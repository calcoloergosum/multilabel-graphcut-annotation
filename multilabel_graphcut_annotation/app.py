"""Application is defined here"""
from __future__ import annotations

from typing import Any, Iterator, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

import bresenham
import cv2
import gco
import numpy as np
import numpy.typing as npt
from skimage.future.graph import RAG, rag_mean_color
import skimage.measure
import skimage.segmentation

from multilabel_graphcut_annotation.gmm import Model, fit_model, pixelwise_likelihood

# from multilabel_graphcut_annotation.single_gaussian import fit_model, pixelwise_likelihood


Point2i = Tuple[int, int]
BGRColor = Tuple[float, float, float]
Labels = List[Tuple[str, str, BGRColor]]

CACHE_DIR = Path(".multilabel_graphcut_annotation")


def read_label_definitions(path: Path) -> Labels:
    """Read label definition"""
    with path.open('r') as f:
        contents = json.load(f)
        for value in contents.values():
            value['color'] = tuple(value['color'])
    return [(k, v["key"], v["color"]) for k, v in contents.items()]


@dataclass
class MultiLabelState:
    """Collection of multi-label graph cut related variables"""
    # pylint: disable=too-many-instance-attributes
    image: npt.NDArray[np.uint8]      # H x W x C
    user_mask: npt.NDArray[np.uint8]  # H x W, value in {0, 255}
    labelmap: npt.NDArray[np.uint8]   # H x W, value in [0, n_class)
    model: Optional[Model] = None

    # Superpixel
    _n_segments: int = 16384
    _segment_vis: Optional[npt.NDArray[np.uint8]] = None
    _segment_labelmap: Optional[npt.NDArray[np.uint8]] = None
    _segment_rag: Optional[RAG] = None
    _segment_regions: Optional[List[skimage.measure._regionprops.RegionProperties]] = None

    grabcut_gamma: float = 100.0

    @classmethod
    def new(cls, image: npt.NDArray[np.uint8]) -> MultiLabelState:
        """Default constructor"""
        return cls(
            image,
            np.zeros(image.shape[:2], dtype=np.uint8),
            np.zeros(image.shape[:2], dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: Path, prefix: str) -> MultiLabelState:
        """load from a path"""
        return cls(
            cv2.imread((path / f"{prefix}_image.png").as_posix(), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH),
            cv2.imread((path / f"{prefix}_user_mask.png").as_posix(), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH),
            cv2.imread((path / f"{prefix}_labels.png").as_posix(), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH),
        )

    def save(self, path: Path, prefix: str) -> None:
        """Save to a file so that it can be continued later"""
        path.mkdir(exist_ok=True, parents=False)
        cv2.imwrite((path / f"{prefix}_image.png").as_posix(), self.image)
        cv2.imwrite((path / f"{prefix}_labels.png").as_posix(), self.labelmap)
        cv2.imwrite((path / f"{prefix}_user_mask.png").as_posix(), self.user_mask)

    # superpixel related
    @property
    def n_segments(self) -> int:
        """Number of segments"""
        return self._n_segments

    @n_segments.setter
    def n_segments(self, value: int) -> None:
        """Number of segments; resets all the cached properties"""
        self._n_segments = value
        self._segment_labelmap = None
        self._segment_rag = None
        self._segment_regions = None
        self._segment_vis = None

    @property
    def segment_labelmap(self) -> npt.NDArray[np.uint8]:
        """cached property of superpixel segments. Pixel -> region id"""
        if self._segment_labelmap is None:
            segment_labelmap = skimage.segmentation.slic(
                self.image,
                n_segments=self.n_segments, compactness=10,
                multichannel=True,
                enforce_connectivity=True,
                convert2lab=True,
                start_label=1)
            self._segment_labelmap = segment_labelmap
        return self._segment_labelmap

    @property
    def segment_regions(self) -> List[skimage.measure._regionprops.RegionProperties]:
        """cached property of superpixel regions"""
        if self._segment_regions is None:
            self._segment_regions = skimage.measure.regionprops(
                self.segment_labelmap,
                intensity_image=self.image,
                cache=True,
            )
            # TODO: make separate test
            assert self._segment_labelmap is not None
            for ridx, region in enumerate(self._segment_regions):
                assert (self._segment_labelmap[tuple(region.coords.T)] - 1 == ridx).all()
                assert ridx == region.label - 1
                assert region.coords.size > 0
        return self._segment_regions

    @property
    def segment_rag(self) -> RAG:
        """cached property of superpixel region affinity graph"""
        if self._segment_rag is None:
            self._segment_rag = rag_mean_color(self.image, self.segment_labelmap, connectivity=2)
        return self._segment_rag

    @property
    def segment_vis(self) -> npt.NDArray[np.uint8]:
        """cached property of superpixel visualization"""
        if self._segment_vis is None:
            self._segment_vis = visualize_superpixel(self.image, self.segment_regions)
        return self._segment_vis


def visualize_superpixel(
    image: npt.NDArray[np.uint8],
    regions: List[skimage.measure._regionprops.RegionProperties]
) -> npt.NDArray[np.uint8]:
    """Return superpixel visualized"""
    superpixels = np.empty_like(image)
    for region in regions:
        ys, xs = region.coords.T
        superpixels[ys, xs] = region.mean_intensity
    return superpixels


@dataclass
class UIState:
    """Collection of UI related variables"""
    # pylint: disable=too-many-instance-attributes
    # Show
    image:       npt.NDArray[np.uint8]  # H x W x C
    image_label: npt.NDArray[np.uint8]  # H x W, value in range [0, n_class)
    window_name: str

    # Scribble
    scribble_stack: List[Tuple[Point2i, Point2i, str, int]]
    mode: Optional[str] = None
    down_at: Optional[Point2i] = None
    cur_label_idx = 0

    # Job Signals
    calculation_pending: bool = False

    # Superpixels
    n_segments: int = 16384
    smoothness: int = 1

    # Undo
    last_user_mask: Optional[npt.NDArray[np.uint8]] = None
    last_labels: Optional[npt.NDArray[np.uint8]] = None

    @classmethod
    def new(cls, image: npt.NDArray[np.uint8]) -> UIState:
        """Constructor"""
        return cls(
            image.copy(),
            40 * np.ones_like(image, dtype=np.uint8),  # type: ignore
            'annotation tool',
            scribble_stack=[],
            calculation_pending=False,
        )


def event_loop(state: MultiLabelState, labels: Labels):
    """Consume user input"""
    n_labels = len(labels)
    assert n_labels < 255, "too many labels .."
    assert n_labels == len(set(n for n, _, _ in labels)), "overlapping names"
    assert n_labels == len(set(k for _, k, _ in labels)), "overlapping shortcuts"
    assert n_labels == len(set(c for _, _, c in labels)), "overlapping colors"

    while True:
        for opname, *user_input in user_input_loop(state, labels):
            if opname == 'scribble':
                state = draw(state, *user_input)
            elif opname == 'superpixel':
                n_segments, = user_input
                state.n_segments = n_segments
            elif opname == 'iterate':
                state = iterate(state, labels, *user_input)
            elif opname == 'quit':
                return state
            elif opname == 'quit and delete':
                return None
            elif opname == 'restart':
                state = MultiLabelState.new(state.image)
                break
            else:
                raise NotImplementedError(opname, *user_input)


def fit(state: MultiLabelState, label2compressed: npt.NDArray[np.uint8]) -> Optional[Model]:
    """Do model fitting here; pixelwise."""
    xs_gt = np.nonzero(state.user_mask.flatten() > 0)
    label_flat = state.labelmap.flatten()
    image_flat = state.image.reshape(-1, state.image.shape[-1])

    # in case of first run, use hard-assigned values only
    if state.model is None:
        img, lab = image_flat[xs_gt], label_flat[xs_gt]
    else:
        img, lab = image_flat, label_flat

    return fit_model(img, label2compressed[lab], label2compressed.max() + 1, None)


def iterate(state: MultiLabelState, labels: Labels):
    """Calculate next iteration, given labels.
    """
    # pylint: disable=too-many-locals
    state.save(CACHE_DIR, "cache")

    # fit the model
    # Add conversion so that non-existent labels are not mendatory
    labels_mask = np.array([(state.labelmap == i).any() for i, _ in enumerate(labels)])
    if labels_mask.sum() < 2:
        print("Need more than 2 labels to be annotated")
        return state
    label2compressed = np.cumsum(labels_mask) - 1
    compressed2labels, = np.nonzero(labels_mask)

    # Routine start
    # fit model
    state.model = fit(state, label2compressed)  # type: ignore
    if state.model is None:
        return state

    # superpixel
    label_flat = np.array([
        np.bincount(state.labelmap[tuple(r.coords.T)]).argmax()
        for r in state.segment_regions
    ])
    image_flat = np.array([r.mean_intensity for r in state.segment_regions])
    xs_gt = np.unique(state.segment_labelmap[np.nonzero(state.user_mask > 0)] - 1)
    weights = np.array([r.area for r in state.segment_regions])
    ls_gt = label_flat[xs_gt]

    # calculate likelihood
    unary_flat = pixelwise_likelihood(image_flat, weights, state.model)

    # hard assignment for user inputs
    unary_flat[xs_gt] = 10000.0
    unary_flat[xs_gt, label2compressed[ls_gt]] = 0.0

    # make pairwise terms
    pairwise = (1 - np.eye(len(compressed2labels)))

    mns = np.array(state.segment_rag.edges()) - 1  # label map index starts from 1
    xys = np.array([r.centroid for r in state.segment_regions])

    val_diff = ((image_flat[mns[:, 0]] - image_flat[mns[:, 1]]) ** 2).sum(axis=1)
    beta = 1 / 2 / val_diff.mean()
    dis = np.linalg.norm(xys[mns[:, 0]] - xys[mns[:, 1]], axis=1)
    edge_costs = (
        state.grabcut_gamma / dis *
        np.exp(- beta * val_diff) *
        np.sqrt(weights[mns[:, 0]] * weights[mns[:, 1]])
    )
    del dis, val_diff

    print("[*] Max unary:", unary_flat[unary_flat != 10000].max())
    print("[*] Max edge:",  edge_costs.max())

    labels_flat = gco.cut_general_graph(mns, edge_costs, unary_flat, pairwise,)

    labelmap = np.empty_like(state.labelmap, dtype=np.uint8)
    for ridx, reg in enumerate(state.segment_regions):
        labelmap[tuple(reg.coords.T)] = compressed2labels[labels_flat[ridx]]

    labelmap[state.user_mask > 0] = state.labelmap[state.user_mask > 0]

    # superpixels
    print('done!')
    state.labelmap = labelmap
    return state


def user_input_loop(state: MultiLabelState, labels: Labels) -> Iterator[Tuple[Any, ...]]:
    """Loop over user inputs and give commands to the main function"""
    # pylint: disable=too-many-statements,too-many-branches
    uistate = UIState.new(image=state.image)
    yield ('superpixel', uistate.n_segments)

    cv2.namedWindow(uistate.window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(uistate.window_name, lambda *args: mouse_callback(uistate, labels, *args))
    cv2.namedWindow('superpixels', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('superpixels', lambda *args: mouse_callback(uistate, labels, *args))

    def reload() -> None:
        uistate.image_label = np.empty_like(state.image)
        for i, label in enumerate(labels):
            uistate.image_label[state.labelmap == i] = label[2]
        uistate.image = cv2.addWeighted(state.image, 0.8, uistate.image_label, 0.2, 1.0)

    while True:
        cv2.imshow("user mask", state.user_mask)
        cv2.imshow('superpixels', state.segment_vis)
        cv2.imshow("labels", uistate.image_label)
        cv2.imshow(uistate.window_name, uistate.image)

        k = cv2.waitKey(50)
        if uistate.calculation_pending:
            yield ("iterate",)
            reload()
            uistate.calculation_pending = False

        # process keyboard
        if k != 0xff:
            if k == ord('r'):
                yield ("restart",)
                break
            if k == ord('q'):
                yield ("quit",)
                break
            if k == ord('x'):
                yield ("quit and delete",)
                break
            if k == ord('+'):
                uistate.n_segments = int(uistate.n_segments * 1.5)
                print("superpixel:", uistate.n_segments)
                yield ('superpixel', uistate.n_segments)
            if k == ord('-'):
                uistate.n_segments = int(uistate.n_segments / 1.5)
                print("superpixel:", uistate.n_segments)
                yield ('superpixel', uistate.n_segments)

            if k == ord('>'):
                state.grabcut_gamma *= 1.2
                print("grabcut_gamma:", state.grabcut_gamma)
            if k == ord('<'):
                state.grabcut_gamma /= 1.2
                print("grabcut_gamma:", state.grabcut_gamma)

            if k == ord(' '):
                yield ("iterate",)
                reload()

            if k == ord('\t'):
                uistate.cur_label_idx += 1
                uistate.cur_label_idx %= len(labels)
                print("Labeling with:", labels[uistate.cur_label_idx][0])

        # process mouse
        if uistate.scribble_stack != []:
            stack = uistate.scribble_stack
            uistate.scribble_stack = []
            uistate.last_user_mask = state.user_mask
            uistate.last_labels = state.labelmap
            yield ("scribble", stack)

        if k == ord('z'):
            if uistate.last_user_mask is None or uistate.last_labels is None:
                print("Cannot undo")
                continue
            state.user_mask = uistate.last_user_mask
            state.labelmap = uistate.last_labels
            yield ("iterate",)

    # cleanup
    cv2.destroyAllWindows()
    return


def draw(state: MultiLabelState, scribble) -> MultiLabelState:
    """Draw given scribble on state"""
    for pt1, pt2, mode, label in scribble:
        mask_color = 255 if mode == 'paint' else 0

        for _x, _y in np.unique(list(bresenham.bresenham(*pt1, *pt2)), axis=0):
            reg = state.segment_regions[state.segment_labelmap[_y, _x] - 1]
            state.labelmap[tuple(reg.coords.T)] = label
            state.user_mask[tuple(reg.coords.T)] = mask_color
    return state


def mouse_callback(
    uistate: UIState, labels: Labels, event, x: int, y: int,
    *_,
) -> None:
    """Mouse event callback for cv2 event listener"""
    if event == cv2.EVENT_RBUTTONDOWN:
        uistate.mode = 'erase'
        uistate.down_at = (x, y)
    elif event == cv2.EVENT_RBUTTONUP:
        if uistate.down_at is not None:
            uistate.calculation_pending = True
        uistate.mode = None
        uistate.down_at = None

    if event == cv2.EVENT_LBUTTONDOWN:
        uistate.mode = 'paint'
        uistate.down_at = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if uistate.down_at is not None:
            uistate.calculation_pending = True
        uistate.mode = None
        uistate.down_at = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if uistate.down_at is not None:
            assert uistate.mode is not None
            uistate.scribble_stack.append((
                uistate.down_at,
                (x, y),
                uistate.mode,
                uistate.cur_label_idx,
            ))
            cv2.line(uistate.image,       uistate.down_at, (x, y), color=labels[uistate.cur_label_idx][2], thickness=5)
            cv2.line(uistate.image_label, uistate.down_at, (x, y), color=labels[uistate.cur_label_idx][2], thickness=5)
            uistate.down_at = (x, y)
