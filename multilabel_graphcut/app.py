from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import bresenham

import cv2
import gco
import numpy as np

from multilabel_graphcut.common import Labels, MultiLabelState, Point2D
from multilabel_graphcut.gmm import fit_model, get_unary
# from multilabel_graphcut.single_gaussian import fit_model, get_unary


CACHE_DIR = Path(".simple_multilabel_graphcut")


@dataclass
class UIState:
    # Show
    image:       np.ndarray  # H x W x C
    image_label: np.ndarray  # H x W x C
    window_name: str

    # Scribble
    mode: str = None
    scribble_stack: List[Point2D] = None
    down_at: Optional[Point2D] = None
    cur_label_idx = 0

    # Job Signals
    calculation_pending: bool = False

    # Superpixels
    n_segments: int = 16384
    smoothness: int = 1

    @classmethod
    def new(cls, image):
        return cls(
            image.copy(), 40 * np.ones_like(image),
            'annotation tool',
            smoothness=1,
            calculation_pending=False,
            scribble_stack=[],
        )


def event_loop(state: MultiLabelState, labels: Labels):
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
                state = update_superpixels(state, *user_input)
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


def iterate(state: MultiLabelState, labels: Labels, smoothness: int, use_superpixels: bool):
    state.save(CACHE_DIR)

    # Routine start
    if not use_superpixels:
        xs_gt = np.nonzero(state.user_mask.flatten() > 0)
        label_flat = state.labels.flatten()
        image_flat = state.image.reshape(-1, state.image.shape[-1])
    else:
        label_flat = np.array([
            np.bincount(state.labels[tuple(r.coords.T)]).argmax()
            for r in state.segment_regions
        ])
        image_flat = np.array([r.mean_intensity for r in state.segment_regions])
        xs_gt = np.unique(state.segment_labelmap[np.nonzero(state.user_mask > 0)] - 1)

    ls_gt = label_flat[xs_gt]
    if state.model is None:
        # in case of first run, use annotated values only!
        img, lab = image_flat[xs_gt], ls_gt
    else:
        img, lab = image_flat, label_flat

    # fit the model
    # Add conversion so that non-existent labels are not mendatory
    labels_mask = np.array([(state.labels == i).any() for i, _ in enumerate(labels)])
    if labels_mask.sum() < 2:
        print("Need more than 2 labels to be annotated")
        return state
    label2compressed = np.cumsum(labels_mask) - 1
    compressed2labels, = np.nonzero(labels_mask)

    # state.model = fit_model(img, label2compressed[lab], len(compressed2labels), None)
    state.model = fit_model(img, label2compressed[lab], labels_mask.sum(), None)

    if state.model is None:
        return state

    # calculate likelihood
    unary_flat = get_unary(image_flat, state.model)

    # hard assignment for user inputs
    unary_flat[xs_gt] = 100.0
    unary_flat[xs_gt, label2compressed[ls_gt]] = 0.0

    # make pairwise terms
    pairwise = (1 - np.eye(len(compressed2labels))) * smoothness

    if use_superpixels:
        mns = np.array(state.segment_rag.edges()) - 1  # label map index starts from 1
        xys = np.array([r.centroid for r in state.segment_regions])
    
        val_diff = ((image_flat[mns[:, 0]] - image_flat[mns[:, 1]]) ** 2).sum(axis=1)
        beta = 1 / 2 / val_diff.mean()
        dis = np.linalg.norm(xys[mns[:, 0]] - xys[mns[:, 1]], axis=1)
        state.grabcut_gamma = 100
        edge_costs = state.grabcut_gamma / dis * np.exp(- beta * val_diff)
        del dis, val_diff

        print("[*] Max unary:", unary_flat[unary_flat != 100].max())
        print("[*] Max edge:",  edge_costs.max())

        labels_flat = gco.cut_general_graph(mns, edge_costs, unary_flat, pairwise,)

        labels = np.empty_like(state.labels, dtype=np.uint8)
        rs = state.segment_regions
        for ridx, r in enumerate(rs):
            labels[tuple(r.coords.T)] = compressed2labels[labels_flat[ridx]]

        labels[state.user_mask > 0] = state.labels[state.user_mask > 0]
    else:
        unary = unary_flat.reshape(*state.image.shape[:2], len(labels))
        labels = gco.cut_grid_graph_simple(unary, pairwise, connect=8, algorithm='expansion')
        labels = compressed2labels[labels].reshape(*state.image.shape[:2])

    # superpixels
    print('done!')
    state.labels = labels
    return state


def user_input_loop(state: MultiLabelState, labels: Labels, use_superpixels: bool = True):
    ui = UIState.new(image=state.image)
    yield ('superpixel', ui.n_segments)

    cv2.namedWindow(ui.window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(ui.window_name, lambda *args: mouse_callback(ui, labels, *args))
    cv2.namedWindow('superpixels', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('superpixels', lambda *args: mouse_callback(ui, labels, *args))

    def reload():
        ui.image_label = np.empty_like(state.image)
        for i, label in enumerate(labels):
            ui.image_label[state.labels == i] = label[2]
        ui.image = cv2.addWeighted(state.image, 0.8, ui.image_label, 0.2, 1.0)

    while True:
        cv2.imshow("user mask", state.user_mask)
        cv2.imshow('superpixels', state.segment_vis)
        cv2.imshow("labels", ui.image_label)
        cv2.imshow(ui.window_name, ui.image)

        k = cv2.waitKey(50)
        if ui.calculation_pending:
            yield ("iterate", ui.smoothness, use_superpixels)
            reload()
            ui.calculation_pending = False

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
                ui.n_segments *= 1.5
                ui.n_segments = int(ui.n_segments)
                print("superpixel:", ui.n_segments)
                yield ('superpixel', ui.n_segments)
            if k == ord('-'):
                ui.n_segments /= 1.5
                ui.n_segments = int(ui.n_segments)
                print("superpixel:", ui.n_segments)
                yield ('superpixel', ui.n_segments)

            if k == ord('>'):
                ui.smoothness += 1
                print("smoothness:", ui.smoothness)
            if k == ord('<'):
                ui.smoothness -= 1
                print("smoothness:", ui.smoothness)

            if k == ord(' '):
                yield ("iterate", ui.smoothness, use_superpixels)
                reload()

            if k == ord('\t'):
                ui.cur_label_idx += 1
                ui.cur_label_idx %= len(labels)
                print("Labeling with:", labels[ui.cur_label_idx][0])

        # process mouse
        if ui.scribble_stack != []:
            stack = ui.scribble_stack
            ui.scribble_stack = []
            yield ("scribble", stack, use_superpixels)

    # cleanup
    cv2.destroyAllWindows()
    return


def draw(state: MultiLabelState, scribble, use_superpixels: bool) -> MultiLabelState:
    for pt1, pt2, mode, label in scribble:
        mask_color = 255 if mode == 'paint' else 0
        if not use_superpixels:
            cv2.line(state.labels,    pt1, pt2, color=label,      thickness=5)
            cv2.line(state.user_mask, pt1, pt2, color=mask_color, thickness=5)
            continue

        for _x, _y in np.unique(list(bresenham.bresenham(*pt1, *pt2)), axis=0):
            r = state.segment_regions[state.segment_labelmap[_y, _x] - 1]
            state.labels[tuple(r.coords.T)] = label
            state.user_mask[tuple(r.coords.T)] = mask_color
    return state


def update_superpixels(state: MultiLabelState, n_segments: int) -> MultiLabelState:
    state.n_segments = n_segments
    return state


def mouse_callback(
    ui: UIState, labels: Labels, event, x, y,
    flags, param,
) -> None:
    if event == cv2.EVENT_RBUTTONDOWN:
        ui.mode = 'erase'
        ui.down_at = (x, y)
    elif event == cv2.EVENT_RBUTTONUP:
        if ui.down_at is not None:
            ui.calculation_pending = True
        ui.mode = None
        ui.down_at = None

    if event == cv2.EVENT_LBUTTONDOWN:
        ui.mode = 'paint'
        ui.down_at = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if ui.down_at is not None:
            ui.calculation_pending = True
        ui.mode = None
        ui.down_at = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if ui.down_at is not None:
            ui.scribble_stack.append((
                ui.down_at,
                (x, y),
                ui.mode,
                ui.cur_label_idx,
            ))
            cv2.line(ui.image,       ui.down_at, (x, y), color=labels[ui.cur_label_idx][2], thickness=5)
            cv2.line(ui.image_label, ui.down_at, (x, y), color=labels[ui.cur_label_idx][2], thickness=5)
            ui.down_at = (x, y)


if __name__ == '__main__':
    main()
