from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import skimage.measure
import skimage.segmentation
from skimage.future.graph import RAG, rag_mean_color

Point2D = Tuple[float, float]
BGRColor = Tuple[float, float, float]
Labels = List[Tuple[str, str, BGRColor]]


@dataclass
class MultiLabelState:
    image: np.ndarray     # H x W x C
    user_mask: np.ndarray # H x W x bool
    labelmap: np.ndarray    # H x W x int
    model: Optional[Any]

    # Superpixel
    _n_segments: int = 16384
    _segment_vis: np.ndarray = None
    _segment_labelmap: np.ndarray = None
    _segment_rag: RAG = None
    _segment_regions: List = None

    grabcut_gamma: float = 100.0

    @classmethod
    def new(cls, image: np.ndarray) -> MultiLabelState:
        return cls(
            image,
            np.zeros(image.shape[:2], dtype=np.uint8),
            np.zeros(image.shape[:2], dtype=np.uint8),
            None,
        )

    @classmethod
    def load(cls, path: Path) -> MultiLabelState:
        return cls(
            cv2.imread((path / "image.png").as_posix(), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH),
            cv2.imread((path / "user_mask.png").as_posix(), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH),
            cv2.imread((path / "labels.png").as_posix(), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH),
            None,
            None,
            None,
        )

    def save(self, path: Path) -> None:
        path.mkdir(exist_ok=True, parents=False)
        cv2.imwrite((path / "image.png").as_posix(), self.image)
        cv2.imwrite((path / "labels.png").as_posix(), self.labelmap)
        cv2.imwrite((path / "user_mask.png").as_posix(), self.user_mask)

    # superpixel related
    @property
    def n_segments(self) -> int:
        return self._n_segments

    @n_segments.setter
    def n_segments(self, value):
        self._n_segments = value
        self._segment_labelmap = None
        self._segment_rag = None
        self._segment_regions = None
        self._segment_vis = None

    @property
    def segment_labelmap(self) -> np.ndarray:
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
    def segment_regions(self) -> List:
        if self._segment_regions is None:
            self._segment_regions = skimage.measure.regionprops(
                self.segment_labelmap,
                intensity_image=self.image,
                cache=True,
            )
            for ridx, r in enumerate(self._segment_regions):
                assert (self._segment_labelmap[tuple(r.coords.T)] - 1 == ridx).all()
                assert ridx == r.label - 1
                assert r.coords.size > 0
        return self._segment_regions

    @property
    def segment_rag(self) -> RAG:
        if self._segment_rag is None:
            self._segment_rag = rag_mean_color(self.image, self.segment_labelmap, connectivity=2)
        return self._segment_rag

    @property
    def segment_vis(self) -> np.ndarray:
        if self._segment_vis is None:
            self._segment_vis = visualize_superpixel(self.image, self.segment_regions)
        return self._segment_vis


def visualize_superpixel(image: np.ndarray, regions):
    superpixels = np.empty_like(image)
    for r in regions:
        ys, xs = r.coords.T
        superpixels[ys, xs] = r.mean_intensity
    return superpixels
