"""Section III-C: retrieve the original image with Eq. 2, then decode from its labels."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .hiding import AuxInfo
from .mapping import ImageIndex

C1 = (0.01 * 255) ** 2
C2 = (0.03 * 255) ** 2
C3 = C2 / 2.0


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def to_comparison_gray(image_bgr: np.ndarray, size: int) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64)


def equation_2(x: np.ndarray, y: np.ndarray) -> float:
    """Eq. 2 as written, with one global window and alpha = beta = gamma = 1."""
    mean_x, mean_y = x.mean(), y.mean()
    std_x, std_y = x.std(), y.std()
    covariance = ((x - mean_x) * (y - mean_y)).mean()
    luminance = (2 * mean_x * mean_y + C1) / (mean_x**2 + mean_y**2 + C1)
    contrast = (2 * std_x * std_y + C2) / (std_x**2 + std_y**2 + C2)
    structure = (covariance + C3) / (std_x * std_y + C3)
    return float(luminance * contrast * structure)


def windowed_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Local-window SSIM with the Gaussian settings of Wang et al. (the paper's [29])."""
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(
            x, y, data_range=255.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False
        )
    )


@dataclass(frozen=True)
class Retrieved:
    image_index: int
    best_score: float
    second_score: float


class Retriever:
    def __init__(self, index: ImageIndex, paths: dict[str, Path], mode: str, size: int, cache_size: int = 20000):
        self.index = index
        self.paths = paths
        self.size = size
        self.score = equation_2 if mode == "global" else windowed_ssim
        self.cache_size = cache_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def _candidate(self, image_index: int) -> np.ndarray:
        cached = self._cache.get(image_index)
        if cached is not None:
            self._cache.move_to_end(image_index)
            return cached
        image = read_image(self.paths[self.index.ids[image_index]])
        gray = to_comparison_gray(image, self.size)
        self._cache[image_index] = gray
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return gray

    def retrieve(self, received_bgr: np.ndarray, cluster: int) -> Retrieved:
        """Returns the highest-scoring image in the given cluster."""
        members = self.index.cluster_members.get(cluster)
        if members is None or members.size == 0:
            raise RuntimeError(f"Cluster C{cluster} has no images")
        query = to_comparison_gray(received_bgr, self.size)
        best_index, best, second = -1, -np.inf, -np.inf
        for image_index in members:
            value = self.score(query, self._candidate(int(image_index)))
            if value > best:
                best_index, second, best = int(image_index), best, value
            elif value > second:
                second = value
        return Retrieved(best_index, float(best), float(second))


def decode_image(image_index: int, aux: AuxInfo, index: ImageIndex) -> list[int | None]:
    """Reads this image's segments in message order. Missing objects become None."""
    sequence = index.sequences[image_index]
    segments: list[int | None] = [None] * aux.mapped_count
    for object_index, position in enumerate(aux.order):
        if position > 0 and object_index < len(sequence):
            segments[position - 1] = sequence[object_index]
    return segments
