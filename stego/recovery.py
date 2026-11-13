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


class DescriptorRetriever:
    """Phase 3 retrieval. Searches every sendable image, so no cluster ID is needed.

    sscd:       the image whose SSCD descriptor is closest to the received image's.
    sscd_align: the SSCD shortlist of `top_k` images, re-ranked after Fourier–Mellin
                alignment. Score = cosine similarity + aligned correlation.
    """

    def __init__(self, index: ImageIndex, paths: dict[str, Path], descriptors: np.ndarray, model, mode: str,
                 top_k: int, size: int, search: np.ndarray | None = None, cache_size: int = 20000):
        if mode not in ("sscd", "sscd_align"):
            raise ValueError(f"Unknown descriptor retrieval mode {mode!r}")
        self.index = index
        self.paths = paths
        self.model = model
        self.mode = mode
        self.top_k = top_k
        self.size = size
        # Both sides know the database, so both can drop images that are never sent.
        search = np.arange(len(index.ids)) if search is None else np.asarray(search, dtype=np.int64)
        if search.size == 0:
            raise RuntimeError("The search set is empty")
        self.search = search
        self.descriptors = np.ascontiguousarray(descriptors[search], dtype=np.float32)
        self.cache_size = cache_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def _candidate(self, image_index: int) -> np.ndarray:
        from .register import fit_gray

        cached = self._cache.get(image_index)
        if cached is not None:
            self._cache.move_to_end(image_index)
            return cached
        gray = fit_gray(read_image(self.paths[self.index.ids[image_index]]), self.size)
        self._cache[image_index] = gray
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return gray

    def similarities(self, received_bgr: np.ndarray) -> np.ndarray:
        query = self.model.describe([received_bgr])[0]
        return self.descriptors @ query

    def retrieve(self, received_bgr: np.ndarray, cluster: int | None = None) -> Retrieved:
        """`cluster` is ignored: these modes search the whole database."""
        cosine = self.similarities(received_bgr)
        if self.mode == "sscd" or len(cosine) == 1:
            order = np.argsort(-cosine, kind="stable")[:2]
            second = float(cosine[order[1]]) if len(order) > 1 else -np.inf
            return Retrieved(int(self.search[order[0]]), float(cosine[order[0]]), second)
        from .register import align

        shortlist = np.argsort(-cosine, kind="stable")[: self.top_k]
        scores = []
        for position in shortlist:
            image_index = int(self.search[position])
            aligned = align(self._candidate(image_index), received_bgr, self.size)
            scores.append(float(cosine[position]) + aligned.score)
        ranking = np.argsort(-np.asarray(scores), kind="stable")
        best = int(self.search[shortlist[ranking[0]]])
        second = scores[ranking[1]] if len(ranking) > 1 else -np.inf
        return Retrieved(best, float(scores[ranking[0]]), float(second))


def make_retriever(config, index: ImageIndex, paths: dict[str, Path], descriptors=None, model=None, search=None):
    if config.retrieval == "cluster_ssim":
        return Retriever(index, paths, config.ssim_mode, config.ssim_size)
    if descriptors is None or model is None:
        raise RuntimeError(f"retrieval={config.retrieval} needs descriptors and a descriptor model")
    return DescriptorRetriever(index, paths, descriptors, model, config.retrieval, config.retrieval_top_k,
                               config.ssim_size, search)
