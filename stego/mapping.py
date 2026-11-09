"""Section III-A: object sequence dictionary, top-M objects, clusters and image index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MappingError(RuntimeError):
    """Raised when a complete sequence index cannot be constructed."""


def top_objects(objects: list[dict], conf_threshold: float, m: int) -> list[dict]:
    """Objects above the threshold, largest box first (III-A2). Ties are deterministic."""

    def area(obj: dict) -> float:
        x1, y1, x2, y2 = obj["box"]
        return float((x2 - x1) * (y2 - y1))

    kept = [obj for obj in objects if obj["score"] > conf_threshold]
    kept.sort(key=lambda obj: (-area(obj), -obj["score"], obj["label"], obj["box"]))
    return kept[:m]


def label_image_counts(objects_per_image: list[list[dict]], conf_threshold: float, m: int) -> dict[str, int]:
    """Number of images able to map each label, meaning it is among their top-M objects.

    Counting only these keeps every group usable, so the index is complete whenever
    there are at least 2^n such labels.
    """
    counts: dict[str, int] = {}
    for objects in objects_per_image:
        for label in {obj["label"] for obj in top_objects(objects, conf_threshold, m)}:
            counts[label] = counts.get(label, 0) + 1
    return counts


def build_dictionary(label_counts: dict[str, int], n: int) -> dict[str, int]:
    """Groups alphabetically ordered labels into 2^n groups (III-A1, Fig. 5).

    Every label gets one group. Only neighbouring labels can share a group. Dynamic
    programming minimises the sum of squared image counts per group, which gives the
    most even split available under that alphabetical constraint.
    """
    labels = sorted(label_counts)
    groups = 2**n
    if len(labels) < groups:
        raise MappingError(
            f"n={n} needs {groups} groups, but the dataset has only {len(labels)} detected labels. "
            "The sequence index would be incomplete. Use a smaller n."
        )
    weights = [label_counts[label] for label in labels]
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    total_labels = len(labels)

    def cost(start: int, end: int) -> int:
        return (prefix[end] - prefix[start]) ** 2

    infinity = float("inf")
    best = [[infinity] * (total_labels + 1) for _ in range(groups + 1)]
    back = [[0] * (total_labels + 1) for _ in range(groups + 1)]
    best[0][0] = 0
    for g in range(1, groups + 1):
        for end in range(g, total_labels - (groups - g) + 1):
            for start in range(g - 1, end):
                if best[g - 1][start] == infinity:
                    continue
                value = best[g - 1][start] + cost(start, end)
                if value < best[g][end]:
                    best[g][end] = value
                    back[g][end] = start

    dictionary: dict[str, int] = {}
    end = total_labels
    for g in range(groups, 0, -1):
        start = back[g][end]
        for label in labels[start:end]:
            dictionary[label] = g - 1
        end = start
    return dictionary


def segment_to_bits(group: int, n: int) -> str:
    return format(group, f"0{n}b")


def deep_hashes(features: np.ndarray, threshold: float) -> np.ndarray:
    """Per-image min-max normalisation followed by Eq. 1."""
    low = features.min(axis=1, keepdims=True)
    high = features.max(axis=1, keepdims=True)
    span = np.where(high - low > 1e-12, high - low, 1.0)
    return ((features - low) / span > threshold).astype(np.float32)


def run_clustering(config, features: np.ndarray) -> np.ndarray:
    from sklearn.cluster import KMeans, MiniBatchKMeans

    hashes = deep_hashes(features, config.hash_threshold)
    if len(hashes) < config.k:
        raise MappingError(f"k={config.k} is larger than the number of images ({len(hashes)})")
    if config.cluster_algorithm == "kmeans":
        model = KMeans(
            n_clusters=config.k,
            init="k-means++",
            n_init=1,
            max_iter=config.cluster_max_iter,
            random_state=config.cluster_seed,
        )
    else:
        model = MiniBatchKMeans(
            n_clusters=config.k,
            init="k-means++",
            n_init=1,
            max_iter=config.cluster_max_iter,
            batch_size=4096,
            random_state=config.cluster_seed,
        )
    return model.fit_predict(hashes).astype(np.int32)


@dataclass
class ImageIndex:
    """Table I for one (n, M) setting, stored as arrays for fast search."""

    ids: list[str]
    n: int
    m: int
    dictionary: dict[str, int]
    sequences: list[tuple[int, ...]]  # group of each top-M object, largest first
    labels: list[tuple[str, ...]]
    clusters: np.ndarray
    counts: np.ndarray  # image x group: how many top-M objects map to that group
    cluster_members: dict[int, np.ndarray]

    @property
    def groups(self) -> int:
        return 2**self.n

    @property
    def usable_images(self) -> int:
        return sum(1 for seq in self.sequences if seq)

    def table_rows(self):
        """Yields rows in the form of the paper's Table I."""
        for image_id, labels, sequence, cluster in zip(self.ids, self.labels, self.sequences, self.clusters):
            for number, (label, group) in enumerate(zip(labels, sequence), start=1):
                yield {
                    "filename": image_id,
                    "object_label": label,
                    "object_number": number,
                    "object_sequence": segment_to_bits(group, self.n),
                    "cluster": f"C{int(cluster)}",
                }


def build_index(
    ids: list[str],
    objects_per_image: list[list[dict]],
    clusters: np.ndarray,
    n: int,
    m: int,
    conf_threshold: float,
) -> ImageIndex:
    dictionary = build_dictionary(label_image_counts(objects_per_image, conf_threshold, m), n)
    sequences: list[tuple[int, ...]] = []
    labels: list[tuple[str, ...]] = []
    counts = np.zeros((len(ids), 2**n), dtype=np.uint8)
    for row, objects in enumerate(objects_per_image):
        chosen = top_objects(objects, conf_threshold, m)
        names = tuple(obj["label"] for obj in chosen)
        sequence = tuple(dictionary[name] for name in names)
        labels.append(names)
        sequences.append(sequence)
        for group in sequence:
            counts[row, group] += 1
    members = {int(c): np.flatnonzero(clusters == c) for c in np.unique(clusters)}
    return ImageIndex(ids, n, m, dictionary, sequences, labels, np.asarray(clusters), counts, members)
