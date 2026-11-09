"""Section III-B: message segmentation, stego image selection and auxiliary data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mapping import ImageIndex


@dataclass(frozen=True)
class AuxInfo:
    """The per-image auxiliary tuple of Example 4."""

    order: tuple[int, ...]  # order[i] = message position of object i; 0 means unused
    cluster: int
    padding: int

    def as_tuple(self) -> tuple:
        return (*self.order, f"C{self.cluster}", -1, self.padding)

    @property
    def mapped_count(self) -> int:
        return sum(1 for value in self.order if value > 0)


@dataclass(frozen=True)
class StegoImage:
    image_index: int
    aux: AuxInfo


@dataclass
class HideResult:
    stego: list[StegoImage]
    complete: bool
    segments_hidden: int
    segments_total: int
    repeated_images: int

    @property
    def completion_percent(self) -> float:
        return 100.0 * self.segments_hidden / self.segments_total if self.segments_total else 100.0


def bytes_to_segments(message: bytes, n: int) -> tuple[list[int], int]:
    """Splits bytes into n-bit segments. Pads the end with zeros and returns the count."""
    bits = "".join(format(byte, "08b") for byte in message)
    padding = (n - len(bits) % n) % n
    bits += "0" * padding
    return [int(bits[i : i + n], 2) for i in range(0, len(bits), n)], padding


def segments_to_bytes(segments: list[int | None], n: int, padding: int) -> bytes:
    """Reverses bytes_to_segments. A lost segment (None) becomes zeros."""
    bits = "".join(format(0 if segment is None else segment, f"0{n}b") for segment in segments)
    if padding:
        bits = bits[:-padding]
    usable = len(bits) - len(bits) % 8
    return bytes(int(bits[i : i + 8], 2) for i in range(0, usable, 8))


def _mapping_order(sequence: tuple[int, ...], window: list[int], m: int) -> tuple[int, ...]:
    """Assigns each message segment to the first unused object with that sequence."""
    order = [0] * m
    for position, target in enumerate(window, start=1):
        for object_index, group in enumerate(sequence):
            if group == target and order[object_index] == 0:
                order[object_index] = position
                break
        else:
            raise AssertionError("The selected image cannot map this window")
    return tuple(order)


def hide_segments(segments: list[int], index: ImageIndex, rng: np.random.Generator, padding: int = 0) -> HideResult:
    """Prefers the image that maps the most consecutive segments (III-B).

    Every image is searched. Among images with the same best count, an image not yet
    used in this message is chosen at random. A used image is repeated only when no
    unused image achieves the same count.
    """
    total = len(segments)
    used = np.zeros(len(index.ids), dtype=bool)
    stego: list[StegoImage] = []
    position = 0
    repeated = 0
    while position < total:
        need = np.zeros(index.groups, dtype=np.int32)
        mask = np.ones(len(index.ids), dtype=bool)
        best_length = 0
        for offset in range(min(index.m, total - position)):
            group = segments[position + offset]
            need[group] += 1
            narrowed = mask & (index.counts[:, group] >= need[group])
            if not narrowed.any():
                break
            mask = narrowed
            best_length = offset + 1
        if best_length == 0:
            return HideResult(stego, False, position, total, repeated)

        candidates = np.flatnonzero(mask)
        unused = candidates[~used[candidates]]
        pool = unused if unused.size else candidates
        if not unused.size:
            repeated += 1
        chosen = int(rng.choice(pool))
        used[chosen] = True
        window = segments[position : position + best_length]
        order = _mapping_order(index.sequences[chosen], window, index.m)
        stego.append(StegoImage(chosen, AuxInfo(order, int(index.clusters[chosen]), padding)))
        position += best_length
    return HideResult(stego, True, position, total, repeated)
