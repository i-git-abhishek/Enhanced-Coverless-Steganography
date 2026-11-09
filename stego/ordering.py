"""Phase 2: ordering schemes that need less or no order data.

The paper sends an order array with every stego image (Example 4). This module
compares it with schemes that read objects in a fixed order instead:

    paper-greedy  the paper's search and order array, exactly as the baseline
    paper-dp      the paper's order array, but images chosen by dynamic programming
    A             fixed order (left to right), no order data at all
    A-area        fixed order (largest first), no order data; shows the rule does not matter
    B             fixed order plus an M-bit mask that marks which objects to read
    C             1-bit flag per image: 0 = fixed order, 1 = flag plus the paper's order array

A, A-area, B and C put a 16-bit message length at the start of the message, so the
receiver can drop the extra segments of the last image. The cluster ID is not
changed here; it is reported separately because Phase 3 removes it.

Everything runs on the cached detections. Robustness does not depend on the scheme,
because labels are always read from the retrieved original image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .hiding import _mapping_order, bytes_to_segments, hide_segments, segments_to_bytes
from .mapping import ImageIndex, top_objects

HEADER_BITS = 16
SCHEMES_FIXED = ("paper-greedy", "paper-dp", "A", "A-area", "B")
INFINITY = float("inf")


class OrderingError(RuntimeError):
    """Raised when a message cannot be framed or decoded."""


# ---------------------------------------------------------------- message framing


def frame_message(message: bytes, n: int) -> list[int]:
    """16-bit length in bytes, then the message, padded with zeros to whole segments."""
    if len(message) >= 2**HEADER_BITS:
        raise OrderingError(f"Messages must be shorter than {2**HEADER_BITS} bytes")
    bits = format(len(message), f"0{HEADER_BITS}b") + "".join(format(byte, "08b") for byte in message)
    bits += "0" * ((n - len(bits) % n) % n)
    return [int(bits[i : i + n], 2) for i in range(0, len(bits), n)]


def unframe_message(segments: list[int], n: int) -> bytes | None:
    """Reverses frame_message. Extra segments at the end are ignored."""
    bits = "".join(format(segment, f"0{n}b") for segment in segments)
    if len(bits) < HEADER_BITS:
        return None
    length = int(bits[:HEADER_BITS], 2)
    end = HEADER_BITS + 8 * length
    if len(bits) < end:
        return None
    return bytes(int(bits[i : i + 8], 2) for i in range(HEADER_BITS, end, 8))


# ---------------------------------------------------------------- fixed reading order


def _centre(obj: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = obj["box"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def canonical_sequences(index: ImageIndex, objects_per_image: list[list[dict]], conf_threshold: float,
                        rule: str) -> list[tuple[int, ...]]:
    """Each image's top-M objects in a fixed order, as groups.

    The objects are the same top-M set the paper uses (largest M above the threshold).
    Only the order changes. The order is computed on the original image, so the
    receiver gets the same order after retrieval whatever the attack did.
    """
    if rule == "area":
        return list(index.sequences)
    if rule != "left_to_right":
        raise OrderingError(f"Unknown order rule {rule!r}")
    sequences = []
    for objects in objects_per_image:
        chosen = top_objects(objects, conf_threshold, index.m)
        chosen.sort(key=lambda obj: (*_centre(obj), obj["label"], obj["box"]))
        sequences.append(tuple(index.dictionary[obj["label"]] for obj in chosen))
    return sequences


# ---------------------------------------------------------------- what each image can carry


def order_array_bits(m: int) -> float:
    """The paper's order array, coded as tightly as possible (same as the baseline's aux_bits_tight)."""
    return math.log2(sum(math.perm(m, used) for used in range(1, m + 1)))


@dataclass
class Lookup:
    """For each window of segments, the images that can carry it and the cheapest order cost."""

    exact: dict[tuple[int, ...], np.ndarray]  # scheme A: whole fixed sequence
    prefix: dict[tuple[int, ...], np.ndarray]  # scheme A, last image only: sequence starts with the window
    subsequence: dict[tuple[int, ...], np.ndarray]  # scheme B: objects picked by a mask, order kept
    multiset: dict[tuple[int, ...], np.ndarray]  # paper: any objects in any order


def build_lookup(fixed: list[tuple[int, ...]], area: list[tuple[int, ...]]) -> Lookup:
    exact: dict = {}
    prefix: dict = {}
    subsequence: dict = {}
    multiset: dict = {}
    for image, sequence in enumerate(fixed):
        if not sequence:
            continue
        exact.setdefault(sequence, []).append(image)
        for length in range(1, len(sequence)):
            prefix.setdefault(sequence[:length], []).append(image)
        seen = set()
        for size in range(1, len(sequence) + 1):
            for positions in combinations(range(len(sequence)), size):
                key = tuple(sequence[p] for p in positions)
                if key not in seen:
                    seen.add(key)
                    subsequence.setdefault(key, []).append(image)
    for image, sequence in enumerate(area):
        seen = set()
        for size in range(1, len(sequence) + 1):
            for positions in combinations(range(len(sequence)), size):
                key = tuple(sorted(sequence[p] for p in positions))
                if key not in seen:
                    seen.add(key)
                    multiset.setdefault(key, []).append(image)
    def arrays(table: dict) -> dict:
        return {key: np.asarray(images, dtype=np.int64) for key, images in table.items()}

    return Lookup(arrays(exact), arrays(prefix), arrays(subsequence), arrays(multiset))


# ---------------------------------------------------------------- choosing images


@dataclass(frozen=True)
class Step:
    """One stego image: where it starts, how many segments it carries, and how."""

    start: int
    length: int
    kind: str  # "fixed", "mask" or "order"
    order_bits: float
    candidates: np.ndarray = field(compare=False, repr=False)


def _options(segments: list[int], i: int, scheme: str, lookup: Lookup, m: int, paper_bits: float):
    """Every way one image can carry segments starting at position i."""
    total = len(segments)
    remaining = total - i
    out = []
    for length in range(1, min(m, remaining) + 1):
        window = tuple(segments[i : i + length])
        if scheme in ("A", "A-area", "C"):
            images = lookup.exact.get(window)
            if images is not None:
                out.append(Step(i, length, "fixed", 1.0 if scheme == "C" else 0.0, images))
        if scheme == "B":
            images = lookup.subsequence.get(window)
            if images is not None:
                # One mask bit per object slot, M slots for every image. A fixed length keeps the
                # extra data readable even when one image is retrieved wrongly.
                out.append(Step(i, length, "mask", float(m), images))
        if scheme in ("paper-dp", "C"):
            images = lookup.multiset.get(tuple(sorted(window)))
            if images is not None:
                cost = paper_bits + (1.0 if scheme == "C" else 0.0)
                out.append(Step(i, length, "order", cost, images))
    if scheme in ("A", "A-area", "C") and remaining < m:
        # The last image may hold more objects than segments are left. The length header
        # tells the receiver where the message ends, so the extra segments are dropped.
        images = lookup.prefix.get(tuple(segments[i:]))
        if images is not None:
            out.append(Step(i, remaining, "fixed", 1.0 if scheme == "C" else 0.0, images))
    return out


def _score(images: int, bits: float, lam: float) -> tuple[float, float]:
    """Lower is better. lam is the price of one image in bits; infinity means fewest images first."""
    if math.isinf(lam):
        return (images, bits)
    return (lam * images + bits, images)


def plan(segments: list[int], scheme: str, lookup: Lookup, m: int, lam: float = INFINITY) -> list[Step] | None:
    """Dynamic programming over message positions. Returns None if the message cannot be hidden."""
    total = len(segments)
    paper_bits = order_array_bits(m)
    best: list[tuple[float, float] | None] = [None] * (total + 1)
    totals: list[tuple[int, float]] = [(0, 0.0)] * (total + 1)
    choice: list[Step | None] = [None] * (total + 1)
    best[total] = _score(0, 0.0, lam)
    for i in range(total - 1, -1, -1):
        for option in _options(segments, i, scheme, lookup, m, paper_bits):
            after = i + option.length
            if best[after] is None:
                continue
            images, bits = totals[after][0] + 1, totals[after][1] + option.order_bits
            score = _score(images, bits, lam)
            # Ties keep the longer step, so the result does not depend on option order.
            if best[i] is None or score < best[i] or (score == best[i] and option.length > choice[i].length):
                best[i], totals[i], choice[i] = score, (images, bits), option
    if best[0] is None:
        return None
    steps, i = [], 0
    while i < total:
        steps.append(choice[i])
        i += choice[i].length
    return steps


# ---------------------------------------------------------------- sending and receiving


@dataclass(frozen=True)
class Sent:
    """What the receiver gets for one image, apart from the image itself."""

    image_index: int
    kind: str
    mask: tuple[bool, ...] = ()
    order: tuple[int, ...] = ()


def _mask_for(sequence: tuple[int, ...], window: tuple[int, ...]) -> tuple[bool, ...]:
    mask, position = [], 0
    for group in sequence:
        if position < len(window) and group == window[position]:
            mask.append(True)
            position += 1
        else:
            mask.append(False)
    if position != len(window):
        raise AssertionError("The chosen image cannot carry this window with a mask")
    return tuple(mask)


def realise(steps: list[Step], segments: list[int], fixed: list[tuple[int, ...]], index: ImageIndex,
            rng: np.random.Generator) -> tuple[list[Sent], int]:
    """Picks one image per step, preferring images not used yet in this message."""
    used = np.zeros(len(index.ids), dtype=bool)
    sent, repeated = [], 0
    for step in steps:
        candidates = step.candidates
        unused = candidates[~used[candidates]]
        pool = unused if unused.size else candidates
        repeated += int(not unused.size)
        image = int(rng.choice(pool))
        used[image] = True
        window = tuple(segments[step.start : step.start + step.length])
        if step.kind == "fixed":
            sent.append(Sent(image, "fixed"))
        elif step.kind == "mask":
            sent.append(Sent(image, "mask", mask=_mask_for(fixed[image], window)))
        else:
            sent.append(Sent(image, "order", order=_mapping_order(index.sequences[image], list(window), index.m)))
    return sent, repeated


def decode(sent: list[Sent], fixed: list[tuple[int, ...]], index: ImageIndex) -> list[int]:
    """The receiver's side, given the retrieved original images."""
    segments: list[int] = []
    for item in sent:
        if item.kind == "fixed":
            segments.extend(fixed[item.image_index])
        elif item.kind == "mask":
            segments.extend(g for g, keep in zip(fixed[item.image_index], item.mask) if keep)
        else:
            area = index.sequences[item.image_index]
            out = [None] * sum(1 for p in item.order if p > 0)
            for object_index, position in enumerate(item.order):
                if position > 0:
                    out[position - 1] = area[object_index]
            segments.extend(out)
    return segments


# ---------------------------------------------------------------- one message, one scheme


@dataclass
class Outcome:
    scheme: str
    lam: float
    complete: bool
    images: int = 0
    segments: int = 0
    order_bits: float = 0.0
    header_bits: int = 0
    repeated_images: int = 0
    fixed_share: float = float("nan")
    round_trip: bool = False


def run_scheme(message: bytes, scheme: str, lam: float, index: ImageIndex, fixed_ltr, area, lookups,
               rng: np.random.Generator) -> Outcome:
    if scheme == "paper-greedy":
        segments, padding = bytes_to_segments(message, index.n)
        result = hide_segments(segments, index, rng, padding)
        if not result.complete:
            return Outcome(scheme, lam, False, segments=len(segments))
        recovered = []
        for item in result.stego:
            sequence = index.sequences[item.image_index]
            part = [None] * item.aux.mapped_count
            for object_index, position in enumerate(item.aux.order):
                if position > 0:
                    part[position - 1] = sequence[object_index]
            recovered.extend(part)
        ok = segments_to_bytes(recovered, index.n, padding) == message
        images = len(result.stego)
        return Outcome(scheme, lam, True, images, len(segments), images * order_array_bits(index.m), 0,
                       result.repeated_images, round_trip=ok)

    if scheme == "paper-dp":
        segments, padding = bytes_to_segments(message, index.n)
        header = 0
    else:
        segments, padding = frame_message(message, index.n), 0
        header = HEADER_BITS
    fixed = area if scheme == "A-area" else fixed_ltr
    lookup = lookups["area"] if scheme == "A-area" else lookups["left_to_right"]
    steps = plan(segments, scheme, lookup, index.m, lam)
    if steps is None:
        return Outcome(scheme, lam, False, segments=len(segments), header_bits=header)
    sent, repeated = realise(steps, segments, fixed, index, rng)
    recovered = decode(sent, fixed, index)
    if scheme == "paper-dp":
        ok = segments_to_bytes(recovered, index.n, padding) == message
    else:
        ok = unframe_message(recovered, index.n) == message
    fixed_count = sum(1 for item in sent if item.kind == "fixed")
    return Outcome(
        scheme, lam, True, len(sent), len(segments), sum(step.order_bits for step in steps), header, repeated,
        fixed_share=100.0 * fixed_count / len(sent), round_trip=ok,
    )
