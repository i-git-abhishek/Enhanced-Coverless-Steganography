"""Tests for Phase 2: fixed-order schemes, the length header and the dynamic programming sender."""

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from stego.config import build_config
from stego.hiding import bytes_to_segments
from stego.mapping import ImageIndex
from stego.ordering import (
    HEADER_BITS,
    INFINITY,
    _options,
    build_lookup,
    canonical_sequences,
    decode,
    frame_message,
    order_array_bits,
    plan,
    realise,
    run_scheme,
    unframe_message,
)
from stego.ordering_experiment import run_ordering
from stego.pipeline import resolve_paths, run_baseline
from stego.synthetic import create_synthetic, write_synthetic_cache


def _index(sequences, n=2, m=3):
    """A tiny index where each image's area order is given directly."""
    sequences = [tuple(s) for s in sequences]
    counts = np.zeros((len(sequences), 2**n), dtype=np.uint8)
    for row, sequence in enumerate(sequences):
        for group in sequence:
            counts[row, group] += 1
    ids = [f"img{i}" for i in range(len(sequences))]
    labels = [tuple(f"l{g}" for g in s) for s in sequences]
    dictionary = {f"l{g}": g for g in range(2**n)}
    clusters = np.zeros(len(sequences), dtype=np.int32)
    return ImageIndex(ids, n, m, dictionary, sequences, labels, clusters, counts, {0: np.arange(len(sequences))})


# ---------------------------------------------------------------- framing


@pytest.mark.parametrize("n", [3, 4, 5, 6])
@pytest.mark.parametrize("length", [0, 1, 10, 100, 1024])
def test_header_round_trip(n, length):
    message = np.random.default_rng(length).bytes(length)
    segments = frame_message(message, n)
    assert len(segments) == math.ceil((HEADER_BITS + 8 * length) / n)
    assert unframe_message(segments, n) == message
    # The last image may carry extra segments; the header makes the receiver drop them.
    assert unframe_message(segments + [1, 2, 0], n) == message


def test_header_too_short_is_detected():
    segments = frame_message(b"hello", 6)
    assert unframe_message(segments[:-2], 6) is None


# ---------------------------------------------------------------- reading order


def test_left_to_right_order_uses_box_centres():
    index = _index([(0, 1, 2)], n=2, m=3)
    objects = [[
        {"label": "l0", "score": 0.99, "box": [200, 0, 400, 300]},  # largest, centre x = 300
        {"label": "l1", "score": 0.99, "box": [0, 0, 150, 200]},    # centre x = 75
        {"label": "l2", "score": 0.99, "box": [150, 0, 250, 100]},  # centre x = 200
    ]]
    assert canonical_sequences(index, objects, 0.9, "left_to_right") == [(1, 2, 0)]
    assert canonical_sequences(index, objects, 0.9, "area") == [(0, 1, 2)]


# ---------------------------------------------------------------- the sender is optimal


def _brute_force(segments, scheme, lookup, m, lam):
    paper_bits = order_array_bits(m)

    @lru_cache(maxsize=None)
    def best(i):
        if i == len(segments):
            return (0, 0.0)
        results = []
        for option in _options(list(segments), i, scheme, lookup, m, paper_bits):
            rest = best(i + option.length)
            if rest is not None:
                results.append((rest[0] + 1, rest[1] + option.order_bits))
        if not results:
            return None
        key = (lambda r: (r[0], r[1])) if math.isinf(lam) else (lambda r: (lam * r[0] + r[1], r[0]))
        return min(results, key=key)

    return best(0)


@pytest.mark.parametrize("scheme", ["paper-dp", "A", "B", "C"])
@pytest.mark.parametrize("lam", [0.0, 3.0, INFINITY])
def test_plan_matches_brute_force(scheme, lam):
    rng = np.random.default_rng(7)
    fixed = [tuple(rng.integers(0, 4, size=rng.integers(1, 4)).tolist()) for _ in range(12)]
    fixed += [(g,) for g in range(4)]  # every group can be sent alone, so A is always possible
    lookup = build_lookup(fixed, fixed)
    for trial in range(30):
        segments = tuple(rng.integers(0, 4, size=rng.integers(1, 12)).tolist())
        steps = plan(list(segments), scheme, lookup, 3, lam)
        expected = _brute_force(segments, scheme, lookup, 3, lam)
        if expected is None:
            assert steps is None
            continue
        images, bits = len(steps), sum(step.order_bits for step in steps)
        assert images == expected[0]
        assert bits == pytest.approx(expected[1])
        assert sum(step.length for step in steps) == len(segments)


def test_scheme_a_sends_no_order_data_and_decodes():
    fixed = [(0,), (1,), (2,), (3,), (0, 1), (2, 3, 1)]
    index = _index(fixed, n=2, m=3)
    lookup = build_lookup(fixed, fixed)
    segments = [2, 3, 1, 0, 1, 3]
    steps = plan(segments, "A", lookup, 3)
    assert [step.order_bits for step in steps] == [0.0] * len(steps)
    assert len(steps) == 3  # (2,3,1) + (0,1) + (3,)
    sent, _ = realise(steps, segments, fixed, index, np.random.default_rng(0))
    assert all(item.kind == "fixed" and not item.order and not item.mask for item in sent)
    assert decode(sent, fixed, index) == segments


def test_scheme_a_can_overhang_only_at_the_end():
    fixed = [(0, 1, 2), (3,)]
    lookup = build_lookup(fixed, fixed)
    assert plan([3, 0, 1], "A", lookup, 3) is not None  # last image carries 0,1 and an extra 2
    assert plan([0, 1, 3], "A", lookup, 3) is None  # 0,1 then 3 would need the image to stop early


def test_scheme_a_reports_when_a_message_cannot_be_hidden():
    fixed = [(0, 1), (1,)]  # no image holds group 0 alone, and nothing holds 2 or 3
    index = _index(fixed, n=2, m=2)
    lookups = {"left_to_right": build_lookup(fixed, fixed), "area": build_lookup(fixed, fixed)}
    outcome = run_scheme(b"\xff", "A", INFINITY, index, fixed, fixed, lookups, np.random.default_rng(0))
    assert outcome.complete is False


def test_scheme_b_mask_keeps_order():
    fixed = [(2, 0, 3, 1)]
    index = _index(fixed, n=2, m=4)
    lookup = build_lookup(fixed, fixed)
    steps = plan([2, 3], "B", lookup, 4)
    sent, _ = realise(steps, [2, 3], fixed, index, np.random.default_rng(0))
    assert sent[0].mask == (True, False, True, False)
    assert steps[0].order_bits == 4.0  # one bit per object in that image
    assert decode(sent, fixed, index) == [2, 3]
    # A mask cannot change the order, so 3 then 2 needs the image twice, one segment each.
    assert [step.length for step in plan([3, 2], "B", lookup, 4)] == [1, 1]


def test_scheme_c_uses_the_flag_only_when_worth_it():
    fixed = [(0,), (1,), (2,), (3,), (1, 0)]  # fixed order 1,0; the paper's order array can send 0,1
    index = _index(fixed, n=2, m=2)
    lookup = build_lookup(fixed, fixed)
    cheap = plan([0, 1], "C", lookup, 2, lam=0.0)
    assert [step.kind for step in cheap] == ["fixed", "fixed"]  # 2 images, 2 bits
    few = plan([0, 1], "C", lookup, 2, lam=INFINITY)
    assert [step.kind for step in few] == ["order"]  # 1 image, 1 + log2(4) bits
    sent, _ = realise(few, [0, 1], fixed, index, np.random.default_rng(0))
    assert decode(sent, fixed, index) == [0, 1]


def test_every_scheme_round_trips_on_random_messages():
    rng = np.random.default_rng(3)
    area = [tuple(rng.integers(0, 4, size=rng.integers(1, 4)).tolist()) for _ in range(40)]
    area += [(g,) for g in range(4)]
    fixed = [tuple(reversed(s)) for s in area]  # a different fixed order than the area order
    index = _index(area, n=2, m=3)
    lookups = {"left_to_right": build_lookup(fixed, area), "area": build_lookup(area, area)}
    for trial in range(20):
        message = rng.bytes(int(rng.integers(1, 30)))
        for scheme in ("paper-greedy", "paper-dp", "A", "A-area", "B", "C"):
            for lam in (0.0, 2.0, INFINITY):
                outcome = run_scheme(message, scheme, lam, index, fixed, area, lookups, np.random.default_rng(trial))
                assert outcome.complete, (scheme, lam)
                assert outcome.round_trip, (scheme, lam)


# ---------------------------------------------------------------- the full experiment


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("ordering")
    ids, data_root = create_synthetic(root / "data")
    write_synthetic_cache(ids, root / "cache")
    return root, data_root


def _config(**extra):
    return build_config({"dataset": "synthetic", "k": 8, "capacity_bytes": [10, 100], "capacity_trials": 2,
                         "robustness_trials": 2, "robustness_segments": 8, "bootstrap_samples": 50, **extra})


@pytest.mark.parametrize("n,m", [(6, 4), (3, 4), (6, 6)])
def test_ordering_run_is_saved_and_checks_pass(synthetic, n, m):
    root, data_root = synthetic
    paths = resolve_paths(str(data_root), str(root / "cache"), str(root / f"exp_{n}_{m}" / "runs"), [])
    paths.read_dirs = []
    folder = run_ordering(_config(n=n, M=m), paths, trials=3, allow_dirty=True, log=lambda message: None)
    checks = json.loads((folder / "checks.json").read_text())
    assert all(item["passed"] for item in checks.values()), checks
    for name in ("ordering_messages.csv", "ordering_summary.csv", "summary.json", "config.json", "source.tar.gz"):
        assert (folder / name).is_file()
    assert (folder.parent.parent / "ORDERING_DASHBOARD.md").is_file()
    assert folder.parent.name == "ordering_runs"


def test_paper_greedy_matches_the_baseline_capacity_run(synthetic):
    """The ordering run's paper-greedy rows must repeat the baseline's capacity numbers exactly."""
    root, data_root = synthetic
    paths = resolve_paths(str(data_root), str(root / "cache"), str(root / "exp_match" / "runs"), [])
    paths.read_dirs = []
    config = _config(n=6, M=4)
    baseline = run_baseline(config, paths, allow_dirty=True, log=lambda message: None)
    ordering = run_ordering(config, paths, trials=2, allow_dirty=True, log=lambda message: None)
    capacity = pd.read_csv(baseline / "capacity.csv")
    messages = pd.read_csv(ordering / "ordering_messages.csv")
    greedy = messages[messages["scheme"] == "paper-greedy"]
    merged = capacity.merge(greedy, on=["message_bytes", "trial"], suffixes=("_baseline", "_ordering"))
    assert len(merged) == len(capacity)
    assert (merged["images_baseline"] == merged["images_ordering"]).all()
    segments, _ = bytes_to_segments(b"x" * 10, 6)
    assert len(segments) == 14
