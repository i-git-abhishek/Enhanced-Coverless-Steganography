"""Capacity (IV-B1) and robustness (IV-B2) experiments with raw, per-segment results."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .attacks import GEOMETRIC, NO_ATTACK, NON_GEOMETRIC, TABLE_IX_ORDER, attack_rng, build_attacks
from .hiding import bytes_to_segments, hide_segments, segments_to_bytes
from .mapping import ImageIndex
from .recovery import Retriever, decode_image, read_image


def auxiliary_bits_per_image(m: int, k: int) -> dict[str, float]:
    """Secret auxiliary bits per stego image. Excludes the marker and message padding."""
    patterns = sum(math.perm(m, used) for used in range(1, m + 1))
    cluster_bits = math.log2(k) if k > 1 else 0.0
    return {
        "aux_bits_fixed": m * math.ceil(math.log2(m + 1)) + math.ceil(cluster_bits),
        "aux_bits_tight": math.log2(patterns) + cluster_bits,
    }


def _message_rng(seed: int, stream: int, a: int, b: int) -> np.random.Generator:
    return np.random.default_rng([seed, stream, a, b])


def utilisation_percent(stego, index: ImageIndex) -> float:
    available = sum(len(index.sequences[item.image_index]) for item in stego)
    mapped = sum(item.aux.mapped_count for item in stego)
    return 100.0 * mapped / available if available else 0.0


def run_capacity(config, index: ImageIndex, retriever: Retriever, paths: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    rows = []
    round_trips_total = 0
    round_trips_ok = 0
    retrieval_round_trips_total = 0
    retrieval_round_trips_ok = 0
    aux_bits = auxiliary_bits_per_image(config.M, len(index.cluster_members))
    for length in config.capacity_bytes:
        for trial in range(config.capacity_trials):
            rng = _message_rng(config.seed, 1, length, trial)
            message = rng.bytes(length)
            segments, padding = bytes_to_segments(message, config.n)
            result = hide_segments(segments, index, rng, padding)
            row = {
                "message_bytes": length,
                "trial": trial,
                "complete": result.complete,
                "completion_percent": result.completion_percent,
                "segments": len(segments),
                "images": len(result.stego),
                "repeated_images": result.repeated_images,
                "utilisation_percent": utilisation_percent(result.stego, index),
                "message_bits_per_image": 8 * length / len(result.stego) if result.stego else 0.0,
                **aux_bits,
            }
            rows.append(row)
            if not result.complete:
                continue
            recovered = []
            for item in result.stego:
                recovered.extend(decode_image(item.image_index, item.aux, index))
            round_trips_total += 1
            round_trips_ok += int(segments_to_bytes(recovered, config.n, padding) == message)
            if trial == 0:
                # A full byte-to-byte pass through retrieval on unattacked images.
                recovered = []
                for item in result.stego:
                    image = read_image(paths[index.ids[item.image_index]])
                    found = retriever.retrieve(image, item.aux.cluster)
                    recovered.extend(decode_image(found.image_index, item.aux, index))
                retrieval_round_trips_total += 1
                retrieval_round_trips_ok += int(segments_to_bytes(recovered, config.n, padding) == message)
    frame = pd.DataFrame(rows)
    checks = {
        "round_trips_total": round_trips_total,
        "round_trips_ok": round_trips_ok,
        "retrieval_round_trips_total": retrieval_round_trips_total,
        "retrieval_round_trips_ok": retrieval_round_trips_ok,
    }
    return frame, checks


def run_robustness(config, index: ImageIndex, retriever: Retriever, paths: dict[str, Path], log=print) -> pd.DataFrame:
    """Runs each attack on the same trial messages, as in Table X."""
    trials = []
    for trial in range(config.robustness_trials):
        rng = _message_rng(config.seed, 2, config.robustness_segments, trial)
        segments = rng.integers(0, index.groups, size=config.robustness_segments).tolist()
        result = hide_segments(segments, index, rng, padding=0)
        if not result.complete:
            raise RuntimeError(
                f"Robustness message {trial} could not be hidden ({result.completion_percent:.1f}% done). "
                "The index is incomplete."
            )
        trials.append((segments, result.stego))

    rows = []
    attacks = build_attacks(config.gamma_exponent)
    for attack_number, (name, attack) in enumerate(attacks.items(), start=1):
        for trial, (segments, stego) in enumerate(trials):
            segment_position = 0
            for image_position, item in enumerate(stego):
                original = read_image(paths[index.ids[item.image_index]])
                attacked = attack(original, attack_rng(config.seed, name, trial, image_position))
                found = retriever.retrieve(attacked, item.aux.cluster)
                decoded = decode_image(found.image_index, item.aux, index)
                for value in decoded:
                    expected = segments[segment_position]
                    rows.append(
                        {
                            "attack": name,
                            "trial": trial,
                            "image_position": image_position,
                            "segment_position": segment_position,
                            "stego_image": index.ids[item.image_index],
                            "retrieved_image": index.ids[found.image_index],
                            "retrieval_hit": found.image_index == item.image_index,
                            "best_score": round(found.best_score, 6),
                            "second_score": round(found.second_score, 6),
                            "expected_segment": expected,
                            "decoded_segment": -1 if value is None else value,
                            "correct": value == expected,
                        }
                    )
                    segment_position += 1
            if segment_position != len(segments):
                raise AssertionError("Decoded segment count does not match the message")
        attack_rows = [row for row in rows if row["attack"] == name]
        accuracy = 100.0 * sum(row["correct"] for row in attack_rows) / len(attack_rows)
        log(f"[robustness] {attack_number:2d}/{len(attacks)} {name:26s} ACC {accuracy:6.2f}%")
    return pd.DataFrame(rows)


def bootstrap_interval(per_trial: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    """Percentile interval, resampling whole messages."""
    if len(per_trial) == 1:
        value = float(per_trial[0])
        return value, value
    rng = np.random.default_rng([seed, 3])
    draws = rng.integers(0, len(per_trial), size=(samples, len(per_trial)))
    means = per_trial[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise_robustness(segments: pd.DataFrame, config, paper_row: dict | None) -> pd.DataFrame:
    rows = []
    for name in dict.fromkeys(segments["attack"]):
        part = segments[segments["attack"] == name]
        per_trial = part.groupby("trial")["correct"].mean().to_numpy(dtype=float) * 100.0
        low, high = bootstrap_interval(per_trial, config.bootstrap_samples, config.seed)
        images = part.drop_duplicates(["trial", "image_position"])
        acc = 100.0 * part["correct"].mean()
        paper = paper_row.get(name) if paper_row else None
        rows.append(
            {
                "attack": name,
                "acc_percent": round(acc, 2),
                "ci95_low": round(low, 2),
                "ci95_high": round(high, 2),
                "retrieval_percent": round(100.0 * images["retrieval_hit"].mean(), 2),
                "paper_acc_percent": paper,
                "difference_from_paper": None if paper is None else round(acc - paper, 2),
                "segments": len(part),
            }
        )
    return pd.DataFrame(rows)


def attack_group_means(summary: pd.DataFrame) -> dict[str, float]:
    values = summary.set_index("attack")["acc_percent"]
    return {
        "mean_acc_22_attacks": round(float(values.loc[list(TABLE_IX_ORDER)].mean()), 2),
        "mean_acc_geometric": round(float(values.loc[list(GEOMETRIC)].mean()), 2),
        "mean_acc_non_geometric": round(float(values.loc[list(NON_GEOMETRIC)].mean()), 2),
        "no_attack_acc": round(float(values.loc[NO_ATTACK]), 2),
    }
