"""Phase 3, Part C: the retrievability check.

Before anything is sent, every database image is attacked with each of the 22 Table IX
attacks and searched for with SSCD. An image counts as "usable" if it is found again
for at least `usable_threshold` of the attacks. The sender only picks usable images,
and the receiver only searches among them: both sides hold the database, so both can
run this check and get the same list. The capacity bound becomes log2(usable images).

The check uses plain SSCD (no alignment) because it runs 22 searches per image. That
makes it cautious: alignment can only rescue images SSCD alone would miss.

The random noise of the check uses its own random stream, so the robustness test later
sees different noise from the check.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zlib
from pathlib import Path

import numpy as np

from . import cache
from .attacks import TABLE_IX_ORDER, build_attacks
from .descriptors import descriptor_key
from .recovery import read_image

FORMAT_VERSION = 1
CHECK_CHUNK = 500
CHECK_STREAM = 77  # differs from every stream the robustness test uses


def check_key(config, ids: list[str]) -> str:
    payload = json.dumps(
        {
            "version": FORMAT_VERSION,
            "descriptors": descriptor_key(config.dataset, ids, config.descriptor),
            "attacks": list(TABLE_IX_ORDER),
            "gamma": config.gamma_exponent,
            "seed": config.seed,
            "conf_threshold": config.conf_threshold,  # decides which images can be sent at all
        },
        sort_keys=True,
    )
    return f"{config.dataset}_retrievability_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def check_rng(seed: int, attack_name: str, image_index: int) -> np.random.Generator:
    return np.random.default_rng([seed, CHECK_STREAM, zlib.crc32(attack_name.encode("utf-8")), image_index])


def sendable_images(index) -> np.ndarray:
    """Images with at least one object above the threshold. Only these can carry data."""
    return np.array([bool(sequence) for sequence in index.sequences])


def usable_mask(hits: np.ndarray, sendable: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0:
        return sendable.copy()
    return sendable & (hits.mean(axis=1) >= threshold - 1e-9)


def run_check(config, ids, image_paths, descriptors: np.ndarray, model, sendable: np.ndarray,
              cache_dir: Path, read_dirs: list[Path], log=print) -> np.ndarray:
    """Returns hits[image, attack]: was the attacked image found again? Cached and resumable."""
    key = check_key(config, ids)
    found = cache._find_complete(key, [Path(cache_dir), *read_dirs])
    if found is not None:
        log(f"[retrievability] Using saved check: {found}")
        return np.load(found / "hits.npy").astype(bool)

    attacks = build_attacks(config.gamma_exponent)
    names = list(TABLE_IX_ORDER)
    search = np.flatnonzero(sendable)
    database = np.ascontiguousarray(descriptors[search], dtype=np.float32)
    position = {int(image): row for row, image in enumerate(search)}
    log(f"[retrievability] Attacking {len(search)} images {len(names)} times each with {config.descriptor}")

    folder = Path(cache_dir) / (key + ".partial")
    folder.mkdir(parents=True, exist_ok=True)
    chunks = (len(ids) + CHECK_CHUNK - 1) // CHECK_CHUNK
    for chunk in range(chunks):
        target = folder / f"chunk_{chunk:05d}.npy"
        if target.is_file():
            continue
        rows = range(chunk * CHECK_CHUNK, min((chunk + 1) * CHECK_CHUNK, len(ids)))
        hits = np.zeros((len(rows), len(names)), dtype=np.uint8)
        for offset, image_index in enumerate(rows):
            if not sendable[image_index]:
                continue
            original = read_image(image_paths[ids[image_index]])
            attacked = [attacks[name](original, check_rng(config.seed, name, image_index)) for name in names]
            best = np.argmax(model.describe(attacked) @ database.T, axis=1)
            hits[offset] = best == position[image_index]
        np.save(str(target) + ".tmp.npy", hits)
        os.replace(str(target) + ".tmp.npy", target)
        log(f"[retrievability] chunk {chunk + 1}/{chunks} saved")

    hits = np.concatenate([np.load(folder / f"chunk_{c:05d}.npy") for c in range(chunks)])
    final = Path(cache_dir) / key
    temp = Path(cache_dir) / (key + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    np.save(temp / "hits.npy", hits)
    (temp / "attacks.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    rates = hits[sendable].mean(axis=0) * 100.0 if sendable.any() else np.zeros(len(names))
    cache.mark_complete(temp, {"images": len(ids), "searched": int(sendable.sum()), "model": config.descriptor,
                               "found_percent_per_attack": dict(zip(names, np.round(rates, 2).tolist()))})
    if final.exists():
        shutil.rmtree(final)
    os.replace(temp, final)
    shutil.rmtree(folder, ignore_errors=True)
    log(f"[retrievability] Finished: {final}")
    return hits.astype(bool)


def usable_report(hits: np.ndarray, sendable: np.ndarray, thresholds=(0.5, 0.8, 0.9, 0.95, 1.0)) -> dict:
    """How many images survive each threshold, and the bound log2(usable) for each."""
    report = {}
    for threshold in thresholds:
        count = int(usable_mask(hits, sendable, threshold).sum())
        report[f"{threshold:g}"] = {"usable_images": count,
                                    "bound_bits_per_image": round(float(np.log2(count)), 3) if count else 0.0}
    return report
