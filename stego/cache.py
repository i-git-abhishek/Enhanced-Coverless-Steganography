"""Stores detector output and deep features so that sweeps do not need a GPU.

The cache key covers everything that changes the stored values. All detections down
to the detector's own low score threshold are kept, so conf_threshold can be changed
later without another GPU pass. If you change what extract.py computes, raise
FORMAT_VERSION or the names below, so that older caches are not reused.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DETECTOR_NAME = "fasterrcnn_resnet50_fpn/COCO_V1"
FEATURE_NAME = "resnet101/IMAGENET1K_V1/avgpool/resize224"
BOX_SCORE_THRESHOLD = 0.05
CHUNK_SIZE = 1000
FORMAT_VERSION = 1
FEATURE_DIM = 2048


class CacheError(RuntimeError):
    """Raised when required cached data is missing or inconsistent."""


def _digest(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(text).hexdigest()[:16]


def extraction_key(dataset: str, ids: list[str]) -> str:
    ids_hash = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    return f"{dataset}_extract_" + _digest(
        {
            "version": FORMAT_VERSION,
            "ids": ids_hash,
            "detector": DETECTOR_NAME,
            "features": FEATURE_NAME,
            "box_score": BOX_SCORE_THRESHOLD,
        }
    )


def cluster_key(ext_key: str, config) -> str:
    return "clusters_" + _digest(
        {
            "extraction": ext_key,
            "k": config.k,
            "algorithm": config.cluster_algorithm,
            "max_iter": config.cluster_max_iter,
            "seed": config.cluster_seed,
            "hash_threshold": config.hash_threshold,
        }
    )


def default_read_dirs() -> list[Path]:
    """On Kaggle, attached datasets can hold an earlier cache."""
    kaggle_input = Path("/kaggle/input")
    return [kaggle_input] if kaggle_input.is_dir() else []


def _find_complete(key: str, search_dirs: list[Path]) -> Path | None:
    for base in search_dirs:
        direct = base / key
        if (direct / "COMPLETE").is_file():
            return direct
        if base.is_dir():
            for current, dirs, files in os.walk(base, followlinks=True):
                current_path = Path(current)
                if current_path.name == key and "COMPLETE" in files:
                    return current_path
                # Image folders are large and never contain a cache.
                dirs[:] = [d for d in dirs if d not in ("JPEGImages", "train2014", "val2014", "test2014")]
    return None


@dataclass
class Extraction:
    ids: list[str]
    objects: list[list[dict]]
    features: np.ndarray
    key: str
    meta: dict


def chunk_stem(chunk_index: int) -> str:
    return f"chunk_{chunk_index:05d}"


def write_chunk(folder: Path, chunk_index: int, records: list[dict], features: np.ndarray) -> None:
    """Writes one chunk atomically, so an interrupted run can resume safely."""
    if len(records) != len(features):
        raise CacheError("Each record needs exactly one feature vector")
    folder.mkdir(parents=True, exist_ok=True)
    stem = chunk_stem(chunk_index)
    json_tmp = folder / f"{stem}.json.gz.tmp"
    with gzip.open(json_tmp, "wt", encoding="utf-8") as handle:
        json.dump(records, handle)
    npy_tmp = folder / f"{stem}.tmp.npy"
    np.save(npy_tmp, features.astype(np.float32))
    os.replace(npy_tmp, folder / f"{stem}.npy")
    os.replace(json_tmp, folder / f"{stem}.json.gz")


def chunk_is_complete(folder: Path, chunk_index: int) -> bool:
    stem = chunk_stem(chunk_index)
    return (folder / f"{stem}.json.gz").is_file() and (folder / f"{stem}.npy").is_file()


def mark_complete(folder: Path, meta: dict) -> None:
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (folder / "COMPLETE").write_text("ok\n", encoding="utf-8")


def load_extraction(ids: list[str], dataset: str, cache_dir: Path, read_dirs: list[Path]) -> Extraction:
    key = extraction_key(dataset, ids)
    folder = _find_complete(key, [Path(cache_dir), *read_dirs])
    if folder is None:
        raise CacheError(
            f"No finished extraction found for {dataset} (cache key {key}). "
            "Run: python run.py extract --config <your config>"
        )
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    objects: list[list[dict]] = []
    found_ids: list[str] = []
    arrays: list[np.ndarray] = []
    chunks = sorted(folder.glob("chunk_*.json.gz"))
    for chunk in chunks:
        with gzip.open(chunk, "rt", encoding="utf-8") as handle:
            records = json.load(handle)
        for record in records:
            found_ids.append(record["id"])
            objects.append(record["objects"])
        arrays.append(np.load(str(chunk)[: -len(".json.gz")] + ".npy"))
    if found_ids != ids:
        raise CacheError(f"Cached image list does not match the dataset: {folder}")
    features = np.concatenate(arrays, axis=0)
    if features.shape != (len(ids), FEATURE_DIM):
        raise CacheError(f"Cached features have the wrong shape {features.shape}: {folder}")
    return Extraction(ids=ids, objects=objects, features=features, key=key, meta=meta)


def load_clusters(key: str, cache_dir: Path, read_dirs: list[Path]) -> np.ndarray | None:
    folder = _find_complete(key, [Path(cache_dir), *read_dirs])
    return None if folder is None else np.load(folder / "labels.npy")


def save_clusters(key: str, cache_dir: Path, labels: np.ndarray, meta: dict) -> None:
    folder = Path(cache_dir) / key
    temp = Path(cache_dir) / f"{key}.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    np.save(temp / "labels.npy", labels.astype(np.int32))
    mark_complete(temp, meta)
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(temp, folder)
