"""A small artificial dataset for checking the pipeline without a GPU or real data.

Its numbers say nothing about the method. They only show that the code runs correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import cache

LABELS = [f"label{index:02d}" for index in range(80)]


def create_synthetic(root: Path, images: int = 700) -> tuple[list[str], Path]:
    """Writes images and metadata. Detections are created by write_synthetic_cache."""
    root = Path(root)
    folder = root / "images"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    ids = []
    for number in range(images):
        coarse = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
        image = cv2.resize(coarse, (240, 180), interpolation=cv2.INTER_CUBIC)
        noise = rng.normal(0, 3, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        image_id = f"synthetic_{number:05d}.jpg"
        cv2.imwrite(str(folder / image_id), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        ids.append(image_id)
    (root / "meta.json").write_text(json.dumps({"images": images}), encoding="utf-8")
    return ids, root


def write_synthetic_cache(ids: list[str], cache_dir: Path) -> Path:
    """Stores made-up detections and features in the real cache format."""
    rng = np.random.default_rng(99)
    key = cache.extraction_key("synthetic", ids)
    folder = Path(cache_dir) / key
    records = []
    features = np.zeros((len(ids), cache.FEATURE_DIM), dtype=np.float32)
    for row, image_id in enumerate(ids):
        count = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7], p=[0.05, 0.2, 0.2, 0.2, 0.15, 0.1, 0.05, 0.05]))
        objects = []
        for _ in range(count):
            x1, y1 = rng.uniform(0, 120), rng.uniform(0, 90)
            width, height = rng.uniform(10, 110), rng.uniform(10, 85)
            objects.append(
                {
                    "label": LABELS[int(rng.integers(0, len(LABELS)))],
                    "score": float(rng.uniform(0.91, 0.999)),
                    "box": [float(x1), float(y1), float(x1 + width), float(y1 + height)],
                }
            )
        # One low-score detection per image checks that the threshold is respected.
        objects.append({"label": LABELS[0], "score": 0.5, "box": [0.0, 0.0, 239.0, 179.0]})
        records.append({"id": image_id, "objects": objects})
        center = rng.integers(0, 20)
        features[row] = rng.normal(center, 1.0, cache.FEATURE_DIM)

    for chunk_index, start in enumerate(range(0, len(ids), cache.CHUNK_SIZE)):
        end = start + cache.CHUNK_SIZE
        cache.write_chunk(folder, chunk_index, records[start:end], features[start:end])
    cache.mark_complete(folder, {"dataset": "synthetic", "images": len(ids), "detector": "made up"})
    return folder
