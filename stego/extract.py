"""GPU pass: Faster R-CNN detections and ResNet-101 features for every image.

PyTorch is imported only here, so tests and sweeps on cached data do not need it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import cache


def _load_models(device):
    import torch
    from torchvision import models, transforms
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    detector = fasterrcnn_resnet50_fpn(
        weights=weights, box_score_thresh=cache.BOX_SCORE_THRESHOLD
    ).to(device).eval()
    categories = list(weights.meta["categories"])

    resnet_weights = models.ResNet101_Weights.IMAGENET1K_V1
    resnet = models.resnet101(weights=resnet_weights)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
    return {
        "detector": detector,
        "categories": categories,
        "det_transform": weights.transforms(),
        "backbone": backbone,
        "feat_transform": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    }


def _records_from_prediction(image_id: str, prediction, categories: list[str]) -> dict:
    boxes = prediction["boxes"].detach().cpu().numpy().tolist()
    scores = prediction["scores"].detach().cpu().numpy().tolist()
    labels = prediction["labels"].detach().cpu().numpy().tolist()
    objects = []
    for box, score, label_id in zip(boxes, scores, labels):
        label = categories[label_id]
        if label in ("N/A", "__background__"):
            continue
        objects.append({"label": label, "score": float(score), "box": [float(v) for v in box]})
    return {"id": image_id, "objects": objects}


def _process_batch(paths, ids, models, device):
    import torch
    from PIL import Image

    pil_images = []
    for path in paths:
        with Image.open(path) as handle:
            pil_images.append(handle.convert("RGB"))
    with torch.inference_mode():
        det_inputs = [models["det_transform"](image).to(device) for image in pil_images]
        predictions = models["detector"](det_inputs)
        feat_inputs = torch.stack([models["feat_transform"](image) for image in pil_images]).to(device)
        features = models["backbone"](feat_inputs).flatten(1).detach().cpu().numpy()
    records = [
        _records_from_prediction(image_id, prediction, models["categories"])
        for image_id, prediction in zip(ids, predictions)
    ]
    return records, features


def run_extraction(
    config,
    ids: list[str],
    paths: dict[str, Path],
    cache_dir: Path,
    read_dirs: list[Path],
    batch_size: int = 1,
    limit: int = 0,
    shard: int = 0,
    shards: int = 1,
    max_hours: float = 0.0,
) -> Path | None:
    """Runs or resumes extraction. With limit > 0, it only prints a preview.

    With shards > 1, this process only handles chunks whose index % shards == shard, so
    one process per GPU can share the work; a later run with shards == 1 finds every
    chunk done and marks the cache complete. With max_hours > 0, no new chunk is started
    once that much time has passed, so a Kaggle version can finish and save its output;
    running again resumes from the first missing chunk.
    """
    import time

    import torch

    started = time.monotonic()
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError(f"shard must be in 0..{shards - 1}, got {shard}")
    tag = f"[extract] shard {shard}/{shards}:" if shards > 1 else "[extract]"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] device: {device}")
    if device.type == "cpu":
        print("[extract] WARNING: no GPU found. Turn on the GPU in Kaggle settings before a full extraction.")

    key = cache.extraction_key(config.dataset, ids)
    if limit <= 0:
        existing = cache._find_complete(key, [Path(cache_dir), *read_dirs])
        if existing is not None:
            print(f"[extract] Finished extraction already exists: {existing}")
            return existing

    if limit > 0:
        models = _load_models(device)
        preview_ids = ids[:limit]
        records, features = _process_batch([paths[i] for i in preview_ids], preview_ids, models, device)
        for record in records:
            above = [o for o in record["objects"] if o["score"] > config.conf_threshold]
            labels = [o["label"] for o in above]
            print(f"[preview] {record['id']}: {len(record['objects'])} raw detections, above threshold: {labels}")
        print(f"[preview] feature shape: {features.shape}. Nothing was saved.")
        return None

    folder = Path(cache_dir) / key
    folder.mkdir(parents=True, exist_ok=True)
    total_chunks = (len(ids) + cache.CHUNK_SIZE - 1) // cache.CHUNK_SIZE
    mine = [c for c in range(total_chunks) if c % shards == shard]
    todo = [c for c in mine if not cache.chunk_is_complete(folder, c)]
    print(f"{tag} {len(mine) - len(todo)} of {len(mine)} chunks already done, {len(todo)} to go", flush=True)
    models = _load_models(device) if todo else None
    for chunk_index in todo:
        if max_hours > 0 and time.monotonic() - started > max_hours * 3600:
            print(f"{tag} time budget of {max_hours} h reached before chunk {chunk_index + 1}; run again to resume",
                  flush=True)
            break
        start = chunk_index * cache.CHUNK_SIZE
        chunk_ids = ids[start : start + cache.CHUNK_SIZE]
        records: list[dict] = []
        vectors: list[np.ndarray] = []
        for offset in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[offset : offset + batch_size]
            batch_records, batch_features = _process_batch(
                [paths[i] for i in batch_ids], batch_ids, models, device
            )
            records.extend(batch_records)
            vectors.append(batch_features)
        cache.write_chunk(folder, chunk_index, records, np.concatenate(vectors, axis=0))
        print(f"{tag} chunk {chunk_index + 1}/{total_chunks} saved", flush=True)

    missing = [c for c in range(total_chunks) if not cache.chunk_is_complete(folder, c)]
    if missing:
        print(f"{tag} INCOMPLETE: {len(missing)} of {total_chunks} chunks still missing. Run extract again to resume.",
              flush=True)
        return None
    if shards > 1:
        print(f"{tag} all chunks present; run extract once without --shard to mark the cache complete", flush=True)
        return None

    import torchvision

    cache.mark_complete(
        folder,
        {
            "dataset": config.dataset,
            "images": len(ids),
            "detector": cache.DETECTOR_NAME,
            "features": cache.FEATURE_NAME,
            "box_score_threshold": cache.BOX_SCORE_THRESHOLD,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
    )
    print(f"[extract] Finished: {folder}")
    return folder


def verify_detections(config, extraction, paths: dict[str, Path], sample: int, seed: int) -> dict:
    """Re-runs the detector on a sample of originals and compares the top-M labels.

    The paper re-runs Faster R-CNN at the receiver. Stored detections are only an
    equivalent shortcut if this agreement is 100%.
    """
    import torch

    from .mapping import top_objects

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _load_models(device)
    rng = np.random.default_rng(seed)
    count = min(sample, len(extraction.ids))
    chosen = sorted(rng.choice(len(extraction.ids), size=count, replace=False).tolist())
    matches = 0
    mismatches = []
    for index in chosen:
        image_id = extraction.ids[index]
        records, _ = _process_batch([paths[image_id]], [image_id], models, device)
        fresh = [o["label"] for o in top_objects(records[0]["objects"], config.conf_threshold, config.M)]
        stored = [o["label"] for o in top_objects(extraction.objects[index], config.conf_threshold, config.M)]
        if fresh == stored:
            matches += 1
        elif len(mismatches) < 10:
            mismatches.append({"image": image_id, "stored": stored, "fresh": fresh})
    result = {"sampled": count, "matching": matches, "agreement_percent": 100.0 * matches / count, "examples": mismatches}
    print(json.dumps(result, indent=2))
    return result
