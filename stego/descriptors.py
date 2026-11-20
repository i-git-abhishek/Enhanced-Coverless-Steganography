"""Phase 3, Part B: one copy-detection descriptor per image, so the receiver can search
the whole database and the cluster ID no longer has to be sent.

SSCD (Pizzi et al., CVPR 2022) is a ResNet-50 trained so that an edited copy of an
image (cropped, rotated, blurred, recompressed ...) lands next to the original. Its
512-number descriptors are L2-normalised, so cosine similarity is a dot product.

"pixels" is a tiny descriptor made from a 16 x 16 thumbnail. It needs no PyTorch and
exists only so the tests and the smoke test can run the whole Phase 3 path. Its
numbers say nothing about the method.

Descriptors of the database images are computed once on the GPU and cached, like the
detections. Received images are described at retrieval time.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from . import cache

FORMAT_VERSION = 1
DESCRIPTOR_CHUNK = 2000
# Every model sees the whole image resized (not cropped) to a square, so a crop attack
# is not undone by a centre crop in the preprocessing.
MODELS = {
    "sscd_disc_mixup": {
        "kind": "sscd",
        "url": "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt",
        "file": "sscd_disc_mixup.torchscript.pt",
        "dim": 512,
        "size": 320,
        "input": "skew_320",  # resize to 320 x 320, as in the SSCD README
    },
    # Descriptor ablation (WP2): general-purpose features, each trained for a different job.
    "resnet101_imagenet": {"kind": "torchvision", "dim": 2048, "size": 224, "input": "skew_224"},  # classification
    "clip_vit_b16": {"kind": "timm", "timm": "vit_base_patch16_clip_224.openai", "dim": None, "size": 224,
                     "input": "skew_224"},  # image-text matching; image embedding after CLIP's projection
    "dinov2_vitb14": {"kind": "timm", "timm": "vit_base_patch14_dinov2.lvd142m", "dim": 768, "size": 224,
                      "input": "skew_224", "pool": True},  # self-supervised; CLS token
    "pixels": {"kind": "pixels", "dim": 256, "input": "gray_16"},
}
SSCD_SIZE = 320
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def prepare_batch(images_bgr: list[np.ndarray], size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """BGR images -> float32 N x 3 x size x size, resized to a square and normalised."""
    batch = []
    for image in images_bgr:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        method = cv2.INTER_AREA if min(rgb.shape[:2]) > size else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (size, size), interpolation=method).astype(np.float32) / 255.0
        batch.append(((rgb - mean) / std).transpose(2, 0, 1))
    return np.stack(batch).astype(np.float32)


class DescriptorError(RuntimeError):
    """Raised when descriptors are missing or cannot be computed."""


def descriptor_key(dataset: str, ids: list[str], model: str) -> str:
    ids_hash = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    payload = json.dumps({"version": FORMAT_VERSION, "ids": ids_hash, "model": model,
                          "input": MODELS[model]["input"]}, sort_keys=True)
    return f"{dataset}_desc_{model}_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class DescriptorModel:
    """Turns BGR images (as read by OpenCV) into L2-normalised descriptors."""

    def __init__(self, name: str, weights: str | Path | None = None, weights_dir: str | Path = "cache/weights"):
        if name not in MODELS:
            raise DescriptorError(f"Unknown descriptor {name!r}. Choose one of {sorted(MODELS)}")
        self.name = name
        self.info = MODELS[name]
        self.dim = self.info["dim"]
        self.size = self.info.get("size", SSCD_SIZE)
        self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        self._model = None
        self._device = None
        kind = self.info["kind"]
        if kind == "sscd":
            self._load_sscd(weights, Path(weights_dir))
        elif kind == "torchvision":
            self._load_torchvision()
        elif kind == "timm":
            self._load_timm()
        if kind != "pixels":
            # One real forward pass, so a wrong model or output shape fails here, not hours later.
            probe = self.describe([np.full((64, 80, 3), 128, np.uint8)])
            if self.dim is None:
                self.dim = int(probe.shape[1])
            elif probe.shape[1] != self.dim:
                raise DescriptorError(f"{name} gave {probe.shape[1]} numbers per image, expected {self.dim}")

    def _torch(self):
        try:
            import torch
        except ImportError as error:  # pragma: no cover - depends on the machine
            raise DescriptorError(f"{self.name} needs PyTorch. On Kaggle it is already installed.") from error
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch

    def _load_sscd(self, weights, weights_dir: Path) -> None:
        torch = self._torch()
        path = Path(weights) if weights else self._find_weights(weights_dir)
        self._model = torch.jit.load(str(path), map_location=self._device).eval()

    def _load_torchvision(self) -> None:  # pragma: no cover - needs torchvision and the network
        torch = self._torch()
        import torchvision

        model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()  # 2048-number global average pool
        self._model = model.to(self._device).eval()

    def _load_timm(self) -> None:  # pragma: no cover - needs timm and the network
        self._torch()
        try:
            import timm
            from timm.data import resolve_data_config
        except ImportError as error:
            raise DescriptorError(f"{self.name} needs timm: run  pip install timm  (Internet on)") from error
        options = {"num_classes": 0} if self.info.get("pool") else {}
        if self.info["timm"].startswith("vit_base_patch14_dinov2"):
            options["img_size"] = self.size  # DINOv2 is trained at 518; its position embedding is resampled
        try:
            model = timm.create_model(self.info["timm"], pretrained=True, **options)
        except Exception as error:
            raise DescriptorError(
                f"Could not load {self.info['timm']} from timm. Turn on Internet in the notebook settings."
            ) from error
        config = resolve_data_config({}, model=model)
        self.mean = np.asarray(config["mean"], np.float32)
        self.std = np.asarray(config["std"], np.float32)
        self._model = model.to(self._device).eval()

    def _find_weights(self, weights_dir: Path) -> Path:
        info = MODELS[self.name]
        # An attached Kaggle dataset can hold the file, so Internet is not needed.
        for base in [weights_dir, *cache.default_read_dirs()]:
            if not base.is_dir():
                continue
            direct = base / info["file"]
            if direct.is_file():
                return direct
            for found in base.rglob(info["file"]):
                return found
        import torch

        weights_dir.mkdir(parents=True, exist_ok=True)
        target = weights_dir / info["file"]
        print(f"[descriptors] Downloading SSCD weights to {target}")
        try:
            torch.hub.download_url_to_file(info["url"], str(target) + ".tmp")
        except Exception as error:  # pragma: no cover - needs the network
            raise DescriptorError(
                "Could not download the SSCD weights. Turn on Internet in the Kaggle notebook settings, "
                f"or attach a dataset containing {info['file']} (from {info['url']})."
            ) from error
        os.replace(str(target) + ".tmp", target)
        return target

    @property
    def device(self) -> str:
        return "cpu" if self._device is None else str(self._device)

    def describe(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        if not images_bgr:
            return np.zeros((0, self.dim), np.float32)
        if self.name == "pixels":
            rows = []
            for image in images_bgr:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
                small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
                rows.append(small - small.mean())
            return _normalise(np.stack(rows))
        import torch

        tensor = torch.from_numpy(prepare_batch(images_bgr, self.size, self.mean, self.std)).to(self._device)
        with torch.inference_mode():
            output = self._model(tensor).float().cpu().numpy()
        return _normalise(output.reshape(len(images_bgr), -1))


def find_descriptors(key: str, cache_dir: Path, read_dirs: list[Path]) -> Path | None:
    return cache._find_complete(key, [Path(cache_dir), *read_dirs])


def load_descriptors(ids: list[str], dataset: str, model: str, cache_dir: Path, read_dirs: list[Path]) -> np.ndarray:
    key = descriptor_key(dataset, ids, model)
    folder = find_descriptors(key, cache_dir, read_dirs)
    if folder is None:
        raise DescriptorError(
            f"No finished {model} descriptors found for {dataset} (cache key {key}). "
            "Run: python run.py descriptors --config <your config>"
        )
    vectors = np.load(folder / "descriptors.npy")
    stored_ids = json.loads((folder / "ids.json").read_text(encoding="utf-8"))
    dim = MODELS[model]["dim"]
    if stored_ids != ids or vectors.ndim != 2 or vectors.shape[0] != len(ids) or (dim and vectors.shape[1] != dim):
        raise DescriptorError(f"Cached descriptors do not match the dataset: {folder}")
    return vectors


def run_descriptor_extraction(config, ids: list[str], image_paths: dict[str, Path], cache_dir: Path,
                              read_dirs: list[Path], batch_size: int = 32, weights=None, log=print) -> Path:
    """Computes or resumes the database descriptors. Saves chunk by chunk."""
    from .recovery import read_image

    key = descriptor_key(config.dataset, ids, config.descriptor)
    existing = find_descriptors(key, cache_dir, read_dirs)
    if existing is not None:
        log(f"[descriptors] Finished descriptors already exist: {existing}")
        return existing
    model = DescriptorModel(config.descriptor, weights, Path(cache_dir) / "weights")
    log(f"[descriptors] {config.descriptor} on {model.device}, {len(ids)} images")
    if model.device == "cpu" and config.descriptor != "pixels":
        log("[descriptors] WARNING: no GPU found. Turn on the GPU in Kaggle settings.")
    folder = Path(cache_dir) / (key + ".partial")
    folder.mkdir(parents=True, exist_ok=True)
    chunks = (len(ids) + DESCRIPTOR_CHUNK - 1) // DESCRIPTOR_CHUNK
    for chunk in range(chunks):
        target = folder / f"chunk_{chunk:05d}.npy"
        if target.is_file():
            continue
        part = ids[chunk * DESCRIPTOR_CHUNK : (chunk + 1) * DESCRIPTOR_CHUNK]
        vectors = []
        for start in range(0, len(part), batch_size):
            vectors.append(model.describe([read_image(image_paths[i]) for i in part[start : start + batch_size]]))
        np.save(str(target) + ".tmp.npy", np.concatenate(vectors))
        os.replace(str(target) + ".tmp.npy", target)
        log(f"[descriptors] chunk {chunk + 1}/{chunks} saved")
    vectors = np.concatenate([np.load(folder / f"chunk_{c:05d}.npy") for c in range(chunks)])
    final = Path(cache_dir) / key
    temp = Path(cache_dir) / (key + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    np.save(temp / "descriptors.npy", vectors.astype(np.float32))
    (temp / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    cache.mark_complete(temp, {"dataset": config.dataset, "model": config.descriptor, "images": len(ids),
                               "dim": int(vectors.shape[1]), "device": model.device})
    if final.exists():
        shutil.rmtree(final)
    os.replace(temp, final)
    shutil.rmtree(folder, ignore_errors=True)
    log(f"[descriptors] Finished: {final}")
    return final
