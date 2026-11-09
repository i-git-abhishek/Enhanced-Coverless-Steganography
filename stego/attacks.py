"""The 22 attacks of the paper's Table IX, plus a required no-attack control.

Where Table IX leaves a detail open, the choice is listed in MANUAL.md section 12.
"""

from __future__ import annotations

import zlib
from typing import Callable

import cv2
import numpy as np

NO_ATTACK = "No attack"
GEOMETRIC = (
    "Center Cropping 5%", "Center Cropping 10%", "Center Cropping 20%",
    "Edge Cropping 5%", "Edge Cropping 10%", "Edge Cropping 20%",
    "Rotation 10°", "Rotation 30°", "Rotation 50°",
    "Translation (36, 20)", "Translation (40, 25)", "Translation (80, 50)",
    "Scaling",
)
NON_GEOMETRIC = (
    "Gauss Noise", "Salt & Pepper Noise", "Speckle Noise",
    "Median Filter", "Mean Filter", "Gauss Filter",
    "JPEG Compression", "Histogram Equalization", "Gamma Correction",
)
TABLE_IX_ORDER = GEOMETRIC + NON_GEOMETRIC


def center_crop(image: np.ndarray, ratio: float) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    crop_h, crop_w = int(height * np.sqrt(ratio)), int(width * np.sqrt(ratio))
    y, x = (height - crop_h) // 2, (width - crop_w) // 2
    out[y : y + crop_h, x : x + crop_w] = 0
    return out


def edge_crop(image: np.ndarray, ratio: float) -> np.ndarray:
    height, width = image.shape[:2]
    keep = np.sqrt(1.0 - ratio)
    keep_h, keep_w = int(height * keep), int(width * keep)
    y, x = (height - keep_h) // 2, (width - keep_w) // 2
    out = np.zeros_like(image)
    out[y : y + keep_h, x : x + keep_w] = image[y : y + keep_h, x : x + keep_w]
    return out


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), borderValue=(0, 0, 0))


def translate(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (width, height), borderValue=(0, 0, 0))


def scale(image: np.ndarray, factor: float) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=cv2.INTER_LINEAR)


def _from_unit(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def gauss_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    unit = image.astype(np.float64) / 255.0
    return _from_unit(unit + rng.normal(0.0, sigma, unit.shape))


def speckle_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    unit = image.astype(np.float64) / 255.0
    return _from_unit(unit + unit * rng.normal(0.0, sigma, unit.shape))


def salt_and_pepper(image: np.ndarray, density: float, rng: np.random.Generator) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    corrupted = int(density * height * width)
    chosen = rng.choice(height * width, size=corrupted, replace=False)
    half = corrupted // 2
    ys, xs = np.unravel_index(chosen[:half], (height, width))
    out[ys, xs] = 255
    ys, xs = np.unravel_index(chosen[half:], (height, width))
    out[ys, xs] = 0
    return out


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    converted[:, :, 0] = cv2.equalizeHist(converted[:, :, 0])
    return cv2.cvtColor(converted, cv2.COLOR_YCrCb2BGR)


def gamma_correction(image: np.ndarray, exponent: float) -> np.ndarray:
    table = np.clip(np.rint(((np.arange(256) / 255.0) ** exponent) * 255.0), 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


AttackFunction = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def build_attacks(gamma_exponent: float = 0.8) -> dict[str, AttackFunction]:
    """No attack first, followed by Table IX in the order of Table X."""
    return {
        NO_ATTACK: lambda image, rng: image.copy(),
        "Center Cropping 5%": lambda image, rng: center_crop(image, 0.05),
        "Center Cropping 10%": lambda image, rng: center_crop(image, 0.10),
        "Center Cropping 20%": lambda image, rng: center_crop(image, 0.20),
        "Edge Cropping 5%": lambda image, rng: edge_crop(image, 0.05),
        "Edge Cropping 10%": lambda image, rng: edge_crop(image, 0.10),
        "Edge Cropping 20%": lambda image, rng: edge_crop(image, 0.20),
        "Rotation 10°": lambda image, rng: rotate(image, 10),
        "Rotation 30°": lambda image, rng: rotate(image, 30),
        "Rotation 50°": lambda image, rng: rotate(image, 50),
        "Translation (36, 20)": lambda image, rng: translate(image, 36, 20),
        "Translation (40, 25)": lambda image, rng: translate(image, 40, 25),
        "Translation (80, 50)": lambda image, rng: translate(image, 80, 50),
        "Scaling": lambda image, rng: scale(image, 3.0),
        "Gauss Noise": lambda image, rng: gauss_noise(image, 0.001, rng),
        "Salt & Pepper Noise": lambda image, rng: salt_and_pepper(image, 0.001, rng),
        "Speckle Noise": lambda image, rng: speckle_noise(image, 0.01, rng),
        "Median Filter": lambda image, rng: cv2.medianBlur(image, 3),
        "Mean Filter": lambda image, rng: cv2.blur(image, (3, 3)),
        "Gauss Filter": lambda image, rng: cv2.GaussianBlur(image, (3, 3), 0),
        "JPEG Compression": lambda image, rng: jpeg(image, 90),
        "Histogram Equalization": lambda image, rng: histogram_equalization(image),
        "Gamma Correction": lambda image, rng: gamma_correction(image, gamma_exponent),
    }


def attack_rng(seed: int, attack_name: str, trial: int, position: int) -> np.random.Generator:
    """Noise depends only on these values, not on the order attacks are run in."""
    return np.random.default_rng([seed, zlib.crc32(attack_name.encode("utf-8")), trial, position])
