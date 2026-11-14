"""The 22 attacks of the paper's Table IX, plus a required no-attack control.

Where Table IX leaves a detail open, the choice is listed in MANUAL.md section 12.

Extended attacks (combined and heavy) stress-test retrieval beyond the paper's
original suite. They are enabled by setting ``extended_attacks: true`` in the
config. The retrievability pre-check always uses only the original 22 attacks,
so the extended attacks are genuinely unseen during filtering.
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

# Extended attacks: combined (chain of operations) and heavy (single but stronger).
# These are NOT used in the retrievability pre-check, so they test generalisation.
COMBINED = (
    "JPEG Q50 + Rotation 15°",
    "JPEG Q30 + Gauss Noise (heavy)",
    "Crop 10% + JPEG Q50 + Resize 0.5×",
    "Rotation 20° + Gauss Noise + JPEG Q70",
    "Edge Crop 15% + S&P Noise + JPEG Q50",
    "Rotation 30° + Crop 10% + Gauss Noise (heavy)",
)
HEAVY = (
    "JPEG Quality 50",
    "JPEG Quality 30",
    "JPEG Quality 10",
    "Heavy Gauss Noise",
    "Heavy Salt & Pepper",
    "Large Median Filter 7×7",
)
EXTENDED_ORDER = COMBINED + HEAVY


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


def build_attacks(gamma_exponent: float = 0.8, *, extended: bool = False) -> dict[str, AttackFunction]:
    """No attack first, followed by Table IX in the order of Table X.

    When *extended* is True, combined and heavy attacks are appended after the
    original 22. The retrievability pre-check never passes ``extended=True``,
    so these attacks are unseen during image filtering.
    """
    attacks: dict[str, AttackFunction] = {
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
    if extended:
        attacks.update(_extended_attacks(gamma_exponent))
    return attacks


def _extended_attacks(gamma_exponent: float) -> dict[str, AttackFunction]:
    """Combined and heavy attacks that stress-test retrieval beyond the paper."""
    return {
        # --- Combined attacks: chain of operations in one pass ---
        "JPEG Q50 + Rotation 15°": lambda image, rng: rotate(jpeg(image, 50), 15),
        "JPEG Q30 + Gauss Noise (heavy)": lambda image, rng: gauss_noise(jpeg(image, 30), 0.01, rng),
        "Crop 10% + JPEG Q50 + Resize 0.5×": lambda image, rng: scale(jpeg(center_crop(image, 0.10), 50), 0.5),
        "Rotation 20° + Gauss Noise + JPEG Q70": lambda image, rng: jpeg(gauss_noise(rotate(image, 20), 0.005, rng), 70),
        "Edge Crop 15% + S&P Noise + JPEG Q50": lambda image, rng: jpeg(salt_and_pepper(edge_crop(image, 0.15), 0.005, rng), 50),
        "Rotation 30° + Crop 10% + Gauss Noise (heavy)": lambda image, rng: gauss_noise(center_crop(rotate(image, 30), 0.10), 0.01, rng),
        # --- Heavy individual attacks: stronger than the paper's versions ---
        "JPEG Quality 50": lambda image, rng: jpeg(image, 50),
        "JPEG Quality 30": lambda image, rng: jpeg(image, 30),
        "JPEG Quality 10": lambda image, rng: jpeg(image, 10),
        "Heavy Gauss Noise": lambda image, rng: gauss_noise(image, 0.01, rng),
        "Heavy Salt & Pepper": lambda image, rng: salt_and_pepper(image, 0.01, rng),
        "Large Median Filter 7×7": lambda image, rng: cv2.medianBlur(image, 7),
    }


def attack_rng(seed: int, attack_name: str, trial: int, position: int) -> np.random.Generator:
    """Noise depends only on these values, not on the order attacks are run in."""
    return np.random.default_rng([seed, zlib.crc32(attack_name.encode("utf-8")), trial, position])
