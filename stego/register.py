"""Phase 3, Part B: Fourier–Mellin alignment of a received image to a candidate original.

The receiver does not know how the stego image was attacked. Before comparing the
received image with a candidate, it estimates the rotation, scale and shift that map
one onto the other, undoes them, and compares only the pixels both images still have.

How it works (Reddy and Chatterji, 1996):
1. The magnitude of the Fourier transform ignores shifts. Rotating the image rotates
   it, and scaling the image scales it.
2. In log-polar coordinates rotation and scale become plain shifts, so phase
   correlation finds both at once.
3. After undoing rotation and scale, a second phase correlation finds the shift.

The magnitude spectrum cannot tell a rotation of t from t + 180 degrees, so both are
tried, together with "no change"; the best match wins. Trying "no change" means a
non-geometric attack (noise, filters, JPEG) is never made worse by a bad estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

MIN_OVERLAP = 0.25  # at least this share of the candidate must still be visible to compare


@dataclass(frozen=True)
class Alignment:
    angle: float  # estimated attack: degrees counter-clockwise about the centre
    scale: float  # estimated attack: zoom about the centre
    shift: tuple[float, float]
    score: float  # masked normalised cross-correlation after alignment, -1 to 1
    overlap: float  # share of the candidate's pixels used for the score


def fit_gray(image_bgr_or_gray: np.ndarray, size: int) -> np.ndarray:
    """Grayscale, resized so the longer side is `size`. The aspect ratio is kept."""
    image = image_bgr_or_gray
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    factor = size / max(height, width)
    new_w, new_h = max(1, round(width * factor)), max(1, round(height * factor))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA).astype(np.float32)


def _canvas(image: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Centres the image on a size x size black canvas. Also returns where the image is."""
    canvas = np.zeros((size, size), np.float32)
    mask = np.zeros((size, size), np.float32)
    height, width = image.shape
    y, x = (size - height) // 2, (size - width) // 2
    canvas[y : y + height, x : x + width] = image
    mask[y : y + height, x : x + width] = 1.0
    return canvas, mask


@lru_cache(maxsize=8)
def _window(size: int) -> np.ndarray:
    return cv2.createHanningWindow((size, size), cv2.CV_32F)


@lru_cache(maxsize=8)
def _highpass(size: int) -> np.ndarray:
    frequencies = np.fft.fftshift(np.fft.fftfreq(size))
    x = np.cos(np.pi * frequencies)[None, :] * np.cos(np.pi * frequencies)[:, None]
    return ((1.0 - x) * (2.0 - x)).astype(np.float32)


@lru_cache(maxsize=64)
def _disk(size: int, radius: int) -> np.ndarray:
    """A soft-edged disk. Rotating about the centre keeps the disk's content."""
    y, x = np.mgrid[:size, :size].astype(np.float32) - (size - 1) / 2.0
    distance = np.sqrt(x * x + y * y) / max(radius, 1)
    edge = np.clip((1.0 - distance) / 0.15, 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * edge)).astype(np.float32)


def _log_polar_spectrum(canvas: np.ndarray, size: int, radius: int) -> np.ndarray:
    # Only the disk inside the image is used: rectangle edges and the black corners
    # that rotation adds would otherwise dominate the spectrum.
    windowed = canvas * _disk(size, radius)
    windowed = windowed - windowed.sum() / max(_disk(size, radius).sum(), 1e-6) * _disk(size, radius)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(windowed))).astype(np.float32)
    spectrum *= _highpass(size)
    centre = (size / 2.0, size / 2.0)
    return cv2.warpPolar(spectrum, (size, size), centre, size / 2.0, cv2.WARP_POLAR_LOG | cv2.INTER_LINEAR)


def estimate_rotation_scale(reference: np.ndarray, received: np.ndarray, size: int,
                            radius: int | None = None) -> tuple[float, float]:
    """Rotation (degrees, counter-clockwise) and scale that turn `reference` into `received`.

    `radius` is the radius of the largest disk inside the image (default: half the canvas).
    """
    radius = int(radius or size // 2)
    ref = _log_polar_spectrum(reference, size, radius)
    got = _log_polar_spectrum(received, size, radius)
    (shift_radius, shift_angle), _ = cv2.phaseCorrelate(ref.astype(np.float64), got.astype(np.float64))
    angle = -360.0 * shift_angle / size
    log_base = np.log(size / 2.0) / size
    # Zooming the image in shrinks its spectrum, hence the minus sign.
    scale = float(np.exp(-shift_radius * log_base))
    angle = (angle + 180.0) % 360.0 - 180.0
    return float(angle), scale


def masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    keep = mask > 0.5
    if not keep.any():
        return -1.0
    x = a[keep].astype(np.float64)
    y = b[keep].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / denominator) if denominator > 1e-9 else -1.0


def _warp(image: np.ndarray, matrix: np.ndarray, size: int) -> np.ndarray:
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)


def _compare(reference, ref_mask, received, got_mask, matrix, size) -> tuple[float, float, tuple[float, float]]:
    """Applies `matrix`, then finds and applies the remaining shift, and scores the overlap."""
    warped = _warp(received, matrix, size)
    warped_mask = _warp(got_mask, matrix, size)
    window = _window(size).astype(np.float64)
    (dx, dy), _ = cv2.phaseCorrelate(reference.astype(np.float64) * window, warped.astype(np.float64) * window)
    shift = np.float32([[1, 0, -dx], [0, 1, -dy]])
    warped = _warp(warped, shift, size)
    warped_mask = _warp(warped_mask, shift, size)
    # Black fill from cropping, rotation and translation carries no information.
    usable = (warped_mask > 0.99) & (ref_mask > 0.5) & (warped > 0.5)
    overlap = float(usable.sum() / max(ref_mask.sum(), 1.0))
    if overlap < MIN_OVERLAP:
        return -1.0, overlap, (float(dx), float(dy))
    return masked_ncc(reference, warped, usable.astype(np.float32)), overlap, (float(dx), float(dy))


def align(candidate_gray: np.ndarray, received_bgr: np.ndarray, size: int) -> Alignment:
    """Aligns the received image to one candidate and scores the match.

    `candidate_gray` comes from fit_gray(original, size). The received image is
    resized to the candidate's shape first, which undoes a uniform rescale.
    """
    height, width = candidate_gray.shape
    received = received_bgr
    if received.ndim == 3:
        received = cv2.cvtColor(received, cv2.COLOR_BGR2GRAY)
    received = cv2.resize(received, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    reference, ref_mask = _canvas(candidate_gray, size)
    got, got_mask = _canvas(received, size)

    angle, scale = estimate_rotation_scale(reference, got, size, min(height, width) // 2)
    centre = (size / 2.0, size / 2.0)
    best = None
    for trial_angle, trial_scale in ((0.0, 1.0), (angle, scale), (angle + 180.0, scale)):
        # Undo the estimated attack: rotate back and zoom back.
        matrix = cv2.getRotationMatrix2D(centre, -trial_angle, 1.0 / trial_scale)
        score, overlap, shift = _compare(reference, ref_mask, got, got_mask, matrix, size)
        if best is None or score > best.score:
            best = Alignment(float((trial_angle + 180.0) % 360.0 - 180.0), float(trial_scale), shift, score, overlap)
    return best
