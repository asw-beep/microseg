"""Contrast enhancement.

Nuclei at the dim end of a field -- out-of-focus, weakly stained, or in a
vignetted corner -- are the ones a global threshold loses.  Local contrast
enhancement pulls them up to comparable intensity with the bright ones before
segmentation ever sees the image.
"""

from __future__ import annotations

import cv2
import numpy as np

from microseg.utils import to_uint8


def clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalisation.

    Equalises within small tiles, so dim regions get boosted independently of
    bright ones.  ``clip_limit`` caps the local histogram slope, which is the
    part that stops plain AHE from amplifying background noise into fake texture.
    """
    op = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid))
    )
    return (op.apply(to_uint8(img)).astype(np.float32) / 255.0)


def adjust_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Power-law intensity transform.

    ``gamma < 1`` brightens dim structures (the usual direction for faint
    fluorescence); ``gamma > 1`` suppresses background haze.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    arr = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    return np.power(arr, float(gamma)).astype(np.float32)


def stretch_contrast(
    img: np.ndarray, low: float = 2.0, high: float = 98.0
) -> np.ndarray:
    """Linear percentile stretch -- global, and the mildest option here."""
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, [low, high])
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def unsharp_mask(img: np.ndarray, sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """Sharpen by adding back the high-frequency residual.

    Useful before watershed: crisper boundaries give a cleaner distance
    transform, and therefore fewer merged nuclei.  Overdone it manufactures
    haloes around objects, so ``amount`` stays modest by default.
    """
    arr = np.asarray(img, dtype=np.float32)
    k = max(3, int(2 * round(3 * sigma) + 1))
    blurred = cv2.GaussianBlur(arr, (k, k), float(sigma))
    return np.clip(arr + float(amount) * (arr - blurred), 0.0, 1.0).astype(np.float32)


def enhance_contrast(img: np.ndarray, method: str | None = "clahe", **kwargs) -> np.ndarray:
    """Dispatch to a contrast operator by name."""
    if method in (None, "none"):
        return np.asarray(img, dtype=np.float32)
    if method == "clahe":
        return clahe(
            img, kwargs.get("clahe_clip_limit", 2.0), kwargs.get("clahe_tile_grid", 8)
        )
    if method == "gamma":
        return adjust_gamma(img, kwargs.get("gamma", 1.0))
    if method == "stretch":
        return stretch_contrast(img)
    if method == "unsharp":
        return unsharp_mask(img)
    raise ValueError(
        f"unknown contrast method {method!r}; expected clahe|gamma|stretch|unsharp|none"
    )
