"""Intensity normalisation.

Microscopy intensities are not comparable across images: exposure time, laser
power, gain and stain concentration all move the histogram.  Normalising each
image (or each channel) is what lets one threshold or one set of network weights
work across a plate.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def percentile_normalize(
    img: np.ndarray, low: float = 1.0, high: float = 99.0
) -> np.ndarray:
    """Rescale so the ``low``/``high`` intensity percentiles map to 0 and 1.

    The default for microscopy.  Unlike min-max it is robust to hot pixels and
    saturated debris, which otherwise compress the whole dynamic range of real
    signal into a narrow band.
    """
    if high <= low:
        raise ValueError(f"high percentile ({high}) must exceed low ({low})")
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [low, high])
    if hi - lo < EPS:
        # A flat field (e.g. an empty tile): return zeros rather than amplify noise.
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def minmax_normalize(img: np.ndarray) -> np.ndarray:
    """Rescale the full observed range to ``[0, 1]``."""
    img = np.asarray(img, dtype=np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < EPS:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def zscore_normalize(img: np.ndarray, clip_sigma: float | None = 3.0) -> np.ndarray:
    """Zero-mean/unit-variance, then squashed back into ``[0, 1]``.

    ``clip_sigma`` bounds the output range so the result is still a displayable
    image; pass ``None`` to return raw z-scores (useful as a network input).
    """
    img = np.asarray(img, dtype=np.float32)
    mean, std = float(img.mean()), float(img.std())
    if std < EPS:
        return np.zeros_like(img, dtype=np.float32)
    z = (img - mean) / std
    if clip_sigma is None:
        return z.astype(np.float32)
    return np.clip((z + clip_sigma) / (2 * clip_sigma), 0.0, 1.0).astype(np.float32)


def normalize(img: np.ndarray, method: str | None = "percentile", **kwargs) -> np.ndarray:
    """Dispatch to a normalisation method by name."""
    if method in (None, "none"):
        return np.asarray(img, dtype=np.float32)
    if method == "percentile":
        return percentile_normalize(
            img, kwargs.get("percentile_low", 1.0), kwargs.get("percentile_high", 99.0)
        )
    if method == "minmax":
        return minmax_normalize(img)
    if method == "zscore":
        return zscore_normalize(img, kwargs.get("clip_sigma", 3.0))
    raise ValueError(
        f"unknown normalisation method {method!r}; "
        "expected percentile|minmax|zscore|none"
    )
