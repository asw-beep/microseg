"""Illumination and artifact correction.

Two artifacts dominate widefield microscopy and both are corrected here:

* **Uneven illumination** -- vignetting and an off-centre light source leave a
  smooth intensity gradient across the field.  A global threshold then cuts
  correctly in the centre and wrongly in the corners.  The fix is to estimate
  the low-frequency background and divide (or subtract) it out.
* **Hot / dead pixels** -- single-pixel detector defects that survive every
  smoothing step short of a median, and which percentile normalisation and
  min-max scaling both react to.

:func:`has_dark_background` and :func:`maybe_invert` handle a third, dataset-
specific issue: BBBC038 mixes fluorescence (bright nuclei on dark) with H&E
brightfield (dark nuclei on light).  Normalising polarity up front is what lets
a single threshold -- and a single set of network weights -- cover both.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6


# --------------------------------------------------------------------- polarity
def has_dark_background(img: np.ndarray, quantile: float = 0.5) -> bool:
    """True when the image has dark background with bright objects.

    Nuclei occupy a minority of pixels in a typical field, so the median tracks
    the background.  Comparing it to the midpoint of the observed range is a
    cheap, threshold-free polarity test that holds for both modalities.
    """
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi - lo < EPS:
        return True
    midpoint = (lo + hi) / 2.0
    return float(np.quantile(arr, quantile)) < midpoint


def maybe_invert(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Invert brightfield images so objects are always bright.

    Returns the (possibly inverted) image and whether inversion happened, so the
    caller can record it -- intensity features must be reported on the original
    polarity, not the segmentation-convenient one.
    """
    arr = np.asarray(img, dtype=np.float32)
    if has_dark_background(arr):
        return arr, False
    return (1.0 - arr).astype(np.float32), True


# ------------------------------------------------------------------- artifacts
def remove_hot_pixels(
    img: np.ndarray, threshold: float = 5.0, size: int = 3
) -> np.ndarray:
    """Replace isolated outlier pixels with the local median.

    A pixel is an outlier when it deviates from its local median by more than
    ``threshold`` robust standard deviations (estimated globally from the MAD).
    Only those pixels are touched, so real structure is untouched -- unlike a
    plain median filter, which softens every edge in the image.
    """
    arr = np.asarray(img, dtype=np.float32)
    size = int(size) | 1
    local_median = cv2.medianBlur(arr, size) if size <= 5 else cv2.medianBlur(
        (np.clip(arr, 0, 1) * 255).astype(np.uint8), size
    ).astype(np.float32) / 255.0

    residual = arr - local_median
    mad = float(np.median(np.abs(residual - np.median(residual))))
    # 1.4826 converts a MAD into a Gaussian-equivalent sigma.
    sigma = 1.4826 * mad
    if sigma < EPS:
        return arr
    outliers = np.abs(residual) > threshold * sigma
    if not outliers.any():
        return arr
    out = arr.copy()
    out[outliers] = local_median[outliers]
    return out


# ----------------------------------------------------------------- illumination
def _morphological_background(
    img: np.ndarray, radius: int, max_kernel: int = 21
) -> np.ndarray:
    """Grayscale opening with a disk larger than any object.

    This is the classic "rolling ball" idea: an opening removes every bright
    structure that cannot contain the structuring element, leaving the slowly
    varying background.  ``radius`` must exceed the largest nucleus radius, or
    the nuclei themselves get absorbed into the background estimate.

    The opening is computed on a **downsampled** copy whenever the structuring
    element would exceed ``max_kernel`` pixels.  A large opening is expensive --
    a 101x101 element costs ~0.25 s on a 512x512 field, which dominated the whole
    preprocessing chain -- and the downsampling is free of consequence here
    because the quantity being estimated is by construction low-frequency.  The
    estimate is bilinearly upsampled back, which also smooths the slight
    blockiness that morphology on a coarse grid introduces.
    """
    arr = np.asarray(img, dtype=np.float32)
    radius = int(radius)
    h, w = arr.shape

    scale = max(1, int(np.ceil((2 * radius + 1) / max_kernel)))
    # Do not shrink below a size where the (scaled) element still fits.
    while scale > 1 and min(h // scale, w // scale) < 4 * max_kernel // 3:
        scale -= 1

    if scale == 1:
        k = 2 * radius + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(arr, cv2.MORPH_OPEN, se)

    small = cv2.resize(arr, (max(w // scale, 1), max(h // scale, 1)), interpolation=cv2.INTER_AREA)
    k = 2 * max(radius // scale, 1) + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(small, cv2.MORPH_OPEN, se)
    return cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)


def _gaussian_background(img: np.ndarray, radius: int) -> np.ndarray:
    """Heavy Gaussian blur as a background estimate.

    Cheapest option, but objects bleed into it, so it slightly under-corrects in
    dense fields.
    """
    sigma = max(1.0, float(radius) / 2.0)
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(np.asarray(img, dtype=np.float32), (k, k), sigma)


def _polynomial_background(img: np.ndarray, degree: int = 2) -> np.ndarray:
    """Least-squares 2-D polynomial surface fit.

    The right model when illumination really is a smooth low-order field (a
    single off-axis source, vignetting).  Fitted on a subsampled grid of the
    darker pixels so bright nuclei do not drag the surface upward.
    """
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = xx / max(w - 1, 1), yy / max(h - 1, 1)

    terms = [xn**i * yn**j for i in range(degree + 1) for j in range(degree + 1 - i)]
    design = np.stack([t.ravel() for t in terms], axis=1)

    values = arr.ravel()
    # Fit on background pixels only (below the 70th percentile).
    background = values <= np.percentile(values, 70)
    if background.sum() < design.shape[1]:
        background = np.ones_like(values, dtype=bool)
    coeffs, *_ = np.linalg.lstsq(design[background], values[background], rcond=None)
    return (design @ coeffs).reshape(h, w).astype(np.float32)


def estimate_background(
    img: np.ndarray, method: str = "morphological", radius: int = 50
) -> np.ndarray:
    """Estimate the smooth illumination field of an image."""
    if method == "morphological":
        return _morphological_background(img, radius)
    if method == "gaussian":
        return _gaussian_background(img, radius)
    if method == "polynomial":
        return _polynomial_background(img)
    raise ValueError(
        f"unknown illumination method {method!r}; "
        "expected morphological|gaussian|polynomial"
    )


def correct_illumination(
    img: np.ndarray,
    method: str | None = "morphological",
    radius: int = 50,
    mode: str = "subtract",
) -> np.ndarray:
    """Flatten uneven illumination.

    ``subtract`` suits an additive background (stray light, camera offset);
    ``divide`` suits a multiplicative one (vignetting, uneven excitation), which
    is the physically correct model for flat-field correction.
    """
    if method in (None, "none"):
        return np.asarray(img, dtype=np.float32)
    arr = np.asarray(img, dtype=np.float32)
    background = estimate_background(arr, method, radius)

    if mode == "subtract":
        corrected = arr - background
        # Restore the original mean level so absolute intensity features stay
        # interpretable rather than being pushed to near-zero.
        corrected = corrected + float(np.median(background))
    elif mode == "divide":
        corrected = arr / np.maximum(background, EPS)
        corrected = corrected * float(np.median(background))
    else:
        raise ValueError(f"unknown correction mode {mode!r}; expected subtract|divide")

    return np.clip(corrected, 0.0, 1.0).astype(np.float32)


def synthetic_illumination_gradient(
    shape: tuple[int, int], strength: float = 0.4, seed: int | None = None
) -> np.ndarray:
    """A smooth multiplicative shading field, for robustness testing.

    Used by the robustness benchmark to measure how much Dice a model loses when
    illumination correction has nothing to fall back on.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    cy, cx = rng.uniform(0.2, 0.8, size=2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= max(h - 1, 1)
    xx /= max(w - 1, 1)
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    field = 1.0 - strength * (dist / (dist.max() + EPS))
    return field.astype(np.float32)
