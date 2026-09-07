"""Denoising.

Fluorescence microscopy at low exposure is shot-noise limited, and the noise is
what breaks a threshold: speckle fragments objects and creates spurious tiny
components that survive into the instance stage.  The trade-off across these
methods is noise suppression versus edge (nuclear boundary) preservation, which
matters directly because touching nuclei are separated at their boundaries.
"""

from __future__ import annotations

import cv2
import numpy as np

from microseg.utils import to_uint8


def gaussian(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur.  Fast and simple, but blurs boundaries along with noise."""
    if sigma <= 0:
        return np.asarray(img, dtype=np.float32)
    # Kernel sized to ~3 sigma each side, forced odd, as OpenCV requires.
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(
        np.asarray(img, dtype=np.float32), (k, k), sigmaX=float(sigma)
    ).astype(np.float32)


def median(img: np.ndarray, size: int = 3) -> np.ndarray:
    """Median filter.  The default: kills salt-and-pepper/hot pixels outright
    while keeping step edges sharp."""
    size = int(size)
    if size < 3:
        return np.asarray(img, dtype=np.float32)
    if size % 2 == 0:
        size += 1
    # OpenCV's float path only supports 3x3/5x5, so larger kernels go via uint8.
    arr = np.asarray(img, dtype=np.float32)
    if size <= 5:
        return cv2.medianBlur(arr, size).astype(np.float32)
    return (cv2.medianBlur(to_uint8(arr), size).astype(np.float32) / 255.0)


def bilateral(
    img: np.ndarray, diameter: int = 5, sigma_color: float = 0.1, sigma_space: float = 5.0
) -> np.ndarray:
    """Edge-preserving smoothing: averages only over similar-intensity
    neighbours, so nuclear boundaries survive."""
    arr = np.asarray(img, dtype=np.float32)
    return cv2.bilateralFilter(
        arr, int(diameter), float(sigma_color), float(sigma_space)
    ).astype(np.float32)


def non_local_means(img: np.ndarray, strength: float = 5.0) -> np.ndarray:
    """Non-local means: averages over similar *patches*, not just neighbours.

    The best texture preservation of the four, and the slowest -- worth it when
    downstream GLCM texture features are the point.
    """
    denoised = cv2.fastNlMeansDenoising(
        to_uint8(img), None, h=float(strength), templateWindowSize=7, searchWindowSize=21
    )
    return (denoised.astype(np.float32) / 255.0)


def total_variation(img: np.ndarray, weight: float = 0.05) -> np.ndarray:
    """Total-variation (Chambolle) denoising: piecewise-flat output with crisp
    edges, which suits thresholding well."""
    from skimage.restoration import denoise_tv_chambolle

    return denoise_tv_chambolle(
        np.asarray(img, dtype=np.float32), weight=float(weight)
    ).astype(np.float32)


def denoise(img: np.ndarray, method: str | None = "median", strength: float = 1.0) -> np.ndarray:
    """Dispatch to a denoiser by name.

    ``strength`` is a single dimensionless knob (1.0 = sensible default) mapped
    onto each method's native parameter, so configs can sweep denoising
    aggressiveness without knowing per-method units.
    """
    if method in (None, "none"):
        return np.asarray(img, dtype=np.float32)
    if method == "gaussian":
        return gaussian(img, sigma=1.0 * strength)
    if method == "median":
        return median(img, size=int(round(3 * strength)) | 1)
    if method == "bilateral":
        return bilateral(img, diameter=5, sigma_color=0.1 * strength, sigma_space=5.0)
    if method == "nlm":
        return non_local_means(img, strength=5.0 * strength)
    if method == "tv":
        return total_variation(img, weight=0.05 * strength)
    raise ValueError(
        f"unknown denoise method {method!r}; expected gaussian|median|bilateral|nlm|tv|none"
    )
