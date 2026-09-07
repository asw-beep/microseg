"""Assembles the configured preprocessing chain and lifts it to multi-channel.

Stage order is fixed and deliberate:

1. **hot-pixel removal** -- before anything that computes statistics, because a
   handful of saturated pixels skew percentiles and background fits.
2. **polarity normalisation** -- so every later stage sees bright objects on a
   dark background.
3. **illumination correction** -- on the raw-ish signal, before contrast
   enhancement bakes the gradient into the histogram.
4. **denoising** -- after flattening, so the noise estimate is spatially uniform.
5. **contrast enhancement** -- on clean data, so CLAHE amplifies signal, not noise.
6. **normalisation** -- last, to hand a fixed ``[0, 1]`` range downstream.

Getting this order wrong is a common and quiet failure: CLAHE before
illumination correction, for instance, locks in the shading it was meant to
remove.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from microseg.channels import MicroscopyImage
from microseg.config import PreprocessConfig
from microseg.preprocessing.contrast import enhance_contrast
from microseg.preprocessing.denoise import denoise
from microseg.preprocessing.illumination import (
    correct_illumination,
    maybe_invert,
    remove_hot_pixels,
)
from microseg.preprocessing.normalize import normalize


@dataclass
class PreprocessResult:
    """Preprocessed image plus the intermediates, for visual QC."""

    image: MicroscopyImage
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    inverted: list[bool] = field(default_factory=list)

    @property
    def array(self) -> np.ndarray:
        return self.image.data


def apply_per_channel(image: MicroscopyImage, fn, channels=None) -> MicroscopyImage:
    """Apply a 2-D ``ndarray -> ndarray`` op to selected channels.

    Unselected channels pass through untouched.  This is the single place that
    knows how to broadcast a single-channel operator over a multiplexed stack,
    which keeps every operator in this package simple and 2-D.
    """
    indices = image.resolve(channels)
    out = image.data.copy()
    for idx in indices:
        out[:, :, idx] = fn(image.data[:, :, idx])
    return image.with_data(out)


def preprocess_array(
    img: np.ndarray, cfg: PreprocessConfig | None = None, collect_stages: bool = False
) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    """Run the chain on one 2-D channel.

    Returns the result, a dict of intermediates (empty unless ``collect_stages``)
    and whether polarity was inverted.
    """
    cfg = cfg or PreprocessConfig()
    arr = np.asarray(img, dtype=np.float32)
    stages: dict[str, np.ndarray] = {}
    if collect_stages:
        stages["raw"] = arr.copy()

    if cfg.hot_pixel_correction:
        arr = remove_hot_pixels(arr, threshold=cfg.hot_pixel_threshold)
        if collect_stages:
            stages["hot_pixels_removed"] = arr.copy()

    inverted = False
    if cfg.auto_invert:
        arr, inverted = maybe_invert(arr)
        if collect_stages and inverted:
            stages["inverted"] = arr.copy()

    if cfg.illumination not in (None, "none"):
        arr = correct_illumination(arr, cfg.illumination, cfg.illumination_radius)
        if collect_stages:
            stages["illumination_corrected"] = arr.copy()

    if cfg.denoise not in (None, "none"):
        arr = denoise(arr, cfg.denoise, cfg.denoise_strength)
        if collect_stages:
            stages["denoised"] = arr.copy()

    if cfg.contrast not in (None, "none"):
        arr = enhance_contrast(
            arr,
            cfg.contrast,
            clahe_clip_limit=cfg.clahe_clip_limit,
            clahe_tile_grid=cfg.clahe_tile_grid,
            gamma=cfg.gamma,
        )
        if collect_stages:
            stages["contrast_enhanced"] = arr.copy()

    arr = normalize(
        arr,
        cfg.normalize,
        percentile_low=cfg.percentile_low,
        percentile_high=cfg.percentile_high,
    )
    if collect_stages:
        stages["normalized"] = arr.copy()

    return arr.astype(np.float32), stages, inverted


def preprocess(
    image: MicroscopyImage,
    cfg: PreprocessConfig | None = None,
    channels=None,
    collect_stages: bool = False,
) -> PreprocessResult:
    """Run the chain over a :class:`MicroscopyImage`.

    Each channel is processed independently -- correct for microscopy, where
    channels have their own background, noise level and dynamic range and must
    never be normalised jointly.
    """
    cfg = cfg or PreprocessConfig()
    indices = image.resolve(channels)
    out = image.data.copy()
    stages: dict[str, np.ndarray] = {}
    inverted: list[bool] = []

    for i, idx in enumerate(indices):
        # Only the first selected channel contributes preview stages, otherwise
        # a 6-channel stack would produce an unreadable QC figure.
        want = collect_stages and i == 0
        processed, chan_stages, was_inverted = preprocess_array(
            image.data[:, :, idx], cfg, collect_stages=want
        )
        out[:, :, idx] = processed
        stages.update(chan_stages)
        inverted.append(was_inverted)

    return PreprocessResult(image.with_data(out), stages, inverted)
