"""Microscopy preprocessing: normalisation, denoising, contrast, illumination.

Each submodule exposes plain ``ndarray -> ndarray`` functions that operate on a
single 2-D float32 channel in ``[0, 1]``.  :mod:`microseg.preprocessing.chain`
lifts them to multi-channel :class:`~microseg.channels.MicroscopyImage` inputs
and assembles them into the configured pipeline.
"""

from microseg.preprocessing.chain import (
    PreprocessResult,
    apply_per_channel,
    preprocess,
    preprocess_array,
)
from microseg.preprocessing.contrast import (
    adjust_gamma,
    clahe,
    stretch_contrast,
    unsharp_mask,
)
from microseg.preprocessing.denoise import (
    bilateral,
    denoise,
    gaussian,
    median,
    non_local_means,
    total_variation,
)
from microseg.preprocessing.illumination import (
    correct_illumination,
    estimate_background,
    has_dark_background,
    maybe_invert,
    remove_hot_pixels,
)
from microseg.preprocessing.normalize import (
    minmax_normalize,
    normalize,
    percentile_normalize,
    zscore_normalize,
)

__all__ = [
    "PreprocessResult",
    "adjust_gamma",
    "apply_per_channel",
    "bilateral",
    "clahe",
    "correct_illumination",
    "denoise",
    "estimate_background",
    "gaussian",
    "has_dark_background",
    "maybe_invert",
    "median",
    "minmax_normalize",
    "non_local_means",
    "normalize",
    "percentile_normalize",
    "preprocess",
    "preprocess_array",
    "remove_hot_pixels",
    "stretch_contrast",
    "total_variation",
    "unsharp_mask",
    "zscore_normalize",
]
