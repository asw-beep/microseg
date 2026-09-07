"""GLCM (Haralick) texture features.

Chromatin texture is a genuine morphological readout, not a generic image
statistic: condensed, coarse chromatin distinguishes apoptotic and mitotic
nuclei from interphase ones, and loss of texture homogeneity is a standard
cytopathology criterion for malignancy.  Two nuclei with identical area and
circularity can be told apart by texture alone.

The grey-level co-occurrence matrix counts how often intensity pairs occur at a
given offset.  From it:

* **contrast** -- local intensity variation; high for coarse, clumped chromatin.
* **homogeneity** -- the inverse view; high for smooth, evenly stained nuclei.
* **energy / ASM** -- textural uniformity; high when a few grey levels dominate.
* **correlation** -- how linearly predictable a pixel is from its neighbour, i.e.
  how organised the texture is.
* **entropy** -- randomness of the co-occurrence distribution.

Two implementation details matter for correctness on *objects* rather than whole
images:

1. Intensities are quantised to ``levels`` grey values, with level 0 reserved
   for "outside the object".  Fewer levels give a denser, less noisy matrix; 32
   is the usual compromise for small regions.
2. After computing the matrix, the row and column for level 0 are zeroed, which
   discards every pixel pair involving background.  Without this step the
   features would mostly describe the object's *silhouette against background*
   -- a shape measurement wearing a texture label.

Averaging over several angles makes the features rotation-invariant, which they
must be: a field of view has no preferred orientation.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops

FEATURE_NAMES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def _quantize(patch: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    """Quantise object pixels to ``1..levels-1``, leaving background at 0.

    Levels are assigned from the object's *own* intensity range, so texture is
    measured independently of how bright the nucleus is overall -- otherwise a
    dim nucleus and a bright one with identical chromatin structure would get
    different features.
    """
    out = np.zeros(patch.shape, dtype=np.uint8)
    if not mask.any():
        return out
    values = patch[mask]
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-8:
        out[mask] = 1
        return out
    scaled = (values - lo) / (hi - lo)
    out[mask] = np.clip((scaled * (levels - 2)).astype(np.uint8) + 1, 1, levels - 1)
    return out


def glcm_features(
    patch: np.ndarray,
    mask: np.ndarray | None = None,
    distances: list[int] | None = None,
    angles_deg: list[float] | None = None,
    levels: int = 32,
) -> dict[str, float]:
    """GLCM features for one image patch, optionally restricted to ``mask``.

    Returns one value per feature, averaged over the requested distances and
    angles.  Distances are also reported separately, because the ratio between a
    short-range and a long-range contrast is itself informative about chromatin
    granularity.
    """
    distances = distances or [1, 3]
    angles_deg = angles_deg or [0.0, 45.0, 90.0, 135.0]
    angles = [np.deg2rad(a) for a in angles_deg]

    patch = np.asarray(patch, dtype=np.float32)
    mask = np.ones(patch.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    empty = {f"glcm_{name.lower()}": 0.0 for name in FEATURE_NAMES}
    empty["glcm_entropy"] = 0.0
    for d in distances:
        empty[f"glcm_contrast_d{d}"] = 0.0
    if mask.sum() < 2:
        return empty

    quantized = _quantize(patch, mask, levels)
    matrix = graycomatrix(
        quantized, distances=distances, angles=angles, levels=levels, symmetric=True, normed=False
    )
    # Drop every pair that involved a background pixel.
    matrix[0, :, :, :] = 0
    matrix[:, 0, :, :] = 0
    if matrix.sum() == 0:
        return empty

    out: dict[str, float] = {}
    for name in FEATURE_NAMES:
        values = graycoprops(matrix, name.lower() if name != "ASM" else "ASM")
        out[f"glcm_{name.lower()}"] = float(np.nanmean(values))

    # Per-distance contrast, averaged over angles.
    contrast = graycoprops(matrix, "contrast")
    for i, d in enumerate(distances):
        out[f"glcm_contrast_d{d}"] = float(np.nanmean(contrast[i]))

    # Entropy is not provided by graycoprops, so normalise and compute it here.
    probs = matrix.astype(np.float64)
    totals = probs.sum(axis=(0, 1), keepdims=True)
    probs = np.divide(probs, totals, out=np.zeros_like(probs), where=totals > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(probs > 0, probs * np.log2(probs), 0.0), axis=(0, 1))
    out["glcm_entropy"] = float(np.nanmean(entropy))
    return out


def object_glcm_features(
    labels: np.ndarray,
    intensity: np.ndarray,
    distances: list[int] | None = None,
    angles_deg: list[float] | None = None,
    levels: int = 32,
) -> list[dict[str, float]]:
    """GLCM features for every instance, computed on its bounding-box crop.

    Cropping to the bounding box keeps the cost proportional to total object
    area rather than (number of objects x image area), which is what makes this
    tractable on fields holding hundreds of nuclei.
    """
    from scipy import ndimage as ndi

    labels = np.asarray(labels, dtype=np.int32)
    intensity = np.asarray(intensity, dtype=np.float32)
    n = int(labels.max())
    if n == 0:
        return []

    slices = ndi.find_objects(labels)
    rows: list[dict[str, float]] = []
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            rows.append({"label": i, **glcm_features(np.zeros((2, 2), np.float32))})
            continue
        patch = intensity[sl]
        mask = labels[sl] == i
        features = glcm_features(patch, mask, distances, angles_deg, levels)
        rows.append({"label": i, **features})
    return rows
