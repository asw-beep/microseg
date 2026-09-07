"""Classical CV segmentation baseline: threshold -> morphology -> watershed.

This is the reference the U-Net has to beat, and it is not a straw man: on
clean, well-separated fluorescence fields a tuned Otsu + distance-transform
watershed is genuinely competitive, runs in milliseconds, needs no GPU and no
training data.  Its failure modes are the interesting part, and they are exactly
the ones the learned model is there to fix:

* heterogeneous modalities (H&E vs fluorescence) break a single global threshold;
* densely packed nuclei defeat distance-transform seeding, because the merged
  distance map has one basin, not two;
* textured or vesicular nuclei fragment into several components.

Every stage is exposed separately so the pipeline can be ablated stage by stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_li, threshold_otsu, threshold_triangle
from skimage.measure import label as cc_label
from skimage.segmentation import watershed

from microseg.utils import to_uint8


@dataclass
class ClassicalParams:
    """Tunable knobs for the classical pipeline."""

    threshold_method: str = "otsu"  # otsu|adaptive|li|triangle|fixed
    fixed_threshold: float = 0.5
    adaptive_block_size: int = 51
    adaptive_offset: float = 0.02

    opening_radius: int = 1
    closing_radius: int = 2
    min_object_area: int = 15
    min_hole_area: int = 30

    use_watershed: bool = True
    # Peaks closer than this merge into one seed.  The single most important
    # parameter: too small over-segments nuclei, too large merges neighbours.
    peak_min_distance: int = 5
    # Erode the distance map's peak set relative to the object; helps when nuclei
    # differ a lot in size.
    seed_relative_threshold: float = 0.35
    watershed_line: bool = False


# ------------------------------------------------------------------ thresholding
def threshold_image(
    img: np.ndarray, method: str = "otsu", params: ClassicalParams | None = None
) -> np.ndarray:
    """Binarise a normalised (bright-objects) float image.

    ``otsu`` maximises between-class variance -- the right default for a
    bimodal fluorescence histogram.  ``adaptive`` thresholds per neighbourhood
    and is the fallback when illumination correction has not fully flattened the
    field.  ``li`` (minimum cross-entropy) is more forgiving than Otsu when
    foreground occupies only a tiny fraction of pixels, which is common in sparse
    fields.
    """
    params = params or ClassicalParams()
    arr = np.asarray(img, dtype=np.float32)

    if method == "fixed":
        return arr > params.fixed_threshold
    if method == "adaptive":
        block = int(params.adaptive_block_size) | 1
        binary = cv2.adaptiveThreshold(
            to_uint8(arr),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            -float(params.adaptive_offset) * 255.0,
        )
        return binary > 0

    # A constant image has no meaningful threshold; skimage raises, so short-circuit.
    if float(arr.max() - arr.min()) < 1e-6:
        return np.zeros(arr.shape, dtype=bool)

    if method == "otsu":
        value = threshold_otsu(arr)
    elif method == "li":
        value = threshold_li(arr)
    elif method == "triangle":
        value = threshold_triangle(arr)
    else:
        raise ValueError(
            f"unknown threshold method {method!r}; expected otsu|adaptive|li|triangle|fixed"
        )
    return arr > value


# -------------------------------------------------------------------- morphology
def morphological_cleanup(
    mask: np.ndarray, params: ClassicalParams | None = None
) -> np.ndarray:
    """Opening -> closing -> hole filling -> small-object removal.

    Opening first (erode then dilate) detaches thin bridges between nuclei that
    a threshold welded together and deletes speckle; closing then repairs the
    concavities that opening bit out of real objects.  Doing it in the other
    order would cement the bridges in place.
    """
    params = params or ClassicalParams()
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return binary

    work = binary.astype(np.uint8)
    if params.opening_radius > 0:
        k = 2 * int(params.opening_radius) + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, se)
    if params.closing_radius > 0:
        k = 2 * int(params.closing_radius) + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, se)

    out = work.astype(bool)
    if params.min_hole_area > 0:
        out = fill_small_holes(out, params.min_hole_area)
    if params.min_object_area > 0:
        out = remove_small_regions(out, params.min_object_area)
    return out


def fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    """Fill background holes smaller than ``max_hole_area``.

    Nucleoli and staining gaps punch holes in an otherwise solid nucleus; left
    open they corrupt area and, worse, split the distance transform.  Only
    enclosed holes are filled -- background regions that reach the image border
    are the actual background, no matter how small they look.
    """
    mask = np.asarray(mask, dtype=bool)
    holes = cc_label(~mask, connectivity=1)
    if holes.max() == 0:
        return mask

    areas = np.bincount(holes.ravel())
    touches_border = np.unique(
        np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]])
    )
    candidates = np.arange(1, holes.max() + 1)
    fillable = candidates[
        ~np.isin(candidates, touches_border) & (areas[candidates] <= int(max_hole_area))
    ]
    if fillable.size == 0:
        return mask
    return mask | np.isin(holes, fillable)


def remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components with fewer than ``min_area`` pixels.

    The size floor that removes debris, staining speckle and single-pixel noise
    survivors -- objects far too small to be a nucleus at any magnification in
    the dataset.
    """
    mask = np.asarray(mask, dtype=bool)
    components = cc_label(mask)
    if components.max() == 0:
        return mask
    areas = np.bincount(components.ravel())
    small = np.flatnonzero(areas < int(min_area))
    if small.size == 0:
        return mask
    return mask & ~np.isin(components, small)


# -------------------------------------------------------------------- watershed
def watershed_split(
    mask: np.ndarray,
    params: ClassicalParams | None = None,
    intensity: np.ndarray | None = None,
) -> np.ndarray:
    """Split touching objects with a distance-transform watershed.

    The distance transform turns each nucleus into a hill peaking at its centre;
    two touching nuclei give two peaks separated by a saddle.  Flooding the
    *negative* distance map from those peaks cuts along the saddle -- which is
    the narrow neck between the nuclei, the boundary a human would draw.
    """
    params = params or ClassicalParams()
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32)

    distance = ndi.distance_transform_edt(binary)

    # Two seeding criteria, intersected: local maxima that are also a decent
    # fraction of the object's own maximum distance.  The relative test is what
    # keeps a small nucleus next to a large one from being swallowed.
    peaks = peak_local_max(
        distance,
        min_distance=int(params.peak_min_distance),
        labels=binary,
        exclude_border=False,
    )
    seeds = np.zeros(binary.shape, dtype=bool)
    if peaks.size:
        seeds[tuple(peaks.T)] = True

    if params.seed_relative_threshold > 0:
        objects = cc_label(binary)
        maxima = ndi.maximum(distance, objects, index=np.arange(1, objects.max() + 1))
        if maxima.size:
            per_pixel_max = np.zeros_like(distance)
            lut = np.concatenate([[0.0], np.atleast_1d(maxima)])
            per_pixel_max = lut[objects]
            seeds &= distance >= params.seed_relative_threshold * per_pixel_max

    markers = cc_label(seeds)
    if markers.max() == 0:
        # No usable seed survived: fall back to plain connected components rather
        # than returning an empty image.
        return cc_label(binary).astype(np.int32)

    surface = -distance if intensity is None else -np.asarray(intensity, dtype=np.float32)
    labels = watershed(
        surface, markers, mask=binary, watershed_line=params.watershed_line
    )
    return labels.astype(np.int32)


# ---------------------------------------------------------------------- pipeline
def classical_segment(
    img: np.ndarray, params: ClassicalParams | None = None
) -> np.ndarray:
    """Preprocessed image -> cleaned binary foreground mask."""
    params = params or ClassicalParams()
    binary = threshold_image(img, params.threshold_method, params)
    return morphological_cleanup(binary, params)


def classical_instances(
    img: np.ndarray, params: ClassicalParams | None = None
) -> np.ndarray:
    """Preprocessed image -> instance label image (0 = background).

    The full classical baseline, and the function the benchmark scripts call.
    """
    params = params or ClassicalParams()
    binary = classical_segment(img, params)
    if not binary.any():
        return np.zeros(np.asarray(img).shape[:2], dtype=np.int32)

    if params.use_watershed:
        labels = watershed_split(binary, params, intensity=None)
    else:
        labels = cc_label(binary).astype(np.int32)

    # Watershed can leave slivers below the size floor; re-apply it on instances.
    if params.min_object_area > 0:
        labels = drop_small_labels(labels, params.min_object_area)
    return relabel_sequential(labels)


def drop_small_labels(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Remove instances below ``min_area`` pixels."""
    labels = np.asarray(labels, dtype=np.int32)
    if labels.max() == 0:
        return labels
    counts = np.bincount(labels.ravel())
    too_small = np.flatnonzero(counts < int(min_area))
    if too_small.size:
        labels = np.where(np.isin(labels, too_small), 0, labels)
    return labels


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Renumber labels to ``1..N`` with no gaps.

    Downstream code (metric matching, feature tables) assumes contiguous ids.
    """
    labels = np.asarray(labels, dtype=np.int32)
    unique = np.unique(labels)
    unique = unique[unique != 0]
    if unique.size == 0:
        return np.zeros_like(labels, dtype=np.int32)
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lut[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
    return lut[labels]
