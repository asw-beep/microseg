"""Semantic -> instance segmentation.

The network predicts three per-pixel classes; a biologist needs *objects* --
"how many nuclei, and how big is each one".  Bridging that gap is a real
algorithmic step, not a formality, because the naive bridge (connected
components on the foreground mask) merges every pair of touching nuclei into one
object and silently undercounts dense fields.

The strategy here uses the boundary class as a separator:

1. **Seeds** = connected components of ``P(interior)``, which the boundary class
   has already pulled apart at the necks between neighbours.  One seed per
   nucleus.
2. **Mask** = ``P(interior) + P(boundary)``, i.e. all foreground.  This is the
   territory the instances are allowed to occupy.
3. **Watershed** floods from the seeds across the mask, so the boundary band is
   reassigned to whichever nucleus it borders, and the cut lands in the middle
   of the neck.

Step 3 matters: without it, objects would be missing their boundary ring and
every area measurement would be biased low.

**Why there is a second seeding strategy.**  Step 1 assumes a boundary exists to
pull neighbours apart, and when nuclei *overlap* rather than touch, it does not:
the annotation gives the pair no shared edge, the network has nothing to predict
there, and the two objects come back as one.  A model trained with a distance
head predicts the normalised distance to each nucleus' own edge instead, which
keeps two separate maxima through an overlap.  Seeds are then the connected
components of ``distance > threshold``.

Thresholding rather than peak-finding is deliberate.  ``peak_local_max`` on the
distance map of a partly occluded nucleus -- a crescent -- returns several
maxima along the arc and over-segments it; a level set at mid-radius stays one
component.  On a synthetic pair swept from touching to 50% overlap, thresholding
returns exactly 2 seeds throughout while peak-finding returns 4 at the extreme.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as cc_label
from skimage.segmentation import watershed

from microseg.classical.segment import (
    drop_small_labels,
    fill_small_holes,
    relabel_sequential,
)
from microseg.config import PostprocessConfig
from microseg.data.dataset import BACKGROUND, BOUNDARY, INTERIOR


def probabilities_to_instances(
    probs: np.ndarray,
    cfg: PostprocessConfig | None = None,
    distance: np.ndarray | None = None,
) -> np.ndarray:
    """3-class probability map ``(3, H, W)`` -> instance labels ``(H, W)``.

    ``distance`` is the optional predicted distance-to-edge map ``(H, W)`` in
    0..1 from a model with a distance head.  When supplied (and
    ``cfg.use_distance_seeds`` is on) it replaces the boundary class as the
    seeding signal; the foreground mask still comes from the class head either
    way, so the two heads keep the roles each is better at -- classes decide
    *what is a nucleus*, distance decides *where one ends and the next begins*.
    """
    cfg = cfg or PostprocessConfig()
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 3 or probs.shape[0] < 2:
        raise ValueError(f"expected a (C, H, W) probability map, got shape {probs.shape}")

    if distance is not None and cfg.use_distance_seeds:
        return instances_from_distance(probs, np.asarray(distance, dtype=np.float32), cfg)

    if probs.shape[0] == 2:
        # Binary model: no boundary channel, fall back to distance seeding.
        foreground = probs[1] > cfg.foreground_threshold
        return instances_from_binary(foreground, cfg)

    interior = probs[INTERIOR]
    foreground_prob = 1.0 - probs[BACKGROUND]

    foreground = foreground_prob > cfg.foreground_threshold
    if cfg.fill_holes and foreground.any():
        foreground = fill_small_holes(foreground, 64)
    if not foreground.any():
        return np.zeros(probs.shape[1:], dtype=np.int32)

    # A seed must be interior-dominant, not merely above a threshold: requiring
    # interior to beat boundary makes the seeding insensitive to the absolute
    # calibration of the network's confidence.
    seeds = (interior > cfg.seed_threshold) & (interior >= probs[BOUNDARY]) & foreground
    markers = cc_label(seeds)
    if cfg.min_seed_area > 0 and markers.max() > 0:
        markers = relabel_sequential(drop_small_labels(markers, cfg.min_seed_area))

    if markers.max() == 0:
        # The model found foreground but no confident centre (very small or very
        # dim objects).  Distance seeding is the graceful degradation.
        return instances_from_binary(foreground, cfg)

    # Flood the *negative* interior probability: water rises from confident
    # centres outward, and the ridge lines land where interior confidence is
    # lowest -- the boundary between neighbours.
    labels = watershed(
        -interior, markers, mask=foreground, watershed_line=cfg.watershed_line
    )
    labels = drop_small_labels(labels.astype(np.int32), cfg.min_object_area)
    return relabel_sequential(labels)


def instances_from_distance(
    probs: np.ndarray, distance: np.ndarray, cfg: PostprocessConfig | None = None
) -> np.ndarray:
    """Seed from the predicted distance map, flood over the class foreground."""
    cfg = cfg or PostprocessConfig()
    distance = np.asarray(distance, dtype=np.float32)
    if distance.ndim == 3 and distance.shape[0] == 1:
        distance = distance[0]
    if distance.shape != probs.shape[1:]:
        raise ValueError(
            f"distance map {distance.shape} does not match probabilities {probs.shape[1:]}"
        )

    foreground_prob = 1.0 - probs[BACKGROUND] if probs.shape[0] > 2 else probs[1]
    foreground = foreground_prob > cfg.foreground_threshold
    if cfg.fill_holes and foreground.any():
        foreground = fill_small_holes(foreground, 64)
    if not foreground.any():
        return np.zeros(probs.shape[1:], dtype=np.int32)

    # Intersecting with the foreground is what lets the training loss leave the
    # background distance unconstrained: a confident distance prediction outside
    # the mask cannot become a seed.
    seeds = (distance > cfg.distance_seed_threshold) & foreground
    markers = cc_label(seeds)
    if cfg.min_seed_area > 0 and markers.max() > 0:
        markers = relabel_sequential(drop_small_labels(markers, cfg.min_seed_area))

    if markers.max() == 0:
        # Every object is smaller than the seed threshold can resolve.  Fall back
        # rather than return an empty field.
        return instances_from_binary(foreground, cfg)

    labels = watershed(
        -distance, markers, mask=foreground, watershed_line=cfg.watershed_line
    )
    labels = drop_small_labels(labels.astype(np.int32), cfg.min_object_area)
    return relabel_sequential(labels)


def instances_from_binary(
    mask: np.ndarray, cfg: PostprocessConfig | None = None
) -> np.ndarray:
    """Binary foreground -> instances via a distance-transform watershed.

    Used for binary models, and as the fallback when no interior seed survives.
    """
    from skimage.feature import peak_local_max

    cfg = cfg or PostprocessConfig()
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32)

    distance = ndi.distance_transform_edt(binary)
    peaks = peak_local_max(
        distance, min_distance=int(cfg.peak_min_distance), labels=binary, exclude_border=False
    )
    seeds = np.zeros(binary.shape, dtype=bool)
    if peaks.size:
        seeds[tuple(peaks.T)] = True

    markers = cc_label(seeds)
    if markers.max() == 0:
        labels = cc_label(binary).astype(np.int32)
    else:
        labels = watershed(-distance, markers, mask=binary).astype(np.int32)

    labels = drop_small_labels(labels, cfg.min_object_area)
    return relabel_sequential(labels)


def instances_from_probabilities(
    probs: np.ndarray,
    cfg: PostprocessConfig | None = None,
    distance: np.ndarray | None = None,
) -> np.ndarray:
    """Alias of :func:`probabilities_to_instances`, kept for readable call sites."""
    return probabilities_to_instances(probs, cfg, distance)


def remove_border_objects(labels: np.ndarray) -> np.ndarray:
    """Drop instances touching the image border.

    Objects clipped by the field of view have truncated area, perimeter and
    shape, so including them biases every morphology statistic downward.  The
    trade-off is a biased *count* (large objects are likelier to touch a border),
    which is why this is opt-in rather than automatic.
    """
    labels = np.asarray(labels, dtype=np.int32)
    if labels.max() == 0:
        return labels
    border_ids = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    border_ids = border_ids[border_ids != 0]
    if border_ids.size:
        labels = np.where(np.isin(labels, border_ids), 0, labels)
    return relabel_sequential(labels)
