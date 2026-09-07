"""Segment Anything (SAM / SAM 2) experiment.

A deliberately small experiment to characterise where a vision foundation model
does and does not help on microscopy, rather than to claim it as a solution.

The interesting finding is a structural one, and it holds regardless of the
exact checkpoint. SAM is class-agnostic: it segments *whatever is there*, with
no notion of "nucleus". On a field of a few hundred small, near-identical,
touching nuclei this produces three characteristic failures:

1. the automatic mask generator's point grid (32x32 by default) is coarser than
   the nuclei themselves, so many are never prompted and are simply missed;
2. its ambiguity resolution favours larger, more "object-like" regions, so it
   readily returns a clump of nuclei as one mask;
3. returned masks may overlap and are unordered, whereas an instance map must be
   a partition -- so a resolution rule is needed before the standard metrics
   even apply.

Two modes are implemented to separate "SAM cannot see nuclei" from "SAM was not
told where to look":

* :func:`sam_automatic_masks` -- ungrided automatic segmentation, the zero-shot
  baseline.
* :func:`sam_prompted_masks` -- SAM prompted with point seeds from this
  project's own classical distance-transform stage.  This is the practically
  interesting configuration: cheap classical CV supplies detection, the
  foundation model supplies the outline, and it usually beats the automatic mode
  by a wide margin on this data.

SAM weights are a large download and are not bundled.  With no checkpoint
present, every function reports a skip status instead of failing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from microseg.classical.segment import relabel_sequential
from microseg.utils import LOGGER

SAM_AVAILABLE = importlib.util.find_spec("segment_anything") is not None
SAM2_AVAILABLE = importlib.util.find_spec("sam2") is not None

# Where the loader looks for weights, in order.
DEFAULT_CHECKPOINTS = (
    ("vit_b", "weights/sam_vit_b_01ec64.pth"),
    ("vit_l", "weights/sam_vit_l_0b3195.pth"),
    ("vit_h", "weights/sam_vit_h_4b8939.pth"),
)


def find_checkpoint(explicit: str | Path | None = None) -> tuple[str, Path] | None:
    """Locate SAM weights, returning ``(model_type, path)`` or None."""
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            return None
        name = path.name.lower()
        for model_type in ("vit_b", "vit_l", "vit_h"):
            if model_type in name:
                return model_type, path
        return "vit_b", path

    for model_type, relative in DEFAULT_CHECKPOINTS:
        path = Path(relative)
        if path.exists():
            return model_type, path
    return None


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """SAM expects an 8-bit 3-channel image; microscopy gives 1 float channel."""
    arr = np.clip(np.asarray(image, dtype=np.float32), 0, 1)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return (arr * 255).astype(np.uint8)


def load_sam(checkpoint: str | Path | None = None, device: str = "cpu"):
    """Build a SAM predictor, or return None when unavailable."""
    if not SAM_AVAILABLE:
        LOGGER.info("segment_anything is not installed; skipping the SAM experiment")
        return None
    found = find_checkpoint(checkpoint)
    if found is None:
        LOGGER.info("no SAM checkpoint found (looked in weights/); skipping")
        return None

    from segment_anything import sam_model_registry

    model_type, path = found
    LOGGER.info("loading SAM %s from %s", model_type, path)
    sam = sam_model_registry[model_type](checkpoint=str(path))
    sam.to(device)
    return sam


def masks_to_labels(masks: list[np.ndarray], min_area: int = 15) -> np.ndarray:
    """Resolve a list of possibly-overlapping binary masks into an instance map.

    Painted smallest-last, so that when a clump mask and its constituent nuclei
    both come back, the smaller (more specific) masks win the contested pixels.
    Without this rule SAM's output is not a valid instance segmentation and any
    IoU-matched metric computed on it is meaningless.
    """
    if not masks:
        return np.zeros((1, 1), dtype=np.int32)

    ordered = sorted(masks, key=lambda m: int(np.count_nonzero(m)), reverse=True)
    labels = np.zeros(ordered[0].shape, dtype=np.int32)
    next_id = 1
    for mask in ordered:
        mask = np.asarray(mask, dtype=bool)
        if np.count_nonzero(mask) < min_area:
            continue
        labels[mask] = next_id
        next_id += 1
    return relabel_sequential(labels)


def sam_automatic_masks(
    image: np.ndarray,
    sam=None,
    points_per_side: int = 32,
    min_area: int = 15,
    max_area_fraction: float = 0.25,
) -> np.ndarray:
    """Zero-shot automatic mask generation.

    ``max_area_fraction`` drops masks covering more than a quarter of the field:
    SAM very often returns the whole background, or the entire tissue region, as
    one "object", and that mask would otherwise swallow every nucleus.
    """
    if sam is None:
        return np.zeros(np.asarray(image).shape[:2], dtype=np.int32)

    from segment_anything import SamAutomaticMaskGenerator

    generator = SamAutomaticMaskGenerator(
        sam, points_per_side=points_per_side, min_mask_region_area=min_area
    )
    records = generator.generate(_to_rgb_uint8(image))

    total = float(np.asarray(image).shape[0] * np.asarray(image).shape[1])
    masks = [
        r["segmentation"]
        for r in records
        if min_area <= r["area"] <= max_area_fraction * total
    ]
    return masks_to_labels(masks, min_area)


def sam_prompted_masks(
    image: np.ndarray, sam=None, seeds: np.ndarray | None = None, min_area: int = 15
) -> np.ndarray:
    """SAM prompted with one point per nucleus from the classical seeding stage.

    The hybrid worth demonstrating: a distance-transform watershed is very good
    at *finding* nuclei centres and mediocre at outlines, while SAM is the
    reverse.  Composing them plays to both.
    """
    if sam is None:
        return np.zeros(np.asarray(image).shape[:2], dtype=np.int32)

    from segment_anything import SamPredictor

    if seeds is None:
        seeds = classical_seed_points(image)
    if seeds.size == 0:
        return np.zeros(np.asarray(image).shape[:2], dtype=np.int32)

    predictor = SamPredictor(sam)
    predictor.set_image(_to_rgb_uint8(image))

    masks: list[np.ndarray] = []
    for y, x in seeds:
        # SAM takes (x, y); getting this backwards is the classic silent bug and
        # produces plausible-looking masks in the wrong places.
        point = np.array([[float(x), float(y)]])
        predicted, scores, _ = predictor.predict(
            point_coords=point, point_labels=np.array([1]), multimask_output=True
        )
        # multimask_output returns three nested candidates (part / part-of /
        # whole).  For a nucleus the tightest plausible mask is wanted, so pick
        # the highest-scoring one that is not implausibly large.
        area_limit = 0.05 * predicted[0].size
        candidates = [
            (score, mask)
            for score, mask in zip(scores, predicted)
            if min_area <= np.count_nonzero(mask) <= area_limit
        ]
        if candidates:
            masks.append(max(candidates, key=lambda c: c[0])[1])

    return masks_to_labels(masks, min_area)


def classical_seed_points(image: np.ndarray, min_distance: int = 5) -> np.ndarray:
    """Nucleus centre candidates from the classical distance transform."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max

    from microseg.classical.segment import ClassicalParams, classical_segment

    mask = classical_segment(image, ClassicalParams())
    if not mask.any():
        return np.zeros((0, 2), dtype=int)
    distance = ndi.distance_transform_edt(mask)
    return peak_local_max(distance, min_distance=min_distance, labels=mask, exclude_border=False)


def run_sam_experiment(
    cache_dir: str | Path,
    image_ids: list[str],
    checkpoint: str | Path | None = None,
    device: str = "cpu",
    preprocess_cfg=None,
    modes: tuple[str, ...] = ("automatic", "prompted"),
) -> dict:
    """Evaluate SAM on a few images in both modes, with the project's metrics.

    Intentionally run on a handful of images: SAM on CPU takes seconds per
    field, and the point being made is qualitative.
    """
    from microseg.data.bbbc038 import load_cached, to_microscopy_image
    from microseg.metrics.instance import instance_metrics
    from microseg.metrics.semantic import aggregate, semantic_metrics
    from microseg.preprocessing.chain import preprocess_array
    from microseg.utils import Timer

    sam = load_sam(checkpoint, device)
    if sam is None:
        return {
            "status": "skipped",
            "reason": (
                "segment_anything not installed or no checkpoint in weights/; "
                "see docs/foundation_models.md"
            ),
        }

    cache_dir = Path(cache_dir)
    results: dict = {"status": "ok", "modes": {}}

    for mode in modes:
        timer = Timer()
        records: list[dict] = []
        for image_id in image_ids:
            sample = load_cached(cache_dir / f"{image_id}.npz")
            nuclei = to_microscopy_image(sample).nuclei()
            if preprocess_cfg is not None:
                nuclei, _, _ = preprocess_array(nuclei, preprocess_cfg)

            with timer.lap():
                if mode == "automatic":
                    labels = sam_automatic_masks(nuclei, sam)
                else:
                    labels = sam_prompted_masks(nuclei, sam)

            record: dict = {"image_id": image_id}
            record.update(semantic_metrics(labels > 0, sample.labels > 0))
            record.update(instance_metrics(labels, sample.labels))
            records.append(record)

        summary = aggregate(records)
        summary["n_images"] = len(records)
        summary.update({f"time_{k}": round(v, 4) for k, v in timer.summary().items()})
        results["modes"][mode] = summary
        LOGGER.info(
            "SAM %-9s | dice %.4f  AP %.4f  F1 %.4f  %.2f s/img",
            mode,
            summary.get("dice", 0),
            summary.get("ap_mean", 0),
            summary.get("f1", 0),
            summary.get("time_mean_s", 0),
        )

    return results
