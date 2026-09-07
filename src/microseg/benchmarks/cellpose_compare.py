"""Cellpose comparison.

Cellpose is the standard general-purpose cell/nuclei segmentation model, so it
is the honest reference point for a custom U-Net: it was trained on far more
data than BBBC038 alone and needs no training at all here.  The comparison is
worth running for what it reveals about the trade-off, not to declare a winner:

* Cellpose predicts spatial *gradient flows* and recovers instances by following
  them to a common attractor.  That is a fundamentally different instance
  mechanism from the boundary-class watershed used in this project, and it
  handles crowded, non-convex cells more gracefully.
* The custom U-Net is ~7M parameters, trains in minutes on one GPU, runs on CPU
  in real time, and can be retrained on a new assay from a few hundred
  annotations.  Cellpose is heavier and less easy to specialise.

The comparison is run through the same metrics and the same preprocessing as
every other backend, so the numbers are directly comparable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from microseg.metrics.instance import instance_metrics
from microseg.metrics.semantic import aggregate, semantic_metrics
from microseg.utils import LOGGER, Timer

CELLPOSE_AVAILABLE = importlib.util.find_spec("cellpose") is not None


def _load_cellpose(model_type: str = "nuclei", gpu: bool = False):
    """Instantiate a Cellpose model across the 2.x/3.x and 4.x APIs.

    Cellpose 4 replaced the per-tissue models with a single ``cpsam`` model and
    dropped the ``model_type`` argument, so both call signatures are tried.
    """
    from cellpose import models

    try:  # Cellpose 2.x / 3.x
        return models.Cellpose(gpu=gpu, model_type=model_type), "legacy"
    except (TypeError, AttributeError, ValueError):
        pass
    try:  # Cellpose 4.x
        return models.CellposeModel(gpu=gpu), "cpsam"
    except Exception as exc:  # pragma: no cover - depends on the installed version
        raise RuntimeError(f"could not construct a Cellpose model: {exc}") from exc


def cellpose_segment(
    image: np.ndarray, model=None, api: str = "legacy", diameter: float | None = None
) -> np.ndarray:
    """Segment one grayscale image with Cellpose, returning instance labels."""
    if model is None:
        model, api = _load_cellpose()

    arr = np.asarray(image, dtype=np.float32)
    # channels=[0, 0] means "grayscale, no second channel", which is the right
    # setting for a nuclei-only field.
    if api == "legacy":
        masks = model.eval(arr, diameter=diameter, channels=[0, 0])[0]
    else:
        masks = model.eval(arr, diameter=diameter)[0]
    return np.asarray(masks, dtype=np.int32)


def run_cellpose_benchmark(
    cache_dir: str | Path,
    image_ids: list[str],
    preprocess_cfg=None,
    diameter: float | None = None,
    gpu: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate Cellpose on cached samples with the project's own metrics.

    Cellpose is given the *preprocessed* image, the same input the U-Net sees,
    so the two are compared on equal footing rather than one getting a cleaner
    signal than the other.
    """
    from microseg.data.bbbc038 import load_cached, to_microscopy_image
    from microseg.preprocessing.chain import preprocess_array

    if not CELLPOSE_AVAILABLE:
        raise RuntimeError("cellpose is not installed; `pip install cellpose`")

    cache_dir = Path(cache_dir)
    model, api = _load_cellpose(gpu=gpu)
    LOGGER.info("cellpose API: %s", api)

    timer = Timer()
    records: list[dict] = []
    for image_id in image_ids:
        sample = load_cached(cache_dir / f"{image_id}.npz")
        nuclei = to_microscopy_image(sample).nuclei()
        if preprocess_cfg is not None:
            nuclei, _, _ = preprocess_array(nuclei, preprocess_cfg)

        with timer.lap():
            labels = cellpose_segment(nuclei, model, api, diameter)

        record: dict = {"image_id": image_id}
        record.update(semantic_metrics(labels > 0, sample.labels > 0))
        record.update(instance_metrics(labels, sample.labels))
        records.append(record)

    df = pd.DataFrame(records)
    summary = aggregate(records)
    summary["n_images"] = len(records)
    summary["backend"] = f"cellpose ({api})"
    summary.update({f"time_{k}": round(v, 4) for k, v in timer.summary().items()})
    return df, summary
