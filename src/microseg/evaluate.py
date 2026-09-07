"""Evaluation: accuracy, robustness and inference time.

Reports three things, because a segmentation model is only deployable if all
three hold:

* **Accuracy** -- pixel Dice/IoU *and* instance precision/recall/AP.  Both,
  always: pixel metrics hide merged nuclei and instance metrics hide sloppy
  outlines, and a model can look good on either alone.
* **Robustness** -- the same metrics after realistic corruptions (noise, defocus,
  exposure drift, uneven illumination).  A model that scores 0.90 on clean data
  and 0.45 under mild defocus is not usable on a real plate, where focus drifts
  across wells.
* **Inference time** -- per-image, split into segmentation and analysis, because
  the analysis stage is often the slower half and that is a surprise worth
  measuring rather than assuming.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from microseg.channels import ChannelSpec, MicroscopyImage
from microseg.config import ExperimentConfig, load_config
from microseg.data.bbbc038 import load_cached, to_microscopy_image
from microseg.data.splits import load_splits
from microseg.metrics.instance import instance_metrics
from microseg.metrics.semantic import aggregate, semantic_metrics
from microseg.pipeline import SegmentationPipeline
from microseg.preprocessing.illumination import synthetic_illumination_gradient
from microseg.utils import LOGGER, Timer, ensure_dir, setup_logging

# ------------------------------------------------------------------ corruptions
# Each simulates a specific, common microscopy failure mode.  Kept mild on
# purpose: these are the conditions a working plate scan actually produces, not
# adversarial worst cases.
CORRUPTIONS = {
    "gaussian_noise": lambda a, r: np.clip(
        a + r.normal(0, 0.05, a.shape).astype(np.float32), 0, 1
    ),
    "heavy_noise": lambda a, r: np.clip(
        a + r.normal(0, 0.12, a.shape).astype(np.float32), 0, 1
    ),
    "defocus_blur": lambda a, r: cv2.GaussianBlur(a, (9, 9), 2.0),
    "low_exposure": lambda a, r: np.clip(a * 0.5, 0, 1),
    "high_exposure": lambda a, r: np.clip(a * 1.6, 0, 1),
    "uneven_illumination": lambda a, r: np.clip(
        a * synthetic_illumination_gradient(a.shape, 0.6, seed=int(r.integers(1 << 30))), 0, 1
    ),
    "jpeg_artifacts": lambda a, r: cv2.imdecode(
        cv2.imencode(".jpg", (a * 255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 30])[1],
        cv2.IMREAD_GRAYSCALE,
    ).astype(np.float32)
    / 255.0,
}


def evaluate_split(
    pipeline: SegmentationPipeline,
    cache_dir: str | Path,
    image_ids: list[str],
    corruption: str | None = None,
    seed: int = 0,
    save_examples: int = 0,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate one backend over a list of cached samples.

    Returns a per-image metric table and an aggregate summary.
    """
    cache_dir = Path(cache_dir)
    rng = np.random.default_rng(seed)
    timer = Timer()

    records: list[dict] = []
    for i, image_id in enumerate(image_ids):
        sample = load_cached(cache_dir / f"{image_id}.npz")
        nuclei = to_microscopy_image(sample).nuclei()

        if corruption:
            nuclei = CORRUPTIONS[corruption](nuclei.astype(np.float32), rng)

        image = MicroscopyImage.from_array(
            nuclei, [ChannelSpec("nuclei", "nuclei")], name=image_id
        )

        with timer.lap():
            result = pipeline.run(image)

        record: dict = {"image_id": image_id}
        record.update(semantic_metrics(result.foreground, sample.labels > 0))
        record.update(instance_metrics(result.instances, sample.labels))
        record.update({k: round(v, 4) for k, v in result.timings.items()})
        records.append(record)

        if save_examples and i < save_examples and output_dir:
            from microseg.viz import comparison_figure

            comparison_figure(
                image.nuclei(),
                result.instances,
                sample.labels,
                title=f"{image_id} ({pipeline.backend})",
                save_path=Path(output_dir) / "examples" / f"{image_id}.png",
            )

    df = pd.DataFrame(records)
    summary = aggregate(records)
    summary["n_images"] = len(records)
    summary["backend"] = pipeline.backend
    summary["corruption"] = corruption or "none"
    summary.update({f"time_{k}": round(v, 4) for k, v in timer.summary().items()})
    return df, summary


def evaluate_robustness(
    pipeline: SegmentationPipeline,
    cache_dir: str | Path,
    image_ids: list[str],
    corruptions: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the clean evaluation plus every corruption, and report the drop.

    ``dice_drop``/``ap_drop`` are reported relative to the clean baseline, which
    is what makes two backends comparable even when their absolute scores differ.
    """
    corruptions = corruptions or list(CORRUPTIONS)
    rows: list[dict] = []

    _, clean = evaluate_split(pipeline, cache_dir, image_ids, None, seed)
    clean_dice, clean_ap = clean.get("dice", 0.0), clean.get("ap_mean", 0.0)
    clean.update({"dice_drop": 0.0, "ap_drop": 0.0, "dice_retention": 1.0})
    rows.append(clean)

    for name in corruptions:
        LOGGER.info("  corruption: %s", name)
        _, summary = evaluate_split(pipeline, cache_dir, image_ids, name, seed)
        summary["dice_drop"] = clean_dice - summary.get("dice", 0.0)
        summary["ap_drop"] = clean_ap - summary.get("ap_mean", 0.0)
        summary["dice_retention"] = (
            summary.get("dice", 0.0) / clean_dice if clean_dice > 0 else 0.0
        )
        rows.append(summary)

    return pd.DataFrame(rows)


def run_evaluation(
    cfg: ExperimentConfig,
    backend: str = "classical",
    checkpoint: str | Path | None = None,
    split: str = "test",
    limit: int | None = None,
    robustness: bool = False,
    device: str = "cpu",
    tta: bool = False,
) -> dict:
    """Evaluate one backend on one split and write results to the run directory."""
    setup_logging()

    cache_dir = Path(cfg.data.processed_root) / "cache"
    split_path = Path(cfg.data.processed_root) / f"splits_seed{cfg.data.split_seed}.json"
    run_split = cfg.run_dir / "splits.json"
    splits = load_splits(run_split if run_split.exists() else split_path)

    image_ids = splits[split]
    if limit:
        image_ids = image_ids[: int(limit)]
    if not image_ids:
        raise RuntimeError(f"split {split!r} is empty")

    if backend == "unet":
        if checkpoint is None:
            checkpoint = cfg.run_dir / "best.pt"
        pipeline = SegmentationPipeline.from_checkpoint(checkpoint, device=device, tta=tta)
    else:
        pipeline = SegmentationPipeline(cfg, backend="classical")

    output_dir = ensure_dir(cfg.run_dir / f"eval_{backend}_{split}")
    LOGGER.info("evaluating %s on %d %s image(s)", backend, len(image_ids), split)

    per_image, summary = evaluate_split(
        pipeline, cache_dir, image_ids, save_examples=6, output_dir=output_dir
    )
    per_image.to_csv(output_dir / "per_image_metrics.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOGGER.info(
        "%s | dice %.4f  iou %.4f  precision %.4f  recall %.4f  F1 %.4f  AP %.4f  %.3f s/img",
        backend,
        summary.get("dice", 0),
        summary.get("iou", 0),
        summary.get("precision", 0),
        summary.get("recall", 0),
        summary.get("f1", 0),
        summary.get("ap_mean", 0),
        summary.get("time_mean_s", 0),
    )

    if robustness:
        LOGGER.info("running robustness sweep")
        robustness_df = evaluate_robustness(pipeline, cache_dir, image_ids[: min(len(image_ids), 25)])
        robustness_df.to_csv(output_dir / "robustness.csv", index=False)
        LOGGER.info(
            "\n%s",
            robustness_df[["corruption", "dice", "ap_mean", "dice_drop", "ap_drop"]].to_string(index=False),
        )

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation backend")
    parser.add_argument("--config", type=str, default="configs/unet_bbbc038.yaml")
    parser.add_argument("--backend", choices=["classical", "unet"], default="classical")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tta", action="store_true", help="8x test-time augmentation")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_evaluation(
        cfg,
        backend=args.backend,
        checkpoint=args.checkpoint,
        split=args.split,
        limit=args.limit,
        robustness=args.robustness,
        device=args.device,
        tta=args.tta,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
