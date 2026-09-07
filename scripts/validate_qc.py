"""Validate the QC layer against ground truth.

    python scripts/validate_qc.py --config configs/unet_cpu.yaml

A QC score is only worth having if it actually correlates with being wrong, so
this measures it rather than asserting it.  For each image in a split it runs the
pipeline, computes the QC verdict (which never sees ground truth) and joins it
against the real metrics.

Reported:

* **catch rate** -- of the images that genuinely failed, how many did QC flag?
* **false-alarm rate** -- of the images that were fine, how many did QC flag?
* **correlation** between QC confidence and actual Dice.

Thresholds are meant to be tuned on ``--split val`` and reported on
``--split test``.  Tuning and reporting on the same split would make the numbers
meaningless, which is the whole reason this takes a ``--split`` argument.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microseg.channels import ChannelSpec, MicroscopyImage  # noqa: E402
from microseg.config import load_config  # noqa: E402
from microseg.data.bbbc038 import load_cached, to_microscopy_image  # noqa: E402
from microseg.data.splits import load_splits  # noqa: E402
from microseg.metrics.instance import instance_metrics  # noqa: E402
from microseg.metrics.semantic import semantic_metrics  # noqa: E402
from microseg.pipeline import SegmentationPipeline  # noqa: E402
from microseg.utils import LOGGER, ensure_dir, setup_logging  # noqa: E402

# What counts as a genuine failure -- a result a biologist would call wrong, not
# merely imperfect.
FAILURE_DICE = 0.5
FAILURE_COUNT_REL_ERROR = 0.5


def collect(cfg, split: str, backend: str, checkpoint, device: str, limit=None) -> pd.DataFrame:
    """Run the pipeline over a split and pair QC verdicts with true metrics."""
    cache_dir = Path(cfg.data.processed_root) / "cache"
    run_split = cfg.run_dir / "splits.json"
    default_split = Path(cfg.data.processed_root) / f"splits_seed{cfg.data.split_seed}.json"
    splits = load_splits(run_split if run_split.exists() else default_split)

    image_ids = splits[split]
    if limit:
        image_ids = image_ids[:limit]

    if backend == "unet":
        pipeline = SegmentationPipeline.from_checkpoint(checkpoint, device)
    else:
        pipeline = SegmentationPipeline(cfg, backend="classical")

    rows: list[dict] = []
    for image_id in image_ids:
        sample = load_cached(cache_dir / f"{image_id}.npz")
        image = MicroscopyImage.from_array(
            to_microscopy_image(sample).nuclei(),
            [ChannelSpec("nuclei", "nuclei")],
            name=image_id,
        )
        result = pipeline.run(image)

        row: dict = {"image_id": image_id}
        row.update(semantic_metrics(result.foreground, sample.labels > 0))
        row.update(instance_metrics(result.instances, sample.labels))
        if result.qc is not None:
            row.update(result.qc.as_dict())
        rows.append(row)

    return pd.DataFrame(rows)


def report(df: pd.DataFrame, split: str) -> dict:
    """Summarise how well QC confidence predicts genuine failure."""
    failed = (df["dice"] < FAILURE_DICE) | (df["count_rel_error"] > FAILURE_COUNT_REL_ERROR)
    flagged = ~df["qc_passed"].astype(bool)

    n = len(df)
    n_failed = int(failed.sum())
    n_flagged = int(flagged.sum())
    caught = int((failed & flagged).sum())
    false_alarms = int((~failed & flagged).sum())
    missed = int((failed & ~flagged).sum())

    stats = {
        "split": split,
        "n_images": n,
        "n_genuine_failures": n_failed,
        "n_flagged_by_qc": n_flagged,
        "caught": caught,
        "missed": missed,
        "false_alarms": false_alarms,
        "catch_rate": round(caught / n_failed, 4) if n_failed else None,
        "false_alarm_rate": round(false_alarms / max(n - n_failed, 1), 4),
        "corr_confidence_dice": round(
            float(np.corrcoef(df["qc_confidence"], df["dice"])[0, 1]), 4
        ),
        "mean_dice_passed": round(float(df.loc[~flagged, "dice"].mean()), 4) if (~flagged).any() else None,
        "mean_dice_flagged": round(float(df.loc[flagged, "dice"].mean()), 4) if flagged.any() else None,
        "mean_ap_passed": round(float(df.loc[~flagged, "ap_mean"].mean()), 4) if (~flagged).any() else None,
        "mean_ap_flagged": round(float(df.loc[flagged, "ap_mean"].mean()), 4) if flagged.any() else None,
    }
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the QC layer against ground truth")
    parser.add_argument("--config", default="configs/unet_cpu.yaml")
    parser.add_argument("--backend", choices=["classical", "unet"], default="unet")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or (cfg.run_dir / "best.pt")
    out_dir = ensure_dir(Path(args.output) if args.output else cfg.run_dir / "qc")

    all_stats = []
    for split in args.splits:
        LOGGER.info("running %s on %s split", args.backend, split)
        df = collect(cfg, split, args.backend, checkpoint, args.device, args.limit)
        df.to_csv(out_dir / f"qc_{args.backend}_{split}.csv", index=False)

        stats = report(df, split)
        all_stats.append(stats)
        LOGGER.info("%s: %s", split, json.dumps(stats, indent=2))

        flagged = df[~df["qc_passed"].astype(bool)]
        if not flagged.empty:
            LOGGER.info(
                "flagged images on %s:\n%s",
                split,
                flagged[["image_id", "dice", "f1", "count_pred", "count_true", "qc_confidence", "qc_flags"]]
                .assign(image_id=lambda d: d["image_id"].str[:12])
                .to_string(index=False),
            )

    (out_dir / "qc_validation.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    LOGGER.info("results -> %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
