"""Compare every segmentation backend on the same held-out images.

    python scripts/run_benchmarks.py --config configs/unet_cpu.yaml

Runs whichever of these are available and writes one comparison table:

* **classical** -- threshold + morphology + distance watershed
* **unet** -- this project's model, if a checkpoint exists
* **cellpose** -- the general-purpose reference model, if installed
* **sam** -- foundation-model experiment, if installed and weights are present

Every method sees the same preprocessed images and the same test split, and is
scored with the same metrics, so the numbers are comparable.  Missing optional
dependencies are reported as skipped rather than failing the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microseg.config import load_config  # noqa: E402
from microseg.data.splits import load_splits  # noqa: E402
from microseg.evaluate import evaluate_split  # noqa: E402
from microseg.pipeline import SegmentationPipeline  # noqa: E402
from microseg.utils import LOGGER, ensure_dir, setup_logging  # noqa: E402

# Columns worth putting in front of a reader, in order.
REPORT_COLUMNS = [
    "backend",
    "dice",
    "iou",
    "precision",
    "recall",
    "f1",
    "ap_mean",
    "mean_matched_iou",
    "count_abs_error",
    "time_mean_s",
]


def _resolve_splits(cfg):
    run_split = cfg.run_dir / "splits.json"
    default = Path(cfg.data.processed_root) / f"splits_seed{cfg.data.split_seed}.json"
    path = run_split if run_split.exists() else default
    if not path.exists():
        raise SystemExit(f"no split file at {path}; run scripts/prepare_data.py first")
    return load_splits(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark all segmentation backends")
    parser.add_argument("--config", default="configs/unet_cpu.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=50, help="images per backend")
    parser.add_argument("--sam-limit", type=int, default=5, help="SAM is slow; use fewer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip", nargs="*", default=[], help="backends to skip")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    cache_dir = Path(cfg.data.processed_root) / "cache"
    image_ids = _resolve_splits(cfg)[args.split][: args.limit]
    LOGGER.info("benchmarking on %d %s image(s)", len(image_ids), args.split)

    output_dir = ensure_dir(Path(args.output) if args.output else cfg.run_dir / "benchmarks")
    rows: list[dict] = []
    skipped: dict[str, str] = {}

    # ------------------------------------------------------------- classical
    if "classical" not in args.skip:
        LOGGER.info("running classical baseline")
        _, summary = evaluate_split(
            SegmentationPipeline(cfg, backend="classical"), cache_dir, image_ids
        )
        rows.append(summary)

    # ------------------------------------------------------------------ unet
    if "unet" not in args.skip:
        checkpoint = Path(args.checkpoint) if args.checkpoint else cfg.run_dir / "best.pt"
        if checkpoint.exists():
            LOGGER.info("running U-Net from %s", checkpoint)
            pipeline = SegmentationPipeline.from_checkpoint(checkpoint, args.device)
            _, summary = evaluate_split(pipeline, cache_dir, image_ids)
            summary["backend"] = "unet"
            rows.append(summary)
        else:
            skipped["unet"] = f"no checkpoint at {checkpoint}"

    # -------------------------------------------------------------- cellpose
    if "cellpose" not in args.skip:
        from microseg.benchmarks import CELLPOSE_AVAILABLE, run_cellpose_benchmark

        if CELLPOSE_AVAILABLE:
            LOGGER.info("running Cellpose")
            try:
                _, summary = run_cellpose_benchmark(
                    cache_dir, image_ids, preprocess_cfg=cfg.preprocess
                )
                rows.append(summary)
            except Exception as exc:
                LOGGER.warning("cellpose failed: %s", exc)
                skipped["cellpose"] = str(exc)
        else:
            skipped["cellpose"] = "cellpose not installed"

    # ------------------------------------------------------------------- sam
    if "sam" not in args.skip:
        from microseg.benchmarks import run_sam_experiment

        LOGGER.info("running SAM experiment on %d image(s)", args.sam_limit)
        sam_results = run_sam_experiment(
            cache_dir,
            image_ids[: args.sam_limit],
            device=args.device,
            preprocess_cfg=cfg.preprocess,
        )
        if sam_results.get("status") == "ok":
            for mode, summary in sam_results["modes"].items():
                summary["backend"] = f"sam ({mode})"
                rows.append(summary)
            (output_dir / "sam_experiment.json").write_text(
                json.dumps(sam_results, indent=2), encoding="utf-8"
            )
        else:
            skipped["sam"] = sam_results.get("reason", "unavailable")

    # ---------------------------------------------------------------- report
    df = pd.DataFrame(rows)
    columns = [c for c in REPORT_COLUMNS if c in df.columns]
    df = df[columns + [c for c in df.columns if c not in columns]]
    df.to_csv(output_dir / "comparison.csv", index=False)

    LOGGER.info("\n%s", df[columns].round(4).to_string(index=False))
    if skipped:
        LOGGER.info("skipped: %s", json.dumps(skipped, indent=2))
        (output_dir / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    LOGGER.info("results -> %s", output_dir / "comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
