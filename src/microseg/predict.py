"""Run the full pipeline on image files and write quantitative results.

The user-facing entry point: point it at a microscopy image (or a folder of
them) and it writes a per-nucleus feature table, a per-image summary, an
instance label map and an overlay figure.

    python -m microseg.predict --input my_field.tif --backend unet \\
        --checkpoint outputs/unet_bbbc038/best.pt --output results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from microseg.config import load_config
from microseg.io_utils import list_images, load_image, save_labels
from microseg.pipeline import SegmentationPipeline, results_to_tables
from microseg.utils import LOGGER, ensure_dir, setup_logging


def predict_paths(
    paths: list[Path],
    pipeline: SegmentationPipeline,
    output_dir: Path,
    save_figures: bool = True,
    pixel_size_um: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the pipeline over image files and write all artifacts."""
    ensure_dir(output_dir)
    results = []

    for path in paths:
        LOGGER.info("processing %s", path.name)
        image = load_image(path, pixel_size_um=pixel_size_um)
        result = pipeline.run(image, collect_stages=save_figures)
        results.append(result)

        save_labels(result.instances, output_dir / "labels" / f"{image.name}_labels.png")
        if save_figures:
            from microseg.viz import instance_figure, preprocessing_figure

            instance_figure(
                result.preprocessed.nuclei(),
                result.instances,
                title=f"{image.name} -- {result.n_objects} nuclei ({pipeline.backend})",
                save_path=output_dir / "figures" / f"{image.name}_segmentation.png",
            )
            if result.stages:
                preprocessing_figure(
                    result.stages, output_dir / "figures" / f"{image.name}_preprocessing.png"
                )

        LOGGER.info(
            "  %d nuclei in %.3f s (preprocess %.3f, segment %.3f, analyse %.3f)",
            result.n_objects,
            result.timings["total_s"],
            result.timings["preprocess_s"],
            result.timings["segment_s"],
            result.timings["analysis_s"],
        )

    per_object, per_image = results_to_tables(results)
    per_object.to_csv(output_dir / "nuclei_features.csv", index=False)
    per_image.to_csv(output_dir / "image_summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps([r.summary for r in results], indent=2, default=str), encoding="utf-8"
    )
    return per_object, per_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Segment and quantify nuclei in microscopy images")
    parser.add_argument("--input", required=True, help="an image file or a folder of images")
    parser.add_argument("--output", default="outputs/predictions")
    parser.add_argument("--config", default="configs/unet_bbbc038.yaml")
    parser.add_argument("--backend", choices=["classical", "unet"], default="classical")
    parser.add_argument("--checkpoint", default=None, help="required for --backend unet")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--pixel-size-um", type=float, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)

    source = Path(args.input)
    paths = list_images(source) if source.is_dir() else [source]
    if not paths:
        raise SystemExit(f"no readable images found at {source}")

    if args.backend == "unet":
        checkpoint = args.checkpoint or (cfg.run_dir / "best.pt")
        if not Path(checkpoint).exists():
            raise SystemExit(f"checkpoint {checkpoint} not found; train a model first")
        pipeline = SegmentationPipeline.from_checkpoint(checkpoint, args.device, tta=args.tta)
    else:
        pipeline = SegmentationPipeline(cfg, backend="classical")

    per_object, per_image = predict_paths(
        paths,
        pipeline,
        Path(args.output),
        save_figures=not args.no_figures,
        pixel_size_um=args.pixel_size_um,
    )
    LOGGER.info(
        "wrote %d object(s) across %d image(s) to %s",
        len(per_object),
        len(per_image),
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
