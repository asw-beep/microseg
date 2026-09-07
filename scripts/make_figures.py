"""Generate the report figures from a trained model.

    python scripts/make_figures.py --config configs/unet_cpu.yaml

Writes to ``assets/`` by default:

* ``preprocessing_stages.png`` -- the chain applied to one real field, which is
  the figure to look at first when a segmentation goes wrong;
* ``qualitative_<id>.png`` -- input / prediction / ground truth / error map, for
  each backend, chosen to include both an easy and a hard case;
* ``probability_maps.png`` -- the three predicted class maps, where a weak
  boundary channel visibly predicts merged instances;
* ``feature_distributions.png`` -- population morphology across the test split;
* ``training_curves.png`` -- copied from the run directory.

Images are picked by *difficulty* rather than at random: showing only easy
fields would misrepresent the method.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microseg.channels import ChannelSpec, MicroscopyImage  # noqa: E402
from microseg.config import load_config  # noqa: E402
from microseg.data.bbbc038 import load_cached, to_microscopy_image  # noqa: E402
from microseg.data.splits import load_splits  # noqa: E402
from microseg.pipeline import SegmentationPipeline  # noqa: E402
from microseg.utils import LOGGER, ensure_dir, setup_logging  # noqa: E402
from microseg.viz import (  # noqa: E402
    comparison_figure,
    feature_distributions,
    preprocessing_figure,
    probability_figure,
)


def pick_examples(cache_dir: Path, image_ids: list[str], n: int = 3) -> list[str]:
    """Choose a sparse, a typical and a crowded field.

    Object count is a good proxy for difficulty here: crowded fields are where
    instance separation actually gets tested.
    """
    counts = []
    for image_id in image_ids:
        sample = load_cached(cache_dir / f"{image_id}.npz")
        counts.append((sample.n_objects, image_id))
    counts.sort()
    if len(counts) < n:
        return [i for _, i in counts]
    picks = [counts[len(counts) // 6], counts[len(counts) // 2], counts[-2]]
    return [image_id for _, image_id in picks]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument("--config", default="configs/unet_cpu.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="assets")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-examples", type=int, default=3)
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    cache_dir = Path(cfg.data.processed_root) / "cache"
    out_dir = ensure_dir(args.output)

    run_split = cfg.run_dir / "splits.json"
    default_split = Path(cfg.data.processed_root) / f"splits_seed{cfg.data.split_seed}.json"
    splits = load_splits(run_split if run_split.exists() else default_split)
    image_ids = splits[args.split]

    # ------------------------------------------------------------- backends
    backends: dict[str, SegmentationPipeline] = {
        "classical": SegmentationPipeline(cfg, backend="classical")
    }
    checkpoint = Path(args.checkpoint) if args.checkpoint else cfg.run_dir / "best.pt"
    if checkpoint.exists():
        backends["unet"] = SegmentationPipeline.from_checkpoint(checkpoint, "cpu")
        LOGGER.info("U-Net loaded from %s", checkpoint)
    else:
        LOGGER.warning("no checkpoint at %s; classical figures only", checkpoint)

    examples = pick_examples(cache_dir, image_ids, args.n_examples)
    LOGGER.info("example images: %s", [i[:12] for i in examples])

    # ------------------------------------------------- preprocessing figure
    sample = load_cached(cache_dir / f"{examples[-1]}.npz")
    image = MicroscopyImage.from_array(
        to_microscopy_image(sample).nuclei(),
        [ChannelSpec("nuclei", "nuclei")],
        name=sample.image_id,
    )
    result = backends["classical"].run(image, collect_stages=True)
    preprocessing_figure(result.stages, out_dir / "preprocessing_stages.png")
    LOGGER.info("wrote preprocessing_stages.png")

    # -------------------------------------------------- qualitative figures
    for image_id in examples:
        sample = load_cached(cache_dir / f"{image_id}.npz")
        image = MicroscopyImage.from_array(
            to_microscopy_image(sample).nuclei(),
            [ChannelSpec("nuclei", "nuclei")],
            name=image_id,
        )
        for name, pipeline in backends.items():
            result = pipeline.run(image)
            comparison_figure(
                result.preprocessed.nuclei(),
                result.instances,
                sample.labels,
                title=(
                    f"{name} -- {image_id[:12]} -- "
                    f"{result.n_objects} predicted vs {sample.n_objects} true"
                ),
                save_path=out_dir / f"qualitative_{name}_{image_id[:12]}.png",
            )
            if name == "unet" and result.probabilities is not None:
                probability_figure(
                    result.probabilities, out_dir / f"probability_maps_{image_id[:12]}.png"
                )
    LOGGER.info("wrote qualitative figures")

    # ------------------------------------------------ population morphology
    pipeline = backends.get("unet", backends["classical"])
    frames = []
    for image_id in image_ids[:40]:
        sample = load_cached(cache_dir / f"{image_id}.npz")
        image = MicroscopyImage.from_array(
            to_microscopy_image(sample).nuclei(),
            [ChannelSpec("nuclei", "nuclei")],
            name=image_id,
        )
        features = pipeline.run(image).features
        if features is not None and not features.empty:
            frames.append(features)

    if frames:
        population = pd.concat(frames, ignore_index=True)
        feature_distributions(
            population,
            ["area_px", "circularity", "eccentricity", "solidity"],
            out_dir / "feature_distributions.png",
        )
        population.to_csv(out_dir / "population_features.csv", index=False)
        LOGGER.info(
            "wrote feature_distributions.png over %d nuclei from %d fields",
            len(population),
            len(frames),
        )

    # ---------------------------------------------------- training curves
    curves = cfg.run_dir / "training_curves.png"
    if curves.exists():
        shutil.copy(curves, out_dir / "training_curves.png")
        LOGGER.info("copied training_curves.png")

    LOGGER.info("figures -> %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
