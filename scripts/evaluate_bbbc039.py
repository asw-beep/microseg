"""Cross-dataset evaluation: a model trained on BBBC038, tested on BBBC039.

    python scripts/evaluate_bbbc039.py --limit 100

BBBC038 and BBBC039 are different experiments -- different cell line, different
microscope, different plate -- so this measures the thing the corruption sweep
only proxies: whether the model generalises past the dataset it was fitted to.
Nothing here is fine-tuned; the checkpoint never saw a BBBC039 pixel.

Two details decide whether the ground truth is trustworthy:

* **The masks are graph-coloured, not instance-numbered.** Values are 0..3, with
  adjacent nuclei given different values so they stay separable. Taking
  ``mask > 0`` merges every touching pair -- on the first field that is 94
  objects where there are really 110, which would flatter any model that
  under-segments. :func:`decode_instances` labels each colour separately.
* **Images are 12-bit stored in uint16.** Values run to 4095, not 65535, so the
  dtype scaling in ``MicroscopyImage.from_array`` leaves them at ~6% of range.
  Percentile normalisation in the preprocessing chain recovers this, which is
  exactly why that stage is not optional.

Metrics come from :mod:`microseg.metrics`, the same functions the in-dataset
evaluation uses, so the numbers sit directly beside the BBBC038 results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from skimage.measure import label as connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microseg.channels import ChannelSpec, MicroscopyImage  # noqa: E402
from microseg.config import load_config  # noqa: E402
from microseg.metrics.instance import instance_metrics  # noqa: E402
from microseg.metrics.semantic import aggregate, semantic_metrics  # noqa: E402
from microseg.pipeline import SegmentationPipeline  # noqa: E402
from microseg.utils import LOGGER, Timer, ensure_dir, setup_logging  # noqa: E402

DATA_ROOT = Path("data/raw/BBBC039")


def decode_instances(mask_path: Path) -> np.ndarray:
    """BBBC039 graph-coloured PNG -> instance labels with one id per nucleus.

    Each colour is labelled independently and the ids are offset so they stay
    unique across colours. Two nuclei that touch carry different colours by
    construction, so they survive as separate objects.
    """
    mask = np.asarray(Image.open(mask_path))
    colours = mask[:, :, 0] if mask.ndim == 3 else mask

    instances = np.zeros(colours.shape, dtype=np.int32)
    next_id = 0
    for colour in np.unique(colours):
        if colour == 0:
            continue
        labelled = connected_components(colours == colour)
        hit = labelled > 0
        instances[hit] = labelled[hit] + next_id
        next_id += int(labelled.max())
    return instances


def pairs(limit: int, offset: int = 0) -> list[tuple[Path, Path]]:
    """Matched (image, mask) paths, skipping the __MACOSX junk in the archive.

    ``offset`` exists so the set can be split into disjoint halves and the
    second used to check that a result from the first replicates. The ordering
    is filename order, which is plate/well order -- arbitrary with respect to
    anything the model cares about, but not random, so a subset is a convenience
    rather than a sample.
    """
    images = sorted(p for p in (DATA_ROOT / "images").iterdir() if p.suffix == ".tif")
    out = []
    for image_path in images:
        mask_path = DATA_ROOT / "masks" / f"{image_path.stem}.png"
        if mask_path.exists():
            out.append((image_path, mask_path))
    return out[offset : offset + limit] if limit else out[offset:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unet_cpu.yaml")
    parser.add_argument("--checkpoint", default="outputs/unet_cpu/best.pt")
    parser.add_argument("--backend", default="unet", choices=["unet", "classical"])
    parser.add_argument("--limit", type=int, default=100,
                        help="how many images to evaluate; 0 for all remaining")
    parser.add_argument("--offset", type=int, default=0,
                        help="skip this many images first, for disjoint subsets")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    setup_logging()
    available = pairs(0, 0)
    if not available:
        LOGGER.error("no BBBC039 data under %s -- download it first", DATA_ROOT)
        return 1

    samples = pairs(args.limit, args.offset)
    if not samples:
        # Distinguish "nothing on disk" from "you asked for a range past the end",
        # which are one typo apart and have completely different fixes.
        LOGGER.error(
            "offset %d is past the %d available fields -- nothing to evaluate",
            args.offset, len(available),
        )
        return 1

    # The field range is part of the run's identity, so it goes in the output
    # directory name rather than just the offset. Keying on backend alone means
    # a five-image smoke test overwrites a full hundred-image run in place, and
    # every downstream statistic then changes with no error.
    first, last = args.offset, args.offset + len(samples)

    if args.backend == "unet":
        pipeline = SegmentationPipeline.from_checkpoint(args.checkpoint, device="cpu")
    else:
        pipeline = SegmentationPipeline(load_config(args.config), backend="classical")

    LOGGER.info("BBBC039: %d images (offset %d), backend %s",
                len(samples), args.offset, args.backend)

    timer = Timer()
    records = []
    for i, (image_path, mask_path) in enumerate(samples, 1):
        raw = tifffile.imread(image_path)
        truth = decode_instances(mask_path)

        image = MicroscopyImage.from_array(
            raw, [ChannelSpec("nuclei", "nuclei")], name=image_path.stem
        )
        with timer.lap():
            result = pipeline.run(image)

        record = {"image_id": image_path.stem}
        record.update(semantic_metrics(result.foreground, truth > 0))
        record.update(instance_metrics(result.instances, truth))
        records.append(record)

        if i % 20 == 0 or i == len(samples):
            LOGGER.info("  %3d/%d  running AP %.4f", i, len(samples),
                        float(np.mean([r["ap_mean"] for r in records])))

    df = pd.DataFrame(records)
    summary = aggregate(records)
    summary["n_images"] = len(records)
    summary["backend"] = pipeline.backend
    summary["dataset"] = "BBBC039"
    summary["offset"] = args.offset
    summary["first_field"] = first
    summary["last_field"] = last
    summary.update({f"time_{k}": round(v, 4) for k, v in timer.summary().items()})

    default_dir = f"outputs/bbbc039/{args.backend}_f{first}-{last}"
    out_dir = ensure_dir(args.output_dir or default_dir)
    df.to_csv(out_dir / "per_image_metrics.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOGGER.info("AP %.4f | F1 %.4f | dice %.4f | count rel err %.3f -> %s",
                summary["ap_mean"], summary["f1"], summary["dice"],
                summary["count_rel_error"], out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
