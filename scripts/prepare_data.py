"""Cache BBBC038 samples and write the train/val/test split.

    python scripts/prepare_data.py --config configs/unet_bbbc038.yaml

Merges each sample's per-nucleus mask PNGs into one instance label array and
stores it as a compressed ``.npz``.  This turns ~30 PNG decodes per sample per
epoch into a single array read, which is the difference between the dataloader
and the GPU being the bottleneck.

The split is derived from ``data.split_seed`` and written to disk so every later
script reads the same partition instead of re-deriving it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microseg.config import load_config  # noqa: E402
from microseg.data.bbbc038 import list_samples, load_cached, prepare_cache  # noqa: E402
from microseg.data.splits import make_splits  # noqa: E402
from microseg.utils import LOGGER, setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cache BBBC038 and write splits")
    parser.add_argument("--config", default="configs/unet_bbbc038.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    limit = args.limit if args.limit is not None else cfg.data.limit

    cache_dir = Path(cfg.data.processed_root) / "cache"
    LOGGER.info("caching %s -> %s", cfg.data.root, cache_dir)
    prepare_cache(cfg.data.root, cache_dir, limit=limit, overwrite=args.overwrite)

    ids = list_samples(cfg.data.root)
    if limit:
        ids = ids[:limit]

    splits = make_splits(ids, cfg.data.val_fraction, cfg.data.test_fraction, cfg.data.split_seed)
    split_path = Path(cfg.data.processed_root) / f"splits_seed{cfg.data.split_seed}.json"
    splits.save(split_path)
    LOGGER.info("splits %s -> %s", splits.counts, split_path)

    # A quick sanity readout: image size and object-count range across the set,
    # which is the first thing worth knowing about a heterogeneous dataset.
    sizes, counts = set(), []
    for image_id in ids[:200]:
        sample = load_cached(cache_dir / f"{image_id}.npz")
        sizes.add(sample.image.shape[:2])
        counts.append(sample.n_objects)
    LOGGER.info("distinct image sizes (first 200): %s", sorted(sizes))
    LOGGER.info(
        "nuclei per image: min %d, median %d, max %d",
        min(counts), sorted(counts)[len(counts) // 2], max(counts),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
