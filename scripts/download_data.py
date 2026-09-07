"""Download the BBBC038 (Data Science Bowl 2018) dataset.

    python scripts/download_data.py

Fetches stage1_train (670 annotated images) from the Broad Bioimage Benchmark
Collection and extracts it under ``data/raw/``.  Only stage1_train carries public
per-nucleus masks, so it is the only split that supports instance evaluation;
stage1_test is optional and unlabelled.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microseg.utils import LOGGER, ensure_dir, setup_logging  # noqa: E402

BASE_URL = "https://data.broadinstitute.org/bbbc/BBBC038"
ARCHIVES = {
    "stage1_train": "stage1_train.zip",  # 83 MB, 670 images with masks
    "stage1_test": "stage1_test.zip",  # 9 MB, 65 images, no public masks
}


def _progress(count: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(count * block_size, total_size)
    pct = 100.0 * done / total_size
    sys.stdout.write(f"\r    {pct:5.1f}%  ({done / 1e6:.1f} / {total_size / 1e6:.1f} MB)")
    sys.stdout.flush()


def download_archive(name: str, raw_dir: Path, force: bool = False) -> Path:
    """Download and extract one archive, skipping work already done."""
    archive_path = raw_dir / ARCHIVES[name]
    extract_dir = raw_dir / name

    if extract_dir.is_dir() and any(extract_dir.iterdir()) and not force:
        LOGGER.info("%s already extracted (%d samples)", name, len(list(extract_dir.iterdir())))
        return extract_dir

    if not archive_path.exists() or force:
        url = f"{BASE_URL}/{ARCHIVES[name]}"
        LOGGER.info("downloading %s", url)
        urllib.request.urlretrieve(url, archive_path, _progress)
        sys.stdout.write("\n")

    LOGGER.info("extracting %s", archive_path.name)
    ensure_dir(extract_dir)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    LOGGER.info("%s ready: %d samples", name, len(list(extract_dir.iterdir())))
    return extract_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download BBBC038")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--splits", nargs="+", default=["stage1_train"], choices=list(ARCHIVES)
    )
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args(argv)

    setup_logging()
    raw_dir = ensure_dir(args.raw_dir)
    for name in args.splits:
        download_archive(name, raw_dir, args.force)
        if not args.keep_archives:
            (raw_dir / ARCHIVES[name]).unlink(missing_ok=True)

    LOGGER.info("done. next: python scripts/prepare_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
