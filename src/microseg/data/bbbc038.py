"""BBBC038 (Broad Bioimage Benchmark Collection / Data Science Bowl 2018) loader.

On-disk layout of ``stage1_train``::

    <image_id>/
        images/<image_id>.png      # RGB or RGBA micrograph
        masks/<mask_id>.png        # one binary PNG per nucleus

The per-nucleus masks are the reason this dataset is worth using: they give true
*instance* ground truth, so instance metrics (matched IoU, count error) are
available rather than only pixel-wise Dice.

The set is deliberately heterogeneous -- fluorescence, brightfield and H&E, at
several magnifications and image sizes.  That heterogeneity is the point: it is
what separates a pipeline that generalises from one tuned to a single assay.

Decoding ~30 mask PNGs per sample per epoch is slow, so :func:`prepare_cache`
converts each sample once into a compressed ``.npz`` holding the image and a
single int32 instance-label array.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from microseg.channels import ChannelSpec, MicroscopyImage
from microseg.utils import LOGGER


@dataclass
class BBBC038Sample:
    """One image plus its instance label map."""

    image_id: str
    image: np.ndarray  # (H, W, C) uint8
    labels: np.ndarray  # (H, W) int32, 0 = background

    @property
    def n_objects(self) -> int:
        return int(self.labels.max())


def list_samples(root: str | Path) -> list[str]:
    """List sample ids under a BBBC038 split directory."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"dataset root {root} does not exist; run scripts/download_data.py first"
        )
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "images").is_dir())


def _read_image(path: Path) -> np.ndarray:
    """Read a PNG as uint8, dropping any alpha channel."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"could not read image {path}")
    if img.ndim == 2:
        return img[:, :, None]
    if img.shape[2] == 4:
        img = img[:, :, :3]
    # OpenCV reads BGR; convert so channel order matches the file's RGB.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_sample(root: str | Path, image_id: str) -> BBBC038Sample:
    """Load one sample, merging the per-nucleus PNGs into an instance map."""
    sample_dir = Path(root) / image_id
    image_paths = sorted((sample_dir / "images").glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"no image PNG under {sample_dir / 'images'}")
    image = _read_image(image_paths[0])

    labels = np.zeros(image.shape[:2], dtype=np.int32)
    mask_dir = sample_dir / "masks"
    if mask_dir.is_dir():
        for i, mask_path in enumerate(sorted(mask_dir.glob("*.png")), start=1):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                LOGGER.warning("skipping unreadable mask %s", mask_path)
                continue
            # Annotations occasionally overlap by a pixel; last writer wins, which
            # keeps every object present rather than dropping one entirely.
            labels[mask > 0] = i
    return BBBC038Sample(image_id, image, labels)


def to_microscopy_image(
    sample: BBBC038Sample, nuclei_from: str = "auto"
) -> MicroscopyImage:
    """Wrap a sample as a single-channel nuclei :class:`MicroscopyImage`.

    BBBC038 images are stored as RGB but are not truly three-channel assays:
    fluorescence images put nearly all signal in one channel, and H&E images are
    genuine colour.  ``auto`` picks the channel with the highest variance, which
    recovers the stained channel in the fluorescent case and a reasonable
    grayscale proxy otherwise.

    A real multiplexed dataset would instead keep every channel and tag roles;
    that path is :meth:`MicroscopyImage.from_array` with explicit specs.
    """
    img = sample.image
    if img.shape[2] == 1:
        gray = img[:, :, 0]
    elif nuclei_from == "auto":
        variances = img.reshape(-1, img.shape[2]).astype(np.float32).var(axis=0)
        spread = variances.max() - variances.min()
        if spread > 0.15 * max(variances.max(), 1e-6):
            gray = img[:, :, int(np.argmax(variances))]
        else:
            # Channels carry comparable information (H&E): use luminance.
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif nuclei_from == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img[:, :, int(nuclei_from)]

    return MicroscopyImage.from_array(
        gray, [ChannelSpec("nuclei", "nuclei")], name=sample.image_id
    )


def prepare_cache(
    root: str | Path,
    cache_dir: str | Path,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Convert a BBBC038 split into per-sample ``.npz`` caches.

    Each cache holds ``image`` (uint8 RGB) and ``labels`` (int32 instances).
    """
    root, cache_dir = Path(root), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ids = list_samples(root)
    if limit is not None:
        ids = ids[: int(limit)]

    written: list[Path] = []
    for i, image_id in enumerate(ids, start=1):
        out_path = cache_dir / f"{image_id}.npz"
        if out_path.exists() and not overwrite:
            written.append(out_path)
            continue
        sample = load_sample(root, image_id)
        np.savez_compressed(out_path, image=sample.image, labels=sample.labels)
        written.append(out_path)
        if i % 50 == 0 or i == len(ids):
            LOGGER.info("cached %d/%d samples", i, len(ids))
    return written


def load_cached(path: str | Path) -> BBBC038Sample:
    """Load a sample written by :func:`prepare_cache`."""
    path = Path(path)
    with np.load(path) as data:
        return BBBC038Sample(path.stem, data["image"], data["labels"].astype(np.int32))
