"""PyTorch dataset and dataloaders.

The central design decision lives in :func:`instances_to_semantic`: instance
ground truth is converted into a **three-class** semantic target -- background,
nucleus interior, nucleus boundary.

A plain binary foreground target cannot represent touching nuclei: two nuclei
that share an edge are one connected component, and no amount of post-processing
recovers the split reliably.  Predicting the boundary as its own class gives the
network an explicit incentive to carve a one-pixel gap between neighbours, and
that gap is what the watershed in :mod:`microseg.instance` floods from.  This is
the same idea as the original U-Net's weighted separation loss, expressed as a
class instead of a per-pixel weight map -- simpler to reason about and it makes
the boundary quality directly measurable.

The boundary class has one failure it cannot be tuned out of, and
:func:`instances_to_distance` is the answer to it.  When two nuclei *overlap*
rather than merely touch, the annotation gives them no shared edge, so there is
no boundary band to predict and nothing for the watershed to cut along.  The
normalised distance transform still has two distinct maxima in that case -- one
per nucleus -- because it is computed per object rather than on the merged
foreground.  A model with both targets can seed from whichever signal survives.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
from torch.utils.data import DataLoader, Dataset

from microseg.config import DataConfig, PreprocessConfig, TrainConfig
from microseg.data.bbbc038 import BBBC038Sample, load_cached, to_microscopy_image
from microseg.data.transforms import AugmentParams, augment, center_crop, random_crop
from microseg.preprocessing.chain import preprocess_array

BACKGROUND, INTERIOR, BOUNDARY = 0, 1, 2


def instances_to_semantic(labels: np.ndarray, boundary_width: int = 2) -> np.ndarray:
    """Instance labels -> 3-class semantic map.

    ``boundary_width`` trades separation power against mask accuracy: a wider
    band separates crowded nuclei more reliably but eats into the interior, so
    predicted objects come out slightly small.  Two pixels is a good compromise
    at BBBC038's typical nucleus size (~20 px across).
    """
    labels = np.asarray(labels, dtype=np.int32)
    target = np.zeros(labels.shape, dtype=np.int64)
    if labels.max() == 0:
        return target

    target[labels > 0] = INTERIOR
    # ``mode='inner'`` keeps the band inside each object, so the union of
    # interior+boundary still equals the true foreground mask exactly.
    boundaries = find_boundaries(labels, mode="inner")
    if boundary_width > 1:
        k = 2 * int(boundary_width) - 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        boundaries = cv2.dilate(boundaries.astype(np.uint8), se).astype(bool)
        # Dilation can spill onto background; clamp it back to the foreground so
        # background pixels are never mislabelled as boundary.
        boundaries &= labels > 0
    target[boundaries] = BOUNDARY
    return target


def instances_to_distance(labels: np.ndarray) -> np.ndarray:
    """Instance labels -> per-object normalised distance map, ``float32`` in 0..1.

    Each pixel holds its distance to the edge of *its own* object, divided by
    that object's maximum, so every nucleus peaks at 1.0 at its centre and falls
    to 0 at its rim regardless of size.  Per-object normalisation is what makes
    a single seed threshold work across a field holding both 8 px and 60 px
    nuclei.

    The implementation computes one EDT for the whole image rather than one per
    object, by zeroing each object's own inner boundary first.  Distance then
    cannot flow between two touching nuclei, which is exactly the property a
    naive ``distance_transform_edt(labels > 0)`` loses -- on the merged blob of
    two touching nuclei that version has a single ridge and one maximum.
    """
    labels = np.asarray(labels, dtype=np.int32)
    distance = np.zeros(labels.shape, dtype=np.float32)
    n_objects = int(labels.max())
    if n_objects == 0:
        return distance

    # Erase the inner rim so the transform measures distance-to-own-edge, not
    # distance-to-background.
    interior = (labels > 0) & ~find_boundaries(labels, mode="inner")
    raw = ndi.distance_transform_edt(interior).astype(np.float32)

    # Normalise each object by its own maximum.  Objects thin enough to be
    # entirely rim have max 0 and are left at zero rather than dividing by it.
    index = np.arange(1, n_objects + 1)
    maxima = np.asarray(ndi.maximum(raw, labels, index), dtype=np.float32)
    maxima[~np.isfinite(maxima)] = 0.0
    lookup = np.zeros(n_objects + 1, dtype=np.float32)
    lookup[1:] = np.where(maxima > 0, 1.0 / np.maximum(maxima, 1e-6), 0.0)

    distance = raw * lookup[labels]
    return np.clip(distance, 0.0, 1.0, out=distance)


def semantic_to_foreground(target: np.ndarray) -> np.ndarray:
    """The foreground mask implied by a 3-class map."""
    return np.asarray(target) > BACKGROUND


class NucleiDataset(Dataset):
    """Cached BBBC038 samples -> preprocessed tiles and 3-class targets.

    Preprocessing runs inside the dataset rather than as an offline step so that
    a config change takes effect without rebuilding the cache, and so training
    and inference provably share one code path (:func:`preprocess_array`).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        image_ids: list[str],
        data_cfg: DataConfig | None = None,
        preprocess_cfg: PreprocessConfig | None = None,
        train: bool = True,
        boundary_width: int | None = None,
        seed: int = 0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.image_ids = list(image_ids)
        self.data_cfg = data_cfg or DataConfig()
        self.preprocess_cfg = preprocess_cfg or PreprocessConfig()
        self.train = train
        self.boundary_width = (
            self.data_cfg.boundary_width if boundary_width is None else boundary_width
        )
        self.augment_params = AugmentParams()
        self.seed = seed

        missing = [i for i in self.image_ids if not (self.cache_dir / f"{i}.npz").exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} cached sample(s) missing from {self.cache_dir} "
                f"(first: {missing[0]}); run scripts/prepare_data.py"
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        sample = load_cached(self.cache_dir / f"{self.image_ids[index]}.npz")
        image = to_microscopy_image(sample).nuclei()
        return image, sample.labels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, labels = self._load(index)

        # Preprocess before cropping: illumination correction and percentile
        # normalisation are global operations, and estimating them from a 256 px
        # tile rather than the full field would make them tile-dependent.
        image, _, _ = preprocess_array(image, self.preprocess_cfg)

        size = self.data_cfg.tile_size
        if self.train:
            # Seeded per-epoch-per-item so a run is reproducible while still
            # drawing different crops across epochs.
            rng = np.random.default_rng(
                (self.seed * 1_000_003 + index * 7919 + int(torch.initial_seed()) % 10_000)
                % (2**32)
            )
            image, labels = random_crop(image, labels, size, rng)
            if self.data_cfg.augment:
                image, labels = augment(image, labels, rng, self.augment_params)
        else:
            image, labels = center_crop(image, labels, size)

        target = instances_to_semantic(labels, self.boundary_width)
        # Always produced, even for models without a distance head: it costs one
        # EDT per sample and keeps the batch schema the same across configs.
        distance = instances_to_distance(labels)
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "target": torch.from_numpy(target),
            "distance": torch.from_numpy(distance[None]),
            "instances": torch.from_numpy(labels.astype(np.int32)),
            "image_id": self.image_ids[index],
        }


def build_dataloaders(
    cache_dir: str | Path,
    splits,
    data_cfg: DataConfig,
    preprocess_cfg: PreprocessConfig,
    train_cfg: TrainConfig,
) -> dict[str, DataLoader]:
    """Construct train/val/test loaders from a :class:`~microseg.data.splits.Splits`."""
    # Pinned host memory makes the host->device copy asynchronous.  It is a
    # measurable win on a GPU and pure overhead on CPU, so it follows the device.
    pin = str(getattr(train_cfg, "device", "cpu")).startswith("cuda") or (
        getattr(train_cfg, "device", "cpu") == "auto" and torch.cuda.is_available()
    )

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        ids = splits[split]
        if not ids:
            continue
        is_train = split == "train"
        dataset = NucleiDataset(
            cache_dir,
            ids,
            data_cfg,
            preprocess_cfg,
            train=is_train,
            seed=train_cfg.seed,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size if is_train else max(1, train_cfg.batch_size // 2),
            shuffle=is_train,
            num_workers=train_cfg.num_workers,
            pin_memory=pin,
            persistent_workers=train_cfg.num_workers > 0,
            drop_last=is_train and len(dataset) > train_cfg.batch_size,
            collate_fn=_collate,
        )
    return loaders


def _collate(batch: list[dict]) -> dict:
    """Stack tensors, keep ``image_id`` as a list of strings."""
    out: dict = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        out[key] = values if key == "image_id" else torch.stack(values)
    return out


def sample_to_arrays(sample: BBBC038Sample) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: raw sample -> (nuclei channel, instance labels)."""
    return to_microscopy_image(sample).nuclei(), sample.labels
