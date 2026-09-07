"""Dataset, splits, augmentation and I/O tests."""

from __future__ import annotations

import numpy as np
import pytest

from microseg.data.splits import Splits, load_splits, make_splits
from microseg.data.synthetic import synthetic_field, synthetic_microscopy_image
from microseg.data.transforms import (
    AugmentParams,
    augment,
    center_crop,
    elastic_deform,
    pad_to_min,
    pad_to_multiple,
    random_affine,
    random_crop,
    random_flip_rot,
)


class TestSplits:
    def test_splits_are_disjoint_and_complete(self):
        ids = [f"img{i}" for i in range(100)]
        splits = make_splits(ids, 0.15, 0.15, seed=42)
        assert splits.counts == {"train": 70, "val": 15, "test": 15}
        assert set(splits.train) | set(splits.val) | set(splits.test) == set(ids)
        assert not (set(splits.train) & set(splits.val))
        assert not (set(splits.train) & set(splits.test))
        assert not (set(splits.val) & set(splits.test))

    def test_same_seed_gives_the_same_split(self):
        ids = [f"img{i}" for i in range(50)]
        assert make_splits(ids, seed=7).train == make_splits(ids, seed=7).train

    def test_different_seed_gives_a_different_split(self):
        ids = [f"img{i}" for i in range(50)]
        assert make_splits(ids, seed=1).train != make_splits(ids, seed=2).train

    def test_split_survives_a_disk_round_trip(self, tmp_path):
        splits = make_splits([f"i{i}" for i in range(20)], seed=3)
        reloaded = load_splits(splits.save(tmp_path / "s.json"))
        assert reloaded.train == splits.train and reloaded.seed == 3

    def test_impossible_fractions_raise(self):
        with pytest.raises(ValueError, match="leave room for training"):
            make_splits(["a", "b"], val_fraction=0.6, test_fraction=0.5)

    def test_unknown_split_name_raises(self):
        with pytest.raises(KeyError, match="unknown split"):
            Splits()["validation"]


class TestTransforms:
    def test_flip_and_rotate_keep_image_and_labels_aligned(self, easy_field):
        """Geometry must be applied identically to both, or labels desynchronise."""
        image, labels = easy_field
        rng = np.random.default_rng(0)
        out_image, out_labels = random_flip_rot(image, labels, rng, AugmentParams(1.0, 1.0))
        assert out_image.shape == out_labels.shape
        # Object identity and pixel count survive a dihedral transform exactly.
        assert sorted(np.unique(out_labels)) == sorted(np.unique(labels))
        assert (out_labels > 0).sum() == (labels > 0).sum()

    def test_labels_stay_integral_after_affine(self, easy_field):
        """Nearest-neighbour interpolation: no invented in-between label ids."""
        image, labels = easy_field
        rng = np.random.default_rng(0)
        _, out_labels = random_affine(image, labels, rng, AugmentParams())
        assert set(np.unique(out_labels)) <= set(np.unique(labels))

    def test_labels_stay_integral_after_elastic_deform(self, easy_field):
        image, labels = easy_field
        rng = np.random.default_rng(0)
        _, out_labels = elastic_deform(image, labels, rng, AugmentParams())
        assert set(np.unique(out_labels)) <= set(np.unique(labels))

    def test_augmentation_keeps_images_in_range(self, easy_field):
        image, labels = easy_field
        rng = np.random.default_rng(5)
        for _ in range(10):
            out_image, out_labels = augment(image, labels, rng, AugmentParams())
            assert 0.0 <= out_image.min() and out_image.max() <= 1.0
            assert out_image.shape == out_labels.shape

    def test_augmentation_actually_changes_the_image(self, easy_field):
        image, labels = easy_field
        rng = np.random.default_rng(1)
        params = AugmentParams(flip_prob=1.0, rot90_prob=1.0)
        out, _ = augment(image, labels, rng, params)
        assert not np.array_equal(out, image)

    def test_random_crop_returns_the_requested_size(self, easy_field):
        image, labels = easy_field
        rng = np.random.default_rng(0)
        out_image, out_labels = random_crop(image, labels, 64, rng)
        assert out_image.shape == (64, 64) and out_labels.shape == (64, 64)

    def test_crop_pads_images_smaller_than_the_tile(self):
        image = np.ones((32, 40), np.float32)
        labels = np.ones((32, 40), np.int32)
        out_image, out_labels = center_crop(image, labels, 64)
        assert out_image.shape == (64, 64) and out_labels.shape == (64, 64)

    def test_center_crop_is_deterministic(self, easy_field):
        image, labels = easy_field
        a, _ = center_crop(image, labels, 64)
        b, _ = center_crop(image, labels, 64)
        assert np.array_equal(a, b)

    def test_padding_zero_fills_labels_but_reflects_images(self):
        image = np.ones((10, 10), np.float32)
        labels = np.ones((10, 10), np.int32)
        out_image, out_labels = pad_to_min(image, labels, 20)
        assert out_image[0, 0] == 1.0  # reflected image content
        assert out_labels[0, 0] == 0  # padded region is background, not an object

    @pytest.mark.parametrize("shape", [(100, 70), (256, 256), (33, 17)])
    def test_pad_to_multiple_rounds_up(self, shape):
        padded, original = pad_to_multiple(np.zeros(shape, np.float32), 16)
        assert padded.shape[0] % 16 == 0 and padded.shape[1] % 16 == 0
        assert original == shape


class TestSyntheticData:
    def test_requested_objects_are_generated(self):
        _, labels = synthetic_field(n_objects=20, allow_touching=False, seed=0)
        assert labels.max() == 20

    def test_image_is_normalised_float32(self):
        image, _ = synthetic_field(seed=0)
        assert image.dtype == np.float32
        assert 0.0 <= image.min() and image.max() <= 1.0

    def test_objects_are_brighter_than_background(self):
        image, labels = synthetic_field(noise=0.01, seed=0)
        assert image[labels > 0].mean() > image[labels == 0].mean() + 0.2

    def test_seed_makes_generation_reproducible(self):
        a, la = synthetic_field(seed=11)
        b, lb = synthetic_field(seed=11)
        assert np.array_equal(a, b) and np.array_equal(la, lb)

    def test_non_touching_mode_produces_separate_components(self):
        from skimage.measure import label as cc_label

        _, labels = synthetic_field(n_objects=10, allow_touching=False, seed=2)
        assert cc_label(labels > 0).max() == labels.max()

    def test_multichannel_shares_one_label_map(self):
        image, labels = synthetic_microscopy_image(n_channels=4, seed=0, n_objects=8)
        assert image.n_channels == 4
        assert image.channels[0].role == "nuclei"
        assert all(c.role == "marker" for c in image.channels[1:])
        assert labels.shape == (image.height, image.width)


class TestSemanticDatasetTargets:
    def test_dataset_item_shapes_and_types(self, tmp_path):
        """Exercises NucleiDataset end to end against a cached synthetic sample."""
        import torch

        from microseg.config import DataConfig, PreprocessConfig
        from microseg.data.dataset import NucleiDataset

        image, labels = synthetic_field(shape=(128, 128), n_objects=10, seed=0)
        cache = tmp_path / "cache"
        cache.mkdir()
        np.savez_compressed(
            cache / "s0.npz",
            image=(image * 255).astype(np.uint8)[:, :, None],
            labels=labels,
        )

        dataset = NucleiDataset(
            cache, ["s0"], DataConfig(tile_size=64), PreprocessConfig(), train=True, seed=0
        )
        item = dataset[0]
        assert item["image"].shape == (1, 64, 64)
        assert item["target"].shape == (64, 64)
        assert item["image"].dtype == torch.float32
        assert set(np.unique(item["target"].numpy())) <= {0, 1, 2}

    def test_missing_cache_file_raises_a_clear_error(self, tmp_path):
        from microseg.data.dataset import NucleiDataset

        with pytest.raises(FileNotFoundError, match="cached sample"):
            NucleiDataset(tmp_path, ["nope"])


class TestImageIO:
    def test_png_round_trip_of_a_label_map(self, tmp_path):
        from microseg.io_utils import save_labels

        _, labels = synthetic_field(n_objects=30, seed=0)
        path = save_labels(labels, tmp_path / "labels.png")

        import cv2

        reloaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(reloaded.astype(np.int32), labels)

    def test_rgb_file_is_reduced_to_one_nuclei_channel(self, tmp_path):
        import cv2

        from microseg.io_utils import load_image

        image, _ = synthetic_field(seed=0)
        rgb = np.stack([np.zeros_like(image), np.zeros_like(image), image], axis=-1)
        path = tmp_path / "field.png"
        cv2.imwrite(str(path), (rgb[:, :, ::-1] * 255).astype(np.uint8))

        loaded = load_image(path)
        assert loaded.n_channels == 1
        assert loaded.channels[0].role == "nuclei"

    def test_multichannel_tiff_keeps_its_channels(self, tmp_path):
        import tifffile

        from microseg.io_utils import load_image

        stack = np.random.default_rng(0).random((4, 64, 64)).astype(np.float32)
        path = tmp_path / "stack.tif"
        tifffile.imwrite(str(path), stack, photometric="minisblack")

        loaded = load_image(path, channel_names=["dapi", "actin", "tubulin", "marker"])
        assert loaded.shape == (64, 64, 4)
        assert loaded.channel_names == ["dapi", "actin", "tubulin", "marker"]
        # With no explicit nuclei channel, the first becomes the segmentation one.
        assert loaded.channels[0].role == "nuclei"

    def test_nuclei_channel_can_be_named_explicitly(self, tmp_path):
        import tifffile

        from microseg.io_utils import load_image

        stack = np.random.default_rng(0).random((3, 32, 32)).astype(np.float32)
        path = tmp_path / "s.tif"
        tifffile.imwrite(str(path), stack, photometric="minisblack")

        loaded = load_image(path, channel_names=["actin", "dapi", "marker"], nuclei_channel="dapi")
        assert loaded.index_of("nuclei") == 1

    def test_z_stack_is_projected(self, tmp_path):
        import tifffile

        from microseg.io_utils import load_image

        stack = np.zeros((5, 1, 32, 32), np.float32)
        stack[2, 0, 10:20, 10:20] = 1.0  # signal only on the in-focus slice
        path = tmp_path / "z.tif"
        tifffile.imwrite(str(path), stack)

        loaded = load_image(path)
        assert loaded.shape[:2] == (32, 32)
        assert loaded.data.max() == pytest.approx(1.0)
