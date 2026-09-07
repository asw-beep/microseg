"""Classical segmentation and semantic -> instance conversion.

The tests that matter here are about *instances*, not pixels: the whole point of
the watershed stage is that touching nuclei come out as separate objects, and
that property is easy to break without any pixel metric noticing.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.measure import label as cc_label

from microseg.classical import (
    ClassicalParams,
    classical_instances,
    morphological_cleanup,
    threshold_image,
    watershed_split,
)
from microseg.classical.segment import (
    drop_small_labels,
    fill_small_holes,
    relabel_sequential,
    remove_small_regions,
)
from microseg.config import PostprocessConfig
from microseg.data.dataset import BOUNDARY, INTERIOR, instances_to_semantic
from microseg.instance import (
    instances_from_binary,
    probabilities_to_instances,
    remove_border_objects,
)
from microseg.preprocessing import preprocess_array


class TestThresholding:
    @pytest.mark.parametrize("method", ["otsu", "li", "triangle", "adaptive", "fixed"])
    def test_every_method_finds_the_objects(self, easy_field, method):
        image, labels = easy_field
        pre, _, _ = preprocess_array(image)
        mask = threshold_image(pre, method, ClassicalParams(fixed_threshold=0.3))
        # At least half of the true foreground should be recovered.
        assert mask[labels > 0].mean() > 0.5, f"{method} missed most of the foreground"

    def test_constant_image_returns_empty_mask(self):
        """skimage raises on a flat histogram; we must degrade gracefully."""
        mask = threshold_image(np.full((32, 32), 0.4, np.float32), "otsu")
        assert not mask.any()

    def test_unknown_method_raises(self, easy_field):
        with pytest.raises(ValueError, match="unknown threshold method"):
            threshold_image(easy_field[0], "magic")


class TestMorphology:
    def test_small_objects_are_removed(self):
        mask = np.zeros((64, 64), bool)
        mask[10:30, 10:30] = True  # 400 px, kept
        mask[50, 50] = True  # 1 px, dropped
        out = remove_small_regions(mask, 15)
        assert out[20, 20] and not out[50, 50]

    def test_enclosed_holes_are_filled(self):
        mask = np.zeros((64, 64), bool)
        mask[10:40, 10:40] = True
        mask[20:23, 20:23] = False  # a 9 px nucleolus-like hole
        assert fill_small_holes(mask, 30)[21, 21]

    def test_background_is_not_filled(self):
        """A background region reaching the border is background, however small."""
        mask = np.zeros((16, 16), bool)
        mask[:, 8:] = True
        assert not fill_small_holes(mask, 10_000)[0, 0]

    def test_cleanup_separates_thinly_bridged_objects(self):
        """Opening must break a one-pixel bridge between two blobs."""
        mask = np.zeros((64, 64), bool)
        mask[20:30, 10:25] = True
        mask[20:30, 35:50] = True
        mask[24:26, 25:35] = True  # a 2 px-wide bridge

        assert cc_label(mask).max() == 1
        cleaned = morphological_cleanup(mask, ClassicalParams(opening_radius=2, closing_radius=0))
        assert cc_label(cleaned).max() == 2


class TestWatershed:
    def test_touching_objects_are_split(self):
        """The core claim of the classical pipeline."""
        import cv2

        mask = np.zeros((80, 140), np.uint8)
        cv2.circle(mask, (50, 40), 24, 1, -1)
        cv2.circle(mask, (90, 40), 24, 1, -1)  # overlapping -> one component
        binary = mask.astype(bool)
        assert cc_label(binary).max() == 1

        labels = watershed_split(binary, ClassicalParams(peak_min_distance=10))
        assert labels.max() == 2, f"expected 2 instances, got {labels.max()}"

    def test_empty_mask_returns_empty_labels(self):
        assert watershed_split(np.zeros((32, 32), bool)).max() == 0

    def test_full_pipeline_counts_separated_objects(self, easy_field):
        image, labels = easy_field
        pre, _, _ = preprocess_array(image)
        found = classical_instances(pre)
        assert abs(int(found.max()) - int(labels.max())) <= 2

    def test_watershed_beats_connected_components_when_crowded(self, crowded_field):
        """The reason the watershed stage exists at all."""
        image, labels = crowded_field
        pre, _, _ = preprocess_array(image)

        with_watershed = classical_instances(pre, ClassicalParams(use_watershed=True))
        without = classical_instances(pre, ClassicalParams(use_watershed=False))
        truth = int(labels.max())
        assert abs(int(with_watershed.max()) - truth) < abs(int(without.max()) - truth)


class TestLabelUtilities:
    def test_relabel_makes_ids_contiguous(self):
        labels = np.array([[0, 3, 3], [7, 7, 0], [0, 0, 9]], np.int32)
        out = relabel_sequential(labels)
        assert sorted(np.unique(out)) == [0, 1, 2, 3]

    def test_drop_small_labels_keeps_ids_of_survivors(self):
        labels = np.zeros((16, 16), np.int32)
        labels[0:4, 0:4] = 1  # 16 px
        labels[10, 10] = 2  # 1 px
        out = drop_small_labels(labels, 10)
        assert (out == 1).sum() == 16 and (out == 2).sum() == 0

    def test_border_objects_are_removed(self):
        labels = np.zeros((32, 32), np.int32)
        labels[0:5, 0:5] = 1  # touches the border
        labels[15:20, 15:20] = 2  # interior
        out = remove_border_objects(labels)
        assert out.max() == 1 and out[17, 17] > 0 and out[1, 1] == 0


class TestSemanticTargets:
    def test_three_classes_are_produced(self, easy_field):
        _, labels = easy_field
        target = instances_to_semantic(labels)
        assert set(np.unique(target)) == {0, INTERIOR, BOUNDARY}

    def test_foreground_is_preserved_exactly(self, easy_field):
        """interior u boundary must equal the true mask, or areas come out wrong."""
        _, labels = easy_field
        target = instances_to_semantic(labels, boundary_width=3)
        assert np.array_equal(target > 0, labels > 0)

    def test_wider_boundary_uses_more_pixels(self, crowded_field):
        _, labels = crowded_field
        narrow = (instances_to_semantic(labels, 1) == BOUNDARY).sum()
        wide = (instances_to_semantic(labels, 3) == BOUNDARY).sum()
        assert wide > narrow

    def test_empty_labels_give_all_background(self):
        assert instances_to_semantic(np.zeros((16, 16), np.int32)).max() == 0


class TestProbabilitiesToInstances:
    def _probs_from_target(self, target, confidence=0.9):
        """Turn a ground-truth class map into a confident probability map."""
        probs = np.stack([(target == c) for c in range(3)]).astype(np.float32)
        return probs * confidence + (1 - confidence) / 3.0

    def test_perfect_probabilities_recover_every_instance(self, crowded_field):
        """The round trip that validates the whole semantic->instance design."""
        _, labels = crowded_field
        probs = self._probs_from_target(instances_to_semantic(labels, boundary_width=2))
        recovered = probabilities_to_instances(probs, PostprocessConfig())
        assert recovered.max() == labels.max()

    def test_recovered_masks_include_the_boundary_band(self, easy_field):
        """Watershed must reassign boundary pixels, or every area is biased low."""
        _, labels = easy_field
        probs = self._probs_from_target(instances_to_semantic(labels, boundary_width=2))
        recovered = probabilities_to_instances(probs, PostprocessConfig())
        assert (recovered > 0).sum() > 0.95 * (labels > 0).sum()

    def test_empty_prediction_returns_no_objects(self):
        probs = np.zeros((3, 32, 32), np.float32)
        probs[0] = 1.0
        assert probabilities_to_instances(probs).max() == 0

    def test_two_channel_input_falls_back_to_distance_seeding(self, easy_field):
        _, labels = easy_field
        probs = np.stack([(labels == 0), (labels > 0)]).astype(np.float32)
        assert probabilities_to_instances(probs).max() > 0

    def test_malformed_input_raises(self):
        with pytest.raises(ValueError, match=r"expected a \(C, H, W\)"):
            probabilities_to_instances(np.zeros((32, 32), np.float32))

    def test_binary_fallback_splits_touching_objects(self):
        import cv2

        mask = np.zeros((80, 140), np.uint8)
        cv2.circle(mask, (50, 40), 24, 1, -1)
        cv2.circle(mask, (90, 40), 24, 1, -1)
        labels = instances_from_binary(mask.astype(bool), PostprocessConfig(peak_min_distance=10))
        assert labels.max() == 2


class TestDistanceTargets:
    """`instances_to_distance` -- the training target the distance head learns."""

    def test_every_object_peaks_at_one_and_background_stays_zero(self, easy_field):
        from microseg.data.dataset import instances_to_distance

        _, labels = easy_field
        distance = instances_to_distance(labels)
        assert distance.shape == labels.shape and distance.dtype == np.float32
        assert distance.min() >= 0.0 and distance.max() <= 1.0
        assert distance[labels == 0].max() == 0.0
        # Per-object normalisation: every nucleus reaches its own maximum of 1.
        for object_id in np.unique(labels[labels > 0]):
            assert np.isclose(distance[labels == object_id].max(), 1.0, atol=1e-5)

    def test_normalisation_is_per_object_not_global(self):
        """A small nucleus must peak as high as a large one, or one seed
        threshold cannot serve both."""
        import cv2

        from microseg.data.dataset import instances_to_distance

        labels = np.zeros((80, 160), np.int32)
        cv2.circle(labels, (40, 40), 25, 1, -1)
        cv2.circle(labels, (120, 40), 7, 2, -1)
        distance = instances_to_distance(labels)
        assert np.isclose(distance[labels == 1].max(), 1.0, atol=1e-5)
        assert np.isclose(distance[labels == 2].max(), 1.0, atol=1e-5)

    def test_touching_objects_get_one_maximum_each(self):
        """The property a plain `distance_transform_edt(labels > 0)` loses.

        Two touching nuclei form one connected blob; measuring distance to the
        blob's outside gives a single ridge, and therefore a single seed.
        Measuring distance to each object's own edge gives two.
        """
        import cv2
        from skimage.measure import label as cc

        from microseg.data.dataset import instances_to_distance

        labels = np.zeros((80, 140), np.int32)
        cv2.circle(labels, (52, 40), 25, 1, -1)
        cv2.circle(labels, (88, 40), 25, 2, -1)
        assert cc(labels > 0).max() == 1, "the fixture must be one connected blob"

        distance = instances_to_distance(labels)
        assert cc(distance > 0.5).max() == 2

    def test_empty_label_map_gives_an_all_zero_target(self):
        from microseg.data.dataset import instances_to_distance

        assert instances_to_distance(np.zeros((32, 32), np.int32)).max() == 0.0

    def test_dataset_item_carries_a_matching_distance_target(self, tmp_path):
        """The batch schema is the same with or without a distance head."""
        import torch

        from microseg.config import DataConfig, PreprocessConfig
        from microseg.data.dataset import NucleiDataset
        from microseg.data.synthetic import synthetic_field

        image, labels = synthetic_field(shape=(128, 128), n_objects=10, seed=0)
        cache = tmp_path / "cache"
        cache.mkdir()
        np.savez_compressed(
            cache / "s0.npz",
            image=(image * 255).astype(np.uint8)[:, :, None],
            labels=labels,
        )
        item = NucleiDataset(
            cache, ["s0"], DataConfig(tile_size=64), PreprocessConfig(), train=False, seed=0
        )[0]

        assert item["distance"].shape == (1, 64, 64)
        assert item["distance"].dtype == torch.float32
        distance = item["distance"].numpy()[0]
        assert distance.min() >= 0.0 and distance.max() <= 1.0
        # Distance lives strictly inside the foreground the class target marks.
        assert distance[item["target"].numpy() == 0].max(initial=0.0) == 0.0


class TestDistanceSeeding:
    """Seeding the watershed from a predicted distance map."""

    @staticmethod
    def _touching_pair():
        """Two overlapping discs, and the perfect predictions for them."""
        import cv2

        from microseg.data.dataset import instances_to_distance

        labels = np.zeros((80, 140), np.int32)
        cv2.circle(labels, (52, 40), 25, 1, -1)
        cv2.circle(labels, (88, 40), 25, 2, -1)
        foreground = labels > 0
        probs = np.zeros((3, *labels.shape), np.float32)
        probs[0][~foreground] = 1.0
        probs[INTERIOR][foreground] = 1.0
        return labels, probs, instances_to_distance(labels)

    def test_distance_seeds_split_a_blob_the_class_map_cannot(self):
        """The reason the head exists: overlapping nuclei leave no boundary band,
        so a class map that says only 'foreground' still separates."""
        labels, probs, distance = self._touching_pair()
        assert probabilities_to_instances(probs).max() == 1
        assert probabilities_to_instances(probs, PostprocessConfig(), distance).max() == 2

    def test_recovered_instances_match_the_truth(self):
        labels, probs, distance = self._touching_pair()
        recovered = probabilities_to_instances(probs, PostprocessConfig(), distance)
        assert recovered.max() == 2
        # Same footprint as the truth, up to the watershed's choice of ids.
        assert ((recovered > 0) == (labels > 0)).mean() > 0.99

    def test_the_flag_turns_distance_seeding_off(self):
        """Keeping both routes available on one checkpoint is what makes them
        comparable."""
        _, probs, distance = self._touching_pair()
        cfg = PostprocessConfig(use_distance_seeds=False)
        assert probabilities_to_instances(probs, cfg, distance).max() == 1

    def test_a_leading_singleton_axis_is_accepted(self):
        """Model output arrives as (1, H, W) often enough to be worth handling."""
        _, probs, distance = self._touching_pair()
        assert probabilities_to_instances(probs, PostprocessConfig(), distance[None]).max() == 2

    def test_mismatched_distance_shape_raises(self):
        _, probs, _ = self._touching_pair()
        with pytest.raises(ValueError, match="does not match probabilities"):
            probabilities_to_instances(probs, PostprocessConfig(), np.zeros((16, 16), np.float32))

    def test_seeding_falls_back_when_no_seed_survives(self):
        """An unconfident distance map must not empty the field."""
        _, probs, distance = self._touching_pair()
        recovered = probabilities_to_instances(probs, PostprocessConfig(), distance * 0.1)
        assert recovered.max() > 0

    def test_empty_foreground_returns_no_objects(self):
        probs = np.zeros((3, 32, 32), np.float32)
        probs[0] = 1.0
        distance = np.ones((32, 32), np.float32)
        assert probabilities_to_instances(probs, PostprocessConfig(), distance).max() == 0

    def test_a_confident_distance_outside_the_mask_cannot_seed(self):
        """Foreground comes from the class head, always: the distance loss is
        masked, so its background values are unconstrained noise."""
        labels, probs, distance = self._touching_pair()
        distance = distance.copy()
        distance[5:15, 120:135] = 1.0  # a confident blob out in the background
        assert probabilities_to_instances(probs, PostprocessConfig(), distance).max() == 2
