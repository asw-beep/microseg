"""Metric tests.

Metrics are the part of a project that must be right even when nothing else is,
because every conclusion downstream rests on them.  These tests pin the values
on cases where the answer can be computed by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from microseg.metrics import (
    average_precision,
    count_metrics,
    dice_score,
    instance_metrics,
    iou_matrix,
    iou_score,
    match_instances,
    per_class_dice,
    semantic_metrics,
)
from microseg.metrics.semantic import aggregate


class TestSemantic:
    def test_identical_masks_score_one(self, easy_field):
        _, labels = easy_field
        mask = labels > 0
        assert dice_score(mask, mask) == 1.0
        assert iou_score(mask, mask) == 1.0

    def test_disjoint_masks_score_zero(self):
        a = np.zeros((16, 16), bool)
        b = np.zeros((16, 16), bool)
        a[:8] = True
        b[8:] = True
        assert dice_score(a, b) == 0.0

    def test_half_overlap_has_known_value(self):
        """Prediction twice the size of a contained truth: Dice 2/3, IoU 1/2."""
        truth = np.zeros((10, 10), bool)
        truth[:5] = True
        pred = np.zeros((10, 10), bool)
        pred[:10] = True
        assert dice_score(pred, truth) == pytest.approx(2 / 3)
        assert iou_score(pred, truth) == pytest.approx(0.5)

    def test_two_empty_masks_score_one(self):
        """Correctly predicting an empty field is a success, not undefined."""
        empty = np.zeros((16, 16), bool)
        assert dice_score(empty, empty) == 1.0
        assert semantic_metrics(empty, empty)["precision"] == 1.0

    def test_dice_never_below_iou(self, easy_field, rng):
        _, labels = easy_field
        truth = labels > 0
        noisy = truth ^ (rng.random(truth.shape) < 0.05)
        assert dice_score(noisy, truth) >= iou_score(noisy, truth)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            dice_score(np.zeros((4, 4), bool), np.zeros((5, 5), bool))

    def test_per_class_dice_names_the_classes(self, easy_field):
        _, labels = easy_field
        from microseg.data.dataset import instances_to_semantic

        target = instances_to_semantic(labels)
        scores = per_class_dice(target, target)
        assert set(scores) == {"dice_background", "dice_interior", "dice_boundary"}
        assert all(v == 1.0 for v in scores.values())

    def test_aggregate_averages_numeric_fields(self):
        out = aggregate([{"dice": 0.8, "iou": 0.6}, {"dice": 0.6, "iou": 0.4}])
        assert out == {"dice": pytest.approx(0.7), "iou": pytest.approx(0.5)}


class TestIoUMatrix:
    def test_perfect_prediction_is_an_identity_of_ones(self, easy_field):
        _, labels = easy_field
        matrix = iou_matrix(labels, labels)
        assert matrix.shape == (labels.max(), labels.max())
        assert np.allclose(np.diag(matrix), 1.0)

    def test_empty_prediction_gives_an_empty_axis(self, easy_field):
        _, labels = easy_field
        assert iou_matrix(np.zeros_like(labels), labels).shape == (labels.max(), 0)

    def test_values_match_a_hand_computation(self):
        truth = np.zeros((10, 10), np.int32)
        truth[0:4, 0:4] = 1  # 16 px
        pred = np.zeros((10, 10), np.int32)
        pred[0:4, 0:2] = 1  # 8 px, fully inside -> IoU 8/16
        assert iou_matrix(pred, truth)[0, 0] == pytest.approx(0.5)


class TestMatching:
    def test_one_to_one_match_on_a_perfect_prediction(self, easy_field):
        _, labels = easy_field
        pairs, missed, extra = match_instances(labels, labels, 0.5)
        assert len(pairs) == labels.max()
        assert missed.size == 0 and extra.size == 0

    def test_merged_objects_produce_a_miss(self):
        """Two truths, one prediction covering both: one match, one false negative.

        This is the failure that pixel Dice cannot see, so it is the single most
        important case in this file.
        """
        truth = np.zeros((10, 20), np.int32)
        truth[:, :10] = 1
        truth[:, 10:] = 2
        pred = np.ones((10, 20), np.int32)  # one big object

        pairs, missed, extra = match_instances(pred, truth, 0.5)
        assert len(pairs) == 1 and len(missed) == 1 and len(extra) == 0
        assert dice_score(pred > 0, truth > 0) == 1.0  # pixel metric sees nothing wrong

    def test_split_object_produces_a_false_positive(self):
        """One truth predicted as two halves: at most one half can match.

        Each half has IoU 100/200 = 0.5 with the truth, so at threshold 0.5 one
        is matched and the other is necessarily a false positive.  Raising the
        threshold rejects both, and the object is then also a false negative.
        """
        truth = np.ones((10, 20), np.int32)
        pred = np.zeros((10, 20), np.int32)
        pred[:, :10] = 1
        pred[:, 10:] = 2

        pairs, missed, extra = match_instances(pred, truth, 0.5)
        assert len(pairs) == 1 and len(missed) == 0 and len(extra) == 1

        pairs, missed, extra = match_instances(pred, truth, 0.6)
        assert len(pairs) == 0 and len(missed) == 1 and len(extra) == 2

    def test_matching_is_optimal_not_greedy(self):
        """A marginal pair must not steal a partner from a better one."""
        truth = np.zeros((10, 30), np.int32)
        truth[:, 0:10] = 1
        truth[:, 10:20] = 2
        pred = np.zeros((10, 30), np.int32)
        pred[:, 0:10] = 1
        pred[:, 10:20] = 2
        pairs, _, _ = match_instances(pred, truth, 0.5)
        assert sorted(map(tuple, pairs)) == [(1, 1), (2, 2)]

    def test_higher_threshold_rejects_loose_matches(self):
        truth = np.zeros((10, 10), np.int32)
        truth[0:8, :] = 1
        pred = np.zeros((10, 10), np.int32)
        pred[0:5, :] = 1  # IoU = 5/8 = 0.625
        assert len(match_instances(pred, truth, 0.5)[0]) == 1
        assert len(match_instances(pred, truth, 0.7)[0]) == 0


class TestAveragePrecision:
    def test_perfect_prediction_scores_one(self, easy_field):
        _, labels = easy_field
        assert average_precision(labels, labels)["ap_mean"] == pytest.approx(1.0)

    def test_empty_versus_empty_scores_one(self):
        empty = np.zeros((16, 16), np.int32)
        assert average_precision(empty, empty)["ap_mean"] == 1.0

    def test_missing_everything_scores_zero(self, easy_field):
        _, labels = easy_field
        assert average_precision(np.zeros_like(labels), labels)["ap_mean"] == 0.0

    def test_reports_every_threshold(self, easy_field):
        _, labels = easy_field
        out = average_precision(labels, labels)
        assert "ap@0.50" in out and "ap@0.95" in out and len(out) == 11


class TestCounts:
    def test_exact_count_has_no_error(self, easy_field):
        _, labels = easy_field
        out = count_metrics(labels, labels)
        assert out["count_error"] == 0 and out["count_rel_error"] == 0

    def test_relative_error_is_signed_correctly(self):
        truth = np.zeros((10, 20), np.int32)
        truth[:, :10] = 1
        truth[:, 10:] = 2
        pred = np.ones((10, 20), np.int32)
        out = count_metrics(pred, truth)
        assert out["count_error"] == -1 and out["count_rel_error"] == pytest.approx(0.5)


class TestInstanceReport:
    def test_perfect_prediction_scores_one_everywhere(self, easy_field):
        _, labels = easy_field
        out = instance_metrics(labels, labels)
        assert out["precision"] == 1.0 and out["recall"] == 1.0 and out["f1"] == 1.0
        assert out["mean_matched_iou"] == pytest.approx(1.0)

    def test_report_contains_the_expected_keys(self, easy_field):
        _, labels = easy_field
        out = instance_metrics(labels, labels)
        for key in ("precision", "recall", "f1", "ap_mean", "count_pred", "mean_matched_iou"):
            assert key in out
