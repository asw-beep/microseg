"""Quality-control layer tests.

QC decides whether a result is handed onward or held for review, so the tests
here are about the *decisions*: a good segmentation must pass, and each specific
failure mode must be caught by the signal that is supposed to catch it.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from microseg.analysis import extract_features
from microseg.channels import ChannelSpec, MicroscopyImage
from microseg.config import AnalysisConfig, ExperimentConfig
from microseg.pipeline import SegmentationPipeline
from microseg.qc import (
    QCConfig,
    QCReport,
    assess,
    compute_signals,
    predictive_confidence,
)


def _features_for(labels: np.ndarray, image: np.ndarray):
    micro = MicroscopyImage.from_array(image, [ChannelSpec("nuclei", "nuclei")], name="t")
    return extract_features(labels, micro, AnalysisConfig(compute_texture=False))


@pytest.fixture
def good_result(easy_field):
    """A clean segmentation of well-separated round objects."""
    image, labels = easy_field
    return labels, _features_for(labels, image), image


class TestSignals:
    def test_signals_describe_a_clean_field(self, good_result):
        labels, features, _ = good_result
        signals = compute_signals(labels, features)
        assert signals["n_objects"] == labels.max()
        assert 0.0 < signals["foreground_fraction"] < 0.45
        assert signals["median_solidity"] > 0.9
        assert signals["median_circularity"] > 0.7

    def test_empty_result_has_no_objects(self):
        signals = compute_signals(np.zeros((64, 64), np.int32))
        assert signals["n_objects"] == 0
        assert signals["foreground_fraction"] == 0.0

    def test_predictive_confidence_is_high_for_confident_maps(self):
        probs = np.zeros((3, 16, 16), np.float32)
        probs[1] = 0.95
        probs[0] = 0.05
        labels = np.ones((16, 16), np.int32)
        assert predictive_confidence(probs, labels) == pytest.approx(0.95, abs=1e-5)

    def test_predictive_confidence_is_low_for_uniform_maps(self):
        probs = np.full((3, 16, 16), 1 / 3, np.float32)
        assert predictive_confidence(probs) == pytest.approx(1 / 3, abs=1e-5)


class TestVerdicts:
    def test_clean_segmentation_passes(self, good_result):
        labels, features, _ = good_result
        report = assess(labels, features)
        assert report.passed
        assert report.confidence > 0.9
        assert report.flags == []

    def test_empty_result_is_flagged_with_a_clear_reason(self):
        report = assess(np.zeros((64, 64), np.int32))
        assert not report.passed
        assert report.confidence == 0.0
        assert report.flags == ["no objects detected"]

    def test_runaway_foreground_is_flagged(self):
        """The pathological case: most of the image labelled as one nucleus."""
        labels = np.ones((64, 64), np.int32)
        report = assess(labels, _features_for(labels, np.ones((64, 64), np.float32)))
        assert not report.passed
        assert any("foreground" in f for f in report.flags)

    def test_merged_objects_are_flagged_by_solidity(self):
        """Two nuclei segmented as one object are concave -- solidity catches it.

        The pairs are placed just touching (centres 44 apart, radius 22) rather
        than overlapping: that is what an under-segmented pair actually looks
        like, and it leaves the pinched waist that depresses solidity. Heavily
        overlapping circles merge into a convex blob and are correctly *not*
        flagged, which is why the geometry here is specific.
        """
        labels = np.zeros((140, 260), np.int32)
        for label, cx in enumerate((45, 165), start=1):
            cv2.circle(labels, (cx, 70), 22, label, -1)
            cv2.circle(labels, (cx + 44, 70), 22, label, -1)

        report = assess(labels, _features_for(labels, (labels > 0).astype(np.float32)))
        assert not report.passed
        assert any("solidity" in f for f in report.flags)

    def test_fragmentation_is_flagged_by_size_dispersion(self):
        """Whole nuclei mixed with slivers gives a high area CV."""
        labels = np.zeros((200, 200), np.int32)
        next_id = 1
        for cx in (40, 100, 160):  # three normal objects
            cv2.circle(labels, (cx, 50), 20, next_id, -1)
            next_id += 1
        for i in range(12):  # a scatter of fragments
            cv2.circle(labels, (20 + 15 * i, 150), 2, next_id, -1)
            next_id += 1

        report = assess(labels, _features_for(labels, (labels > 0).astype(np.float32)))
        assert not report.passed
        assert any("fragmentation" in f or "vary implausibly" in f for f in report.flags)

    def test_uncertain_model_is_flagged(self, good_result):
        """Plausible geometry but a hedging model is still worth review."""
        labels, features, _ = good_result
        probs = np.full((3, *labels.shape), 1 / 3, np.float32)
        report = assess(labels, features, probs)
        assert any("uncertain" in f for f in report.flags)

    def test_confident_model_does_not_trip_the_uncertainty_flag(self, good_result):
        labels, features, _ = good_result
        probs = np.zeros((3, *labels.shape), np.float32)
        probs[1] = 0.9
        probs[0] = 0.1
        report = assess(labels, features, probs)
        assert not any("uncertain" in f for f in report.flags)

    def test_thresholds_are_configurable(self, good_result):
        """A caller can tighten QC without touching code."""
        labels, features, _ = good_result
        strict = QCConfig(min_median_circularity=0.999, circularity_width=0.01)
        assert not assess(labels, features, cfg=strict).passed
        assert assess(labels, features).passed


class TestReport:
    def test_report_serialises_for_a_results_table(self, good_result):
        labels, features, _ = good_result
        row = assess(labels, features).as_dict()
        assert row["qc_passed"] is True
        assert 0.0 <= row["qc_confidence"] <= 1.0
        assert isinstance(row["qc_flags"], str)
        assert row["qc_n_flags"] == 0

    def test_flags_are_human_readable(self):
        report = assess(np.ones((64, 64), np.int32))
        assert all(isinstance(f, str) and len(f) > 10 for f in report.flags)


class TestPenaltyRamp:
    def test_penalty_is_zero_inside_the_band(self):
        from microseg.qc import _penalty_above, _penalty_below

        assert _penalty_above(0.5, 1.0, 0.5) == 0.0
        assert _penalty_below(1.5, 1.0, 0.5) == 0.0

    def test_penalty_saturates_at_one_width_out(self):
        from microseg.qc import _penalty_above, _penalty_below

        assert _penalty_above(1.5, 1.0, 0.5) == pytest.approx(1.0)
        assert _penalty_below(0.5, 1.0, 0.5) == pytest.approx(1.0)

    def test_penalty_ramps_linearly(self):
        from microseg.qc import _penalty_above

        assert _penalty_above(1.25, 1.0, 0.5) == pytest.approx(0.5)

    def test_width_is_in_signal_units_not_relative(self):
        """A band near 1.0 must be usable -- this is why width is explicit.

        Solidity separates good from bad results over a range of about 0.04.
        A relative ramp scaled to the limit would make that check inert.
        """
        from microseg.qc import _penalty_below

        assert _penalty_below(0.845, 0.88, 0.04) > 0.8
        assert _penalty_below(0.878, 0.88, 0.04) < 0.1


class TestPipelineIntegration:
    def test_pipeline_attaches_a_qc_report(self, easy_field):
        image, _ = easy_field
        micro = MicroscopyImage.from_array(
            image, [ChannelSpec("nuclei", "nuclei")], name="f"
        )
        cfg = ExperimentConfig()
        cfg.analysis.compute_texture = False
        result = SegmentationPipeline(cfg, backend="classical").run(micro)

        assert isinstance(result.qc, QCReport)
        assert result.trustworthy
        assert "qc_confidence" in result.summary
        assert result.summary["qc_passed"] is True

    def test_qc_can_be_disabled(self, easy_field):
        image, _ = easy_field
        micro = MicroscopyImage.from_array(
            image, [ChannelSpec("nuclei", "nuclei")], name="f"
        )
        cfg = ExperimentConfig()
        cfg.analysis.compute_texture = False
        result = SegmentationPipeline(cfg, backend="classical", run_qc=False).run(micro)

        assert result.qc is None
        assert result.trustworthy  # absence of QC is not a failure
        assert "qc_confidence" not in result.summary

    def test_blank_image_is_flagged_by_the_pipeline(self):
        cfg = ExperimentConfig()
        cfg.analysis.compute_texture = False
        blank = MicroscopyImage.from_array(np.zeros((64, 64), np.float32), name="blank")
        result = SegmentationPipeline(cfg, backend="classical").run(blank)

        assert not result.trustworthy
        assert result.qc.flags == ["no objects detected"]
