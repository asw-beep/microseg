"""End-to-end pipeline integration tests.

These are the tests that would catch a wiring mistake between two stages that
are individually correct -- for example intensity features accidentally being
measured on the CLAHE-enhanced image instead of the raw one.
"""

from __future__ import annotations

import numpy as np
import pytest

from microseg.channels import ChannelSpec, MicroscopyImage
from microseg.config import AnalysisConfig, ExperimentConfig, ModelConfig
from microseg.models import build_model
from microseg.pipeline import SegmentationPipeline, results_to_tables


@pytest.fixture
def field_image(easy_field):
    image, labels = easy_field
    micro = MicroscopyImage.from_array(
        image, [ChannelSpec("nuclei", "nuclei")], name="field0"
    )
    return micro, labels


class TestClassicalPipeline:
    def test_produces_objects_and_a_feature_row_each(self, field_image):
        micro, labels = field_image
        cfg = ExperimentConfig()
        cfg.analysis.compute_texture = False
        result = SegmentationPipeline(cfg, backend="classical").run(micro)

        assert result.n_objects > 0
        assert len(result.features) == result.n_objects
        assert result.summary["n_objects"] == result.n_objects

    def test_object_count_is_close_to_truth(self, field_image):
        micro, labels = field_image
        cfg = ExperimentConfig()
        cfg.analysis.compute_texture = False
        result = SegmentationPipeline(cfg, backend="classical").run(micro)
        assert abs(result.n_objects - int(labels.max())) <= 2

    def test_every_stage_is_timed(self, field_image):
        micro, _ = field_image
        cfg = ExperimentConfig()
        cfg.analysis.compute_texture = False
        result = SegmentationPipeline(cfg, backend="classical").run(micro)
        for key in ("preprocess_s", "segment_s", "analysis_s", "total_s"):
            assert result.timings[key] >= 0
        assert result.timings["total_s"] == pytest.approx(
            result.timings["preprocess_s"]
            + result.timings["segment_s"]
            + result.timings["analysis_s"],
            rel=1e-6,
        )

    def test_stages_are_captured_on_request(self, field_image):
        micro, _ = field_image
        cfg = ExperimentConfig()
        cfg.analysis.compute_texture = False
        result = SegmentationPipeline(cfg, backend="classical").run(micro, collect_stages=True)
        assert "raw" in result.stages and "normalized" in result.stages

    def test_intensity_is_measured_on_the_raw_image(self, field_image):
        """The correctness property that keeps measurements quantitative.

        Preprocessing (CLAHE, percentile normalisation) deliberately destroys
        the mapping from pixel value to stain concentration.  Features must
        therefore come from the raw input, and this test detects a regression
        that swapped the two.
        """
        micro, _ = field_image
        cfg = ExperimentConfig()
        cfg.analysis.compute_texture = False
        pipeline = SegmentationPipeline(cfg, backend="classical")
        result = pipeline.run(micro)

        raw_values = micro.nuclei()[result.instances > 0]
        measured = result.features["mean_intensity"].to_numpy()
        # Measured means must sit inside the raw intensity range, not the
        # renormalised [0, 1] range of the preprocessed image.
        assert measured.min() >= raw_values.min() - 1e-6
        assert measured.max() <= raw_values.max() + 1e-6

    def test_scaling_the_input_scales_the_measured_intensity(self, field_image):
        """A stronger version of the same property, independent of the range."""
        micro, _ = field_image
        cfg = ExperimentConfig()
        cfg.analysis.compute_texture = False
        pipeline = SegmentationPipeline(cfg, backend="classical")

        bright = pipeline.run(micro)
        dim = pipeline.run(micro.with_data(micro.data * 0.5))
        assert dim.features["mean_intensity"].mean() < 0.75 * bright.features[
            "mean_intensity"
        ].mean()

    def test_empty_image_produces_no_objects(self):
        cfg = ExperimentConfig()
        cfg.analysis.compute_texture = False
        blank = MicroscopyImage.from_array(np.zeros((64, 64), np.float32), name="blank")
        result = SegmentationPipeline(cfg, backend="classical").run(blank)
        assert result.n_objects == 0 and result.features.empty
        assert result.summary["n_objects"] == 0


class TestMultiChannel:
    def test_segments_one_channel_and_measures_all(self, multichannel_image):
        """The multiplexed-assay workflow, end to end."""
        image, _ = multichannel_image
        cfg = ExperimentConfig()
        cfg.analysis.compute_texture = False
        result = SegmentationPipeline(cfg, backend="classical").run(image)

        assert result.n_objects > 0
        for name in ("nuclei", "marker1", "marker2"):
            assert f"{name}_mean_intensity" in result.features.columns

    def test_measure_channels_can_be_restricted(self, multichannel_image):
        image, _ = multichannel_image
        cfg = ExperimentConfig()
        cfg.analysis.compute_texture = False
        result = SegmentationPipeline(cfg, backend="classical").run(
            image, measure_channels=["nuclei"]
        )
        assert "marker1_mean_intensity" not in result.features.columns


class TestUNetPipeline:
    def test_untrained_model_runs_end_to_end(self, field_image):
        """Shape and plumbing check -- accuracy is not the point here."""
        micro, _ = field_image
        cfg = ExperimentConfig(model=ModelConfig(base_filters=4, depth=2))
        cfg.analysis.compute_texture = False
        model = build_model(cfg.model)

        result = SegmentationPipeline(cfg, backend="unet", model=model).run(micro)
        assert result.probabilities is not None
        assert result.probabilities.shape == (3, micro.height, micro.width)
        assert np.allclose(result.probabilities.sum(axis=0), 1.0, atol=1e-5)

    def test_unet_backend_requires_a_model(self):
        with pytest.raises(ValueError, match="requires a trained model"):
            SegmentationPipeline(ExperimentConfig(), backend="unet")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown backend"):
            SegmentationPipeline(ExperimentConfig(), backend="cellpose")


class TestBatchTables:
    def test_results_concatenate_into_two_tables(self, field_image):
        micro, _ = field_image
        cfg = ExperimentConfig()
        cfg.analysis.compute_texture = False
        pipeline = SegmentationPipeline(cfg, backend="classical")

        results = pipeline.run_batch([micro, micro.with_data(micro.data)])
        per_object, per_image = results_to_tables(results)
        assert len(per_image) == 2
        assert len(per_object) == sum(r.n_objects for r in results)
        assert "image_id" in per_object.columns


class TestAnalysisConfigEffects:
    def test_texture_columns_appear_only_when_enabled(self, field_image):
        micro, _ = field_image
        with_texture = ExperimentConfig(analysis=AnalysisConfig(compute_texture=True))
        without = ExperimentConfig(analysis=AnalysisConfig(compute_texture=False))

        a = SegmentationPipeline(with_texture, backend="classical").run(micro)
        b = SegmentationPipeline(without, backend="classical").run(micro)
        assert any("glcm" in c for c in a.features.columns)
        assert not any("glcm" in c for c in b.features.columns)
