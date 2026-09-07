"""Quantitative feature extraction tests.

Features are measurements, so they are tested against analytically known values
wherever one exists -- a disc of radius r has area pi*r^2 and circularity 1 --
rather than against whatever the code happened to produce first.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from microseg.analysis import extract_features, morphology_features, summarize_image
from microseg.analysis.features import intensity_features
from microseg.analysis.texture import glcm_features, object_glcm_features
from microseg.channels import ChannelSpec, MicroscopyImage
from microseg.config import AnalysisConfig


def _disc(radius: int, size: int = 120) -> np.ndarray:
    labels = np.zeros((size, size), np.int32)
    cv2.circle(labels, (size // 2, size // 2), radius, 1, -1)
    return labels


class TestMorphology:
    def test_area_matches_the_analytic_circle(self):
        labels = _disc(20)
        area = morphology_features(labels)["area_px"].iloc[0]
        assert area == pytest.approx(np.pi * 20**2, rel=0.03)

    def test_circle_has_circularity_near_one(self):
        """Uses the Crofton perimeter; the naive pixel perimeter scores ~0.8."""
        assert morphology_features(_disc(25))["circularity"].iloc[0] > 0.95

    def test_elongated_object_has_high_eccentricity(self):
        labels = np.zeros((80, 80), np.int32)
        cv2.ellipse(labels, (40, 40), (30, 6), 0, 0, 360, 1, -1)
        row = morphology_features(labels).iloc[0]
        assert row["eccentricity"] > 0.9
        assert row["aspect_ratio"] > 3.0
        assert row["circularity"] < 0.8

    def test_concave_object_has_low_solidity(self):
        """A merged pair of nuclei is concave -- solidity flags it."""
        labels = np.zeros((80, 120), np.int32)
        cv2.circle(labels, (40, 40), 22, 1, -1)
        cv2.circle(labels, (78, 40), 22, 1, -1)
        assert morphology_features(labels)["solidity"].iloc[0] < 0.95

    def test_holes_are_reported(self):
        labels = np.zeros((80, 80), np.int32)
        cv2.circle(labels, (40, 40), 25, 1, -1)
        cv2.circle(labels, (40, 40), 6, 0, -1)  # a nucleolus-sized hole
        assert morphology_features(labels)["hole_area_px"].iloc[0] > 90

    def test_border_objects_are_flagged(self):
        labels = np.zeros((60, 60), np.int32)
        labels[0:10, 0:10] = 1  # clipped by the field of view
        labels[30:40, 30:40] = 2
        df = morphology_features(labels).set_index("label")
        assert bool(df.loc[1, "touches_border"]) and not bool(df.loc[2, "touches_border"])

    def test_physical_units_are_added_when_calibrated(self):
        df = morphology_features(_disc(20), pixel_size_um=0.65)
        assert "area_um2" in df
        assert df["area_um2"].iloc[0] == pytest.approx(df["area_px"].iloc[0] * 0.65**2)

    def test_empty_label_image_returns_empty_frame(self):
        assert morphology_features(np.zeros((32, 32), np.int32)).empty


class TestIntensity:
    def test_statistics_match_a_known_constant_object(self):
        labels = np.zeros((40, 40), np.int32)
        labels[10:20, 10:20] = 1  # 100 px
        intensity = np.zeros((40, 40), np.float32)
        intensity[10:20, 10:20] = 0.5

        row = intensity_features(labels, intensity).iloc[0]
        assert row["mean_intensity"] == pytest.approx(0.5)
        assert row["integrated_intensity"] == pytest.approx(50.0)
        assert row["std_intensity"] == pytest.approx(0.0, abs=1e-6)

    def test_integrated_intensity_scales_with_area(self):
        """The property that makes it the right measure for total DNA content."""
        small, large = np.zeros((60, 60), np.int32), np.zeros((60, 60), np.int32)
        small[10:20, 10:20] = 1
        large[30:50, 30:50] = 1
        intensity = np.full((60, 60), 0.5, np.float32)
        s = intensity_features(small, intensity)["integrated_intensity"].iloc[0]
        l = intensity_features(large, intensity)["integrated_intensity"].iloc[0]
        assert l == pytest.approx(4 * s)


class TestTexture:
    def test_smooth_region_is_more_homogeneous_than_noisy(self, rng):
        """The defining property of the homogeneity feature."""
        mask = np.ones((48, 48), bool)
        smooth = np.linspace(0.3, 0.6, 48 * 48).reshape(48, 48).astype(np.float32)
        noisy = rng.uniform(0, 1, (48, 48)).astype(np.float32)
        assert (
            glcm_features(smooth, mask)["glcm_homogeneity"]
            > glcm_features(noisy, mask)["glcm_homogeneity"]
        )

    def test_noisy_region_has_higher_contrast_and_entropy(self, rng):
        mask = np.ones((48, 48), bool)
        smooth = np.full((48, 48), 0.5, np.float32)
        noisy = rng.uniform(0, 1, (48, 48)).astype(np.float32)
        a, b = glcm_features(smooth, mask), glcm_features(noisy, mask)
        assert b["glcm_contrast"] > a["glcm_contrast"]
        assert b["glcm_entropy"] > a["glcm_entropy"]

    def test_features_are_rotation_invariant(self, rng):
        """Averaging over four angles is what buys this; a field has no 'up'."""
        patch = rng.uniform(0, 1, (40, 40)).astype(np.float32)
        mask = np.ones((40, 40), bool)
        a = glcm_features(patch, mask)
        b = glcm_features(np.rot90(patch).copy(), np.rot90(mask).copy())
        assert a["glcm_contrast"] == pytest.approx(b["glcm_contrast"], rel=0.02)

    def test_tiny_object_returns_zeros_not_an_error(self):
        out = glcm_features(np.zeros((3, 3), np.float32), np.zeros((3, 3), bool))
        assert out["glcm_contrast"] == 0.0

    def test_one_row_per_object(self, easy_field):
        image, labels = easy_field
        rows = object_glcm_features(labels, image)
        assert len(rows) == labels.max()
        assert all("glcm_entropy" in r for r in rows)


class TestFeatureTable:
    def test_one_row_per_object_with_expected_columns(self, easy_field):
        image, labels = easy_field
        micro = MicroscopyImage.from_array(image, [ChannelSpec("nuclei", "nuclei")], name="t")
        df = extract_features(labels, micro, AnalysisConfig(compute_texture=True))

        assert len(df) == labels.max()
        for column in ("area_px", "circularity", "mean_intensity", "glcm_contrast", "image_id"):
            assert column in df.columns

    def test_every_channel_is_quantified(self, multichannel_image):
        """Segment on one channel, measure on all -- the multiplexed-assay case."""
        image, labels = multichannel_image
        df = extract_features(labels, image, AnalysisConfig(compute_texture=False))
        for name in ("nuclei", "marker1", "marker2"):
            assert f"{name}_mean_intensity" in df.columns

    def test_texture_can_be_disabled(self, easy_field):
        image, labels = easy_field
        micro = MicroscopyImage.from_array(image, [ChannelSpec("nuclei", "nuclei")])
        df = extract_features(labels, micro, AnalysisConfig(compute_texture=False))
        assert not [c for c in df.columns if "glcm" in c]

    def test_border_objects_can_be_excluded(self, easy_field):
        image, labels = easy_field
        micro = MicroscopyImage.from_array(image, [ChannelSpec("nuclei", "nuclei")])
        keep_all = extract_features(labels, micro, AnalysisConfig(compute_texture=False))
        excluded = extract_features(
            labels, micro, AnalysisConfig(compute_texture=False, exclude_border_objects=True)
        )
        assert len(excluded) <= len(keep_all)
        assert not excluded["touches_border"].any()

    def test_summary_reports_the_count(self, easy_field):
        image, labels = easy_field
        micro = MicroscopyImage.from_array(image, [ChannelSpec("nuclei", "nuclei")])
        df = extract_features(labels, micro, AnalysisConfig(compute_texture=False))
        summary = summarize_image(df, "t")
        assert summary["n_objects"] == labels.max()
        assert "area_px_mean" in summary

    def test_summary_of_empty_frame_is_safe(self):
        import pandas as pd

        assert summarize_image(pd.DataFrame(), "empty")["n_objects"] == 0
