"""Preprocessing tests.

Each test states the property the operator is supposed to have, so a failure
says what broke rather than merely that a number changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from microseg.channels import ChannelSpec, MicroscopyImage
from microseg.config import PreprocessConfig
from microseg.preprocessing import (
    adjust_gamma,
    clahe,
    correct_illumination,
    denoise,
    has_dark_background,
    maybe_invert,
    normalize,
    percentile_normalize,
    preprocess,
    preprocess_array,
    remove_hot_pixels,
    zscore_normalize,
)
from microseg.preprocessing.illumination import synthetic_illumination_gradient


class TestNormalization:
    def test_percentile_normalize_maps_to_unit_range(self, rng):
        img = rng.uniform(0.2, 0.6, (64, 64)).astype(np.float32)
        out = percentile_normalize(img, 1, 99)
        assert out.min() == pytest.approx(0.0, abs=1e-5)
        assert out.max() == pytest.approx(1.0, abs=1e-5)

    def test_percentile_normalize_is_robust_to_outliers(self, rng):
        """A single saturated pixel must not compress the real signal.

        This is the whole reason percentile normalisation is the default over
        min-max for microscopy.
        """
        img = rng.uniform(0.2, 0.4, (64, 64)).astype(np.float32)
        clean = percentile_normalize(img)
        img[0, 0] = 1000.0
        with_hot_pixel = percentile_normalize(img)
        # The bulk of the image should be almost unchanged.
        assert np.abs(clean[1:, 1:] - with_hot_pixel[1:, 1:]).mean() < 0.02

    def test_flat_image_does_not_amplify_noise(self):
        assert np.all(percentile_normalize(np.full((16, 16), 0.5, np.float32)) == 0.0)

    def test_zscore_output_is_bounded_when_clipped(self, rng):
        out = zscore_normalize(rng.normal(0.5, 0.1, (64, 64)).astype(np.float32))
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown normalisation"):
            normalize(np.zeros((8, 8), np.float32), "nonsense")


class TestPolarity:
    def test_detects_fluorescence_polarity(self, easy_field):
        image, _ = easy_field
        assert has_dark_background(image)

    def test_detects_brightfield_polarity(self, easy_field):
        image, _ = easy_field
        assert not has_dark_background(1.0 - image)

    def test_inversion_restores_bright_objects(self, easy_field):
        """A brightfield image must come out with the same polarity as fluorescence."""
        image, labels = easy_field
        inverted, was_inverted = maybe_invert(1.0 - image)
        assert was_inverted
        assert np.allclose(inverted, image, atol=1e-5)
        # Objects should now be brighter than background.
        assert inverted[labels > 0].mean() > inverted[labels == 0].mean()

    def test_fluorescence_is_left_alone(self, easy_field):
        image, _ = easy_field
        out, was_inverted = maybe_invert(image)
        assert not was_inverted
        assert np.array_equal(out, image)


class TestArtifactCorrection:
    def test_hot_pixels_are_replaced(self, easy_field, rng):
        image, _ = easy_field
        corrupted = image.copy()
        coords = [(5, 5), (30, 70), (100, 20)]
        for y, x in coords:
            corrupted[y, x] = 1.0
        cleaned = remove_hot_pixels(corrupted, threshold=4.0)
        for y, x in coords:
            assert cleaned[y, x] < 0.9, f"hot pixel at {(y, x)} survived"

    def test_real_structure_is_preserved(self, easy_field):
        """Only outliers may change; nuclei are not outliers."""
        image, labels = easy_field
        cleaned = remove_hot_pixels(image, threshold=5.0)
        changed = np.abs(cleaned - image) > 0.05
        assert changed.mean() < 0.02


def _background_spread(image: np.ndarray, background: np.ndarray) -> float:
    """Range of mean background intensity across a 3x3 grid of the field.

    This is the quantity a global threshold actually depends on: if background
    level varies across the image, one threshold cannot be right everywhere.
    """
    h, w = image.shape
    means = []
    for i in range(3):
        for j in range(3):
            tile = (slice(i * h // 3, (i + 1) * h // 3), slice(j * w // 3, (j + 1) * w // 3))
            pixels = image[tile][background[tile]]
            if pixels.size:
                means.append(float(pixels.mean()))
    return max(means) - min(means)


class TestIllumination:
    def test_subtract_flattens_an_additive_background(self, easy_field):
        """The headline property: after correction, background level is uniform.

        Modelled as stray light / camera offset, i.e. a smooth additive gradient
        -- the component that a *subtractive* correction is the right model for.
        """
        image, labels = easy_field
        # 1 - field gives a smooth radial ramp spanning roughly 0 to 0.35.
        ramp = 0.35 * (1.0 - synthetic_illumination_gradient(image.shape, 1.0, seed=1))
        shaded = np.clip(image + ramp, 0, 1).astype(np.float32)

        background = labels == 0
        corrected = correct_illumination(shaded, "morphological", radius=30, mode="subtract")

        before = _background_spread(shaded, background)
        after = _background_spread(corrected, background)
        assert after < 0.5 * before, f"background spread {before:.3f} -> {after:.3f}"

    def test_divide_equalises_multiplicative_shading(self, easy_field):
        """Vignetting scales signal *and* background, so division is the model.

        With a realistic background pedestal present, dividing by the estimated
        field should restore comparable object-to-background contrast across the
        image.
        """
        image, labels = easy_field
        shading = synthetic_illumination_gradient(image.shape, strength=0.6, seed=2)
        # A pedestal makes the shading observable in the background, which is
        # what division needs in order to estimate it.
        shaded = np.clip((image + 0.2) * shading, 0, 1).astype(np.float32)
        corrected = correct_illumination(shaded, "morphological", radius=30, mode="divide")

        def contrast_ratio(a):
            """Object-vs-background contrast in the brightest vs dimmest corner."""
            h, w = a.shape
            corners = [(slice(0, h // 3), slice(0, w // 3)), (slice(-h // 3, None), slice(-w // 3, None))]
            values = []
            for corner in corners:
                objects = a[corner][labels[corner] > 0]
                back = a[corner][labels[corner] == 0]
                if objects.size and back.size:
                    values.append(objects.mean() - back.mean())
            return max(values) / max(min(values), 1e-6) if len(values) == 2 else 1.0

        assert contrast_ratio(corrected) < contrast_ratio(shaded)

    def test_polynomial_backend_runs(self, easy_field):
        image, _ = easy_field
        out = correct_illumination(image, "polynomial")
        assert out.shape == image.shape and out.dtype == np.float32

    def test_correction_preserves_objects(self, easy_field):
        """Flattening must not erase the nuclei along with the background."""
        image, labels = easy_field
        corrected = correct_illumination(image, "morphological", radius=30)
        assert corrected[labels > 0].mean() > corrected[labels == 0].mean() + 0.1

    def test_downsampled_background_matches_the_exact_opening(self, easy_field):
        """Large openings are estimated on a downsampled copy for speed.

        The background is low-frequency by construction, so the approximation
        must stay close to the exact morphological opening; this pins that
        equivalence so the optimisation cannot silently drift.
        """
        import cv2

        from microseg.preprocessing.illumination import _morphological_background

        image, _ = easy_field
        radius = 40
        k = 2 * radius + 1
        exact = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
        approx = _morphological_background(image, radius)

        assert approx.shape == image.shape
        assert np.abs(approx - exact).mean() < 0.05

    def test_small_radius_uses_the_exact_opening(self, easy_field):
        """Below the kernel-size threshold there is nothing to gain, so no scaling."""
        import cv2

        from microseg.preprocessing.illumination import _morphological_background

        image, _ = easy_field
        k = 2 * 5 + 1
        exact = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
        assert np.allclose(_morphological_background(image, 5), exact)

    def test_unknown_method_raises(self, easy_field):
        with pytest.raises(ValueError, match="unknown illumination"):
            correct_illumination(easy_field[0], "wavelet")


class TestDenoise:
    @pytest.mark.parametrize("method", ["gaussian", "median", "bilateral", "nlm", "tv"])
    def test_every_denoiser_reduces_noise(self, easy_field, method, rng):
        image, _ = easy_field
        noisy = np.clip(image + rng.normal(0, 0.08, image.shape), 0, 1).astype(np.float32)
        cleaned = denoise(noisy, method)
        assert np.abs(cleaned - image).mean() < np.abs(noisy - image).mean(), (
            f"{method} did not move the image closer to the clean original"
        )

    def test_none_is_a_passthrough(self, easy_field):
        image, _ = easy_field
        assert np.array_equal(denoise(image, "none"), image)


class TestContrast:
    def test_clahe_increases_local_contrast(self, easy_field):
        image, _ = easy_field
        low_contrast = (image * 0.3 + 0.35).astype(np.float32)
        assert clahe(low_contrast).std() > low_contrast.std()

    def test_gamma_below_one_brightens(self, easy_field):
        image, _ = easy_field
        assert adjust_gamma(image, 0.5).mean() > image.mean()

    def test_invalid_gamma_raises(self, easy_field):
        with pytest.raises(ValueError, match="gamma must be positive"):
            adjust_gamma(easy_field[0], 0.0)


class TestChain:
    def test_output_is_normalised_float32(self, easy_field):
        image, _ = easy_field
        out, _, _ = preprocess_array(image, PreprocessConfig())
        assert out.dtype == np.float32
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_stages_are_collected_in_order(self, easy_field):
        image, _ = easy_field
        _, stages, _ = preprocess_array(image, PreprocessConfig(), collect_stages=True)
        assert list(stages)[0] == "raw"
        assert list(stages)[-1] == "normalized"

    def test_disabling_every_stage_only_normalises(self, easy_field):
        image, _ = easy_field
        cfg = PreprocessConfig(
            auto_invert=False, hot_pixel_correction=False,
            illumination=None, denoise=None, contrast=None, normalize="none",
        )
        out, _, _ = preprocess_array(image, cfg)
        assert np.allclose(out, image)

    def test_channels_are_processed_independently(self, multichannel_image):
        """A multiplexed stack must not be normalised jointly.

        Each channel has its own background and dynamic range, so each should
        end up spanning the full [0, 1] range on its own.
        """
        image, _ = multichannel_image
        result = preprocess(image, PreprocessConfig())
        for c in range(result.image.n_channels):
            channel = result.image.data[:, :, c]
            assert channel.max() > 0.9, f"channel {c} was not normalised on its own"

    def test_selector_leaves_other_channels_untouched(self, multichannel_image):
        image, _ = multichannel_image
        result = preprocess(image, PreprocessConfig(), channels="nuclei")
        assert not np.array_equal(result.image.channel("nuclei"), image.channel("nuclei"))
        assert np.array_equal(result.image.channel("marker1"), image.channel("marker1"))


class TestMicroscopyImage:
    def test_resolves_channels_by_name_and_role(self, multichannel_image):
        image, _ = multichannel_image
        assert image.index_of("nuclei") == 0
        assert image.index_of("marker1") == 1
        assert image.resolve(None) == [0, 1, 2]
        assert image.resolve("marker2") == [2]

    def test_missing_channel_raises(self, multichannel_image):
        image, _ = multichannel_image
        with pytest.raises(KeyError, match="no channel named"):
            image.index_of("dapi")

    def test_2d_input_is_promoted_to_one_channel(self):
        image = MicroscopyImage.from_array(np.zeros((8, 8), np.float32))
        assert image.shape == (8, 8, 1)
        assert image.channels[0].role == "nuclei"

    def test_uint8_is_scaled_to_unit_range(self):
        image = MicroscopyImage.from_array(np.full((4, 4), 255, np.uint8))
        assert image.data.max() == pytest.approx(1.0)

    def test_channel_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="channel specs"):
            MicroscopyImage(np.zeros((4, 4, 2), np.float32), [ChannelSpec("only")])

    def test_unknown_role_raises(self):
        with pytest.raises(ValueError, match="unknown channel role"):
            ChannelSpec("x", "mitochondria")
