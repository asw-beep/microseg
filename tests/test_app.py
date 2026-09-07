"""Gradio app entry points.

The app is an optional extra (`pip install .[app]`), so the whole module skips
when gradio is absent -- a headless training container must still run the suite.

What is worth testing here is not the widgets but the seam between them and the
pipeline: `_to_image` turns whatever the upload widget hands over into the 0..1
single-channel field every downstream stage assumes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gradio")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import gradio_app as app  # noqa: E402


def _grey_field(dtype=np.uint8, channels=3):
    """A dim fluorescence field saved the way an image viewer would save it:
    identical RGB channels, nuclei well below full scale."""
    peak = np.iinfo(dtype).max
    rng = np.random.default_rng(0)
    grey = rng.integers(1, 6, size=(64, 64)).astype(np.float64)
    grey[20:30, 20:30] = peak * 0.31  # one bright nucleus
    grey = grey.astype(dtype)
    if channels == 1:
        return grey
    stacked = np.repeat(grey[:, :, None], 3, axis=2)
    if channels == 4:
        alpha = np.full((64, 64, 1), peak, dtype=dtype)
        stacked = np.concatenate([stacked, alpha], axis=2)
    return stacked


class TestUploadConversion:
    @pytest.mark.parametrize("channels", [1, 3, 4])
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
    def test_uploads_arrive_scaled_to_unit_range(self, dtype, channels):
        """The regression this file exists for.

        Collapsing identical RGB channels with `.mean()` produced a *float*
        array, so the integer-dtype check inside `MicroscopyImage.from_array`
        stopped firing and the /255 scaling was skipped -- an 8-bit upload
        reached the pipeline in 0..255. Nothing raised; CLAHE simply returned
        garbage and a dim field came back with no nuclei at all.
        """
        data = app._to_image(_grey_field(dtype, channels)).nuclei()
        assert data.dtype == np.float32
        assert 0.0 <= data.min() and data.max() <= 1.0
        # The bright nucleus must survive as a real value, not be flattened.
        assert 0.2 < data.max() < 0.5

    def test_alpha_channel_is_dropped_not_averaged_in(self):
        """A constant alpha would drag every pixel toward full scale."""
        rgb = app._to_image(_grey_field(np.uint8, 3)).nuclei()
        rgba = app._to_image(_grey_field(np.uint8, 4)).nuclei()
        assert np.allclose(rgb, rgba, atol=1e-6)

    def test_a_dominant_stain_channel_is_picked_over_luminance(self):
        """The other branch: one channel carrying the signal is used as-is,
        rather than being diluted by two empty ones."""
        arr = np.zeros((64, 64, 3), np.uint8)
        arr[:, :, 1] = 10
        arr[20:30, 20:30, 1] = 200  # signal lives only in green
        data = app._to_image(arr).nuclei()
        assert np.isclose(data.max(), 200 / 255, atol=1e-3)

    def test_float_input_is_left_alone(self):
        """Gradio can hand over a float array; it must not be rescaled again."""
        arr = np.zeros((32, 32), np.float32)
        arr[8:16, 8:16] = 0.75
        assert np.isclose(app._to_image(arr).nuclei().max(), 0.75)


class TestAnalyseEndToEnd:
    def test_a_dim_grayscale_upload_yields_objects(self):
        """The user-visible symptom: 'No nuclei found' on a field that has them."""
        from microseg.config import ExperimentConfig

        app.STATE["cfg"] = ExperimentConfig()
        outputs = app.analyze(
            _grey_field(np.uint8, 4),
            app.CLASSICAL,
            "morphological",
            "median",
            "clahe",
            15,
            0.0,
            False,
        )
        table = outputs[5]
        assert table is not None and len(table) > 0
        assert "note" not in table.columns, "pipeline reported no objects"


class TestThemeControls:
    """Guards on the custom CSS, not on rendering.

    Gradio draws radios and checkboxes with `appearance: none`, so the *only*
    thing distinguishing a selected option from an unselected one is the
    background it paints. Applying the `background` shorthand to those inputs
    erases both the colour and the indicator image, and because there is no
    native control underneath, every option then renders identically -- the
    segmentation-method picker looked unselectable even though it worked.
    """

    @staticmethod
    def _rules_for(selector_fragment: str) -> list[str]:
        """Every CSS rule body whose selector mentions the fragment."""
        from theme import CSS

        rules = []
        for block in CSS.split("}"):
            if "{" not in block:
                continue
            selector, _, body = block.partition("{")
            if selector_fragment in selector:
                rules.append(body)
        return rules

    def test_the_background_shorthand_is_never_used_on_toggles(self):
        for kind in ('input[type="radio"]', 'input[type="checkbox"]'):
            for body in self._rules_for(kind):
                for declaration in body.split(";"):
                    prop = declaration.split(":")[0].strip()
                    assert prop != "background", (
                        f"{kind} rule sets the `background` shorthand, which wipes "
                        "the selected-state indicator"
                    )

    def test_a_checked_state_is_styled(self):
        """Something must visibly change when an option is chosen."""
        assert self._rules_for('input[type="radio"]:checked'), (
            "no :checked rule for radios -- selection would be invisible"
        )
        assert self._rules_for('input[type="checkbox"]:checked')

    def test_checked_and_unchecked_backgrounds_differ(self):
        unchecked = " ".join(
            b for b in self._rules_for('input[type="radio"]') if ":checked" not in b
        )
        checked = " ".join(self._rules_for('input[type="radio"]:checked'))
        assert "background-color" in unchecked and "background-color" in checked
        assert unchecked != checked
