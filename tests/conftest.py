"""Shared fixtures.

Every test runs on synthetic data, so the suite is hermetic: no 80 MB download,
no trained checkpoint, and ground truth that exact assertions can be made
against.
"""

from __future__ import annotations

import numpy as np
import pytest

from microseg.data.synthetic import synthetic_field, synthetic_microscopy_image


@pytest.fixture(scope="session")
def easy_field():
    """Well-separated round objects: the case every backend must get right."""
    return synthetic_field(
        shape=(192, 192), n_objects=15, radius_range=(8, 12), noise=0.02,
        allow_touching=False, seed=0,
    )


@pytest.fixture(scope="session")
def crowded_field():
    """Touching objects: the case that separates instance from semantic methods."""
    return synthetic_field(
        shape=(192, 192), n_objects=40, radius_range=(7, 11), noise=0.03,
        allow_touching=True, seed=3,
    )


@pytest.fixture(scope="session")
def multichannel_image():
    """A 3-channel stack (nuclei + two markers) with its instance labels."""
    return synthetic_microscopy_image(
        n_channels=3, shape=(160, 160), n_objects=12, allow_touching=False, seed=7
    )


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
