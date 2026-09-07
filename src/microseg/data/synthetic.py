"""Synthetic microscopy field generator.

Used for two things:

* **hermetic tests** -- the test suite must run in CI without an 80 MB download,
  and needs ground truth it can assert exact counts against;
* **controlled stress tests** -- density, noise and shading are independent
  knobs, so a failure can be attributed to one factor rather than guessed at.

The forward model is a simplified but physically motivated one: elliptical
nuclei with a smooth intensity profile, blurred by a Gaussian PSF, scaled by a
multiplicative illumination field, then corrupted with Poisson (shot) plus
Gaussian (read) noise -- the two noise sources that actually dominate a
fluorescence camera.
"""

from __future__ import annotations

import cv2
import numpy as np

from microseg.channels import ChannelSpec, MicroscopyImage


def synthetic_field(
    shape: tuple[int, int] = (256, 256),
    n_objects: int = 30,
    radius_range: tuple[int, int] = (6, 14),
    noise: float = 0.03,
    psf_sigma: float = 1.2,
    illumination_strength: float = 0.0,
    allow_touching: bool = True,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one synthetic field.

    Returns ``(image, labels)`` where ``image`` is float32 in ``[0, 1]`` and
    ``labels`` is an int32 instance map.  Objects are placed one at a time and,
    when ``allow_touching`` is False, rejected if they overlap an existing one --
    which gives the easy case for the classical baseline.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    image = np.zeros((h, w), dtype=np.float32)
    labels = np.zeros((h, w), dtype=np.int32)

    next_label = 1
    attempts = 0
    max_attempts = n_objects * 40

    while next_label <= n_objects and attempts < max_attempts:
        attempts += 1
        ry = int(rng.integers(radius_range[0], radius_range[1] + 1))
        rx = int(rng.integers(radius_range[0], radius_range[1] + 1))
        cy = int(rng.integers(ry, max(ry + 1, h - ry)))
        cx = int(rng.integers(rx, max(rx + 1, w - rx)))
        angle = float(rng.uniform(0, 180))

        candidate = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(candidate, (cx, cy), (rx, ry), angle, 0, 360, 1, -1)
        candidate_mask = candidate.astype(bool)
        if not candidate_mask.any():
            continue

        overlap = labels[candidate_mask] > 0
        if not allow_touching and overlap.any():
            continue
        # Even when touching is allowed, an object almost entirely buried under
        # an existing one is not a useful annotation.
        if overlap.mean() > 0.5:
            continue

        brightness = float(rng.uniform(0.55, 1.0))
        # A soft radial falloff rather than a flat disc: real nuclei are brighter
        # at the centre, which is what makes the distance transform meaningful.
        distance = cv2.distanceTransform(candidate, cv2.DIST_L2, 3)
        peak = float(distance.max())
        if peak > 0:
            profile = 0.55 + 0.45 * (distance / peak)
        else:
            profile = candidate.astype(np.float32)
        image = np.maximum(image, brightness * profile * candidate_mask)

        labels[candidate_mask] = next_label
        next_label += 1

    if psf_sigma > 0:
        k = int(2 * round(3 * psf_sigma) + 1)
        image = cv2.GaussianBlur(image, (k, k), psf_sigma)

    if illumination_strength > 0:
        from microseg.preprocessing.illumination import synthetic_illumination_gradient

        image = image * synthetic_illumination_gradient(
            shape, illumination_strength, seed=seed
        )

    if noise > 0:
        # Shot noise scales with signal; read noise does not.
        shot = rng.poisson(np.clip(image, 0, None) * 255.0) / 255.0
        image = 0.5 * image + 0.5 * shot.astype(np.float32)
        image = image + rng.normal(0.0, noise, image.shape).astype(np.float32)

    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return image, labels


def synthetic_microscopy_image(
    n_channels: int = 1, seed: int | None = 0, **kwargs
) -> tuple[MicroscopyImage, np.ndarray]:
    """Multi-channel variant, for exercising the multi-channel code paths.

    Channel 0 is the nuclear stain and carries the ground-truth labels;
    additional channels are marker stains generated over the same objects with
    independent brightness, mimicking a multiplexed assay where segmentation
    happens on one channel and quantification on all of them.
    """
    image, labels = synthetic_field(seed=seed, **kwargs)
    stack = [image]
    specs = [ChannelSpec("nuclei", "nuclei")]

    rng = np.random.default_rng(None if seed is None else seed + 1000)
    for c in range(1, n_channels):
        gains = rng.uniform(0.2, 1.0, size=int(labels.max()) + 1).astype(np.float32)
        gains[0] = 0.0
        marker = gains[labels] * (image > 0.05)
        marker = cv2.GaussianBlur(marker, (5, 5), 1.0)
        marker += rng.normal(0.0, 0.02, marker.shape).astype(np.float32)
        stack.append(np.clip(marker, 0, 1).astype(np.float32))
        specs.append(ChannelSpec(f"marker{c}", "marker"))

    data = np.stack(stack, axis=-1)
    return MicroscopyImage(data, specs, name="synthetic"), labels
