"""Joint image/label augmentation, implemented directly on numpy + OpenCV.

Written by hand rather than pulled from Albumentations for two reasons: label
images need nearest-neighbour interpolation everywhere (bilinear resampling of
an instance map invents label ids that belong to no object), and the geometric
choices here are microscopy-specific.

What is and is not appropriate for microscopy:

* **flips and 90-degree rotations are free** -- a field of view has no canonical
  orientation, unlike natural images where the sky is up;
* **elastic deformation** is the classic U-Net augmentation for cell data,
  standing in for genuine biological shape variation;
* **scale jitter** matters because BBBC038 spans magnifications;
* **intensity jitter, blur and noise** simulate exposure, focus and gain drift
  across a plate -- the variation the model must be invariant to at inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AugmentParams:
    flip_prob: float = 0.5
    rot90_prob: float = 0.5
    affine_prob: float = 0.5
    max_rotation_deg: float = 30.0
    scale_range: tuple[float, float] = (0.85, 1.2)
    elastic_prob: float = 0.2
    elastic_alpha: float = 20.0
    elastic_sigma: float = 6.0
    intensity_prob: float = 0.5
    brightness_delta: float = 0.15
    contrast_range: tuple[float, float] = (0.8, 1.25)
    gamma_range: tuple[float, float] = (0.75, 1.35)
    blur_prob: float = 0.15
    noise_prob: float = 0.2
    noise_sigma: float = 0.03


def random_flip_rot(
    image: np.ndarray, labels: np.ndarray, rng: np.random.Generator, params: AugmentParams
) -> tuple[np.ndarray, np.ndarray]:
    """Dihedral (flip + 90-degree rotation) augmentation."""
    if rng.random() < params.flip_prob:
        image, labels = image[:, ::-1], labels[:, ::-1]
    if rng.random() < params.flip_prob:
        image, labels = image[::-1], labels[::-1]
    if rng.random() < params.rot90_prob:
        k = int(rng.integers(1, 4))
        image, labels = np.rot90(image, k), np.rot90(labels, k)
    return np.ascontiguousarray(image), np.ascontiguousarray(labels)


def random_affine(
    image: np.ndarray, labels: np.ndarray, rng: np.random.Generator, params: AugmentParams
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation + scale about the image centre.

    ``BORDER_REFLECT_101`` keeps the padded margin statistically like the image;
    zero padding would teach the model that black borders mean background.
    """
    h, w = image.shape[:2]
    angle = float(rng.uniform(-params.max_rotation_deg, params.max_rotation_deg))
    scale = float(rng.uniform(*params.scale_range))
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)

    image = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    labels = cv2.warpAffine(
        labels.astype(np.int32),
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return image, labels.astype(np.int32)


def elastic_deform(
    image: np.ndarray, labels: np.ndarray, rng: np.random.Generator, params: AugmentParams
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth random displacement field (Simard et al., as used by U-Net).

    Random per-pixel offsets are smoothed with a Gaussian so the field is
    locally coherent: the result looks like a genuinely different specimen
    rather than a scrambled one.
    """
    h, w = image.shape[:2]
    dx = rng.uniform(-1, 1, (h, w)).astype(np.float32)
    dy = rng.uniform(-1, 1, (h, w)).astype(np.float32)
    k = int(2 * round(3 * params.elastic_sigma) + 1)
    dx = cv2.GaussianBlur(dx, (k, k), params.elastic_sigma) * params.elastic_alpha
    dy = cv2.GaussianBlur(dy, (k, k), params.elastic_sigma) * params.elastic_alpha

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x, map_y = xx + dx, yy + dy

    image = cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    labels = cv2.remap(
        labels.astype(np.int32),
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return image, labels.astype(np.int32)


def random_intensity(
    image: np.ndarray, rng: np.random.Generator, params: AugmentParams
) -> np.ndarray:
    """Brightness, contrast and gamma jitter -- labels are unaffected."""
    out = image.astype(np.float32)
    out = out * float(rng.uniform(*params.contrast_range))
    out = out + float(rng.uniform(-params.brightness_delta, params.brightness_delta))
    out = np.clip(out, 0.0, 1.0)
    out = np.power(out, float(rng.uniform(*params.gamma_range)))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def augment(
    image: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    params: AugmentParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the full augmentation stack to a 2-D image and its label map."""
    params = params or AugmentParams()
    image = np.asarray(image, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    image, labels = random_flip_rot(image, labels, rng, params)
    if rng.random() < params.affine_prob:
        image, labels = random_affine(image, labels, rng, params)
    if rng.random() < params.elastic_prob:
        image, labels = elastic_deform(image, labels, rng, params)
    if rng.random() < params.intensity_prob:
        image = random_intensity(image, rng, params)
    if rng.random() < params.blur_prob:
        sigma = float(rng.uniform(0.5, 1.5))
        k = int(2 * round(3 * sigma) + 1)
        image = cv2.GaussianBlur(image, (k, k), sigma)
    if rng.random() < params.noise_prob:
        image = image + rng.normal(0, params.noise_sigma, image.shape).astype(np.float32)

    return np.clip(image, 0.0, 1.0).astype(np.float32), labels


def random_crop(
    image: np.ndarray, labels: np.ndarray, size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random ``size x size`` crop, reflect-padding images smaller than ``size``."""
    image, labels = pad_to_min(image, labels, size)
    h, w = image.shape[:2]
    top = int(rng.integers(0, h - size + 1))
    left = int(rng.integers(0, w - size + 1))
    return (
        image[top : top + size, left : left + size],
        labels[top : top + size, left : left + size],
    )


def center_crop(
    image: np.ndarray, labels: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic centre crop -- used for validation and test tiles."""
    image, labels = pad_to_min(image, labels, size)
    h, w = image.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return (
        image[top : top + size, left : left + size],
        labels[top : top + size, left : left + size],
    )


def pad_to_min(
    image: np.ndarray, labels: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reflect-pad an image (and zero-pad its labels) up to ``size``."""
    h, w = image.shape[:2]
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    if pad_h == 0 and pad_w == 0:
        return image, labels
    top, left = pad_h // 2, pad_w // 2
    image = cv2.copyMakeBorder(
        image, top, pad_h - top, left, pad_w - left, cv2.BORDER_REFLECT_101
    )
    labels = cv2.copyMakeBorder(
        labels.astype(np.int32),
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    return image, labels.astype(np.int32)


def pad_to_multiple(image: np.ndarray, multiple: int = 16) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad an image so both sides divide ``multiple``.

    A U-Net with depth *d* halves the resolution *d* times, so inputs whose
    dimensions are not a multiple of ``2**d`` produce skip connections of
    mismatched size.  Returns the padded image and the original shape so the
    prediction can be cropped back.
    """
    h, w = image.shape[:2]
    new_h = int(np.ceil(h / multiple) * multiple)
    new_w = int(np.ceil(w / multiple) * multiple)
    if (new_h, new_w) == (h, w):
        return image, (h, w)
    padded = cv2.copyMakeBorder(
        image, 0, new_h - h, 0, new_w - w, cv2.BORDER_REFLECT_101
    )
    return padded, (h, w)
