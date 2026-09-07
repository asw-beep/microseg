"""Reading microscopy images from disk.

TIFF gets special handling because it is what microscopes actually write, and a
multi-page TIFF is ambiguous: the extra axis may be channels, z-slices or time.
:func:`load_image` guesses using the heuristic that a small leading axis (<= 8)
is channels, and says what it assumed in ``meta`` so a caller can correct it
rather than silently analysing a z-stack as a multiplexed field.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from microseg.channels import ChannelSpec, MicroscopyImage

TIFF_SUFFIXES = {".tif", ".tiff"}
IMAGE_SUFFIXES = TIFF_SUFFIXES | {".png", ".jpg", ".jpeg", ".bmp"}


def load_image(
    path: str | Path,
    channel_names: list[str] | None = None,
    nuclei_channel: str | int | None = None,
    pixel_size_um: float | None = None,
) -> MicroscopyImage:
    """Load an image file into a :class:`MicroscopyImage`.

    Parameters
    ----------
    channel_names:
        Names for the channels, in file order.  Without them, a 1-channel image
        is assumed to be nuclei and an RGB image is reduced to a single nuclei
        channel (see below).
    nuclei_channel:
        Which channel to segment on.  Its role is set to ``nuclei``, which is
        what :meth:`MicroscopyImage.nuclei` looks for.
    """
    path = Path(path)
    if path.suffix.lower() in TIFF_SUFFIXES:
        arr, note = _read_tiff(path)
    else:
        arr, note = _read_standard(path), "2-D/RGB image"

    image = MicroscopyImage.from_array(
        arr,
        [ChannelSpec(n) for n in channel_names] if channel_names else None,
        name=path.stem,
        pixel_size_um=pixel_size_um,
    )
    image.meta["source"] = str(path)
    image.meta["layout"] = note

    if nuclei_channel is not None:
        index = image.resolve(nuclei_channel)[0]
        image.channels[index] = ChannelSpec(image.channels[index].name, "nuclei")
    elif image.n_channels == 3 and channel_names is None:
        # An unlabelled RGB file: collapse to the single most informative
        # channel, matching how BBBC038 samples are handled.
        image = _rgb_to_nuclei(image)
    elif not any(c.role == "nuclei" for c in image.channels):
        image.channels[0] = ChannelSpec(image.channels[0].name, "nuclei")

    return image


def _read_standard(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"could not read image {path}")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _read_tiff(path: Path) -> tuple[np.ndarray, str]:
    """Read a TIFF and normalise it to ``(H, W, C)``."""
    import tifffile

    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        return arr[:, :, None], "single-channel TIFF"
    if arr.ndim == 3:
        # Channels-first is the common convention for multiplexed TIFFs, and a
        # leading axis of at most 8 is far likelier to be channels than rows.
        if arr.shape[0] <= 8 and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
            return np.moveaxis(arr, 0, -1), f"channels-first TIFF ({arr.shape[0]} channels)"
        return arr, f"channels-last TIFF ({arr.shape[2]} channels)"
    if arr.ndim == 4:
        # A z-stack or time series: take a maximum intensity projection over the
        # leading axis, the standard 2-D reduction for nuclei counting.
        projected = arr.max(axis=0)
        if projected.shape[0] <= 8 and projected.shape[0] < min(projected.shape[1:]):
            projected = np.moveaxis(projected, 0, -1)
        return projected, "4-D TIFF, max-intensity projection over axis 0"
    raise ValueError(f"unsupported TIFF with shape {arr.shape}")


def _rgb_to_nuclei(image: MicroscopyImage) -> MicroscopyImage:
    """Reduce an unlabelled RGB image to one nuclei channel.

    Same rule as the BBBC038 loader: if one channel carries most of the variance
    it is the stain, otherwise the image is genuine colour (H&E) and luminance
    is the better proxy.
    """
    arr = image.data
    variances = arr.reshape(-1, arr.shape[2]).var(axis=0)
    if variances.max() - variances.min() > 0.15 * max(float(variances.max()), 1e-6):
        gray = arr[:, :, int(np.argmax(variances))]
        note = f"RGB reduced to channel {int(np.argmax(variances))} (highest variance)"
    else:
        gray = cv2.cvtColor((np.clip(arr, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        note = "RGB reduced to luminance"

    out = MicroscopyImage(
        gray[:, :, None],
        [ChannelSpec("nuclei", "nuclei")],
        pixel_size_um=image.pixel_size_um,
        name=image.name,
        meta=dict(image.meta),
    )
    out.meta["rgb_reduction"] = note
    return out


def list_images(folder: str | Path) -> list[Path]:
    """Every readable image file directly inside ``folder``."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def save_labels(labels: np.ndarray, path: str | Path) -> Path:
    """Save an instance map losslessly.

    16-bit PNG, so up to 65535 objects survive a round trip.  An 8-bit PNG would
    silently wrap around on a dense field, which is the sort of bug that only
    shows up in the counts weeks later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels)
    if labels.max() > 65535:
        raise ValueError(f"{labels.max()} objects exceeds the 16-bit PNG limit")
    cv2.imwrite(str(path), labels.astype(np.uint16))
    return path
