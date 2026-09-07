"""Model inference on full-size microscopy fields.

Training runs on 256 px tiles, but a real field is whatever the microscope
produced -- BBBC038 alone ranges from 256x256 to 1024x1024.  Two details make
full-field inference correct rather than approximately correct:

* **Padding to the network's size divisor.**  A depth-4 U-Net pools four times,
  so any dimension not divisible by 16 produces skip connections that do not
  align.  The image is reflect-padded up, then the prediction is cropped back.
* **Optional test-time augmentation.**  Averaging predictions over the eight
  dihedral transforms costs 8x compute and reliably buys a little boundary
  accuracy -- worth it for a final evaluation, not for interactive use, so it is
  off by default.

A model with a distance head returns one extra channel that must be passed
through a sigmoid, never the softmax the class channels take.  Keeping that
split in one place -- :func:`predict_maps` -- is why the rest of the codebase
never indexes the raw output tensor directly.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from microseg.data.transforms import pad_to_multiple


class Prediction(NamedTuple):
    """What a forward pass yields, already split by head.

    ``distance`` is ``None`` for a model without a distance head, which is the
    signal every consumer uses to pick a seeding strategy.
    """

    probs: np.ndarray  # (n_classes, H, W), softmax
    distance: np.ndarray | None  # (H, W) in 0..1, sigmoid


@torch.no_grad()
def predict_maps(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device | str = "auto",
    tta: bool = False,
) -> Prediction:
    """Preprocessed 2-D image (or ``(H, W, C)`` stack) -> class probabilities and distance."""
    from microseg.utils import resolve_device

    model.eval()
    device = resolve_device(device) if isinstance(device, str) else torch.device(device)
    model.to(device)

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    original_shape = arr.shape[:2]

    divisor = getattr(model, "size_divisor", 16)
    padded, _ = pad_to_multiple(arr, divisor)
    if padded.ndim == 2:
        padded = padded[:, :, None]

    tensor = torch.from_numpy(padded.transpose(2, 0, 1)[None]).to(device)
    n_classes = int(getattr(model, "n_classes", getattr(model, "out_channels", 3)))

    if not tta:
        maps = _activate(model(tensor), n_classes)[0].cpu().numpy()
    else:
        maps = _dihedral_tta(model, tensor, n_classes)

    h, w = original_shape
    maps = maps[:, :h, :w].astype(np.float32)
    if maps.shape[0] == n_classes:
        return Prediction(maps, None)
    return Prediction(maps[:n_classes], maps[n_classes])


def _activate(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Softmax the class channels, sigmoid the distance channel, keep them stacked.

    Both activations happen before TTA averaging, not after: the mean of eight
    softmaxes is a probability; the softmax of eight averaged logits is a
    different quantity, and it is the subtly wrong one.
    """
    from microseg.models.losses import split_output

    class_logits, distance_logits = split_output(logits, n_classes)
    probs = torch.softmax(class_logits, dim=1)
    if distance_logits is None:
        return probs
    return torch.cat([probs, torch.sigmoid(distance_logits)], dim=1)


def _dihedral_tta(model: torch.nn.Module, tensor: torch.Tensor, n_classes: int) -> np.ndarray:
    """Average activated outputs over the 8 flip/rotate symmetries.

    Each prediction is mapped back into the original frame before averaging --
    forgetting the inverse transform is the classic TTA bug, and it produces a
    blurred mess rather than an error.

    The distance channel rides along unchanged: it is a scalar field, so a
    rotation of the input rotates it exactly as it rotates a class map.  A flow
    field would not be -- its vectors would need rotating too -- which is worth
    knowing before anyone swaps this head for one.
    """
    accumulator = None
    for k in range(4):
        for flip in (False, True):
            view = torch.rot90(tensor, k, dims=(2, 3))
            if flip:
                view = torch.flip(view, dims=(3,))

            maps = _activate(model(view), n_classes)

            if flip:
                maps = torch.flip(maps, dims=(3,))
            maps = torch.rot90(maps, -k, dims=(2, 3))
            accumulator = maps if accumulator is None else accumulator + maps
    return (accumulator / 8.0)[0].cpu().numpy()


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device | str = "auto",
    tta: bool = False,
) -> np.ndarray:
    """Class probabilities only.  Kept because most call sites want only these."""
    return predict_maps(model, image, device, tta).probs


def load_checkpoint(path, device: torch.device | str = "auto"):
    """Load a training checkpoint and rebuild its model.

    Checkpoints carry their own :class:`ModelConfig`, so a saved model can always
    be reconstructed without the caller knowing the architecture it was trained
    with.
    """
    from microseg.config import config_from_dict
    from microseg.models import build_model
    from microseg.utils import resolve_device

    device = resolve_device(device) if isinstance(device, str) else torch.device(device)
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    cfg = config_from_dict(checkpoint["config"])
    model = build_model(cfg.model)
    model.load_state_dict(checkpoint["model_state"])
    model.to(torch.device(device)).eval()
    return model, cfg, checkpoint
