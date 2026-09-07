"""Loss functions for 3-class nuclei segmentation.

The combination used here is weighted cross-entropy + soft Dice, which is the
standard pairing for medical segmentation and for a concrete reason:

* **Cross-entropy** is a per-pixel likelihood.  It gives dense, well-conditioned
  gradients everywhere, but it is dominated by the majority class -- background
  is ~80% of pixels and boundary ~5%, so a model that predicts "background
  everywhere" already scores well.
* **Soft Dice** optimises region overlap directly, which is scale-invariant with
  respect to class frequency.  On its own it is unstable early in training
  (near-empty predictions give vanishing gradients) and ignores confidence.

Together, CE provides the stable learning signal and Dice supplies the class
balance.  The explicit ``class_weights`` on top upweight the boundary class,
because separating touching nuclei is worth more than a marginal gain in mask
Dice for this task.

**The distance term.**  When the model carries a distance head, an L1 regression
loss on the normalised distance-to-edge map is added.  L1 rather than MSE
because the distance target is bounded in 0..1 and its interesting structure is
the *ridge* near 1.0: MSE weights the many easy mid-range pixels quadratically
and flattens exactly the peaks the watershed seeds from.

That term is masked to the foreground.  Regressing distance on background would
spend most of the loss on pixels whose answer is a constant 0, and it is not
needed: seeds are intersected with the predicted foreground mask before they
ever reach the watershed, so an unconstrained background prediction cannot
invent an object.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from microseg.config import LossConfig


class SoftDiceLoss(nn.Module):
    """Multi-class soft Dice, averaged over classes.

    "Soft" because it uses the predicted probabilities rather than a hard
    argmax, which keeps it differentiable.
    """

    def __init__(self, smooth: float = 1.0, ignore_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target.long(), n_classes).permute(0, 3, 1, 2).float()

        start = 1 if self.ignore_background and n_classes > 1 else 0
        probs, onehot = probs[:, start:], onehot[:, start:]

        dims = (0, 2, 3)  # sum over batch and space, keep classes separate
        intersection = (probs * onehot).sum(dims)
        cardinality = probs.sum(dims) + onehot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """``ce_weight * weighted CE + dice_weight * soft Dice + distance_weight * L1``.

    ``forward`` accepts the model output whether or not it carries a distance
    channel.  The extra channel is detected from the shape against ``n_classes``,
    so one loss object serves both architectures and a config that sets
    ``distance_weight`` without a distance head is simply ignored rather than
    being a crash at epoch 1.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: list[float] | None = None,
        distance_weight: float = 0.0,
        n_classes: int = 3,
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.distance_weight = float(distance_weight)
        self.n_classes = int(n_classes)
        self.dice = SoftDiceLoss()
        if class_weights is not None:
            # Registered as a buffer so it follows the model across .to(device).
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        distance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = target.long()
        class_logits, distance_logits = split_output(logits, self.n_classes)

        loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if self.ce_weight:
            weight = None
            if self.class_weights is not None:
                weight = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss = loss + self.ce_weight * F.cross_entropy(class_logits, target, weight=weight)
        if self.dice_weight:
            loss = loss + self.dice_weight * self.dice(class_logits, target)
        if self.distance_weight and distance_logits is not None and distance is not None:
            loss = loss + self.distance_weight * masked_l1(
                torch.sigmoid(distance_logits), distance, target > 0
            )
        return loss


def split_output(logits: torch.Tensor, n_classes: int):
    """``(B, C, H, W)`` -> ``(class logits, distance logits or None)``."""
    if logits.shape[1] == n_classes:
        return logits, None
    if logits.shape[1] == n_classes + 1:
        return logits[:, :n_classes], logits[:, n_classes : n_classes + 1]
    raise ValueError(
        f"model output has {logits.shape[1]} channels; expected {n_classes} "
        f"or {n_classes + 1} (with a distance head)"
    )


def masked_l1(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean absolute error over ``mask`` only.

    Returns 0 for an all-background tile rather than NaN.  Empty tiles are real
    -- BBBC038 has fields where a random 256 px crop lands on nothing -- and a
    single NaN would poison the epoch average and every gradient after it.
    """
    if target.dim() == prediction.dim() - 1:
        target = target.unsqueeze(1)
    mask = mask.unsqueeze(1) if mask.dim() == prediction.dim() - 1 else mask
    mask = mask.to(prediction.dtype)

    denominator = mask.sum()
    if denominator.item() == 0:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    return ((prediction - target.to(prediction.dtype)).abs() * mask).sum() / denominator


def build_loss(
    cfg: LossConfig, n_classes: int = 3, distance_head: bool = False
) -> CombinedLoss:
    """Build the configured loss, validating the class-weight length."""
    weights = cfg.class_weights
    if weights is not None and len(weights) != n_classes:
        raise ValueError(
            f"loss.class_weights has {len(weights)} entries but the model has "
            f"{n_classes} output classes"
        )
    return CombinedLoss(
        cfg.ce_weight,
        cfg.dice_weight,
        weights,
        distance_weight=cfg.distance_weight if distance_head else 0.0,
        n_classes=n_classes,
    )
