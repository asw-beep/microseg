"""Pixel-wise segmentation metrics.

These answer "did we label the right pixels as nucleus".  They are necessary but
not sufficient: a model that merges every touching pair of nuclei into one blob
can still score a Dice of 0.95, because only the thin boundary pixels are wrong.
That is exactly why :mod:`microseg.metrics.instance` exists, and why both are
reported side by side throughout this project.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def confusion_counts(pred: np.ndarray, true: np.ndarray) -> dict[str, int]:
    """True/false positive/negative pixel counts for two binary masks."""
    pred = np.asarray(pred).astype(bool)
    true = np.asarray(true).astype(bool)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: prediction {pred.shape} vs truth {true.shape}")
    return {
        "tp": int(np.count_nonzero(pred & true)),
        "fp": int(np.count_nonzero(pred & ~true)),
        "fn": int(np.count_nonzero(~pred & true)),
        "tn": int(np.count_nonzero(~pred & ~true)),
    }


def dice_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Dice / F1 over pixels: ``2|A n B| / (|A| + |B|)``.

    Two empty masks score 1.0 -- an image with no nuclei, correctly predicted to
    have none, is a perfect result rather than an undefined one.
    """
    c = confusion_counts(pred, true)
    denom = 2 * c["tp"] + c["fp"] + c["fn"]
    if denom == 0:
        return 1.0
    return float(2 * c["tp"] / denom)


def iou_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Jaccard index: ``|A n B| / |A u B|``.  Always <= Dice."""
    c = confusion_counts(pred, true)
    denom = c["tp"] + c["fp"] + c["fn"]
    if denom == 0:
        return 1.0
    return float(c["tp"] / denom)


def semantic_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Dice, IoU, precision, recall, specificity and accuracy for one image."""
    c = confusion_counts(pred, true)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    dice_denom = 2 * tp + fp + fn
    iou_denom = tp + fp + fn
    return {
        "dice": float(2 * tp / dice_denom) if dice_denom else 1.0,
        "iou": float(tp / iou_denom) if iou_denom else 1.0,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 1.0,
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
    }


def per_class_dice(pred: np.ndarray, true: np.ndarray, n_classes: int = 3) -> dict[str, float]:
    """Dice for each class of a multi-class label map.

    Reported per class because the boundary class is the informative one: its
    Dice predicts how well instances will separate, while the interior class
    saturates early in training and stops discriminating between models.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    names = {0: "background", 1: "interior", 2: "boundary"}
    out: dict[str, float] = {}
    for c in range(n_classes):
        key = names.get(c, f"class{c}")
        out[f"dice_{key}"] = dice_score(pred == c, true == c)
    return out


def aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across images, ignoring non-numeric entries.

    Averaging per-image scores (rather than pooling all pixels) weights every
    field of view equally, which is what you want when image sizes vary by 10x
    as they do in BBBC038.
    """
    if not records:
        return {}
    keys = [k for k, v in records[0].items() if isinstance(v, (int, float, np.floating))]
    return {k: float(np.mean([r[k] for r in records if k in r])) for k in keys}
