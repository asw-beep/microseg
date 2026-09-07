"""Object-wise (instance) segmentation metrics.

This is the evaluation that matters for the biology.  A pipeline is useful when
it finds *each nucleus as its own object*, and pixel Dice cannot see the
difference between that and one merged blob.

The metrics implemented here are the BBBC038 / Data Science Bowl 2018 standard:

* predictions are matched to ground-truth objects one-to-one at an IoU
  threshold;
* every unmatched prediction is a false positive (an over-segmentation or a
  debris detection), every unmatched truth a false negative (a missed or merged
  nucleus);
* precision/recall/F1 follow, and averaging the score over IoU thresholds
  0.50 to 0.95 in steps of 0.05 gives the summary number, which rewards accurate
  outlines rather than just correct detection.

Matching is solved optimally with the Hungarian algorithm rather than greedily;
greedy matching over-counts when a large prediction overlaps two truths.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.5, 1.0, 0.05), 2))


def iou_matrix(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Pairwise IoU between every predicted and every true object.

    Returns an ``(n_true, n_pred)`` matrix.  Built from a 2-D histogram of label
    pairs, which is a single pass over the pixels regardless of object count --
    the naive per-pair loop is O(n_true * n_pred) mask operations and becomes the
    bottleneck on fields with hundreds of nuclei.
    """
    pred = np.asarray(pred, dtype=np.int64)
    true = np.asarray(true, dtype=np.int64)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: prediction {pred.shape} vs truth {true.shape}")

    n_true, n_pred = int(true.max()), int(pred.max())
    if n_true == 0 or n_pred == 0:
        return np.zeros((n_true, n_pred), dtype=np.float64)

    # Row 0 / column 0 are background and are dropped after the histogram.
    overlap = np.bincount(
        (true.ravel() * (n_pred + 1) + pred.ravel()),
        minlength=(n_true + 1) * (n_pred + 1),
    ).reshape(n_true + 1, n_pred + 1)

    true_areas = overlap.sum(axis=1, keepdims=True)
    pred_areas = overlap.sum(axis=0, keepdims=True)
    intersection = overlap[1:, 1:]
    union = true_areas[1:] + pred_areas[:, 1:] - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1), 0.0)


def match_instances(
    pred: np.ndarray, true: np.ndarray, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimally match objects at an IoU threshold.

    Returns ``(matched_pairs, unmatched_true_ids, unmatched_pred_ids)`` with
    1-based label ids.
    """
    ious = iou_matrix(pred, true)
    n_true, n_pred = ious.shape
    if n_true == 0 or n_pred == 0:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.arange(1, n_true + 1),
            np.arange(1, n_pred + 1),
        )

    # Maximise total IoU, then discard pairs that fall below the threshold.  The
    # assignment is computed on the full matrix so a marginal pair cannot steal a
    # partner from a better one.
    rows, cols = linear_sum_assignment(-ious)
    keep = ious[rows, cols] >= threshold
    rows, cols = rows[keep], cols[keep]

    pairs = np.stack([rows + 1, cols + 1], axis=1) if rows.size else np.zeros((0, 2), np.int64)
    unmatched_true = np.setdiff1d(np.arange(1, n_true + 1), pairs[:, 0] if pairs.size else [])
    unmatched_pred = np.setdiff1d(np.arange(1, n_pred + 1), pairs[:, 1] if pairs.size else [])
    return pairs, unmatched_true, unmatched_pred


def average_precision(
    pred: np.ndarray, true: np.ndarray, thresholds=DEFAULT_THRESHOLDS
) -> dict[str, float]:
    """The DSB2018 score: mean of ``TP / (TP + FP + FN)`` over IoU thresholds.

    Note this is a detection-quality ratio, not the area under a
    precision-recall curve -- the competition called it average precision and the
    name has stuck, so it is spelled out here to avoid confusion.
    """
    ious = iou_matrix(pred, true)
    n_true, n_pred = ious.shape

    per_threshold: dict[str, float] = {}
    scores: list[float] = []
    for threshold in thresholds:
        if n_true == 0 and n_pred == 0:
            score = 1.0
        elif n_true == 0 or n_pred == 0:
            score = 0.0
        else:
            rows, cols = linear_sum_assignment(-ious)
            tp = int(np.count_nonzero(ious[rows, cols] >= threshold))
            fp, fn = n_pred - tp, n_true - tp
            score = tp / max(tp + fp + fn, 1)
        per_threshold[f"ap@{threshold:.2f}"] = float(score)
        scores.append(score)

    per_threshold["ap_mean"] = float(np.mean(scores)) if scores else 0.0
    return per_threshold


def count_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """How close the object count is -- the number a biologist reads first."""
    n_pred, n_true = int(np.asarray(pred).max()), int(np.asarray(true).max())
    error = n_pred - n_true
    return {
        "count_pred": float(n_pred),
        "count_true": float(n_true),
        "count_error": float(error),
        "count_abs_error": float(abs(error)),
        "count_rel_error": float(abs(error) / n_true) if n_true else 0.0,
    }


def instance_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    threshold: float = 0.5,
    thresholds=DEFAULT_THRESHOLDS,
) -> dict[str, float]:
    """Full instance report for one image at the primary threshold, plus AP."""
    pairs, unmatched_true, unmatched_pred = match_instances(pred, true, threshold)
    tp = int(len(pairs))
    fn = int(len(unmatched_true))
    fp = int(len(unmatched_pred))

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Mean IoU of the matched pairs: how well the outlines agree, given that the
    # object was found at all.  Separates detection failures from sloppy masks.
    if tp:
        ious = iou_matrix(pred, true)
        matched_iou = float(np.mean([ious[t - 1, p - 1] for t, p in pairs]))
    else:
        matched_iou = 0.0

    out = {
        f"tp@{threshold:.2f}": float(tp),
        f"fp@{threshold:.2f}": float(fp),
        f"fn@{threshold:.2f}": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": matched_iou,
    }
    out.update(average_precision(pred, true, thresholds))
    out.update(count_metrics(pred, true))
    return out
