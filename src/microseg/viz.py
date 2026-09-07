"""Visualisation helpers.

Deliberately opinionated about what to draw.  For segmentation the useful
picture is not a pretty overlay but an *error* picture: where the prediction
split one nucleus into two, merged two into one, or missed a dim one entirely.
:func:`comparison_figure` therefore always shows ground truth and prediction
side by side with a colour-coded difference map.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs in Docker and in CI

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

from microseg.utils import ensure_dir


def label_to_rgb(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Colour each instance randomly, background black.

    Random colours (rather than a sequential colormap) are the right choice for
    instance maps: neighbouring ids get visibly different colours, so a merge or
    a split is obvious at a glance.
    """
    labels = np.asarray(labels, dtype=np.int32)
    n = int(labels.max())
    rng = np.random.default_rng(seed)
    # Bias toward bright, saturated colours so objects read clearly on black.
    colors = rng.uniform(0.35, 1.0, size=(n + 1, 3))
    colors[0] = 0.0
    return colors[labels].astype(np.float32)


def overlay_boundaries(
    image: np.ndarray, labels: np.ndarray, color=(1.0, 0.25, 0.1), width: int = 1
) -> np.ndarray:
    """Draw instance outlines on a grayscale image.

    Outlines rather than filled masks, because a filled overlay hides exactly
    the pixels a reviewer needs to check.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        rgb = np.repeat(np.clip(image, 0, 1)[:, :, None], 3, axis=2)
    else:
        rgb = np.clip(image[:, :, :3], 0, 1).astype(np.float32)

    boundaries = find_boundaries(np.asarray(labels, dtype=np.int32), mode="outer")
    if width > 1:
        import cv2

        k = 2 * int(width) - 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        boundaries = cv2.dilate(boundaries.astype(np.uint8), se).astype(bool)

    out = rgb.copy()
    out[boundaries] = color
    return out


def error_map(pred_mask: np.ndarray, true_mask: np.ndarray) -> np.ndarray:
    """Colour-coded pixel agreement: green TP, red FP, blue FN, black TN."""
    pred = np.asarray(pred_mask).astype(bool)
    true = np.asarray(true_mask).astype(bool)
    out = np.zeros((*pred.shape, 3), dtype=np.float32)
    out[pred & true] = (0.15, 0.8, 0.3)
    out[pred & ~true] = (0.9, 0.2, 0.2)
    out[~pred & true] = (0.2, 0.45, 0.95)
    return out


def _show(ax, image, title: str, cmap: str | None = None) -> None:
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def preprocessing_figure(stages: dict[str, np.ndarray], save_path=None):
    """Grid of preprocessing intermediates, in pipeline order.

    The QC figure to look at when segmentation behaves oddly -- most such
    failures are visible here as an over-corrected background or a CLAHE step
    that amplified noise.
    """
    if not stages:
        raise ValueError("no preprocessing stages captured; pass collect_stages=True")

    n = len(stages)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)

    for ax, (name, image) in zip(axes.ravel(), stages.items()):
        _show(ax, image, name.replace("_", " "), cmap="gray")
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle("Preprocessing stages", fontsize=11)
    fig.tight_layout()
    return _finish(fig, save_path)


def comparison_figure(
    image: np.ndarray,
    pred_labels: np.ndarray,
    true_labels: np.ndarray | None = None,
    title: str = "",
    save_path=None,
):
    """Image / prediction / ground truth / error, in one row."""
    panels = 2 if true_labels is None else 4
    fig, axes = plt.subplots(1, panels, figsize=(4.2 * panels, 4.4))

    _show(axes[0], np.asarray(image), "input", cmap="gray")
    _show(
        axes[1],
        overlay_boundaries(image, pred_labels),
        f"prediction ({int(np.asarray(pred_labels).max())} objects)",
    )
    if true_labels is not None:
        _show(
            axes[2],
            overlay_boundaries(image, true_labels, color=(0.2, 0.9, 0.4)),
            f"ground truth ({int(np.asarray(true_labels).max())} objects)",
        )
        _show(
            axes[3],
            error_map(np.asarray(pred_labels) > 0, np.asarray(true_labels) > 0),
            "green TP / red FP / blue FN",
        )

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _finish(fig, save_path)


def instance_figure(image: np.ndarray, labels: np.ndarray, title: str = "", save_path=None):
    """Input, coloured instances, and outlines over the input."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    _show(axes[0], np.asarray(image), "input", cmap="gray")
    _show(axes[1], label_to_rgb(labels), f"{int(np.asarray(labels).max())} instances")
    _show(axes[2], overlay_boundaries(image, labels), "outlines")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _finish(fig, save_path)


def probability_figure(probs: np.ndarray, save_path=None):
    """The three predicted class maps.

    Worth inspecting directly: a weak boundary channel is the single best
    predictor that instances will come out merged.
    """
    names = ["P(background)", "P(interior)", "P(boundary)"]
    n = min(len(names), probs.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.2))
    axes = np.atleast_1d(axes)
    for i in range(n):
        im = axes[i].imshow(probs[i], cmap="magma", vmin=0, vmax=1)
        axes[i].set_title(names[i], fontsize=9)
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046)
    fig.tight_layout()
    return _finish(fig, save_path)


def training_curves(history: dict[str, list[float]], save_path=None):
    """Loss and validation metric curves from a training run."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.get("train_loss", []), label="train")
    if history.get("val_loss"):
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for key in ("val_dice", "val_dice_boundary", "val_ap"):
        if history.get(key):
            axes[1].plot(history[key], label=key.replace("val_", ""))
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return _finish(fig, save_path)


def feature_distributions(df, columns=None, save_path=None):
    """Histograms of the headline morphology features across a population."""
    columns = columns or ["area_px", "circularity", "eccentricity", "solidity"]
    columns = [c for c in columns if c in df.columns]
    if not columns:
        raise ValueError("none of the requested feature columns are present")

    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3.4))
    axes = np.atleast_1d(axes)
    for ax, column in zip(axes, columns):
        values = df[column].to_numpy(dtype=float)
        ax.hist(values, bins=30, color="#4C72B0", edgecolor="white")
        ax.set_title(f"{column}\nmedian={np.median(values):.2f}", fontsize=9)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return _finish(fig, save_path)


def _finish(fig, save_path):
    """Save if a path was given, then return the figure."""
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig
