"""Reproducible U-Net training.

Everything needed to reproduce a run is written into the run directory: the
resolved config, the split file, the full epoch history, and checkpoints that
embed their own config.  Rerunning ``python -m microseg.train --config <that
config>`` reproduces the result, because the split is seeded and stored rather
than recomputed.

Model selection uses **instance AP on the validation set, not pixel Dice**.
This is the important choice in this file.  Dice saturates early and then barely
moves, while the thing that actually improves -- separating touching nuclei --
shows up mostly in the instance metric.  Measured on the shipped BBBC038 run,
over the last 10 epochs validation Dice spanned 2.1% of its mean and instance AP
spanned 13.3%, so AP carries several times more signal about which checkpoint to
keep.  (The two peaked at the same epoch in that run; this is a choice about
which metric to trust, not a claim that it changed the outcome there.)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from microseg.config import ExperimentConfig, load_config
from microseg.data.bbbc038 import list_samples, prepare_cache
from microseg.data.dataset import build_dataloaders
from microseg.data.splits import load_splits, make_splits
from microseg.instance import probabilities_to_instances
from microseg.metrics.instance import average_precision
from microseg.metrics.semantic import dice_score, per_class_dice
from microseg.models import build_loss, build_model, count_parameters
from microseg.utils import LOGGER, ensure_dir, resolve_device, set_seed, setup_logging


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True
        )
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}; expected adamw|adam|sgd")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)
    if cfg.scheduler in ("none", None):
        return None
    raise ValueError(f"unknown scheduler {cfg.scheduler!r}; expected cosine|plateau|none")


def train_one_epoch(model, loader, criterion, optimizer, device, cfg, scaler=None) -> float:
    model.train()
    losses: list[float] = []

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        distances = batch["distance"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(images), targets, distances)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), targets, distances)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        losses.append(float(loss.detach()))
        if cfg.log_every and step % cfg.log_every == 0:
            LOGGER.info("    step %4d/%d  loss %.4f", step, len(loader), losses[-1])

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate(model, loader, criterion, device, cfg) -> dict[str, float]:
    """Loss, per-class Dice, distance MAE and instance AP over the validation set.

    AP is computed with whatever seeding the model supports -- the distance map
    when it has a distance head, the boundary class otherwise -- so the number
    driving checkpoint selection is the one the deployed pipeline will produce.
    """
    from microseg.models.losses import split_output

    model.eval()
    losses: list[float] = []
    dice_records: list[dict[str, float]] = []
    foreground_dice: list[float] = []
    distance_mae: list[float] = []
    aps: list[float] = []

    n_classes = int(getattr(model, "n_classes", getattr(model, "out_channels", 3)))

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        distances = batch["distance"].to(device, non_blocking=True)

        logits = model(images)
        losses.append(float(criterion(logits, targets, distances)))

        class_logits, distance_logits = split_output(logits, n_classes)
        probs = torch.softmax(class_logits, dim=1).cpu().numpy()

        predicted_distance = None
        if distance_logits is not None:
            predicted_distance = torch.sigmoid(distance_logits).cpu().numpy()[:, 0]
            foreground = (targets > 0).cpu().numpy()
            if foreground.any():
                error = np.abs(predicted_distance - distances.cpu().numpy()[:, 0])
                distance_mae.append(float(error[foreground].mean()))

        predictions = probs.argmax(axis=1)
        truth = targets.cpu().numpy()
        instances_true = batch["instances"].numpy()

        for i in range(predictions.shape[0]):
            dice_records.append(per_class_dice(predictions[i], truth[i]))
            foreground_dice.append(dice_score(predictions[i] > 0, truth[i] > 0))
            instances_pred = probabilities_to_instances(
                probs[i],
                cfg.postprocess,
                distance=None if predicted_distance is None else predicted_distance[i],
            )
            aps.append(average_precision(instances_pred, instances_true[i])["ap_mean"])

    metrics = {
        "val_loss": float(np.mean(losses)) if losses else 0.0,
        "val_dice": float(np.mean(foreground_dice)) if foreground_dice else 0.0,
        "val_ap": float(np.mean(aps)) if aps else 0.0,
    }
    if distance_mae:
        metrics["val_distance_mae"] = float(np.mean(distance_mae))
    if dice_records:
        for key in dice_records[0]:
            metrics[f"val_{key}"] = float(np.mean([r[key] for r in dice_records]))
    return metrics


def prepare_data(cfg: ExperimentConfig) -> tuple[Path, object]:
    """Ensure the sample cache and split file exist, then return both."""
    cache_dir = Path(cfg.data.processed_root) / "cache"
    split_path = Path(cfg.data.processed_root) / f"splits_seed{cfg.data.split_seed}.json"

    ids = list_samples(cfg.data.root)
    if cfg.data.limit:
        ids = ids[: int(cfg.data.limit)]

    missing = [i for i in ids if not (cache_dir / f"{i}.npz").exists()]
    if missing:
        LOGGER.info("caching %d sample(s) to %s", len(missing), cache_dir)
        prepare_cache(cfg.data.root, cache_dir, limit=cfg.data.limit)

    if split_path.exists():
        splits = load_splits(split_path)
        # A stale split from a larger run would reference uncached ids.
        known = set(ids)
        if not set(splits.train) <= known:
            LOGGER.info("existing split does not match current id set; regenerating")
            splits = make_splits(ids, cfg.data.val_fraction, cfg.data.test_fraction, cfg.data.split_seed)
            splits.save(split_path)
    else:
        splits = make_splits(ids, cfg.data.val_fraction, cfg.data.test_fraction, cfg.data.split_seed)
        splits.save(split_path)

    LOGGER.info("splits: %s", splits.counts)
    return cache_dir, splits


def train(cfg: ExperimentConfig) -> dict:
    """Run a full training job and return its history."""
    setup_logging()
    set_seed(cfg.seed)

    run_dir = ensure_dir(cfg.run_dir)
    cfg.save(run_dir / "config.yaml")

    device = resolve_device(cfg.train.device)
    LOGGER.info("device: %s", device)

    cache_dir, splits = prepare_data(cfg)
    splits.save(run_dir / "splits.json")
    loaders = build_dataloaders(cache_dir, splits, cfg.data, cfg.preprocess, cfg.train)
    if "train" not in loaders:
        raise RuntimeError("no training data; check data.root and data.limit")

    model = build_model(cfg.model).to(device)
    LOGGER.info("model: %s params", f"{count_parameters(model):,}")

    criterion = build_loss(
        cfg.loss, cfg.model.out_channels, cfg.model.distance_head
    ).to(device)
    optimizer = build_optimizer(model, cfg.train)
    scheduler = build_scheduler(optimizer, cfg.train)
    # GradScaler is CUDA-only; on CPU we simply train in float32.
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history: dict[str, list] = {}
    best_score = -np.inf
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(cfg.train.epochs):
        started = time.perf_counter()
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, cfg.train, scaler
        )

        metrics = {"train_loss": train_loss}
        if "val" in loaders:
            metrics.update(validate(model, loaders["val"], criterion, device, cfg))

        if scheduler is not None:
            if cfg.train.scheduler == "plateau":
                scheduler.step(metrics.get("val_ap", 0.0))
            else:
                scheduler.step()

        metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        metrics["epoch_s"] = time.perf_counter() - started
        for key, value in metrics.items():
            history.setdefault(key, []).append(value)

        # The distance column only appears for a model that has the head, so
        # the log line stays readable for the plain 3-class runs.
        distance_note = (
            f"  dist_mae {metrics['val_distance_mae']:.4f}"
            if "val_distance_mae" in metrics
            else ""
        )
        LOGGER.info(
            "epoch %3d/%d  train %.4f  val %.4f  dice %.4f  boundary %.4f%s  AP %.4f  (%.1fs)",
            epoch + 1,
            cfg.train.epochs,
            train_loss,
            metrics.get("val_loss", float("nan")),
            metrics.get("val_dice", float("nan")),
            metrics.get("val_dice_boundary", float("nan")),
            distance_note,
            metrics.get("val_ap", float("nan")),
            metrics["epoch_s"],
        )

        # Selection metric: instance AP, falling back to loss when there is no
        # validation split at all (a smoke-test config).
        score = metrics.get("val_ap", -metrics["train_loss"])
        if score > best_score:
            best_score, best_epoch = score, epoch
            epochs_without_improvement = 0
            save_checkpoint(run_dir / "best.pt", model, cfg, epoch, metrics)
            LOGGER.info("    new best (AP %.4f) -> best.pt", score)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.train.early_stopping_patience:
                LOGGER.info("early stopping: no improvement for %d epochs", epochs_without_improvement)
                break

    save_checkpoint(run_dir / "last.pt", model, cfg, epoch, metrics)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    try:
        from microseg.viz import training_curves

        training_curves(history, run_dir / "training_curves.png")
    except Exception as exc:  # plotting must never fail a training run
        LOGGER.warning("could not write training curves: %s", exc)

    LOGGER.info("done: best AP %.4f at epoch %d -> %s", best_score, best_epoch + 1, run_dir)
    return history


def save_checkpoint(path: Path, model, cfg: ExperimentConfig, epoch: int, metrics: dict) -> None:
    """Save weights together with the config that produced them."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        str(path),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the U-Net nuclei segmentation model")
    parser.add_argument("--config", type=str, default="configs/unet_bbbc038.yaml")
    parser.add_argument("--name", type=str, default=None, help="override the run name")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap the number of images")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.name:
        cfg.name = args.name
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.limit is not None:
        cfg.data.limit = args.limit
    if args.device:
        cfg.train.device = args.device

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
