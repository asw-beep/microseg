"""Small shared helpers: seeding, devices, timing, logging."""

from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("microseg")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the package logger once, with a terse format."""
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False
    return LOGGER


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python/numpy/torch RNGs.

    Torch is imported lazily so the classical-CV half of the package stays
    usable without a torch install.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cudnn autotuning picks different algorithms run to run, which shows
            # up as small metric jitter; off by default for reproducibility.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_device(device: str = "auto"):
    """Resolve ``auto`` to cuda when available, else cpu."""
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class Timer:
    """Accumulating wall-clock timer, used for inference-time benchmarks."""

    def __init__(self) -> None:
        self.laps: list[float] = []

    @contextmanager
    def lap(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.laps.append(time.perf_counter() - start)

    @property
    def total(self) -> float:
        return float(sum(self.laps))

    @property
    def mean(self) -> float:
        return float(np.mean(self.laps)) if self.laps else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.laps)) if self.laps else 0.0

    def summary(self) -> dict[str, float]:
        if not self.laps:
            return {"n": 0, "mean_s": 0.0, "std_s": 0.0, "total_s": 0.0}
        laps = np.asarray(self.laps, dtype=float)
        return {
            "n": int(laps.size),
            "mean_s": float(laps.mean()),
            "std_s": float(laps.std()),
            "median_s": float(np.median(laps)),
            "p95_s": float(np.percentile(laps, 95)),
            "total_s": float(laps.sum()),
            "fps": float(1.0 / laps.mean()) if laps.mean() > 0 else float("inf"),
        }


def as_float01(arr: np.ndarray) -> np.ndarray:
    """Coerce an array to float32 in ``[0, 1]``, scaling by dtype range."""
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a float image in ``[0, 1]`` to uint8 for OpenCV routines."""
    return np.clip(np.asarray(arr) * 255.0, 0, 255).astype(np.uint8)
