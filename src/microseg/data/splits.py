"""Deterministic train/val/test splits.

Splits are computed from a seed and written to JSON.  Every downstream script
reads that file rather than re-deriving the split, so a model is never evaluated
on an image some earlier run trained on -- the quiet mistake that inflates
reported scores.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Splits:
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    seed: int = 42

    def __getitem__(self, key: str) -> list[str]:
        if key not in ("train", "val", "test"):
            raise KeyError(f"unknown split {key!r}; expected train|val|test")
        return getattr(self, key)

    @property
    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def make_splits(
    ids: list[str], val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 42
) -> Splits:
    """Shuffle ids with a seeded RNG and cut them into three disjoint sets."""
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must leave room for training data")

    shuffled = list(ids)
    np.random.default_rng(seed).shuffle(shuffled)

    n = len(shuffled)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    return Splits(
        train=shuffled[n_val + n_test :],
        val=shuffled[:n_val],
        test=shuffled[n_val : n_val + n_test],
        seed=seed,
    )


def load_splits(path: str | Path) -> Splits:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Splits(**data)
