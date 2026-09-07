"""Experiment configuration.

Plain dataclasses plus a YAML loader.  Every run writes its resolved config next
to its checkpoints, so an experiment can be reproduced from the output directory
alone.  Deliberately lighter than Hydra/OmegaConf: the config surface here is
small enough that a nested dataclass tree is easier to read than a framework.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PreprocessConfig:
    """Microscopy preprocessing chain, applied in the order listed below.

    Every stage can be disabled with ``None``/``"none"`` so the effect of each
    can be ablated from a config file without touching code.
    """

    # BBBC038 mixes fluorescence (bright objects on dark) with H&E brightfield
    # (dark objects on light).  Detecting and inverting the latter is what makes
    # one threshold/model work across the whole set.
    auto_invert: bool = True
    hot_pixel_correction: bool = True
    hot_pixel_threshold: float = 5.0  # in MADs above the local median

    illumination: str | None = "morphological"  # morphological|polynomial|gaussian|none
    illumination_radius: int = 50

    denoise: str | None = "median"  # gaussian|median|bilateral|nlm|tv|none
    denoise_strength: float = 1.0

    contrast: str | None = "clahe"  # clahe|gamma|stretch|none
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    gamma: float = 1.0

    normalize: str = "percentile"  # percentile|zscore|minmax|none
    percentile_low: float = 1.0
    percentile_high: float = 99.0


@dataclass
class DataConfig:
    # Which reader interprets ``root``.  See :mod:`microseg.data.registry`.
    dataset: str = "bbbc038"  # bbbc038|bbbc039|monuseg
    root: str = "data/raw/stage1_train"
    processed_root: str = "data/processed"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 42
    tile_size: int = 256
    # Held-out test tiles are centre-cropped rather than randomly cropped so the
    # evaluation set is deterministic across runs.
    augment: bool = True
    limit: int | None = None  # cap the number of images, for smoke tests
    # Width of the boundary band in the 3-class target.  Wider separates crowded
    # nuclei more reliably but eats into the interior, biasing areas low.
    boundary_width: int = 2


@dataclass
class ModelConfig:
    name: str = "unet"
    in_channels: int = 1
    # 3 classes: background / nucleus interior / touching boundary.  The boundary
    # class is what turns a semantic map into separable instances.
    out_channels: int = 3
    base_filters: int = 32
    depth: int = 4
    dropout: float = 0.1
    norm: str = "batch"  # batch|instance|none
    bilinear_upsample: bool = False
    # Adds a 1-channel regression output predicting the normalised distance to
    # the object edge.  It is the seeding signal that survives when two nuclei
    # overlap and there is no boundary band between them to predict.
    distance_head: bool = False


@dataclass
class LossConfig:
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    # Boundary pixels are a small minority; upweighting them is what stops the
    # network collapsing touching nuclei into one blob.
    class_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 3.0])
    # Weight on the distance-regression term.  Ignored when the model has no
    # distance head.  1.0 puts it on par with the CE term at convergence.
    distance_weight: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"  # cosine|plateau|none
    num_workers: int = 0  # 0 is the safe default on Windows
    device: str = "auto"  # auto|cpu|cuda
    amp: bool = True
    early_stopping_patience: int = 10
    grad_clip: float = 1.0
    log_every: int = 20
    seed: int = 42


@dataclass
class PostprocessConfig:
    """Semantic probability maps -> instance labels."""

    foreground_threshold: float = 0.5
    seed_threshold: float = 0.5  # on the interior (non-boundary) probability
    # Prefer the predicted distance map for seeding when the model has a
    # distance head.  Set False to force the boundary-class seeding, which is
    # what makes the two mechanisms comparable on one checkpoint.
    use_distance_seeds: bool = True
    # A seed is a pixel closer to the object centre than this, on the 0..1
    # normalised distance scale.  0.5 is the mid-radius contour: high enough to
    # separate two overlapping nuclei, low enough to survive on small ones.
    distance_seed_threshold: float = 0.5
    min_object_area: int = 15
    min_seed_area: int = 4
    fill_holes: bool = True
    watershed_line: bool = False
    # Distance-transform peaks are the fallback seeding strategy when a model has
    # no boundary head (e.g. the classical baseline).
    peak_min_distance: int = 5


@dataclass
class AnalysisConfig:
    compute_texture: bool = True
    glcm_distances: list[int] = field(default_factory=lambda: [1, 3])
    glcm_angles_deg: list[float] = field(default_factory=lambda: [0.0, 45.0, 90.0, 135.0])
    glcm_levels: int = 32
    # Objects touching the field of view are truncated, so their morphology is
    # biased; flag them and let the caller decide.
    exclude_border_objects: bool = False


@dataclass
class ExperimentConfig:
    name: str = "unet_bbbc038"
    output_dir: str = "outputs"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dict, rejecting unknown keys.

    Silently ignoring a typo'd config key is the classic way to spend an hour
    wondering why a flag had no effect, so unknown keys are a hard error.
    """
    if not is_dataclass(cls):
        return data
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"unknown config key(s) for {cls.__name__}: {sorted(unknown)}; "
            f"valid keys are {sorted(known)}"
        )
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        f = known[key]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[key] = _from_dict(f.type, value)
        elif isinstance(value, dict) and dataclasses.is_dataclass(
            getattr(f, "default_factory", None)
        ):
            kwargs[key] = value
        else:
            kwargs[key] = value
    return cls(**kwargs)


# Nested sections need explicit types because ``from __future__ import
# annotations`` turns field types into strings.
_SECTIONS = {
    "data": DataConfig,
    "preprocess": PreprocessConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "train": TrainConfig,
    "postprocess": PostprocessConfig,
    "analysis": AnalysisConfig,
}


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a (possibly partial) dict."""
    data = dict(data or {})
    sections = {}
    for key, cls in _SECTIONS.items():
        section = data.pop(key, None) or {}
        if not isinstance(section, dict):
            raise ValueError(f"config section {key!r} must be a mapping")
        sections[key] = _from_dict(cls, section)

    top_level = {f.name for f in fields(ExperimentConfig)} - set(_SECTIONS)
    unknown = set(data) - top_level
    if unknown:
        raise ValueError(f"unknown top-level config key(s): {sorted(unknown)}")
    return ExperimentConfig(**data, **sections)


def load_config(path: str | Path | None = None, **overrides: Any) -> ExperimentConfig:
    """Load a YAML config, applying dotted-path overrides.

    Overrides use ``section__key`` (double underscore) so they can be passed from
    a CLI without quoting, e.g. ``train__epochs=2``.
    """
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    for dotted, value in overrides.items():
        keys = dotted.split("__")
        node = data
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value

    return config_from_dict(data)
