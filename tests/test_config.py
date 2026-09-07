"""Configuration loading tests.

The important behaviour is that a typo'd key is an *error*.  Silently ignoring
an unknown config key is how you spend an afternoon wondering why a flag had no
effect on the results.
"""

from __future__ import annotations

import pytest
import yaml

from microseg.config import (
    ExperimentConfig,
    PreprocessConfig,
    config_from_dict,
    load_config,
)


class TestDefaults:
    def test_defaults_are_self_consistent(self):
        cfg = ExperimentConfig()
        assert cfg.model.out_channels == 3
        assert len(cfg.loss.class_weights) == cfg.model.out_channels

    def test_run_dir_combines_output_dir_and_name(self):
        cfg = ExperimentConfig(name="exp1", output_dir="runs")
        assert cfg.run_dir.as_posix() == "runs/exp1"


class TestFromDict:
    def test_partial_config_fills_in_defaults(self):
        cfg = config_from_dict({"name": "partial", "train": {"epochs": 5}})
        assert cfg.name == "partial"
        assert cfg.train.epochs == 5
        assert cfg.train.batch_size == ExperimentConfig().train.batch_size

    def test_empty_config_is_all_defaults(self):
        assert config_from_dict({}).name == ExperimentConfig().name

    def test_unknown_section_key_raises(self):
        with pytest.raises(ValueError, match="unknown config key"):
            config_from_dict({"train": {"epocs": 5}})

    def test_unknown_top_level_key_raises(self):
        with pytest.raises(ValueError, match="unknown top-level config key"):
            config_from_dict({"learning_rate": 0.1})

    def test_non_mapping_section_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            config_from_dict({"train": 5})


class TestRoundTrip:
    def test_save_and_reload_preserves_values(self, tmp_path):
        cfg = ExperimentConfig(name="rt")
        cfg.train.epochs = 7
        cfg.preprocess.denoise = "bilateral"
        cfg.loss.class_weights = [1.0, 2.0, 4.0]

        path = cfg.save(tmp_path / "config.yaml")
        reloaded = load_config(path)

        assert reloaded.name == "rt"
        assert reloaded.train.epochs == 7
        assert reloaded.preprocess.denoise == "bilateral"
        assert reloaded.loss.class_weights == [1.0, 2.0, 4.0]

    def test_overrides_use_double_underscore_paths(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"name": "base"}), encoding="utf-8")
        cfg = load_config(path, train__epochs=3, data__tile_size=64)
        assert cfg.train.epochs == 3 and cfg.data.tile_size == 64

    def test_load_without_a_path_gives_defaults(self):
        assert load_config(None).name == ExperimentConfig().name


class TestShippedConfigs:
    """Every config in configs/ must parse -- they are part of the interface."""

    @pytest.mark.parametrize(
        "name",
        [
            "unet_bbbc038.yaml",
            "unet_cpu.yaml",
            "unet_cpu_distance.yaml",
            "classical_baseline.yaml",
            "smoke.yaml",
        ],
    )
    def test_config_loads(self, name):
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "configs" / name
        if not path.exists():
            pytest.skip(f"{name} not present")
        cfg = load_config(path)
        assert cfg.name and cfg.model.out_channels >= 2

    def test_experiment_configs_share_a_split_seed(self):
        """The comparison is only fair if the held-out images are identical."""
        from pathlib import Path

        configs_dir = Path(__file__).resolve().parents[1] / "configs"
        seeds = {
            name: load_config(configs_dir / name).data.split_seed
            for name in (
                "unet_bbbc038.yaml",
                "unet_cpu.yaml",
                "unet_cpu_distance.yaml",
                "classical_baseline.yaml",
            )
            if (configs_dir / name).exists()
        }
        assert len(set(seeds.values())) == 1, f"split seeds differ: {seeds}"


class TestPreprocessConfig:
    def test_stages_can_be_disabled_with_none(self):
        cfg = PreprocessConfig(illumination=None, denoise=None, contrast=None)
        assert cfg.illumination is None and cfg.normalize == "percentile"
