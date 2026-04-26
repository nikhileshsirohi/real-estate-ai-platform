"""Tests for YAML config loading."""

from pathlib import Path

from src.utils.config_loader import load_yaml_config


def test_load_yaml_config_returns_expected_keys() -> None:
    config = load_yaml_config(Path("configs/model_config.yaml"))

    assert config["experiment_name"] == "real_estate_baseline"
    assert config["model_name"] == "linear_regression"
    assert config["test_size"] == 0.2
    assert config["random_state"] == 42
