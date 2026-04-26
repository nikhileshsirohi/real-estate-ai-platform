"""Helpers for loading YAML configuration files."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    return config
