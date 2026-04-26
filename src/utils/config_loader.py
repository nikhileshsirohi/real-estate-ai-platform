"""Helpers for loading YAML configuration files."""

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path_like: str | Path) -> Path:
    """Resolve a path relative to the project root when needed."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return get_project_root() / path


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    resolved_path = resolve_project_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    return config
