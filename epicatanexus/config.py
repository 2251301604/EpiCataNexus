"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return config


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    """Resolve a path relative to the repository, not the caller's CWD."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    config_path = Path(config_path).resolve()
    repository_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    return repository_root / candidate

