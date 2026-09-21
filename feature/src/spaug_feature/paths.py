"""Paths and configuration for the standalone feature module."""
from pathlib import Path
import yaml

FEATURE_ROOT = Path(__file__).resolve().parents[2]

def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else FEATURE_ROOT / path

def load_config(path: str | Path) -> dict:
    with resolve_path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}

load_yaml = load_config
