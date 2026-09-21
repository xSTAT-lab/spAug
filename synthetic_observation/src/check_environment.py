"""Check the Python runtime and an optional R runtime."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "environment.yaml"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_version(executable: Path, args: list[str]) -> str:
    if not executable.is_file():
        raise FileNotFoundError(f"Configured executable does not exist: {executable}")
    proc = subprocess.run(
        [str(executable), *args],
        text=True,
        capture_output=True,
        cwd=str(PROJECT_ROOT),
    )
    output = (proc.stdout or proc.stderr).strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Runtime check failed for {executable}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return output


def check_environment(config_path: str | Path | None = None) -> dict:
    """Check the active interpreter or an optional local YAML configuration."""
    cfg = {}
    if config_path is not None:
        config_path = resolve_path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    py = Path(cfg.get("python", {}).get("executable", sys.executable))
    result = {
        "python": {
            "executable": str(py),
            "version": run_version(py, ["--version"]),
        },
    }
    r_value = cfg.get("r", {}).get("scdesign3", {}).get("executable")
    if r_value:
        r = Path(r_value)
        result["r_scdesign3"] = {"executable": str(r), "version": run_version(r, ["--version"])}
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check configured benchmark runtimes.")
    parser.add_argument("--config", default=None, help="Optional environment YAML path")
    return parser


def main():
    args = build_parser().parse_args()
    result = check_environment(args.config)
    for name, info in result.items():
        print(f"{name}: {info['executable']} | {info['version']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Environment check failed: {exc}", file=sys.stderr)
        raise
