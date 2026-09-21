#!/usr/bin/env python3
"""Validate feature imports, configuration, and numerical helpers in isolation."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_one(notebook: Path) -> None:
    cells = [c for c in json.loads(notebook.read_text())["cells"] if c["cell_type"] == "code"]
    with tempfile.TemporaryDirectory(prefix="spaug_feature_setup_") as temp:
        scope = {"__name__": "__spaug_setup__"}
        for cell in cells[:2]:
            source = "".join(cell["source"])
            source = source.replace('FEATURE_ROOT / "outputs/', f'Path({temp!r}) / "')
            exec(compile(source, notebook.name, "exec"), scope)
        assert scope["FEATURE_ROOT"].resolve() == ROOT
        for name, module in tuple(sys.modules.items()):
            if name.startswith("spaug_feature") and getattr(module, "__file__", None):
                assert Path(module.__file__).resolve().is_relative_to(ROOT)
    print(f"{notebook.name}: setup PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated", action="store_true", help="Test a standalone copy of this module")
    parser.add_argument("--notebook", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.notebook:
        check_one(args.notebook)
        return 0
    if args.isolated:
        with tempfile.TemporaryDirectory(prefix="spaug_feature_isolated_") as temp:
            destination = Path(temp) / "feature"
            shutil.copytree(ROOT, destination, ignore=shutil.ignore_patterns(
                "data", "outputs", "__pycache__", ".git", ".venv", ".ipynb_checkpoints"))
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["PYTHONNOUSERSITE"] = "1"
            return subprocess.call([sys.executable, str(destination / "scripts/check_setup.py")], cwd=destination, env=env)
    for notebook in sorted((ROOT / "notebooks").glob("*.ipynb")):
        for cwd in (ROOT, notebook.parent):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--notebook", str(notebook)], cwd=cwd, capture_output=True, text=True)
            if result.returncode:
                print(result.stdout + result.stderr, file=sys.stderr)
                return result.returncode
        print(f"{notebook.name}: module and notebook working directories PASS", flush=True)
    return subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
