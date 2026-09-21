#!/usr/bin/env python3
"""Fast repository-health check for the public source distribution."""

from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DIRS = ("src/01_data_prep", "src/02_generators", "src/04_paradigm", "src/05_downstream", "src/06_evaluation", "configs")
REQUIRED_FILES = ("README.md", "LICENSE", "requirements.txt", "docs/data_contract.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-mechanisms", action="store_true", help="also run the small generator mechanism checks")
    args = parser.parse_args()

    missing = [p for p in (*REQUIRED_DIRS, *REQUIRED_FILES) if not (ROOT / p).exists()]
    if missing:
        print("missing required paths:", ", ".join(missing), file=sys.stderr)
        return 1

    forbidden = []
    non_ascii = []
    machine_paths = (
        "/" + "home/",
        "/" + "Users/",
        "/" + "mnt/",
        "/" + "workspace/",
        "C:\\Users\\",
    )
    text_suffixes = {".py", ".yaml", ".yml", ".md", ".txt", ".sh", ".cff", ".toml", ".ini"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix not in text_suffixes:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if any(ord(char) > 127 for char in content):
            non_ascii.append(str(path.relative_to(ROOT)))
        if any(marker in content for marker in machine_paths):
            forbidden.append(str(path.relative_to(ROOT)))
    if non_ascii:
        print("non-ASCII text found in public files:", ", ".join(non_ascii), file=sys.stderr)
        return 1
    if forbidden:
        print("machine-specific path marker found in:", ", ".join(forbidden), file=sys.stderr)
        return 1

    if not compileall.compile_dir(ROOT / "src", quiet=1, force=True):
        print("Python bytecode compilation failed", file=sys.stderr)
        return 1

    if args.with_mechanisms:
        checks = [
            "check_srtsim_faithful_core.py",
            "check_splatter_faithful_core.py",
            "check_sparsim_faithful_core.py",
            "check_scgan_mechanism.py",
            "check_scdiffusion_mechanism.py",
        ]
        for check in checks:
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / check)], cwd=ROOT)
            if result.returncode:
                return result.returncode

    print(f"public repository check passed: {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
