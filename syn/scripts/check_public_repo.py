#!/usr/bin/env python3
"""Check the standalone synthetic-observation source distribution."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-mechanisms", action="store_true")
    args = parser.parse_args()
    for name in ("src", "configs", "README.md", "LICENSE", "requirements.txt", "docs/data_contract.md"):
        if not (ROOT / name).exists():
            raise SystemExit(f"Required module path: {name}")
    for directory in (ROOT / "src", ROOT / "scripts"):
        for path in directory.rglob("*.py"):
            compile(path.read_text(), str(path.relative_to(ROOT)), "exec")
    if args.with_mechanisms:
        for name in ("srtsim_faithful_core", "splatter_faithful_core", "sparsim_faithful_core", "scgan_mechanism", "scdiffusion_mechanism"):
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / f"check_{name}.py")], cwd=ROOT)
            if result.returncode:
                return result.returncode
    print("Synthetic-observation module checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
