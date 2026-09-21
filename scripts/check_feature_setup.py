#!/usr/bin/env python3
"""Run the feature module's standalone validation."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parents[1] / "feature/scripts/check_setup.py"
    raise SystemExit(subprocess.call([sys.executable, str(script), "--isolated"]))
