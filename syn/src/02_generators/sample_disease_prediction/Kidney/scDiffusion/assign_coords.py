"""CLI wrapper for shared coordinate assignment."""

from pathlib import Path
import sys

def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coord_assignment import main


if __name__ == "__main__":
    main()
