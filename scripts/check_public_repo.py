#!/usr/bin/env python3
"""Validate the combined spAug source distribution using the standard library."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "LICENSE", "CITATION.cff", "THIRD_PARTY_NOTICES.md",
    "requirements.txt", "feature/requirements.txt", "syn/requirements.txt",
    "syn/src", "syn/configs", "syn/docs/data_contract.md", "syn/LICENSE",
    "feature/src/spaug_feature", "feature/configs", "feature/LICENSE",
    "feature/scripts/prepare_dlpfc.py", "feature/scripts/check_setup.py",
    "feature/scripts/extract_pfm_embeddings.py", "feature/spagcn/LICENSE",
    "feature/notebooks/feature_spatial_cluster.ipynb",
    "feature/notebooks/feature_low_label_classification.ipynb",
    "feature/notebooks/feature_cross_slice_classification.ipynb",
    "feature/notebooks/feature_disease_prediction.ipynb",
)
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".sh", ".cff", ".toml", ".ipynb"}
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".ipynb_checkpoints"}
CREDENTIAL_PATTERN = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
LOCAL_METADATA = {".DS_Store", "Thumbs.db", ".netrc", ".pypirc"}

MACHINE_PATHS = tuple("/" + part for part in ("home/", "Users/", "mnt/", "workspace/", "disk16T")) + ("C:" + "\\Users\\",)


def validate_repository(root: Path = ROOT) -> list[str]:
    errors = [f"Required path: {name}" for name in REQUIRED if not (root / name).exists()]
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.name == ".git" and path.parent != root:
            errors.append(f"Nested Git metadata: {rel}")
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.name in LOCAL_METADATA or (
            (path.name == ".env" or path.name.startswith(".env."))
            and path.name not in {".env.example", ".env.template"}
        ) or path.suffix in {".pem", ".key"}:
            errors.append(f"Keep local metadata and credential files outside the release: {rel}")
        if path.is_symlink():
            errors.append(f"Distribute source files directly: {rel}")
            continue
        if not path.is_file() or (path.suffix not in TEXT_SUFFIXES and path.name not in {"LICENSE", "Makefile", ".gitignore"}):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            if CREDENTIAL_PATTERN.search(text):
                errors.append(f"Credential pattern in source: {rel}")
            if any(marker in text for marker in MACHINE_PATHS):
                errors.append(f"Local machine path: {rel}")
            if path.suffix == ".py":
                compile(text, str(rel), "exec")
            elif path.suffix == ".ipynb":
                notebook = json.loads(text)
                metadata = notebook.get("metadata", {})
                if set(metadata) - {"kernelspec", "language_info"}:
                    errors.append(f"Use generic notebook metadata: {rel}")
                kernel = metadata.get("kernelspec", {})
                if kernel.get("display_name") not in {"Python 3", "Python 3 (ipykernel)"}:
                    errors.append(f"Use a generic notebook kernel display name: {rel}")
                if kernel.get("name") != "python3":
                    errors.append(f"Use the generic python3 kernel: {rel}")
                for index, cell in enumerate(notebook["cells"]):
                    source = "".join(cell["source"])
                    if any(marker in source for marker in MACHINE_PATHS):
                        errors.append(f"Local machine path: {rel}, cell {index}")
                    if cell.get("metadata") or cell.get("attachments"):
                        errors.append(f"Clear cell metadata and attachments: {rel}, cell {index}")
                    if cell["cell_type"] == "code":
                        if cell.get("outputs") or cell.get("execution_count") is not None:
                            errors.append(f"Clear execution state: {rel}, cell {index}")
                        compile(source, f"{rel}:cell {index}", "exec")
            elif path.suffix == ".md":
                for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                    if "://" in target or target.startswith(("#", "mailto:")):
                        continue
                    local = target.split("#", 1)[0]
                    if local and not (path.parent / local).exists():
                        errors.append(f"Unresolved local link in {rel}: {target}")
            if path.name.startswith("requirements") and path.suffix == ".txt":
                for line in text.splitlines():
                    if line.startswith("-r ") and not (path.parent / line[3:].strip()).is_file():
                        errors.append(f"Unresolved requirement include in {rel}")
        except (UnicodeError, SyntaxError, ValueError, KeyError) as exc:
            errors.append(f"{rel}: {type(exc).__name__}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-mechanisms", action="store_true", help="Run synthetic generator mechanism checks")
    args = parser.parse_args()
    errors = validate_repository()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    if args.with_mechanisms:
        for name in ("srtsim_faithful_core", "splatter_faithful_core", "sparsim_faithful_core", "scgan_mechanism", "scdiffusion_mechanism"):
            result = subprocess.run([sys.executable, str(ROOT / "syn/scripts" / f"check_{name}.py")], cwd=ROOT / "syn")
            if result.returncode:
                return result.returncode
    print("spAug source, notebook, link, and portable-path checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
