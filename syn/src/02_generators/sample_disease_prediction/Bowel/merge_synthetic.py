"""Merge label-wise or slice-wise synthetic AnnData outputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path = [p for p in sys.path if Path(p or os.getcwd()).resolve() != PROJECT_ROOT]

import anndata as ad


KNOWN_LABEL_STRATEGIES = {
    "label_wise_generation",
    "explicit_synthetic_labels",
    "none",
    "unsupervised",
}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_synthetic_files(input_root: Path) -> list[Path]:
    paths = []
    for path in sorted(input_root.glob("**/synthetic.h5ad")):
        if path.name == "synthetic_pool_40x.h5ad":
            continue
        if path.parent == input_root:
            continue
        if any(child.is_dir() for child in path.parent.iterdir()):
            continue
        paths.append(path)
    return paths


def concat(paths: list[Path]) -> ad.AnnData:
    if not paths:
        raise ValueError("No synthetic.h5ad files found")
    adatas = [ad.read_h5ad(path) for path in paths]
    merged = ad.concat(adatas, axis=0, join="outer", merge="same", index_unique=None)
    merged.obs_names_make_unique()
    if adatas:
        merged.uns.update(adatas[0].uns)
    merged.uns["n_generated"] = int(merged.n_obs)
    merged.uns["n_genes"] = int(merged.n_vars)
    if "generate_ratio" in merged.uns and "n_generate_ratio" not in merged.uns:
        merged.uns["n_generate_ratio"] = merged.uns["generate_ratio"]
    if "n_generate_ratio" in merged.uns and "generate_ratio" not in merged.uns:
        merged.uns["generate_ratio"] = merged.uns["n_generate_ratio"]
    if "slice_id" in merged.obs.columns:
        slice_ids = sorted(merged.obs["slice_id"].astype(str).unique().tolist())
        merged.uns["slice_ids"] = slice_ids
        if len(slice_ids) == 1:
            merged.uns["slice_id"] = slice_ids[0]
        else:
            merged.uns.pop("slice_id", None)
            merged.uns.pop("coord_slice_id", None)
    if "label" in merged.obs.columns:
        labels = sorted(merged.obs["label"].astype(str).unique().tolist())
        merged.uns["merged_label_conditions"] = labels
        if len(labels) != 1 and "label_condition" in merged.uns:
            del merged.uns["label_condition"]
        if "label_strategy" not in merged.uns:
            merged.uns["label_strategy"] = "label_wise_generation"
    elif "label_strategy" not in merged.uns:
        merged.uns["label_strategy"] = "unsupervised"
    condition_source = str(merged.uns.get("condition_source", "") or "")
    if not condition_source or condition_source not in KNOWN_LABEL_STRATEGIES:
        label_strategy = str(merged.uns.get("label_strategy", "") or "")
        merged.uns["condition_source"] = label_strategy if label_strategy else "none"
    merged.uns["merged_from"] = [str(path) for path in paths]
    merged.uns["n_merged_files"] = len(paths)
    return merged


def write_per_slice(input_root: Path, paths: list[Path]) -> list[Path]:
    by_slice: dict[str, list[Path]] = {}
    for path in paths:
        adata = ad.read_h5ad(path, backed="r")
        if "slice_id" not in adata.obs.columns:
            adata.file.close()
            continue
        slices = sorted(adata.obs["slice_id"].astype(str).unique().tolist())
        adata.file.close()
        if len(slices) != 1:
            continue
        by_slice.setdefault(slices[0], []).append(path)

    outputs = []
    for slice_id, slice_paths in sorted(by_slice.items()):
        out = input_root / slice_id / "synthetic.h5ad"
        out.parent.mkdir(parents=True, exist_ok=True)
        merged = concat(slice_paths)
        merged.write_h5ad(out)
        outputs.append(out)
    return outputs


def merge_synthetic(
    input_root: str | Path,
    output: str | Path | None = None,
    per_slice: bool = False,
) -> Path:
    input_root = resolve_path(input_root)
    paths = find_synthetic_files(input_root)
    if per_slice:
        write_per_slice(input_root, paths)
    output = resolve_path(output) if output is not None else input_root / "synthetic_pool_40x.h5ad"
    output.parent.mkdir(parents=True, exist_ok=True)
    merged = concat(paths)
    merged.uns["pool_ratio"] = 40
    merged.uns["pool_name"] = "pool_40x"
    merged.uns.setdefault("generate_ratio", 40.0)
    merged.uns.setdefault("n_generate_ratio", 40.0)
    merged.uns["n_generated"] = int(merged.n_obs)
    merged.uns["n_genes"] = int(merged.n_vars)
    merged.write_h5ad(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge synthetic AnnData files.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument(
        "--per-slice",
        action="store_true",
        help="Also write <input-root>/<slice_id>/synthetic.h5ad from label-wise files.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    out = merge_synthetic(args.input_root, output=args.output, per_slice=args.per_slice)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
