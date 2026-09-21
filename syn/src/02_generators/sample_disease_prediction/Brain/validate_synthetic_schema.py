"""Validate the shared synthetic AnnData contract for generator outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np

def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
LEAKAGE_LABEL_COLUMNS = {
    "label",
    "spatialLIBD",
    "ground_truth",
    "manual_annotation",
    "manual_label",
    "cluster",
    "clusters",
    "refined_label",
    "refined_labels",
    "response",
    "Response",
}
SUPPORTED_GENERATORS = {"SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion"}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def validate_synthetic_schema(
    synthetic_path: str | Path,
    generator: str | None = None,
    dataset: str | None = None,
    mode: str | None = None,
    allow_zero_coords: bool = False,
) -> dict:
    path = resolve_path(synthetic_path)
    syn = ad.read_h5ad(str(path))

    generator = generator or str(syn.uns.get("generator", "") or "")
    dataset = dataset or str(syn.uns.get("dataset", "") or "")
    mode = mode or str(syn.uns.get("generation_mode", "") or "")

    _require(generator in SUPPORTED_GENERATORS, f"Unknown or missing generator: {generator!r}")
    _require(dataset, "Missing dataset in synthetic schema")
    _require(mode in {"supervised", "unsupervised"}, f"Invalid generation mode: {mode!r}")
    _require(syn.n_obs > 0, "Synthetic AnnData has no observations")
    _require(syn.n_vars > 0, "Synthetic AnnData has no genes")
    _require("highly_variable" in syn.var.columns, "Synthetic var must include highly_variable")
    _require(bool(syn.var["highly_variable"].astype(bool).all()), "Synthetic var['highly_variable'] must be True for all generated HVGs")
    _require("spatial" in syn.obsm, "Synthetic AnnData must include obsm['spatial']")
    coords = np.asarray(syn.obsm["spatial"])
    _require(coords.shape == (syn.n_obs, 2), f"obsm['spatial'] must have shape ({syn.n_obs}, 2), got {coords.shape}")
    _require(np.isfinite(coords).all(), "Synthetic coordinates contain NaN or inf")
    if not allow_zero_coords:
        _require(not np.allclose(coords, 0.0), "Synthetic coordinates are all zero; run coordinate assignment for non-spatial generators")

    required_obs = {"source", "split", "dataset", "synthetic_id"}
    missing_obs = sorted(required_obs.difference(syn.obs.columns))
    _require(not missing_obs, f"Missing synthetic obs columns: {missing_obs}")
    _require((syn.obs["split"].astype(str) == "synthetic").all(), "All synthetic obs rows must have split='synthetic'")

    if dataset == "DLPFC":
        _require("slice_id" in syn.obs.columns, "DLPFC synthetic obs must include slice_id")
    elif dataset == "Brain":
        _require("sample_id" in syn.obs.columns, "Brain synthetic obs must include sample_id")
        _require("disease_label" in syn.obs.columns, "Brain synthetic obs must include disease_label")
        sample_ids = sorted(syn.obs["sample_id"].astype(str).unique().tolist())
        _require(len(sample_ids) == 1, f"Brain synthetic file must contain one sample_id, got {sample_ids}")
    elif dataset == "Trastuzumab":
        _require("sample_id" in syn.obs.columns, "Trastuzumab synthetic obs must include sample_id")

    if mode == "supervised":
        _require("label" in syn.obs.columns, "Supervised synthetic obs must include trusted label")
        strategy = str(syn.uns.get("label_strategy", "") or "")
        _require(strategy in {"label_wise_generation", "slice_conditional_label_generation", "explicit_synthetic_labels"}, f"Unsupported supervised label_strategy: {strategy!r}")
    else:
        leakage = sorted(c for c in LEAKAGE_LABEL_COLUMNS if c in syn.obs.columns)
        _require(not leakage, f"Unsupervised synthetic obs contains leakage columns: {leakage}")

    for key in ("generator", "dataset", "generation_mode", "n_generated", "n_genes", "gene_space"):
        _require(key in syn.uns, f"Missing synthetic uns key: {key}")
    _require(int(syn.uns["n_generated"]) == syn.n_obs, "uns['n_generated'] does not match n_obs")
    _require(int(syn.uns["n_genes"]) == syn.n_vars, "uns['n_genes'] does not match n_vars")
    _require(str(syn.uns["gene_space"]) == "hvg", "Synthetic output must declare gene_space='hvg'")

    return {
        "path": str(path),
        "generator": generator,
        "dataset": dataset,
        "mode": mode,
        "n_obs": int(syn.n_obs),
        "n_vars": int(syn.n_vars),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate synthetic AnnData schema.")
    parser.add_argument("--synthetic", "-s", required=True, help="Synthetic .h5ad path")
    parser.add_argument("--generator", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--mode", default=None, choices=["supervised", "unsupervised"])
    parser.add_argument("--allow-zero-coords", action="store_true", help="Allow zero coordinates before coordinate assignment")
    return parser


def main():
    args = build_parser().parse_args()
    summary = validate_synthetic_schema(
        synthetic_path=args.synthetic,
        generator=args.generator,
        dataset=args.dataset,
        mode=args.mode,
        allow_zero_coords=args.allow_zero_coords,
    )
    print("Synthetic schema OK:", summary)


if __name__ == "__main__":
    main()
