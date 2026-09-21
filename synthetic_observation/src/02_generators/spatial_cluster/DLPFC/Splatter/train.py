"""Fit project-local Splatter parameters on prepared training data."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import yaml

def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "02_generators" / "_faithful"))

from common import infer_dataset, infer_mode, minimal_reference_obs, prepare_training_adata
from splatter_core import SpatialSplatterPython


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("Splatter.train")


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def train(
    input_path: str,
    output_path: str,
    config_path: str = "configs/spatial_cluster/DLPFC/generators.yaml",
    slice_id: str | None = None,
    label_value: str | None = None,
    random_seed: int | None = None,
    n_regions: int | None = None,
) -> str:
    cfg = load_yaml(resolve_path(config_path)).get("Splatter", {})
    random_seed = int(cfg.get("random_seed", 42) if random_seed is None else random_seed)
    n_regions = int(cfg.get("n_regions", 7) if n_regions is None else n_regions)

    input_path_resolved = resolve_path(input_path)
    logger.info("loaddata: %s", input_path_resolved)
    adata = ad.read_h5ad(input_path_resolved)
    train_adata, dataset, mode, slice_id = prepare_training_adata(
        adata,
        input_path=str(input_path_resolved),
        slice_id=slice_id,
        label_value=label_value,
    )
    x = train_adata.X.toarray() if hasattr(train_adata.X, "toarray") else np.asarray(train_adata.X)
    coords = np.asarray(train_adata.obsm["spatial"], dtype=np.float64) if "spatial" in train_adata.obsm else None
    gene_names = train_adata.var_names.astype(str).tolist()

    logger.info(
        "training set: %s observations x %s genes | dataset=%s | mode=%s | slice_id=%s | label=%s",
        train_adata.n_obs,
        train_adata.n_vars,
        dataset,
        mode,
        slice_id,
        label_value,
    )

    model = SpatialSplatterPython(
        n_regions=n_regions,
        min_region_size=int(cfg.get("min_region_size", 20)),
        coord_sigma_factor=float(cfg.get("coord_sigma_factor", 0.15)),
        random_seed=random_seed,
    ).fit(x, gene_names, coords=coords)

    checkpoint = {
        "model_state": model.to_dict(),
        "model_backend": "python_reimplementation",
        "model_backend_detail": "Splat single-population Python implementation with SpatialSimBench region simAdaptor",
        "model_source_reference": "splatter::splatEstimate.matrix and splatter::splatSimulate(method='single')",
        "n_genes": len(gene_names),
        "gene_names": gene_names,
        "n_train_spots": int(train_adata.n_obs),
        "data_shape": [int(train_adata.n_obs), int(train_adata.n_vars)],
        "dataset": infer_dataset(train_adata, input_path),
        "generation_mode": infer_mode(train_adata, input_path),
        "slice_id": str(slice_id) if slice_id is not None else None,
        "label_condition": str(label_value) if label_value is not None else None,
        "label_strategy": "label_wise_generation" if label_value is not None else None,
        "label_conditioned": bool(label_value is not None),
        "gene_space": "hvg",
        "reference_obs": minimal_reference_obs(train_adata, dataset, mode),
        "region_labels": None if model.region_labels is None else model.region_labels.tolist(),
        "region_method": str(cfg.get("region_method", "spatial_kmeans")),
    }

    output_path_resolved = resolve_path(output_path)
    output_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    with output_path_resolved.open("wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Splatter checkpoint saved: %s", output_path_resolved)
    logger.info("regions=%s, region_probs=%s", len(model.region_models), np.round(model.region_probs, 3).tolist())
    return str(output_path_resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit Python Splatter-style simulator")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--config", "-c", default="configs/spatial_cluster/DLPFC/generators.yaml")
    parser.add_argument("--slice-id", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-regions", type=int, default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        slice_id=args.slice_id,
        label_value=args.label,
        random_seed=args.seed,
        n_regions=args.n_regions,
    )
