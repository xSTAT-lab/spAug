"""Generate synthetic expression with the project-local Splatter model."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import yaml
from scipy import sparse

def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "02_generators" / "_faithful"))

from common import (
    add_common_uns,
    make_synthetic_obs,
    make_synthetic_var,
    prepare_training_adata,
    validate_trusted_label_condition,
)
from model import SplatterModel as ProjectLocalSplatterModel
from splatter_core import SpatialSplatterPython


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("Splatter.generate")


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def restore_model(checkpoint: dict):
    backend = checkpoint.get("model_backend", "project_local_model")
    if backend == "python_reimplementation":
        return SpatialSplatterPython.from_dict(checkpoint["model_state"]), backend
    return ProjectLocalSplatterModel.from_dict(checkpoint["model_state"]), backend


def generate(
    input_path: str,
    checkpoint_path: str,
    output_path: str,
    config_path: str = "configs/sample_disease_prediction/Bowel/generators.yaml",
    n_generate_ratio: float | None = None,
    n_samples: int | None = None,
    random_seed: int | None = None,
    slice_id: str | None = None,
    label_value: str | None = None,
) -> str:
    cfg = load_yaml(resolve_path(config_path)).get("Splatter", {})
    if n_generate_ratio is None:
        n_generate_ratio = float(cfg.get("n_generate_ratio", 1.0))

    with resolve_path(checkpoint_path).open("rb") as f:
        checkpoint = pickle.load(f)
    model, model_backend = restore_model(checkpoint)
    gene_names = list(checkpoint["gene_names"])
    ref_obs = checkpoint.get("reference_obs")
    slice_id = str(checkpoint.get("slice_id")) if slice_id is None and checkpoint.get("slice_id") is not None else slice_id
    label_value = str(checkpoint.get("label_condition")) if label_value is None and checkpoint.get("label_condition") is not None else label_value

    adata = ad.read_h5ad(resolve_path(input_path))
    train_adata, dataset, mode, slice_id = prepare_training_adata(
        adata,
        input_path=str(resolve_path(input_path)),
        slice_id=slice_id,
        label_value=label_value,
    )
    if train_adata.var_names.astype(str).tolist() != gene_names:
        raise ValueError("Input HVG gene order does not match Splatter checkpoint")
    if ref_obs is None:
        ref_obs = train_adata.obs.copy()
    trusted_label, label_strategy = validate_trusted_label_condition(mode, checkpoint, label_value)
    if n_samples is None:
        n_samples = max(1, int(train_adata.n_obs * float(n_generate_ratio)))

    x, coords, regions = model.generate(int(n_samples), random_seed=random_seed)
    obs = make_synthetic_obs(
        ref_obs=ref_obs,
        n_obs=int(n_samples),
        generator="Splatter",
        dataset=dataset,
        mode=mode,
        slice_id=slice_id,
        synthetic_labels=trusted_label,
        label_strategy=label_strategy,
    )
    obs["synthetic_region"] = regions
    var = make_synthetic_var(gene_names)
    syn = ad.AnnData(X=sparse.csr_matrix(x.astype(np.float32)), obs=obs, var=var)
    syn.obsm["spatial"] = coords.astype(np.float64)

    add_common_uns(
        syn,
        "Splatter",
        dataset,
        mode,
        int(checkpoint.get("n_train_spots", train_adata.n_obs)),
        float(n_generate_ratio),
        slice_id=slice_id,
        label_strategy=label_strategy,
        label_condition=trusted_label,
    )
    syn.uns["generator_backend"] = model_backend
    syn.uns["model_backend"] = model_backend
    syn.uns["model_backend_detail"] = checkpoint.get("model_backend_detail", "")
    syn.uns["model_source_reference"] = checkpoint.get("model_source_reference", "")
    syn.uns["model_type"] = "splat_single_region_simadaptor" if model_backend == "python_reimplementation" else "regional_negative_binomial"
    syn.uns["region_method"] = str(checkpoint.get("region_method", "spatial_kmeans"))
    if model_backend == "python_reimplementation":
        syn.uns["n_regions"] = int(len(model.region_models))
    else:
        syn.uns["n_regions"] = int(len(model.region_params))
    syn.uns["coord_method"] = "simadaptor_region_resampling"
    syn.uns["coord_source"] = "simadaptor_region_resampling"
    syn.uns["label_conditioned"] = bool(trusted_label is not None)

    output_path_resolved = resolve_path(output_path)
    output_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    syn.write(output_path_resolved)
    logger.info("synthetic data saved: %s", output_path_resolved)
    logger.info("shape=%s x %s, regions=%s", syn.n_obs, syn.n_vars, syn.obs["synthetic_region"].nunique())
    return str(output_path_resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate from Python Splatter-style simulator")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--config", "-c", default="configs/sample_disease_prediction/Bowel/generators.yaml")
    parser.add_argument("--ratio", "--generate-ratio", dest="ratio", type=float, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--slice-id", default=None)
    parser.add_argument("--label", default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    generate(
        input_path=args.input,
        checkpoint_path=args.ckpt,
        output_path=args.output,
        config_path=args.config,
        n_generate_ratio=args.ratio,
        n_samples=args.n_samples,
        random_seed=args.seed,
        slice_id=args.slice_id,
        label_value=args.label,
    )
