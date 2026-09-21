"""Generate synthetic spatial data from fitted SRTsim parameters.

The command loads a project-local checkpoint, samples synthetic expression,
restores the configured spatial coordinate policy, and writes AnnData output.
"""

import os
import sys
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
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

from model import SRTsimModel as ProjectLocalSRTsimModel
from srtsim_core import SRTsimPython
from common import (
    add_common_uns,
    make_synthetic_obs,
    make_synthetic_var,
    prepare_training_adata,
    repeat_reference_coords,
    validate_trusted_label_condition,
)

# ============================================================
# log
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SRTsim.generate")


# ============================================================
# helper functions
# ============================================================

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def sample_perturbed_coords(
    train_coords: np.ndarray,
    n_samples: int,
    random_seed: int,
    noise_fraction: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample slice-native coordinates with small jitter for synthetic new locations."""
    train_coords = np.asarray(train_coords, dtype=np.float64)
    if train_coords.shape[0] == 0:
        raise ValueError("Cannot sample coordinates from empty train_coords")
    rng = np.random.default_rng(int(random_seed))
    source_idx = rng.integers(0, train_coords.shape[0], size=int(n_samples))
    coords = train_coords[source_idx].astype(np.float64, copy=True)
    coord_range = np.maximum(train_coords.max(axis=0) - train_coords.min(axis=0), 1.0)
    coords += rng.normal(0.0, coord_range * float(noise_fraction), size=coords.shape)
    return coords, source_idx


def restore_model_from_checkpoint(checkpoint: dict):
    """Restore new faithful SRTsim checkpoint, with checkpoint compatibility fallback."""
    if isinstance(checkpoint.get("model"), SRTsimPython):
        return checkpoint["model"], checkpoint.get("model_backend", "python_reimplementation")
    if "model_state" in checkpoint:
        return ProjectLocalSRTsimModel.from_dict(checkpoint["model_state"]), "project_local_model"
    raise ValueError("Checkpoint must contain a supported SRTsim model")


# ============================================================
# main generation function
# ============================================================

def generate(
    input_path: str,
    checkpoint_path: str,
    output_path: str,
    config_path: str = "configs/cross_slice_generalization/DLPFC/generators.yaml",
    n_generate_ratio: float = None,
    n_samples: int = None,
    random_seed: int = None,
    slice_id: str = None,
    label_value: str = None,
):
    """
    Load model parameters and generate synthetic data.

    Args:
        input_path:       used during training h5ad file (get gene names and n_train_spots)
        checkpoint_path:  trained .pkl model parameters
        output_path:      write synthetic AnnData path
        config_path:      generators.yaml configuration path
        n_generate_ratio: generation ratio (override YAML)
        n_samples:        explicit number of generated observations (override ratio)
        random_seed:      random seed (override checkpoint)
    """
    # Load configuration
    cfg = load_yaml(str(resolve_path(config_path)))
    srtsim_cfg = cfg.get("SRTsim", {})

    if n_generate_ratio is None:
        n_generate_ratio = srtsim_cfg.get("n_generate_ratio", 1.0)

    # ---- load checkpoint ----
    checkpoint_path = resolve_path(checkpoint_path)
    logger.info(f"load model: {checkpoint_path}")

    with open(str(checkpoint_path), "rb") as f:
        checkpoint = pickle.load(f)

    #  model
    model, model_backend = restore_model_from_checkpoint(checkpoint)
    gene_names = checkpoint["gene_names"]
    X_ref = checkpoint["X_ref"]
    train_coords = checkpoint.get("train_coords")
    train_labels = checkpoint.get("train_labels")
    ref_obs = checkpoint.get("reference_obs", None)
    ckpt_slice_id = checkpoint.get("slice_id", None)
    if slice_id is None and ckpt_slice_id is not None:
        slice_id = str(ckpt_slice_id)
    if label_value is None and checkpoint.get("label_condition") is not None:
        label_value = str(checkpoint["label_condition"])
    n_train_spots = checkpoint["n_train_spots"]

    logger.info(f"model parameters: {len(gene_names)} genes, {n_train_spots} training spots")
    logger.info(f"model backend: {model_backend}")
    logger.info(f"data type: {getattr(model, 'data_type', 'count')}")
    logger.info(f"simulation scheme: {model.sim_scheme}")

    # ---- validate input data ----
    input_path = resolve_path(input_path)
    logger.info(f"read input data: {input_path}")
    adata = ad.read_h5ad(str(input_path))

    adata_hvg, dataset, mode, slice_id = prepare_training_adata(
        adata, input_path=input_path, slice_id=slice_id, label_value=label_value
    )
    n_train = adata_hvg.n_obs
    if ref_obs is None:
        ref_obs = adata_hvg.obs.copy()

    n_genes_expected = adata_hvg.n_vars
    if n_genes_expected != len(gene_names):
        logger.error(
            f"gene count mismatch: input HVG count: {n_genes_expected}, model: {len(gene_names)}"
        )
        sys.exit(1)

    logger.info(f"gene count validation: {n_genes_expected}")
    trusted_label, label_strategy = validate_trusted_label_condition(mode, checkpoint, label_value)

    # ---- Determine the number of observations to generate ----
    if n_samples is None:
        n_samples = max(1, int(n_train * n_generate_ratio))
    logger.info(f"generate {n_samples} synthetic spot (ratio={n_generate_ratio})")

    # ---- generate synthetic data ----
    logger.info("start generation...")

    # overriderandom seed (ensureeach run resultdifferent)
    if random_seed is not None:
        model.random_seed = random_seed

    backend_seed = int(random_seed if random_seed is not None else getattr(model, "random_seed", 42))
    if model_backend == "python_reimplementation":
        if n_samples == n_train_spots:
            synthetic = model.generate(X_ref, labels=train_labels, random_seed=backend_seed)
            spatial_coords = train_coords.copy() if train_coords is not None else np.zeros((n_samples, 2), dtype=np.float64)
            coord_method = "srtsim_same_location_rank"
        elif n_samples < n_train_spots:
            X_ref_sub = X_ref[:n_samples]
            labels_sub = train_labels[:n_samples] if train_labels is not None else None
            synthetic = model.generate(X_ref_sub, labels=labels_sub, random_seed=backend_seed)
            spatial_coords = train_coords[:n_samples].copy() if train_coords is not None else np.zeros((n_samples, 2), dtype=np.float64)
            coord_method = "srtsim_same_location_rank_subset"
        else:
            if train_coords is None:
                raise ValueError("SRTsim faithful new-location generation requires train_coords")
            spatial_coords, source_idx = sample_perturbed_coords(train_coords, n_samples, backend_seed)
            if train_labels is not None:
                new_labels = np.asarray(train_labels).astype(str)[source_idx]
            else:
                new_labels = None
            synthetic = model.generate(
                X_ref,
                labels=train_labels,
                ref_coords=train_coords,
                new_coords=spatial_coords,
                new_labels=new_labels,
                random_seed=backend_seed,
            )
            coord_method = "srtsim_knn_rank_with_reference_coord_perturbation"
    else:
        # Checkpoint-based generation path for stored model parameters.
        if n_samples == n_train_spots:
            synthetic = model.generate(X_ref, n_generate=n_samples, labels=train_labels)
        elif n_samples <= n_train_spots:
            X_ref_sub = X_ref[:n_samples]
            labels_sub = train_labels[:n_samples] if train_labels is not None else None
            synthetic = model.generate(X_ref_sub, n_generate=n_samples, labels=labels_sub)
        else:
            logger.info(f"  checkpoint number of generated observations ({n_samples}) > referencecount ({n_train_spots}),use generate")
            n_batches = int(np.ceil(n_samples / n_train_spots))
            all_synth = []
            for b in range(n_batches):
                model.random_seed = backend_seed + b
                batch = model.generate(X_ref, n_generate=n_train_spots, labels=train_labels)
                all_synth.append(batch)
            synthetic = np.concatenate(all_synth, axis=0)[:n_samples]
        if train_coords is not None:
            spatial_coords, _ = sample_perturbed_coords(train_coords, n_samples, backend_seed)
        else:
            spatial_coords = np.zeros((n_samples, 2), dtype=np.float64)
        coord_method = "reference_coord_resampling"

    logger.info(f"generation completed: {synthetic.shape}")

    # ---- post-processing ----
    if getattr(model, "data_type", "count") == "count":
        # count-valued: ensurenon-negative number
        synthetic = np.maximum(np.round(synthetic), 0)

    # ---- Build output AnnData ----
    obs_df = make_synthetic_obs(
        ref_obs=ref_obs,
        n_obs=n_samples,
        generator="SRTsim",
        dataset=dataset,
        mode=mode,
        slice_id=slice_id,
        synthetic_labels=trusted_label,
        label_strategy=label_strategy,
    )
    var_df = make_synthetic_var(gene_names)

    syn_adata = ad.AnnData(
        X=sparse.csr_matrix(synthetic.astype(np.float32)),
        obs=obs_df,
        var=var_df,
    )

    # coordinates
    syn_adata.obsm["spatial"] = spatial_coords

    #  data
    add_common_uns(
        syn_adata, "SRTsim", dataset, mode, int(n_train_spots),
        float(n_generate_ratio), slice_id=slice_id,
        label_strategy=label_strategy,
        label_condition=trusted_label,
    )
    syn_adata.uns["generator_backend"] = model_backend
    syn_adata.uns["model_backend"] = model_backend
    syn_adata.uns["model_backend_detail"] = checkpoint.get("model_backend_detail", "")
    syn_adata.uns["model_source_reference"] = checkpoint.get("model_source_reference", "")
    syn_adata.uns["data_type"] = getattr(model, "data_type", "count")
    syn_adata.uns["sim_scheme"] = model.sim_scheme
    syn_adata.uns["coord_method"] = coord_method
    syn_adata.uns["coord_source"] = "srtsim_reference_rank_pattern"
    syn_adata.uns["label_conditioned"] = bool(trusted_label is not None or model.sim_scheme == "domain")

    # savedistribution selectionstatistics
    if model_backend == "python_reimplementation":
        model_counts = {}
        for block in model.blocks.values():
            for param in block.params:
                if param is not None:
                    model_counts[param.model_selected] = model_counts.get(param.model_selected, 0) + 1
        syn_adata.uns["model_distribution"] = model_counts
    elif getattr(model, "params", None):
        model_counts = {}
        for p in model.params.values():
            m = p["model"]
            model_counts[m] = model_counts.get(m, 0) + 1
        syn_adata.uns["model_distribution"] = model_counts

    # ---- save ----
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    syn_adata.write(str(output_path))

    logger.info(f"\nsynthetic data saved: {output_path}")
    logger.info(f"  dimension: {syn_adata.n_obs} spots x {syn_adata.n_vars} genes")
    logger.info(f"  coordinates: {spatial_coords.shape} (has_coords=true)")
    logger.info(f"  X range: [{synthetic.min():.4f}, {synthetic.max():.4f}]")
    logger.info(f"  X mean: {synthetic.mean():.4f}")
    if getattr(model, "data_type", "count") == "count":
        logger.info(f"   : {np.mean(synthetic == 0) * 100:.1f}%")

    return str(output_path)


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRTsim Synthetic data generation")

    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="input h5ad file (get gene names and training-set size)",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="trained model parameters (.pkl file)",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="synthetic-data output path (.h5ad)",
    )
    parser.add_argument(
        "--config", "-c", type=str, default="configs/cross_slice_generalization/DLPFC/generators.yaml",
        help="generators.yaml configuration file path",
    )
    parser.add_argument(
        "--ratio", "--generate-ratio", dest="ratio", type=float, default=None,
        help="generation ratio (override YAML, default 1.0)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="explicit number of generated observations (override ratio)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed (override checkpoint in )",
    )
    parser.add_argument(
        "--slice-id", type=str, default=None,
        help="DLPFC mode generates the selected slice",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="supervised generation condition: output synthetic label fixed to this label",
    )

    args = parser.parse_args()

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
