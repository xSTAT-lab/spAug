"""Generate synthetic expression with the project-local scGAN model."""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import yaml
import torch
from scipy import sparse

def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import ConditionalGenerator, Generator
from common import (
    add_common_uns,
    make_synthetic_obs,
    make_synthetic_var,
    prepare_training_adata,
    repeat_reference_coords,
    validate_trusted_label_condition,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scGAN.generate")


# ============================================================
#  
# ============================================================

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def sample_reference_coords(adata: ad.AnnData, n_obs: int, seed: int | None = None) -> tuple[np.ndarray, str]:
    """Sample slice-native training coordinates and add small jitter."""
    if "spatial" not in adata.obsm:
        return np.zeros((int(n_obs), 2), dtype=np.float64), "missing_spatial_zero_fallback"
    ref = np.asarray(adata.obsm["spatial"][:, :2], dtype=np.float64)
    if ref.shape[0] == 0:
        return np.zeros((int(n_obs), 2), dtype=np.float64), "empty_spatial_zero_fallback"
    rng = np.random.default_rng(seed)
    base = repeat_reference_coords(ref, int(n_obs)).astype(np.float64, copy=True)
    if base.shape[0] != int(n_obs):
        idx = rng.integers(0, ref.shape[0], size=int(n_obs))
        base = ref[idx].astype(np.float64, copy=True)
    sigma = np.maximum(ref.std(axis=0) * 0.01, 1e-6)
    return base + rng.normal(0.0, sigma, size=base.shape), "reference_coord_resampling_with_jitter"


# ============================================================
# main generation function
# ============================================================

def generate(
    input_path: str,
    checkpoint_path: str,
    output_path: str,
    config_path: str = "configs/cross_slice_generalization/DLPFC/generators.yaml",
    n_generate_ratio: float = None,
    batch_size: int = 256,
    device: str = "cuda",
    n_samples: int = None,
    slice_id: str = None,
    label_value: str = None,
    condition_label: str = None,
):
    """
    load modelandgenerate synthetic data.

    Args:
        input_path:       used during training h5ad file (get gene names and n_train_spots)
        checkpoint_path:  trained .pth model weights
        output_path:      write synthetic AnnData path
        config_path:      generators.yaml configuration path
        n_generate_ratio: generation ratio (override YAML)
        batch_size:       inference batch size
        device:           "cuda" / "cpu"
        n_samples:        explicit number of generated observations (override ratio)
    """
    # ---- Load configuration ----
    cfg = load_yaml(str(resolve_path(config_path)))
    scgan_cfg = cfg.get("scGAN", {})

    if n_generate_ratio is None:
        n_generate_ratio = scgan_cfg.get("n_generate_ratio", 1.0)

    # ---- read input metadata ----
    input_path = resolve_path(input_path)
    logger.info(f"read input data: {input_path}")
    adata = ad.read_h5ad(str(input_path))

    # ---- load model ----
    checkpoint_path = resolve_path(checkpoint_path)
    logger.info(f"load model: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_backend = ckpt.get("model_backend", "project_local_model")
    if slice_id is None and ckpt.get("slice_id") is not None:
        slice_id = str(ckpt["slice_id"])
    if label_value is None and ckpt.get("label_condition") is not None:
        label_value = str(ckpt["label_condition"])

    adata, dataset, mode, slice_id = prepare_training_adata(
        adata, input_path=input_path, slice_id=slice_id, label_value=label_value
    )
    n_train = adata.n_obs
    logger.info(
        f"training set: {n_train} observations | dataset={dataset} | "
        f"mode={mode} | slice_id={slice_id}"
    )
    trusted_label, label_strategy = validate_trusted_label_condition(mode, ckpt, label_value)
    if condition_label is None:
        condition_label = trusted_label

    n_genes = adata.n_vars
    gene_names = adata.var_names.tolist()
    logger.info(f"HVG gene count: {n_genes}")

    # ---- Determine the number of observations to generate ----
    if n_samples is None:
        n_samples = max(1, int(n_train * n_generate_ratio))
    logger.info(f"generate {n_samples} synthetic spot (ratio={n_generate_ratio})")

    gen_hidden = ckpt.get("gen_hidden", [256, 512])
    latent_dim = ckpt.get("latent_dim", 128)
    use_lsn = ckpt.get("use_lsn", False)
    lsn_lib_size = ckpt.get("lsn_lib_size", None)
    lib_size_mean = ckpt.get("lib_size_mean", None)
    lib_size_std = ckpt.get("lib_size_std", None)
    conditional = bool(ckpt.get("conditional", False))
    condition_classes = [str(x) for x in ckpt.get("condition_classes") or []]
    condition_dim = int(ckpt.get("condition_dim", 32) or 32)
    condition_index = None
    if conditional:
        if condition_label is None:
            raise ValueError("Conditional scGAN checkpoint requires --label or --condition-label at generation time")
        if str(condition_label) not in condition_classes:
            raise ValueError(
                f"Requested condition {condition_label!r} not in checkpoint condition_classes={condition_classes}"
            )
        condition_index = condition_classes.index(str(condition_label))

    #  gene count consistency
    ckpt_n_genes = ckpt.get("n_genes", n_genes)
    if ckpt_n_genes != n_genes:
        logger.error(
            f"gene count used during training {ckpt_n_genes} with input data HVG number {n_genes} does not match!"
        )
        sys.exit(1)

    #  outputactivation
    adata_train = adata
    if hasattr(adata_train.X, "toarray"):
        X_train = adata_train.X.toarray()
    else:
        X_train = np.array(adata_train.X)
    data_min = X_train.min()
    output_activation = "relu" if data_min >= -0.01 else "none"

    # create Generator andloadweights
    if conditional:
        generator = ConditionalGenerator(
            latent_dim=latent_dim,
            gen_hidden=gen_hidden,
            n_genes=n_genes,
            n_conditions=len(condition_classes),
            condition_dim=condition_dim,
            output_activation=output_activation,
            lsn_lib_size=lsn_lib_size if use_lsn else None,
        )
    else:
        generator = Generator(
            latent_dim=latent_dim,
            gen_hidden=gen_hidden,
            n_genes=n_genes,
            output_activation=output_activation,
            lsn_lib_size=lsn_lib_size if use_lsn else None,
        )
    generator.load_state_dict(ckpt["generator"])

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    generator.to(device)
    generator.eval()
    logger.info(f"Generator loaded successfully,device: {device}")
    if conditional:
        logger.info(
            f"Conditional scGAN enabled: {ckpt.get('conditional_key')}={condition_label} "
            f"(class_index={condition_index})"
        )
    if use_lsn:
        logger.info(f"LSN enabled: lib_size_mean={lib_size_mean:.2f}, lib_size_std={lib_size_std:.2f}")

    # ---- generate synthetic data ----
    logger.info(f"start generation...")
    synthetic_samples = []

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            n = end - start
            z = torch.randn(n, latent_dim, device=device)
            if conditional:
                cond = torch.full((n,), int(condition_index), device=device, dtype=torch.long)
            else:
                cond = None

            if use_lsn and lib_size_mean is not None:
                # sample size (reference truncated normal approximate)
                if lib_size_std is not None and lib_size_std > 0:
                    lib_sizes = torch.normal(
                        mean=lib_size_mean,
                        std=lib_size_std,
                        size=(n,),
                    ).clamp(min=1.0).to(device)
                else:
                    lib_sizes = torch.full((n,), lib_size_mean, device=device)
                fake = generator(z, cond, library_size=lib_sizes) if conditional else generator(z, library_size=lib_sizes)
            else:
                fake = generator(z, cond) if conditional else generator(z)
            synthetic_samples.append(fake.cpu().numpy())

    synthetic_X = np.concatenate(synthetic_samples, axis=0)
    logger.info(f"generation completed: {synthetic_X.shape}")

    # post-processing: clip non-negative expression data (for example, logcounts).
    # Z-score datasets such as Trastuzumab must retain negative values.
    if output_activation == "relu":
        synthetic_X = np.maximum(synthetic_X, 0)

    # ---- Build output AnnData ----
    ref_obs = ckpt.get("reference_obs", adata.obs.copy())
    obs_df = make_synthetic_obs(
        ref_obs=ref_obs,
        n_obs=n_samples,
        generator="scGAN",
        dataset=dataset,
        mode=mode,
        slice_id=slice_id,
        synthetic_labels=trusted_label,
        label_strategy=label_strategy,
    )
    var_df = make_synthetic_var(gene_names)

    syn_adata = ad.AnnData(
        X=sparse.csr_matrix(synthetic_X.astype(np.float32)),
        obs=obs_df,
        var=var_df,
    )
    add_common_uns(
        syn_adata, "scGAN", dataset, mode, int(n_train),
        float(n_generate_ratio), slice_id=slice_id,
        label_strategy=label_strategy,
        label_condition=trusted_label,
    )

    coord_seed = int(ckpt.get("random_seed", scgan_cfg.get("random_seed", 42)))
    coords, coord_method = sample_reference_coords(adata, n_samples, seed=coord_seed)
    syn_adata.obsm["spatial"] = coords.astype(np.float64)
    syn_adata.uns["generator_backend"] = model_backend
    syn_adata.uns["model_backend"] = model_backend
    syn_adata.uns["model_backend_detail"] = ckpt.get("model_backend_detail", "")
    syn_adata.uns["model_source_reference"] = ckpt.get("model_source_reference", "")
    syn_adata.uns["model_type"] = "cscgan_wgan_gp" if conditional else "scgan_wgan_gp"
    syn_adata.uns["conditional_generation"] = {
        "enabled": bool(conditional),
        "condition_label": None if condition_label is None else str(condition_label),
        "condition_index": None if condition_index is None else int(condition_index),
        "condition_classes": condition_classes,
        "condition_strategy": ckpt.get("condition_strategy", ""),
    }
    syn_adata.uns["coord_method"] = coord_method
    syn_adata.uns["coord_source"] = coord_method

    # ---- save ----
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    syn_adata.write(str(output_path))
    logger.info(f"synthetic data saved: {output_path}")
    logger.info(f"  dimension: {syn_adata.n_obs} spots x {syn_adata.n_vars} genes")
    logger.info(f"  coordinates: {syn_adata.obsm['spatial'].shape} ({coord_method})")

    return str(output_path)


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scGAN Synthetic data generation")

    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="input h5ad file (get gene names and training-set size)",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="trained model weights .pth file",
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
        "--batch-size", type=int, default=256,
        help="inference batch size (default 256)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--slice-id", type=str, default=None,
        help="DLPFC mode generates the selected slice",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="supervised generation condition: output synthetic label fixed to this label",
    )
    parser.add_argument(
        "--condition-label", type=str, default=None,
        help="internal cscGAN condition label;use by default --label / checkpoint label_condition",
    )

    args = parser.parse_args()

    generate(
        input_path=args.input,
        checkpoint_path=args.ckpt,
        output_path=args.output,
        config_path=args.config,
        n_generate_ratio=args.ratio,
        batch_size=args.batch_size,
        device=args.device,
        n_samples=args.n_samples,
        slice_id=args.slice_id,
        label_value=args.label,
        condition_label=args.condition_label,
    )
