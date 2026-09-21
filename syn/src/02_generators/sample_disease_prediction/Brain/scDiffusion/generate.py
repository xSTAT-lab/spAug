"""Generate synthetic expression with the project-local scDiffusion model."""

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
import torch.nn.functional as F
from scipy import sparse

def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# local import: model and diffusion process
from model import Cell_Unet, DiffusionClassifier, GeneVAE
from diffusion import GaussianDiffusion, get_named_beta_schedule
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
logger = logging.getLogger("scDiffusion.generate")


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


def calibrate_generated_expression(
    synthetic_X: np.ndarray,
    train_X: np.ndarray,
    gene_quantile: float = 0.999,
    library_quantile: float = 0.995,
) -> tuple[np.ndarray, dict]:
    """Constrain generated log-expression to the current training condition scale."""
    synthetic_X = np.asarray(synthetic_X, dtype=np.float32)
    train_X = np.asarray(train_X, dtype=np.float32)
    finite_train = train_X[np.isfinite(train_X)]
    train_fill = float(np.median(finite_train)) if finite_train.size else 0.0
    train_cap = float(np.max(finite_train)) if finite_train.size else 0.0
    n_nonfinite_before = int(np.size(synthetic_X) - np.isfinite(synthetic_X).sum())
    train_X = np.nan_to_num(train_X, nan=0.0, posinf=train_cap, neginf=0.0)
    synthetic_X = np.nan_to_num(synthetic_X, nan=train_fill, posinf=train_cap, neginf=0.0)
    synthetic_X = np.maximum(synthetic_X, 0.0)

    gene_cap = np.quantile(train_X, float(gene_quantile), axis=0).astype(np.float32)
    gene_max = train_X.max(axis=0).astype(np.float32)
    gene_cap = np.maximum(gene_cap, gene_max)
    gene_cap = np.maximum(gene_cap, 0.0)
    synthetic_X = np.minimum(synthetic_X, gene_cap[None, :])

    train_lib = train_X.sum(axis=1).astype(np.float64)
    lib_cap = float(np.quantile(train_lib, float(library_quantile)))
    syn_lib = synthetic_X.sum(axis=1).astype(np.float64)
    high = syn_lib > lib_cap
    if np.any(high) and lib_cap > 0:
        scale = (lib_cap / np.maximum(syn_lib[high], 1e-8)).astype(np.float32)
        synthetic_X[high] *= scale[:, None]

    return synthetic_X.astype(np.float32, copy=False), {
        "enabled": True,
        "gene_quantile": float(gene_quantile),
        "library_quantile": float(library_quantile),
        "train_max": float(np.max(train_X)),
        "train_library_cap": lib_cap,
        "n_nonfinite_replaced": n_nonfinite_before,
        "n_rows_scaled_by_library": int(np.sum(high)),
        "synthetic_max_after": float(np.max(synthetic_X)),
        "synthetic_library_mean_after": float(np.mean(synthetic_X.sum(axis=1))),
    }


# ============================================================
# modelbuild
# ============================================================

def build_model_and_diffusion(
    n_genes: int,
    hidden_dim: list,
    diffusion_steps: int,
    noise_schedule: str,
    use_vae: bool = False,
    vae_latent_dim: int = 128,
) -> tuple:
    """
    Build Cell_Unet and GaussianDiffusion using the training configuration.

    Returns:
        (model, diffusion)
    """
    model = Cell_Unet(
        input_dim=vae_latent_dim if use_vae else n_genes,
        hidden_num=hidden_dim,
        dropout=0.0,  #  disable dropout
    )

    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type="epsilon",
        model_var_type="fixed_small",
        loss_type="mse",
        rescale_timesteps=False,
    )

    return model, diffusion


# ============================================================
# main generation function
# ============================================================

def generate(
    input_path: str,
    checkpoint_path: str,
    output_path: str,
    config_path: str = "configs/sample_disease_prediction/Brain/generators.yaml",
    n_generate_ratio: float = None,
    batch_size: int = 256,
    device: str = "cuda",
    n_samples: int = None,
    use_ddim: bool = False,
    clip_denoised: bool = False,
    slice_id: str = None,
    label_value: str = None,
    guidance_label: str = None,
    guidance_scale: float = None,
):
    """
    load modelandgenerate synthetic data.

    Args:
        input_path:       used during training h5ad file (get gene names and n_train_spots)
        checkpoint_path:  trained .pt model weights
        output_path:      write synthetic AnnData path
        config_path:      generators.yaml configuration path
        n_generate_ratio: generation ratio (override YAML)
        batch_size:       inference batch size ( )
        device:           "cuda" / "cpu"
        n_samples:        explicit number of generated observations (override ratio)
        use_ddim:         whether to use DDIM accelerated sampling
        clip_denoised:    is resultto [-1, 1]
    """
    # ---- Load configuration ----
    cfg = load_yaml(str(resolve_path(config_path)))
    sd_cfg = cfg.get("scDiffusion", {})

    if n_generate_ratio is None:
        n_generate_ratio = sd_cfg.get("n_generate_ratio", 1.0)

    # ---- read input metadata ----
    input_path = resolve_path(input_path)
    logger.info(f"read input data: {input_path}")
    adata = ad.read_h5ad(str(input_path))

    # ---- load model ----
    checkpoint_path = resolve_path(checkpoint_path)
    logger.info(f"load checkpoint: {checkpoint_path}")

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
    n_genes = adata.n_vars
    gene_names = adata.var_names.tolist()
    logger.info(
        f"training set: {n_train} observations | dataset={dataset} | "
        f"mode={mode} | slice_id={slice_id}"
    )
    trusted_label, label_strategy = validate_trusted_label_condition(mode, ckpt, label_value)
    if guidance_label is None:
        guidance_label = trusted_label
    if guidance_scale is None:
        guidance_scale = float(sd_cfg.get("classifier_guidance_scale", 1.0))
    logger.info(f"HVG gene count: {n_genes}")

    X_train = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    data_min = float(X_train.min())
    nonnegative_expression = data_min >= -0.01
    logger.info(
        f"training data min={data_min:.4f}, "
        f"nonnegative_expression={nonnegative_expression}"
    )

    # ---- Determine the number of observations to generate ----
    if n_samples is None:
        n_samples = max(1, int(n_train * n_generate_ratio))
    logger.info(f"generate {n_samples} synthetic spot (ratio={n_generate_ratio})")

    # fromcheckpoint model parameters
    ckpt_n_genes = ckpt["n_genes"]
    hidden_dim = ckpt["hidden_dim"]
    n_timesteps = ckpt["n_timesteps"]
    beta_schedule = ckpt["beta_schedule"]
    use_vae = ckpt.get("use_vae", False)
    vae_latent_dim = ckpt.get("vae_latent_dim", None)

    if ckpt_n_genes != n_genes:
        logger.error(
            f"gene count used during training {ckpt_n_genes} with input data HVG number {n_genes} does not match!"
        )
        sys.exit(1)

    logger.info(
        f"model parameters: n_genes={ckpt_n_genes}, hidden_dim={hidden_dim}, "
        f"n_timesteps={n_timesteps}, beta_schedule={beta_schedule}, "
        f"use_vae={use_vae}"
    )

    # buildmodel and diffusion process
    model, diffusion = build_model_and_diffusion(
        n_genes=ckpt_n_genes,
        hidden_dim=hidden_dim,
        diffusion_steps=n_timesteps,
        noise_schedule=beta_schedule,
        use_vae=use_vae,
        vae_latent_dim=vae_latent_dim or 128,
    )
    model.load_state_dict(ckpt["model"])

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Cell_Unet loaded successfully,device: {device}")

    guidance_info = {
        "enabled": False,
        "requested_label": None if guidance_label is None else str(guidance_label),
        "scale": float(guidance_scale),
        "reason": "classifier_not_available",
    }
    cond_fn = None
    classifier_state = ckpt.get("classifier_state")
    classifier_classes = ckpt.get("classifier_classes")
    if classifier_state is not None and classifier_classes is not None and guidance_label is not None:
        classifier_classes = [str(x) for x in classifier_classes]
        if str(guidance_label) in classifier_classes:
            classifier = DiffusionClassifier(
                input_dim=vae_latent_dim if use_vae else n_genes,
                n_classes=len(classifier_classes),
                hidden_num=ckpt.get("classifier_hidden_dim", [512, 512, 256, 128]),
                dropout=float(ckpt.get("classifier_dropout", 0.0)),
            )
            classifier.load_state_dict(classifier_state)
            classifier.to(device)
            classifier.eval()
            target_idx = classifier_classes.index(str(guidance_label))
            guidance_t_max = int(ckpt.get("classifier_guidance_t_max", max(0, int(n_timesteps) // 2)))

            def _cond_fn(x, t, _out):
                active = (t <= guidance_t_max).float().view(-1, *([1] * (x.ndim - 1)))
                if float(active.sum().item()) == 0.0:
                    return torch.zeros_like(x)
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    logits = classifier(
                        x_in.float(),
                        diffusion._scale_timesteps(t).unsqueeze(1),
                    )
                    log_probs = F.log_softmax(logits, dim=1)
                    selected = log_probs[:, target_idx].sum()
                    grad = torch.autograd.grad(selected, x_in)[0]
                    beta_t = torch.from_numpy(diffusion.betas).to(t.device)[t].float()
                    beta_t = beta_t.view(-1, *([1] * (x.ndim - 1)))
                    return grad * active * beta_t * float(guidance_scale)

            cond_fn = _cond_fn
            guidance_info = {
                "enabled": True,
                "label": str(guidance_label),
                "class_index": int(target_idx),
                "classes": classifier_classes,
                "scale": float(guidance_scale),
                "guidance_t_max": int(guidance_t_max),
                "timestep_policy": ckpt.get("classifier_timestep_policy", "guide_only_0_to_T_over_2"),
                "beta_scaled": True,
                "classifier_key": ckpt.get("classifier_key", ""),
            }
            logger.info(f"enable classifier guidance: {guidance_info}")
        else:
            guidance_info["reason"] = f"label_not_in_classifier_classes:{guidance_label}"
            guidance_info["classes"] = classifier_classes
            logger.warning(f"skip classifier guidance: {guidance_info}")
    else:
        logger.info(f"notenable classifier guidance: {guidance_info}")

    # ---- load VAE (ifuse) ----
    vae_model = None
    if use_vae:
        vae_path = str(checkpoint_path).replace(".pt", "_vae.pt")
        if not Path(vae_path).exists():
            # attempt directory
            vae_path = str(checkpoint_path.with_suffix("")) + "_vae.pt"
        if Path(vae_path).exists():
            logger.info(f"load VAE: {vae_path}")
            vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
            vae_hidden_dim = vae_ckpt.get("vae_hidden_dim", [1024, 1024, 1024])
            vae_model = GeneVAE(
                n_genes=n_genes,
                latent_dim=vae_latent_dim,
                hidden_dim=vae_hidden_dim,
            )
            vae_model.encoder.load_state_dict(vae_ckpt["encoder"])
            vae_model.decoder.load_state_dict(vae_ckpt["decoder"])
            vae_model.to(device)
            vae_model.eval()
            logger.info(f"VAE loaded successfully (latent_dim={vae_latent_dim})")
        else:
            logger.error(f" to VAE checkpoint: {vae_path}")
            sys.exit(1)

    # ----  generatedimension ----
    gen_dim = vae_latent_dim if use_vae else n_genes

    # ----  generate (avoid ) ----
    logger.info(
        f"start to  (DDIM={use_ddim}, clip={clip_denoised})..."
    )
    all_samples = []
    remaining = n_samples

    with torch.no_grad():
        while remaining > 0:
            cur_batch = min(remaining, batch_size)

            if use_ddim:
                sample = diffusion.ddim_sample_loop(
                    model,
                    shape=(cur_batch, gen_dim),
                    clip_denoised=clip_denoised,
                    device=device,
                    progress=False,
                    eta=0.0,
                    cond_fn=cond_fn,
                )
            else:
                sample = diffusion.p_sample_loop(
                    model,
                    shape=(cur_batch, gen_dim),
                    clip_denoised=clip_denoised,
                    device=device,
                    progress=False,
                    cond_fn=cond_fn,
                )

            # VAE  : latent -> gene space
            if vae_model is not None:
                sample = vae_model.decode(sample)
                if nonnegative_expression:
                    sample = F.relu(sample)

            all_samples.append(sample.cpu().numpy())
            remaining -= cur_batch
            generated_so_far = n_samples - remaining
            logger.info(f"  alreadygenerate: {generated_so_far}/{n_samples}")

    synthetic_X = np.concatenate(all_samples, axis=0)[:n_samples]
    if nonnegative_expression:
        synthetic_X = np.maximum(synthetic_X, 0)
    synthetic_X, postprocess_info = calibrate_generated_expression(
        synthetic_X,
        X_train,
        gene_quantile=float(sd_cfg.get("postprocess_gene_quantile", 0.999)),
        library_quantile=float(sd_cfg.get("postprocess_library_quantile", 0.995)),
    )
    logger.info(f"generation completed: {synthetic_X.shape}")
    logger.info(f"post-processing: {postprocess_info}")

    # ---- Build output AnnData ----
    ref_obs = ckpt.get("reference_obs", adata.obs.copy())
    obs_df = make_synthetic_obs(
        ref_obs=ref_obs,
        n_obs=n_samples,
        generator="scDiffusion",
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
        syn_adata, "scDiffusion", dataset, mode, int(n_train),
        float(n_generate_ratio), slice_id=slice_id,
        label_strategy=label_strategy,
        label_condition=trusted_label,
    )
    syn_adata.uns["diffusion_params"] = {
        "n_timesteps": n_timesteps,
        "beta_schedule": beta_schedule,
        "hidden_dim": hidden_dim,
        "use_ddim": use_ddim,
        "use_vae": bool(use_vae),
        "vae_latent_dim": None if vae_latent_dim is None else int(vae_latent_dim),
    }
    syn_adata.uns["classifier_guidance"] = guidance_info
    syn_adata.uns["expression_postprocess"] = postprocess_info
    coord_seed = int(ckpt.get("random_seed", sd_cfg.get("random_seed", 42)))
    coords, coord_method = sample_reference_coords(adata, n_samples, seed=coord_seed)
    syn_adata.obsm["spatial"] = coords.astype(np.float64)
    syn_adata.uns["generator_backend"] = model_backend
    syn_adata.uns["model_backend"] = model_backend
    syn_adata.uns["model_backend_detail"] = ckpt.get("model_backend_detail", "")
    syn_adata.uns["model_source_reference"] = ckpt.get("model_source_reference", "")
    syn_adata.uns["model_type"] = "scdiffusion_local_vae_ddpm"
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
    parser = argparse.ArgumentParser(description="scDiffusion Synthetic data generation")

    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="input h5ad file (get gene names and training-set size)",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="trained model weights .pt file",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="synthetic-data output path (.h5ad)",
    )
    parser.add_argument(
        "--config", "-c", type=str, default="configs/sample_disease_prediction/Brain/generators.yaml",
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
        "--use-ddim", action="store_true",
        help="use DDIM accelerated sampling",
    )
    parser.add_argument(
        "--clip-denoised", action="store_true",
        help=" resultto [-1, 1]",
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
        "--guidance-label", type=str, default=None,
        help="classifier guidance oftargetcondition;use by default --label / checkpoint label_condition",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=None,
        help="classifier guidance gradient ;defaultreadconfiguration classifier_guidance_scale or 1.0",
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
        use_ddim=args.use_ddim,
        clip_denoised=args.clip_denoised,
        slice_id=args.slice_id,
        label_value=args.label,
        guidance_label=args.guidance_label,
        guidance_scale=args.guidance_scale,
    )
