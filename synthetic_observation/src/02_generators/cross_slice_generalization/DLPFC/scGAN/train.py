"""Train the project-local scGAN model on prepared training data."""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import anndata as ad
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset

# fromProject rootorcurrentdirectory 
def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import WGAN_GP
from common import infer_dataset, infer_mode, minimal_reference_obs, prepare_training_adata

# ============================================================
# log
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scGAN.train")


# ============================================================
# configurationload
# ============================================================

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str) -> Path:
    """will forpath as for PROJECT_ROOT of forpath"""
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# ============================================================
# data loading
# ============================================================

def load_training_data(
    input_path: str,
    batch_size: int = 128,
    num_workers: int = 0,
    slice_id: Optional[str] = None,
    label_value: Optional[str] = None,
    conditional_key: str = "label",
    allow_multiclass_supervised: bool = False,
) -> tuple:
    """
    Load training data.

    Args:
        input_path: processed_with_split.h5ad filepath
        batch_size: DataLoader batch size
        num_workers: DataLoader number of workers

    Returns:
        (train_loader, adata_train, n_genes, gene_names, output_activation)
    """
    input_path = resolve_path(input_path)
    logger.info(f"loaddata: {input_path}")

    adata = ad.read_h5ad(str(input_path))
    logger.info(f"full data: {adata.n_obs} spots x {adata.n_vars} genes")

    adata_train, dataset, mode, slice_id = prepare_training_adata(
        adata,
        input_path=input_path,
        slice_id=slice_id,
        label_value=label_value,
        allow_multiclass_supervised=allow_multiclass_supervised,
    )
    label_msg = (
        f", label distribution: {adata_train.obs['label'].value_counts().to_dict()}"
        if "label" in adata_train.obs.columns else ""
    )
    logger.info(
        f"training set: {adata_train.n_obs} observations | dataset={dataset} | "
        f"mode={mode} | slice_id={slice_id}{label_msg}"
    )
    logger.info(f"HVG subset: {adata_train.n_vars} genes")

    n_genes = adata_train.n_vars
    gene_names = adata_train.var_names.tolist()

    # ---- extract expression matrix ----
    if hasattr(adata_train.X, "toarray"):
        X = adata_train.X.toarray()
    else:
        X = np.array(adata_train.X)

    X = X.astype(np.float32)

    # ----  outputactivation ----
    # DLPFC: logcounts >= 0 -> ReLU
    # Trastuzumab: Z-score (can contain negative values) -> identity
    data_min = X.min()
    if data_min >= -0.01:
        output_activation = "relu"
        logger.info(f"data min={data_min:.4f},use output activation ReLU")
    else:
        output_activation = "none"
        logger.info(f"data min={data_min:.4f} (can contain negative values),use output activation identity")

    # ---- compute sizestatistics  (usein LSN) ----
    lib_sizes = X.sum(axis=1)
    lib_size_mean = float(np.mean(lib_sizes))
    lib_size_std = float(np.std(lib_sizes))
    logger.info(f" sizestatistics: mean={lib_size_mean:.2f}, std={lib_size_std:.2f}")

    # ---- build DataLoader ----
    effective_batch_size = min(batch_size, X.shape[0])
    drop_last = X.shape[0] > batch_size
    if effective_batch_size < batch_size:
        logger.warning(
            f"training sample count ({X.shape[0]}) less than batch_size ({batch_size}),"
            f"automatically use batch_size={effective_batch_size}, drop_last=False"
        )

    condition_classes = None
    condition_codes = None
    if conditional_key in adata_train.obs.columns:
        condition_values = adata_train.obs[conditional_key].astype(str).to_numpy()
        condition_classes = sorted(np.unique(condition_values).tolist())
        if len(condition_classes) >= 2:
            class_to_idx = {value: i for i, value in enumerate(condition_classes)}
            condition_codes = np.asarray([class_to_idx[str(v)] for v in condition_values], dtype=np.int64)
            logger.info(f"condition column {conditional_key!r}: {condition_classes}")
        else:
            logger.info(f"condition column {conditional_key!r} only {len(condition_classes)} classes,use unconditional WGAN-GP")

    if condition_codes is not None:
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(condition_codes))
    else:
        dataset = TensorDataset(torch.from_numpy(X))
    train_loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    if len(train_loader) == 0:
        logger.error(
            f"DataLoader is empty: n_train={X.shape[0]}, batch_size={batch_size}, "
            f"effective_batch_size={effective_batch_size}, drop_last={drop_last}"
        )
        sys.exit(1)

    logger.info(f"DataLoader: {len(train_loader)} batches x {effective_batch_size}")
    return (
        train_loader, adata_train, n_genes, gene_names, output_activation,
        lib_size_mean, lib_size_std, condition_classes,
    )


# ============================================================
# training configuration 
# ============================================================

def train(
    input_path: str,
    output_path: str,
    config_path: str = "configs/cross_slice_generalization/DLPFC/generators.yaml",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    latent_dim: Optional[int] = None,
    gen_hidden: Optional[list] = None,
    dis_hidden: Optional[list] = None,
    lr_gen: Optional[float] = None,
    lr_dis: Optional[float] = None,
    n_critic: Optional[int] = None,
    lambda_gp: Optional[float] = None,
    device: str = "cuda",
    log_interval: int = 50,
    save_interval: int = 50,
    num_workers: int = 0,
    use_lsn: Optional[bool] = None,
    lr_decay: Optional[bool] = None,
    lr_final: Optional[float] = None,
    slice_id: Optional[str] = None,
    label_value: Optional[str] = None,
    conditional_key: str = "label",
    conditional_gan: Optional[bool] = None,
    condition_dim: Optional[int] = None,
):
    """
    WGAN-GP training entry point.

    Args:
        input_path:  input h5ad filepath
        output_path: model-weight output path (.pth)
        config_path: generators.yaml configuration file path
        additional parameter override configuration file
    """
    # ---- Load configuration ----
    cfg = load_yaml(str(resolve_path(config_path)))
    scgan_cfg = cfg.get("scGAN", {})
    global_cfg = cfg.get("global", {})

    # parameter precedence: CLI > YAML > defaults
    latent_dim = latent_dim or scgan_cfg.get("latent_dim", 128)
    gen_hidden = gen_hidden or scgan_cfg.get("gen_hidden", [256, 512])
    dis_hidden = dis_hidden or scgan_cfg.get("dis_hidden", [512, 256])
    lr_gen = lr_gen or scgan_cfg.get("lr_gen", 2e-4)
    lr_dis = lr_dis or scgan_cfg.get("lr_dis", 2e-4)
    n_critic = n_critic or scgan_cfg.get("n_critic", 5)
    lambda_gp = lambda_gp or scgan_cfg.get("lambda_gp", 10.0)
    if epochs is None:
        epochs = scgan_cfg.get("epochs", 500)
    if batch_size is None:
        batch_size = scgan_cfg.get("batch_size", 128)
    if use_lsn is None:
        use_lsn = scgan_cfg.get("use_lsn", False)
    if lr_decay is None:
        lr_decay = scgan_cfg.get("lr_decay", False)
    if lr_final is None:
        lr_final = scgan_cfg.get("lr_final", None)
    if conditional_gan is None:
        conditional_gan = bool(scgan_cfg.get("conditional_gan", True))
    if condition_dim is None:
        condition_dim = int(scgan_cfg.get("condition_dim", 32))
    if device == "cuda":
        device = global_cfg.get("device", "cuda")

    logger.info("=" * 60)
    logger.info("scGAN WGAN-GP training configuration")
    logger.info("=" * 60)
    logger.info(f"  epochs:       {epochs}")
    logger.info(f"  batch_size:   {batch_size}")
    logger.info(f"  latent_dim:   {latent_dim}")
    logger.info(f"  gen_hidden:   {gen_hidden}")
    logger.info(f"  dis_hidden:   {dis_hidden}")
    logger.info(f"  lr_gen:       {lr_gen}")
    logger.info(f"  lr_dis:       {lr_dis}")
    logger.info(f"  n_critic:     {n_critic}")
    logger.info(f"  lambda_gp:    {lambda_gp}")
    logger.info(f"  use_lsn:      {use_lsn}")
    logger.info(f"  lr_decay:     {lr_decay}")
    logger.info(f"  lr_final:     {lr_final}")
    logger.info(f"  conditional:  {conditional_gan} key={conditional_key} dim={condition_dim}")
    logger.info(f"  device:       {device}")
    logger.info("=" * 60)

    # ---- loaddata ----
    train_loader, adata_train, n_genes, gene_names, output_activation, \
        lib_size_mean, lib_size_std, condition_classes = \
        load_training_data(
            input_path, batch_size=batch_size, num_workers=num_workers,
            slice_id=slice_id, label_value=label_value,
            conditional_key=conditional_key,
            allow_multiclass_supervised=bool(conditional_gan and label_value is None),
        )
    dataset = infer_dataset(adata_train, input_path)
    mode = infer_mode(adata_train, input_path)

    # ---- compute max_steps (usein LR decay) ----
    max_steps = epochs * len(train_loader) if lr_decay else None
    is_conditional = bool(conditional_gan and condition_classes is not None and len(condition_classes) >= 2)
    if is_conditional:
        logger.info(f"enable cscGAN-style internal condition training: {conditional_key}={condition_classes}")
    else:
        logger.info("use unconditional scGAN/WGAN-GP training")

    # ---- create model ----
    model = WGAN_GP(
        n_genes=n_genes,
        latent_dim=latent_dim,
        gen_hidden=gen_hidden,
        dis_hidden=dis_hidden,
        lambda_gp=lambda_gp,
        lr_gen=lr_gen,
        lr_dis=lr_dis,
        n_critic=n_critic,
        device=device,
        output_activation=output_activation,
        use_lsn=use_lsn,
        lsn_lib_size=lib_size_mean if use_lsn else None,
        lr_decay=lr_decay,
        lr_final=lr_final,
        max_steps=max_steps,
        conditional=is_conditional,
        n_conditions=len(condition_classes or []),
        condition_dim=condition_dim,
    )

    n_params_g = sum(p.numel() for p in model.generator.parameters())
    n_params_d = sum(p.numel() for p in model.critic.parameters())
    logger.info(f"Generator number of parameters: {n_params_g:,}")
    logger.info(f"Critic number of parameters:    {n_params_d:,}")
    logger.info(f" number of parameters:         {n_params_g + n_params_d:,}")

    # ---- training loop ----
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nstart training ({epochs} epochs)...\n")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_gp = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if is_conditional:
                real_batch, conditions = batch
                result = model.train_step(real_batch, conditions)
            else:
                (real_batch,) = batch
                result = model.train_step(real_batch)
            epoch_d_loss += result["d_loss"]
            epoch_g_loss += result["g_loss"]
            epoch_gp += result["gp"]

            step = model.step
            if step % log_interval == 0:
                logger.info(
                    f"  Epoch {epoch:4d}/{epochs} | "
                    f"Step {step:6d} | "
                    f"D_loss: {result['d_loss']:+.4f} | "
                    f"G_loss: {result['g_loss']:+.4f} | "
                    f"GP: {result['gp']:.4f}"
                )

        n_batches = len(train_loader)

        # each epoch  
        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch:4d}/{epochs} | "
            f"D_loss: {epoch_d_loss / n_batches:+.4f} | "
            f"G_loss: {epoch_g_loss / n_batches:+.4f} | "
            f"GP: {epoch_gp / n_batches:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # save periodically
        if epoch % save_interval == 0 or epoch == epochs:
            ckpt_path = output_path.with_suffix(f".epoch{epoch}.pth")
            model.save(str(ckpt_path))
            logger.info(f"  save checkpoint: {ckpt_path}")

    # save final checkpoint
    extra_stats = {
        "lib_size_mean": lib_size_mean,
        "lib_size_std": lib_size_std,
        "model_backend": "python_reimplementation",
        "model_backend_detail": (
            "PyTorch scGAN/cscGAN-style WGAN-GP with AMSGrad, optional LSN, "
            "and internal condition embeddings when multiple conditions are available"
        ),
        "model_source_reference": "published imsb-uke/scGAN design; published project licensing applies; conditional batchnorm generator, projection critic, LSN, AMSGrad, and WGAN-GP",
        "dataset": dataset,
        "generation_mode": mode,
        "slice_id": str(slice_id) if slice_id is not None else (
            str(adata_train.obs["slice_id"].iloc[0]) if "slice_id" in adata_train.obs else None
        ),
        "label_condition": str(label_value) if label_value is not None else None,
        "label_strategy": "label_wise_generation" if label_value is not None else None,
        "conditional": is_conditional,
        "conditional_key": conditional_key if is_conditional else None,
        "condition_classes": condition_classes if is_conditional else None,
        "condition_dim": condition_dim if is_conditional else None,
        "condition_strategy": "internal_cscgan_conditional_bn_projection" if is_conditional else None,
        "gene_space": "hvg",
        "reference_obs": minimal_reference_obs(adata_train, dataset, mode),
    }
    model.save(str(output_path), extra=extra_stats)
    logger.info(f"\ntraining completed,model saved to: {output_path}")
    logger.info(f"total elapsed time: {time.time() - start_time:.1f}s")


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="scGAN WGAN-GP training script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="input h5ad filepath (e.g. data/02_interim/cross_slice_generalization/DLPFC/model_inputs/SRTsim/fixed_8train_4test/processed_with_split.h5ad)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="model-weight output path (e.g. models/scGAN_DLPFC.pth)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/cross_slice_generalization/DLPFC/generators.yaml",
        help="generators.yaml configuration file path (default: configs/cross_slice_generalization/DLPFC/generators.yaml)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="epochs (override YAML configuration)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="batch size (override YAML configuration)",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=None,
    )
    parser.add_argument(
        "--lr-gen", type=float, default=None,
    )
    parser.add_argument(
        "--lr-dis", type=float, default=None,
    )
    parser.add_argument(
        "--n-critic", type=int, default=None,
    )
    parser.add_argument(
        "--lambda-gp", type=float, default=None,
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--log-interval", type=int, default=50,
        help="logoutput  ( number)",
    )
    parser.add_argument(
        "--save-interval", type=int, default=50,
        help="modelsave  (epoch)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader number of workers",
    )
    parser.add_argument(
        "--use-lsn", action="store_true", default=None,
        help="enable Library Size Normalization following the output_LSN convention",
    )
    parser.add_argument(
        "--lr-decay", action="store_true", default=None,
        help="enable exponential learning-rate decay following the decay=True convention",
    )
    parser.add_argument(
        "--lr-final", type=float, default=None,
        help="final learning rate (usein LR decay, corresponding  alpha_final)",
    )
    parser.add_argument(
        "--slice-id", type=str, default=None,
        help="DLPFC mode trains on the selected slice",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="supervised generation condition: train on selected observations with this label and record label-wise synthetic labels",
    )
    parser.add_argument(
        "--conditional-key", type=str, default="label",
        help="internal cscGAN condition column;class count >= 2 enables conditional training",
    )
    parser.add_argument(
        "--no-conditional-gan", dest="conditional_gan", action="store_false",
        default=None, help="disable internal scGAN conditioning and use unconditional WGAN-GP",
    )
    parser.add_argument(
        "--condition-dim", type=int, default=None,
        help="condition embedding dimension",
    )

    args = parser.parse_args()

    train(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        lr_gen=args.lr_gen,
        lr_dis=args.lr_dis,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        device=args.device,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        num_workers=args.num_workers,
        use_lsn=args.use_lsn,
        lr_decay=args.lr_decay,
        lr_final=args.lr_final,
        slice_id=args.slice_id,
        label_value=args.label,
        conditional_key=args.conditional_key,
        conditional_gan=args.conditional_gan,
        condition_dim=args.condition_dim,
    )
