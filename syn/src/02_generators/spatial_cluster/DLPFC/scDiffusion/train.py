"""Train the project-local scDiffusion model on prepared training data."""

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
from common import infer_dataset, infer_mode, minimal_reference_obs, prepare_training_adata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scDiffusion.train")


# ============================================================
#  
# ============================================================

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ============================================================
# data loading
# ============================================================

def load_training_data(
    input_path: str,
    batch_size: int = 128,
    num_workers: int = 0,
    slice_id: Optional[str] = None,
    label_value: Optional[str] = None,
) -> tuple:
    """
    Load the HVG training data.

    Returns:
        (train_loader, n_genes, gene_names)
    """
    input_path = resolve_path(input_path)
    logger.info(f"loaddata: {input_path}")
    adata = ad.read_h5ad(str(input_path))
    logger.info(f"full data: {adata.n_obs} spots x {adata.n_vars} genes")

    adata, dataset, mode, slice_id = prepare_training_adata(
        adata, input_path=input_path, slice_id=slice_id, label_value=label_value
    )
    logger.info(
        f"training set: {adata.n_obs} observations | dataset={dataset} | "
        f"mode={mode} | slice_id={slice_id}"
    )
    logger.info(f"HVG subset: {adata.n_vars} genes")

    n_genes = adata.n_vars
    gene_names = adata.var_names.tolist()

    #   dense
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    X = X.astype(np.float32)

    effective_batch_size = min(batch_size, X.shape[0])
    drop_last = X.shape[0] > batch_size
    if effective_batch_size < batch_size:
        logger.warning(
            f"training sample count ({X.shape[0]}) less than batch_size ({batch_size}),"
            f"automatically use batch_size={effective_batch_size}, drop_last=False"
        )

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
    return train_loader, adata, n_genes, gene_names


# ============================================================
# model with diffusion process
# ============================================================

def build_model_and_diffusion(
    n_genes: int,
    hidden_dim: list,
    diffusion_steps: int,
    noise_schedule: str,
    dropout: float = 0.0,
    use_vae: bool = False,
    vae_latent_dim: int = 128,
) -> tuple:
    """
    build Cell_Unet + GaussianDiffusion.

    when use_vae=True  , Cell_Unet of input_dim as vae_latent_dim (latent-space dimension).

    Args:
        n_genes:         input/output dimension (gene count when VAE is disabled) or VAE latent_dim
        hidden_dim:      hidden layer dimensionlist (default [512, 512, 256, 128])
        diffusion_steps: number of diffusion steps (default 1000)
        noise_schedule:  "linear" / "cosine"
        dropout:         dropout  
        use_vae:         whether to use VAE  process
        vae_latent_dim:  VAE latent-space dimension

    Returns:
        (model, diffusion)
    """
    input_dim = vae_latent_dim if use_vae else n_genes

    model = Cell_Unet(
        input_dim=input_dim,
        hidden_num=hidden_dim,
        dropout=dropout,
    )

    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type="epsilon",  # predicted noise
        model_var_type="fixed_small",
        loss_type="mse",
        rescale_timesteps=False,
    )

    return model, diffusion


# ============================================================
# main training parameters
# ============================================================

def train(
    input_path: str,
    output_path: str,
    config_path: str = "configs/spatial_cluster/DLPFC/generators.yaml",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    hidden_dim: Optional[list] = None,
    lr: Optional[float] = None,
    n_timesteps: Optional[int] = None,
    beta_schedule: Optional[str] = None,
    device: str = "cuda",
    log_interval: int = 50,
    save_interval: int = 50,
    num_workers: int = 0,
    use_vae: Optional[bool] = None,
    vae_latent_dim: Optional[int] = None,
    vae_epochs: Optional[int] = None,
    vae_ckpt: Optional[str] = None,
    slice_id: Optional[str] = None,
    label_value: Optional[str] = None,
    classifier_key: str = "label",
    classifier_epochs: Optional[int] = None,
    classifier_lr: Optional[float] = None,
    classifier_guidance: bool = True,
):
    """scDiffusion training entry point."""

    # ---- Load configuration ----
    cfg = load_yaml(str(resolve_path(config_path)))
    sd_cfg = cfg.get("scDiffusion", {})
    global_cfg = cfg.get("global", {})

    # YAML in hidden_dim can is int or list
    yaml_hidden = sd_cfg.get("hidden_dim", [512, 512, 256, 128])
    if isinstance(yaml_hidden, int):
        yaml_hidden = [yaml_hidden, yaml_hidden, yaml_hidden // 2, yaml_hidden // 4]

    hidden_dim = hidden_dim or yaml_hidden
    n_timesteps = n_timesteps or sd_cfg.get("n_timesteps", 1000)
    beta_schedule = beta_schedule or sd_cfg.get("beta_schedule", "linear")
    lr = lr or sd_cfg.get("lr", 1e-4)
    if use_vae is None:
        use_vae = sd_cfg.get("use_vae", False)
    if vae_latent_dim is None:
        vae_latent_dim = sd_cfg.get("vae_latent_dim", 128)
    if vae_epochs is None:
        vae_epochs = sd_cfg.get("vae_epochs", 200)
    vae_lr = sd_cfg.get("vae_lr", 5e-4)
    vae_hidden = sd_cfg.get("vae_hidden_dim", [1024, 1024, 1024])
    classifier_hidden = sd_cfg.get("classifier_hidden_dim", [512, 512, 256, 128])
    classifier_dropout = float(sd_cfg.get("classifier_dropout", 0.1))
    if classifier_lr is None:
        classifier_lr = float(sd_cfg.get("classifier_lr", 1e-3))
    if epochs is None:
        epochs = sd_cfg.get("epochs", 1000)
    if classifier_epochs is None:
        classifier_epochs = int(sd_cfg.get("classifier_epochs", max(1, min(int(epochs), 50))))
    if batch_size is None:
        batch_size = sd_cfg.get("batch_size", 128)
    if device == "cuda":
        device = global_cfg.get("device", "cuda")

    logger.info("=" * 60)
    logger.info("scDiffusion training configuration")
    logger.info("=" * 60)
    logger.info(f"  epochs:         {epochs}")
    logger.info(f"  batch_size:     {batch_size}")
    logger.info(f"  hidden_dim:     {hidden_dim}")
    logger.info(f"  lr:             {lr}")
    logger.info(f"  n_timesteps:    {n_timesteps}")
    logger.info(f"  beta_schedule:  {beta_schedule}")
    logger.info(f"  use_vae:        {use_vae}")
    if use_vae:
        logger.info(f"  vae_latent_dim: {vae_latent_dim}")
        logger.info(f"  vae_epochs:     {vae_epochs}")
    logger.info(f"  device:         {device}")
    logger.info(f"  classifier:     {classifier_guidance} key={classifier_key} epochs={classifier_epochs}")
    logger.info("=" * 60)

    # ---- loaddata ----
    train_loader, adata_train, n_genes, gene_names = load_training_data(
        input_path, batch_size=batch_size, num_workers=num_workers,
        slice_id=slice_id, label_value=label_value,
    )
    dataset = infer_dataset(adata_train, input_path)
    mode = infer_mode(adata_train, input_path)
    x_classifier_np = (
        adata_train.X.toarray().astype(np.float32)
        if hasattr(adata_train.X, "toarray")
        else np.asarray(adata_train.X, dtype=np.float32)
    )

    # ---- create model ----
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ---- VAE  process ----
    vae_model = None
    if use_vae:
        vae_model = _train_or_load_vae(
            train_loader, n_genes, vae_latent_dim, vae_hidden,
            vae_epochs, vae_lr, device, vae_ckpt,
            output_path, log_interval,
        )
        # encode training data into latent space
        logger.info("encode training data into latent space...")
        latent_data = []
        vae_model.eval()
        with torch.no_grad():
            for (batch_x,) in train_loader:
                z = vae_model.encode(batch_x.to(device))
                latent_data.append(z.cpu())
        latent_tensor = torch.cat(latent_data, dim=0)
        with torch.no_grad():
            classifier_input_tensor = vae_model.encode(torch.from_numpy(x_classifier_np).to(device)).cpu()
        latent_dataset = TensorDataset(latent_tensor)
        latent_batch_size = min(batch_size, latent_tensor.shape[0])
        latent_drop_last = latent_tensor.shape[0] > batch_size
        train_loader = DataLoader(
            latent_dataset, batch_size=latent_batch_size, shuffle=True,
            num_workers=num_workers, drop_last=latent_drop_last,
        )
        if len(train_loader) == 0:
            logger.error(
                f"Latent DataLoader is empty: n_train={latent_tensor.shape[0]}, "
                f"batch_size={batch_size}, effective_batch_size={latent_batch_size}, "
                f"drop_last={latent_drop_last}"
            )
            sys.exit(1)
        logger.info(f"Latent space: {latent_tensor.shape}")
    else:
        latent_tensor = None
        classifier_input_tensor = torch.from_numpy(x_classifier_np)

    # ---- create model ----

    model, diffusion = build_model_and_diffusion(
        n_genes=n_genes,
        hidden_dim=hidden_dim,
        diffusion_steps=n_timesteps,
        noise_schedule=beta_schedule,
        use_vae=use_vae,
        vae_latent_dim=vae_latent_dim,
    )
    model.to(device)

    diff_input_dim = vae_latent_dim if use_vae else n_genes
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Cell_Unet number of parameters: {n_params:,} (input_dim={diff_input_dim})")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    # ---- training loop ----
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nstart training ({epochs} epochs)...\n")
    start_time = time.time()
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (real_batch,) in enumerate(train_loader):
            x_0 = real_batch.to(device)

            # randomsampletimestep
            t = torch.randint(0, n_timesteps, (x_0.shape[0],), device=device)

            #  
            noise = torch.randn_like(x_0)
            x_t = diffusion.q_sample(x_0, t, noise=noise)

            # model-predicted noise
            t_emb = t.unsqueeze(1)  # (B, 1)
            model_output = model(x_t.float(), t_emb.float())

            # MSE loss: predicted noise vs realnoise
            loss = torch.mean((noise - model_output) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                logger.info(
                    f"  Epoch {epoch:4d}/{epochs} | "
                    f"Step {global_step:6d} | "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch:4d}/{epochs} | "
            f"Avg Loss: {avg_loss:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # save periodically
        if epoch % save_interval == 0 or epoch == epochs:
            ckpt_path = output_path.with_suffix(f".epoch{epoch}.pt")
            save_checkpoint(
                model, optimizer, diffusion, epoch, global_step,
                n_genes, hidden_dim, n_timesteps, beta_schedule,
                str(ckpt_path),
            )
            logger.info(f"  save checkpoint: {ckpt_path}")

    classifier_model = None
    classifier_classes = None
    classifier_state = None
    classifier_metrics = None
    if classifier_guidance:
        classifier_model, classifier_classes, classifier_metrics = train_guidance_classifier(
            adata=adata_train,
            x_tensor=classifier_input_tensor,
            classifier_key=classifier_key,
            input_dim=diff_input_dim,
            hidden_dim=classifier_hidden,
            dropout=classifier_dropout,
            diffusion=diffusion,
            n_timesteps=n_timesteps,
            epochs=classifier_epochs,
            batch_size=batch_size,
            lr=float(classifier_lr),
            device=device,
            log_interval=log_interval,
        )
        if classifier_model is not None:
            classifier_state = classifier_model.cpu().state_dict()

    # save final checkpoint
    save_checkpoint(
        model, optimizer, diffusion, epochs, global_step,
        n_genes, hidden_dim, n_timesteps, beta_schedule,
        str(output_path),
        use_vae=use_vae,
        vae_latent_dim=vae_latent_dim if use_vae else None,
        metadata={
            "model_backend": "python_reimplementation",
            "model_backend_detail": "Project-local PyTorch implementation following the published EperLuo/scDiffusion design with VAE, Cell_Unet backbone, Cell_classifier guidance, and guided sampling",
            "model_source_reference": "EperLuo/scDiffusion published design: VAE/VAE_model.py, guided_diffusion/cell_model.py, guided_diffusion/gaussian_diffusion.py, classifier_sample.py",
            "dataset": dataset,
            "generation_mode": mode,
            "slice_id": str(slice_id) if slice_id is not None else (
                str(adata_train.obs["slice_id"].iloc[0]) if "slice_id" in adata_train.obs else None
            ),
            "label_condition": str(label_value) if label_value is not None else None,
            "label_strategy": "label_wise_generation" if label_value is not None else None,
            "gene_space": "hvg",
            "reference_obs": minimal_reference_obs(adata_train, dataset, mode),
            "classifier_key": classifier_key,
            "classifier_classes": classifier_classes,
            "classifier_state": classifier_state,
            "classifier_hidden_dim": classifier_hidden,
            "classifier_dropout": classifier_dropout,
            "classifier_metrics": classifier_metrics,
            "classifier_guidance_t_max": None if classifier_metrics is None else classifier_metrics.get("guidance_t_max"),
            "classifier_timestep_policy": None if classifier_metrics is None else classifier_metrics.get("timestep_policy"),
        },
    )
    # also save the VAE when available
    if vae_model is not None:
        vae_path = str(output_path).replace(".pt", "_vae.pt")
        torch.save({
            "encoder": vae_model.encoder.state_dict(),
            "decoder": vae_model.decoder.state_dict(),
            "n_genes": n_genes,
            "latent_dim": vae_latent_dim,
            "vae_hidden_dim": vae_hidden,
        }, vae_path)
        logger.info(f"VAE saveto: {vae_path}")

    logger.info(f"\ntraining completed,model saved to: {output_path}")
    logger.info(f"total elapsed time: {time.time() - start_time:.1f}s")


def save_checkpoint(
    model, optimizer, diffusion, epoch, step,
    n_genes, hidden_dim, n_timesteps, beta_schedule, path,
    use_vae=False, vae_latent_dim=None, metadata=None,
):
    """save modelcheckpoint."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "n_genes": n_genes,
            "hidden_dim": hidden_dim,
            "n_timesteps": n_timesteps,
            "beta_schedule": beta_schedule,
            "use_vae": use_vae,
            "vae_latent_dim": vae_latent_dim,
            **(metadata or {}),
        },
        path,
    )


def train_guidance_classifier(
    adata: ad.AnnData,
    x_tensor: torch.Tensor,
    classifier_key: str,
    input_dim: int,
    hidden_dim: list,
    dropout: float,
    diffusion: GaussianDiffusion,
    n_timesteps: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_interval: int,
):
    """Train classifier guidance model on noisy x_t."""
    if classifier_key not in adata.obs.columns:
        logger.info(f"skip classifier guidance: obs in has {classifier_key!r}")
        return None, None, None

    labels = adata.obs[classifier_key].astype(str).to_numpy()
    classes = sorted(np.unique(labels).tolist())
    if len(classes) < 2:
        logger.info(
            f"skip classifier guidance: {classifier_key!r} only {len(classes)} classes "
            f"({classes})"
        )
        return None, classes, {"enabled": False, "reason": "fewer_than_two_classes"}

    class_to_idx = {value: i for i, value in enumerate(classes)}
    y = torch.tensor([class_to_idx[str(v)] for v in labels], dtype=torch.long)
    x_tensor = x_tensor.detach().float().cpu()
    effective_batch_size = min(int(batch_size), int(x_tensor.shape[0]))
    ds = TensorDataset(x_tensor, y)
    loader = DataLoader(ds, batch_size=effective_batch_size, shuffle=True, drop_last=False)

    classifier = DiffusionClassifier(
        input_dim=input_dim,
        n_classes=len(classes),
        hidden_num=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    classifier.train()

    logger.info(
        f"training classifier guidance: key={classifier_key}, classes={classes}, "
        f"input_dim={input_dim}, epochs={epochs}"
    )
    guidance_t_max = max(0, int(n_timesteps) // 2)
    guidance_t_upper = max(1, guidance_t_max + 1)
    logger.info(f"  classifier timesteps: 0..{guidance_t_max} (scDiffusion paper)")
    global_step = 0
    last_loss = 0.0
    last_acc = 0.0
    for epoch in range(1, int(epochs) + 1):
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            t = torch.randint(0, guidance_t_upper, (batch_x.shape[0],), device=device)
            noise = torch.randn_like(batch_x)
            x_t = diffusion.q_sample(batch_x, t, noise=noise)
            logits = classifier(x_t.float(), diffusion._scale_timesteps(t).unsqueeze(1))
            loss = torch.nn.functional.cross_entropy(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(batch_x.shape[0])
            total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total_n += int(batch_x.shape[0])
            global_step += 1
            if log_interval > 0 and global_step % log_interval == 0:
                logger.info(f"  Classifier epoch={epoch} step={global_step} loss={loss.item():.6f}")
        last_loss = total_loss / max(total_n, 1)
        last_acc = total_correct / max(total_n, 1)
        logger.info(f"Classifier Epoch {epoch}/{epochs} | Loss={last_loss:.6f} | Acc={last_acc:.4f}")

    classifier.eval()
    return classifier, classes, {
        "enabled": True,
        "classifier_key": classifier_key,
        "n_classes": len(classes),
        "classes": classes,
        "epochs": int(epochs),
        "guidance_t_max": int(guidance_t_max),
        "timestep_policy": "train_classifier_on_0_to_T_over_2",
        "final_loss": float(last_loss),
        "train_accuracy": float(last_acc),
    }


def _train_or_load_vae(
    train_loader, n_genes, latent_dim, hidden_dim,
    epochs, lr, device, vae_ckpt, output_path, log_interval,
):
    """train or load the VAE model."""
    vae = GeneVAE(n_genes=n_genes, latent_dim=latent_dim, hidden_dim=hidden_dim)
    vae.to(device)

    if vae_ckpt is not None:
        logger.info(f"load pretrained VAE: {vae_ckpt}")
        ckpt = torch.load(vae_ckpt, map_location=device, weights_only=False)
        vae.encoder.load_state_dict(ckpt["encoder"])
        vae.decoder.load_state_dict(ckpt["decoder"])
        return vae

    logger.info(f"\n{'='*40}")
    logger.info(f"VAE pretrained ({epochs} epochs, lr={lr})")
    logger.info(f"{'='*40}")

    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=0.01)
    vae.train()
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for (batch_x,) in train_loader:
            x = batch_x.to(device)
            loss = vae.train_step(x, optimizer)
            epoch_loss += loss
            n_batches += 1
            global_step += 1

            if global_step % log_interval == 0:
                logger.info(f"  VAE Epoch {epoch} | Step {global_step} | Loss: {loss:.6f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % 50 == 0 or epoch == epochs:
            logger.info(f"VAE Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.6f}")

    vae.eval()
    logger.info(f"VAE  training completed (final loss: {avg_loss:.6f})")
    return vae


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scDiffusion training script")

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input h5ad path (contains split column)")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="model output path (.pt)")
    parser.add_argument("--config", "-c", type=str, default="configs/spatial_cluster/DLPFC/generators.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-timesteps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-vae", dest="use_vae", action="store_true", default=None,
                        help="Enable VAE latent diffusion and override the configuration")
    parser.add_argument("--no-vae", dest="use_vae", action="store_false",
                        help="Train diffusion directly in expression space")
    parser.add_argument("--vae-latent-dim", type=int, default=None,
                        help="VAE latent-space dimension (default 128)")
    parser.add_argument("--vae-epochs", type=int, default=None,
                        help="VAE epochs (default 200)")
    parser.add_argument("--vae-ckpt", type=str, default=None,
                        help="Path to a pretrained VAE checkpoint")
    parser.add_argument("--slice-id", type=str, default=None,
                        help="DLPFC mode trains on the selected slice")
    parser.add_argument("--label", type=str, default=None,
                        help="supervised generation condition: train on selected observations with this label and record label-wise synthetic labels")
    parser.add_argument("--classifier-key", type=str, default="label",
                        help="classifier guidance useof obs condition column;class  2  automaticskip")
    parser.add_argument("--classifier-epochs", type=int, default=None,
                        help="classifier guidance classification epochs")
    parser.add_argument("--classifier-lr", type=float, default=None,
                        help="classifier guidance classification learning rate")
    parser.add_argument("--no-classifier-guidance", dest="classifier_guidance", action="store_false",
                        default=True, help="disable classifier guidance classification training")

    args = parser.parse_args()

    train(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_timesteps=args.n_timesteps,
        device=args.device,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        num_workers=args.num_workers,
        use_vae=args.use_vae,
        vae_latent_dim=args.vae_latent_dim,
        vae_epochs=args.vae_epochs,
        vae_ckpt=args.vae_ckpt,
        slice_id=args.slice_id,
        label_value=args.label,
        classifier_key=args.classifier_key,
        classifier_epochs=args.classifier_epochs,
        classifier_lr=args.classifier_lr,
        classifier_guidance=args.classifier_guidance,
    )
