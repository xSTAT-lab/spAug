#!/usr/bin/env python3
"""Task-entry CLI smoke tests for deep generators.

The mechanism tests validate model classes directly. This script exercises the
representative DLPFC supervised task entrypoints end-to-end on a tiny synthetic
AnnData file:

``train.py -> generate.py -> output h5ad schema/provenance checks``.

It uses a compact synthetic fixture for a formal pre-run validation gate.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SCGAN_DIR = PROJECT_ROOT / "src/02_generators/supervised_low_label/DLPFC/scGAN"
SCDIFFUSION_DIR = PROJECT_ROOT / "src/02_generators/supervised_low_label/DLPFC/scDiffusion"


def assert_true(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        suffix = f": {detail}" if detail else ""
        raise AssertionError(f"{name} failed{suffix}")


def write_toy_input(path: Path) -> None:
    rng = np.random.default_rng(2026)
    n_obs = 24
    n_vars = 12
    labels = np.array(["A", "B"] * (n_obs // 2), dtype=object)
    x = rng.poisson(lam=np.where(labels[:, None] == "A", 3.0, 6.0), size=(n_obs, n_vars))
    x[:, :2] += (labels[:, None] == "B").astype(int) * 3
    obs = pd.DataFrame(
        {
            "split": ["train"] * n_obs,
            "slice_id": ["toy_slice"] * n_obs,
            "label": labels,
        },
        index=[f"toy_spot_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame({"highly_variable": [True] * n_vars}, index=[f"gene_{i}" for i in range(n_vars)])
    adata = ad.AnnData(X=x.astype(np.float32), obs=obs, var=var)
    coords = np.column_stack(
        [
            np.linspace(0.0, 10.0, n_obs),
            np.repeat(np.arange(6), 4)[:n_obs].astype(float),
        ]
    )
    adata.obsm["spatial"] = coords
    adata.uns["dataset"] = "DLPFC"
    adata.uns["split_mode"] = "supervised"
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def write_tiny_config(path: Path) -> None:
    cfg = {
        "global": {"device": "cpu"},
        "scGAN": {
            "latent_dim": 6,
            "gen_hidden": [16],
            "dis_hidden": [16],
            "epochs": 1,
            "batch_size": 8,
            "lr_gen": 1e-3,
            "lr_dis": 1e-3,
            "n_critic": 1,
            "lambda_gp": 1.0,
            "use_lsn": True,
            "conditional_gan": True,
            "condition_dim": 4,
            "n_generate_ratio": 1.0,
            "random_seed": 2026,
        },
        "scDiffusion": {
            "hidden_dim": [16, 16, 8, 4],
            "classifier_hidden_dim": [16, 16, 8, 4],
            "classifier_dropout": 0.0,
            "classifier_lr": 1e-3,
            "classifier_epochs": 1,
            "classifier_guidance_scale": 0.5,
            "n_timesteps": 12,
            "beta_schedule": "linear",
            "epochs": 1,
            "batch_size": 8,
            "lr": 1e-3,
            "use_vae": False,
            "vae_latent_dim": 8,
            "vae_hidden_dim": [32, 32, 32],
            "n_generate_ratio": 1.0,
            "postprocess_gene_quantile": 0.999,
            "postprocess_library_quantile": 0.995,
            "random_seed": 2026,
        },
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def run(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        payload = {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-80:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-80:]),
        }
        raise RuntimeError(json.dumps(payload, indent=2, ensure_ascii=False))


def dense_x(adata: ad.AnnData) -> np.ndarray:
    return adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)


def check_synthetic_output(path: Path, generator: str, label_strategy: str) -> None:
    syn = ad.read_h5ad(path)
    x = dense_x(syn)
    assert_true(f"{generator} output shape", syn.n_obs == 6 and syn.n_vars == 12, str(syn.shape))
    assert_true(f"{generator} finite expression", bool(np.isfinite(x).all()))
    assert_true(f"{generator} nonnegative expression", bool((x >= 0).all()))
    assert_true(f"{generator} source", set(syn.obs["source"].astype(str)) == {generator})
    assert_true(f"{generator} split", set(syn.obs["split"].astype(str)) == {"synthetic"})
    assert_true(f"{generator} label", set(syn.obs["label"].astype(str)) == {"A"})
    assert_true(f"{generator} label strategy", set(syn.obs["label_strategy"].astype(str)) == {label_strategy})
    assert_true(f"{generator} backend", syn.uns.get("generator_backend") == "python_reimplementation")
    assert_true(f"{generator} spatial present", "spatial" in syn.obsm)
    coords = np.asarray(syn.obsm["spatial"], dtype=float)
    assert_true(f"{generator} finite spatial", bool(np.isfinite(coords).all()))
    assert_true(f"{generator} nonconstant spatial", float(coords.std()) > 0.0)
    assert_true(
        f"{generator} coord method",
        syn.uns.get("coord_method") == "reference_coord_resampling_with_jitter",
        str(syn.uns.get("coord_method")),
    )


def check_scgan(tmp: Path, input_h5ad: Path, config: Path) -> None:
    ckpt = tmp / "scgan_toy.pth"
    out = tmp / "scgan_synthetic.h5ad"
    run(
        [
            PYTHON,
            str(SCGAN_DIR / "train.py"),
            "--input",
            str(input_h5ad),
            "--output",
            str(ckpt),
            "--config",
            str(config),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--latent-dim",
            "6",
            "--n-critic",
            "1",
            "--condition-dim",
            "4",
            "--device",
            "cpu",
            "--log-interval",
            "9999",
        ]
    )
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert_true("scGAN checkpoint conditional", bool(checkpoint.get("conditional")))
    assert_true(
        "scGAN condition strategy",
        checkpoint.get("condition_strategy") == "internal_cscgan_conditional_bn_projection",
    )
    assert_true("scGAN source reference", "imsb-uke/scGAN" in str(checkpoint.get("model_source_reference", "")))
    run(
        [
            PYTHON,
            str(SCGAN_DIR / "generate.py"),
            "--input",
            str(input_h5ad),
            "--ckpt",
            str(ckpt),
            "--output",
            str(out),
            "--config",
            str(config),
            "--n-samples",
            "6",
            "--batch-size",
            "6",
            "--device",
            "cpu",
            "--slice-id",
            "toy_slice",
            "--label",
            "A",
        ]
    )
    check_synthetic_output(out, "scGAN", "internal_cscgan_conditional_bn_projection")
    syn = ad.read_h5ad(out)
    conditional = syn.uns.get("conditional_generation", {})
    assert_true("scGAN conditional output", bool(conditional.get("enabled")))
    assert_true("scGAN model type", syn.uns.get("model_type") == "cscgan_wgan_gp")


def check_scdiffusion(tmp: Path, input_h5ad: Path, config: Path) -> None:
    ckpt = tmp / "scdiffusion_toy.pt"
    out = tmp / "scdiffusion_synthetic.h5ad"
    run(
        [
            PYTHON,
            str(SCDIFFUSION_DIR / "train.py"),
            "--input",
            str(input_h5ad),
            "--output",
            str(ckpt),
            "--config",
            str(config),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--n-timesteps",
            "12",
            "--device",
            "cpu",
            "--no-vae",
            "--classifier-key",
            "label",
            "--classifier-epochs",
            "1",
            "--log-interval",
            "9999",
        ]
    )
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert_true("scDiffusion classifier state", checkpoint.get("classifier_state") is not None)
    assert_true("scDiffusion classifier classes", checkpoint.get("classifier_classes") == ["A", "B"])
    assert_true("scDiffusion source reference", "EperLuo/scDiffusion" in str(checkpoint.get("model_source_reference", "")))
    run(
        [
            PYTHON,
            str(SCDIFFUSION_DIR / "generate.py"),
            "--input",
            str(input_h5ad),
            "--ckpt",
            str(ckpt),
            "--output",
            str(out),
            "--config",
            str(config),
            "--n-samples",
            "6",
            "--batch-size",
            "6",
            "--use-ddim",
            "--device",
            "cpu",
            "--slice-id",
            "toy_slice",
            "--label",
            "A",
            "--guidance-label",
            "A",
            "--guidance-scale",
            "0.5",
        ]
    )
    check_synthetic_output(out, "scDiffusion", "classifier_guidance")
    syn = ad.read_h5ad(out)
    guidance = syn.uns.get("classifier_guidance", {})
    assert_true("scDiffusion guidance enabled", bool(guidance.get("enabled")))
    assert_true("scDiffusion beta-scaled guidance", bool(guidance.get("beta_scaled")))
    assert_true("scDiffusion model type", syn.uns.get("model_type") == "scdiffusion_local_vae_ddpm")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="spaug_deep_cli_smoke_") as tmpdir:
        tmp = Path(tmpdir)
        input_h5ad = tmp / "toy_supervised_dlpfc.h5ad"
        config = tmp / "generators.yaml"
        write_toy_input(input_h5ad)
        write_tiny_config(config)
        check_scgan(tmp, input_h5ad, config)
        check_scdiffusion(tmp, input_h5ad, config)
    print("deep generator task CLI smoke checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
