#!/usr/bin/env python3
"""Regression checks for the scDiffusion-style implementation.

The checks use tiny tensors and the representative supervised_low_label module.
They validate published-design structure using locally generated tensors.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCDIFFUSION_DIR = PROJECT_ROOT / "src" / "02_generators" / "supervised_low_label" / "DLPFC" / "scDiffusion"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_true(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        suffix = f": {detail}" if detail else ""
        raise AssertionError(f"{name} failed{suffix}")


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    model_mod = _load_module("spaug_scdiffusion_model", SCDIFFUSION_DIR / "model.py")
    diffusion_mod = _load_module("spaug_scdiffusion_diffusion", SCDIFFUSION_DIR / "diffusion.py")

    x = torch.rand(8, 16)
    t_vec = torch.randint(0, 20, (8,))
    t_col = t_vec.view(-1, 1).float()

    vae = model_mod.GeneVAE(n_genes=16, latent_dim=6, hidden_dim=[32, 32, 32])
    z = vae.encode(x)
    x_hat = vae.decode(z)
    assert_true("vae latent shape", z.shape == (8, 6), str(z.shape))
    assert_true("vae decoder shape", x_hat.shape == x.shape, str(x_hat.shape))
    assert_true("vae l2 normalize", torch.allclose(z.norm(dim=1), torch.ones(8), atol=1e-5))
    encoder_linear = sum(isinstance(m, torch.nn.Linear) for m in vae.encoder.network)
    decoder_linear = sum(isinstance(m, torch.nn.Linear) for m in vae.decoder.network)
    assert_true("vae three hidden layers plus output", encoder_linear == 4 and decoder_linear == 4)

    denoiser = model_mod.Cell_Unet(input_dim=6, hidden_num=[32, 32, 16, 8], dropout=0.0)
    pred_noise = denoiser(torch.randn(8, 6), t_col)
    assert_true("cell_unet output shape", pred_noise.shape == (8, 6), str(pred_noise.shape))
    assert_true("cell_unet finite", torch.isfinite(pred_noise).all().item())

    classifier = model_mod.DiffusionClassifier(
        input_dim=6,
        n_classes=3,
        hidden_num=[32, 32, 16, 8],
        dropout=0.1,
    )
    logits_vec = classifier(torch.randn(8, 6), t_vec)
    logits_col = classifier(torch.randn(8, 6), t_col)
    assert_true("classifier output shape vector t", logits_vec.shape == (8, 3), str(logits_vec.shape))
    assert_true("classifier output shape column t", logits_col.shape == (8, 3), str(logits_col.shape))
    assert_true("classifier published-design batchnorm blocks", hasattr(classifier, "norm1") and hasattr(classifier, "norm2") and hasattr(classifier, "norm3"))
    assert_true("classifier finite", torch.isfinite(logits_vec).all().item() and torch.isfinite(logits_col).all().item())

    x_guided = torch.randn(8, 6, requires_grad=True)
    logits = classifier(x_guided, t_vec)
    selected = torch.nn.functional.log_softmax(logits, dim=-1)[:, 1]
    grad = torch.autograd.grad(selected.sum(), x_guided)[0]
    assert_true("classifier guidance grad shape", grad.shape == x_guided.shape, str(grad.shape))
    assert_true("classifier guidance grad finite", torch.isfinite(grad).all().item())

    betas = diffusion_mod.get_named_beta_schedule("linear", 20)
    diffusion = diffusion_mod.GaussianDiffusion(betas=betas)
    assert_true("linear beta clipped", float(betas.max()) <= 0.999)
    noisy = diffusion.q_sample(z.detach(), t_vec)
    losses = diffusion.training_losses(denoiser, z.detach(), t_vec)
    assert_true("q_sample shape", noisy.shape == z.shape, str(noisy.shape))
    assert_true("training loss finite", torch.isfinite(losses["loss"]).all().item())

    print("scDiffusion mechanism checks: PASS")


if __name__ == "__main__":
    main()
