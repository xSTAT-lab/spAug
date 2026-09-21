#!/usr/bin/env python
"""Regression checks for scGAN/cscGAN-style implementation.

This script validates the project-local scGAN mechanism on tiny tensors. It verifies the project-local implementation contract used by formal spAug
experiments and records the published design reference.
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import numpy as np
import torch


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
SCGAN_DIR = PROJECT_ROOT / "src" / "02_generators" / "supervised_low_label" / "DLPFC" / "scGAN"
sys.path.insert(0, str(SCGAN_DIR))

from model import (  # noqa: E402
    ConditionalCritic,
    ConditionalGenerator,
    Generator,
    WGAN_GP,
    compute_gradient_penalty,
)


def assert_true(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def check_lsn() -> None:
    torch.manual_seed(1)
    gen = Generator(latent_dim=5, gen_hidden=[8], n_genes=6, use_batch_norm=False, lsn_lib_size=1.0)
    z = torch.randn(4, 5)
    target = torch.tensor([10.0, 20.0, 30.0, 40.0])
    out = gen(z, library_size=target)
    assert_true("lsn output shape", tuple(out.shape) == (4, 6))
    assert_true("lsn nonnegative", bool((out >= 0).all()))
    assert_true("lsn target library", bool(torch.allclose(out.sum(dim=1), target, rtol=1e-4, atol=1e-4)))


def check_conditional_modules() -> None:
    torch.manual_seed(2)
    z = torch.randn(6, 7)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    gen = ConditionalGenerator(
        latent_dim=7,
        gen_hidden=[10],
        n_genes=9,
        n_conditions=3,
        condition_dim=4,
        use_batch_norm=False,
    )
    out = gen(z, y)
    assert_true("conditional generator shape", tuple(out.shape) == (6, 9))
    assert_true("conditional generator finite", bool(torch.isfinite(out).all()))
    assert_true("conditional generator nonnegative", bool((out >= 0).all()))

    critic = ConditionalCritic(n_genes=9, dis_hidden=[11], n_conditions=3, condition_dim=4)
    score = critic(out, y)
    assert_true("conditional critic shape", tuple(score.shape) == (6, 1))
    assert_true("conditional critic finite", bool(torch.isfinite(score).all()))

    try:
        gen(z, None)
    except ValueError:
        pass
    else:
        raise AssertionError("ConditionalGenerator must require condition")
    try:
        critic(out, None)
    except ValueError:
        pass
    else:
        raise AssertionError("ConditionalCritic must require condition")


def check_wgan_gp_step_and_checkpoint() -> None:
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    real = torch.tensor(rng.gamma(shape=2.0, scale=1.0, size=(12, 10)), dtype=torch.float32)
    conditions = torch.tensor([0, 1, 2] * 4, dtype=torch.long)
    model = WGAN_GP(
        n_genes=10,
        latent_dim=6,
        gen_hidden=[12],
        dis_hidden=[12],
        lambda_gp=10.0,
        lr_gen=1e-3,
        lr_dis=1e-3,
        n_critic=1,
        device="cpu",
        use_lsn=True,
        lsn_lib_size=float(real.sum(dim=1).mean()),
        conditional=True,
        n_conditions=3,
        condition_dim=4,
    )
    result = model.train_step(real, conditions)
    assert_true("train step increments", model.step == 1)
    for key in ("d_loss", "g_loss", "gp"):
        assert_true(f"{key} finite", np.isfinite(float(result[key])))
    assert_true("gp nonnegative", float(result["gp"]) >= 0)

    generated = model.generate(
        9,
        batch_size=4,
        library_size_mean=float(real.sum(dim=1).mean()),
        library_size_std=float(real.sum(dim=1).std()),
        condition=1,
    )
    assert_true("generated shape", generated.shape == (9, 10))
    assert_true("generated finite", bool(np.isfinite(generated).all()))
    assert_true("generated nonnegative", bool((generated >= 0).all()))

    fake = torch.tensor(generated[: real.shape[0]], dtype=torch.float32)
    if fake.shape[0] < real.shape[0]:
        fake = torch.cat([fake, fake[: real.shape[0] - fake.shape[0]]], dim=0)
    gp = compute_gradient_penalty(model.critic, real, fake, condition=conditions)
    assert_true("gradient penalty finite", bool(torch.isfinite(gp)))
    assert_true("gradient penalty nonnegative", float(gp.detach()) >= 0)

    with tempfile.TemporaryDirectory(prefix="spaug_scgan_check_") as tmp:
        path = Path(tmp) / "toy_scgan.pth"
        model.save(
            str(path),
            extra={
                "model_backend": "python_reimplementation",
                "condition_strategy": "internal_cscgan_conditional_bn_projection",
                "condition_classes": ["A", "B", "C"],
            },
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert_true("checkpoint backend", ckpt.get("model_backend") == "python_reimplementation")
        assert_true("checkpoint conditional", bool(ckpt.get("conditional")))
        assert_true(
            "checkpoint condition strategy",
            ckpt.get("condition_strategy") == "internal_cscgan_conditional_bn_projection",
        )
        assert_true("checkpoint classes", ckpt.get("condition_classes") == ["A", "B", "C"])


def main() -> int:
    check_lsn()
    check_conditional_modules()
    check_wgan_gp_step_and_checkpoint()
    print("scGAN mechanism checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
