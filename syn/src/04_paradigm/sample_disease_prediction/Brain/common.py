"""Shared paradigm utilities for Brain sample-level prediction."""

from __future__ import annotations

import anndata as ad
import numpy as np


def concat_packages(parts: list[ad.AnnData]) -> ad.AnnData:
    parts = [p.copy() for p in parts if p is not None and p.n_obs > 0]
    if not parts:
        raise ValueError("Cannot concatenate an empty sample package")
    if len(parts) == 1:
        return parts[0]
    return ad.concat(parts, axis=0, join="inner", merge="same")


def sample_synthetic(synthetic: ad.AnnData, n: int, seed: int) -> ad.AnnData:
    if synthetic is None or synthetic.n_obs == 0:
        raise ValueError("Synthetic package is required but empty")
    n = max(1, min(int(n), int(synthetic.n_obs)))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(synthetic.n_obs, size=n, replace=False))
    sampled = synthetic[idx]
    if getattr(sampled, "isbacked", False):
        return sampled.to_memory()
    return sampled.copy()


def real_with_source(real: ad.AnnData) -> ad.AnnData:
    out = real.copy()
    out.obs["augmentation_source"] = "real"
    if "source" not in out.obs.columns:
        out.obs["source"] = "real"
    return out


def package_weights(package: ad.AnnData, alpha: float | None = None) -> np.ndarray:
    source = package.obs.get("augmentation_source", package.obs.get("source")).astype(str).to_numpy()
    real_mask = np.isin(source, ["real"])
    syn_mask = ~real_mask
    weights = np.zeros(package.n_obs, dtype=np.float64)
    if alpha is None:
        weights[:] = 1.0 / max(1, package.n_obs)
        return weights
    alpha = float(alpha)
    if real_mask.any():
        weights[real_mask] = (1.0 - alpha) / max(1, int(real_mask.sum()))
    if syn_mask.any():
        weights[syn_mask] = alpha / max(1, int(syn_mask.sum()))
    if weights.sum() <= 0:
        weights[:] = 1.0 / max(1, package.n_obs)
    return weights
