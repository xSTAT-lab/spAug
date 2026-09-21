"""Global weighted ERM for DLPFC cross-slice classification."""

from __future__ import annotations

import anndata as ad
import numpy as np

from .common import concat_train, require_synthetic, sample_global_by_label


def build_package(real_train: ad.AnnData, synthetic_slice: ad.AnnData | None, ratio: float, alpha: float, seed: int = 0):
    synthetic_slice = require_synthetic(synthetic_slice, "weighted_erm_package")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("weighted_erm_package alpha must be in [0, 1]")
    syn = sample_global_by_label(real_train, synthetic_slice, ratio=float(ratio), seed=seed)
    weights = np.concatenate([
        np.full(real_train.n_obs, (1.0 - alpha) / real_train.n_obs, dtype=np.float64),
        np.full(syn.n_obs, alpha / syn.n_obs, dtype=np.float64),
    ])
    train = concat_train(real_train, syn)
    train.uns["weighted_erm_objective"] = "(1-alpha)*mean(real_loss)+alpha*mean(synthetic_loss)"
    train.uns["alpha"] = alpha
    return train, weights, real_train.copy()
