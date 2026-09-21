"""Global real + synthetic augmentation for DLPFC cross-slice classification."""

from __future__ import annotations

import anndata as ad

from .common import concat_train, require_synthetic, sample_global_by_label


def build_package(real_train: ad.AnnData, synthetic_slice: ad.AnnData | None, ratio: float, alpha=None, seed: int = 0):
    synthetic_slice = require_synthetic(synthetic_slice, "global_real_plus_synthetic")
    syn = sample_global_by_label(real_train, synthetic_slice, ratio=float(ratio), seed=seed)
    return concat_train(real_train, syn), None, real_train.copy()
