"""Real-only baseline for DLPFC cross-slice classification."""

from __future__ import annotations

import anndata as ad


def build_package(real_train: ad.AnnData, synthetic_slice: ad.AnnData | None = None, ratio=None, alpha=None, seed: int = 0):
    return real_train.copy(), None, real_train.copy()
