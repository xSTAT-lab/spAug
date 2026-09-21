"""Synthetic-only same-sample package."""

from __future__ import annotations

import anndata as ad

from .common import package_weights, sample_synthetic


def build_package(real_sample: ad.AnnData, synthetic_sample: ad.AnnData | None, ratio: float, alpha=None, seed: int = 0):
    n_syn = max(1, int(round(real_sample.n_obs * float(ratio))))
    syn = sample_synthetic(synthetic_sample, n=n_syn, seed=seed)
    syn.obs["augmentation_source"] = "synthetic"
    return syn, package_weights(syn), {"n_real": 0, "n_synthetic": int(syn.n_obs)}
