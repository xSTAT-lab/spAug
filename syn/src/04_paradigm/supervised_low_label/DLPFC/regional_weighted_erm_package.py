"""Regional weighted ERM for DLPFC supervised low-label classification."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from .common import attach_slice_regions, concat_train, require_synthetic, sample_local_by_label_region


def regional_alpha_map(real_train: ad.AnnData, alpha_max: float) -> dict[str, float]:
    """Assign region-specific synthetic weights from dense-low to sparse-high."""
    alpha_max = float(alpha_max)
    if alpha_max <= 0:
        regions = sorted(real_train.obs["low_label_region"].astype(str).unique().tolist())
        return {region: 0.0 for region in regions}

    counts = real_train.obs["low_label_region"].astype(str).value_counts().sort_values(ascending=False)
    regions = counts.index.astype(str).tolist()
    if len(regions) == 1:
        return {regions[0]: alpha_max}

    levels = np.linspace(0.0, alpha_max, num=len(regions), dtype=np.float64)
    # Dense regions are first and get lower alpha; sparse regions are last and get higher alpha.
    return {region: float(levels[i]) for i, region in enumerate(regions)}


def build_package(real_train: ad.AnnData, synthetic_slice: ad.AnnData | None, ratio: float, alpha: float, seed: int = 0):
    synthetic_slice = require_synthetic(synthetic_slice, "regional_weighted_erm_package")
    real_region, synthetic_region = attach_slice_regions(real_train, synthetic_slice)
    syn = sample_local_by_label_region(real_region, synthetic_region, ratio=float(ratio), seed=seed)
    alpha_by_region = regional_alpha_map(real_region, float(alpha))
    syn_alpha = (
        syn.obs["low_label_region"]
        .astype(str)
        .map(alpha_by_region)
        .fillna(float(alpha))
        .astype(float)
        .to_numpy()
    )
    syn.obs["regional_alpha"] = syn_alpha
    real_region.obs["regional_alpha"] = 1.0
    weights = np.concatenate([
        np.ones(real_region.n_obs, dtype=np.float64),
        syn_alpha.astype(np.float64),
    ])
    train = concat_train(real_region, syn)
    train.uns["regional_alpha_strategy"] = "real_region_density_sparse_high"
    train.uns["regional_alpha_max"] = float(alpha)
    train.uns["regional_alpha_map"] = alpha_by_region
    train.uns["regional_alpha_table"] = pd.DataFrame(
        {
            "low_label_region": list(alpha_by_region.keys()),
            "alpha": list(alpha_by_region.values()),
        }
    ).to_dict(orient="records")
    return train, weights, real_region.copy()
