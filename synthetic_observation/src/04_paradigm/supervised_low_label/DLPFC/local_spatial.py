"""Local spatial augmentation for DLPFC supervised low-label classification."""

from __future__ import annotations

import anndata as ad

from .common import concat_train, require_synthetic, sample_local_by_label_region_vector


def build_package(real_train: ad.AnnData, synthetic_slice: ad.AnnData | None, ratio: float, alpha=None, seed: int = 0):
    synthetic_slice = require_synthetic(synthetic_slice, "local_spatial")
    real_region, syn, ratio_map = sample_local_by_label_region_vector(
        real_train,
        synthetic_slice,
        max_ratio=float(ratio),
        seed=seed,
    )
    train = concat_train(real_region, syn)
    train.uns["local_spatial_ratio_role"] = "region_ratio_vector_max"
    train.uns["local_spatial_max_ratio"] = float(ratio)
    train.uns["local_spatial_ratio_map"] = ratio_map
    train.uns["local_spatial_ratio_table"] = [
        {"low_label_region": region, "ratio": int(region_ratio)}
        for region, region_ratio in sorted(ratio_map.items())
    ]
    return train, None, real_region.copy()
