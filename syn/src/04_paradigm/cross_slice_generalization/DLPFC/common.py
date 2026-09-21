"""Shared helpers for DLPFC cross-slice supervised paradigms."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd


def _axis_bins(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.asarray([], dtype=int)
    if np.nanmax(values) <= np.nanmin(values):
        return np.zeros(values.shape[0], dtype=int)
    q = min(int(bins), max(1, values.shape[0]))
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, q + 1)))
    if edges.size <= 2:
        return np.zeros(values.shape[0], dtype=int)
    return np.searchsorted(edges[1:-1], values, side="right").astype(int)


def spatial_regions(coords: np.ndarray, bins: int = 3) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return np.asarray([], dtype=str)
    x_bin = _axis_bins(coords[:, 0], bins)
    y_bin = _axis_bins(coords[:, 1], bins)
    return np.asarray([f"r{int(x)}_{int(y)}" for x, y in zip(x_bin, y_bin)], dtype=str)


def attach_slice_regions(real_train: ad.AnnData, synthetic: ad.AnnData | None) -> tuple[ad.AnnData, ad.AnnData | None]:
    real_train = real_train.copy()
    if synthetic is None or synthetic.n_obs == 0:
        real_train.obs["low_label_region"] = spatial_regions(real_train.obsm["spatial"][:, :2])
        return real_train, synthetic
    synthetic = synthetic.copy()
    n_real = real_train.n_obs
    coords = np.vstack([
        np.asarray(real_train.obsm["spatial"][:, :2], dtype=np.float64),
        np.asarray(synthetic.obsm["spatial"][:, :2], dtype=np.float64),
    ])
    regions = spatial_regions(coords)
    real_train.obs["low_label_region"] = regions[:n_real]
    synthetic.obs["low_label_region"] = regions[n_real:]
    return real_train, synthetic


def sample_indices(indices: np.ndarray, n: int, rng: np.random.Generator) -> list[int]:
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0 or n <= 0:
        return []
    n_take = min(int(n), int(indices.size))
    return rng.choice(indices, size=n_take, replace=False).astype(int).tolist()


def sample_global_by_label(real_train: ad.AnnData, synthetic: ad.AnnData, ratio: float, seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    syn_labels = synthetic.obs["label"].astype(str).reset_index(drop=True)
    for label, real_idx in real_train.obs.groupby(real_train.obs["label"].astype(str), sort=True).groups.items():
        n = max(1, int(round(len(real_idx) * float(ratio))))
        candidates = np.flatnonzero(syn_labels.to_numpy() == str(label))
        chosen.extend(sample_indices(candidates, n, rng))
    if not chosen:
        raise ValueError("No synthetic observations matched real labels")
    syn = synthetic[chosen].copy()
    syn.obs["augmentation_source"] = "synthetic"
    return syn


def sample_local_by_label_region(real_train: ad.AnnData, synthetic: ad.AnnData, ratio: float, seed: int) -> ad.AnnData:
    real_train, synthetic = attach_slice_regions(real_train, synthetic)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    syn_key = synthetic.obs[["label", "low_label_region"]].astype(str).agg("||".join, axis=1).reset_index(drop=True)
    real_key = real_train.obs[["label", "low_label_region"]].astype(str).agg("||".join, axis=1)
    for key, real_idx in real_key.groupby(real_key, sort=True).groups.items():
        n = max(1, int(round(len(real_idx) * float(ratio))))
        candidates = np.flatnonzero(syn_key.to_numpy() == str(key))
        if candidates.size == 0:
            label = str(key).split("||", 1)[0]
            candidates = np.flatnonzero(synthetic.obs["label"].astype(str).to_numpy() == label)
        chosen.extend(sample_indices(candidates, n, rng))
    if not chosen:
        raise ValueError("No local synthetic observations matched real labels/regions")
    syn = synthetic[chosen].copy()
    syn.obs["augmentation_source"] = "synthetic"
    return syn


def local_region_ratio_map(real_train: ad.AnnData, max_ratio: float) -> dict[str, int]:
    """Build a region-specific synthetic ratio vector capped by ``max_ratio``.

    Dense regions receive lower ratios and sparse regions receive higher ratios.
    ``max_ratio=1`` makes every region use ratio 1.
    """
    max_ratio_i = max(1, int(round(float(max_ratio))))
    regions = real_train.obs["low_label_region"].astype(str)
    counts = regions.value_counts().sort_values(ascending=False)
    ordered = counts.index.astype(str).tolist()
    if not ordered:
        raise ValueError("No spatial regions found for local_spatial")
    if len(ordered) == 1:
        return {ordered[0]: max_ratio_i}
    levels = np.arange(1, max_ratio_i + 1, dtype=int)
    ratio_map: dict[str, int] = {}
    for rank, region in enumerate(ordered):
        level_idx = int(round(rank * (len(levels) - 1) / (len(ordered) - 1)))
        ratio_map[str(region)] = int(levels[level_idx])
    return ratio_map


def sample_local_by_label_region_vector(
    real_train: ad.AnnData,
    synthetic: ad.AnnData,
    max_ratio: float,
    seed: int,
) -> tuple[ad.AnnData, ad.AnnData, dict[str, int]]:
    """Sample local synthetic spots with a per-region ratio vector."""
    real_region, synthetic_region = attach_slice_regions(real_train, synthetic)
    ratio_map = local_region_ratio_map(real_region, max_ratio=max_ratio)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    syn_key = synthetic_region.obs[["label", "low_label_region"]].astype(str).agg("||".join, axis=1).reset_index(drop=True)
    real_key = real_region.obs[["label", "low_label_region"]].astype(str).agg("||".join, axis=1)
    syn_labels = synthetic_region.obs["label"].astype(str).to_numpy()
    for key, real_idx in real_key.groupby(real_key, sort=True).groups.items():
        label, region = str(key).split("||", 1)
        region_ratio = int(ratio_map.get(region, max(1, int(round(float(max_ratio))))))
        n = max(1, int(round(len(real_idx) * region_ratio)))
        candidates = np.flatnonzero(syn_key.to_numpy() == str(key))
        if candidates.size == 0:
            candidates = np.flatnonzero(syn_labels == label)
        chosen.extend(sample_indices(candidates, n, rng))
    if not chosen:
        raise ValueError("No local synthetic observations matched real labels/regions")
    syn = synthetic_region[chosen].copy()
    syn.obs["target_region_ratio"] = syn.obs["low_label_region"].astype(str).map(ratio_map).fillna(max(ratio_map.values())).astype(int)
    syn.obs["augmentation_source"] = "synthetic"
    return real_region, syn, ratio_map


def concat_train(real_train: ad.AnnData | None, syn_train: ad.AnnData | None) -> ad.AnnData:
    parts = []
    if real_train is not None and real_train.n_obs > 0:
        real = real_train.copy()
        real.obs["augmentation_source"] = "real"
        parts.append(real)
    if syn_train is not None and syn_train.n_obs > 0:
        syn = syn_train.copy()
        syn.obs["augmentation_source"] = "synthetic"
        parts.append(syn)
    if not parts:
        raise ValueError("Training package is empty")
    out = ad.concat(parts, join="inner", merge="same", index_unique=None)
    out.obs_names_make_unique()
    return out


def require_synthetic(synthetic: ad.AnnData | None, family: str) -> ad.AnnData:
    if synthetic is None or synthetic.n_obs == 0:
        raise ValueError(f"{family} requires same-slice synthetic data")
    return synthetic
