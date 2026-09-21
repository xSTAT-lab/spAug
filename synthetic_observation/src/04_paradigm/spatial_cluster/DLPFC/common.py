"""Shared utilities for parameterized augmentation paradigms."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path = [p for p in sys.path if Path(p or os.getcwd()).resolve() != PROJECT_ROOT]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "02_generators" / "spatial_cluster" / "DLPFC"))

from coord_assignment import assign_coordinates_to_adata, load_yaml as load_mapping_yaml


TRUSTED_LABEL_STRATEGIES = {"label_wise_generation", "explicit_synthetic_labels"}
KNOWN_GENERATORS = ("SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion")
DEFAULT_SYNTHETIC_RATIOS = list(range(1, 31))
DEFAULT_REAL_FRACTIONS = [1.0]
COORD_POLICIES = {"pool_default", "spatial_resampling", "knn_mapping", "spatial_perturbation", "gmm_sampling"}
DEFAULT_FAMILIES = [
    "global_real_plus_synthetic",
    "local_spatial",
]
DLPFC_ONLY_FAMILIES = {"local_spatial"}
UNSUPERVISED_INCOMPATIBLE_FAMILIES: set[str] = set()


@dataclass
class ParadigmPackage:
    """In-memory paradigm package used by streaming downstream runs."""

    train: Optional[ad.AnnData]
    manifest: dict
    pretrain: Optional[ad.AnnData] = None
    finetune: Optional[ad.AnnData] = None


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _stringify_obs_value(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def sanitize_obs_for_h5ad(adata: ad.AnnData) -> ad.AnnData:
    """Convert mixed object/string obs columns into h5ad-writable strings."""
    adata = adata.copy()
    for col in adata.obs.columns:
        series = adata.obs[col]
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            adata.obs[col] = series.map(_stringify_obs_value).astype(str)
    return adata


def infer_dataset(adata: ad.AnnData, path: str | Path | None = None) -> str:
    dataset = str(adata.uns.get("dataset", "") or "")
    if dataset:
        return dataset
    if path is not None:
        parts = set(resolve_path(path).parts)
        if "DLPFC" in parts:
            return "DLPFC"
        if "Trastuzumab" in parts:
            return "Trastuzumab"
    return "unknown"


def infer_mode(adata: ad.AnnData, path: str | Path | None = None) -> str:
    mode = str(adata.uns.get("split_mode", "") or adata.uns.get("generation_mode", "") or "").lower()
    if mode in {"supervised", "unsupervised"}:
        return mode
    if path is not None:
        parts = set(resolve_path(path).parts)
        if "unsupervised" in parts:
            return "unsupervised"
        if "supervised" in parts:
            return "supervised"
    return "supervised" if "label" in adata.obs.columns else "unsupervised"


def align_gene_space(
    real: ad.AnnData,
    synthetic: ad.AnnData,
    strategy: str = "intersection",
) -> tuple[ad.AnnData, ad.AnnData]:
    if strategy != "intersection":
        raise NotImplementedError("Gene alignment uses the intersection strategy")
    common = real.var_names.intersection(synthetic.var_names)
    if len(common) == 0:
        raise ValueError("No common genes between real and synthetic AnnData")
    return real[:, common].copy(), synthetic[:, common].copy()


def subset_real_for_paradigm(
    real: ad.AnnData,
    dataset: str,
    mode: str,
    slice_id: Optional[str] = None,
) -> ad.AnnData:
    subset = real
    if "split" in subset.obs.columns:
        split = subset.obs["split"].astype(str)
        if (split == "train").any():
            subset = subset[split == "train"].copy()
        elif (split == "all").any():
            subset = subset[split == "all"].copy()

    if dataset == "DLPFC" and slice_id is not None:
        if "slice_id" not in subset.obs.columns:
            raise ValueError("DLPFC real AnnData is missing obs['slice_id']")
        subset = subset[subset.obs["slice_id"].astype(str) == str(slice_id)].copy()

    if subset.n_obs == 0:
        raise ValueError("No real observations remain for paradigm construction")
    return subset


def validate_synthetic_for_paradigm(
    synthetic: ad.AnnData,
    dataset: str,
    mode: str,
    slice_id: Optional[str] = None,
):
    if "spatial" not in synthetic.obsm:
        raise ValueError("Synthetic AnnData must contain obsm['spatial']")
    if mode == "supervised":
        if "label" not in synthetic.obs.columns:
            raise ValueError("Supervised synthetic must contain trusted obs['label']")
        label_strategy = str(synthetic.uns.get("label_strategy", "") or "")
        if label_strategy not in TRUSTED_LABEL_STRATEGIES:
            raise ValueError(
                "Supervised synthetic labels require a trusted source. Expected "
                f"label_strategy in {sorted(TRUSTED_LABEL_STRATEGIES)}, got {label_strategy!r}. "
                "Regenerate supervised synthetic data with label-wise generation."
            )
    if dataset == "DLPFC":
        if "slice_id" not in synthetic.obs.columns:
            raise ValueError("DLPFC synthetic is missing obs['slice_id']")
        syn_slices = sorted(synthetic.obs["slice_id"].astype(str).unique().tolist())
        if slice_id is not None:
            if len(syn_slices) != 1:
                raise ValueError(f"DLPFC per-slice synthetic should contain one slice, found {syn_slices}")
            if syn_slices[0] != str(slice_id):
                raise ValueError(f"Synthetic slice {syn_slices[0]} != requested {slice_id}")
        if mode == "unsupervised" and "label" in synthetic.obs.columns:
            raise ValueError("Unsupervised synthetic must not contain label")
    if dataset == "Trastuzumab":
        if "slice_id" in synthetic.obs.columns:
            raise ValueError("Trastuzumab synthetic must not contain slice_id")
        if "sample_id" not in synthetic.obs.columns:
            raise ValueError("Trastuzumab synthetic is missing sample_id")


def load_yaml(path: str | Path | None) -> dict:
    if path is None:
        return {}
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_capability(model: str, config_path: str | Path = "configs/spatial_cluster/DLPFC/model_capabilities.yaml") -> dict:
    cfg = load_yaml(config_path)
    return dict((cfg or {}).get(model, {}) or {})


def parse_float_list(values: Optional[list[str]], default: list[float]) -> list[float]:
    if not values:
        return default
    out: list[float] = []
    for value in values:
        out.extend(float(x) for x in str(value).split(",") if x != "")
    return out


def parse_str_list(values: Optional[list[str]], default: list[str]) -> list[str]:
    if not values:
        return default
    out: list[str] = []
    for value in values:
        out.extend(x for x in str(value).split(",") if x)
    return out


def subset_fraction(adata: ad.AnnData, fraction: float, seed: int, group_cols: Iterable[str]) -> ad.AnnData:
    if not 0 < fraction <= 1:
        raise ValueError("real_fraction must be in (0, 1]")
    if fraction >= 1:
        return adata.copy()
    obs = adata.obs.copy()
    cols = [col for col in group_cols if col in obs.columns]
    rng = np.random.default_rng(seed)
    if not cols:
        n = max(1, int(round(adata.n_obs * fraction)))
        return adata[np.sort(rng.choice(adata.n_obs, size=n, replace=False))].copy()
    key = obs[cols].astype(str).agg("||".join, axis=1)
    key.index = np.arange(adata.n_obs)
    chosen = []
    for _, idx in key.groupby(key).groups.items():
        idx = np.asarray(list(idx))
        n = max(1, int(round(idx.size * fraction)))
        chosen.extend(rng.choice(idx, size=min(n, idx.size), replace=False).tolist())
    return adata[np.sort(np.asarray(chosen, dtype=int))].copy()


def limit_synthetic_like_real(
    real: ad.AnnData,
    synthetic: ad.AnnData,
    ratio: float,
    seed: int,
) -> ad.AnnData:
    if ratio <= 0:
        raise ValueError("synthetic_ratio must be positive")
    group_cols = [col for col in ("slice_id", "label") if col in real.obs.columns and col in synthetic.obs.columns]
    rng = np.random.default_rng(seed)
    if not group_cols:
        n = min(synthetic.n_obs, max(1, int(round(real.n_obs * ratio))))
        return synthetic[np.sort(rng.choice(synthetic.n_obs, size=n, replace=False))].copy()

    real_key = real.obs[group_cols].astype(str).agg("||".join, axis=1)
    syn_key = synthetic.obs[group_cols].astype(str).agg("||".join, axis=1)
    chosen = []
    for key, real_idx in real_key.groupby(real_key).groups.items():
        syn_idx = np.flatnonzero(syn_key.to_numpy() == key)
        if syn_idx.size == 0:
            continue
        n = min(syn_idx.size, max(1, int(round(len(real_idx) * ratio))))
        chosen.extend(rng.choice(syn_idx, size=n, replace=False).tolist())
    if not chosen:
        raise ValueError("No synthetic observations matched real grouping for ratio selection")
    return synthetic[np.sort(np.asarray(chosen, dtype=int))].copy()


def nearest_local_synthetic(
    real: ad.AnnData,
    synthetic: ad.AnnData,
    ratio: float,
    seed: int,
    group_cols: Iterable[str] = ("slice_id",),
) -> ad.AnnData:
    if "spatial" not in real.obsm or "spatial" not in synthetic.obsm:
        raise ValueError("local_spatial requires obsm['spatial'] in real and synthetic")
    cols = [col for col in group_cols if col in real.obs.columns and col in synthetic.obs.columns]
    rng = np.random.default_rng(seed)
    chosen = []
    real_key = real.obs[cols].astype(str).agg("||".join, axis=1) if cols else pd.Series("all", index=np.arange(real.n_obs))
    real_key.index = np.arange(real.n_obs)
    syn_key = (
        synthetic.obs[cols].astype(str).agg("||".join, axis=1)
        if cols
        else pd.Series("all", index=np.arange(synthetic.n_obs))
    )
    syn_key.index = np.arange(synthetic.n_obs)
    for key, real_idx in real_key.groupby(real_key).groups.items():
        real_pos = np.asarray(list(real_idx), dtype=int)
        syn_pos = np.flatnonzero(syn_key.to_numpy() == key)
        if syn_pos.size == 0:
            continue
        n = min(syn_pos.size, max(1, int(round(real_pos.size * ratio))))
        nn = NearestNeighbors(n_neighbors=min(max(1, n), syn_pos.size))
        nn.fit(np.asarray(synthetic.obsm["spatial"])[syn_pos])
        order = nn.kneighbors(np.asarray(real.obsm["spatial"])[real_pos], return_distance=False).ravel()
        candidates = syn_pos[np.unique(order)]
        if candidates.size < n:
            extra = np.setdiff1d(syn_pos, candidates)
            if extra.size:
                candidates = np.concatenate(
                    [candidates, rng.choice(extra, size=min(n - candidates.size, extra.size), replace=False)]
                )
        chosen.extend(candidates[:n].tolist())
    if not chosen:
        raise ValueError("No local synthetic observations selected")
    return synthetic[np.sort(np.unique(chosen))].copy()


def _spatial_grid_regions(adata: ad.AnnData, n_bins: int = 2) -> pd.Series:
    """Assign slice-aware spatial grid regions used by adaptive local sampling."""
    if "spatial" not in adata.obsm:
        raise ValueError("Spatial grid regions require obsm['spatial']")
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    slices = (
        adata.obs["slice_id"].astype(str).to_numpy()
        if "slice_id" in adata.obs.columns
        else np.repeat("all", adata.n_obs)
    )
    out = np.empty(adata.n_obs, dtype=object)
    for sid in sorted(set(slices)):
        idx = np.flatnonzero(slices == sid)
        part = coords[idx]
        lo = part.min(axis=0)
        hi = part.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        bins = np.floor((part - lo) / span * n_bins).astype(int).clip(0, n_bins - 1)
        out[idx] = [f"slice_{sid}_grid_{x}_{y}" for x, y in bins]
    return pd.Series(out, index=np.arange(adata.n_obs), dtype="string")


def _spatial_grid_regions_with_reference(
    adata: ad.AnnData,
    reference: ad.AnnData,
    n_bins: int = 2,
) -> pd.Series:
    """Assign regions to ``adata`` using per-slice coordinate ranges from ``reference``."""
    if "spatial" not in adata.obsm or "spatial" not in reference.obsm:
        raise ValueError("Spatial grid regions require obsm['spatial']")
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    ref_coords = np.asarray(reference.obsm["spatial"], dtype=np.float64)
    slices = (
        adata.obs["slice_id"].astype(str).to_numpy()
        if "slice_id" in adata.obs.columns
        else np.repeat("all", adata.n_obs)
    )
    ref_slices = (
        reference.obs["slice_id"].astype(str).to_numpy()
        if "slice_id" in reference.obs.columns
        else np.repeat("all", reference.n_obs)
    )
    out = np.empty(adata.n_obs, dtype=object)
    for sid in sorted(set(slices)):
        idx = np.flatnonzero(slices == sid)
        ref_idx = np.flatnonzero(ref_slices == sid)
        ref_part = ref_coords[ref_idx] if ref_idx.size else coords[idx]
        lo = ref_part.min(axis=0)
        hi = ref_part.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        bins = np.floor((coords[idx] - lo) / span * n_bins).astype(int).clip(0, n_bins - 1)
        out[idx] = [f"slice_{sid}_grid_{x}_{y}" for x, y in bins]
    return pd.Series(out, index=np.arange(adata.n_obs), dtype="string")


def adaptive_local_synthetic(
    real: ad.AnnData,
    synthetic: ad.AnnData,
    max_ratio: float,
    seed: int,
    n_bins: int = 2,
) -> ad.AnnData:
    """Select local synthetic spots with region-specific ratios capped by availability.

    ``max_ratio`` is the largest allowed per-region synthetic/real ratio. For
    DLPFC we map spatial regions to ratios in ``1..max_ratio``: sparser real
    regions receive higher ratios, denser regions receive lower ratios. The
    final number in every region is capped by available synthetic spots.
    """
    if max_ratio <= 0:
        raise ValueError("local_spatial max_ratio must be positive")
    if "spatial" not in real.obsm or "spatial" not in synthetic.obsm:
        raise ValueError("adaptive local_spatial requires obsm['spatial'] in real and synthetic")
    max_ratio_i = max(1, int(round(float(max_ratio))))
    ratio_levels = np.arange(1, max_ratio_i + 1, dtype=int)
    rng = np.random.default_rng(seed)
    real_region = _spatial_grid_regions(real, n_bins=n_bins)
    syn_region = _spatial_grid_regions_with_reference(synthetic, real, n_bins=n_bins)
    real_counts = real_region.value_counts().sort_index()
    ordered_regions = real_counts.sort_values(ascending=False).index.tolist()
    if len(ordered_regions) == 1:
        region_ratio = {ordered_regions[0]: max_ratio_i}
    else:
        # Dense regions get lower ratios; sparse regions get higher ratios.
        region_ratio = {}
        for rank, region in enumerate(ordered_regions):
            level_idx = int(round(rank * (len(ratio_levels) - 1) / (len(ordered_regions) - 1)))
            region_ratio[str(region)] = int(ratio_levels[level_idx])

    chosen: list[int] = []
    real_coords = np.asarray(real.obsm["spatial"], dtype=np.float64)
    syn_coords = np.asarray(synthetic.obsm["spatial"], dtype=np.float64)
    syn_region_arr = syn_region.astype(str).to_numpy()
    for region, real_idx_values in real_region.groupby(real_region).groups.items():
        region = str(region)
        real_idx = np.asarray(list(real_idx_values), dtype=int)
        syn_idx = np.flatnonzero(syn_region_arr == region)
        if real_idx.size == 0 or syn_idx.size == 0:
            continue
        target_ratio = int(region_ratio.get(region, max_ratio_i))
        n_target = min(syn_idx.size, max(1, int(round(real_idx.size * target_ratio))))
        nn = NearestNeighbors(n_neighbors=min(max(1, n_target), syn_idx.size))
        nn.fit(syn_coords[syn_idx])
        order = nn.kneighbors(real_coords[real_idx], return_distance=False).ravel()
        candidates = syn_idx[np.unique(order)]
        if candidates.size < n_target:
            extra = np.setdiff1d(syn_idx, candidates)
            if extra.size:
                candidates = np.concatenate(
                    [candidates, rng.choice(extra, size=min(n_target - candidates.size, extra.size), replace=False)]
                )
        chosen.extend(candidates[:n_target].tolist())
    if not chosen:
        raise ValueError("No adaptive local synthetic observations selected")
    out = synthetic[np.sort(np.unique(chosen))].copy()
    out.uns["local_spatial_region_policy"] = "slice_spatial_grid_sparse_regions_higher_ratio"
    out.uns["local_spatial_max_ratio"] = int(max_ratio_i)
    return out


def load_regional_alpha_file(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    df = pd.read_csv(resolve_path(path))
    if "region_id" not in df.columns or "alpha" not in df.columns:
        raise ValueError("--regional-alpha-file must contain columns: region_id, alpha")
    return {str(row["region_id"]): float(row["alpha"]) for _, row in df.iterrows()}


def assign_region_ids(
    real: ad.AnnData,
    synthetic: ad.AnnData,
    dataset: str,
    n_bins: int = 2,
) -> tuple[ad.AnnData, ad.AnnData]:
    real = real.copy()
    synthetic = synthetic.copy()
    if dataset == "DLPFC" and "spatial" in real.obsm and "spatial" in synthetic.obsm:
        real.obs["region_id"] = _spatial_grid_regions(real, n_bins=n_bins).to_numpy(dtype=str)
        synthetic.obs["region_id"] = _spatial_grid_regions_with_reference(synthetic, real, n_bins=n_bins).to_numpy(dtype=str)
        return real, synthetic

    if "label" in real.obs.columns and "label" in synthetic.obs.columns:
        real.obs["region_id"] = "label_" + real.obs["label"].astype(str)
        synthetic.obs["region_id"] = "label_" + synthetic.obs["label"].astype(str)
        return real, synthetic

    real.obs["region_id"] = "all"
    synthetic.obs["region_id"] = "all"
    return real, synthetic


def ensure_obs_columns(real: ad.AnnData, synthetic: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData]:
    cols = list(dict.fromkeys(list(real.obs.columns) + list(synthetic.obs.columns)))
    real = real.copy()
    synthetic = synthetic.copy()
    for col in cols:
        if col not in real.obs:
            real.obs[col] = pd.NA
        if col not in synthetic.obs:
            synthetic.obs[col] = pd.NA
    real.obs = real.obs[cols]
    synthetic.obs = synthetic.obs[cols]
    return real, synthetic


def concat_train(real: ad.AnnData, synthetic: ad.AnnData) -> ad.AnnData:
    real, synthetic = ensure_obs_columns(real, synthetic)
    merged = ad.concat(
        [real, synthetic],
        axis=0,
        join="outer",
        merge="same",
        label="augmentation_source",
        keys=["real", "synthetic"],
        index_unique=None,
    )
    merged.var = real.var.copy()
    if sparse.issparse(real.X) or sparse.issparse(synthetic.X):
        merged.X = sparse.csr_matrix(merged.X)
    merged.obs_names_make_unique()
    return sanitize_obs_for_h5ad(merged)


def add_weight_columns(
    adata: ad.AnnData,
    alpha: float,
    regional: bool = False,
    alpha_vector: Optional[dict[str, float]] = None,
) -> ad.AnnData:
    adata = adata.copy()
    source = adata.obs["augmentation_source"].astype(str)
    real_mask = source == "real"
    syn_mask = source == "synthetic"
    if not real_mask.any() or not syn_mask.any():
        raise ValueError("weighted ERM packages require both real and synthetic observations")
    weights = np.zeros(adata.n_obs, dtype=np.float64)
    alpha_vector = alpha_vector or {}
    if not regional or "region_id" not in adata.obs.columns:
        weights[real_mask.to_numpy()] = float(alpha) / int(real_mask.sum())
        weights[syn_mask.to_numpy()] = (1.0 - float(alpha)) / int(syn_mask.sum())
    else:
        regions = adata.obs["region_id"].astype(str)
        for region_id in regions.unique():
            region_alpha = float(alpha_vector.get(str(region_id), alpha))
            region = regions == region_id
            r = real_mask & region
            s = syn_mask & region
            if r.any():
                weights[r.to_numpy()] = region_alpha / int(r.sum())
            if s.any():
                weights[s.to_numpy()] = (1.0 - region_alpha) / int(s.sum())
    adata.obs["loss_weight"] = weights
    adata.obs["erm_group"] = source.values
    return adata


def variant_id(
    family: str,
    synthetic_ratio: float,
    real_fraction: float,
    alpha: Optional[float],
    selection_policy: str,
) -> str:
    parts = [
        family,
        f"syn{synthetic_ratio:g}",
        f"real{real_fraction:g}",
        selection_policy,
    ]
    if alpha is not None:
        parts.append(f"alpha{alpha:g}")
    return "_".join(part.replace(".", "p").replace("/", "-") for part in parts)


def write_manifest(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_package(
    output_root: Path,
    family: str,
    variant: str,
    manifest: dict,
    train: Optional[ad.AnnData] = None,
    pretrain: Optional[ad.AnnData] = None,
    finetune: Optional[ad.AnnData] = None,
) -> Path:
    out = output_root / family / variant
    out.mkdir(parents=True, exist_ok=True)
    if train is not None:
        train.write_h5ad(out / "train.h5ad")
        manifest["train_path"] = str(out / "train.h5ad")
    if pretrain is not None:
        pretrain.write_h5ad(out / "pretrain_synthetic.h5ad")
        manifest["pretrain_path"] = str(out / "pretrain_synthetic.h5ad")
    if finetune is not None:
        finetune.write_h5ad(out / "finetune_real.h5ad")
        manifest["finetune_path"] = str(out / "finetune_real.h5ad")
    write_manifest(out / "manifest.json", manifest)
    return out


def build_manifest(
    *,
    family: str,
    variant: str,
    dataset: str,
    mode: str,
    model: str,
    synthetic_ratio: float,
    real_fraction: float,
    alpha: Optional[float],
    selection_policy: str,
    real: ad.AnnData,
    synthetic: ad.AnnData,
    capability: dict,
    training_regime: str,
    pca_fit_policy: str = "real_train_only",
    alpha_vector: Optional[dict[str, float]] = None,
    coord_policy: str = "pool_default",
) -> dict:
    payload = {
        "paradigm_family": family,
        "variant_id": variant,
        "dataset": dataset,
        "mode": mode,
        "model": model,
        "generator": model,
        "generator_capability": capability,
        "synthetic_ratio": float(synthetic_ratio),
        "real_fraction": float(real_fraction),
        "alpha": None if alpha is None else float(alpha),
        "alpha_vector": alpha_vector or {},
        "sampling_strategy": selection_policy,
        "selection_policy": selection_policy,
        "training_regime": training_regime,
        "pca_fit_policy": pca_fit_policy,
        "condition_source": str(synthetic.uns.get("condition_source", "")),
        "coord_policy": str(coord_policy),
        "coord_source": str(synthetic.uns.get("coord_source", synthetic.uns.get("coord_method", ""))),
        "n_real_used": int(real.n_obs),
        "n_synthetic_used": int(synthetic.n_obs),
    }
    for key in ("task_mode", "label_fraction", "low_label_seed"):
        if key in real.uns:
            payload[key] = real.uns[key]
        elif key in synthetic.uns:
            payload[key] = synthetic.uns[key]
    return payload


def build_memory_package(
    *,
    family: str,
    real: ad.AnnData,
    synthetic: ad.AnnData,
    dataset: str,
    mode: str,
    model: str,
    capability: dict,
    synthetic_ratio: float,
    real_fraction: float,
    seed: int,
    pca_fit_policy: str,
    alpha: Optional[float] = None,
    loss_file: str | Path | None = None,
    regional_alpha: Optional[dict[str, float]] = None,
    coord_policy: str = "pool_default",
    mapping_config: str | Path = "configs/spatial_cluster/DLPFC/mapping.yaml",
    reference_path: str | Path = "",
) -> ParadigmPackage:
    """Build one paradigm package in memory and return it to the caller."""
    if family == "local_spatial":
        syn_subset = adaptive_local_synthetic(real, synthetic, synthetic_ratio, seed)
        selection_policy = f"adaptive_region_local_spatial_max{synthetic_ratio:g}_{coord_policy}"
    elif family == "global_real_plus_synthetic":
        syn_subset = limit_synthetic_like_real(real, synthetic, synthetic_ratio, seed)
        selection_policy = f"matched_group_ratio_{coord_policy}"
    else:
        raise ValueError(f"Unsupported DLPFC spatial-cluster paradigm family: {family}")

    syn_subset = apply_coord_policy(syn_subset, real, coord_policy, mapping_config, reference_path)
    manifest_kwargs = {
        "family": family,
        "variant": variant_id(family, synthetic_ratio, real_fraction, alpha, selection_policy),
        "dataset": dataset,
        "mode": mode,
        "model": model,
        "synthetic_ratio": synthetic_ratio,
        "real_fraction": real_fraction,
        "alpha": alpha,
        "selection_policy": selection_policy,
        "real": real,
        "synthetic": syn_subset,
        "capability": capability,
        "pca_fit_policy": pca_fit_policy,
        "coord_policy": coord_policy,
    }

    if family == "global_real_plus_synthetic":
        train = concat_train(real, syn_subset)
        manifest = build_manifest(**manifest_kwargs, training_regime="concat")
        return ParadigmPackage(train=train, manifest=manifest)

    if family == "local_spatial":
        train = concat_train(real, syn_subset)
        manifest = build_manifest(**manifest_kwargs, training_regime="concat")
        return ParadigmPackage(train=train, manifest=manifest)

    raise ValueError(f"Unknown paradigm family: {family}")


def infer_task_name(dataset: str, mode: str, task_name: Optional[str] = None) -> str:
    if task_name:
        return str(task_name)
    if dataset == "DLPFC" and mode == "supervised":
        return "spot_clf"
    if dataset == "DLPFC" and mode == "unsupervised":
        return "spatial_cluster"
    if dataset == "Trastuzumab" and mode == "supervised":
        return "response_pred"
    return "unknown_task"


def standard_output_root(base: str | Path, model: str, dataset: str, task_name: str, mode: str) -> Path:
    return resolve_path(base) / task_name / dataset / model / mode


def annotate_manifest(output: Path, task_name: str, standard_layout: bool, synthetic_pool_path: str | Path):
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["task_name"] = task_name
    payload["standard_layout"] = bool(standard_layout)
    payload["synthetic_pool_path"] = str(resolve_path(synthetic_pool_path))
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_paradigm_inputs(
    real_path: str | Path,
    synthetic_pool_path: str | Path,
    output_root: str | Path,
    dataset: Optional[str],
    mode: Optional[str],
    model: Optional[str],
    task_name: Optional[str],
    standard_layout: bool,
) -> tuple[ad.AnnData, ad.AnnData, Path, str, str, str, str]:
    real_path = resolve_path(real_path)
    synthetic_pool_path = resolve_path(synthetic_pool_path)
    real = ad.read_h5ad(real_path)
    synthetic = ad.read_h5ad(synthetic_pool_path)
    dataset = dataset or infer_dataset(real, real_path)
    mode = mode or infer_mode(real, real_path)
    if model is None:
        model = str(synthetic.uns.get("generator", "") or "")
    if not model:
        model = next((name for name in KNOWN_GENERATORS if name in synthetic_pool_path.parts), "unknown")
    task_name = infer_task_name(dataset, mode, task_name)
    out_root = (
        standard_output_root(output_root, model=model, dataset=dataset, task_name=task_name, mode=mode)
        if standard_layout
        else resolve_path(output_root)
    )
    validate_synthetic_for_paradigm(synthetic, dataset=dataset, mode=mode)
    real_train = subset_real_for_paradigm(real, dataset=dataset, mode=mode)
    real_train, synthetic = align_gene_space(real_train, synthetic, strategy="intersection")
    return real_train, synthetic, out_root, dataset, mode, model, task_name


def add_common_cli_args(parser):
    parser.add_argument("-r", "--real", required=True)
    parser.add_argument("-s", "--synthetic-pool", required=True)
    parser.add_argument(
        "-o",
        "--output-root",
        default="data/04_paradigm_augmented",
        help="Base output directory. By default, task/dataset/model/mode are appended.",
    )
    parser.add_argument("-c", "--config", default="configs/spatial_cluster/DLPFC/paradigm.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--mode", default=None, choices=[None, "supervised", "unsupervised"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--ratios", nargs="*", default=None, help="Synthetic ratios sampled from the 40x pool.")
    parser.add_argument("--real-fractions", nargs="*", default=None)
    parser.add_argument("--seed", "--random-seed", dest="random_seed", type=int, default=42)
    parser.add_argument("--no-standard-layout", action="store_true")
    parser.add_argument(
        "--coord-policy",
        default="pool_default",
        choices=sorted(COORD_POLICIES),
        help="Coordinate ablation applied after sampling this paradigm package.",
    )
    parser.add_argument(
        "--mapping-config",
        default="configs/spatial_cluster/DLPFC/mapping.yaml",
        help="Coordinate mapping config used with non-default coordinate policies.",
    )


def apply_coord_policy(
    synthetic: ad.AnnData,
    real_reference: ad.AnnData,
    coord_policy: str,
    mapping_config: str | Path,
    reference_path: str | Path,
) -> ad.AnnData:
    if coord_policy == "pool_default":
        return synthetic.copy()
    if coord_policy not in COORD_POLICIES:
        raise ValueError(f"Unknown coord_policy={coord_policy}")
    cfg = load_mapping_yaml(mapping_config)
    out_parts = []
    group_cols = []
    if "slice_id" in synthetic.obs.columns:
        group_cols.append("slice_id")
    if (
        str(synthetic.uns.get("label_strategy", "") or "") == "label_wise_generation"
        and "label" in synthetic.obs.columns
        and "label" in real_reference.obs.columns
    ):
        group_cols.append("label")
    if group_cols:
        keys = synthetic.obs[group_cols].astype(str).agg("||".join, axis=1)
        keys.index = np.arange(synthetic.n_obs)
        for _, idx in keys.groupby(keys).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            syn_part = synthetic[idx].copy()
            slice_id = str(syn_part.obs["slice_id"].astype(str).iloc[0]) if "slice_id" in syn_part.obs else None
            mapped, _ = assign_coordinates_to_adata(
                syn_part,
                real_reference,
                method=coord_policy,
                config=cfg,
                reference_path=reference_path,
                slice_id=slice_id,
            )
            out_parts.append(mapped)
        out = ad.concat(out_parts, join="outer", merge="same", index_unique=None)
        out.obs_names_make_unique()
    else:
        out, _ = assign_coordinates_to_adata(
            synthetic,
            real_reference,
            method=coord_policy,
            config=cfg,
            reference_path=reference_path,
            slice_id=None,
        )
    out.uns.update(synthetic.uns)
    out.uns["coord_method"] = coord_policy
    out.uns["coord_source"] = f"{coord_policy}_paradigm_ablation"
    return out


def build_single_family_packages(
    *,
    builder,
    family: str,
    real_path: str | Path,
    synthetic_pool_path: str | Path,
    output_root: str | Path,
    config_path: str | Path,
    dataset: Optional[str],
    mode: Optional[str],
    model: Optional[str],
    task_name: Optional[str],
    standard_layout: bool,
    synthetic_ratios: Optional[list[float]],
    real_fractions: Optional[list[float]],
    alphas: Optional[list[float]],
    random_seed: int,
    coord_policy: str = "pool_default",
    mapping_config: str | Path = "configs/spatial_cluster/DLPFC/mapping.yaml",
    extra_kwargs: Optional[dict] = None,
) -> list[Path]:
    suite_cfg = load_yaml(config_path).get("paradigm_suite", {})
    family_ratio_key = "local_spatial_max_ratios" if family == "local_spatial" else "synthetic_ratios"
    synthetic_ratios = synthetic_ratios or suite_cfg.get(family_ratio_key, DEFAULT_SYNTHETIC_RATIOS)
    real_fractions = real_fractions or suite_cfg.get("real_fractions", DEFAULT_REAL_FRACTIONS)
    pca_fit_policy = str(suite_cfg.get("pca_fit_policy", "real_train_only"))
    real_train, synthetic, out_root, dataset, mode, model, task_name = load_paradigm_inputs(
        real_path=real_path,
        synthetic_pool_path=synthetic_pool_path,
        output_root=output_root,
        dataset=dataset,
        mode=mode,
        model=model,
        task_name=task_name,
        standard_layout=standard_layout,
    )
    capability = load_capability(model)
    if dataset != "DLPFC" and family in DLPFC_ONLY_FAMILIES:
        raise ValueError(f"{family} is only defined for DLPFC spatial tasks")
    if mode == "unsupervised" and family in UNSUPERVISED_INCOMPATIBLE_FAMILIES:
        raise ValueError(f"{family} requires supervised labels and cannot run in unsupervised mode")
    outputs: list[Path] = []
    extra_kwargs = extra_kwargs or {}
    extra_kwargs["reference_path"] = real_path
    for real_fraction in real_fractions:
        real_subset = subset_fraction(
            real_train,
            real_fraction,
            seed=random_seed,
            group_cols=("slice_id", "label"),
        )
        for synthetic_ratio in synthetic_ratios:
            kwargs = dict(extra_kwargs)
            kwargs["coord_policy"] = coord_policy
            kwargs["mapping_config"] = mapping_config
            output = builder(
                real=real_subset,
                synthetic=synthetic,
                output_root=out_root,
                dataset=dataset,
                mode=mode,
                model=model,
                capability=capability,
                synthetic_ratio=synthetic_ratio,
                real_fraction=real_fraction,
                seed=random_seed,
                pca_fit_policy=pca_fit_policy,
                **kwargs,
            )
            annotate_manifest(output, task_name, standard_layout, synthetic_pool_path)
            outputs.append(output)
    return outputs
