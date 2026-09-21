"""Shared coordinate assignment utilities for non-spatial generators.

The low-label generation default for scGAN/scDiffusion is
``label_spatial_perturbation``: sample coordinates from the same slice and
label training spots, then add a small jitter. Other coordinate policies are
kept here for downstream/paradigm ablations so the 40x synthetic pool is not
materialized three times.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import yaml
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
GENERATOR_COMMON_PATH = Path(__file__).resolve().parent / "common.py"
_spec = importlib.util.spec_from_file_location("_generator_common", GENERATOR_COMMON_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load generator common helpers from {GENERATOR_COMMON_PATH}")
_generator_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generator_common)
infer_dataset = _generator_common.infer_dataset
infer_mode = _generator_common.infer_mode
select_hvg = _generator_common.select_hvg
validate_no_unsupervised_leakage = _generator_common.validate_no_unsupervised_leakage

logger = logging.getLogger("coord_assignment")


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def to_dense(x) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def validate_gene_order(syn_adata: ad.AnnData, ref_adata: ad.AnnData):
    if syn_adata.n_vars != ref_adata.n_vars or not syn_adata.var_names.equals(ref_adata.var_names):
        raise ValueError(
            "Synthetic and reference HVG gene spaces do not match exactly. "
            f"synthetic_n_vars={syn_adata.n_vars}, reference_n_vars={ref_adata.n_vars}, "
            f"synthetic_first5={syn_adata.var_names[:5].tolist()}, "
            f"reference_first5={ref_adata.var_names[:5].tolist()}"
        )


def filter_reference_by_label_condition(ref_adata: ad.AnnData, syn_adata: ad.AnnData, mode: str) -> ad.AnnData:
    if mode != "supervised":
        return ref_adata
    if str(syn_adata.uns.get("label_strategy", "") or "") != "label_wise_generation":
        return ref_adata
    if "label" not in syn_adata.obs or "label" not in ref_adata.obs:
        raise ValueError("Label-wise coordinate mapping requires obs['label'] in synthetic and reference")
    labels = sorted(syn_adata.obs["label"].astype(str).unique().tolist())
    if len(labels) != 1:
        raise ValueError(f"Coordinate assignment expects one label-wise condition, found labels={labels}")
    label = labels[0]
    filtered = ref_adata[ref_adata.obs["label"].astype(str) == label].copy()
    if filtered.n_obs == 0:
        raise ValueError(f"No reference observations found for label={label}")
    logger.info("Label-wise coordinate mapping: label=%s, reference=%d", label, filtered.n_obs)
    return filtered


def knn_mapping(
    synthetic_X: np.ndarray,
    reference_X: np.ndarray,
    reference_coords: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
    aggregation: str = "mean",
    pca_n_components: Optional[int] = None,
) -> np.ndarray:
    if pca_n_components and pca_n_components > 0:
        n_comp = min(pca_n_components, reference_X.shape[0] + synthetic_X.shape[0] - 1, reference_X.shape[1])
        if n_comp >= 1:
            pca = PCA(n_components=n_comp, random_state=42)
            combined = pca.fit_transform(np.vstack([reference_X, synthetic_X]))
            reference_X = combined[: reference_X.shape[0]]
            synthetic_X = combined[reference_X.shape[0] :]

    distances = cdist(synthetic_X, reference_X, metric=metric)
    k_eff = min(max(int(k), 1), reference_X.shape[0])
    assigned = np.zeros((synthetic_X.shape[0], 2), dtype=np.float64)
    for i in range(synthetic_X.shape[0]):
        idx = np.argpartition(distances[i], k_eff - 1)[:k_eff]
        if aggregation == "weighted":
            d = np.maximum(distances[i][idx], 1e-8)
            w = 1.0 / d
            w /= w.sum()
            assigned[i] = np.average(reference_coords[idx], axis=0, weights=w)
        else:
            assigned[i] = reference_coords[idx].mean(axis=0)
    return assigned


def spatial_perturbation(
    synthetic_X: np.ndarray,
    reference_X: np.ndarray,
    reference_coords: np.ndarray,
    base_k: int = 10,
    sigma_factor: float = 1.0,
    check_overlap: bool = True,
    min_distance_ratio: float = 0.5,
    max_resample: int = 10,
    random_seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    coords = knn_mapping(synthetic_X, reference_X, reference_coords, k=base_k)
    mean_dist = float(pdist(reference_coords).mean()) if reference_coords.shape[0] > 1 else 1.0
    min_dist = mean_dist * float(min_distance_ratio)
    sigma = mean_dist * float(sigma_factor)
    out = coords.copy()
    for i in range(out.shape[0]):
        for _ in range(max_resample):
            candidate = coords[i] + rng.normal(0.0, sigma, size=2)
            if not check_overlap or np.linalg.norm(reference_coords - candidate, axis=1).min() >= min_dist:
                out[i] = candidate
                break
        else:
            out[i] = coords[i] + rng.normal(0.0, sigma * 0.1, size=2)
    return out


def label_spatial_perturbation(
    reference_coords: np.ndarray,
    n_generate: int,
    sigma_factor: float = 0.08,
    min_sigma: float = 1e-6,
    random_seed: int = 42,
) -> np.ndarray:
    """Sample same-condition reference coordinates and add a small jitter."""
    reference_coords = np.asarray(reference_coords, dtype=np.float64)
    if reference_coords.shape[0] == 0:
        raise ValueError("Cannot sample coordinates from an empty reference")
    rng = np.random.default_rng(random_seed)
    base = reference_coords[rng.integers(0, reference_coords.shape[0], size=int(n_generate)), :2].astype(
        np.float64,
        copy=True,
    )
    sd = np.maximum(reference_coords[:, :2].std(axis=0) * float(sigma_factor), float(min_sigma))
    return base + rng.normal(0.0, sd, size=base.shape)


def gmm_sampling(
    reference_coords: np.ndarray,
    n_generate: int,
    n_components: int = 10,
    covariance_type: str = "full",
    sampling_mode: str = "low_density",
    density_threshold: float = 0.3,
    random_seed: int = 42,
) -> np.ndarray:
    n_components = min(max(int(n_components), 1), reference_coords.shape[0])
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_seed)
    gmm.fit(reference_coords)
    if sampling_mode == "uniform":
        samples, _ = gmm.sample(n_generate)
        return samples.astype(np.float64)
    weights = gmm.weights_.copy()
    mean_weight = weights.mean()
    adjusted = np.minimum(weights, mean_weight * float(density_threshold)) if density_threshold > 0 else 1.0 / (weights + 1e-8)
    adjusted = adjusted / adjusted.sum()
    counts = np.random.default_rng(random_seed).multinomial(n_generate, adjusted)
    samples = []
    for j, count in enumerate(counts):
        if count <= 0:
            continue
        cov = gmm.covariances_
        if covariance_type == "full":
            cov_j = cov[j]
        elif covariance_type == "diag":
            cov_j = np.diag(cov[j])
        elif covariance_type == "spherical":
            cov_j = np.eye(reference_coords.shape[1]) * cov[j]
        else:
            cov_j = cov
        samples.append(np.random.default_rng(random_seed + j).multivariate_normal(gmm.means_[j], cov_j, size=count))
    out = np.vstack(samples) if samples else np.zeros((0, 2), dtype=np.float64)
    np.random.default_rng(random_seed).shuffle(out)
    return out[:n_generate].astype(np.float64)


def prepare_reference_for_mapping(
    syn_adata: ad.AnnData,
    ref_adata: ad.AnnData,
    reference_path: str | Path,
    slice_id: Optional[str] = None,
) -> tuple[ad.AnnData, str, str, Optional[str]]:
    dataset = infer_dataset(ref_adata, str(reference_path))
    mode = infer_mode(ref_adata, str(reference_path))
    validate_no_unsupervised_leakage(ref_adata, mode)
    if "split" in ref_adata.obs:
        split = ref_adata.obs["split"].astype(str)
        mask = split == "train"
        if not mask.any() and (split == "all").any():
            mask = split == "all"
        ref_adata = ref_adata[mask].copy()

    if dataset == "DLPFC":
        if "slice_id" not in ref_adata.obs:
            raise ValueError("DLPFC reference must contain obs['slice_id']")
        if slice_id is None:
            if "slice_id" in syn_adata.obs:
                syn_slices = sorted(syn_adata.obs["slice_id"].astype(str).unique().tolist())
                if len(syn_slices) != 1:
                    raise ValueError(f"Assign coordinates per slice or pass --slice-id. Found {syn_slices}")
                slice_id = syn_slices[0]
            else:
                ref_slices = sorted(ref_adata.obs["slice_id"].astype(str).unique().tolist())
                if len(ref_slices) != 1:
                    raise ValueError(f"Pass --slice-id for DLPFC coordinate assignment. Found {ref_slices}")
                slice_id = ref_slices[0]
        ref_adata = ref_adata[ref_adata.obs["slice_id"].astype(str) == str(slice_id)].copy()
        if ref_adata.n_obs == 0:
            raise ValueError(f"No reference observations found for slice_id={slice_id}")

    ref_adata = filter_reference_by_label_condition(ref_adata, syn_adata, mode)
    ref_adata = select_hvg(ref_adata)
    validate_gene_order(syn_adata, ref_adata)
    return ref_adata, dataset, mode, slice_id


def assign_coordinates_to_adata(
    syn_adata: ad.AnnData,
    ref_adata: ad.AnnData,
    method: str,
    config: dict,
    reference_path: str | Path,
    slice_id: Optional[str] = None,
) -> tuple[ad.AnnData, Optional[str]]:
    ref_adata, _, _, slice_id = prepare_reference_for_mapping(syn_adata, ref_adata, reference_path, slice_id=slice_id)
    syn_X = to_dense(syn_adata.X).astype(np.float32)
    ref_X = to_dense(ref_adata.X).astype(np.float32)
    ref_coords = np.asarray(ref_adata.obsm["spatial"], dtype=np.float64)

    if method == "label_spatial_perturbation":
        cfg = config.get("label_spatial_perturbation", {})
        coords = label_spatial_perturbation(
            ref_coords,
            syn_X.shape[0],
            sigma_factor=cfg.get("sigma_factor", 0.08),
            min_sigma=cfg.get("min_sigma", 1e-6),
            random_seed=cfg.get("random_seed", 42),
        )
    elif method == "knn_mapping":
        cfg = config.get("knn_mapping", {})
        coords = knn_mapping(
            syn_X,
            ref_X,
            ref_coords,
            k=cfg.get("k", 10),
            metric=cfg.get("metric", "cosine"),
            aggregation=cfg.get("aggregation", "mean"),
            pca_n_components=cfg.get("pca_n_components"),
        )
    elif method == "spatial_perturbation":
        cfg = config.get("spatial_perturbation", {})
        knn_cfg = config.get("knn_mapping", {})
        coords = spatial_perturbation(
            syn_X,
            ref_X,
            ref_coords,
            base_k=knn_cfg.get("k", 10),
            sigma_factor=cfg.get("sigma_factor", 1.0),
            check_overlap=cfg.get("check_overlap", True),
            min_distance_ratio=cfg.get("min_distance_ratio", 0.5),
            max_resample=cfg.get("max_resample", 10),
            random_seed=cfg.get("random_seed", 42),
        )
    elif method == "gmm_sampling":
        cfg = config.get("gmm_sampling", {})
        coords = gmm_sampling(
            ref_coords,
            syn_X.shape[0],
            n_components=cfg.get("n_components", 10),
            covariance_type=cfg.get("covariance_type", "full"),
            sampling_mode=cfg.get("sampling_mode", "low_density"),
            density_threshold=cfg.get("density_threshold", 0.3),
            random_seed=cfg.get("random_seed", 42),
        )
    else:
        raise ValueError(f"Unknown coordinate assignment method: {method}")

    out = syn_adata.copy()
    out.obsm["spatial"] = coords.astype(np.float64)
    out.uns["coord_method"] = method
    out.uns["coord_source"] = method
    if slice_id is not None:
        out.uns["coord_slice_id"] = str(slice_id)
    return out, slice_id


def assign_coordinates(
    synthetic_path: str | Path,
    reference_path: str | Path,
    method: str = "knn_mapping",
    config_path: str | Path = "configs/supervised_low_label/DLPFC/mapping.yaml",
    slice_id: Optional[str] = None,
    generation_default: bool = False,
) -> str:
    synthetic_path = resolve_path(synthetic_path)
    reference_path = resolve_path(reference_path)
    config = load_yaml(config_path)
    syn = ad.read_h5ad(str(synthetic_path))
    ref = ad.read_h5ad(str(reference_path))
    original_uns = dict(syn.uns)
    group_cols = []
    if slice_id is None and "slice_id" in syn.obs:
        group_cols.append("slice_id")
    if (
        str(syn.uns.get("label_strategy", "") or "") == "label_wise_generation"
        and "label" in syn.obs
        and "label" in ref.obs
    ):
        group_cols.append("label")
    if group_cols:
        original_var = syn.var.copy()
        parts = []
        keys = syn.obs[group_cols].astype(str).agg("||".join, axis=1)
        keys.index = np.arange(syn.n_obs)
        for _, idx in keys.groupby(keys).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            syn_part = syn[idx].copy()
            part_slice = str(syn_part.obs["slice_id"].astype(str).iloc[0]) if "slice_id" in syn_part.obs else slice_id
            mapped, _ = assign_coordinates_to_adata(
                syn_part,
                ref,
                method,
                config,
                reference_path,
                slice_id=part_slice,
            )
            parts.append(mapped)
        syn = ad.concat(parts, join="outer", merge="same", index_unique=None)
        syn.obs_names_make_unique()
        syn = syn[:, original_var.index].copy()
        syn.var = original_var.loc[syn.var_names].copy()
        syn.uns.update(original_uns)
        syn.uns["coord_method"] = method
        syn.uns["coord_source"] = method
        if slice_id is not None:
            syn.uns["coord_slice_id"] = str(slice_id)
        elif "slice_id" in syn.obs:
            syn_slices = sorted(syn.obs["slice_id"].astype(str).unique().tolist())
            if len(syn_slices) == 1:
                syn.uns["coord_slice_id"] = syn_slices[0]
        syn.uns["n_generated"] = int(syn.n_obs)
        syn.uns["n_genes"] = int(syn.n_vars)
    else:
        syn, slice_id = assign_coordinates_to_adata(syn, ref, method, config, reference_path, slice_id=slice_id)
    if generation_default:
        syn.uns["coord_source"] = f"{method}_generation_default"
    syn.write(str(synthetic_path))
    return str(synthetic_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assign coordinates to synthetic AnnData.")
    parser.add_argument("--synthetic", "-s", required=True)
    parser.add_argument("--reference", "-r", required=True)
    parser.add_argument(
        "--method",
        "-m",
        default="label_spatial_perturbation",
        choices=["label_spatial_perturbation", "knn_mapping", "spatial_perturbation", "gmm_sampling"],
    )
    parser.add_argument("--config", "-c", default="configs/supervised_low_label/DLPFC/mapping.yaml")
    parser.add_argument("--slice-id", default=None)
    parser.add_argument("--generation-default", action="store_true")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    assign_coordinates(
        synthetic_path=args.synthetic,
        reference_path=args.reference,
        method=args.method,
        config_path=args.config,
        slice_id=args.slice_id,
        generation_default=args.generation_default,
    )


if __name__ == "__main__":
    main()
