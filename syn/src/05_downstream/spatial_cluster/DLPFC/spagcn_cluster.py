"""Spatial clustering for DLPFC augmented AnnData.

The formal DLPFC spatial clustering experiment uses one downstream method:
PCA expression features concatenated with scaled spatial coordinates, followed
by kNN graph construction and Leiden clustering. Evaluation labels are loaded
only after clustering.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path = [p for p in sys.path if Path(p or os.getcwd()).resolve() != PROJECT_ROOT]

import anndata as ad
LEAKAGE_COLUMNS = {"label", "spatialLIBD", "cluster", "ground_truth", "manual_annotation"}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_manifest(path: str | Path | dict | None) -> dict:
    if path is None:
        return {}
    if isinstance(path, dict):
        return dict(path)
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def to_dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def validate_unsupervised_input(adata: ad.AnnData):
    leakage = sorted(c for c in LEAKAGE_COLUMNS if c in adata.obs.columns)
    if leakage:
        raise ValueError(f"Unsupervised clustering input contains label columns: {leakage}")
    if "spatial" not in adata.obsm:
        raise ValueError("Spatial clustering input requires obsm['spatial']")
    if "slice_id" not in adata.obs.columns:
        raise ValueError("DLPFC spatial clustering input requires obs['slice_id']")


def subset_slice(adata: ad.AnnData, slice_id: Optional[str]) -> ad.AnnData:
    if slice_id is None:
        slices = sorted(adata.obs["slice_id"].astype(str).unique().tolist())
        if len(slices) != 1:
            raise ValueError(f"Pass --slice-id for multi-slice clustering input: {slices}")
        slice_id = slices[0]
    subset = adata[adata.obs["slice_id"].astype(str) == str(slice_id)].copy()
    if subset.n_obs == 0:
        raise ValueError(f"No observations found for slice_id={slice_id}")
    return subset


def build_features(
    adata: ad.AnnData,
    n_components: int = 50,
    spatial_weight: float = 1.0,
    random_seed: int = 42,
) -> np.ndarray:
    x = adata.X
    n_components = min(n_components, x.shape[1], max(1, x.shape[0] - 1))
    if n_components >= 2:
        if sparse.issparse(x):
            expr = TruncatedSVD(n_components=n_components, random_state=random_seed).fit_transform(x.astype(np.float32))
        else:
            expr = PCA(n_components=n_components, random_state=random_seed).fit_transform(np.asarray(x, dtype=np.float32))
    else:
        expr = x.toarray() if sparse.issparse(x) else np.asarray(x, dtype=np.float32)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    coords = StandardScaler().fit_transform(coords) * float(spatial_weight)
    return np.hstack([StandardScaler().fit_transform(expr), coords])


def knn_connectivities_sklearn(features: np.ndarray, k: int, weighted: bool = False) -> sparse.csr_matrix:
    k = min(max(1, k), features.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(features)
    distances, idx = nn.kneighbors(return_distance=True)
    distances = distances[:, 1:]
    idx = idx[:, 1:]
    rows = np.repeat(np.arange(features.shape[0]), k)
    cols = idx.reshape(-1)
    if weighted:
        sigma = float(np.nanmedian(distances[distances > 0])) if np.any(distances > 0) else 1.0
        sigma = max(sigma, 1e-6)
        data = np.exp(-np.square(distances.reshape(-1)) / (2.0 * sigma * sigma)).astype(np.float32)
    else:
        data = np.ones(rows.shape[0], dtype=np.float32)
    graph = sparse.coo_matrix((data, (rows, cols)), shape=(features.shape[0], features.shape[0]))
    graph = graph.maximum(graph.T)
    return graph.tocsr()


def knn_connectivities_torch_cuda(
    features: np.ndarray,
    k: int,
    weighted: bool = False,
    chunk_size: int = 1024,
) -> sparse.csr_matrix:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA is not available")
    features = np.asarray(features, dtype=np.float32)
    n_obs = int(features.shape[0])
    k = min(max(1, k), n_obs - 1)
    device = torch.device("cuda")
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    norms = torch.sum(x * x, dim=1)
    all_indices: list[np.ndarray] = []
    all_distances: list[np.ndarray] = []
    chunk_size = max(1, int(chunk_size))
    with torch.no_grad():
        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)
            chunk = x[start:end]
            distances = norms[start:end, None] + norms[None, :] - 2.0 * (chunk @ x.T)
            distances.clamp_min_(0.0)
            row = torch.arange(end - start, device=device)
            col = torch.arange(start, end, device=device)
            distances[row, col] = float("inf")
            vals, idx = torch.topk(distances, k=k, dim=1, largest=False, sorted=False)
            all_indices.append(idx.cpu().numpy())
            all_distances.append(vals.cpu().numpy())
            del distances, vals, idx
    idx_np = np.vstack(all_indices)
    dist_np = np.sqrt(np.maximum(np.vstack(all_distances), 0.0))
    rows = np.repeat(np.arange(n_obs), k)
    cols = idx_np.reshape(-1)
    if weighted:
        positive = dist_np[dist_np > 0]
        sigma = float(np.nanmedian(positive)) if positive.size else 1.0
        sigma = max(sigma, 1e-6)
        data = np.exp(-np.square(dist_np.reshape(-1)) / (2.0 * sigma * sigma)).astype(np.float32)
    else:
        data = np.ones(rows.shape[0], dtype=np.float32)
    graph = sparse.coo_matrix((data, (rows, cols)), shape=(n_obs, n_obs))
    graph = graph.maximum(graph.T)
    return graph.tocsr()


def knn_connectivities(
    features: np.ndarray,
    k: int,
    weighted: bool = False,
    backend: str = "sklearn",
    torch_chunk_size: int = 1024,
) -> sparse.csr_matrix:
    backend = str(backend)
    if backend in {"torch_cuda", "cuda", "gpu"}:
        return knn_connectivities_torch_cuda(
            features,
            k=k,
            weighted=weighted,
            chunk_size=torch_chunk_size,
        )
    if backend == "auto":
        try:
            return knn_connectivities_torch_cuda(
                features,
                k=k,
                weighted=weighted,
                chunk_size=torch_chunk_size,
            )
        except Exception:
            return knn_connectivities_sklearn(features, k=k, weighted=weighted)
    return knn_connectivities_sklearn(features, k=k, weighted=weighted)


def cluster_with_leiden(
    features: np.ndarray,
    k: int,
    resolution: float,
    random_seed: int,
) -> np.ndarray:
    graph = knn_connectivities(features, k)
    return cluster_with_graph_leiden(graph, resolution=resolution, random_seed=random_seed, fallback_features=features)


def cluster_with_graph_leiden(
    graph: sparse.csr_matrix,
    resolution: float,
    random_seed: int,
    fallback_features: np.ndarray,
) -> np.ndarray:
    try:
        import scanpy as sc
        tmp = ad.AnnData(np.zeros((graph.shape[0], 1), dtype=np.float32))
        tmp.obsp["connectivities"] = graph
        tmp.uns["neighbors"] = {
            "connectivities_key": "connectivities",
            "distances_key": "connectivities",
        }
        sc.tl.leiden(tmp, resolution=resolution, random_state=random_seed, key_added="cluster")
        return tmp.obs["cluster"].astype(str).to_numpy()
    except Exception:
        n_clusters = max(2, min(10, int(np.sqrt(fallback_features.shape[0] / 2))))
        return KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=20).fit_predict(fallback_features)


def cluster_with_backend(
    features: np.ndarray,
    coords: np.ndarray,
    backend: str,
    k: int,
    resolution: float,
    random_seed: int,
) -> np.ndarray:
    backend = str(backend)
    if backend == "pca_spatial_leiden":
        return cluster_with_leiden(features, k=k, resolution=resolution, random_seed=random_seed).astype(str)
    raise ValueError(f"Unsupported spatial clustering backend {backend!r}; expected pca_spatial_leiden")


def build_backend_graph(
    features: np.ndarray,
    coords: np.ndarray,
    backend: str,
    k: int,
    knn_backend: str = "sklearn",
    torch_chunk_size: int = 1024,
) -> sparse.csr_matrix:
    backend = str(backend)
    if backend == "pca_spatial_leiden":
        return knn_connectivities(features, k=k, backend=knn_backend, torch_chunk_size=torch_chunk_size)
    raise ValueError(f"Unsupported spatial clustering backend {backend!r}; expected pca_spatial_leiden")


def load_evaluation_labels(labels_path: str | Path) -> pd.Series:
    labels = pd.read_csv(resolve_path(labels_path))
    if {"obs_name", "label"}.issubset(labels.columns):
        return pd.Series(labels["label"].astype(str).values, index=labels["obs_name"].astype(str))
    first = labels.columns[0]
    label_col = "label" if "label" in labels.columns else labels.columns[-1]
    return pd.Series(labels[label_col].astype(str).values, index=labels[first].astype(str))


def load_clustering_input(
    input_path: str | Path | None,
    input_adata: ad.AnnData | None,
    slice_id: Optional[str],
) -> ad.AnnData:
    pre_sliced = False
    if input_adata is not None:
        if slice_id is not None and "slice_id" in input_adata.obs.columns:
            mask = input_adata.obs["slice_id"].astype(str).to_numpy() == str(slice_id)
            if not bool(np.any(mask)):
                raise ValueError(f"No observations found for slice_id={slice_id}")
            adata = input_adata[mask].copy()
            pre_sliced = True
        else:
            adata = input_adata.copy()
    elif input_path is not None:
        adata = ad.read_h5ad(resolve_path(input_path))
    else:
        raise ValueError("run_spatial_clustering requires input_path or input_adata")
    validate_unsupervised_input(adata)
    if not pre_sliced:
        adata = subset_slice(adata, slice_id=slice_id)
    return adata


def write_spatial_cluster_result(
    adata: ad.AnnData,
    clusters: np.ndarray,
    output_dir: str | Path,
    manifest: dict,
    slice_id: Optional[str],
    ablation: str,
    backend: str,
    random_seed: int,
) -> pd.DataFrame:
    result_obs = adata.obs.copy()
    result_obs["pred_cluster"] = clusters.astype(str)
    result_obs.insert(0, "obs_name", adata.obs_names.astype(str))
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    result_obs["spatial_x"] = coords[:, 0]
    result_obs["spatial_y"] = coords[:, 1]
    result_obs["task"] = "dlpfc_spatial_clustering"
    result_obs["ablation"] = ablation
    result_obs["downstream_backend"] = backend
    result_obs["cluster_seed"] = int(random_seed)
    result_obs["slice_id"] = str(slice_id or adata.obs["slice_id"].astype(str).iloc[0])
    result_obs["n_obs"] = int(adata.n_obs)
    result_obs["n_real"] = int((result_obs.get("augmentation_source", "real").astype(str) != "synthetic").sum()) \
        if "augmentation_source" in result_obs else int(adata.n_obs)
    result_obs["n_synthetic"] = int((result_obs["augmentation_source"].astype(str) == "synthetic").sum()) \
        if "augmentation_source" in result_obs else 0
    if "augmentation_source" not in result_obs:
        result_obs["augmentation_source"] = "real"
    result_obs["model"] = str(manifest.get("generator", manifest.get("model", "baseline")))
    result_obs["paradigm_family"] = str(manifest.get("paradigm_family", "real_only"))
    result_obs["variant_id"] = str(manifest.get("variant_id", "real_only"))
    result_obs["synthetic_ratio"] = manifest.get("synthetic_ratio", 0.0)
    result_obs["real_fraction"] = manifest.get("real_fraction", 1.0)
    for key in (
        "dataset",
        "mode",
        "task_name",
        "paradigm_family",
        "variant_id",
        "synthetic_ratio",
        "real_fraction",
        "alpha",
        "selection_policy",
        "generator",
        "condition_source",
        "coord_source",
        "coord_policy",
        "training_regime",
        "n_real_used",
        "n_synthetic_used",
        "input_mode",
        "downstream_backend",
        "cluster_seed",
    ):
        if key in manifest:
            result_obs[key] = manifest[key]
    if "generator" in manifest:
        result_obs["model"] = manifest["generator"]

    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_obs.to_csv(output_dir / "spatial_clusters.csv", index=False)
    return result_obs


def run_spatial_clustering_many_seeds(
    input_path: str | Path | None,
    output_dirs_by_seed: dict[int, str | Path],
    labels_path: Optional[str | Path] = None,
    config_path: str | Path = "configs/spatial_cluster/DLPFC/downstream.yaml",
    slice_id: Optional[str] = None,
    ablation: str = "path_B",
    manifest_path: str | Path | dict | None = None,
    input_adata: ad.AnnData | None = None,
    backend: str | None = None,
    cluster_seeds: list[int] | tuple[int, ...] | None = None,
    skip_existing: bool = False,
) -> dict[int, pd.DataFrame]:
    cfg = load_config(config_path)
    manifest = load_manifest(manifest_path)
    task_cfg = cfg.get("task3_spatial_cluster", {})
    cluster_cfg = task_cfg.get("clustering", {})
    backend = str(backend or cluster_cfg.get("backend", "pca_spatial_leiden"))
    if backend != "pca_spatial_leiden":
        raise ValueError("DLPFC spatial_cluster formal run uses backend='pca_spatial_leiden'")
    if ablation != "path_B":
        raise ValueError("DLPFC spatial_cluster formal run uses ablation='path_B'")
    if cluster_seeds is None:
        cluster_seeds = sorted(output_dirs_by_seed)
    cluster_seeds = [int(seed) for seed in cluster_seeds]
    if cluster_seeds != [42]:
        raise ValueError("DLPFC spatial_cluster formal run uses cluster_seeds=[42]")
    pending_seeds = [
        seed for seed in cluster_seeds
        if not (skip_existing and (resolve_path(output_dirs_by_seed[seed]) / "spatial_clusters.csv").exists())
    ]
    if not pending_seeds:
        return {}

    k = int(cluster_cfg.get("adj_k", 6))
    spatial_weight = float(cluster_cfg.get("spatial_weight", 1.0))
    resolution = float(cluster_cfg.get("resolution", 1.0))
    feature_seed = int(cluster_cfg.get("random_seed", cfg.get("global", {}).get("random_seed", 42)))
    knn_backend = str(cluster_cfg.get("knn_backend", os.environ.get("SPAUG_KNN_BACKEND", "sklearn")))
    torch_chunk_size = int(cluster_cfg.get("torch_knn_chunk_size", os.environ.get("SPAUG_TORCH_KNN_CHUNK_SIZE", 1024)))

    adata = load_clustering_input(input_path=input_path, input_adata=input_adata, slice_id=slice_id)
    features = build_features(
        adata,
        n_components=int(cluster_cfg.get("pca_n_components", task_cfg.get("pca_n_components", 50))),
        spatial_weight=spatial_weight,
        random_seed=feature_seed,
    )
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    results: dict[int, pd.DataFrame] = {}
    graph = build_backend_graph(
        features,
        coords,
        backend=backend,
        k=k,
        knn_backend=knn_backend,
        torch_chunk_size=torch_chunk_size,
    )
    for seed in pending_seeds:
        seed_manifest = dict(manifest)
        seed_manifest["downstream_backend"] = backend
        seed_manifest["cluster_seed"] = int(seed)
        clusters = cluster_with_graph_leiden(
            graph,
            resolution=resolution,
            random_seed=seed,
            fallback_features=features,
        ).astype(str)
        results[seed] = write_spatial_cluster_result(
            adata=adata,
            clusters=clusters,
            output_dir=output_dirs_by_seed[seed],
            manifest=seed_manifest,
            slice_id=slice_id,
            ablation=ablation,
            backend=backend,
            random_seed=seed,
        )
    return results


def run_spatial_clustering(
    input_path: str | Path | None,
    output_dir: str | Path,
    labels_path: Optional[str | Path] = None,
    config_path: str | Path = "configs/spatial_cluster/DLPFC/downstream.yaml",
    slice_id: Optional[str] = None,
    ablation: str = "path_B",
    manifest_path: str | Path | dict | None = None,
    input_adata: ad.AnnData | None = None,
    backend: str | None = None,
    cluster_seed: int | None = None,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    cluster_cfg = cfg.get("task3_spatial_cluster", {}).get("clustering", {})
    seed = int(cluster_seed if cluster_seed is not None else cluster_cfg.get("random_seed", cfg.get("global", {}).get("random_seed", 42)))
    results = run_spatial_clustering_many_seeds(
        input_path=input_path,
        output_dirs_by_seed={seed: output_dir},
        labels_path=labels_path,
        config_path=config_path,
        slice_id=slice_id,
        ablation=ablation,
        manifest_path=manifest_path,
        input_adata=input_adata,
        backend=backend,
        cluster_seeds=[seed],
        skip_existing=False,
    )
    return results[seed]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DLPFC spatial clustering.")
    parser.add_argument("-i", "--input", default=None, help="Unsupervised real/augmented AnnData")
    parser.add_argument("--manifest", default=None, help="Paradigm package manifest.json")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--labels", default=None, help="Labels CSV used for evaluation")
    parser.add_argument("-c", "--config", default="configs/spatial_cluster/DLPFC/downstream.yaml")
    parser.add_argument("--slice-id", default=None)
    parser.add_argument("--ablation", default="path_B", choices=["path_B"])
    parser.add_argument("--backend", default=None, help="The formal run uses pca_spatial_leiden")
    parser.add_argument("--cluster-seed", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    input_path = args.input
    if args.manifest and input_path is None:
        manifest = load_manifest(args.manifest)
        input_path = manifest.get("train_path")
    if input_path is None:
        raise ValueError("Pass --input or --manifest with train_path")
    run_spatial_clustering(
        input_path=input_path,
        output_dir=args.output_dir,
        labels_path=args.labels,
        config_path=args.config,
        slice_id=args.slice_id,
        ablation=args.ablation,
        manifest_path=args.manifest,
        backend=args.backend,
        cluster_seed=args.cluster_seed,
    )


if __name__ == "__main__":
    main()
