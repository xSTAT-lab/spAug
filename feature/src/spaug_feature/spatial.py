"""Expression, spatial features, graph clustering, and result serialization."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from .paths import resolve_path, load_config


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
