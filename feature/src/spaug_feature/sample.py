"""Sample-level expression summaries and classifiers."""
from __future__ import annotations
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .paths import resolve_path, load_yaml
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
PCA_CACHE = {}
REAL_SAMPLE_CACHE = {}
FEATURE_CACHE = {}
TRANSFORM_CACHE = {}
_CACHE_DATASET = None


def activate_dataset(adata: ad.AnnData) -> None:
    """Scope sample and projection caches to one input dataset."""
    global _CACHE_DATASET
    if _CACHE_DATASET is not adata:
        for cache in (PCA_CACHE, REAL_SAMPLE_CACHE, FEATURE_CACHE, TRANSFORM_CACHE):
            cache.clear()
        _CACHE_DATASET = adata


def dense(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def subset_sample(adata: ad.AnnData, sample_id: str) -> ad.AnnData:
    activate_dataset(adata)
    key = str(sample_id)
    if key in REAL_SAMPLE_CACHE:
        return REAL_SAMPLE_CACHE[key]
    out = adata[adata.obs["sample_id"].astype(str) == str(sample_id)].copy()
    if out.n_obs == 0:
        raise ValueError(f"No real spots for sample_id={sample_id}")
    REAL_SAMPLE_CACHE[key] = out
    return out


def align_to_genes(adata: ad.AnnData, genes: pd.Index | list[str]) -> ad.AnnData:
    common = pd.Index(genes).intersection(adata.var_names)
    if len(common) == 0:
        raise ValueError("No common genes for feature projection")
    if len(common) != len(genes):
        raise ValueError(f"Gene mismatch: expected {len(genes)}, found {len(common)}")
    return adata[:, list(genes)].copy()


def fit_pca(real: ad.AnnData, train_samples: list[str], n_pcs: int) -> tuple[PCA, pd.Index]:
    train = real[real.obs["sample_id"].astype(str).isin(train_samples)].copy()
    x = dense(train.X).astype(np.float32)
    n_comp = max(1, min(int(n_pcs), x.shape[0] - 1, x.shape[1]))
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(x)
    return pca, train.var_names.copy()


def get_fold_pca(real: ad.AnnData, fold: int, train_samples: list[str], n_pcs: int) -> tuple[PCA, pd.Index]:
    activate_dataset(real)
    key = (int(fold), int(n_pcs), tuple(sorted(map(str, train_samples))))
    if key not in PCA_CACHE:
        PCA_CACHE[key] = fit_pca(real, train_samples, n_pcs=n_pcs)
    return PCA_CACHE[key]


def weighted_mean_and_std(z: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    mean = np.average(z, axis=0, weights=weights)
    var = np.average((z - mean[None, :]) ** 2, axis=0, weights=weights)
    return mean, np.sqrt(np.maximum(var, 0.0))


def sample_package_weights(package: ad.AnnData, alpha: float | None = None) -> np.ndarray:
    source = package.obs.get("augmentation_source", package.obs.get("source")).astype(str).to_numpy()
    real_mask = np.isin(source, ["real"])
    syn_mask = ~real_mask
    weights = np.zeros(package.n_obs, dtype=np.float64)
    if alpha is None:
        weights[:] = 1.0 / max(1, package.n_obs)
        return weights
    alpha = float(alpha)
    if real_mask.any():
        weights[real_mask] = (1.0 - alpha) / max(1, int(real_mask.sum()))
    if syn_mask.any():
        weights[syn_mask] = alpha / max(1, int(syn_mask.sum()))
    if weights.sum() <= 0:
        weights[:] = 1.0 / max(1, package.n_obs)
    return weights


def sample_feature(
    package: ad.AnnData,
    weights: np.ndarray,
    pca: PCA,
    genes: pd.Index,
    quantiles: list[float],
    cache_key: tuple | None = None,
    transform_cache_key: tuple | None = None,
) -> np.ndarray:
    if cache_key is not None and cache_key in FEATURE_CACHE:
        return FEATURE_CACHE[cache_key]
    if transform_cache_key is not None and transform_cache_key in TRANSFORM_CACHE:
        z, qs, coord_stats = TRANSFORM_CACHE[transform_cache_key]
    else:
        package = align_to_genes(package, genes)
        x = dense(package.X).astype(np.float32)
        z = pca.transform(x)
        qs = np.quantile(z, quantiles, axis=0).reshape(-1)
        coords = np.asarray(package.obsm["spatial"][:, :2], dtype=np.float64)
        coord_stats = np.concatenate([
            coords.mean(axis=0),
            coords.std(axis=0),
            coords.max(axis=0) - coords.min(axis=0),
        ])
        if transform_cache_key is not None:
            TRANSFORM_CACHE[transform_cache_key] = (z, qs, coord_stats)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[0] != z.shape[0]:
        raise ValueError("Feature weights length does not match package observations")
    mean, sd = weighted_mean_and_std(z, weights)
    feature = np.concatenate([mean, sd, qs, coord_stats]).astype(np.float32)
    if cache_key is not None:
        FEATURE_CACHE[cache_key] = feature
    return feature


def classifier_params(name: str, n_jobs: int, seed: int):
    name = name.upper()
    if name == "LR":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ])
    if name == "XGBOOST":
        if XGBClassifier is None:
            return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=seed)
        return XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "MLP":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, early_stopping=True, random_state=seed)),
        ])
    raise ValueError(f"Unsupported classifier: {name}")
