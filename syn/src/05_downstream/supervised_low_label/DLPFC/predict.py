"""Run within-slice DLPFC supervised low-label prediction.

Each slice is an independent low-label classification task:
- train: configured fraction of real spots per label in that slice
- test: the remaining real spots in that same slice
- augmented paradigms may use only synthetic spots from the same slice
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import tempfile
import sys
import gc
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import torch
except Exception:
    torch = None

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from io_utils import dense_array, downstream_root, fraction_tag, read_split, resolve_path, sanitize_obs, synthetic_pool_path
else:
    from .io_utils import dense_array, downstream_root, fraction_tag, read_split, resolve_path, sanitize_obs, synthetic_pool_path


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "04_paradigm" / "supervised_low_label"))
from classifiers import encode_labels, train_classifier  # noqa: E402


DEFAULT_MODELS = ["SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion"]
DEFAULT_CLASSIFIERS = ["LR", "XGBoost", "MLP"]
GLOBAL_RATIOS = list(range(1, 31))
LOCAL_RATIOS = [1, 2, 3, 4]
REGIONAL_RATIOS = list(range(1, 31))
ALPHAS = [0, 0.2, 0.4, 0.6, 0.8, 1]
DEFAULT_FAMILIES = [
    "real_baseline",
    "global_real_plus_synthetic",
    "global_synthetic_only",
    "local_spatial",
    "weighted_erm_package",
]
ALL_FAMILIES = [
    *DEFAULT_FAMILIES,
    "regional_weighted_erm_package",
]


def load_paradigm_builder(family: str, dataset: str = "DLPFC"):
    module = importlib.import_module(f"{dataset}.{family}")
    return module.build_package


def align_gene_space(real: ad.AnnData, synthetic: ad.AnnData | None) -> tuple[ad.AnnData, ad.AnnData | None]:
    if synthetic is None:
        return real.copy(), None
    common = real.var_names.intersection(synthetic.var_names)
    if len(common) == 0:
        raise ValueError("No common genes between real and synthetic data")
    return real[:, common].copy(), synthetic[:, common].copy()


def synthetic_slice_map_from_pool(pool_path: str | Path) -> dict[str, Path]:
    """Return per-slice synthetic files for a generated pool.

    Low-label 40x pools are too large to load repeatedly as a single AnnData.
    Prefer explicit ``<pool>/<slice_id>/synthetic.h5ad`` files, then fall back
    to statistical generator shards under ``<pool>/slices/slice_<id>.h5ad``.
    """
    pool_path = resolve_path(pool_path)
    pool_dir = pool_path if pool_path.is_dir() else pool_path.parent
    out: dict[str, Path] = {}
    for path in sorted(pool_dir.glob("*/synthetic.h5ad")):
        if path.parent.name in {"conditions", "slices"}:
            continue
        out[str(path.parent.name)] = path
    slice_root = pool_dir / "slices"
    if slice_root.exists():
        for path in sorted(slice_root.glob("slice_*.h5ad")):
            slice_id = path.stem.replace("slice_", "")
            out.setdefault(str(slice_id), path)
    return out


def load_synthetic_for_slice(synthetic: ad.AnnData | dict[str, Path] | None, slice_id: str) -> ad.AnnData | None:
    if synthetic is None:
        return None
    if isinstance(synthetic, dict):
        path = synthetic.get(str(slice_id))
        if path is None or not path.exists():
            return None
        return ad.read_h5ad(path)
    return synthetic[synthetic.obs["slice_id"].astype(str) == str(slice_id)].copy()


def format_split_path(template: str, fraction_tag_value: str, seed: int, model: str) -> str:
    return template.format(fraction_tag=fraction_tag_value, seed=seed, model=model)


def classifier_params(classifier: str, use_gpu: str, n_jobs: int) -> dict:
    name = classifier.upper()
    if name == "LR":
        return {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "lr": 0.01,
            "epochs": 60,
            "batch_size": 4096,
            "weight_decay": 0.0001,
            "early_stopping_patience": 6,
            "use_gpu": use_gpu,
        }
    if name == "XGBOOST":
        return {
            "n_estimators": 160,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": n_jobs,
            "use_gpu": use_gpu,
        }
    if name == "MLP":
        return {
            "hidden_layers": [256, 128],
            "dropout": 0.25,
            "lr": 0.001,
            "epochs": 50,
            "batch_size": 4096,
            "weight_decay": 0.0001,
            "early_stopping_patience": 5,
            "use_gpu": use_gpu,
        }
    raise ValueError(f"Unsupported classifier: {classifier}")


def choose_pca_components(n_train: int, n_features: int, requested: int) -> int:
    return max(1, min(int(requested), n_features, n_train - 1))


def gpu_enabled_flag(use_gpu: str) -> bool:
    value = str(use_gpu).lower()
    if value in {"0", "false", "no", "cpu"}:
        return False
    if torch is None or not torch.cuda.is_available():
        return False
    return value in {"1", "true", "yes", "cuda", "gpu", "auto"}


def project_gpu_pca(
    x_fit: np.ndarray,
    x_train: np.ndarray,
    x_test: np.ndarray,
    n_components: int,
    use_gpu: str,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not gpu_enabled_flag(use_gpu):
        return None
    try:
        device = torch.device("cuda")
        with torch.no_grad():
            x_fit_t = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
            mean = x_fit_t.mean(dim=0, keepdim=True)
            centered = x_fit_t - mean
            denom = max(1, int(x_fit_t.shape[0]) - 1)
            cov = centered.transpose(0, 1).matmul(centered) / float(denom)
            del centered, x_fit_t
            eigvals, eigvecs = torch.linalg.eigh(cov)
            order = torch.argsort(eigvals, descending=True)[:n_components]
            components = eigvecs[:, order].contiguous()
            del eigvals, eigvecs, cov

            def transform(x: np.ndarray) -> np.ndarray:
                parts = []
                for start in range(0, x.shape[0], chunk_size):
                    xb = torch.as_tensor(x[start : start + chunk_size], dtype=torch.float32, device=device)
                    z = (xb - mean).matmul(components)
                    parts.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
                return np.concatenate(parts, axis=0)

            out_train = transform(x_train)
            out_test = transform(x_test)
        torch.cuda.empty_cache()
        return out_train, out_test
    except Exception as exc:
        print(f"[GPU PCA fallback] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def transform_features(
    train: ad.AnnData,
    test: ad.AnnData,
    pca_fit: ad.AnnData,
    n_components: int,
    use_gpu: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_fit = dense_array(pca_fit.X).astype(np.float32)
    x_train = dense_array(train.X).astype(np.float32)
    x_test = dense_array(test.X).astype(np.float32)
    n_comp = choose_pca_components(x_fit.shape[0], x_fit.shape[1], n_components)
    if n_comp < 2:
        return x_train, x_test
    gpu_result = project_gpu_pca(x_fit, x_train, x_test, n_comp, use_gpu=use_gpu)
    if gpu_result is not None:
        return gpu_result
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(x_fit)
    return pca.transform(x_train).astype(np.float32), pca.transform(x_test).astype(np.float32)


def fit_pca_basis(
    pca_fit: ad.AnnData,
    n_components: int,
    use_gpu: str,
) -> dict[str, np.ndarray | int]:
    x_fit = dense_array(pca_fit.X).astype(np.float32)
    n_comp = choose_pca_components(x_fit.shape[0], x_fit.shape[1], n_components)
    if n_comp < 2:
        return {"identity": 1, "n_components": int(n_comp)}
    if gpu_enabled_flag(use_gpu):
        try:
            device = torch.device("cuda")
            with torch.no_grad():
                x_fit_t = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
                mean = x_fit_t.mean(dim=0, keepdim=True)
                centered = x_fit_t - mean
                denom = max(1, int(x_fit_t.shape[0]) - 1)
                cov = centered.transpose(0, 1).matmul(centered) / float(denom)
                eigvals, eigvecs = torch.linalg.eigh(cov)
                order = torch.argsort(eigvals, descending=True)[:n_comp]
                components = eigvecs[:, order].contiguous()
                out = {
                    "mean": mean.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1),
                    "components": components.detach().cpu().numpy().astype(np.float32, copy=False),
                    "n_components": int(n_comp),
                }
                del x_fit_t, centered, cov, eigvals, eigvecs, components, mean
            torch.cuda.empty_cache()
            return out
        except Exception as exc:
            print(f"[GPU PCA basis fallback] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(x_fit)
    return {
        "mean": pca.mean_.astype(np.float32, copy=False),
        "components": pca.components_.T.astype(np.float32, copy=False),
        "n_components": int(n_comp),
    }


def apply_pca_basis(
    adata: ad.AnnData,
    basis: dict[str, np.ndarray | int],
    use_gpu: str,
    chunk_size: int = 65536,
) -> np.ndarray:
    x = dense_array(adata.X).astype(np.float32)
    if int(basis.get("identity", 0)) == 1:
        return x
    mean = np.asarray(basis["mean"], dtype=np.float32)
    components = np.asarray(basis["components"], dtype=np.float32)
    if gpu_enabled_flag(use_gpu):
        try:
            device = torch.device("cuda")
            mean_t = torch.as_tensor(mean.reshape(1, -1), dtype=torch.float32, device=device)
            comp_t = torch.as_tensor(components, dtype=torch.float32, device=device)
            parts = []
            with torch.no_grad():
                for start in range(0, x.shape[0], chunk_size):
                    xb = torch.as_tensor(x[start : start + chunk_size], dtype=torch.float32, device=device)
                    z = (xb - mean_t).matmul(comp_t)
                    parts.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
            del mean_t, comp_t
            torch.cuda.empty_cache()
            return np.concatenate(parts, axis=0)
        except Exception as exc:
            print(f"[GPU PCA transform fallback] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
    return ((x - mean) @ components).astype(np.float32, copy=False)


def _obs_index(names) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(names.astype(str) if hasattr(names, "astype") else names)}


def _take_projected_rows(names, index: dict[str, int], values: np.ndarray) -> np.ndarray:
    positions = [index[str(name)] for name in names]
    return values[np.asarray(positions, dtype=int)]


def projected_feature_cache_key(
    model_name: str,
    seed: int,
    slice_id: str,
    family: str,
    pca_components: int,
) -> tuple:
    if family == "global_synthetic_only":
        pca_scope = "synthetic_pool"
    else:
        pca_scope = "real_train"
    return (model_name, int(seed), str(slice_id), family, pca_scope, int(pca_components))


def get_projected_slice_features(
    *,
    model_name: str,
    seed: int,
    slice_id: str,
    family: str,
    real_train: ad.AnnData,
    real_test: ad.AnnData,
    synthetic_slice: ad.AnnData | None,
    pca_fit: ad.AnnData,
    pca_components: int,
    use_gpu: str,
    feature_cache: dict | None,
) -> dict[str, object] | None:
    if feature_cache is None or synthetic_slice is None:
        return None
    if family not in {"global_real_plus_synthetic", "global_synthetic_only", "local_spatial"}:
        return None
    key = projected_feature_cache_key(model_name, seed, slice_id, family, pca_components)
    projected = feature_cache.setdefault("projected_slice_features", {})
    if key in projected:
        return projected[key]
    basis_fit = synthetic_slice if family == "global_synthetic_only" else pca_fit
    basis = fit_pca_basis(basis_fit, n_components=pca_components, use_gpu=use_gpu)
    payload: dict[str, object] = {
        "basis": basis,
        "x_real_train": apply_pca_basis(real_train, basis, use_gpu=use_gpu),
        "x_real_test": apply_pca_basis(real_test, basis, use_gpu=use_gpu),
        "real_index": _obs_index(real_train.obs_names),
        "test_index": _obs_index(real_test.obs_names),
        "syn_index": _obs_index(synthetic_slice.obs_names),
        "x_syn": apply_pca_basis(synthetic_slice, basis, use_gpu=use_gpu),
        "pca_scope": "synthetic_pool" if family == "global_synthetic_only" else "real_train",
    }
    projected[key] = payload
    return payload


def compose_projected_train(
    train_pkg: ad.AnnData,
    family: str,
    projected: dict[str, object],
) -> np.ndarray:
    if family == "global_synthetic_only":
        return _take_projected_rows(train_pkg.obs_names, projected["syn_index"], projected["x_syn"])
    if "augmentation_source" not in train_pkg.obs.columns:
        return None
    sources = train_pkg.obs["augmentation_source"].astype(str).to_numpy()
    out = np.empty((train_pkg.n_obs, np.asarray(projected["x_real_train"]).shape[1]), dtype=np.float32)
    real_mask = sources == "real"
    syn_mask = sources == "synthetic"
    if real_mask.any():
        out[real_mask] = _take_projected_rows(
            train_pkg.obs_names[real_mask],
            projected["real_index"],
            projected["x_real_train"],
        )
    if syn_mask.any():
        out[syn_mask] = _take_projected_rows(
            train_pkg.obs_names[syn_mask],
            projected["syn_index"],
            projected["x_syn"],
        )
    if (~(real_mask | syn_mask)).any():
        return None
    return out


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


def attach_regions(real_train: ad.AnnData, synthetic: ad.AnnData | None) -> tuple[ad.AnnData, ad.AnnData | None]:
    real_train = real_train.copy()
    if synthetic is None or synthetic.n_obs == 0:
        real_train.obs["low_label_region"] = spatial_regions(real_train.obsm["spatial"][:, :2])
        return real_train, synthetic
    synthetic = synthetic.copy()
    # synthetic_region is generator provenance. For label-wise Splatter/SPARsim
    # it is local to one label condition, so local paradigms must use a slice-level
    # region grid recomputed from actual coordinates instead.
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
    return synthetic[chosen].copy()


def sample_local_by_label_region(real_train: ad.AnnData, synthetic: ad.AnnData, ratio: float, seed: int) -> ad.AnnData:
    real_train, synthetic = attach_regions(real_train, synthetic)
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
    return synthetic[chosen].copy()


def concat_train(real_train: ad.AnnData | None, syn_train: ad.AnnData | None) -> ad.AnnData:
    parts = [x for x in (real_train, syn_train) if x is not None and x.n_obs > 0]
    if not parts:
        raise ValueError("Training package is empty")
    out = sanitize_obs(ad.concat(parts, join="inner", merge="same", index_unique=None))
    out.obs_names_make_unique()
    return out


def build_training_package(
    family: str,
    real_train: ad.AnnData,
    synthetic_slice: ad.AnnData | None,
    ratio: float | None,
    alpha: float | None,
    seed: int,
) -> tuple[ad.AnnData, np.ndarray | None, ad.AnnData]:
    builder = load_paradigm_builder(family, dataset="DLPFC")
    return builder(real_train, synthetic_slice, ratio=ratio, alpha=alpha, seed=seed)


def variant_grid(family: str) -> Iterable[tuple[float | None, float | None]]:
    if family == "real_baseline":
        yield None, None
    elif family in {"global_real_plus_synthetic", "global_synthetic_only"}:
        for ratio in GLOBAL_RATIOS:
            yield float(ratio), None
    elif family == "local_spatial":
        for ratio in LOCAL_RATIOS:
            yield float(ratio), None
    elif family == "weighted_erm_package":
        for ratio in GLOBAL_RATIOS:
            for alpha in ALPHAS:
                yield float(ratio), float(alpha)
    elif family == "regional_weighted_erm_package":
        for ratio in REGIONAL_RATIOS:
            for alpha in ALPHAS:
                yield float(ratio), float(alpha)
    else:
        raise ValueError(f"Unknown family: {family}")


def variant_id(family: str, ratio: float | None, alpha: float | None) -> str:
    if family == "real_baseline":
        return "real_baseline"
    base = f"{family}_ratio{ratio:g}"
    if alpha is not None:
        base += f"_alpha{alpha:g}"
    return base


def keep_variant(
    ratio: float | None,
    alpha: float | None,
    ratio_filter: set[float] | None,
    alpha_filter: set[float] | None,
) -> bool:
    if ratio_filter is not None:
        if ratio is None or float(ratio) not in ratio_filter:
            return False
    if alpha_filter is not None and alpha is not None:
        if float(alpha) not in alpha_filter:
            return False
    return True


def load_weighted_ratio_windows(path: str | Path | None) -> dict[tuple[float, str, str], set[float]]:
    """Load weighted ERM ratio windows keyed by fraction/model/classifier."""
    if path is None:
        return {}
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weighted ratio window file not found: {path}")
    table = pd.read_csv(path)
    required = {"label_fraction", "model", "classifier", "weighted_ratio_window"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Weighted ratio window file missing columns: {sorted(missing)}")
    out: dict[tuple[float, str, str], set[float]] = {}
    for row in table.itertuples(index=False):
        fraction = float(getattr(row, "label_fraction"))
        model = str(getattr(row, "model"))
        classifier = str(getattr(row, "classifier"))
        raw = getattr(row, "weighted_ratio_window")
        ratios = {
            float(part.strip())
            for part in str(raw).split(",")
            if part is not None and str(part).strip() != ""
        }
        if not ratios:
            continue
        out[(fraction, model, classifier)] = ratios
    return out


def allowed_weighted_ratios(
    windows: dict[tuple[float, str, str], set[float]],
    label_fraction: float,
    model_name: str,
    classifiers: list[str],
) -> set[float] | None:
    if not windows:
        return None
    selected: set[float] = set()
    missing = []
    for classifier in classifiers:
        key = (float(label_fraction), str(model_name), str(classifier))
        values = windows.get(key)
        if values is None:
            missing.append(str(classifier))
            continue
        selected.update(values)
    if missing:
        raise ValueError(
            "Weighted ratio window file has no rows for "
            f"label_fraction={label_fraction}, model={model_name}, classifiers={missing}"
        )
    return selected


def classifier_scoped_output_dir(output_dir: Path, family: str, classifiers: list[str]) -> Path:
    """Avoid output collisions when one classifier is run per process.

    The evaluation stage recursively reads predictions.csv and uses the CSV
    columns, so adding this leaf directory preserves the result schema while
    allowing model x classifier tmux parallelism for weighted ERM.
    """
    if family == "weighted_erm_package" and len(classifiers) == 1:
        return output_dir / f"classifier_{classifiers[0]}"
    return output_dir


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(base_seed) + int(digest[:8], 16) % 100000


def sampling_alpha_key(family: str, alpha: float | None) -> float | None:
    """Keep weighted ERM alpha sweeps on the same sampled synthetic package."""
    if family == "weighted_erm_package":
        return None
    return alpha


def run_one_variant(
    reference: ad.AnnData,
    synthetic: ad.AnnData | dict[str, Path] | None,
    model_name: str,
    label_fraction: float,
    seed: int,
    family: str,
    ratio: float | None,
    alpha: float | None,
    classifiers: list[str],
    output_dir: Path,
    pca_components: int,
    use_gpu: str,
    n_jobs: int,
    feature_cache: dict | None = None,
):
    rows = []
    manifest = {
        "dataset": "DLPFC",
        "task_family": "supervised_low_label",
        "task_name": "dlpfc_slice_low_label_spot_clf",
        "model": model_name,
        "paradigm": family,
        "variant_id": variant_id(family, ratio, alpha),
        "ratio": ratio,
        "alpha": alpha,
        "label_fraction": float(label_fraction),
        "low_label_seed": int(seed),
        "prediction_scope": "within_slice",
        "pca_components": int(pca_components),
        "classifiers": classifiers,
    }
    if family == "regional_weighted_erm_package":
        manifest["alpha_role"] = "regional_alpha_max"
    regional_alpha_profiles = {}
    local_spatial_ratio_profiles = {}
    for slice_id in sorted(reference.obs["slice_id"].astype(str).unique().tolist()):
        real_slice = reference[reference.obs["slice_id"].astype(str) == str(slice_id)].copy()
        syn_slice = load_synthetic_for_slice(synthetic, str(slice_id))
        if syn_slice is not None:
            real_slice, syn_slice = align_gene_space(real_slice, syn_slice)
        real_train = real_slice[real_slice.obs["split"].astype(str) == "train"].copy()
        real_test = real_slice[real_slice.obs["split"].astype(str) == "test"].copy()
        if real_train.n_obs == 0 or real_test.n_obs == 0:
            continue
        if real_train.obs["label"].astype(str).nunique() < 2 or real_test.obs["label"].astype(str).nunique() < 1:
            continue
        train_pkg, sample_weight, pca_fit = build_training_package(
            family=family,
            real_train=real_train,
            synthetic_slice=syn_slice,
            ratio=ratio,
            alpha=alpha,
            seed=stable_seed(seed, family, ratio, sampling_alpha_key(family, alpha), slice_id),
        )
        regional_alpha_map = train_pkg.uns.get("regional_alpha_map")
        if regional_alpha_map is not None:
            regional_alpha_profiles[str(slice_id)] = {
                str(k): float(v) for k, v in dict(regional_alpha_map).items()
            }
        local_ratio_map = train_pkg.uns.get("local_spatial_ratio_map")
        if local_ratio_map is not None:
            local_spatial_ratio_profiles[str(slice_id)] = {
                str(k): int(v) for k, v in dict(local_ratio_map).items()
            }
        y_train = train_pkg.obs["label"].astype(str).to_numpy()
        y_test = real_test.obs["label"].astype(str).to_numpy()
        y_train_enc, y_test_enc, encoder = encode_labels(y_train, y_test)
        cache_key = None
        projected = get_projected_slice_features(
            model_name=model_name,
            seed=seed,
            slice_id=str(slice_id),
            family=family,
            real_train=real_train,
            real_test=real_test,
            synthetic_slice=syn_slice,
            pca_fit=pca_fit,
            pca_components=pca_components,
            use_gpu=use_gpu,
            feature_cache=feature_cache,
        )
        if projected is not None:
            x_train = compose_projected_train(train_pkg, family, projected)
            x_test = projected["x_real_test"]
            if x_train is None:
                projected = None
        if projected is None:
            if family == "weighted_erm_package" and feature_cache is not None:
                cache_key = (
                    model_name,
                    int(seed),
                    str(slice_id),
                    "weighted_erm_package",
                    float(ratio),
                    int(pca_components),
                )
            if cache_key is not None and cache_key in feature_cache:
                x_train, x_test = feature_cache[cache_key]
            else:
                x_train, x_test = transform_features(
                    train_pkg,
                    real_test,
                    pca_fit,
                    n_components=pca_components,
                    use_gpu=use_gpu,
                )
                if cache_key is not None:
                    feature_cache[cache_key] = (x_train, x_test)
        for classifier in classifiers:
            params = classifier_params(classifier, use_gpu=use_gpu, n_jobs=n_jobs)
            clf = train_classifier(
                x_train,
                y_train_enc,
                classifier=classifier,
                params=params,
                random_seed=seed,
                sample_weight=sample_weight,
            )
            pred_enc = clf.predict(x_test)
            pred = encoder.inverse_transform(np.asarray(pred_enc, dtype=int))
            for obs_name, true, pred_label in zip(real_test.obs_names.astype(str), y_test, pred):
                rows.append(
                    {
                        "obs_name": obs_name,
                        "slice_id": str(slice_id),
                        "y_true": str(true),
                        "y_pred": str(pred_label),
                        "classifier": classifier,
                        "model": model_name,
                        "paradigm": family,
                        "variant_id": manifest["variant_id"],
                        "ratio": "" if ratio is None else float(ratio),
                        "alpha": "" if alpha is None else float(alpha),
                        "label_fraction": float(label_fraction),
                        "low_label_seed": int(seed),
                        "n_train_real_slice": int(real_train.n_obs),
                        "n_test_slice": int(real_test.n_obs),
                        "n_train_package": int(train_pkg.n_obs),
                        "regional_alpha_strategy": str(train_pkg.uns.get("regional_alpha_strategy", "")),
                        "local_spatial_ratio_role": str(train_pkg.uns.get("local_spatial_ratio_role", "")),
                    }
                )
    if not rows:
        raise ValueError(f"No predictions produced for {manifest['variant_id']}")
    if regional_alpha_profiles:
        manifest["regional_alpha_profiles"] = regional_alpha_profiles
    if local_spatial_ratio_profiles:
        manifest["local_spatial_ratio_profiles"] = local_spatial_ratio_profiles
    if family == "weighted_erm_package":
        manifest["weighted_erm_alpha_reuse"] = "same synthetic sample and PCA projection for all alphas within each ratio"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run_matrix(args):
    label_fraction = float(args.label_fraction)
    tag = fraction_tag(label_fraction)
    classifiers = args.classifiers or DEFAULT_CLASSIFIERS
    families = args.families or DEFAULT_FAMILIES
    baseline_model = (args.models or DEFAULT_MODELS)[0]
    weighted_windows = load_weighted_ratio_windows(args.weighted_ratio_window_file)
    for seed in args.seeds:
        if "real_baseline" in families and not args.skip_baseline:
            reference = read_split(format_split_path(args.split_template, tag, seed, baseline_model))
            real, _ = align_gene_space(reference, None)
            out_root = downstream_root(args.results_root, "baseline", label_fraction, seed) / "real_baseline" / variant_id("real_baseline", None, None)
            if args.skip_existing and (out_root / "predictions.csv").exists():
                print(f"[baseline seed={seed}] skip existing real_baseline", flush=True)
            else:
                run_one_variant(
                    real,
                    None,
                    "baseline",
                    label_fraction,
                    seed,
                    "real_baseline",
                    None,
                    None,
                    classifiers,
                    out_root,
                    args.pca_components,
                    args.use_gpu,
                    args.n_jobs,
                )
                print(f"[baseline seed={seed}] finished real_baseline", flush=True)
        model_families = [f for f in families if f != "real_baseline"]
        if not model_families:
            continue
        for model_name in args.models:
            reference = read_split(format_split_path(args.split_template, tag, seed, model_name))
            if args.synthetic_template == "__default__":
                syn_path = synthetic_pool_path(
                    args.synthetic_root,
                    model_name,
                    label_fraction,
                    seed,
                    pool_ratio=args.synthetic_pool_ratio,
                )
            else:
                syn_path = Path(args.synthetic_template.format(
                    model=model_name,
                    fraction_tag=tag,
                    seed=seed,
                    output_root=args.synthetic_root,
                ))
            synthetic_map = synthetic_slice_map_from_pool(syn_path)
            if synthetic_map:
                synthetic = synthetic_map
                real = reference
            else:
                synthetic_full = ad.read_h5ad(resolve_path(syn_path))
                real, synthetic = align_gene_space(reference, synthetic_full)
            feature_cache: dict = {}
            for family in model_families:
                weighted_ratio_window = None
                if family == "weighted_erm_package":
                    weighted_ratio_window = allowed_weighted_ratios(weighted_windows, label_fraction, model_name, classifiers)
                for ratio, alpha in variant_grid(family):
                    if weighted_ratio_window is not None and ratio is not None and float(ratio) not in weighted_ratio_window:
                        continue
                    if not keep_variant(ratio, alpha, args.ratio_filter, args.alpha_filter):
                        continue
                    vid = variant_id(family, ratio, alpha)
                    out_root = downstream_root(args.results_root, model_name, label_fraction, seed) / family / vid
                    out_root = classifier_scoped_output_dir(out_root, family, classifiers)
                    if args.dry_run:
                        print(f"[dry-run {model_name} seed={seed}] would run {vid}", flush=True)
                        continue
                    if args.skip_existing and (out_root / "predictions.csv").exists():
                        print(f"[{model_name} seed={seed}] skip existing {vid}", flush=True)
                        continue
                    run_one_variant(
                        real,
                        synthetic,
                        model_name,
                        label_fraction,
                        seed,
                        family,
                        ratio,
                        alpha,
                        classifiers,
                        out_root,
                        args.pca_components,
                        args.use_gpu,
                        args.n_jobs,
                        feature_cache=feature_cache,
                    )
                    print(f"[{model_name} seed={seed}] finished {vid}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DLPFC supervised low-label within-slice prediction.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--label-fraction", type=float, default=0.50)
    parser.add_argument("--classifiers", nargs="*", default=DEFAULT_CLASSIFIERS)
    parser.add_argument("--families", nargs="*", default=DEFAULT_FAMILIES, choices=ALL_FAMILIES)
    parser.add_argument(
        "--split-template",
        default="data/02_interim/supervised_low_label/DLPFC/model_inputs/{model}/{fraction_tag}/seed{seed}/processed_with_split.h5ad",
    )
    parser.add_argument("--synthetic-root", default="data/03_synthetic")
    parser.add_argument("--synthetic-pool-ratio", type=float, default=40.0)
    parser.add_argument("--synthetic-template", default="__default__")
    parser.add_argument("--results-root", default="data/05_results")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--use-gpu", default="auto", choices=["auto", "true", "false", "cuda", "cpu"])
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ratio-filter", nargs="*", type=float, default=None)
    parser.add_argument("--alpha-filter", nargs="*", type=float, default=None)
    parser.add_argument(
        "--weighted-ratio-window-file",
        default=None,
        help=(
            "Optional CSV from low-label evaluation with columns label_fraction, model, classifier, "
            "weighted_ratio_window. When set, weighted_erm_package runs across the union of windows "
            "for the selected classifiers."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview selected variants and leave prediction files unchanged.")
    return parser


def main():
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "spaug_numba_cache"))
    args = build_parser().parse_args()
    args.ratio_filter = None if args.ratio_filter is None else {float(x) for x in args.ratio_filter}
    args.alpha_filter = None if args.alpha_filter is None else {float(x) for x in args.alpha_filter}
    run_matrix(args)


if __name__ == "__main__":
    main()
