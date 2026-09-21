"""Run DLPFC cross-slice supervised prediction.

For each paradigm, every training slice is processed with its own synthetic
spots first. The resulting slice-level training packages are then concatenated
and one classifier is trained for cross-slice testing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import tempfile
import sys
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
    from io_utils import dense_array, downstream_root, read_split, resolve_path, sanitize_obs, synthetic_pool_path
else:
    from .io_utils import dense_array, downstream_root, read_split, resolve_path, sanitize_obs, synthetic_pool_path


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "04_paradigm" / "cross_slice_generalization"))
from classifiers import encode_labels, train_classifier  # noqa: E402


TASK_FAMILY = "cross_slice_generalization"
TASK_NAME = "dlpfc_cross_slice_spot_clf"
DEFAULT_MODELS = ["SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion"]
DEFAULT_CLASSIFIERS = ["LR", "XGBoost", "MLP"]
GLOBAL_RATIOS = list(range(1, 31))
LOCAL_RATIOS = [1, 2, 3, 4]
ALPHAS = [0, 0.2, 0.4, 0.6, 0.8, 1]
DEFAULT_FAMILIES = [
    "real_baseline",
    "global_real_plus_synthetic",
    "global_synthetic_only",
    "weighted_erm_package",
    "local_spatial",
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
            out.setdefault(path.stem.replace("slice_", ""), path)
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
    sample_weight: np.ndarray | None = None,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not gpu_enabled_flag(use_gpu):
        return None
    try:
        device = torch.device("cuda")
        with torch.no_grad():
            x_fit_t = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
            if sample_weight is None:
                weights = None
                mean = x_fit_t.mean(dim=0, keepdim=True)
            else:
                weights = torch.as_tensor(np.asarray(sample_weight, dtype=np.float32), device=device).reshape(-1, 1)
                weights = weights / weights.sum().clamp_min(1e-12)
                mean = (x_fit_t * weights).sum(dim=0, keepdim=True)
            centered = x_fit_t - mean
            if weights is None:
                denom = max(1, int(x_fit_t.shape[0]) - 1)
                cov = centered.transpose(0, 1).matmul(centered) / float(denom)
            else:
                cov = centered.transpose(0, 1).matmul(centered * weights)
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
    n_components: int,
    use_gpu: str,
    sample_weight: np.ndarray | None = None,
    pca_fit: ad.AnnData | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = dense_array(train.X).astype(np.float32)
    x_test = dense_array(test.X).astype(np.float32)
    if pca_fit is not None:
        x_fit = dense_array(pca_fit.X).astype(np.float32)
        fit_weight = None
    elif sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        positive = sample_weight > 0
        if not positive.any():
            raise ValueError("PCA sample_weight has no positive entries")
        x_fit = x_train[positive]
        fit_weight = sample_weight[positive]
    else:
        x_fit = x_train
        fit_weight = None
    n_comp = choose_pca_components(x_fit.shape[0], x_fit.shape[1], n_components)
    if n_comp < 2:
        return x_train, x_test
    gpu_result = project_gpu_pca(x_fit, x_train, x_test, n_comp, use_gpu=use_gpu, sample_weight=fit_weight)
    if gpu_result is not None:
        return gpu_result
    if fit_weight is None:
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(x_fit)
        return pca.transform(x_train).astype(np.float32), pca.transform(x_test).astype(np.float32)
    fit_weight = fit_weight / max(float(fit_weight.sum()), 1e-12)
    mean = np.average(x_fit, axis=0, weights=fit_weight).astype(np.float32)
    centered = x_fit - mean
    cov = centered.T @ (centered * fit_weight[:, None])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:n_comp]
    components = eigvecs[:, order].astype(np.float32, copy=False)
    return (x_train - mean) @ components, (x_test - mean) @ components


def build_training_package(
    family: str,
    real_train: ad.AnnData,
    synthetic_slice: ad.AnnData | None,
    ratio: float | None,
    alpha: float | None,
    seed: int,
) -> tuple[ad.AnnData, np.ndarray | None, ad.AnnData]:
    if family == "weighted_erm_package" and alpha is not None and np.isclose(float(alpha), 0.0):
        real = real_train.copy()
        real.obs["augmentation_source"] = "real"
        weights = np.full(real.n_obs, 1.0 / max(1, real.n_obs), dtype=np.float64)
        real.uns["weighted_erm_objective"] = "(1-alpha)*mean(real_loss)+alpha*mean(synthetic_loss)"
        real.uns["alpha"] = 0.0
        real.uns["alpha_zero_real_only"] = True
        return real, weights, real_train.copy()
    builder = load_paradigm_builder(family, dataset="DLPFC")
    return builder(real_train, synthetic_slice, ratio=ratio, alpha=alpha, seed=seed)


def variant_grid(family: str) -> Iterable[tuple[float | None, float | None]]:
    if family == "real_baseline":
        yield None, None
    elif family in {"global_real_plus_synthetic", "global_synthetic_only"}:
        for ratio in GLOBAL_RATIOS:
            yield float(ratio), None
    elif family == "weighted_erm_package":
        for ratio in GLOBAL_RATIOS:
            for alpha in ALPHAS:
                yield float(ratio), float(alpha)
    elif family == "local_spatial":
        for ratio in LOCAL_RATIOS:
            yield float(ratio), None
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


def load_weighted_ratio_windows(path: str | Path | None) -> dict[tuple[str, str], set[float]]:
    """Load cross-slice weighted ERM ratio windows keyed by model/classifier."""
    if path is None:
        return {}
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weighted ratio window file not found: {path}")
    table = pd.read_csv(path)
    required = {"model", "classifier", "weighted_ratio_window"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Weighted ratio window file missing columns: {sorted(missing)}")
    out: dict[tuple[str, str], set[float]] = {}
    for row in table.itertuples(index=False):
        model = str(getattr(row, "model"))
        classifier = str(getattr(row, "classifier"))
        raw = getattr(row, "weighted_ratio_window")
        ratios = {
            float(part.strip())
            for part in str(raw).split(",")
            if part is not None and str(part).strip() != ""
        }
        if ratios:
            out[(model, classifier)] = ratios
    return out


def allowed_weighted_ratios(
    windows: dict[tuple[str, str], set[float]],
    model_name: str,
    classifiers: list[str],
) -> set[float] | None:
    if not windows:
        return None
    if len(classifiers) != 1:
        raise ValueError(
            "--weighted-ratio-window-file requires running exactly one classifier per process "
            "to avoid taking the union of classifier-specific windows."
        )
    key = (str(model_name), str(classifiers[0]))
    values = windows.get(key)
    if values is None:
        raise ValueError(f"Weighted ratio window file has no row for model={model_name}, classifier={classifiers[0]}")
    return values


def classifier_scoped_output_dir(output_dir: Path, family: str, classifiers: list[str]) -> Path:
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


def concat_packages(parts: list[ad.AnnData]) -> ad.AnnData:
    out = sanitize_obs(ad.concat(parts, join="inner", merge="same", index_unique=None))
    out.obs_names_make_unique()
    return out


def concat_weights(weights: list[np.ndarray | None], packages: list[ad.AnnData]) -> np.ndarray | None:
    if all(w is None for w in weights):
        return None
    normalized = []
    for weight, package in zip(weights, packages):
        if weight is None:
            normalized.append(np.ones(package.n_obs, dtype=np.float64))
        else:
            weight = np.asarray(weight, dtype=np.float64)
            if weight.shape[0] != package.n_obs:
                raise ValueError(f"Sample weight length {weight.shape[0]} != package size {package.n_obs}")
            normalized.append(weight)
    return np.concatenate(normalized)


def global_weighted_erm_weights(train_pkg: ad.AnnData, alpha: float) -> np.ndarray:
    source = train_pkg.obs.get("augmentation_source")
    if source is None:
        source = train_pkg.obs.get("source")
    if source is None:
        raise ValueError("weighted_erm_package requires obs['augmentation_source'] or obs['source']")
    is_syn = source.astype(str).to_numpy() == "synthetic"
    is_real = ~is_syn
    n_real = int(is_real.sum())
    n_syn = int(is_syn.sum())
    if n_real == 0 or n_syn == 0:
        raise ValueError(f"weighted_erm_package requires both real and synthetic samples; got real={n_real}, syn={n_syn}")
    weights = np.zeros(train_pkg.n_obs, dtype=np.float64)
    weights[is_real] = (1.0 - float(alpha)) / n_real
    weights[is_syn] = float(alpha) / n_syn
    return weights


def train_slices(reference: ad.AnnData) -> list[str]:
    train = reference.obs[reference.obs["split"].astype(str) == "train"]
    return sorted(train["slice_id"].astype(str).unique().tolist())


def test_slices(reference: ad.AnnData) -> list[str]:
    test = reference.obs[reference.obs["split"].astype(str) == "test"]
    return sorted(test["slice_id"].astype(str).unique().tolist())


def run_one_variant(
    reference: ad.AnnData,
    synthetic: ad.AnnData | dict[str, Path] | None,
    model_name: str,
    split_tag: str,
    synthetic_split_tag: str,
    random_seed: int,
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
    manifest = {
        "dataset": "DLPFC",
        "task_family": TASK_FAMILY,
        "task_name": TASK_NAME,
        "model": model_name,
        "paradigm": family,
        "variant_id": variant_id(family, ratio, alpha),
        "ratio": ratio,
        "alpha": alpha,
        "split_tag": str(split_tag),
        "synthetic_split_tag": str(synthetic_split_tag),
        "synthetic_usage_policy": "pre_generated_pool_train_slices_only",
        "test_synthetic_used": False,
        "random_seed": int(random_seed),
        "prediction_scope": "cross_slice",
        "train_slices": train_slices(reference),
        "test_slices": test_slices(reference),
        "pca_fit_scope": "real_train_only" if family == "weighted_erm_package" else "combined_train_package",
        "pca_components": int(pca_components),
        "classifiers": classifiers,
    }

    package_parts: list[ad.AnnData] = []
    pca_fit_parts: list[ad.AnnData] = []
    weight_parts: list[np.ndarray | None] = []
    local_spatial_ratio_profiles = {}
    synthetic_slices_used: list[str] = []
    for slice_id in manifest["train_slices"]:
        real_slice = reference[reference.obs["slice_id"].astype(str) == str(slice_id)].copy()
        real_train = real_slice[real_slice.obs["split"].astype(str) == "train"].copy()
        if real_train.n_obs == 0:
            continue
        syn_slice = load_synthetic_for_slice(synthetic, str(slice_id))
        if syn_slice is not None:
            real_train, syn_slice = align_gene_space(real_train, syn_slice)
            synthetic_slices_used.append(str(slice_id))
        pca_fit_parts.append(real_train.copy())
        train_pkg, sample_weight, _ = build_training_package(
            family=family,
            real_train=real_train,
            synthetic_slice=syn_slice,
            ratio=ratio,
            alpha=alpha,
            seed=stable_seed(random_seed, family, ratio, sampling_alpha_key(family, alpha), slice_id),
        )
        local_ratio_map = train_pkg.uns.get("local_spatial_ratio_map")
        if local_ratio_map is not None:
            local_spatial_ratio_profiles[str(slice_id)] = {str(k): int(v) for k, v in dict(local_ratio_map).items()}
        train_pkg.obs["train_origin_slice_id"] = str(slice_id)
        package_parts.append(train_pkg)
        weight_parts.append(sample_weight)

    if not package_parts:
        raise ValueError(f"No training packages produced for {manifest['variant_id']}")
    leaked = sorted(set(synthetic_slices_used) & set(manifest["test_slices"]))
    if leaked:
        raise RuntimeError(f"Test-slice synthetic data would be used for split {split_tag}: {leaked}")
    manifest["synthetic_slices_used"] = sorted(set(synthetic_slices_used))
    train_pkg_all = concat_packages(package_parts)
    pca_fit_all = concat_packages(pca_fit_parts)
    sample_weight_all = concat_weights(weight_parts, package_parts)
    if family == "weighted_erm_package" and not (alpha is not None and np.isclose(float(alpha), 0.0)):
        sample_weight_all = global_weighted_erm_weights(train_pkg_all, alpha=float(alpha))
    if local_spatial_ratio_profiles:
        manifest["local_spatial_ratio_profiles"] = local_spatial_ratio_profiles
    train_model = train_pkg_all
    sample_weight_model = sample_weight_all
    if sample_weight_all is not None:
        manifest["weighted_erm_objective"] = "(1-alpha)*mean(real_loss)+alpha*mean(synthetic_loss)"
        manifest["sample_weight_sum"] = float(np.sum(sample_weight_all))
        positive = np.asarray(sample_weight_all) > 0
        if not positive.any():
            raise ValueError(f"No positive sample weights for {manifest['variant_id']}")
        if not positive.all():
            train_model = train_pkg_all[positive].copy()
            sample_weight_model = np.asarray(sample_weight_all, dtype=np.float64)[positive]
        manifest["n_train_effective"] = int(train_model.n_obs)

    real_test_all = reference[reference.obs["split"].astype(str) == "test"].copy()
    real_test_all = real_test_all[:, train_model.var_names].copy()
    if real_test_all.n_obs == 0:
        raise ValueError("No held-out test slices found")
    y_train = train_model.obs["label"].astype(str).to_numpy()
    y_test = real_test_all.obs["label"].astype(str).to_numpy()
    y_train_enc, y_test_enc, encoder = encode_labels(y_train, y_test)
    cache_key = None
    if family == "weighted_erm_package" and feature_cache is not None:
        if alpha is not None and np.isclose(float(alpha), 0.0):
            weight_scope = "real_only"
        elif alpha is not None and np.isclose(float(alpha), 1.0):
            weight_scope = "synthetic_only"
        else:
            weight_scope = "with_synthetic"
        cache_key = (
            model_name,
            str(split_tag),
            "weighted_erm_package",
            float(ratio),
            weight_scope,
            int(pca_components),
        )
    if cache_key is not None and cache_key in feature_cache:
        x_train, x_test = feature_cache[cache_key]
    else:
        x_train, x_test = transform_features(
            train_model,
            real_test_all,
            n_components=pca_components,
            use_gpu=use_gpu,
            sample_weight=None,
            pca_fit=pca_fit_all if family == "weighted_erm_package" else None,
        )
        if cache_key is not None:
            feature_cache[cache_key] = (x_train, x_test)

    rows = []
    for classifier in classifiers:
        params = classifier_params(classifier, use_gpu=use_gpu, n_jobs=n_jobs)
        clf = train_classifier(
            x_train,
            y_train_enc,
            classifier=classifier,
            params=params,
            random_seed=random_seed,
            sample_weight=sample_weight_model,
        )
        pred_enc = clf.predict(x_test)
        pred = encoder.inverse_transform(np.asarray(pred_enc, dtype=int))
        for obs_name, slice_id, true, pred_label in zip(
            real_test_all.obs_names.astype(str),
            real_test_all.obs["slice_id"].astype(str),
            y_test,
            pred,
        ):
            rows.append(
                {
                    "obs_name": obs_name,
                    "slice_id": str(slice_id),
                    "test_scope": "heldout_slice",
                    "y_true": str(true),
                    "y_pred": str(pred_label),
                    "classifier": classifier,
                    "model": model_name,
                    "paradigm": family,
                    "variant_id": manifest["variant_id"],
                    "ratio": "" if ratio is None else float(ratio),
                    "alpha": "" if alpha is None else float(alpha),
                    "split_tag": str(split_tag),
                    "random_seed": int(random_seed),
                    "n_train_real_total": int((reference.obs["split"].astype(str) == "train").sum()),
                    "n_test_total": int(real_test_all.n_obs),
                    "n_train_package": int(train_pkg_all.n_obs),
                    "n_train_effective": int(train_model.n_obs),
                    "n_train_slices": int(len(manifest["train_slices"])),
                    "n_test_slices": int(len(manifest["test_slices"])),
                    "local_spatial_ratio_role": str(train_pkg_all.uns.get("local_spatial_ratio_role", "")),
                }
            )
    if not rows:
        raise ValueError(f"No predictions produced for {manifest['variant_id']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run_matrix(args):
    classifiers = args.classifiers or DEFAULT_CLASSIFIERS
    families = args.families or DEFAULT_FAMILIES
    baseline_model = (args.models or DEFAULT_MODELS)[0]
    split_tag = str(args.split_tag)
    synthetic_split_tag = str(args.synthetic_split_tag or split_tag)
    weighted_windows = load_weighted_ratio_windows(args.weighted_ratio_window_file)
    if "real_baseline" in families and not args.skip_baseline:
        reference = read_split(args.split_template.format(model=baseline_model, split_tag=split_tag))
        real, _ = align_gene_space(reference, None)
        out_root = downstream_root(args.results_root, "baseline", split_tag) / "real_baseline" / "real_baseline"
        out_root = classifier_scoped_output_dir(out_root, "real_baseline", classifiers)
        if args.dry_run:
            print(f"[dry-run baseline split={split_tag}] would run real_baseline", flush=True)
        elif args.skip_existing and (out_root / "predictions.csv").exists():
            print(f"[baseline split={split_tag}] skip existing real_baseline", flush=True)
        else:
            run_one_variant(
                real,
                None,
                "baseline",
                split_tag,
                synthetic_split_tag,
                args.random_seed,
                "real_baseline",
                None,
                None,
                classifiers,
                out_root,
                args.pca_components,
                args.use_gpu,
                args.n_jobs,
            )
            print(f"[baseline split={split_tag}] finished real_baseline", flush=True)

    model_families = [f for f in families if f != "real_baseline"]
    if not model_families:
        return
    for model_name in args.models:
        reference = read_split(args.split_template.format(model=model_name, split_tag=split_tag))
        aligned_real = None
        synthetic = None
        synthetic_map: dict[str, Path] | None = None
        feature_cache: dict = {}
        for family in model_families:
            weighted_ratio_window = None
            if family == "weighted_erm_package":
                weighted_ratio_window = allowed_weighted_ratios(weighted_windows, model_name, classifiers)
            for ratio, alpha in variant_grid(family):
                if weighted_ratio_window is not None and ratio is not None and float(ratio) not in weighted_ratio_window:
                    continue
                if not keep_variant(ratio, alpha, args.ratio_filter, args.alpha_filter):
                    continue
                needs_synthetic = not (
                    family == "weighted_erm_package"
                    and alpha is not None
                    and np.isclose(float(alpha), 0.0)
                )
                if needs_synthetic:
                    if synthetic is None:
                        if args.synthetic_template == "__default__":
                            syn_path = synthetic_pool_path(
                                args.synthetic_root,
                                model_name,
                                synthetic_split_tag,
                                pool_ratio=args.synthetic_pool_ratio,
                            )
                        else:
                            syn_path = resolve_path(
                                args.synthetic_template.format(
                                    model=model_name,
                                    split_tag=split_tag,
                                    synthetic_split_tag=synthetic_split_tag,
                                    output_root=args.synthetic_root,
                                )
                            )
                        synthetic_map = synthetic_slice_map_from_pool(syn_path)
                        if synthetic_map:
                            synthetic = synthetic_map
                            aligned_real = reference
                        else:
                            aligned_real, synthetic = align_gene_space(reference, ad.read_h5ad(resolve_path(syn_path)))
                    real_for_variant = aligned_real
                    synthetic_for_variant = synthetic
                else:
                    real_for_variant = reference.copy()
                    synthetic_for_variant = None
                vid = variant_id(family, ratio, alpha)
                out_root = downstream_root(args.results_root, model_name, split_tag) / family / vid
                out_root = classifier_scoped_output_dir(out_root, family, classifiers)
                if args.dry_run:
                    print(f"[dry-run {model_name} split={split_tag}] would run {vid}", flush=True)
                    continue
                if args.skip_existing and (out_root / "predictions.csv").exists():
                    print(f"[{model_name} split={split_tag}] skip existing {vid}", flush=True)
                    continue
                run_one_variant(
                    real_for_variant,
                    synthetic_for_variant,
                    model_name,
                    split_tag,
                    synthetic_split_tag,
                    args.random_seed,
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
                print(f"[{model_name} split={split_tag}] finished {vid}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DLPFC cross-slice supervised prediction.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--split-tag", default="fixed_8train_4test")
    parser.add_argument(
        "--synthetic-split-tag",
        default=None,
        help="Split tag used to locate pre-generated synthetic pools. Defaults to --split-tag; use all_slices_pool for no-leakage shared pools.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--classifiers", nargs="*", default=DEFAULT_CLASSIFIERS)
    parser.add_argument("--families", nargs="*", default=DEFAULT_FAMILIES, choices=DEFAULT_FAMILIES)
    parser.add_argument(
        "--split-template",
        default="data/02_interim/cross_slice_generalization/DLPFC/model_inputs/{model}/{split_tag}/processed_with_split.h5ad",
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
            "Optional CSV with model, classifier, weighted_ratio_window. "
            "When set, weighted_erm_package runs only that classifier-specific "
            "10-ratio window; exactly one classifier must be selected."
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
