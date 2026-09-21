"""Run Kidney sample-level disease prediction.

Each sample is a bag of spots. Paradigms operate inside each training sample,
then every sample package is aggregated to one feature vector for sample-level
classification. Test samples are always represented by real spots only.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
TASK = "sample_disease_prediction"
DATASET = "Kidney"
sys.path.insert(0, str(PROJECT_ROOT / "src" / "04_paradigm" / TASK))
PCA_CACHE: dict[tuple[int, int, tuple[str, ...]], tuple[PCA, pd.Index]] = {}
REAL_SAMPLE_CACHE: dict[str, ad.AnnData] = {}
SYNTHETIC_CACHE: dict[tuple[str, str, str], ad.AnnData] = {}
FEATURE_CACHE: dict[tuple, np.ndarray] = {}
TRANSFORM_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
WEIGHTED_PACKAGE_CACHE: dict[tuple, tuple[ad.AnnData, dict]] = {}
ACTIVE_WEIGHTED_CACHE_GROUP: tuple | None = None


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dense(x) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def read_splits(path: str | Path) -> pd.DataFrame:
    splits = pd.read_csv(resolve_path(path))
    splits["sample_id"] = splits["sample_id"].astype(str)
    splits["split"] = splits["split"].astype(str)
    return splits


def load_synthetic(generated_root: Path, model: str, sample_id: str) -> ad.AnnData:
    path = generated_root / model / f"{sample_id}.h5ad"
    if not path.exists():
        raise FileNotFoundError(f"Missing synthetic file: {path}")
    key = (str(generated_root), str(model), str(sample_id))
    if key not in SYNTHETIC_CACHE:
        SYNTHETIC_CACHE[key] = ad.read_h5ad(path)
    return SYNTHETIC_CACHE[key]


def subset_sample(adata: ad.AnnData, sample_id: str) -> ad.AnnData:
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


def predict_score(model, x: np.ndarray, positive_index: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] > positive_index:
            return proba[:, positive_index]
    if hasattr(model, "decision_function"):
        score = model.decision_function(x)
        score = score[:, positive_index] if getattr(score, "ndim", 1) == 2 else score
        return 1.0 / (1.0 + np.exp(-score))
    pred = model.predict(x)
    return np.asarray(pred == positive_index, dtype=float)


def paradigm_grid(family: str, ratios: list[float], alphas: list[float]) -> Iterable[tuple[float | None, float | None, str]]:
    if family == "real_baseline":
        yield None, None, "baseline"
    elif family in {"global_real_plus_synthetic", "global_synthetic_only"}:
        for ratio in ratios:
            yield float(ratio), None, f"ratio_{ratio:g}"
    elif family == "weighted_erm_package":
        for ratio in ratios:
            for alpha in alphas:
                yield float(ratio), float(alpha), f"ratio_{ratio:g}_alpha_{alpha:g}"
    else:
        raise ValueError(f"Unsupported paradigm for Kidney sample prediction: {family}")


def load_builder(paradigm: str):
    return importlib.import_module(f"{DATASET}.{paradigm}").build_package


def build_weighted_package_cached(
    builder,
    real_sample: ad.AnnData,
    synthetic_sample: ad.AnnData,
    generated_root: Path,
    model_name: str,
    sample_id: str,
    ratio: float,
    alpha: float,
    seed: int,
) -> tuple[ad.AnnData, np.ndarray, dict, tuple]:
    package_key = (
        "weighted_package",
        str(generated_root),
        model_name,
        str(sample_id),
        float(ratio),
        int(seed),
    )
    if package_key not in WEIGHTED_PACKAGE_CACHE:
        package, _, info = builder(real_sample, synthetic_sample, ratio=ratio, alpha=alpha, seed=seed)
        WEIGHTED_PACKAGE_CACHE[package_key] = (package, dict(info))
    package, info = WEIGHTED_PACKAGE_CACHE[package_key]
    weights = sample_package_weights(package, alpha=alpha)
    return package, weights, dict(info), package_key


def build_matrix(
    real: ad.AnnData,
    generated_root: Path,
    train_samples: list[str],
    test_samples: list[str],
    model_name: str,
    paradigm: str,
    ratio: float | None,
    alpha: float | None,
    pca: PCA,
    genes: pd.Index,
    quantiles: list[float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    builder = load_builder(paradigm)
    x_train, y_train, x_test, y_test, package_rows = [], [], [], [], []
    fold_tag = tuple(sorted(map(str, train_samples)))
    for i, sample_id in enumerate(train_samples):
        real_sample = subset_sample(real, sample_id)
        package_seed = seed + i * 31
        needs_synthetic = model_name != "baseline" and not (
            paradigm == "weighted_erm_package" and float(alpha or 0.0) == 0.0
        )
        syn = load_synthetic(generated_root, model_name, sample_id) if needs_synthetic else None
        transform_cache_key = None
        if paradigm == "weighted_erm_package" and syn is not None and float(alpha or 0.0) > 0.0:
            package, weights, info, package_key = build_weighted_package_cached(
                builder,
                real_sample,
                syn,
                generated_root=generated_root,
                model_name=model_name,
                sample_id=sample_id,
                ratio=float(ratio),
                alpha=float(alpha),
                seed=package_seed,
            )
            transform_cache_key = (
                "weighted_transform",
                package_key,
                fold_tag,
                tuple(genes.astype(str)),
                tuple(float(q) for q in quantiles),
            )
        else:
            package, weights, info = builder(real_sample, syn, ratio=ratio, alpha=alpha, seed=package_seed)
        real_only = model_name == "baseline" or info.get("n_synthetic", 0) == 0
        cache_key = (
            "train_real",
            fold_tag,
            sample_id,
            tuple(genes.astype(str)),
            tuple(float(q) for q in quantiles),
        ) if real_only else None
        x_train.append(
            sample_feature(
                package,
                weights,
                pca,
                genes,
                quantiles,
                cache_key=cache_key,
                transform_cache_key=transform_cache_key,
            )
        )
        y_train.append(str(real_sample.obs["disease_label"].iloc[0]))
        row = {"sample_id": sample_id, "split": "train", **info}
        package_rows.append(row)
    for sample_id in test_samples:
        real_sample = subset_sample(real, sample_id)
        weights = np.full(real_sample.n_obs, 1.0 / max(1, real_sample.n_obs), dtype=np.float64)
        cache_key = (
            "test_real",
            fold_tag,
            sample_id,
            tuple(genes.astype(str)),
            tuple(float(q) for q in quantiles),
        )
        x_test.append(sample_feature(real_sample, weights, pca, genes, quantiles, cache_key=cache_key))
        y_test.append(str(real_sample.obs["disease_label"].iloc[0]))
        package_rows.append({"sample_id": sample_id, "split": "test", "n_real": int(real_sample.n_obs), "n_synthetic": 0})
    return (
        np.vstack(x_train),
        np.asarray(y_train),
        np.vstack(x_test),
        np.asarray(y_test),
        package_rows,
    )


def run_variant(
    real: ad.AnnData,
    splits: pd.DataFrame,
    cfg: dict,
    model_name: str,
    paradigm: str,
    ratio: float | None,
    alpha: float | None,
    variant_id: str,
    classifiers: list[str],
    fold_values: list[int],
    overwrite: bool = False,
) -> None:
    global ACTIVE_WEIGHTED_CACHE_GROUP
    generated_root = resolve_path(cfg["generated_root"])
    out_root = resolve_path(cfg["prediction_root"])
    n_pcs = int(cfg.get("n_pcs", 30))
    quantiles = [float(q) for q in cfg.get("feature_quantiles", [0.1, 0.5, 0.9])]
    seed = int(cfg.get("random_seed", 42))
    n_jobs = int(cfg.get("n_jobs", 4))
    if paradigm == "weighted_erm_package":
        cache_group = (model_name, float(ratio) if ratio is not None else None)
        if ACTIVE_WEIGHTED_CACHE_GROUP != cache_group:
            WEIGHTED_PACKAGE_CACHE.clear()
            TRANSFORM_CACHE.clear()
            ACTIVE_WEIGHTED_CACHE_GROUP = cache_group
    for fold in fold_values:
        expected_outputs = [
            out_root / model_name / paradigm / variant_id / f"fold_{fold}" / classifier / "predictions.csv"
            for classifier in classifiers
        ]
        if not overwrite and all(path.exists() for path in expected_outputs):
            print(f"skip existing {model_name}/{paradigm}/{variant_id}/fold_{fold}", flush=True)
            continue
        split_fold = splits[splits["fold"] == fold]
        train_samples = sorted(split_fold.loc[split_fold["split"] == "train", "sample_id"].astype(str).unique())
        test_samples = sorted(split_fold.loc[split_fold["split"] == "test", "sample_id"].astype(str).unique())
        pca, genes = get_fold_pca(real, int(fold), train_samples, n_pcs=n_pcs)
        x_train, y_train, x_test, y_test, package_rows = build_matrix(
            real,
            generated_root=generated_root,
            train_samples=train_samples,
            test_samples=test_samples,
            model_name=model_name,
            paradigm=paradigm,
            ratio=ratio,
            alpha=alpha,
            pca=pca,
            genes=genes,
            quantiles=quantiles,
            seed=seed + int(fold) * 1009,
        )
        encoder = LabelEncoder()
        y_train_enc = encoder.fit_transform(y_train)
        y_test_enc = encoder.transform(y_test)
        positive_index = int(np.where(encoder.classes_ == "SCCRCC")[0][0]) if "SCCRCC" in encoder.classes_ else 1
        for classifier in classifiers:
            dest = out_root / model_name / paradigm / variant_id / f"fold_{fold}" / classifier
            if not overwrite and (dest / "predictions.csv").exists():
                print(f"skip existing {dest / 'predictions.csv'}", flush=True)
                continue
            clf = classifier_params(classifier, n_jobs=n_jobs, seed=seed + int(fold))
            clf.fit(x_train, y_train_enc)
            pred_enc = clf.predict(x_test)
            score = predict_score(clf, x_test, positive_index=positive_index)
            y_pred = encoder.inverse_transform(np.asarray(pred_enc, dtype=int))
            rows = []
            for sample_id, yt, yp, sc in zip(test_samples, y_test, y_pred, score):
                rows.append({
                    "fold": int(fold),
                    "sample_id": sample_id,
                    "y_true": str(yt),
                    "y_pred": str(yp),
                    "score_SCCRCC": float(sc),
                    "model": model_name,
                    "paradigm": paradigm,
                    "variant_id": variant_id,
                    "ratio": ratio,
                    "alpha": alpha,
                    "classifier": classifier,
                })
            dest.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(dest / "predictions.csv", index=False)
            pd.DataFrame(package_rows).to_csv(dest / "packages.csv", index=False)
            manifest = {
                "model": model_name,
                "paradigm": paradigm,
                "variant_id": variant_id,
                "ratio": ratio,
                "alpha": alpha,
                "fold": int(fold),
                "classifier": classifier,
                "n_train_samples": int(len(train_samples)),
                "n_test_samples": int(len(test_samples)),
                "quick_accuracy": float(accuracy_score(y_test, y_pred)),
            }
            (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"wrote {dest / 'predictions.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sample_disease_prediction/Kidney/downstream.yaml")
    parser.add_argument("--interim-path", default=None)
    parser.add_argument("--splits-path", default=None)
    parser.add_argument("--generated-root", default=None)
    parser.add_argument("--prediction-root", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--paradigms", nargs="*", default=None)
    parser.add_argument("--classifiers", nargs="*", default=None)
    parser.add_argument("--ratios", nargs="*", type=float, default=None)
    parser.add_argument("--alphas", nargs="*", type=float, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Retained for compatibility; predictions skip existing outputs by default.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.interim_path is not None:
        cfg["interim_path"] = args.interim_path
    if args.splits_path is not None:
        cfg["splits_path"] = args.splits_path
    if args.generated_root is not None:
        cfg["generated_root"] = args.generated_root
    if args.prediction_root is not None:
        cfg["prediction_root"] = args.prediction_root
    if args.n_jobs is not None:
        cfg["n_jobs"] = int(args.n_jobs)
    if args.quick:
        quick = cfg.get("quick", {}) or {}
        cfg["ratios"] = quick.get("ratios", [1])
        cfg["alphas"] = quick.get("alphas", [0, 0.5])
        cfg["paradigms"] = quick.get("paradigms", ["real_baseline", "global_real_plus_synthetic", "weighted_erm_package"])
        cfg["classifiers"] = quick.get("classifiers", ["LR"])
    models = args.models or list(cfg.get("models", []))
    paradigms = args.paradigms or list(cfg.get("paradigms", []))
    classifiers = args.classifiers or list(cfg.get("classifiers", ["LR"]))
    ratios = [float(x) for x in (args.ratios if args.ratios is not None else cfg.get("ratios", [1]))]
    alphas = [float(x) for x in (args.alphas if args.alphas is not None else cfg.get("alphas", [0, 0.5]))]

    real = ad.read_h5ad(resolve_path(cfg["interim_path"]))
    splits = read_splits(cfg["splits_path"])
    fold_values = sorted(splits["fold"].unique().tolist())
    if args.quick:
        max_folds = int((cfg.get("quick", {}) or {}).get("max_folds", 2))
        fold_values = fold_values[:max_folds]

    if "real_baseline" in paradigms:
        for _, _, variant_id in paradigm_grid("real_baseline", ratios, alphas):
            run_variant(
                real,
                splits,
                cfg,
                "baseline",
                "real_baseline",
                None,
                None,
                variant_id,
                classifiers,
                fold_values,
                overwrite=args.overwrite,
            )
    for model_name in models:
        for paradigm in paradigms:
            if paradigm == "real_baseline":
                continue
            for ratio, alpha, variant_id in paradigm_grid(paradigm, ratios, alphas):
                run_variant(
                    real,
                    splits,
                    cfg,
                    model_name,
                    paradigm,
                    ratio,
                    alpha,
                    variant_id,
                    classifiers,
                    fold_values,
                    overwrite=args.overwrite,
                )


if __name__ == "__main__":
    main()
