"""Evaluate Brain sample-level disease prediction outputs."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_prediction_file(file: str | Path) -> pd.DataFrame:
    file = Path(file)
    df = pd.read_csv(file)
    df["prediction_file"] = str(file)
    return df


def read_predictions(root: Path, workers: int = 1) -> pd.DataFrame:
    files = sorted(root.rglob("predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No predictions.csv found under {root}")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(_read_prediction_file, files))
    else:
        frames = [_read_prediction_file(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def metric_row(group: pd.DataFrame, positive_label: str) -> dict:
    y_true = group["y_true"].astype(str).to_numpy()
    y_pred = group["y_pred"].astype(str).to_numpy()
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    out = {
        "n_test_samples": int(len(group)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(labels) > 1 else 0.0,
    }
    y_bin = (y_true == positive_label).astype(int)
    if "score_EPM" in group.columns and len(np.unique(y_bin)) > 1:
        score = pd.to_numeric(group["score_EPM"], errors="coerce").fillna(0.0).to_numpy()
        out["auroc"] = float(roc_auc_score(y_bin, score))
        out["auprc"] = float(average_precision_score(y_bin, score))
    else:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[positive_label],
        zero_division=0,
    )
    out["sensitivity"] = float(recall[0])
    tn_label = [x for x in labels if x != positive_label]
    if tn_label:
        neg = tn_label[0]
        out["specificity"] = float(((y_true == neg) & (y_pred == neg)).sum() / max(1, (y_true == neg).sum()))
    else:
        out["specificity"] = np.nan
    return out


def summarize(pred: pd.DataFrame, positive_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = ["model", "paradigm", "variant_id", "ratio", "alpha", "classifier", "fold"]
    rows = []
    for key, group in pred.groupby(group_cols, dropna=False, sort=True):
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(metric_row(group, positive_label))
        rows.append(row)
    by_fold = pd.DataFrame(rows)
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "mcc", "auroc", "auprc", "sensitivity", "specificity"]
    agg_cols = ["model", "paradigm", "variant_id", "ratio", "alpha", "classifier"]
    aggregate = by_fold.groupby(agg_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    aggregate.columns = ["_".join([str(x) for x in c if str(x)]) for c in aggregate.columns.to_flat_index()]

    baseline = by_fold[by_fold["model"].astype(str) == "baseline"][
        ["fold", "classifier"] + metric_cols
    ].rename(columns={m: f"baseline_{m}" for m in metric_cols})
    delta = by_fold.merge(baseline, on=["fold", "classifier"], how="left")
    for metric in metric_cols:
        delta[f"delta_{metric}"] = delta[metric] - delta[f"baseline_{metric}"]
    return by_fold, aggregate, delta


def confusion_counts(pred: pd.DataFrame) -> pd.DataFrame:
    cols = ["model", "paradigm", "variant_id", "classifier", "y_true", "y_pred"]
    return pred.groupby(cols, dropna=False).size().reset_index(name="count")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sample_disease_prediction/Brain/evaluation.yaml")
    parser.add_argument("--prediction-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    pred_root = resolve_path(args.prediction_root or cfg["prediction_root"])
    out_root = resolve_path(args.output_root or cfg["evaluation_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    pred = read_predictions(pred_root, workers=max(1, int(args.workers)))
    positive = str(cfg.get("positive_label", "EPM"))
    by_fold, aggregate, delta = summarize(pred, positive)
    conf = confusion_counts(pred)
    pred.to_csv(out_root / "all_predictions.csv", index=False)
    by_fold.to_csv(out_root / "metrics_by_fold.csv", index=False)
    aggregate.to_csv(out_root / "metrics_summary.csv", index=False)
    delta.to_csv(out_root / "metrics_with_baseline_delta.csv", index=False)
    conf.to_csv(out_root / "confusion_counts.csv", index=False)
    best = (
        delta[delta["model"].astype(str) != "baseline"]
        .sort_values(["delta_macro_f1", "macro_f1"], ascending=[False, False])
        .head(20)
    )
    best.to_csv(out_root / "top_delta_macro_f1.csv", index=False)
    manifest = {"prediction_root": str(pred_root), "n_prediction_rows": int(pred.shape[0]), "output_root": str(out_root)}
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
