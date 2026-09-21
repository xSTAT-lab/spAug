"""Evaluate DLPFC supervised low-label within-slice predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    confusion_matrix,
)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from io_utils import resolve_path
else:
    from .io_utils import resolve_path


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())


def read_manifest(path: Path) -> dict:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_predictions(results_root: str | Path) -> pd.DataFrame:
    root = resolve_path(results_root)
    files = sorted(root.rglob("predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No predictions.csv found under {root}")
    frames = []
    for file in files:
        df = pd.read_csv(file)
        manifest = read_manifest(file)
        for key, value in manifest.items():
            if key not in df.columns and (isinstance(value, (str, int, float, bool)) or value is None):
                df[key] = value
        df["prediction_file"] = str(file)
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


PREDICTION_USECOLS = ["slice_id", "y_true", "y_pred", "classifier"]


def fraction_tag(label_fraction: float) -> str:
    return f"fraction{int(round(float(label_fraction) * 100))}"


def iter_prediction_frames(results_root: str | Path, minimal: bool = True, label_fraction: float | None = None):
    root = resolve_path(results_root)
    files = sorted(root.rglob("predictions.csv"))
    if label_fraction is not None:
        tag = fraction_tag(label_fraction)
        files = [file for file in files if tag in file.parts]
    if not files:
        raise FileNotFoundError(f"No predictions.csv found under {root}")
    for file in files:
        if minimal:
            df = pd.read_csv(file, usecols=lambda col: col in PREDICTION_USECOLS)
        else:
            df = pd.read_csv(file)
        manifest = read_manifest(file)
        for key, value in manifest.items():
            if key not in df.columns and (isinstance(value, (str, int, float, bool)) or value is None):
                df[key] = value
        df["prediction_file"] = str(file)
        yield file, df


def compute_metrics(group: pd.DataFrame) -> pd.Series:
    y_true = group["y_true"].astype(str).to_numpy()
    y_pred = group["y_pred"].astype(str).to_numpy()
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    return pd.Series(
        {
            "n_test": int(len(group)),
            "n_classes_true": int(pd.Series(y_true).nunique()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(labels) > 1 else 0.0,
        }
    )


def normalize_empty_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_baseline_delta(summary: pd.DataFrame) -> pd.DataFrame:
    key = ["label_fraction", "low_label_seed", "slice_id", "classifier"]
    baseline = summary[summary["model"].astype(str) == "baseline"][
        key + ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc"]
    ].rename(
        columns={
            "accuracy": "baseline_accuracy",
            "balanced_accuracy": "baseline_balanced_accuracy",
            "macro_f1": "baseline_macro_f1",
            "weighted_f1": "baseline_weighted_f1",
            "mcc": "baseline_mcc",
        }
    )
    out = summary.merge(baseline, on=key, how="left")
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc"]:
        out[f"delta_{metric}"] = out[metric] - out[f"baseline_{metric}"]
    return out


def summarize_by_slice(pred: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in pred.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(compute_metrics(group).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_label(pred: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in pred.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        base = {col: value for col, value in zip(group_cols, key)}
        y_true = group["y_true"].astype(str).to_numpy()
        y_pred = group["y_pred"].astype(str).to_numpy()
        labels = sorted(pd.Series(y_true).unique().tolist())
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
        for label, p, r, f, s in zip(labels, precision, recall, f1, support):
            row = dict(base)
            row.update(
                {
                    "label": str(label),
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f),
                    "support": int(s),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def add_label_baseline_delta(label_metrics: pd.DataFrame) -> pd.DataFrame:
    key = ["label_fraction", "low_label_seed", "slice_id", "classifier", "label"]
    baseline = label_metrics[label_metrics["model"].astype(str) == "baseline"][
        key + ["precision", "recall", "f1"]
    ].rename(
        columns={
            "precision": "baseline_precision",
            "recall": "baseline_recall",
            "f1": "baseline_f1",
        }
    )
    out = label_metrics.merge(baseline, on=key, how="left")
    for metric in ["precision", "recall", "f1"]:
        out[f"delta_{metric}"] = out[metric] - out[f"baseline_{metric}"]
    return out


def summarize_best_slice_performance(by_slice: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    best_keys = (
        aggregate.sort_values(
            ["model", "classifier", "delta_macro_f1_mean", "macro_f1_mean"],
            ascending=[True, True, False, False],
        )
        .groupby(["model", "classifier"], dropna=False)
        .head(1)
    )
    key_cols = ["model", "paradigm", "variant_id", "ratio", "alpha", "label_fraction", "classifier"]
    keys = best_keys[key_cols].copy()
    merged = by_slice.merge(keys, on=key_cols, how="inner")
    if merged.empty:
        return pd.DataFrame()
    rows = []
    for key, group in merged.groupby(["model", "classifier", "paradigm", "variant_id"], dropna=False):
        model, classifier, paradigm, variant_id = key
        rows.append(
            {
                "model": model,
                "classifier": classifier,
                "paradigm": paradigm,
                "variant_id": variant_id,
                "n_slice_seed": int(group.shape[0]),
                "win_rate_macro_f1": float((group["delta_macro_f1"] > 0).mean()),
                "win_rate_accuracy": float((group["delta_accuracy"] > 0).mean()),
                "mean_delta_macro_f1": float(group["delta_macro_f1"].mean()),
                "median_delta_macro_f1": float(group["delta_macro_f1"].median()),
                "min_delta_macro_f1": float(group["delta_macro_f1"].min()),
                "max_delta_macro_f1": float(group["delta_macro_f1"].max()),
                "mean_macro_f1": float(group["macro_f1"].mean()),
                "mean_accuracy": float(group["accuracy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_paradigms(by_slice: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in by_slice.groupby(["model", "paradigm", "classifier"], dropna=False, sort=True):
        model, paradigm, classifier = key
        rows.append(
            {
                "model": model,
                "paradigm": paradigm,
                "classifier": classifier,
                "n_evaluations": int(group.shape[0]),
                "mean_macro_f1": float(group["macro_f1"].mean()),
                "mean_delta_macro_f1": float(group["delta_macro_f1"].mean()),
                "median_delta_macro_f1": float(group["delta_macro_f1"].median()),
                "win_rate_macro_f1": float((group["delta_macro_f1"] > 0).mean()),
                "mean_accuracy": float(group["accuracy"].mean()),
                "mean_delta_accuracy": float(group["delta_accuracy"].mean()),
                "mean_mcc": float(group["mcc"].mean()),
                "mean_delta_mcc": float(group["delta_mcc"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_parameter_response(aggregate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameter_cols = [
        "model",
        "paradigm",
        "classifier",
        "ratio",
        "alpha",
        "macro_f1_mean",
        "delta_macro_f1_mean",
        "balanced_accuracy_mean",
        "delta_balanced_accuracy_mean",
        "accuracy_mean",
        "delta_accuracy_mean",
        "mcc_mean",
        "delta_mcc_mean",
    ]
    existing = [c for c in parameter_cols if c in aggregate.columns]
    response = aggregate[existing].copy()
    weighted = response[response["paradigm"].astype(str) == "weighted_erm_package"].copy()
    if weighted.empty:
        best_alpha = pd.DataFrame()
    else:
        best_alpha = (
            weighted.sort_values(
                ["model", "classifier", "ratio", "delta_macro_f1_mean", "macro_f1_mean"],
                ascending=[True, True, True, False, False],
            )
            .groupby(["model", "classifier", "ratio"], dropna=False)
            .head(1)
            .reset_index(drop=True)
        )
    return response, best_alpha


def summarize_confusion_counts(group: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cols = group_cols + ["y_true", "y_pred"]
    out = group.groupby(cols, dropna=False, sort=True).size().reset_index(name="count")
    return out


def filter_confusion_for_best(confusion_counts: pd.DataFrame, aggregate: pd.DataFrame, max_variants: int = 30) -> pd.DataFrame:
    key_cols = ["model", "paradigm", "variant_id", "ratio", "alpha", "label_fraction", "classifier"]
    best = (
        aggregate.sort_values(
            ["model", "classifier", "delta_macro_f1_mean", "macro_f1_mean"],
            ascending=[True, True, False, False],
        )
        .groupby(["model", "classifier"], dropna=False)
        .head(1)
        .head(max_variants)
    )
    keys = best[key_cols].copy()
    for col in ["ratio", "alpha", "label_fraction"]:
        if col in keys.columns:
            keys[col] = pd.to_numeric(keys[col], errors="coerce")
    merged = confusion_counts.merge(keys, on=key_cols, how="inner")
    rows = []
    if merged.empty:
        return pd.DataFrame()
    for key, group in merged.groupby(["model", "classifier", "paradigm", "variant_id"], dropna=False):
        labels = sorted(set(group["y_true"].astype(str)).union(set(group["y_pred"].astype(str))))
        mat_df = (
            group.assign(y_true=group["y_true"].astype(str), y_pred=group["y_pred"].astype(str))
            .pivot_table(index="y_true", columns="y_pred", values="count", aggfunc="sum", fill_value=0)
            .reindex(index=labels, columns=labels, fill_value=0)
        )
        mat = mat_df.to_numpy(dtype=np.int64)
        denom = mat.sum(axis=1, keepdims=True)
        norm = np.divide(mat, denom, out=np.zeros_like(mat, dtype=float), where=denom != 0)
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                rows.append(
                    {
                        "model": key[0],
                        "classifier": key[1],
                        "paradigm": key[2],
                        "variant_id": key[3],
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": int(mat[i, j]),
                        "row_fraction": float(norm[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def evaluate(
    results_root: str | Path,
    output_root: str | Path,
    write_predictions: bool = False,
    label_fraction: float | None = None,
) -> dict[str, Path]:
    group_cols = [
        "model",
        "paradigm",
        "variant_id",
        "ratio",
        "alpha",
        "label_fraction",
        "low_label_seed",
        "classifier",
        "slice_id",
    ]
    by_slice_frames = []
    by_label_frames = []
    confusion_frames = []
    prediction_export_frames = [] if write_predictions else None
    for i, (_, frame) in enumerate(
        iter_prediction_frames(results_root, minimal=not write_predictions, label_fraction=label_fraction),
        start=1,
    ):
        frame = normalize_empty_numeric(frame, ["ratio", "alpha", "label_fraction", "low_label_seed"])
        if label_fraction is not None:
            frame = frame[np.isclose(pd.to_numeric(frame["label_fraction"], errors="coerce"), float(label_fraction))]
            if frame.empty:
                continue
        for col in group_cols:
            if col not in frame.columns:
                frame[col] = np.nan
        by_slice_frames.append(summarize_by_slice(frame, group_cols))
        by_label_frames.append(summarize_by_label(frame, group_cols))
        confusion_frames.append(summarize_confusion_counts(frame, aggregate_cols := [
            "model",
            "paradigm",
            "variant_id",
            "ratio",
            "alpha",
            "label_fraction",
            "classifier",
        ]))
        if prediction_export_frames is not None:
            prediction_export_frames.append(frame)
        if i % 10 == 0:
            print(f"[low-label evaluate] scanned {i} prediction files", flush=True)

    if not by_slice_frames:
        fraction_msg = "" if label_fraction is None else f" for label_fraction={label_fraction}"
        raise FileNotFoundError(f"No prediction rows found under {results_root}{fraction_msg}")

    by_slice = pd.concat(by_slice_frames, ignore_index=True, sort=False)
    by_slice = add_baseline_delta(by_slice)

    by_label = pd.concat(by_label_frames, ignore_index=True, sort=False)
    by_label = add_label_baseline_delta(by_label)

    aggregate_cols = [
        "model",
        "paradigm",
        "variant_id",
        "ratio",
        "alpha",
        "label_fraction",
        "classifier",
    ]
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "delta_accuracy",
        "delta_balanced_accuracy",
        "delta_macro_f1",
        "delta_weighted_f1",
        "delta_mcc",
    ]
    aggregate = (
        by_slice.groupby(aggregate_cols, dropna=False, sort=True)[metrics]
        .agg(["mean", "std", "median", "max"])
        .reset_index()
    )
    aggregate.columns = ["_".join([str(x) for x in col if str(x) != ""]).rstrip("_") for col in aggregate.columns]

    best = (
        aggregate.sort_values(["classifier", "delta_macro_f1_mean", "macro_f1_mean"], ascending=[True, False, False])
        .groupby(["model", "classifier"], dropna=False)
        .head(5)
        .reset_index(drop=True)
    )

    label_aggregate_cols = aggregate_cols + ["label"]
    label_metrics = ["precision", "recall", "f1", "delta_precision", "delta_recall", "delta_f1", "support"]
    label_aggregate = (
        by_label.groupby(label_aggregate_cols, dropna=False, sort=True)[label_metrics]
        .agg(["mean", "std", "median", "max"])
        .reset_index()
    )
    label_aggregate.columns = ["_".join([str(x) for x in col if str(x) != ""]).rstrip("_") for col in label_aggregate.columns]

    paradigm_summary = summarize_paradigms(by_slice)
    best_slice_summary = summarize_best_slice_performance(by_slice, aggregate)
    parameter_response, best_alpha = summarize_parameter_response(aggregate)
    confusion_all = pd.concat(confusion_frames, ignore_index=True, sort=False)
    confusion_best = filter_confusion_for_best(confusion_all, aggregate)

    out_root = resolve_path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "by_slice": out_root / "metrics_by_slice.csv",
        "aggregate": out_root / "metrics_aggregate.csv",
        "best": out_root / "best_variants_by_model_classifier.csv",
        "by_label": out_root / "metrics_by_label.csv",
        "label_aggregate": out_root / "metrics_by_label_aggregate.csv",
        "paradigm_summary": out_root / "paradigm_summary.csv",
        "best_slice_summary": out_root / "best_variant_slice_winrate.csv",
        "parameter_response": out_root / "parameter_response_summary.csv",
        "best_alpha": out_root / "weighted_erm_best_alpha_by_ratio.csv",
        "confusion_best": out_root / "confusion_matrix_best_variants.csv",
    }
    if prediction_export_frames is not None:
        pred = pd.concat(prediction_export_frames, ignore_index=True, sort=False)
        paths["predictions"] = out_root / "all_predictions.csv"
        pred.to_csv(paths["predictions"], index=False)
    by_slice.to_csv(paths["by_slice"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    best.to_csv(paths["best"], index=False)
    by_label.to_csv(paths["by_label"], index=False)
    label_aggregate.to_csv(paths["label_aggregate"], index=False)
    paradigm_summary.to_csv(paths["paradigm_summary"], index=False)
    best_slice_summary.to_csv(paths["best_slice_summary"], index=False)
    parameter_response.to_csv(paths["parameter_response"], index=False)
    best_alpha.to_csv(paths["best_alpha"], index=False)
    confusion_best.to_csv(paths["confusion_best"], index=False)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DLPFC supervised low-label predictions.")
    parser.add_argument("--results-root", default="data/05_results/supervised_low_label/DLPFC")
    parser.add_argument("--output-root", default="data/05_results/summary/supervised_low_label/DLPFC")
    parser.add_argument("--label-fraction", type=float, default=None, help="Optional label fraction filter, e.g. 0.5.")
    parser.add_argument("--write-predictions", action="store_true", help="Also write the very large all_predictions.csv.")
    return parser


def main():
    args = build_parser().parse_args()
    paths = evaluate(
        args.results_root,
        args.output_root,
        write_predictions=args.write_predictions,
        label_fraction=args.label_fraction,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
