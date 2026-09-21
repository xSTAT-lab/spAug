"""Visualize DLPFC supervised low-label evaluation summaries."""

from __future__ import annotations

import argparse
import os
import tempfile
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spaug_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import seaborn as sns

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from io_utils import resolve_path
else:
    from .io_utils import resolve_path


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def plot_best_bar(agg: pd.DataFrame, output_dir: Path):
    if "delta_macro_f1_mean" not in agg.columns:
        return
    best = (
        agg.sort_values(["model", "classifier", "delta_macro_f1_mean"], ascending=[True, True, False])
        .groupby(["model", "classifier"], dropna=False)
        .head(1)
        .copy()
    )
    plt.figure(figsize=(11, 5))
    sns.barplot(data=best, x="model", y="delta_macro_f1_mean", hue="classifier")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Delta macro F1 vs real baseline")
    plt.xlabel("Generator")
    plt.title("Best Supervised Low-Label Variant")
    savefig(output_dir / "best_delta_macro_f1.png")


def plot_ratio_curves(agg: pd.DataFrame, output_dir: Path):
    df = agg[
        agg["paradigm"].isin(["global_real_plus_synthetic", "global_synthetic_only", "local_spatial"])
        & agg["ratio"].notna()
    ].copy()
    if df.empty:
        return
    for classifier, sub in df.groupby("classifier", sort=True):
        plt.figure(figsize=(12, 5))
        sns.lineplot(
            data=sub,
            x="ratio",
            y="delta_macro_f1_mean",
            hue="model",
            style="paradigm",
            markers=True,
            dashes=False,
        )
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("Delta macro F1 vs real baseline")
        plt.xlabel("Synthetic ratio")
        plt.title(f"Ratio Sensitivity ({classifier})")
        savefig(output_dir / f"ratio_sensitivity_{classifier}.png")


def plot_alpha_heatmaps(agg: pd.DataFrame, output_dir: Path):
    df = agg[
        agg["paradigm"].isin(["weighted_erm_package", "regional_weighted_erm_package"])
        & agg["ratio"].notna()
        & agg["alpha"].notna()
    ].copy()
    if df.empty:
        return
    for (classifier, model, paradigm), sub in df.groupby(["classifier", "model", "paradigm"], sort=True):
        pivot = sub.pivot_table(index="alpha", columns="ratio", values="delta_macro_f1_mean", aggfunc="mean")
        plt.figure(figsize=(8, 4.5))
        sns.heatmap(pivot, center=0, cmap="vlag", annot=False)
        plt.xlabel("Synthetic ratio")
        plt.ylabel("Synthetic sample weight alpha")
        plt.title(f"{model} {paradigm} ({classifier})")
        savefig(output_dir / f"alpha_heatmap_{classifier}_{model}_{paradigm}.png")


def plot_paradigm_heatmaps(agg: pd.DataFrame, output_dir: Path):
    df = agg[agg["model"].astype(str) != "baseline"].copy()
    if df.empty or "delta_macro_f1_mean" not in df.columns:
        return
    best = (
        df.sort_values(
            ["model", "paradigm", "classifier", "delta_macro_f1_mean", "macro_f1_mean"],
            ascending=[True, True, True, False, False],
        )
        .groupby(["model", "paradigm", "classifier"], dropna=False)
        .head(1)
    )
    for classifier, sub in best.groupby("classifier", sort=True):
        pivot = sub.pivot_table(
            index="model",
            columns="paradigm",
            values="delta_macro_f1_mean",
            aggfunc="mean",
        )
        if pivot.empty:
            continue
        plt.figure(figsize=(10, 4.8))
        sns.heatmap(pivot, center=0, cmap="vlag", annot=True, fmt=".3f")
        plt.xlabel("Paradigm")
        plt.ylabel("Generator")
        plt.title(f"Best Delta Macro F1 by Paradigm ({classifier})")
        savefig(output_dir / "paradigm_heatmaps" / f"best_delta_macro_f1_{classifier}.png")


def plot_metric_scoreboard(agg: pd.DataFrame, output_dir: Path):
    df = agg[agg["model"].astype(str) != "baseline"].copy()
    if df.empty:
        return
    best = (
        df.sort_values(
            ["model", "classifier", "delta_macro_f1_mean", "macro_f1_mean"],
            ascending=[True, True, False, False],
        )
        .groupby(["model", "classifier"], dropna=False)
        .head(1)
        .copy()
    )
    metrics = [
        "accuracy_mean",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "weighted_f1_mean",
        "mcc_mean",
        "delta_macro_f1_mean",
        "delta_mcc_mean",
    ]
    metrics = [m for m in metrics if m in best.columns]
    if not metrics:
        return
    table = best.set_index(["model", "classifier"])[metrics]
    z = table.copy()
    for col in z.columns:
        std = z[col].std(ddof=0)
        z[col] = 0.0 if std == 0 or pd.isna(std) else (z[col] - z[col].mean()) / std
    plt.figure(figsize=(9, max(4, 0.32 * len(z))))
    sns.heatmap(z, cmap="viridis", annot=table.round(3), fmt="", cbar_kws={"label": "column z-score"})
    plt.xlabel("Metric")
    plt.ylabel("Best generator/classifier variant")
    plt.title("Low-Label Best Variant Scoreboard")
    savefig(output_dir / "scoreboards" / "best_variant_metric_scoreboard.png")


def plot_slice_stability(by_slice: pd.DataFrame, best: pd.DataFrame, output_dir: Path):
    if by_slice.empty or best.empty:
        return
    key_cols = ["model", "paradigm", "variant_id", "ratio", "alpha", "label_fraction", "classifier"]
    keys = (
        best.sort_values(["model", "classifier", "delta_macro_f1_mean"], ascending=[True, True, False])
        .groupby(["model", "classifier"], dropna=False)
        .head(1)[key_cols]
    )
    df = by_slice.merge(keys, on=key_cols, how="inner")
    if df.empty:
        return
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="model", y="delta_macro_f1", hue="classifier", showfliers=False)
    sns.stripplot(data=df, x="model", y="delta_macro_f1", hue="classifier", dodge=True, alpha=0.25, size=2, legend=False)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Delta macro F1 by slice/seed")
    plt.xlabel("Generator")
    plt.title("Best Variant Stability Across Slices and Seeds")
    savefig(output_dir / "slice_diagnostics" / "best_variant_delta_macro_f1_boxplot.png")

    pivot = df.pivot_table(
        index="slice_id",
        columns=["model", "classifier"],
        values="delta_macro_f1",
        aggfunc="mean",
    )
    if not pivot.empty:
        plt.figure(figsize=(max(10, 0.35 * len(pivot.columns)), 5))
        sns.heatmap(pivot, center=0, cmap="vlag")
        plt.xlabel("Generator / classifier")
        plt.ylabel("Slice")
        plt.title("Best Variant Delta Macro F1 by Slice")
        savefig(output_dir / "slice_diagnostics" / "best_variant_slice_heatmap.png")


def plot_label_heatmaps(label_agg: pd.DataFrame, best: pd.DataFrame, output_dir: Path):
    if label_agg.empty or best.empty:
        return
    key_cols = ["model", "paradigm", "variant_id", "ratio", "alpha", "label_fraction", "classifier"]
    keys = (
        best.sort_values(["model", "classifier", "delta_macro_f1_mean"], ascending=[True, True, False])
        .groupby(["model", "classifier"], dropna=False)
        .head(1)[key_cols]
    )
    df = label_agg.merge(keys, on=key_cols, how="inner")
    if df.empty:
        return
    for classifier, sub in df.groupby("classifier", sort=True):
        pivot = sub.pivot_table(index="model", columns="label", values="delta_f1_mean", aggfunc="mean")
        if pivot.empty:
            continue
        plt.figure(figsize=(10, 4.8))
        sns.heatmap(pivot, center=0, cmap="vlag", annot=True, fmt=".2f")
        plt.xlabel("True label")
        plt.ylabel("Generator")
        plt.title(f"Best Variant Delta F1 by Label ({classifier})")
        savefig(output_dir / "label_diagnostics" / f"best_variant_delta_f1_by_label_{classifier}.png")


def plot_weighted_best_alpha(best_alpha: pd.DataFrame, output_dir: Path):
    if best_alpha.empty:
        return
    df = best_alpha.copy()
    for col in ["ratio", "alpha", "delta_macro_f1_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for classifier, sub in df.groupby("classifier", sort=True):
        plt.figure(figsize=(11, 5))
        sns.lineplot(data=sub, x="ratio", y="alpha", hue="model", marker="o")
        plt.ylim(-0.03, 1.03)
        plt.xlabel("Synthetic ratio")
        plt.ylabel("Best alpha")
        plt.title(f"Weighted ERM Best Alpha by Ratio ({classifier})")
        savefig(output_dir / "weighted_erm" / f"best_alpha_by_ratio_{classifier}.png")

        plt.figure(figsize=(11, 5))
        sns.lineplot(data=sub, x="ratio", y="delta_macro_f1_mean", hue="model", marker="o")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("Synthetic ratio")
        plt.ylabel("Best delta macro F1")
        plt.title(f"Weighted ERM Best Delta by Ratio ({classifier})")
        savefig(output_dir / "weighted_erm" / f"best_delta_by_ratio_{classifier}.png")


def plot_confusion_matrices(confusion: pd.DataFrame, output_dir: Path):
    if confusion.empty:
        return
    best_groups = list(confusion.groupby(["model", "classifier", "paradigm", "variant_id"], sort=True))
    for (model, classifier, paradigm, variant), sub in best_groups:
        pivot = sub.pivot_table(index="true_label", columns="pred_label", values="row_fraction", aggfunc="sum").fillna(0)
        labels = sorted(set(pivot.index).union(set(pivot.columns)))
        pivot = pivot.reindex(index=labels, columns=labels, fill_value=0)
        plt.figure(figsize=(6.5, 5.5))
        sns.heatmap(pivot, cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f", cbar_kws={"label": "row fraction"})
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"{model} {classifier}\\n{paradigm}")
        safe = f"{model}_{classifier}_{paradigm}".replace("/", "_")
        savefig(output_dir / "confusion" / f"confusion_{safe}.png")


def visualize(summary_root: str | Path, output_dir: str | Path):
    summary_root = resolve_path(summary_root)
    output_dir = resolve_path(output_dir)
    agg_path = summary_root / "metrics_aggregate.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"Run evaluate.py first; missing {agg_path}")
    agg = pd.read_csv(agg_path)
    by_slice = read_csv_or_empty(summary_root / "metrics_by_slice.csv")
    best = read_csv_or_empty(summary_root / "best_variants_by_model_classifier.csv")
    label_agg = read_csv_or_empty(summary_root / "metrics_by_label_aggregate.csv")
    best_alpha = read_csv_or_empty(summary_root / "weighted_erm_best_alpha_by_ratio.csv")
    confusion = read_csv_or_empty(summary_root / "confusion_matrix_best_variants.csv")
    for col in ["ratio", "alpha", "delta_macro_f1_mean"]:
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce")
    sns.set_theme(style="whitegrid", context="notebook")
    plot_best_bar(agg, output_dir)
    plot_ratio_curves(agg, output_dir)
    plot_alpha_heatmaps(agg, output_dir)
    plot_paradigm_heatmaps(agg, output_dir)
    plot_metric_scoreboard(agg, output_dir)
    plot_slice_stability(by_slice, best, output_dir)
    plot_label_heatmaps(label_agg, best, output_dir)
    plot_weighted_best_alpha(best_alpha, output_dir)
    plot_confusion_matrices(confusion, output_dir)
    print(f"wrote figures to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize supervised low-label evaluation.")
    parser.add_argument("--summary-root", default="data/05_results/summary/supervised_low_label/DLPFC")
    parser.add_argument("--output-dir", default="data/05_results/figures/supervised_low_label/DLPFC")
    return parser


def main():
    args = build_parser().parse_args()
    visualize(args.summary_root, args.output_dir)


if __name__ == "__main__":
    main()
