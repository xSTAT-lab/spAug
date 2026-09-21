"""Extended visualizations for DLPFC clustering paradigm evaluation."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spaug_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
MODEL_ORDER = ["SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion"]
PALETTE = {
    "global_real_plus_synthetic": "#4C78A8",
    "local_spatial": "#54A24B",
}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_global_ratio_curves(param: pd.DataFrame, out: Path):
    data = param[param["paradigm_family"] == "global_real_plus_synthetic"].copy()
    if data.empty:
        return
    for metric in ["mean_delta_ari", "mean_delta_nmi", "mean_ari", "mean_nmi"]:
        plt.figure(figsize=(8.5, 4.8))
        sns.lineplot(
            data=data,
            x="synthetic_ratio",
            y=metric,
            hue="model",
            hue_order=[m for m in MODEL_ORDER if m in set(data["model"])],
            marker="o",
        )
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("global synthetic ratio")
        plt.ylabel(metric)
        plt.title(f"Global real + synthetic: {metric} across ratios")
        savefig(out / "parameter_response" / f"global_ratio_{metric}.png")


def plot_local_ratio_curves(param: pd.DataFrame, out: Path):
    data = param[param["paradigm_family"] == "local_spatial"].copy()
    if data.empty:
        return
    for metric in ["mean_delta_ari", "mean_delta_nmi", "mean_ari", "mean_nmi"]:
        plt.figure(figsize=(7.5, 4.6))
        sns.lineplot(
            data=data,
            x="synthetic_ratio",
            y=metric,
            hue="model",
            hue_order=[m for m in MODEL_ORDER if m in set(data["model"])],
            marker="o",
        )
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("local max synthetic ratio")
        plt.ylabel(metric)
        plt.title(f"Local spatial: {metric} across max ratios")
        savefig(out / "parameter_response" / f"local_max_ratio_{metric}.png")


def plot_best_heatmap(best: pd.DataFrame, out: Path):
    if best.empty:
        return
    for metric in ["mean_delta_ari", "mean_delta_nmi"]:
        data = best[best["best_by"] == metric].copy()
        if data.empty:
            continue
        pivot = data.pivot_table(index="model", columns="paradigm_family", values="synthetic_ratio", aggfunc="first")
        pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])
        plt.figure(figsize=(5.8, 3.8))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, cbar_kws={"label": "best ratio"})
        plt.title(f"Best parameter by {metric}")
        plt.xlabel("")
        plt.ylabel("")
        savefig(out / "parameter_response" / f"best_ratio_{metric}.png")


def plot_metric_heatmap(param: pd.DataFrame, out: Path):
    if param.empty:
        return
    data = param.copy()
    data["column"] = data["paradigm_family"].astype(str) + " / " + data["synthetic_ratio"].astype(int).astype(str)
    for metric in ["mean_delta_ari", "mean_delta_nmi"]:
        pivot = data.pivot_table(index="model", columns="column", values=metric, aggfunc="mean")
        pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])
        plt.figure(figsize=(13, 4.2))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, linewidths=0.4)
        plt.title(f"Parameter grid: {metric}")
        plt.xlabel("")
        plt.ylabel("")
        savefig(out / "parameter_response" / f"parameter_grid_{metric}.png")


def plot_local_region_usage(region: pd.DataFrame, out: Path):
    if region.empty:
        return
    data = region.copy()
    plt.figure(figsize=(8.5, 4.8))
    sns.boxplot(data=data, x="synthetic_ratio", y="observed_region_ratio", hue="model", hue_order=[m for m in MODEL_ORDER if m in set(data["model"])])
    plt.xlabel("local max synthetic ratio")
    plt.ylabel("observed region synthetic / real ratio")
    plt.title("Local spatial: region-level observed ratios")
    savefig(out / "local_region" / "region_observed_ratio_boxplot.png")

    profile = (
        data.groupby(["model", "synthetic_ratio"], dropna=False)
        .agg(
            mean_region_ratio=("observed_region_ratio", "mean"),
            median_region_ratio=("observed_region_ratio", "median"),
            p90_region_ratio=("observed_region_ratio", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )
    for metric in ["mean_region_ratio", "median_region_ratio", "p90_region_ratio"]:
        plt.figure(figsize=(7.5, 4.6))
        sns.lineplot(data=profile, x="synthetic_ratio", y=metric, hue="model", marker="o")
        plt.xlabel("local max synthetic ratio")
        plt.ylabel(metric)
        plt.title(f"Local spatial: {metric}")
        savefig(out / "local_region" / f"{metric}.png")


def plot_local_region_performance(region: pd.DataFrame, best_region: pd.DataFrame, out: Path):
    if not region.empty:
        for metric in ["region_ari", "region_nmi"]:
            if metric not in region:
                continue
            plt.figure(figsize=(8.5, 4.8))
            sns.boxplot(
                data=region,
                x="synthetic_ratio",
                y=metric,
                hue="model",
                hue_order=[m for m in MODEL_ORDER if m in set(region["model"])],
            )
            plt.xlabel("local max synthetic ratio")
            plt.ylabel(metric)
            plt.title(f"Local spatial: region-level {metric}")
            savefig(out / "local_region" / f"{metric}_boxplot.png")

    if best_region.empty or "region_ari" not in best_region:
        return
    data = best_region[best_region["best_by"] == "region_ari"].copy()
    if data.empty:
        return
    data["slice_region"] = data["slice_id"].astype(str) + " / " + data["region_id"].astype(str).str.replace(r"^slice_[^_]+_", "", regex=True)
    for model in [m for m in MODEL_ORDER if m in set(data["model"])]:
        sub = data[data["model"] == model]
        pivot = sub.pivot_table(index="slice_id", columns="region_id", values="synthetic_ratio", aggfunc="first")
        plt.figure(figsize=(8, max(4, 0.34 * len(pivot))))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.4, cbar_kws={"label": "best local max ratio"})
        plt.title(f"{model}: region-level best local max ratio by ARI")
        plt.xlabel("region_id")
        plt.ylabel("slice_id")
        savefig(out / "local_region" / f"{model}_region_best_ratio_by_ari.png")


def plot_slice_heatmaps(metrics: pd.DataFrame, out: Path):
    data = metrics[(metrics["model"] != "baseline") & metrics["delta_ari"].notna()].copy()
    if data.empty:
        return
    # Keep best ARI variant per model/family for compact slice diagnostics.
    idx = data.groupby(["model", "paradigm_family"], dropna=False)["delta_ari"].idxmax()
    best = data.loc[idx, ["model", "paradigm_family", "variant_id"]]
    merged = data.merge(best, on=["model", "paradigm_family", "variant_id"], how="inner")
    merged["row"] = merged["model"].astype(str) + " / " + merged["paradigm_family"].astype(str)
    for metric in ["delta_ari", "delta_nmi"]:
        pivot = merged.pivot_table(index="row", columns="slice_id", values=metric, aggfunc="mean")
        plt.figure(figsize=(12, max(4, 0.38 * len(pivot))))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, linewidths=0.4)
        plt.title(f"Best variant per model/family: {metric} by slice")
        plt.xlabel("slice_id")
        plt.ylabel("")
        savefig(out / "slice_diagnostics" / f"best_variant_{metric}_by_slice.png")


def visualize(
    extended_dir: str | Path = "data/05_results/summary/spatial_cluster/DLPFC",
    output_dir: str | Path = "data/05_results/figures/spatial_cluster/DLPFC",
):
    extended_dir = resolve_path(extended_dir)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    metrics = pd.read_csv(extended_dir / "extended_spatial_clustering_metrics.csv")
    param = pd.read_csv(extended_dir / "parameter_response_summary.csv")
    best = pd.read_csv(extended_dir / "best_parameter_by_model_family.csv")
    region_path = extended_dir / "local_spatial_region_usage_summary.csv"
    region = pd.read_csv(region_path) if region_path.exists() else pd.DataFrame()
    best_region_path = extended_dir / "local_spatial_region_best_parameter.csv"
    best_region = pd.read_csv(best_region_path) if best_region_path.exists() else pd.DataFrame()
    plot_global_ratio_curves(param, output_dir)
    plot_local_ratio_curves(param, output_dir)
    plot_best_heatmap(best, output_dir)
    plot_metric_heatmap(param, output_dir)
    plot_local_region_usage(region, output_dir)
    plot_local_region_performance(region, best_region, output_dir)
    plot_slice_heatmaps(metrics, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize extended DLPFC clustering evaluation.")
    parser.add_argument("--extended-dir", default="data/05_results/summary/spatial_cluster/DLPFC")
    parser.add_argument("--output-dir", default="data/05_results/figures/spatial_cluster/DLPFC")
    return parser


def main():
    args = build_parser().parse_args()
    visualize(args.extended_dir, args.output_dir)


if __name__ == "__main__":
    main()
