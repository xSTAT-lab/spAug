"""Extended DLPFC spatial-clustering evaluation and parameter analysis."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
MODELS = ["SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion"]
MODEL_ORDER = ["baseline", *MODELS]
GLOBAL_RATIOS = list(range(1, 31))
LOCAL_MAX_RATIOS = [1, 2, 3, 4]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_labels(path: str | Path) -> pd.Series:
    labels = pd.read_csv(resolve_path(path))
    if {"obs_name", "label"}.issubset(labels.columns):
        return pd.Series(labels["label"].astype(str).values, index=labels["obs_name"].astype(str))
    first = labels.columns[0]
    label_col = "label" if "label" in labels.columns else labels.columns[-1]
    return pd.Series(labels[label_col].astype(str).values, index=labels[first].astype(str))


def load_reference_coords(path: str | Path) -> pd.DataFrame:
    adata = ad.read_h5ad(resolve_path(path), backed="r")
    try:
        obs = adata.obs.copy()
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        out = pd.DataFrame(
            {
                "obs_name": adata.obs_names.astype(str),
                "ref_slice_id": obs["slice_id"].astype(str).to_numpy(),
                "ref_x": coords[:, 0],
                "ref_y": coords[:, 1],
            }
        )
    finally:
        adata.file.close()
    return out


def spatial_grid_regions(coords: pd.DataFrame, n_bins: int = 2) -> pd.Series:
    out = pd.Series(index=coords.index, dtype="object")
    slice_col = "ref_slice_id" if "ref_slice_id" in coords.columns else "slice_id"
    for sid, idx in coords.groupby(coords[slice_col].astype(str)).groups.items():
        idx = list(idx)
        xy = coords.loc[idx, ["ref_x", "ref_y"]].to_numpy(dtype=np.float64)
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        bins = np.floor((xy - lo) / span * n_bins).astype(int).clip(0, n_bins - 1)
        out.loc[idx] = [f"slice_{sid}_grid_{x}_{y}" for x, y in bins]
    return out.astype(str)


def parse_context(path: Path, results_root: Path) -> dict:
    rel = path.relative_to(results_root)
    parts = rel.parts
    context = {"result_file": str(path), "relative_dir": str(rel.parent)}
    full_parts = path.parts
    for model in MODEL_ORDER:
        if model in parts or model in full_parts:
            context["model"] = model
            break
    context.setdefault("model", "unknown")
    context["dataset"] = "DLPFC" if "DLPFC" in parts else "unknown"
    context["mode"] = "unsupervised" if "unsupervised" in parts else "unknown"
    for family in ("global_real_plus_synthetic", "local_spatial"):
        if family in parts:
            context["paradigm_family"] = family
            context["paradigm"] = family
            idx = parts.index(family)
            if idx + 1 < len(parts):
                context["variant_id"] = parts[idx + 1]
            break
    if context.get("model") == "baseline":
        context["paradigm"] = "baseline"
        context["paradigm_family"] = "baseline"
    context["ablation"] = "path_B" if "path_B" in str(path) else "unknown"
    context.setdefault("variant_id", "baseline")
    m = re.search(r"_syn([0-9]+(?:p[0-9]+)?)_", context["variant_id"])
    if m:
        context["synthetic_ratio"] = float(m.group(1).replace("p", "."))
    else:
        context["synthetic_ratio"] = np.nan
    context["parameter_type"] = "baseline"
    if context.get("paradigm_family") == "global_real_plus_synthetic":
        context["parameter_type"] = "global_ratio"
    if context.get("paradigm_family") == "local_spatial":
        context["parameter_type"] = "local_max_ratio"
    return context


def mapped_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_labels = pd.Index(pd.Series(y_true).astype(str).unique())
    pred_labels = pd.Index(pd.Series(y_pred).astype(str).unique())
    table = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred"))
    cost = -table.to_numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    mapped_pred = np.array(["__unmapped__"] * len(y_pred), dtype=object)
    pred_to_true = {str(table.columns[c]): str(table.index[r]) for r, c in zip(row_ind, col_ind)}
    for i, pred in enumerate(y_pred.astype(str)):
        mapped_pred[i] = pred_to_true.get(pred, "__unmapped__")
    labels = sorted(set(y_true.astype(str)))
    precision = []
    recall = []
    f1 = []
    fdr = []
    for label in labels:
        yt = y_true.astype(str) == label
        yp = mapped_pred.astype(str) == label
        tp = int(np.sum(yt & yp))
        fp = int(np.sum(~yt & yp))
        fn = int(np.sum(yt & ~yp))
        p = tp / (tp + fp) if (tp + fp) else np.nan
        r = tp / (tp + fn) if (tp + fn) else np.nan
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if np.isfinite(p) and np.isfinite(r) and (p + r) else np.nan)
        fdr.append(fp / (tp + fp) if (tp + fp) else np.nan)
    return {
        "mapped_macro_precision": float(np.nanmean(precision)),
        "mapped_macro_recall": float(np.nanmean(recall)),
        "mapped_macro_f1": float(np.nanmean(f1)),
        "mapped_macro_fdr": float(np.nanmean(fdr)),
        "n_true_labels": int(len(true_labels)),
        "n_pred_clusters": int(len(pred_labels)),
        "cluster_count_delta": int(len(pred_labels) - len(true_labels)),
    }


def safe_spatial_silhouette(real: pd.DataFrame, max_n: int, seed: int) -> float:
    if real["pred_cluster"].nunique() < 2 or real.shape[0] < 3:
        return np.nan
    data = real.dropna(subset=["ref_x", "ref_y"]).copy()
    if data.shape[0] < 3 or data["pred_cluster"].nunique() < 2:
        return np.nan
    if data.shape[0] > max_n:
        data = data.sample(max_n, random_state=seed)
    counts = data["pred_cluster"].astype(str).value_counts()
    keep = set(counts[counts >= 2].index)
    data = data[data["pred_cluster"].astype(str).isin(keep)]
    if data["pred_cluster"].nunique() < 2:
        return np.nan
    x = StandardScaler().fit_transform(data[["ref_x", "ref_y"]].to_numpy(dtype=np.float64))
    return float(silhouette_score(x, data["pred_cluster"].astype(str).to_numpy()))


def evaluate_one(path: Path, labels: pd.Series, ref_coord_map: pd.DataFrame, results_root: Path, max_silhouette_n: int) -> tuple[dict, pd.DataFrame]:
    context = parse_context(path, results_root)
    df = pd.read_csv(path)
    if "slice_id" not in context and "slice_id" in df.columns and not df.empty:
        context["slice_id"] = str(df["slice_id"].astype(str).iloc[0])
    if "ablation" in df.columns and not df.empty:
        context["ablation"] = str(df["ablation"].astype(str).iloc[0])
    source = df["augmentation_source"].astype(str) if "augmentation_source" in df.columns else pd.Series("real", index=df.index)
    real = df[source != "synthetic"].copy()
    common = [x for x in real["obs_name"].astype(str) if x in labels.index]
    pred = pd.Series(real["pred_cluster"].astype(str).values, index=real["obs_name"].astype(str))
    y_true = labels.loc[common].astype(str).to_numpy()
    y_pred = pred.loc[common].astype(str).to_numpy()
    row = {
        **context,
        "status": "ok",
        "n_obs": int(df.shape[0]),
        "n_real": int(real.shape[0]),
        "n_synthetic": int(df.shape[0] - real.shape[0]),
        "n_eval": int(len(common)),
        "observed_synthetic_ratio": float((df.shape[0] - real.shape[0]) / real.shape[0]) if real.shape[0] else np.nan,
        "synthetic_fraction": float((df.shape[0] - real.shape[0]) / df.shape[0]) if df.shape[0] else np.nan,
    }
    if common:
        row.update(
            {
                "ari": float(adjusted_rand_score(y_true, y_pred)),
                "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
                "ami": float(adjusted_mutual_info_score(y_true, y_pred)),
                "homogeneity": float(homogeneity_score(y_true, y_pred)),
                "completeness": float(completeness_score(y_true, y_pred)),
                "v_measure": float(v_measure_score(y_true, y_pred)),
                "fowlkes_mallows": float(fowlkes_mallows_score(y_true, y_pred)),
            }
        )
        row.update(mapped_classification_metrics(y_true, y_pred))
    else:
        for key in ["ari", "nmi", "ami", "homogeneity", "completeness", "v_measure", "fowlkes_mallows"]:
            row[key] = np.nan
    real_with_coords = real.merge(ref_coord_map, on="obs_name", how="left")
    row["spatial_silhouette_real"] = safe_spatial_silhouette(real_with_coords, max_silhouette_n, seed=42)
    region_rows = pd.DataFrame()
    if context.get("paradigm_family") == "local_spatial":
        region_rows = local_region_usage(df, ref_coord_map, context, labels)
    return row, region_rows


def local_region_usage(df: pd.DataFrame, ref_coords: pd.DataFrame, context: dict, labels: pd.Series) -> pd.DataFrame:
    source = df["augmentation_source"].astype(str) if "augmentation_source" in df.columns else pd.Series("real", index=df.index)
    work = df.copy()
    work["_source"] = source.values

    if {"spatial_x", "spatial_y", "slice_id"}.issubset(work.columns):
        merged = work.copy()
        merged["ref_slice_id"] = merged["slice_id"].astype(str)
        merged["ref_x"] = pd.to_numeric(merged["spatial_x"], errors="coerce")
        merged["ref_y"] = pd.to_numeric(merged["spatial_y"], errors="coerce")
    else:
        ref = ref_coords.set_index("obs_name")
        coord_key = np.where(
            work["_source"].eq("synthetic"),
            work.get("source_obs_name", work["obs_name"]).astype(str),
            work["obs_name"].astype(str),
        )
        work["_coord_key"] = coord_key
        coords = ref.reindex(work["_coord_key"].astype(str))[["ref_slice_id", "ref_x", "ref_y"]].reset_index(drop=True)
        merged = pd.concat([work.reset_index(drop=True), coords], axis=1)

    merged = merged.dropna(subset=["slice_id", "ref_x", "ref_y"]).copy()
    merged["_region"] = spatial_grid_regions(merged[["ref_slice_id", "ref_x", "ref_y"]])
    counts = (
        merged.groupby(["ref_slice_id", "_region", "_source"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"ref_slice_id": "slice_id", "_region": "region_id", "real": "n_real_region", "synthetic": "n_synthetic_region"})
    )
    if "n_real_region" not in counts:
        counts["n_real_region"] = 0
    if "n_synthetic_region" not in counts:
        counts["n_synthetic_region"] = 0
    counts["observed_region_ratio"] = counts["n_synthetic_region"] / counts["n_real_region"].replace(0, np.nan)
    counts["target_max_ratio"] = context.get("synthetic_ratio", np.nan)
    counts["hit_target_or_cap_proxy"] = counts["observed_region_ratio"] >= (counts["target_max_ratio"] - 1e-9)
    real_eval = merged[merged["_source"].eq("real") & merged["obs_name"].astype(str).isin(labels.index)].copy()
    region_metrics = []
    for (slice_id, region_id), sub in real_eval.groupby(["ref_slice_id", "_region"], dropna=False):
        if sub.shape[0] < 2:
            ari = np.nan
            nmi = np.nan
        else:
            y_true = labels.loc[sub["obs_name"].astype(str)].astype(str).to_numpy()
            y_pred = sub["pred_cluster"].astype(str).to_numpy()
            ari = float(adjusted_rand_score(y_true, y_pred))
            nmi = float(normalized_mutual_info_score(y_true, y_pred))
        region_metrics.append(
            {
                "slice_id": str(slice_id),
                "region_id": str(region_id),
                "n_eval_region": int(sub.shape[0]),
                "region_ari": ari,
                "region_nmi": nmi,
            }
        )
    if region_metrics:
        counts = counts.merge(pd.DataFrame(region_metrics), on=["slice_id", "region_id"], how="left")
    else:
        counts["n_eval_region"] = 0
        counts["region_ari"] = np.nan
        counts["region_nmi"] = np.nan
    for key, value in context.items():
        counts[key] = value
    return counts


def discover_prediction_files(results_root: Path) -> list[Path]:
    return sorted(results_root.glob("**/spatial_clusters.csv"))


def add_baseline_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    baseline = out[out["model"] == "baseline"][["slice_id", "ablation", "ari", "nmi", "ami", "v_measure", "spatial_silhouette_real"]]
    baseline = baseline.rename(columns={c: f"baseline_{c}" for c in baseline.columns if c not in {"slice_id", "ablation"}})
    out = out.merge(baseline, on=["slice_id", "ablation"], how="left")
    for metric in ["ari", "nmi", "ami", "v_measure", "spatial_silhouette_real"]:
        base = f"baseline_{metric}"
        if base in out:
            out[f"delta_{metric}"] = out[metric] - out[base]
    return out


def summarize_parameter_response(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = metrics[metrics["model"] != "baseline"].copy()
    group_cols = ["model", "paradigm_family", "parameter_type", "synthetic_ratio", "variant_id"]
    summary = (
        data.groupby(group_cols, dropna=False)
        .agg(
            n_slices=("slice_id", "nunique"),
            mean_ari=("ari", "mean"),
            mean_delta_ari=("delta_ari", "mean"),
            mean_nmi=("nmi", "mean"),
            mean_delta_nmi=("delta_nmi", "mean"),
            mean_ami=("ami", "mean"),
            mean_v_measure=("v_measure", "mean"),
            mean_observed_synthetic_ratio=("observed_synthetic_ratio", "mean"),
            mean_spatial_silhouette_real=("spatial_silhouette_real", "mean"),
        )
        .reset_index()
    )
    best_rows = []
    for (model, family), sub in summary.groupby(["model", "paradigm_family"], dropna=False):
        for metric in ["mean_delta_ari", "mean_delta_nmi", "mean_ari", "mean_nmi"]:
            row = sub.sort_values(metric, ascending=False).iloc[0].to_dict()
            row["best_by"] = metric
            best_rows.append(row)
    best = pd.DataFrame(best_rows)
    return summary, best


def filter_formal_matrix(
    metrics: pd.DataFrame,
    region_usage: pd.DataFrame,
    expected_slices: int,
    global_max_ratio: int | None,
    keep_incomplete: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the formal matrix consistent before summary/best-parameter analysis."""
    out = metrics.copy()
    if global_max_ratio is not None:
        drop_global = (
            out["paradigm_family"].eq("global_real_plus_synthetic")
            & out["synthetic_ratio"].notna()
            & (out["synthetic_ratio"] > float(global_max_ratio))
        )
        out = out.loc[~drop_global].copy()

    if not keep_incomplete and expected_slices > 0:
        group_cols = ["model", "paradigm_family", "variant_id"]
        complete = (
            out.groupby(group_cols, dropna=False)["slice_id"]
            .nunique()
            .reset_index(name="n_complete_slices")
        )
        keep_keys = complete.loc[
            (complete["model"].eq("baseline")) | (complete["n_complete_slices"] >= expected_slices),
            group_cols,
        ]
        out = out.merge(keep_keys, on=group_cols, how="inner")

    if region_usage.empty:
        return out, region_usage
    region = region_usage.copy()
    keep_metric_keys = out[["model", "paradigm_family", "variant_id"]].drop_duplicates()
    region = region.merge(keep_metric_keys, on=["model", "paradigm_family", "variant_id"], how="inner")
    return out, region


def summarize_local_regions(region: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if region.empty:
        return pd.DataFrame(), pd.DataFrame()
    by_region = (
        region.groupby(["model", "synthetic_ratio", "slice_id", "region_id"], dropna=False)
        .agg(
            n_runs=("result_file", "count"),
            n_real_region=("n_real_region", "mean"),
            n_synthetic_region=("n_synthetic_region", "mean"),
            observed_region_ratio=("observed_region_ratio", "mean"),
            target_max_ratio=("target_max_ratio", "mean"),
            n_eval_region=("n_eval_region", "mean"),
            region_ari=("region_ari", "mean"),
            region_nmi=("region_nmi", "mean"),
        )
        .reset_index()
    )
    best_region = (
        region.groupby(["model", "slice_id", "region_id"], dropna=False)
        .agg(
            max_observed_region_ratio=("observed_region_ratio", "max"),
            mean_observed_region_ratio=("observed_region_ratio", "mean"),
            n_real_region=("n_real_region", "mean"),
        )
        .reset_index()
    )
    return by_region, best_region


def summarize_local_region_performance(region_summary: pd.DataFrame) -> pd.DataFrame:
    if region_summary.empty:
        return pd.DataFrame()
    rows = []
    metric_cols = ["region_ari", "region_nmi"]
    for (model, slice_id, region_id), sub in region_summary.groupby(["model", "slice_id", "region_id"], dropna=False):
        for metric in metric_cols:
            valid = sub.dropna(subset=[metric])
            if valid.empty:
                continue
            row = valid.sort_values(metric, ascending=False).iloc[0].to_dict()
            row["best_by"] = metric
            rows.append(row)
    return pd.DataFrame(rows)


def metric_coverage(output_dir: Path):
    rows = [
        ("SpatialSimBench clustering", "ARI", "covered", "evaluate_tasks + extended"),
        ("SpatialSimBench clustering", "NMI", "covered", "evaluate_tasks + extended"),
        ("SpatialSimBench clustering", "mapped precision/recall/FDR", "covered_with_hungarian_mapping", "extended"),
        ("SpatialSimBench clustering", "silhouette width", "partially_covered", "spatial-coordinate silhouette on real spots; expression silhouette requires X in predictions"),
        ("SpatialSimBench downstream spatial", "cell type proportion/deconvolution JSD/RMSE", "not_applicable_current_task", "requires deconvolution/cell proportion task"),
        ("SpatialSimBench downstream spatial", "Moran cosine/Mantel/correlation", "covered_in_intrinsic_not_downstream", "intrinsic metrics already compute Moran's I difference; full cosine/Mantel can be added for generation fidelity"),
        ("Paradigm parameter", "global synthetic ratio response", "covered", "parameter_response_summary and plots"),
        ("Paradigm parameter", "local region observed ratio", "covered_proxy", "uses spatial grid and source_obs_name/ref coords for current results; future outputs include spatial_x/y"),
        ("Paradigm parameter", "local region best max ratio", "covered", "region-level ARI/NMI are recomputed on real spots within spatial grid regions"),
    ]
    pd.DataFrame(rows, columns=["dimension", "metric_or_view", "status", "note"]).to_csv(output_dir / "metric_coverage_vs_spatialsimbench.csv", index=False)


def run_extended(
    results_root: str | Path = "data/05_results/spatial_cluster/DLPFC",
    output_dir: str | Path = "data/05_results/summary/spatial_cluster/DLPFC",
    labels_path: str | Path = "data/02_interim/spatial_cluster/DLPFC/evaluation/labels_for_evaluation.csv",
    reference_path: str | Path = "data/02_interim/spatial_cluster/DLPFC/model_inputs/SRTsim/unsupervised/processed_with_split.h5ad",
    max_silhouette_n: int = 2000,
    expected_slices: int = 12,
    global_max_ratio: int | None = 20,
    keep_incomplete: bool = False,
) -> dict[str, pd.DataFrame]:
    results_root = resolve_path(results_root)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(labels_path)
    ref_coords = load_reference_coords(reference_path)
    files = discover_prediction_files(results_root)
    rows = []
    region_frames = []
    for i, path in enumerate(files, start=1):
        row, region = evaluate_one(path, labels, ref_coords, results_root, max_silhouette_n=max_silhouette_n)
        rows.append(row)
        if not region.empty:
            region_frames.append(region)
        if i % 100 == 0:
            print(f"[extended] {i}/{len(files)} files", flush=True)
    metrics_all = add_baseline_deltas(pd.DataFrame(rows))
    region_usage = pd.concat(region_frames, ignore_index=True, sort=False) if region_frames else pd.DataFrame()
    metrics, region_usage = filter_formal_matrix(
        metrics_all,
        region_usage,
        expected_slices=expected_slices,
        global_max_ratio=global_max_ratio,
        keep_incomplete=keep_incomplete,
    )
    param_summary, best = summarize_parameter_response(metrics)
    region_by, region_best = summarize_local_regions(region_usage)
    region_perf_best = summarize_local_region_performance(region_by)
    metrics_all.to_csv(output_dir / "extended_spatial_clustering_metrics_all_discovered.csv", index=False)
    metrics.to_csv(output_dir / "extended_spatial_clustering_metrics.csv", index=False)
    param_summary.to_csv(output_dir / "parameter_response_summary.csv", index=False)
    best.to_csv(output_dir / "best_parameter_by_model_family.csv", index=False)
    region_usage.to_csv(output_dir / "local_spatial_region_usage_long.csv", index=False)
    region_by.to_csv(output_dir / "local_spatial_region_usage_summary.csv", index=False)
    region_best.to_csv(output_dir / "local_spatial_region_ratio_profile.csv", index=False)
    region_perf_best.to_csv(output_dir / "local_spatial_region_best_parameter.csv", index=False)
    metric_coverage(output_dir)
    pd.DataFrame(
        [
            {
                "expected_slices": expected_slices,
                "global_max_ratio": global_max_ratio,
                "keep_incomplete": keep_incomplete,
                "n_discovered_files": len(files),
                "n_metric_rows_all": int(metrics_all.shape[0]),
                "n_metric_rows_formal": int(metrics.shape[0]),
            }
        ]
    ).to_csv(output_dir / "evaluation_filter_report.csv", index=False)
    return {
        "metrics": metrics,
        "parameter_response": param_summary,
        "best": best,
        "region_usage": region_usage,
        "region_summary": region_by,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extended DLPFC clustering evaluation.")
    parser.add_argument("--results-root", default="data/05_results/spatial_cluster/DLPFC")
    parser.add_argument("-o", "--output-dir", default="data/05_results/summary/spatial_cluster/DLPFC")
    parser.add_argument("--labels", default="data/02_interim/spatial_cluster/DLPFC/evaluation/labels_for_evaluation.csv")
    parser.add_argument("--reference", default="data/02_interim/spatial_cluster/DLPFC/model_inputs/SRTsim/unsupervised/processed_with_split.h5ad")
    parser.add_argument("--max-silhouette-n", type=int, default=2000)
    parser.add_argument("--expected-slices", type=int, default=12)
    parser.add_argument("--global-max-ratio", type=int, default=20)
    parser.add_argument("--keep-incomplete", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    run_extended(
        results_root=args.results_root,
        output_dir=args.output_dir,
        labels_path=args.labels,
        reference_path=args.reference,
        max_silhouette_n=args.max_silhouette_n,
        expected_slices=args.expected_slices,
        global_max_ratio=args.global_max_ratio,
        keep_incomplete=args.keep_incomplete,
    )


if __name__ == "__main__":
    main()
