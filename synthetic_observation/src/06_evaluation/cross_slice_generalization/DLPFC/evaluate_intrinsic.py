"""Intrinsic-fidelity evaluation for DLPFC cross-slice synthetic pools."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.spatial.distance import jensenshannon  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402


def load_spatial_cluster_intrinsic_module():
    path = PROJECT_ROOT / "src" / "06_evaluation" / "spatial_cluster" / "DLPFC" / "evaluate_intrinsic.py"
    spec = importlib.util.spec_from_file_location("_spaug_spatial_cluster_intrinsic", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load intrinsic helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_intrinsic = load_spatial_cluster_intrinsic_module()
FidelityResult = _intrinsic.FidelityResult
_align_and_sample = _intrinsic._align_and_sample
_dense = _intrinsic._dense
_hist_prob = _intrinsic._hist_prob
_mean_morans_i = _intrinsic._mean_morans_i
_metadata_from_path = _intrinsic._metadata_from_path
_safe_corr = _intrinsic._safe_corr
load_intrinsic_defaults = _intrinsic.load_intrinsic_defaults
resolve_path = _intrinsic.resolve_path


KNOWN_MODELS = ("SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion")


def model_from_path(path: Path) -> str:
    for part in path.parts:
        if part in KNOWN_MODELS:
            return part
    return "unknown"


def slice_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("slice_"):
        return stem.replace("slice_", "", 1)
    raise ValueError(f"Cannot infer slice_id from {path}")


def reference_path_for(model: str, split_tag: str) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "02_interim"
        / "cross_slice_generalization"
        / "DLPFC"
        / "model_inputs"
        / model
        / split_tag
        / "processed_with_split.h5ad"
    )


def discover_slice_files(
    synthetic_root: Path,
    split_tag: str,
    models: set[str] | None = None,
    slice_ids: set[str] | None = None,
) -> list[Path]:
    out: list[Path] = []
    for model in KNOWN_MODELS:
        if models is not None and model not in models:
            continue
        root = synthetic_root / model / split_tag / "pool_40x" / "slices"
        for path in sorted(root.glob("slice_*.h5ad")):
            if slice_ids is not None and slice_id_from_path(path) not in slice_ids:
                continue
            out.append(path)
    return sorted(out)


def subset_reference_slice(backed: ad.AnnData, slice_id: str) -> ad.AnnData:
    mask = backed.obs["slice_id"].astype(str).to_numpy() == str(slice_id)
    if not bool(np.any(mask)):
        raise ValueError(f"Reference has no slice_id={slice_id}")
    return backed[mask, :].to_memory()


def compute_fidelity_pair(
    real: ad.AnnData,
    synthetic_path: Path,
    config_path: str | Path,
    mode: str,
    sample_size: int | None,
    sample_frac: float | None,
    random_seed: int,
) -> FidelityResult:
    del config_path
    synthetic_backed = ad.read_h5ad(synthetic_path, backed="r")
    try:
        real_sampled, synthetic = _align_and_sample(real, synthetic_backed, sample_size, sample_frac, random_seed)
    finally:
        synthetic_backed.file.close()

    xr = np.maximum(_dense(real_sampled.X), 0.0)
    xs = np.maximum(_dense(synthetic.X), 0.0)
    max_val = float(max(np.max(xr) if xr.size else 0.0, np.max(xs) if xs.size else 0.0, 1.0))
    pr = _hist_prob(xr.ravel(), bins=50, value_range=(0.0, max_val))
    ps = _hist_prob(xs.ravel(), bins=50, value_range=(0.0, max_val))
    global_jsd = float(jensenshannon(pr, ps, base=2.0) ** 2)

    gene_jsd = []
    for j in range(xr.shape[1]):
        pr_g = _hist_prob(xr[:, j], bins=30, value_range=(0.0, max_val))
        ps_g = _hist_prob(xs[:, j], bins=30, value_range=(0.0, max_val))
        gene_jsd.append(float(jensenshannon(pr_g, ps_g, base=2.0) ** 2))

    lib_r = xr.sum(axis=1)
    lib_s = xs.sum(axis=1)
    mean_r = xr.mean(axis=0)
    mean_s = xs.mean(axis=0)
    var_r = xr.var(axis=0)
    var_s = xs.var(axis=0)
    spot_totals_r = np.sort(lib_r)
    spot_totals_s = np.sort(lib_s)
    n_pair = min(spot_totals_r.size, spot_totals_s.size)
    spot_total_corr = _safe_corr(spot_totals_r[:n_pair], spot_totals_s[:n_pair]) if n_pair > 1 else 0.0

    coords_r = np.asarray(real_sampled.obsm["spatial"][:, :2], dtype=np.float64) if "spatial" in real_sampled.obsm else np.empty((0, 2))
    coords_s = np.asarray(synthetic.obsm["spatial"][:, :2], dtype=np.float64) if "spatial" in synthetic.obsm else np.empty((0, 2))
    moran_r = _mean_morans_i(xr, coords_r)
    moran_s = _mean_morans_i(xs, coords_s)
    model, dataset, inferred_mode = _metadata_from_path(synthetic_path, synthetic)

    return FidelityResult(
        synthetic_path=str(synthetic_path),
        reference_path="slice_subset_from_all_slices_pool",
        model=model,
        dataset=dataset,
        mode=str(mode or inferred_mode),
        n_real=int(real_sampled.n_obs),
        n_synthetic=int(synthetic.n_obs),
        n_genes=int(real_sampled.n_vars),
        global_jsd=global_jsd,
        mean_gene_jsd=float(np.mean(gene_jsd)) if gene_jsd else np.nan,
        median_gene_jsd=float(np.median(gene_jsd)) if gene_jsd else np.nan,
        library_size_log_ratio=float(np.log1p(np.mean(lib_s)) - np.log1p(np.mean(lib_r))),
        library_size_wasserstein=float(wasserstein_distance(lib_r, lib_s)),
        fraction_zero_delta=float(np.mean(xs == 0) - np.mean(xr == 0)),
        mean_gene_mean_abs_diff=float(np.mean(np.abs(mean_s - mean_r))),
        mean_gene_var_abs_diff=float(np.mean(np.abs(var_s - var_r))),
        gene_mean_correlation=_safe_corr(mean_r, mean_s),
        gene_variance_correlation=_safe_corr(var_r, var_s),
        spot_total_correlation=spot_total_corr,
        mean_morans_i_real=moran_r,
        mean_morans_i_synthetic=moran_s,
        morans_i_abs_diff=float(abs(moran_s - moran_r)) if not (np.isnan(moran_s) or np.isnan(moran_r)) else np.nan,
    )


def evaluate_intrinsic_directory(
    synthetic_root: str | Path = "data/03_synthetic/cross_slice_generalization/DLPFC",
    output_path: str | Path = "data/05_results/summary/cross_slice_generalization/DLPFC/intrinsic_metrics.csv",
    config_path: str | Path = "configs/cross_slice_generalization/DLPFC/evaluation.yaml",
    split_tag: str = "all_slices_pool",
    mode: str = "supervised",
    sample_size: int | None = None,
    sample_frac: float | None = None,
    random_seed: int = 42,
    models: list[str] | None = None,
    slice_ids: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    synthetic_root = resolve_path(synthetic_root)
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intrinsic_defaults = load_intrinsic_defaults(config_path)
    sampling_defaults = intrinsic_defaults.get("sampling", {})
    if sample_size is None and sample_frac is None:
        sample_size = sampling_defaults.get("sample_size", 2000)

    model_filter = {str(m) for m in models} if models else None
    slice_filter = {str(s) for s in slice_ids} if slice_ids else None
    paths = discover_slice_files(synthetic_root, split_tag=split_tag, models=model_filter, slice_ids=slice_filter)
    if verbose:
        print(f"[cross-slice intrinsic] discovered {len(paths)} slice files under {synthetic_root}", flush=True)
        print(f"[cross-slice intrinsic] split_tag={split_tag} sample_size={sample_size} sample_frac={sample_frac}", flush=True)
        if model_filter:
            print(f"[cross-slice intrinsic] model_filter={sorted(model_filter)}", flush=True)
        if slice_filter:
            print(f"[cross-slice intrinsic] slice_filter={sorted(slice_filter)}", flush=True)

    rows: list[dict] = []
    reference_cache: dict[str, ad.AnnData] = {}

    def flush():
        pd.DataFrame(rows).to_csv(output_path, index=False)

    try:
        for i, synthetic_path in enumerate(paths, start=1):
            model = model_from_path(synthetic_path)
            slice_id = slice_id_from_path(synthetic_path)
            ref_path = reference_path_for(model, split_tag)
            row_base = {
                "synthetic_path": str(synthetic_path),
                "reference_path": str(ref_path),
                "model": model,
                "dataset": "DLPFC",
                "mode": mode,
                "split_tag": split_tag,
                "slice_id": slice_id,
            }
            if not ref_path.exists():
                rows.append({**row_base, "status": "skipped", "reason": "missing_reference"})
                flush()
                continue
            if model not in reference_cache:
                reference_cache[model] = ad.read_h5ad(ref_path, backed="r")
            try:
                real_slice = subset_reference_slice(reference_cache[model], slice_id)
                result = compute_fidelity_pair(
                    real=real_slice,
                    synthetic_path=synthetic_path,
                    config_path=config_path,
                    mode=mode,
                    sample_size=sample_size,
                    sample_frac=sample_frac,
                    random_seed=random_seed,
                )
                rows.append({**asdict(result), "split_tag": split_tag, "slice_id": slice_id, "status": "ok", "reason": ""})
                flush()
                if verbose:
                    print(f"[cross-slice intrinsic] {i}/{len(paths)} ok {model} slice={slice_id}", flush=True)
            except Exception as exc:
                rows.append({**row_base, "status": "failed", "reason": str(exc)})
                flush()
                if verbose:
                    print(f"[cross-slice intrinsic] {i}/{len(paths)} failed {model} slice={slice_id}: {exc}", flush=True)
    finally:
        for backed in reference_cache.values():
            backed.file.close()

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    if verbose:
        print(f"[cross-slice intrinsic] finished: {len(df)} rows -> {output_path}", flush=True)
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate intrinsic fidelity for DLPFC cross-slice synthetic pools.")
    parser.add_argument("--synthetic-root", default="data/03_synthetic/cross_slice_generalization/DLPFC")
    parser.add_argument("-o", "--output", default="data/05_results/summary/cross_slice_generalization/DLPFC/intrinsic_metrics.csv")
    parser.add_argument("-c", "--config", default="configs/cross_slice_generalization/DLPFC/evaluation.yaml")
    parser.add_argument("--split-tag", default="all_slices_pool")
    parser.add_argument("--mode", default="supervised")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--slice-ids", nargs="+", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    evaluate_intrinsic_directory(
        synthetic_root=args.synthetic_root,
        output_path=args.output,
        config_path=args.config,
        split_tag=args.split_tag,
        mode=args.mode,
        sample_size=args.sample_size,
        sample_frac=args.sample_frac,
        random_seed=args.random_seed,
        models=args.models,
        slice_ids=args.slice_ids,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
