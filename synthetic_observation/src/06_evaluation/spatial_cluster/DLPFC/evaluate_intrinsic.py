"""Batch intrinsic-fidelity evaluation for generated AnnData files."""

from __future__ import annotations

import argparse
import yaml
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class FidelityResult:
    synthetic_path: str
    reference_path: str
    model: str
    dataset: str
    mode: str
    n_real: int
    n_synthetic: int
    n_genes: int
    global_jsd: float
    mean_gene_jsd: float
    median_gene_jsd: float
    library_size_log_ratio: float
    library_size_wasserstein: float
    fraction_zero_delta: float
    mean_gene_mean_abs_diff: float
    mean_gene_var_abs_diff: float
    gene_mean_correlation: float
    gene_variance_correlation: float
    spot_total_correlation: float
    mean_morans_i_real: float
    mean_morans_i_synthetic: float
    morans_i_abs_diff: float


def _dense(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    arr = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _sample_indices(adata: ad.AnnData, sample_size: Optional[int], sample_frac: Optional[float], seed: int) -> np.ndarray:
    n = adata.n_obs
    if sample_frac is not None:
        k = max(1, int(round(n * float(sample_frac))))
    elif sample_size is not None:
        k = min(n, int(sample_size))
    else:
        k = n
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False)) if k < n else np.arange(n)


def _align_and_sample(
    real: ad.AnnData,
    synthetic: ad.AnnData,
    sample_size: Optional[int],
    sample_frac: Optional[float],
    seed: int,
) -> tuple[ad.AnnData, ad.AnnData]:
    common = real.var_names.intersection(synthetic.var_names)
    if len(common) == 0:
        raise ValueError("No common genes between real and synthetic AnnData")
    real_idx = _sample_indices(real, sample_size, sample_frac, seed)
    synthetic_idx = _sample_indices(synthetic, sample_size, sample_frac, seed + 997)
    # Keep backed reads cheap: subset observations before materializing the
    # sampled matrix in memory.
    real = real[real_idx, :].to_memory()[:, common].copy()
    synthetic = synthetic[synthetic_idx, :].to_memory()[:, common].copy()
    return real, synthetic


def _hist_prob(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    prob = hist.astype(np.float64) + 1e-8
    return prob / prob.sum()


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _mean_morans_i(x: np.ndarray, coords: np.ndarray, n_top_genes: int = 100, k: int = 6) -> float:
    if x.shape[0] < 3 or coords.shape[0] != x.shape[0]:
        return np.nan
    n_neighbors = min(k + 1, x.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords[:, :2])
    indices = nn.kneighbors(return_distance=False)[:, 1:]
    variances = np.var(x, axis=0)
    gene_idx = np.argsort(variances)[-min(n_top_genes, x.shape[1]) :]
    values = []
    for j in gene_idx:
        v = x[:, j].astype(np.float64)
        centered = v - v.mean()
        denom = float(np.sum(centered**2))
        if denom <= 0:
            continue
        num = 0.0
        w_sum = 0
        for i in range(x.shape[0]):
            neigh = indices[i]
            num += float(np.sum(centered[i] * centered[neigh]))
            w_sum += len(neigh)
        if w_sum > 0:
            values.append((x.shape[0] / w_sum) * (num / denom))
    return float(np.mean(values)) if values else np.nan


def _metadata_from_path(path: Path, adata: ad.AnnData) -> tuple[str, str, str]:
    parts = path.parts
    model = str(adata.uns.get("generator", "") or next((m for m in ("SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion") if m in parts), "unknown"))
    dataset = str(adata.uns.get("dataset", "") or ("DLPFC" if "DLPFC" in parts else "unknown"))
    mode = str(adata.uns.get("generation_mode", "") or adata.uns.get("split_mode", "") or ("unsupervised" if "unsupervised" in parts else "supervised"))
    return model, dataset, mode


def compute_fidelity_many(
    real_path: str | Path,
    synthetic_path: str | Path,
    config_path: str | Path,
    mode: Optional[str] = None,
    sample_size: Optional[int] = None,
    sample_frac: Optional[float] = None,
    random_seed: int = 42,
    stratify_label: bool = False,
) -> list[FidelityResult]:
    del config_path, stratify_label
    real_path = resolve_path(real_path)
    synthetic_path = resolve_path(synthetic_path)
    real_backed = ad.read_h5ad(real_path, backed="r")
    synthetic_backed = ad.read_h5ad(synthetic_path, backed="r")
    try:
        real, synthetic = _align_and_sample(real_backed, synthetic_backed, sample_size, sample_frac, random_seed)
    finally:
        real_backed.file.close()
        synthetic_backed.file.close()
    xr = np.maximum(_dense(real.X), 0.0)
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

    coords_r = np.asarray(real.obsm["spatial"][:, :2], dtype=np.float64) if "spatial" in real.obsm else np.empty((0, 2))
    coords_s = np.asarray(synthetic.obsm["spatial"][:, :2], dtype=np.float64) if "spatial" in synthetic.obsm else np.empty((0, 2))
    moran_r = _mean_morans_i(xr, coords_r)
    moran_s = _mean_morans_i(xs, coords_s)
    model, dataset, inferred_mode = _metadata_from_path(synthetic_path, synthetic)

    return [
        FidelityResult(
            synthetic_path=str(synthetic_path),
            reference_path=str(real_path),
            model=model,
            dataset=dataset,
            mode=str(mode or inferred_mode),
            n_real=int(real.n_obs),
            n_synthetic=int(synthetic.n_obs),
            n_genes=int(real.n_vars),
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
    ]


def load_intrinsic_defaults(config_path: str | Path) -> dict:
    path = resolve_path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("intrinsic", {})


def infer_reference_path(synthetic_path: Path, mode: Optional[str] = None) -> Optional[Path]:
    parts = synthetic_path.parts
    dataset = "DLPFC" if "DLPFC" in parts else "Trastuzumab" if "Trastuzumab" in parts else None
    model = None
    for known in ("SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion"):
        if known in parts:
            model = known
            break
    if dataset is None or model is None:
        return None
    if "supervised_low_label" in parts:
        fraction = next((part for part in parts if part.startswith("fraction")), None)
        seed_part = next((part for part in parts if part.startswith("seed")), None)
        if fraction is None or seed_part is None:
            return None
        return (
            PROJECT_ROOT
            / "data"
            / "02_interim"
            / "supervised_low_label"
            / dataset
            / "model_inputs"
            / model
            / fraction
            / seed_part
            / "processed_with_split.h5ad"
        )
    if "cross_slice_generalization" in parts:
        split_name = next((part for part in parts if part.startswith("fixed_") or part.startswith("leave_")), None)
        if split_name is None:
            split_name = "fixed_8train_4test"
        return (
            PROJECT_ROOT
            / "data"
            / "02_interim"
            / "cross_slice_generalization"
            / dataset
            / "model_inputs"
            / model
            / split_name
            / "processed_with_split.h5ad"
        )
    if "spatial_cluster" in parts:
        return (
            PROJECT_ROOT
            / "data"
            / "02_interim"
            / "spatial_cluster"
            / dataset
            / "model_inputs"
            / model
            / "unsupervised"
            / "processed_with_split.h5ad"
        )
    if mode is None:
        if "spatial_cluster" in parts:
            mode = "unsupervised"
        elif "unsupervised" in parts:
            mode = "unsupervised"
        elif "supervised" in parts:
            mode = "supervised"
        else:
            mode = "supervised"
    return PROJECT_ROOT / "data" / "02_interim" / dataset / "model_inputs" / model / mode / "processed_with_split.h5ad"


def _is_dlpfc_slice_file(path: Path) -> bool:
    parts = path.parts
    if "DLPFC" not in parts or path.name != "synthetic.h5ad":
        return False
    if "pool_40x" not in parts:
        return False
    pool_idx = parts.index("pool_40x")
    rel_after_pool = parts[pool_idx + 1 :]
    return len(rel_after_pool) == 2


def discover_synthetic_paths(synthetic_root: Path) -> list[Path]:
    """Prefer small per-slice DLPFC files and pooled non-DLPFC files."""
    all_h5ad = sorted(synthetic_root.glob("**/*.h5ad"))
    pool_paths = [p for p in all_h5ad if p.name == "synthetic_pool_40x.h5ad"]
    selected: set[Path] = set()

    # Low-label and clustering DLPFC pools may be represented by per-slice
    # files to avoid repeatedly loading very large 40x pooled AnnData objects.
    for path in all_h5ad:
        if "DLPFC" not in path.parts:
            continue
        if path.name != "synthetic.h5ad":
            continue
        if "conditions" in path.parts:
            continue
        if "pool_40x" in path.parts and (
            path.parent.parent.name == "pool_40x" or path.parent.parent.name == "slices"
        ):
            selected.add(path)
    for path in all_h5ad:
        if "DLPFC" in path.parts and path.parent.name == "slices" and path.name.startswith("slice_"):
            selected.add(path)

    for pool in pool_paths:
        parts = pool.parts
        if "DLPFC" in parts and pool.parent.name == "pool_40x":
            slice_files = sorted(p for p in pool.parent.glob("*/synthetic.h5ad") if _is_dlpfc_slice_file(p))
            named_slice_files = sorted(pool.parent.glob("slices/slice_*.h5ad"))
            if slice_files or named_slice_files:
                selected.update(slice_files)
                selected.update(named_slice_files)
            else:
                # Conditional supervised generators keep per-label files under
                # pool_40x/conditions and a combined pool at pool_40x. The
                # combined pool is the right unit for intrinsic fidelity.
                selected.add(pool)
        else:
            selected.add(pool)

    if selected:
        return sorted(selected)
    return all_h5ad


def filter_synthetic_paths(
    paths: list[Path],
    dataset: Optional[str] = None,
    models: Optional[list[str]] = None,
) -> list[Path]:
    if dataset is None and not models:
        return paths
    model_set = set(models or [])
    out = []
    for path in paths:
        parts = path.parts
        if dataset is not None and dataset not in parts:
            continue
        if model_set:
            model = None
            for known in ("SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion"):
                if known in parts:
                    model = known
                    break
            if model not in model_set:
                continue
        out.append(path)
    return out


def evaluate_intrinsic_directory(
    synthetic_root: str | Path = "data/03_synthetic/spatial_cluster/DLPFC",
    output_path: str | Path = "data/05_results/summary/spatial_cluster/DLPFC/intrinsic_metrics.csv",
    config_path: str | Path = "configs/spatial_cluster/DLPFC/evaluation.yaml",
    reference_path: Optional[str | Path] = None,
    mode: Optional[str] = None,
    dataset: Optional[str] = None,
    models: Optional[list[str]] = None,
    sample_size: Optional[int] = None,
    sample_frac: Optional[float] = None,
    random_seed: int = 42,
    stratify_label: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    synthetic_root = resolve_path(synthetic_root)
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intrinsic_defaults = load_intrinsic_defaults(config_path)
    sampling_defaults = intrinsic_defaults.get("sampling", {})
    if sample_size is None and sample_frac is None:
        sample_size = sampling_defaults.get("sample_size", 2000)
    rows: list[dict] = []
    synthetic_paths = discover_synthetic_paths(synthetic_root)

    def flush():
        pd.DataFrame(rows).to_csv(output_path, index=False)

    if verbose:
        print(f"[intrinsic] discovered {len(synthetic_paths)} h5ad files under {synthetic_root}", flush=True)
    synthetic_paths = filter_synthetic_paths(synthetic_paths, dataset=dataset, models=models)

    if verbose:
        filters = []
        if dataset:
            filters.append(f"dataset={dataset}")
        if models:
            filters.append(f"models={','.join(models)}")
        if filters:
            print(f"[intrinsic] after filters ({'; '.join(filters)}): {len(synthetic_paths)} h5ad files", flush=True)
        if sample_size is not None:
            print(f"[intrinsic] sample_size={sample_size}", flush=True)
        if sample_frac is not None:
            print(f"[intrinsic] sample_frac={sample_frac}", flush=True)
        print(f"[intrinsic] output will be updated incrementally: {output_path}", flush=True)

    for i, synthetic_path in enumerate(synthetic_paths, start=1):
        ref = resolve_path(reference_path) if reference_path is not None else infer_reference_path(synthetic_path, mode=mode)
        row_base = {"synthetic_path": str(synthetic_path), "reference_path": str(ref) if ref else ""}
        if ref is None or not ref.exists():
            rows.append({**row_base, "status": "skipped", "reason": "missing_reference"})
            flush()
            if verbose:
                print(f"[intrinsic] {i}/{len(synthetic_paths)} skipped missing reference: {synthetic_path}", flush=True)
            continue
        try:
            if verbose:
                print(f"[intrinsic] {i}/{len(synthetic_paths)} evaluating: {synthetic_path}", flush=True)
            results = compute_fidelity_many(
                real_path=ref,
                synthetic_path=synthetic_path,
                config_path=config_path,
                mode=mode,
                sample_size=sample_size,
                sample_frac=sample_frac,
                random_seed=random_seed,
                stratify_label=stratify_label,
            )
            for result in results:
                rows.append({**asdict(result), "status": "ok", "reason": ""})
            flush()
            if verbose:
                print(f"[intrinsic] {i}/{len(synthetic_paths)} done: {synthetic_path} -> {len(results)} rows", flush=True)
        except Exception as exc:
            rows.append({**row_base, "status": "failed", "reason": str(exc)})
            flush()
            if verbose:
                print(f"[intrinsic] {i}/{len(synthetic_paths)} failed: {synthetic_path}: {exc}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    if verbose:
        print(f"[intrinsic] finished: {len(df)} rows written to {output_path}", flush=True)
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate intrinsic fidelity for synthetic outputs.")
    parser.add_argument("--synthetic-root", default="data/03_synthetic/spatial_cluster/DLPFC")
    parser.add_argument("-o", "--output", default="data/05_results/summary/spatial_cluster/DLPFC/intrinsic_metrics.csv")
    parser.add_argument("-c", "--config", default="configs/spatial_cluster/DLPFC/evaluation.yaml")
    parser.add_argument("--reference", default=None, help="Optional single reference AnnData")
    parser.add_argument("--mode", default=None, choices=[None, "supervised", "unsupervised"])
    parser.add_argument("--dataset", default=None, choices=[None, "DLPFC"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--stratify-label", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging")
    return parser


def main():
    args = build_parser().parse_args()
    evaluate_intrinsic_directory(
        synthetic_root=args.synthetic_root,
        output_path=args.output,
        config_path=args.config,
        reference_path=args.reference,
        mode=args.mode,
        dataset=args.dataset,
        models=args.models,
        sample_size=args.sample_size,
        sample_frac=args.sample_frac,
        random_seed=args.random_seed,
        stratify_label=args.stratify_label,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
