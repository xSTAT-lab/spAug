"""Intrinsic-fidelity evaluation for Brain sample-level synthetic outputs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spaug_matplotlib"))


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
MODELS = ("SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion")


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class FidelityResult:
    synthetic_path: str
    reference_path: str
    model: str
    dataset: str
    task: str
    sample_id: str
    disease_label: str
    n_real_total: int
    n_real_sampled: int
    n_synthetic_total: int
    n_synthetic_sampled: int
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


def _sample_indices(n: int, sample_size: Optional[int], sample_frac: Optional[float], seed: int) -> np.ndarray:
    if sample_frac is not None:
        k = max(1, int(round(n * float(sample_frac))))
    elif sample_size is not None:
        k = min(n, int(sample_size))
    else:
        k = n
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False)) if k < n else np.arange(n)


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
    indices = NearestNeighbors(n_neighbors=n_neighbors).fit(coords[:, :2]).kneighbors(return_distance=False)[:, 1:]
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


def discover_synthetic_paths(root: str | Path, models: Optional[list[str]] = None) -> list[Path]:
    root = resolve_path(root)
    model_set = set(models or MODELS)
    paths = []
    for model in MODELS:
        if model not in model_set:
            continue
        paths.extend(sorted((root / model).glob("*.h5ad")))
    return sorted(paths)


def _single_sample_metadata(synthetic: ad.AnnData) -> tuple[str, str, str]:
    if "sample_id" not in synthetic.obs.columns:
        raise ValueError("Synthetic obs missing sample_id")
    if "disease_label" not in synthetic.obs.columns:
        raise ValueError("Synthetic obs missing disease_label")
    samples = synthetic.obs["sample_id"].astype(str).unique().tolist()
    if len(samples) != 1:
        raise ValueError(f"Synthetic file must contain one sample_id, got {samples}")
    labels = synthetic.obs["disease_label"].astype(str).unique().tolist()
    disease = labels[0] if len(labels) == 1 else ",".join(sorted(labels))
    source = str(synthetic.obs["source"].astype(str).unique()[0]) if "source" in synthetic.obs.columns else str(synthetic.uns.get("generator", "unknown"))
    return samples[0], disease, source


def _load_aligned_sample(
    reference_path: Path,
    synthetic_path: Path,
    sample_size: Optional[int],
    sample_frac: Optional[float],
    seed: int,
) -> tuple[ad.AnnData, ad.AnnData, str, str, str, int, int]:
    ref = ad.read_h5ad(reference_path, backed="r")
    syn = ad.read_h5ad(synthetic_path, backed="r")
    try:
        sample_id, disease_label, model = _single_sample_metadata(syn)
        if "sample_id" not in ref.obs.columns:
            raise ValueError("Reference obs missing sample_id")
        ref_mask = ref.obs["sample_id"].astype(str).to_numpy() == sample_id
        ref_indices_all = np.flatnonzero(ref_mask)
        if ref_indices_all.size == 0:
            raise ValueError(f"Reference contains no real spots for sample_id={sample_id}")
        common = ref.var_names.intersection(syn.var_names)
        if len(common) == 0:
            raise ValueError("No common genes between real and synthetic AnnData")
        ref_local = _sample_indices(ref_indices_all.size, sample_size, sample_frac, seed)
        syn_indices = _sample_indices(syn.n_obs, sample_size, sample_frac, seed + 997)
        real = ref[ref_indices_all[ref_local], :].to_memory()[:, common].copy()
        synthetic = syn[syn_indices, :].to_memory()[:, common].copy()
        return real, synthetic, sample_id, disease_label, model, int(ref_indices_all.size), int(syn.n_obs)
    finally:
        ref.file.close()
        syn.file.close()


def compute_fidelity_one(
    synthetic_path: str | Path,
    reference_path: str | Path,
    sample_size: Optional[int],
    sample_frac: Optional[float],
    random_seed: int,
) -> dict:
    synthetic_path = resolve_path(synthetic_path)
    reference_path = resolve_path(reference_path)
    real, synthetic, sample_id, disease_label, model, n_real_total, n_syn_total = _load_aligned_sample(
        reference_path=reference_path,
        synthetic_path=synthetic_path,
        sample_size=sample_size,
        sample_frac=sample_frac,
        seed=random_seed,
    )
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

    result = FidelityResult(
        synthetic_path=str(synthetic_path),
        reference_path=str(reference_path),
        model=model,
        dataset="Brain",
        task="sample_disease_prediction",
        sample_id=sample_id,
        disease_label=disease_label,
        n_real_total=n_real_total,
        n_real_sampled=int(real.n_obs),
        n_synthetic_total=n_syn_total,
        n_synthetic_sampled=int(synthetic.n_obs),
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
    return {**asdict(result), "status": "ok", "reason": ""}


def _worker(payload: tuple[str, str, Optional[int], Optional[float], int]) -> dict:
    synthetic_path, reference_path, sample_size, sample_frac, seed = payload
    try:
        return compute_fidelity_one(synthetic_path, reference_path, sample_size, sample_frac, seed)
    except Exception as exc:
        return {
            "synthetic_path": synthetic_path,
            "reference_path": reference_path,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def evaluate_intrinsic_directory(
    synthetic_root: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    models: Optional[list[str]] = None,
    sample_size: Optional[int] = None,
    sample_frac: Optional[float] = None,
    random_seed: int = 42,
    workers: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    synthetic_paths = discover_synthetic_paths(synthetic_root, models=models)
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ref = resolve_path(reference_path)
    if not ref.exists():
        raise FileNotFoundError(f"Missing Brain reference: {ref}")
    if verbose:
        print(f"[brain intrinsic] synthetic files: {len(synthetic_paths)}", flush=True)
        print(f"[brain intrinsic] reference: {ref}", flush=True)
        print(f"[brain intrinsic] output: {output_path}", flush=True)
        print(f"[brain intrinsic] sample_size={sample_size}, sample_frac={sample_frac}, workers={workers}", flush=True)

    payloads = [(str(p), str(ref), sample_size, sample_frac, random_seed + i) for i, p in enumerate(synthetic_paths)]
    rows: list[dict] = []
    if workers <= 1:
        for i, payload in enumerate(payloads, start=1):
            row = _worker(payload)
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_path, index=False)
            if verbose:
                print(f"[brain intrinsic] {i}/{len(payloads)} {row.get('status')}: {payload[0]}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_path = {pool.submit(_worker, payload): payload[0] for payload in payloads}
            for i, fut in enumerate(as_completed(future_to_path), start=1):
                row = fut.result()
                rows.append(row)
                pd.DataFrame(rows).to_csv(output_path, index=False)
                if verbose:
                    print(f"[brain intrinsic] {i}/{len(payloads)} {row.get('status')}: {future_to_path[fut]}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    manifest = {
        "synthetic_root": str(resolve_path(synthetic_root)),
        "reference_path": str(ref),
        "output_path": str(output_path),
        "n_files": int(len(synthetic_paths)),
        "n_rows": int(len(df)),
        "status_counts": df["status"].value_counts(dropna=False).to_dict() if "status" in df else {},
        "models": sorted(df["model"].dropna().astype(str).unique().tolist()) if "model" in df else [],
    }
    (output_path.parent / "intrinsic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if verbose:
        print(json.dumps(manifest, indent=2), flush=True)
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate intrinsic fidelity for Brain synthetic outputs.")
    parser.add_argument("--config", default="configs/sample_disease_prediction/Brain/evaluation.yaml")
    parser.add_argument("--synthetic-root", default=None)
    parser.add_argument("--reference", default=None)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    intrinsic = cfg.get("intrinsic", {})
    sampling = intrinsic.get("sampling", {})
    evaluate_intrinsic_directory(
        synthetic_root=args.synthetic_root or intrinsic.get("synthetic_root", "data/03_synthetic/sample_disease_prediction/Brain"),
        reference_path=args.reference or intrinsic.get("reference_path", "data/02_interim/sample_disease_prediction/Brain/brain_samples.h5ad"),
        output_path=args.output or intrinsic.get("output", "data/05_results/summary/sample_disease_prediction/Brain/intrinsic_metrics.csv"),
        models=args.models,
        sample_size=args.sample_size if args.sample_size is not None else sampling.get("sample_size", 2000),
        sample_frac=args.sample_frac,
        random_seed=args.random_seed,
        workers=args.workers if args.workers is not None else int(intrinsic.get("workers", 1)),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
