"""Splatter/Splat faithful Python reimplementation core.

SpatialSimBench applies Splatter to spatial data through a simAdaptor pattern:
spots are ordered/split by ``spatial.cluster``, ``splatter::splatEstimate`` is
called inside each region, and ``splatter::splatSimulate`` produces expression
for the same number of cells before the original spatial metadata is restored.

This module implements the Splat single-population parameter chain in Python:

``splatEstimate.matrix``:
    library sizes -> normalized counts -> gamma gene means -> expression
    outliers -> BCV -> dropout logistic -> ``nGenes`` and ``batchCells``.

``splatSimulate(method="single")``:
    library sizes -> gene means/outliers -> batch/cell means -> BCV gamma means
    -> Poisson true counts -> dropout.

The implementation is a Python implementation following the published
package call. Provenance should be ``generator_backend=python_reimplementation``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import optimize, sparse, stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _to_dense_count(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    arr = np.asarray(x, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(arr, 0.0)


def _winsorize(x: np.ndarray, q: float = 0.1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo, hi = np.quantile(x, [q, 1.0 - q])
    return np.clip(x, lo, hi)


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) * 1.4826)


def _gamma_moments(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return 0.6, 0.3
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if x.size > 1 else mean
    var = max(var, 1e-8)
    shape = max(mean * mean / var, 1e-6)
    rate = max(mean / var, 1e-6)
    return shape, rate


def _fit_lognormal_params(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return 0.0, 0.2
    lx = np.log(x)
    sd = float(np.std(lx, ddof=1)) if lx.size > 1 else 1e-6
    return float(np.mean(lx)), max(sd, 1e-6)


def _logistic(x: np.ndarray, x0: float, k: float) -> np.ndarray:
    z = np.clip(-k * (x - x0), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))


def _fit_dropout(log_means: np.ndarray, zero_fraction: np.ndarray) -> tuple[str, float, float]:
    valid = np.isfinite(log_means) & np.isfinite(zero_fraction)
    x = np.asarray(log_means[valid], dtype=np.float64)
    y = np.asarray(zero_fraction[valid], dtype=np.float64)
    if x.size < 4 or np.nanmax(y) <= 0:
        return "none", 0.0, -1.0
    mid_candidates = x[(y > 0.2) & (y < 0.8)]
    x0 = float(np.median(mid_candidates)) if mid_candidates.size else float(np.median(x))
    try:
        popt, _ = optimize.curve_fit(
            lambda xx, x0_, k_: _logistic(xx, x0_, k_),
            x,
            np.clip(y, 1e-6, 1.0 - 1e-6),
            p0=np.array([x0, -1.0]),
            maxfev=5000,
        )
        x0_fit = float(popt[0])
        k_fit = float(popt[1])
        if not np.isfinite(x0_fit) or not np.isfinite(k_fit):
            raise RuntimeError("non-finite dropout fit")
        return "experiment", x0_fit, k_fit
    except Exception:
        # Splatter tries several nls algorithms; if all fail, keep dropout off.
        return "none", 0.0, -1.0


def _estimate_common_dispersion(x: np.ndarray) -> float:
    means = x.mean(axis=0)
    vars_ = x.var(axis=0, ddof=1) if x.shape[0] > 1 else means
    valid = means > 0
    if not np.any(valid):
        return 0.0
    disp = (vars_[valid] - means[valid]) / np.maximum(means[valid] ** 2, 1e-8)
    disp = disp[np.isfinite(disp) & (disp > 0)]
    if disp.size == 0:
        return 0.0
    return float(np.median(disp))


@dataclass
class SplatParamsPython:
    n_genes: int
    batch_cells: int
    seed: int = 105122
    mean_shape: float = 0.6
    mean_rate: float = 0.3
    lib_loc: float = 11.0
    lib_scale: float = 0.2
    lib_norm: bool = False
    out_prob: float = 0.05
    out_fac_loc: float = 4.0
    out_fac_scale: float = 0.5
    bcv_common: float = 0.1
    bcv_df: float = 60.0
    dropout_type: str = "none"
    dropout_mid: float = 0.0
    dropout_shape: float = -1.0
    coord_mean: np.ndarray | None = None
    coord_sd: np.ndarray | None = None
    n_spots: int = 0
    region_id: str = "global"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if self.coord_mean is not None:
            out["coord_mean"] = np.asarray(self.coord_mean, dtype=np.float64).tolist()
        if self.coord_sd is not None:
            out["coord_sd"] = np.asarray(self.coord_sd, dtype=np.float64).tolist()
        return out

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "SplatParamsPython":
        state = dict(state)
        if state.get("coord_mean") is not None:
            state["coord_mean"] = np.asarray(state["coord_mean"], dtype=np.float64)
        if state.get("coord_sd") is not None:
            state["coord_sd"] = np.asarray(state["coord_sd"], dtype=np.float64)
        return cls(**state)


class SplatSinglePython:
    """Single-population Splat estimator/simulator."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = int(random_seed)
        self.params: SplatParamsPython | None = None

    def fit(self, x_spot_by_gene, coords: np.ndarray | None = None, region_id: str = "global") -> "SplatSinglePython":
        x = _to_dense_count(x_spot_by_gene)
        n_spots, n_genes = x.shape
        lib_sizes = np.maximum(x.sum(axis=1), 1.0)
        lib_med = float(np.median(lib_sizes))
        norm = x / lib_sizes[:, None] * max(lib_med, 1.0)
        gene_has_signal = (norm > 0).sum(axis=0) > 1
        norm_fit = norm[:, gene_has_signal] if np.any(gene_has_signal) else norm

        means = norm_fit.mean(axis=0)
        means_nonzero = means[means > 0]
        mean_shape, mean_rate = _gamma_moments(_winsorize(means_nonzero, q=0.1))

        if lib_sizes.size >= 3:
            sample = lib_sizes if lib_sizes.size <= 5000 else np.random.default_rng(self.random_seed).choice(lib_sizes, 5000, replace=False)
            try:
                lib_norm = bool(stats.shapiro(sample).pvalue > 0.2)
            except Exception:
                lib_norm = False
        else:
            lib_norm = False
        if lib_norm:
            lib_loc = float(np.mean(lib_sizes))
            lib_scale = float(max(np.std(lib_sizes, ddof=1) if lib_sizes.size > 1 else 1e-6, 1e-6))
        else:
            lib_loc, lib_scale = _fit_lognormal_params(lib_sizes)

        lmeans = np.log(np.maximum(means, 1e-12))
        med = float(np.median(lmeans)) if lmeans.size else 0.0
        mad = _mad(lmeans) if lmeans.size else 0.0
        out_mask = lmeans > (med + 2.0 * mad)
        out_prob = float(out_mask.sum() / max(norm_fit.shape[1], 1))
        out_fac_loc, out_fac_scale = 4.0, 0.5
        if out_mask.sum() > 1 and np.median(means) > 0:
            out_fac_loc, out_fac_scale = _fit_lognormal_params(means[out_mask] / max(float(np.median(means)), 1e-12))

        common_disp = _estimate_common_dispersion(x)
        bcv_common = float(0.1 + 0.25 * common_disp)
        bcv_df = 60.0

        obs_zero = (norm_fit == 0).sum(axis=0) / max(norm_fit.shape[0], 1)
        dropout_type, dropout_mid, dropout_shape = _fit_dropout(np.log(np.maximum(means, 1e-12)), obs_zero)

        if coords is None or len(coords) == 0:
            coord_mean = np.zeros(2, dtype=np.float64)
            coord_sd = np.ones(2, dtype=np.float64)
        else:
            c = np.asarray(coords[:, :2], dtype=np.float64)
            coord_mean = c.mean(axis=0)
            coord_sd = np.maximum(c.std(axis=0), 1.0)

        self.params = SplatParamsPython(
            n_genes=int(n_genes),
            batch_cells=int(n_spots),
            seed=int(self.random_seed),
            mean_shape=float(mean_shape),
            mean_rate=float(mean_rate),
            lib_loc=float(lib_loc),
            lib_scale=float(lib_scale),
            lib_norm=bool(lib_norm),
            out_prob=float(np.clip(out_prob, 0.0, 1.0)),
            out_fac_loc=float(out_fac_loc),
            out_fac_scale=float(out_fac_scale),
            bcv_common=max(float(bcv_common), 1e-8),
            bcv_df=bcv_df,
            dropout_type=dropout_type,
            dropout_mid=float(dropout_mid),
            dropout_shape=float(dropout_shape),
            coord_mean=coord_mean,
            coord_sd=coord_sd,
            n_spots=int(n_spots),
            region_id=str(region_id),
        )
        return self

    @staticmethod
    def _lognormal_factors(n: int, prob: float, loc: float, scale: float, rng: np.random.Generator) -> np.ndarray:
        keep = rng.random(n) < float(prob)
        factors = np.ones(n, dtype=np.float64)
        if np.any(keep):
            factors[keep] = rng.lognormal(mean=float(loc), sigma=max(float(scale), 1e-8), size=int(keep.sum()))
        return factors

    def simulate_counts(self, n_cells: int, random_seed: int | None = None) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("SplatSinglePython must be fitted before simulation")
        p = self.params
        rng = np.random.default_rng(p.seed if random_seed is None else int(random_seed))
        n_cells = int(n_cells)

        if p.lib_norm:
            exp_lib = rng.normal(p.lib_loc, max(p.lib_scale, 1e-8), size=n_cells)
            positive = exp_lib[exp_lib > 0]
            min_lib = float(positive.min()) if positive.size else 1.0
            exp_lib[exp_lib < 0] = min_lib / 2.0
        else:
            exp_lib = rng.lognormal(mean=p.lib_loc, sigma=max(p.lib_scale, 1e-8), size=n_cells)
        exp_lib = np.maximum(exp_lib, 1.0)

        base_means = rng.gamma(shape=max(p.mean_shape, 1e-8), scale=1.0 / max(p.mean_rate, 1e-8), size=p.n_genes)
        outlier_factors = self._lognormal_factors(p.n_genes, p.out_prob, p.out_fac_loc, p.out_fac_scale, rng)
        median_base = float(np.median(base_means)) if base_means.size else 0.0
        gene_means = base_means.copy()
        is_outlier = outlier_factors != 1.0
        gene_means[is_outlier] = median_base * outlier_factors[is_outlier]

        batch_cell_means = np.repeat(gene_means[:, None], n_cells, axis=1)
        denom = np.maximum(batch_cell_means.sum(axis=0), 1e-12)
        cell_props = batch_cell_means / denom[None, :]
        base_cell_means = cell_props * exp_lib[None, :]

        if np.isfinite(p.bcv_df) and p.bcv_df > 0:
            chi = rng.chisquare(df=p.bcv_df, size=(p.n_genes, n_cells))
            bcv = (p.bcv_common + 1.0 / np.sqrt(np.maximum(base_cell_means, 1e-8))) * np.sqrt(p.bcv_df / np.maximum(chi, 1e-8))
        else:
            bcv = p.bcv_common + 1.0 / np.sqrt(np.maximum(base_cell_means, 1e-8))
        shape = 1.0 / np.maximum(bcv * bcv, 1e-8)
        scale = base_cell_means * np.maximum(bcv * bcv, 1e-8)
        cell_means = rng.gamma(shape=shape, scale=np.maximum(scale, 1e-12))
        true_counts = rng.poisson(np.maximum(cell_means, 0.0))

        if p.dropout_type != "none":
            eta = np.log(np.maximum(cell_means, 1e-12))
            drop_prob = _logistic(eta, p.dropout_mid, p.dropout_shape)
            keep = rng.binomial(1, 1.0 - np.clip(drop_prob, 0.0, 1.0))
            true_counts = true_counts * keep
        return np.maximum(true_counts.T, 0).astype(np.float32)


class SpatialSplatterPython:
    """SpatialSimBench-style region-wise Splat wrapper."""

    def __init__(
        self,
        n_regions: int = 7,
        min_region_size: int = 20,
        coord_sigma_factor: float = 0.15,
        random_seed: int = 42,
    ):
        self.n_regions = int(n_regions)
        self.min_region_size = int(min_region_size)
        self.coord_sigma_factor = float(coord_sigma_factor)
        self.random_seed = int(random_seed)
        self.gene_names: list[str] = []
        self.region_labels: np.ndarray | None = None
        self.region_models: list[SplatSinglePython] = []
        self.region_probs: np.ndarray | None = None
        self.region_coords: dict[str, np.ndarray] = {}

    def assign_regions(self, coords: np.ndarray | None, n_obs: int) -> np.ndarray:
        if coords is None:
            return np.zeros(n_obs, dtype=int)
        coords = np.asarray(coords, dtype=np.float64)
        if coords.shape[0] != n_obs or coords.shape[1] < 2:
            return np.zeros(n_obs, dtype=int)
        n_clusters = min(self.n_regions, max(1, n_obs // max(1, self.min_region_size)))
        if n_clusters <= 1:
            return np.zeros(n_obs, dtype=int)
        scaled = StandardScaler().fit_transform(coords[:, :2])
        labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.random_seed).fit_predict(scaled)
        counts = np.bincount(labels, minlength=n_clusters)
        small = np.where(counts < self.min_region_size)[0]
        large = np.where(counts >= self.min_region_size)[0]
        if small.size == 0:
            return labels.astype(int)
        if large.size == 0:
            return np.zeros(n_obs, dtype=int)
        centroids = np.vstack([scaled[labels == k].mean(axis=0) for k in range(n_clusters)])
        for k in small:
            idx = np.where(labels == k)[0]
            d = ((scaled[idx, None, :] - centroids[large][None, :, :]) ** 2).sum(axis=2)
            labels[idx] = large[np.argmin(d, axis=1)]
        _, compact = np.unique(labels, return_inverse=True)
        return compact.astype(int)

    def fit(self, x, gene_names: list[str], coords: np.ndarray | None = None) -> "SpatialSplatterPython":
        x = _to_dense_count(x)
        self.gene_names = list(gene_names)
        labels = self.assign_regions(coords, x.shape[0])
        self.region_labels = labels
        self.region_models = []
        self.region_coords = {}
        counts = []
        for rid in np.unique(labels):
            mask = labels == rid
            if int(mask.sum()) < self.min_region_size:
                continue
            region_id = str(int(rid))
            region_coords = None if coords is None else np.asarray(coords[mask, :2], dtype=np.float64)
            model = SplatSinglePython(random_seed=self.random_seed + int(rid)).fit(x[mask], coords=region_coords, region_id=region_id)
            self.region_models.append(model)
            self.region_coords[region_id] = np.empty((0, 2), dtype=np.float64) if region_coords is None else region_coords
            counts.append(int(mask.sum()))
        if not self.region_models:
            model = SplatSinglePython(random_seed=self.random_seed).fit(x, coords=coords, region_id="global")
            self.region_models = [model]
            self.region_coords = {"global": np.empty((0, 2), dtype=np.float64) if coords is None else np.asarray(coords[:, :2], dtype=np.float64)}
            counts = [int(x.shape[0])]
        self.region_probs = np.asarray(counts, dtype=np.float64)
        self.region_probs = self.region_probs / self.region_probs.sum()
        return self

    def _sample_coords(self, region_id: str, params: SplatParamsPython, n: int, rng: np.random.Generator) -> np.ndarray:
        ref = self.region_coords.get(region_id)
        if ref is not None and ref.shape[0] > 0:
            base = ref[rng.integers(0, ref.shape[0], size=n)].astype(np.float64, copy=True)
        else:
            base = np.repeat(np.asarray(params.coord_mean, dtype=np.float64)[None, :], n, axis=0)
        coord_sd = np.asarray(params.coord_sd, dtype=np.float64) if params.coord_sd is not None else np.ones(2)
        return base + rng.normal(0.0, np.maximum(coord_sd * self.coord_sigma_factor, 1e-6), size=(n, 2))

    def generate(self, n_samples: int, random_seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.region_models or self.region_probs is None:
            raise RuntimeError("SpatialSplatterPython must be fitted before generation")
        rng = np.random.default_rng(self.random_seed if random_seed is None else int(random_seed))
        n_samples = int(n_samples)
        picks = rng.choice(len(self.region_models), size=n_samples, p=self.region_probs)
        x = np.zeros((n_samples, len(self.gene_names)), dtype=np.float32)
        coords = np.zeros((n_samples, 2), dtype=np.float64)
        regions = np.empty(n_samples, dtype=object)
        for i, model in enumerate(self.region_models):
            loc = np.where(picks == i)[0]
            if loc.size == 0:
                continue
            params = model.params
            if params is None:
                continue
            x[loc] = model.simulate_counts(int(loc.size), random_seed=int(rng.integers(0, 2**31 - 1)))
            coords[loc] = self._sample_coords(params.region_id, params, int(loc.size), rng)
            regions[loc] = params.region_id
        return x, coords, regions.astype(str)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_regions": self.n_regions,
            "min_region_size": self.min_region_size,
            "coord_sigma_factor": self.coord_sigma_factor,
            "random_seed": self.random_seed,
            "gene_names": self.gene_names,
            "region_labels": None if self.region_labels is None else self.region_labels.tolist(),
            "region_probs": None if self.region_probs is None else self.region_probs.tolist(),
            "region_params": [m.params.to_dict() for m in self.region_models if m.params is not None],
            "region_coords": {k: v.tolist() for k, v in self.region_coords.items()},
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "SpatialSplatterPython":
        model = cls(
            n_regions=state.get("n_regions", 7),
            min_region_size=state.get("min_region_size", 20),
            coord_sigma_factor=state.get("coord_sigma_factor", 0.15),
            random_seed=state.get("random_seed", 42),
        )
        model.gene_names = list(state.get("gene_names", []))
        labels = state.get("region_labels")
        model.region_labels = None if labels is None else np.asarray(labels, dtype=int)
        probs = state.get("region_probs")
        model.region_probs = None if probs is None else np.asarray(probs, dtype=np.float64)
        model.region_coords = {str(k): np.asarray(v, dtype=np.float64) for k, v in state.get("region_coords", {}).items()}
        model.region_models = []
        for p_state in state.get("region_params", []):
            sm = SplatSinglePython(random_seed=model.random_seed)
            sm.params = SplatParamsPython.from_dict(p_state)
            model.region_models.append(sm)
        return model

