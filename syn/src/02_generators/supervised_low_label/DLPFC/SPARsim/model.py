"""Project-local region-wise SPARsim approximation and checkpoint utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _to_dense_count(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(arr, 0.0)


def _safe_var(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.maximum(x.mean(axis=0), 1e-8)
    return np.maximum(x.var(axis=0, ddof=1), 1e-8)


def _size_factors(x: np.ndarray) -> np.ndarray:
    library = np.maximum(x.sum(axis=1), 1.0)
    median = float(np.median(library))
    if not np.isfinite(median) or median <= 0:
        median = 1.0
    return np.maximum(library / median, 1e-6)


@dataclass
class SPARsimRegionParams:
    region_id: str
    n_spots: int
    norm_mean: np.ndarray
    theta: np.ndarray
    dropout_prob: np.ndarray
    log_size_factor_mean: float
    log_size_factor_sd: float
    coord_mean: np.ndarray
    coord_sd: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("norm_mean", "theta", "dropout_prob", "coord_mean", "coord_sd"):
            out[key] = np.asarray(out[key]).tolist()
        return out

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "SPARsimRegionParams":
        return cls(
            region_id=str(state["region_id"]),
            n_spots=int(state["n_spots"]),
            norm_mean=np.asarray(state["norm_mean"], dtype=np.float64),
            theta=np.asarray(state["theta"], dtype=np.float64),
            dropout_prob=np.asarray(state["dropout_prob"], dtype=np.float64),
            log_size_factor_mean=float(state["log_size_factor_mean"]),
            log_size_factor_sd=float(state["log_size_factor_sd"]),
            coord_mean=np.asarray(state["coord_mean"], dtype=np.float64),
            coord_sd=np.asarray(state["coord_sd"], dtype=np.float64),
        )


class SPARsimModel:
    """Region-wise SPARsim approximation using normalized NB marginals."""

    def __init__(
        self,
        n_regions: int = 7,
        min_region_size: int = 20,
        dispersion_min: float = 0.01,
        dispersion_max: float = 1_000_000.0,
        dropout_quantile: float = 0.99,
        coord_sigma_factor: float = 0.12,
        random_seed: int = 42,
    ):
        self.n_regions = int(n_regions)
        self.min_region_size = int(min_region_size)
        self.dispersion_min = float(dispersion_min)
        self.dispersion_max = float(dispersion_max)
        self.dropout_quantile = float(dropout_quantile)
        self.coord_sigma_factor = float(coord_sigma_factor)
        self.random_seed = int(random_seed)
        self.gene_names: list[str] = []
        self.region_labels: np.ndarray | None = None
        self.region_params: list[SPARsimRegionParams] = []
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

    def _fit_region(self, region_id: str, x: np.ndarray, coords: np.ndarray | None) -> SPARsimRegionParams:
        sf = _size_factors(x)
        norm = x / sf[:, None]
        norm_mean = np.maximum(norm.mean(axis=0), 1e-8)
        norm_var = _safe_var(norm)
        overdisp = np.maximum(norm_var - norm_mean, 1e-8)
        theta = np.clip((norm_mean * norm_mean) / overdisp, self.dispersion_min, self.dispersion_max)

        expected_zero = np.power(theta / (theta + norm_mean), theta)
        observed_zero = (x <= 0).mean(axis=0)
        dropout_prob = (observed_zero - expected_zero) / np.maximum(1.0 - expected_zero, 1e-8)
        dropout_prob = np.clip(dropout_prob, 0.0, self.dropout_quantile)

        log_sf = np.log(sf)
        log_sf_sd = float(log_sf.std(ddof=1)) if x.shape[0] > 1 else 0.0
        if coords is None or coords.shape[0] == 0:
            coord_mean = np.zeros(2, dtype=np.float64)
            coord_sd = np.ones(2, dtype=np.float64)
        else:
            c = np.asarray(coords[:, :2], dtype=np.float64)
            coord_mean = c.mean(axis=0)
            coord_sd = np.maximum(c.std(axis=0), 1.0)

        return SPARsimRegionParams(
            region_id=str(region_id),
            n_spots=int(x.shape[0]),
            norm_mean=norm_mean,
            theta=theta,
            dropout_prob=dropout_prob,
            log_size_factor_mean=float(log_sf.mean()),
            log_size_factor_sd=float(max(log_sf_sd, 1e-6)),
            coord_mean=coord_mean,
            coord_sd=coord_sd,
        )

    def fit(self, x, gene_names: list[str], coords: np.ndarray | None = None) -> "SPARsimModel":
        x = _to_dense_count(x)
        self.gene_names = list(gene_names)
        labels = self.assign_regions(coords, x.shape[0])
        self.region_labels = labels
        self.region_params = []
        self.region_coords = {}
        counts = []

        for rid in np.unique(labels):
            mask = labels == rid
            if int(mask.sum()) < self.min_region_size:
                continue
            region_id = str(int(rid))
            region_coords = None if coords is None else np.asarray(coords[mask, :2], dtype=np.float64)
            self.region_params.append(self._fit_region(region_id, x[mask], region_coords))
            self.region_coords[region_id] = np.empty((0, 2), dtype=np.float64) if region_coords is None else region_coords
            counts.append(int(mask.sum()))

        if not self.region_params:
            self.region_params = [self._fit_region("global", x, coords)]
            self.region_coords = {"global": np.empty((0, 2), dtype=np.float64) if coords is None else np.asarray(coords[:, :2], dtype=np.float64)}
            counts = [int(x.shape[0])]

        self.region_probs = np.asarray(counts, dtype=np.float64)
        self.region_probs = self.region_probs / self.region_probs.sum()
        return self

    def _sample_counts(self, params: SPARsimRegionParams, n: int, rng: np.random.Generator) -> np.ndarray:
        size_factor = np.exp(rng.normal(params.log_size_factor_mean, params.log_size_factor_sd, size=n))
        size_factor = np.maximum(size_factor, 1e-6)
        mu = np.maximum(params.norm_mean[None, :] * size_factor[:, None], 1e-8)
        theta = np.maximum(params.theta, self.dispersion_min)
        p = theta[None, :] / (theta[None, :] + mu)
        out = rng.negative_binomial(theta[None, :], p).astype(np.float64)
        dropout = rng.random(out.shape) < params.dropout_prob[None, :]
        out[dropout] = 0.0
        return np.maximum(out, 0).astype(np.float32)

    def _sample_coords(self, params: SPARsimRegionParams, n: int, rng: np.random.Generator) -> np.ndarray:
        ref = self.region_coords.get(params.region_id)
        if ref is not None and ref.shape[0] > 0:
            base = ref[rng.integers(0, ref.shape[0], size=n)].astype(np.float64, copy=True)
        else:
            base = np.repeat(params.coord_mean[None, :], n, axis=0)
        noise_sd = np.maximum(params.coord_sd * self.coord_sigma_factor, 1e-6)
        return base + rng.normal(0.0, noise_sd, size=(n, 2))

    def generate(self, n_samples: int, random_seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.region_params or self.region_probs is None:
            raise RuntimeError("SPARsimModel must be fitted before generation")
        rng = np.random.default_rng(self.random_seed if random_seed is None else int(random_seed))
        n_samples = int(n_samples)
        region_idx = rng.choice(len(self.region_params), size=n_samples, p=self.region_probs)
        x = np.zeros((n_samples, len(self.gene_names)), dtype=np.float32)
        coords = np.zeros((n_samples, 2), dtype=np.float64)
        regions = np.empty(n_samples, dtype=object)
        for i, params in enumerate(self.region_params):
            loc = np.where(region_idx == i)[0]
            if loc.size == 0:
                continue
            x[loc] = self._sample_counts(params, int(loc.size), rng)
            coords[loc] = self._sample_coords(params, int(loc.size), rng)
            regions[loc] = params.region_id
        return x, coords, regions.astype(str)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_regions": self.n_regions,
            "min_region_size": self.min_region_size,
            "dispersion_min": self.dispersion_min,
            "dispersion_max": self.dispersion_max,
            "dropout_quantile": self.dropout_quantile,
            "coord_sigma_factor": self.coord_sigma_factor,
            "random_seed": self.random_seed,
            "gene_names": self.gene_names,
            "region_labels": None if self.region_labels is None else self.region_labels.tolist(),
            "region_params": [p.to_dict() for p in self.region_params],
            "region_probs": None if self.region_probs is None else self.region_probs.tolist(),
            "region_coords": {k: v.tolist() for k, v in self.region_coords.items()},
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "SPARsimModel":
        model = cls(
            n_regions=state.get("n_regions", 7),
            min_region_size=state.get("min_region_size", 20),
            dispersion_min=state.get("dispersion_min", 0.01),
            dispersion_max=state.get("dispersion_max", 1_000_000.0),
            dropout_quantile=state.get("dropout_quantile", 0.99),
            coord_sigma_factor=state.get("coord_sigma_factor", 0.12),
            random_seed=state.get("random_seed", 42),
        )
        model.gene_names = list(state.get("gene_names", []))
        labels = state.get("region_labels")
        model.region_labels = None if labels is None else np.asarray(labels, dtype=int)
        model.region_params = [SPARsimRegionParams.from_dict(p) for p in state.get("region_params", [])]
        probs = state.get("region_probs")
        model.region_probs = None if probs is None else np.asarray(probs, dtype=np.float64)
        model.region_coords = {str(k): np.asarray(v, dtype=np.float64) for k, v in state.get("region_coords", {}).items()}
        return model

