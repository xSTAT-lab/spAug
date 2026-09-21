"""Project-local region-wise Splatter approximation and checkpoint utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _to_dense_float(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(arr, 0.0)


def _safe_var(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.maximum(x.mean(axis=0), 1e-8)
    return np.maximum(x.var(axis=0, ddof=1), 1e-8)


@dataclass
class RegionParams:
    region_id: str
    n_spots: int
    mean: np.ndarray
    theta: np.ndarray
    zero_rate: np.ndarray
    log_library_mean: float
    log_library_sd: float
    coord_mean: np.ndarray
    coord_sd: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("mean", "theta", "zero_rate", "coord_mean", "coord_sd"):
            out[key] = np.asarray(out[key]).tolist()
        return out

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "RegionParams":
        return cls(
            region_id=str(state["region_id"]),
            n_spots=int(state["n_spots"]),
            mean=np.asarray(state["mean"], dtype=np.float64),
            theta=np.asarray(state["theta"], dtype=np.float64),
            zero_rate=np.asarray(state["zero_rate"], dtype=np.float64),
            log_library_mean=float(state["log_library_mean"]),
            log_library_sd=float(state["log_library_sd"]),
            coord_mean=np.asarray(state["coord_mean"], dtype=np.float64),
            coord_sd=np.asarray(state["coord_sd"], dtype=np.float64),
        )


class SplatterModel:
    """Region-wise count simulator with Splatter-like NB marginals."""

    def __init__(
        self,
        n_regions: int = 7,
        min_region_size: int = 20,
        dispersion_min: float = 0.01,
        dispersion_max: float = 1_000_000.0,
        dropout_match: bool = True,
        library_size_match: bool = True,
        coord_sigma_factor: float = 0.15,
        random_seed: int = 42,
    ):
        self.n_regions = int(n_regions)
        self.min_region_size = int(min_region_size)
        self.dispersion_min = float(dispersion_min)
        self.dispersion_max = float(dispersion_max)
        self.dropout_match = bool(dropout_match)
        self.library_size_match = bool(library_size_match)
        self.coord_sigma_factor = float(coord_sigma_factor)
        self.random_seed = int(random_seed)
        self.gene_names: list[str] = []
        self.region_labels: np.ndarray | None = None
        self.region_params: list[RegionParams] = []
        self.region_probs: np.ndarray | None = None
        self.region_coords: dict[str, np.ndarray] = {}
        self.global_params: RegionParams | None = None

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
        if small.size == 0:
            return labels.astype(int)
        large = np.where(counts >= self.min_region_size)[0]
        if large.size == 0:
            return np.zeros(n_obs, dtype=int)
        centroids = np.vstack([scaled[labels == k].mean(axis=0) for k in range(n_clusters)])
        for k in small:
            idx = np.where(labels == k)[0]
            d = ((scaled[idx, None, :] - centroids[large][None, :, :]) ** 2).sum(axis=2)
            labels[idx] = large[np.argmin(d, axis=1)]
        _, compact = np.unique(labels, return_inverse=True)
        return compact.astype(int)

    def _fit_one_region(self, region_id: str, x: np.ndarray, coords: np.ndarray | None) -> RegionParams:
        mean = np.maximum(x.mean(axis=0), 1e-8)
        var = _safe_var(x)
        overdisp = np.maximum(var - mean, 1e-8)
        theta = np.clip((mean * mean) / overdisp, self.dispersion_min, self.dispersion_max)
        zero_rate = np.clip((x <= 0).mean(axis=0), 0.0, 0.995)
        library = np.maximum(x.sum(axis=1), 1.0)
        log_library = np.log1p(library)
        if coords is None or coords.shape[0] == 0:
            coord_mean = np.zeros(2, dtype=np.float64)
            coord_sd = np.ones(2, dtype=np.float64)
        else:
            c = np.asarray(coords[:, :2], dtype=np.float64)
            coord_mean = c.mean(axis=0)
            coord_sd = np.maximum(c.std(axis=0), 1.0)
        return RegionParams(
            region_id=str(region_id),
            n_spots=int(x.shape[0]),
            mean=mean,
            theta=theta,
            zero_rate=zero_rate,
            log_library_mean=float(log_library.mean()),
            log_library_sd=float(max(log_library.std(ddof=1) if x.shape[0] > 1 else 0.0, 1e-6)),
            coord_mean=coord_mean,
            coord_sd=coord_sd,
        )

    def fit(self, x, gene_names: list[str], coords: np.ndarray | None = None) -> "SplatterModel":
        x = _to_dense_float(x)
        self.gene_names = list(gene_names)
        labels = self.assign_regions(coords, x.shape[0])
        self.region_labels = labels
        self.global_params = self._fit_one_region("global", x, coords)

        self.region_params = []
        self.region_coords = {}
        counts = []
        for rid in np.unique(labels):
            mask = labels == rid
            if int(mask.sum()) < self.min_region_size:
                continue
            region_id = str(int(rid))
            region_coords = None if coords is None else np.asarray(coords[mask, :2], dtype=np.float64)
            self.region_params.append(self._fit_one_region(region_id, x[mask], region_coords))
            self.region_coords[region_id] = np.empty((0, 2), dtype=np.float64) if region_coords is None else region_coords
            counts.append(int(mask.sum()))
        if not self.region_params:
            self.region_params = [self.global_params]
            self.region_coords = {"global": np.empty((0, 2), dtype=np.float64)}
            counts = [x.shape[0]]
        self.region_probs = np.asarray(counts, dtype=np.float64)
        self.region_probs = self.region_probs / self.region_probs.sum()
        return self

    def _sample_counts_for_region(self, params: RegionParams, n: int, rng: np.random.Generator) -> np.ndarray:
        mu = np.maximum(params.mean, 1e-8)
        theta = np.maximum(params.theta, self.dispersion_min)
        p = theta / (theta + mu)
        out = np.empty((n, mu.shape[0]), dtype=np.float64)
        poisson_mask = theta >= (self.dispersion_max * 0.5)
        if np.any(~poisson_mask):
            out[:, ~poisson_mask] = rng.negative_binomial(theta[~poisson_mask], p[~poisson_mask], size=(n, int((~poisson_mask).sum())))
        if np.any(poisson_mask):
            out[:, poisson_mask] = rng.poisson(mu[poisson_mask], size=(n, int(poisson_mask.sum())))
        if self.dropout_match:
            dropout = rng.random(out.shape) < params.zero_rate
            out[dropout] = 0.0
        if self.library_size_match:
            target = np.expm1(rng.normal(params.log_library_mean, params.log_library_sd, size=n))
            target = np.maximum(target, 1.0)
            current = np.maximum(out.sum(axis=1), 1.0)
            out = out * (target[:, None] / current[:, None])
            out = rng.poisson(np.maximum(out, 0.0))
        return np.maximum(np.rint(out), 0).astype(np.float32)

    def _sample_coords_for_region(self, params: RegionParams, n: int, rng: np.random.Generator) -> np.ndarray:
        ref = self.region_coords.get(params.region_id)
        if ref is not None and ref.shape[0] > 0:
            idx = rng.integers(0, ref.shape[0], size=n)
            base = ref[idx].astype(np.float64, copy=True)
        else:
            base = np.repeat(params.coord_mean[None, :], n, axis=0)
        noise_sd = np.maximum(params.coord_sd * self.coord_sigma_factor, 1e-6)
        return base + rng.normal(0.0, noise_sd, size=(n, 2))

    def generate(self, n_samples: int, random_seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.region_params or self.region_probs is None:
            raise RuntimeError("SplatterModel must be fitted before generation")
        rng = np.random.default_rng(self.random_seed if random_seed is None else int(random_seed))
        n_samples = int(n_samples)
        region_idx = rng.choice(len(self.region_params), size=n_samples, p=self.region_probs)
        x_parts = []
        coord_parts = []
        region_parts = []
        for i, params in enumerate(self.region_params):
            loc = np.where(region_idx == i)[0]
            if loc.size == 0:
                continue
            x_parts.append((loc, self._sample_counts_for_region(params, int(loc.size), rng)))
            coord_parts.append((loc, self._sample_coords_for_region(params, int(loc.size), rng)))
            region_parts.append((loc, np.repeat(params.region_id, int(loc.size))))
        x = np.zeros((n_samples, len(self.gene_names)), dtype=np.float32)
        coords = np.zeros((n_samples, 2), dtype=np.float64)
        regions = np.empty(n_samples, dtype=object)
        for loc, block in x_parts:
            x[loc] = block
        for loc, block in coord_parts:
            coords[loc] = block
        for loc, block in region_parts:
            regions[loc] = block
        return x, coords, regions.astype(str)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_regions": self.n_regions,
            "min_region_size": self.min_region_size,
            "dispersion_min": self.dispersion_min,
            "dispersion_max": self.dispersion_max,
            "dropout_match": self.dropout_match,
            "library_size_match": self.library_size_match,
            "coord_sigma_factor": self.coord_sigma_factor,
            "random_seed": self.random_seed,
            "gene_names": self.gene_names,
            "region_labels": None if self.region_labels is None else self.region_labels.tolist(),
            "region_params": [p.to_dict() for p in self.region_params],
            "region_probs": None if self.region_probs is None else self.region_probs.tolist(),
            "region_coords": {k: v.tolist() for k, v in self.region_coords.items()},
            "global_params": None if self.global_params is None else self.global_params.to_dict(),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "SplatterModel":
        model = cls(
            n_regions=state.get("n_regions", 7),
            min_region_size=state.get("min_region_size", 20),
            dispersion_min=state.get("dispersion_min", 0.01),
            dispersion_max=state.get("dispersion_max", 1_000_000.0),
            dropout_match=state.get("dropout_match", True),
            library_size_match=state.get("library_size_match", True),
            coord_sigma_factor=state.get("coord_sigma_factor", 0.15),
            random_seed=state.get("random_seed", 42),
        )
        model.gene_names = list(state.get("gene_names", []))
        labels = state.get("region_labels")
        model.region_labels = None if labels is None else np.asarray(labels, dtype=int)
        model.region_params = [RegionParams.from_dict(p) for p in state.get("region_params", [])]
        probs = state.get("region_probs")
        model.region_probs = None if probs is None else np.asarray(probs, dtype=np.float64)
        model.region_coords = {str(k): np.asarray(v, dtype=np.float64) for k, v in state.get("region_coords", {}).items()}
        global_state = state.get("global_params")
        model.global_params = None if global_state is None else RegionParams.from_dict(global_state)
        return model

