"""SPARsim-style Gamma-Multivariate-Hypergeometric simulator.

This module follows the interface exposed by SpatialSimBench:

1. normalize raw counts and define condition/spatial-region labels;
2. estimate condition-specific gene composition parameters;
3. generate a latent Gamma composition per synthetic cell;
4. draw observed counts with a multivariate hypergeometric sampler so each
   synthetic cell has an explicit library-size constraint.

This module implements the mechanism described by the SpatialSimBench call
pattern and SPARsim's Gamma-Multivariate-Hypergeometric design in Python.
Provenance records the published design reference.
"""

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
    return np.maximum(np.rint(arr), 0.0)


def _size_factors(x: np.ndarray) -> np.ndarray:
    lib = np.maximum(x.sum(axis=1), 1.0)
    med = float(np.median(lib))
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return np.maximum(lib / med, 1e-8)


def _normalize_counts(x: np.ndarray) -> np.ndarray:
    sf = _size_factors(x)
    return x / sf[:, None]


def _safe_log_library(x: np.ndarray) -> tuple[float, float, list[float]]:
    lib = np.maximum(x.sum(axis=1), 1.0)
    log_lib = np.log(lib)
    sd = float(log_lib.std(ddof=1)) if log_lib.size > 1 else 1e-6
    return float(log_lib.mean()), float(max(sd, 1e-6)), lib.astype(float).tolist()


@dataclass
class SPARsimConditionParams:
    condition_id: str
    n_spots: int
    gamma_shape: np.ndarray
    gamma_scale: np.ndarray
    mean_proportion: np.ndarray
    dropout_prob: np.ndarray
    log_library_mean: float
    log_library_sd: float
    empirical_libraries: list[float]
    coord_mean: np.ndarray
    coord_sd: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("gamma_shape", "gamma_scale", "mean_proportion", "dropout_prob", "coord_mean", "coord_sd"):
            out[key] = np.asarray(out[key]).tolist()
        out["empirical_libraries"] = [float(v) for v in self.empirical_libraries]
        return out

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "SPARsimConditionParams":
        return cls(
            condition_id=str(state["condition_id"]),
            n_spots=int(state["n_spots"]),
            gamma_shape=np.asarray(state["gamma_shape"], dtype=np.float64),
            gamma_scale=np.asarray(state["gamma_scale"], dtype=np.float64),
            mean_proportion=np.asarray(state["mean_proportion"], dtype=np.float64),
            dropout_prob=np.asarray(state["dropout_prob"], dtype=np.float64),
            log_library_mean=float(state["log_library_mean"]),
            log_library_sd=float(state["log_library_sd"]),
            empirical_libraries=[float(v) for v in state.get("empirical_libraries", [])],
            coord_mean=np.asarray(state["coord_mean"], dtype=np.float64),
            coord_sd=np.asarray(state["coord_sd"], dtype=np.float64),
        )


class SPARsimGMHPython:
    """Condition-wise Gamma-Multivariate-Hypergeometric simulator."""

    def __init__(
        self,
        n_regions: int = 7,
        min_region_size: int = 20,
        gamma_shape_min: float = 0.05,
        gamma_shape_max: float = 1_000.0,
        finite_pool_factor: float = 20.0,
        library_size_mode: str = "empirical",
        dropout_match: bool = True,
        dropout_quantile: float = 0.99,
        coord_sigma_factor: float = 0.12,
        random_seed: int = 42,
    ):
        self.n_regions = int(n_regions)
        self.min_region_size = int(min_region_size)
        self.gamma_shape_min = float(gamma_shape_min)
        self.gamma_shape_max = float(gamma_shape_max)
        self.finite_pool_factor = float(finite_pool_factor)
        self.library_size_mode = str(library_size_mode)
        self.dropout_match = bool(dropout_match)
        self.dropout_quantile = float(dropout_quantile)
        self.coord_sigma_factor = float(coord_sigma_factor)
        self.random_seed = int(random_seed)
        self.gene_names: list[str] = []
        self.condition_labels: np.ndarray | None = None
        self.condition_params: list[SPARsimConditionParams] = []
        self.condition_probs: np.ndarray | None = None
        self.condition_coords: dict[str, np.ndarray] = {}

    def assign_conditions(self, coords: np.ndarray | None, n_obs: int) -> np.ndarray:
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

    def _fit_condition(self, condition_id: str, raw: np.ndarray, norm: np.ndarray, coords: np.ndarray | None) -> SPARsimConditionParams:
        norm_total = np.maximum(norm.sum(axis=1), 1e-8)
        prop = norm / norm_total[:, None]
        mean_prop = np.maximum(prop.mean(axis=0), 1e-12)
        mean_prop = mean_prop / mean_prop.sum()
        var_prop = np.maximum(prop.var(axis=0, ddof=1) if prop.shape[0] > 1 else mean_prop * 0.1, 1e-12)

        gamma_shape = np.clip((mean_prop * mean_prop) / var_prop, self.gamma_shape_min, self.gamma_shape_max)
        gamma_scale = np.maximum(mean_prop / gamma_shape, 1e-12)

        if self.dropout_match:
            observed_zero = (raw <= 0).mean(axis=0)
            expected_zero = np.exp(-np.maximum(mean_prop * np.median(np.maximum(raw.sum(axis=1), 1.0)), 1e-8))
            dropout_prob = (observed_zero - expected_zero) / np.maximum(1.0 - expected_zero, 1e-8)
            dropout_prob = np.clip(dropout_prob, 0.0, self.dropout_quantile)
        else:
            dropout_prob = np.zeros(raw.shape[1], dtype=np.float64)

        log_mean, log_sd, empirical_lib = _safe_log_library(raw)
        if coords is None or coords.shape[0] == 0:
            coord_mean = np.zeros(2, dtype=np.float64)
            coord_sd = np.ones(2, dtype=np.float64)
        else:
            c = np.asarray(coords[:, :2], dtype=np.float64)
            coord_mean = c.mean(axis=0)
            coord_sd = np.maximum(c.std(axis=0), 1.0)

        return SPARsimConditionParams(
            condition_id=str(condition_id),
            n_spots=int(raw.shape[0]),
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            mean_proportion=mean_prop,
            dropout_prob=dropout_prob,
            log_library_mean=log_mean,
            log_library_sd=log_sd,
            empirical_libraries=empirical_lib,
            coord_mean=coord_mean,
            coord_sd=coord_sd,
        )

    def fit(self, x, gene_names: list[str], coords: np.ndarray | None = None, conditions: np.ndarray | None = None) -> "SPARsimGMHPython":
        raw = _to_dense_count(x)
        norm = _normalize_counts(raw)
        self.gene_names = list(gene_names)
        if conditions is None:
            labels = self.assign_conditions(coords, raw.shape[0])
        else:
            labels = np.asarray(conditions)
            if labels.shape[0] != raw.shape[0]:
                raise ValueError("conditions length does not match n_obs")
        self.condition_labels = labels
        self.condition_params = []
        self.condition_coords = {}
        counts = []

        for cond in np.unique(labels):
            mask = labels == cond
            if int(mask.sum()) < self.min_region_size:
                continue
            cond_id = str(cond)
            cond_coords = None if coords is None else np.asarray(coords[mask, :2], dtype=np.float64)
            self.condition_params.append(self._fit_condition(cond_id, raw[mask], norm[mask], cond_coords))
            self.condition_coords[cond_id] = np.empty((0, 2), dtype=np.float64) if cond_coords is None else cond_coords
            counts.append(int(mask.sum()))

        if not self.condition_params:
            self.condition_params = [self._fit_condition("global", raw, norm, coords)]
            self.condition_coords = {"global": np.empty((0, 2), dtype=np.float64) if coords is None else np.asarray(coords[:, :2], dtype=np.float64)}
            counts = [int(raw.shape[0])]

        self.condition_probs = np.asarray(counts, dtype=np.float64)
        self.condition_probs = self.condition_probs / self.condition_probs.sum()
        return self

    def _sample_library_size(self, params: SPARsimConditionParams, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.library_size_mode == "lognormal" or not params.empirical_libraries:
            lib = np.rint(np.exp(rng.normal(params.log_library_mean, params.log_library_sd, size=n)))
        else:
            empirical = np.asarray(params.empirical_libraries, dtype=np.float64)
            base = empirical[rng.integers(0, empirical.size, size=n)]
            jitter = np.exp(rng.normal(0.0, min(params.log_library_sd, 0.15), size=n))
            lib = np.rint(base * jitter)
        return np.maximum(lib.astype(int), 1)

    def _sample_counts(self, params: SPARsimConditionParams, n: int, rng: np.random.Generator) -> np.ndarray:
        libraries = self._sample_library_size(params, n, rng)
        out = np.zeros((n, len(self.gene_names)), dtype=np.float32)
        for i, lib in enumerate(libraries):
            gamma_weight = rng.gamma(shape=params.gamma_shape, scale=params.gamma_scale)
            if not np.isfinite(gamma_weight).all() or gamma_weight.sum() <= 0:
                gamma_weight = params.mean_proportion.copy()
            prop = np.maximum(gamma_weight, 1e-12)
            prop = prop / prop.sum()
            pool_total = int(max(lib, np.ceil(lib * self.finite_pool_factor), len(prop)))
            colors = np.maximum(np.rint(prop * pool_total).astype(int), 1)
            pool_sum = int(colors.sum())
            draw_n = int(min(lib, pool_sum))
            sampled = rng.multivariate_hypergeometric(colors, draw_n).astype(np.float32)
            if self.dropout_match:
                sampled[rng.random(sampled.shape[0]) < params.dropout_prob] = 0.0
            out[i] = sampled
        return out

    def _sample_coords(self, params: SPARsimConditionParams, n: int, rng: np.random.Generator) -> np.ndarray:
        ref = self.condition_coords.get(params.condition_id)
        if ref is not None and ref.shape[0] > 0:
            base = ref[rng.integers(0, ref.shape[0], size=n)].astype(np.float64, copy=True)
        else:
            base = np.repeat(params.coord_mean[None, :], n, axis=0)
        noise_sd = np.maximum(params.coord_sd * self.coord_sigma_factor, 1e-6)
        return base + rng.normal(0.0, noise_sd, size=(n, 2))

    def generate(self, n_samples: int, random_seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.condition_params or self.condition_probs is None:
            raise RuntimeError("SPARsimGMHPython must be fitted before generation")
        rng = np.random.default_rng(self.random_seed if random_seed is None else int(random_seed))
        n_samples = int(n_samples)
        condition_idx = rng.choice(len(self.condition_params), size=n_samples, p=self.condition_probs)
        x = np.zeros((n_samples, len(self.gene_names)), dtype=np.float32)
        coords = np.zeros((n_samples, 2), dtype=np.float64)
        conditions = np.empty(n_samples, dtype=object)
        for i, params in enumerate(self.condition_params):
            loc = np.where(condition_idx == i)[0]
            if loc.size == 0:
                continue
            x[loc] = self._sample_counts(params, int(loc.size), rng)
            coords[loc] = self._sample_coords(params, int(loc.size), rng)
            conditions[loc] = params.condition_id
        return x, coords, conditions.astype(str)

    @property
    def region_params(self) -> list[SPARsimConditionParams]:
        return self.condition_params

    @property
    def region_probs(self) -> np.ndarray | None:
        return self.condition_probs

    @property
    def region_labels(self) -> np.ndarray | None:
        return None if self.condition_labels is None else np.asarray(self.condition_labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_regions": self.n_regions,
            "min_region_size": self.min_region_size,
            "gamma_shape_min": self.gamma_shape_min,
            "gamma_shape_max": self.gamma_shape_max,
            "finite_pool_factor": self.finite_pool_factor,
            "library_size_mode": self.library_size_mode,
            "dropout_match": self.dropout_match,
            "dropout_quantile": self.dropout_quantile,
            "coord_sigma_factor": self.coord_sigma_factor,
            "random_seed": self.random_seed,
            "gene_names": self.gene_names,
            "condition_labels": None if self.condition_labels is None else np.asarray(self.condition_labels).tolist(),
            "condition_params": [p.to_dict() for p in self.condition_params],
            "condition_probs": None if self.condition_probs is None else self.condition_probs.tolist(),
            "condition_coords": {k: v.tolist() for k, v in self.condition_coords.items()},
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "SPARsimGMHPython":
        model = cls(
            n_regions=state.get("n_regions", 7),
            min_region_size=state.get("min_region_size", 20),
            gamma_shape_min=state.get("gamma_shape_min", 0.05),
            gamma_shape_max=state.get("gamma_shape_max", 1_000.0),
            finite_pool_factor=state.get("finite_pool_factor", 20.0),
            library_size_mode=state.get("library_size_mode", "empirical"),
            dropout_match=state.get("dropout_match", True),
            dropout_quantile=state.get("dropout_quantile", 0.99),
            coord_sigma_factor=state.get("coord_sigma_factor", 0.12),
            random_seed=state.get("random_seed", 42),
        )
        model.gene_names = list(state.get("gene_names", []))
        labels = state.get("condition_labels", state.get("region_labels"))
        model.condition_labels = None if labels is None else np.asarray(labels)
        param_state = state.get("condition_params", state.get("region_params", []))
        model.condition_params = [SPARsimConditionParams.from_dict(p) for p in param_state]
        probs = state.get("condition_probs", state.get("region_probs"))
        model.condition_probs = None if probs is None else np.asarray(probs, dtype=np.float64)
        coord_state = state.get("condition_coords", state.get("region_coords", {}))
        model.condition_coords = {str(k): np.asarray(v, dtype=np.float64) for k, v in coord_state.items()}
        return model
