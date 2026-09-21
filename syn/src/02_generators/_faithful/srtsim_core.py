"""SRTsim core Python implementation following the published design.

This module ports the reference-based SRTsim flow used by SpatialSimBench:

1. ``createSRT(count_in, loc_in)``: expression matrix plus x/y/label metadata.
2. ``srtsim_fit(..., sim_scheme="tissue"|"domain")``: fit marginal
   Poisson/NB/ZIP/ZINB distributions gene-wise, optionally per spatial domain.
3. ``srtsim_count()``: sample from fitted marginals and reorder sampled values
   according to the rank of the reference expression, preserving spatial
   expression patterns.

The implementation follows the published SRTsim design and package call;
provenance records
``generator_backend=python_reimplementation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
from scipy import optimize, sparse, special, stats
from sklearn.neighbors import NearestNeighbors


def _to_dense_counts(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    arr = np.asarray(x, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(arr, 0.0)


def _rankdata_random_ties(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return zero-based ranks with random tie breaking, matching R rank(..., random)."""
    x = np.asarray(x)
    jitter = rng.uniform(0.0, 1e-12, size=x.shape)
    order = np.lexsort((jitter, x))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(x.size)
    return ranks


def rank_preserving_assign(values: np.ndarray, reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Equivalent to R ``sort(simRawExpr)[rank(realdata, ties.method="random")]``."""
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    ranks = _rankdata_random_ties(reference, rng)
    return np.sort(values)[ranks]


def _nb_nll(param: Iterable[float], y: np.ndarray) -> float:
    theta = max(float(param[0]), 1e-10)
    mu = max(float(param[1]), 1e-10)
    ll = (
        special.gammaln(theta + y)
        - special.gammaln(theta)
        - special.gammaln(y + 1.0)
        + theta * np.log(theta)
        + y * np.log(mu + (y == 0))
        - (theta + y) * np.log(theta + mu)
    )
    return float(-np.sum(ll))


def _nb_nll_fixed_mu(theta: float, mu: float, y: np.ndarray) -> float:
    return _nb_nll((theta, mu), y)


def _poisson_nll(mu: float, y: np.ndarray) -> float:
    mu = max(float(mu), 1e-10)
    return float(-np.sum(y * np.log(mu) - mu - special.gammaln(y + 1.0)))


def _zip_nll(lam: float, y: np.ndarray) -> float:
    lam = max(float(lam), 1e-10)
    zero = y == 0
    n0 = int(zero.sum())
    n = int(y.size)
    nz_y = y[~zero]
    denom = max(1.0 - np.exp(-lam), 1e-12)
    pi0 = (n0 / max(n, 1) - np.exp(-lam)) / denom
    pi0 = float(np.clip(pi0, 1e-12, 1.0 - 1e-12))
    ll_zero = n0 * np.log(pi0 + (1.0 - pi0) * np.exp(-lam))
    ll_nonzero = np.sum(np.log(1.0 - pi0) + nz_y * np.log(lam) - lam - special.gammaln(nz_y + 1.0))
    return float(-(ll_zero + ll_nonzero))


def _zinb_nll(param: Iterable[float], y: np.ndarray) -> float:
    theta = max(float(param[0]), 1e-10)
    mu = max(float(param[1]), 1e-10)
    logit_pi = float(param[2])
    pi0 = 1.0 / (1.0 + np.exp(-logit_pi))
    zero = y == 0
    n0 = int(zero.sum())
    nz_y = y[~zero]
    nb_zero = (theta / (mu + theta)) ** theta
    ll_zero = n0 * np.log(np.maximum(pi0 + (1.0 - pi0) * nb_zero, 1e-300))
    ll_nonzero = np.sum(
        np.log(np.maximum(1.0 - pi0, 1e-300))
        + special.gammaln(theta + nz_y)
        - special.gammaln(theta)
        - special.gammaln(nz_y + 1.0)
        + theta * np.log(theta)
        + nz_y * np.log(mu)
        - (theta + nz_y) * np.log(theta + mu)
    )
    return float(-(ll_zero + ll_nonzero))


@dataclass
class MarginalParam:
    pi0: float
    theta: float
    mu: float
    llk: float
    model_selected: str
    converged: bool = True


@dataclass
class FittedBlock:
    params: list[Optional[MarginalParam]]
    gene_sel1: np.ndarray
    gene_sel2: np.ndarray
    n_cell: int
    n_read: float


def fit_poisson(y: np.ndarray, maxiter: int = 500) -> MarginalParam:
    y = np.asarray(y, dtype=np.float64)
    upper = max(float(np.max(y)), float(np.mean(y)), 1.0)
    try:
        res = optimize.minimize_scalar(
            _poisson_nll,
            args=(y,),
            method="bounded",
            bounds=(1e-10, upper),
            options={"maxiter": maxiter},
        )
        mu = float(res.x)
        llk = -float(res.fun)
        converged = bool(res.success)
    except Exception:
        mu = float(np.mean(y))
        llk = -_poisson_nll(mu, y)
        converged = False
    return MarginalParam(0.0, np.inf, max(mu, 1e-10), llk, "Poisson", converged)


def fit_nb(y: np.ndarray, maxiter: int = 500) -> MarginalParam:
    y = np.asarray(y, dtype=np.float64)
    mean = float(np.mean(y))
    var = float(np.var(y, ddof=1)) if y.size > 1 else mean
    size = mean * mean / (var - mean) if var > mean and mean > 0 else 100.0
    size = float(np.clip(size, 1e-4, 1e4))
    try:
        if y.size < 1000:
            res = optimize.minimize(
                _nb_nll,
                np.array([size, max(mean, 1e-4)]),
                args=(y,),
                method="BFGS",
                options={"maxiter": maxiter},
            )
            theta, mu = res.x
            llk = -float(res.fun)
            converged = bool(res.success)
        else:
            res = optimize.minimize_scalar(
                _nb_nll_fixed_mu,
                args=(max(mean, 1e-10), y),
                method="bounded",
                bounds=(1e-4, 1000.0),
                options={"maxiter": maxiter},
            )
            theta = float(res.x)
            mu = mean
            llk = -float(res.fun)
            converged = bool(res.success)
    except Exception:
        theta = size
        mu = max(mean, 1e-10)
        llk = -_nb_nll((theta, mu), y)
        converged = False
    return MarginalParam(0.0, max(float(theta), 1e-8), max(float(mu), 1e-10), llk, "NB", converged)


def fit_zip(y: np.ndarray, maxiter: int = 500) -> MarginalParam:
    y = np.asarray(y, dtype=np.float64)
    n0 = int((y == 0).sum())
    n = int(y.size)
    mean = float(np.mean(y))
    lower = min(-np.log(n0 / n), 0.0) if n0 > 0 and n > 0 else 0.0
    upper = max(float(np.max(y)), mean, 1.0)
    try:
        res = optimize.minimize_scalar(
            _zip_nll,
            args=(y,),
            method="bounded",
            bounds=(max(lower, 1e-10), upper),
            options={"maxiter": maxiter},
        )
        lam = max(float(res.x), 1e-10)
        pi0 = (n0 / max(n, 1) - np.exp(-lam)) / max(1.0 - np.exp(-lam), 1e-12)
        if pi0 <= 0.0 or pi0 >= 1.0 or not res.success:
            return fit_poisson(y, maxiter=maxiter)
        return MarginalParam(float(pi0), np.inf, lam, -float(res.fun), "ZIP", True)
    except Exception:
        return fit_poisson(y, maxiter=maxiter)


def fit_zinb(y: np.ndarray, maxiter: int = 500) -> MarginalParam:
    y = np.asarray(y, dtype=np.float64)
    n0 = int((y == 0).sum())
    n = int(y.size)
    mean = float(np.mean(y))
    var = float(np.var(y, ddof=1)) if n > 1 else mean
    size = mean * mean / (var - mean) if var > mean and mean > 0 else 1.0
    if n0 == 0 or n0 == n:
        return fit_nb(y, maxiter=maxiter)
    try:
        init = np.array([max(size, 1e-4), max(mean, 1e-4), np.log(n0 / max(n - n0, 1))])
        res = optimize.minimize(
            _zinb_nll,
            init,
            args=(y,),
            method="Nelder-Mead",
            options={"maxiter": maxiter},
        )
        theta, mu, logit_pi = res.x
        pi0 = 1.0 / (1.0 + np.exp(-float(logit_pi)))
        if pi0 <= 0.0 or pi0 >= 1.0 or not res.success:
            return fit_nb(y, maxiter=maxiter)
        return MarginalParam(float(pi0), max(float(theta), 1e-8), max(float(mu), 1e-10), -float(res.fun), "ZINB", True)
    except Exception:
        return fit_nb(y, maxiter=maxiter)


def fit_gene_auto(y: np.ndarray, maxiter: int = 500, pval_cutoff: float = 0.05) -> MarginalParam:
    """Port of SRTsim ``getparams(..., marginal="auto_choose")``."""
    y = np.asarray(y, dtype=np.float64)
    mean = float(np.mean(y))
    var = float(np.var(y, ddof=1)) if y.size > 1 else mean
    if mean >= var:
        return fit_poisson(y, maxiter=maxiter)

    nb = fit_nb(y, maxiter=maxiter)
    expected_zero = (nb.theta / (nb.theta + nb.mu)) ** nb.theta
    empirical_zero = float((y == 0).mean())
    if empirical_zero < expected_zero:
        return nb

    try:
        zinb = fit_zinb(y, maxiter=maxiter)
        chisq = 2.0 * (zinb.llk - nb.llk)
        pvalue = 1.0 - stats.chi2.cdf(max(chisq, 0.0), 1)
        if pvalue < pval_cutoff:
            return zinb
    except Exception:
        return nb

    try:
        zip_param = fit_zip(y, maxiter=maxiter)
        # The published R workflow uses a likelihood gain threshold:
        # aic_diff <- 2 * (mle_ZIP[3] - mle_NB[3]); if (aic_diff > 2) ZIP.
        if 2.0 * (zip_param.llk - nb.llk) > 2.0:
            return zip_param
    except Exception:
        pass
    return nb


def fit_single_block(x_gene_by_spot, min_nonzero_num: int = 2, maxiter: int = 500) -> FittedBlock:
    """Port of SRTsim ``fit_single`` for a genes x spots matrix."""
    x = _to_dense_counts(x_gene_by_spot)
    if x.shape[0] > x.shape[1]:
        # Heuristic guard: this function expects genes x spots.
        pass
    n_spots = x.shape[1]
    zero_prop = 1.0 - (x > 0).sum(axis=1) / max(n_spots, 1)
    gene_sel1 = np.where(zero_prop < 1.0 - min_nonzero_num / max(n_spots, 1))[0]
    gene_sel2 = np.setdiff1d(np.arange(x.shape[0]), gene_sel1, assume_unique=True)
    params: list[Optional[MarginalParam]] = [None] * x.shape[0]
    for gene_idx in gene_sel1:
        params[gene_idx] = fit_gene_auto(x[gene_idx], maxiter=maxiter)
    return FittedBlock(params=params, gene_sel1=gene_sel1, gene_sel2=gene_sel2, n_cell=n_spots, n_read=float(x.sum()))


@dataclass
class SRTsimPython:
    """Reference-based SRTsim object for Python pipelines."""

    sim_scheme: str = "tissue"
    min_nonzero_num: int = 2
    maxiter: int = 500
    nn_num: int = 5
    nn_func: str = "mean"
    random_seed: int = 42
    gene_names: list[str] = field(default_factory=list)
    blocks: Dict[str, FittedBlock] = field(default_factory=dict)
    block_labels: Optional[np.ndarray] = None

    def fit(self, x_spot_by_gene, gene_names: list[str], labels: Optional[np.ndarray] = None) -> "SRTsimPython":
        x = _to_dense_counts(x_spot_by_gene)
        self.gene_names = list(gene_names)
        if self.sim_scheme not in {"tissue", "domain"}:
            raise ValueError("sim_scheme must be 'tissue' or 'domain'")
        if self.sim_scheme == "domain":
            if labels is None:
                raise ValueError("domain scheme requires labels")
            labels = np.asarray(labels).astype(str)
            self.block_labels = labels
            self.blocks = {}
            for label in np.unique(labels):
                mask = labels == label
                self.blocks[str(label)] = fit_single_block(
                    x[mask].T,
                    min_nonzero_num=self.min_nonzero_num,
                    maxiter=self.maxiter,
                )
        else:
            self.blocks = {
                "tissue": fit_single_block(
                    x.T,
                    min_nonzero_num=self.min_nonzero_num,
                    maxiter=self.maxiter,
                )
            }
            self.block_labels = None
        return self

    def _sample_gene(self, param: MarginalParam, n: int, rng: np.random.Generator, rr: float = 1.0) -> np.ndarray:
        if param.model_selected == "Poisson":
            raw = rng.poisson(max(rr * param.mu, 1e-10), size=n)
        elif param.model_selected in {"NB", "ZINB"}:
            theta = max(param.theta, 1e-8)
            mu = max(rr * param.mu, 1e-10)
            prob = theta / (theta + mu)
            raw = rng.negative_binomial(theta, prob, size=n)
        elif param.model_selected == "ZIP":
            raw = rng.poisson(max(rr * param.mu, 1e-10), size=n)
        else:
            raw = np.zeros(n, dtype=np.int64)
        if param.pi0 > 0.0:
            keep = rng.binomial(1, 1.0 - min(max(param.pi0, 0.0), 1.0), size=n)
            raw = raw * keep
        return np.asarray(raw, dtype=np.float64)

    def _simulate_block_same_locations(
        self,
        block: FittedBlock,
        real_gene_by_spot: np.ndarray,
        rng: np.random.Generator,
        rr: float = 1.0,
    ) -> np.ndarray:
        n_genes = len(block.params)
        n_spots = real_gene_by_spot.shape[1]
        out = np.zeros((n_genes, n_spots), dtype=np.float64)
        for gene_idx in block.gene_sel1:
            param = block.params[gene_idx]
            if param is None:
                continue
            sampled = self._sample_gene(param, n_spots, rng=rng, rr=rr)
            out[gene_idx] = rank_preserving_assign(sampled, real_gene_by_spot[gene_idx], rng)
        self._rescue_all_zero_genes(out, block, rng, max_nonzero=1)
        return out

    def _pseudo_reference_for_new_locations(
        self,
        real_gene_by_spot: np.ndarray,
        ref_coords: np.ndarray,
        new_coords: np.ndarray,
    ) -> np.ndarray:
        k = min(max(int(self.nn_num), 1), ref_coords.shape[0])
        nn = NearestNeighbors(n_neighbors=k).fit(ref_coords[:, :2])
        indices = nn.kneighbors(new_coords[:, :2], return_distance=False)
        if k == 1:
            return real_gene_by_spot[:, indices[:, 0]]
        if self.nn_func == "median":
            return np.median(real_gene_by_spot[:, indices], axis=2)
        if self.nn_func == "ransam":
            rng = np.random.default_rng(self.random_seed + 17)
            pick = rng.integers(0, k, size=indices.shape[0])
            return real_gene_by_spot[:, indices[np.arange(indices.shape[0]), pick]]
        return np.mean(real_gene_by_spot[:, indices], axis=2)

    def _simulate_block_new_locations(
        self,
        block: FittedBlock,
        real_gene_by_spot: np.ndarray,
        ref_coords: np.ndarray,
        new_coords: np.ndarray,
        rng: np.random.Generator,
        rr: float = 1.0,
    ) -> np.ndarray:
        pseudo_ref = self._pseudo_reference_for_new_locations(real_gene_by_spot, ref_coords, new_coords)
        n_genes = len(block.params)
        n_spots = new_coords.shape[0]
        out = np.zeros((n_genes, n_spots), dtype=np.float64)
        for gene_idx in block.gene_sel1:
            param = block.params[gene_idx]
            if param is None:
                continue
            sampled = self._sample_gene(param, n_spots, rng=rng, rr=rr)
            out[gene_idx] = rank_preserving_assign(sampled, pseudo_ref[gene_idx], rng)
        self._rescue_all_zero_genes(out, block, rng, max_nonzero=max(round(n_spots / max(real_gene_by_spot.shape[1], 1)), 1))
        return out

    @staticmethod
    def _rescue_all_zero_genes(out: np.ndarray, block: FittedBlock, rng: np.random.Generator, max_nonzero: int) -> None:
        all_zero = [int(gene_idx) for gene_idx in block.gene_sel1 if out[gene_idx].sum() == 0]
        if not all_zero:
            return
        # SRTsim R rescues one location when exactly one fitted gene is all-zero;
        # only the multi-gene branch uses max(round(num_new / num_old), 1).
        n_mark = int(max_nonzero) if len(all_zero) > 1 else 1
        n_mark = min(max(n_mark, 1), out.shape[1])
        for gene_idx in all_zero:
            idx = rng.choice(out.shape[1], size=n_mark, replace=False)
            out[gene_idx, idx] = 1.0

    def generate(
        self,
        x_ref_spot_by_gene,
        labels: Optional[np.ndarray] = None,
        ref_coords: Optional[np.ndarray] = None,
        new_coords: Optional[np.ndarray] = None,
        new_labels: Optional[np.ndarray] = None,
        rr: float = 1.0,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a spots x genes count matrix."""
        x_ref = _to_dense_counts(x_ref_spot_by_gene)
        rng = np.random.default_rng(self.random_seed if random_seed is None else int(random_seed))

        if self.sim_scheme == "tissue":
            block = self.blocks["tissue"]
            if new_coords is None:
                out = self._simulate_block_same_locations(block, x_ref.T, rng=rng, rr=rr)
            else:
                if ref_coords is None:
                    raise ValueError("new_coords generation requires ref_coords")
                out = self._simulate_block_new_locations(
                    block,
                    x_ref.T,
                    np.asarray(ref_coords, dtype=np.float64),
                    np.asarray(new_coords, dtype=np.float64),
                    rng=rng,
                    rr=rr,
                )
            return np.maximum(np.rint(out.T), 0).astype(np.int64)

        if labels is None:
            raise ValueError("domain generation requires reference labels")
        labels = np.asarray(labels).astype(str)
        if new_coords is None:
            new_labels = labels if new_labels is None else np.asarray(new_labels).astype(str)
            out = np.zeros((new_labels.size, x_ref.shape[1]), dtype=np.float64)
            offset_by_label: dict[str, np.ndarray] = {}
            for label in np.unique(new_labels):
                offset_by_label[label] = np.where(new_labels == label)[0]
            for label, dest_idx in offset_by_label.items():
                if label not in self.blocks:
                    continue
                ref_mask = labels == label
                block_out = self._simulate_block_same_locations(self.blocks[label], x_ref[ref_mask].T, rng=rng, rr=rr)
                repeats = int(np.ceil(dest_idx.size / max(block_out.shape[1], 1)))
                out[dest_idx] = np.tile(block_out.T, (repeats, 1))[: dest_idx.size]
            return np.maximum(np.rint(out), 0).astype(np.int64)

        if ref_coords is None or new_labels is None:
            raise ValueError("domain new_coords generation requires ref_coords and new_labels")
        new_labels = np.asarray(new_labels).astype(str)
        out = np.zeros((new_labels.size, x_ref.shape[1]), dtype=np.float64)
        for label in np.unique(new_labels):
            if label not in self.blocks:
                continue
            ref_mask = labels == label
            dest_mask = new_labels == label
            block_out = self._simulate_block_new_locations(
                self.blocks[label],
                x_ref[ref_mask].T,
                np.asarray(ref_coords, dtype=np.float64)[ref_mask],
                np.asarray(new_coords, dtype=np.float64)[dest_mask],
                rng=rng,
                rr=rr,
            )
            out[dest_mask] = block_out.T
        return np.maximum(np.rint(out), 0).astype(np.int64)
