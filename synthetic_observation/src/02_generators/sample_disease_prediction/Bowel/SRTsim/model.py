"""Project-local SRTsim statistical model.

Fits gene-wise count distributions and generates rank-preserving synthetic
expression with optional spatial metadata. The implementation follows the
published SRTsim design and stores provenance in checkpoint metadata.
"""

import logging
import zlib
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import numpy as np
from scipy import optimize, sparse, stats as sp_stats

# ============================================================
# log
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SRTsim.model")

# ============================================================
# data type 
# ============================================================

DATA_TYPE_COUNT = "count"
DATA_TYPE_CONTINUOUS = "continuous"


def rank_preserving_assign(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Assign sorted simulated values back to spots according to the reference rank.

    This mirrors the R implementation `sort(simRawExpr)[rank(realdata)]`: spots
    with lower reference expression receive lower simulated values, preserving
    the spatial rank pattern through rank-based value assignment.
    """
    values = np.asarray(values)
    reference = np.asarray(reference)
    order = np.argsort(reference, kind="mergesort")
    assigned = np.empty_like(values, dtype=np.float64)
    assigned[order] = np.sort(values)
    return assigned


# ============================================================
# statisticshelper functions
# ============================================================

def nb_loglik(params, y):
    """negative distributionnegative log-likelihood (size=theta, mu)"""
    theta = max(params[0], 1e-10)
    mu = max(params[1], 1e-10)
    # numerical stability: for log clip parameters to lower bounds,avoid log(0) = -inf
    log_mu = np.log(np.maximum(mu, 1e-300))
    log_theta_mu = np.log(np.maximum(theta + mu, 1e-300))
    ll = np.sum(
        sp_stats.gammaln(theta + y)
        - sp_stats.gammaln(theta)
        - sp_stats.gammaln(y + 1)
        + theta * np.log(np.maximum(theta, 1e-300))
        + y * log_mu
        - (theta + y) * log_theta_mu
    )
    return -ll


def nb_loglik_fixed_mu(theta, mu, y):
    """fixed mu  of NB negative log-likelihood"""
    theta = max(theta, 1e-10)
    mu = max(mu, 1e-10)
    log_mu = np.log(np.maximum(mu, 1e-300))
    log_theta_mu = np.log(np.maximum(theta + mu, 1e-300))
    ll = np.sum(
        sp_stats.gammaln(theta + y)
        - sp_stats.gammaln(theta)
        - sp_stats.gammaln(y + 1)
        + theta * np.log(np.maximum(theta, 1e-300))
        + y * log_mu
        - (theta + y) * log_theta_mu
    )
    return -ll


def poisson_loglik(mu, y):
    """Poisson negative log-likelihood"""
    mu = max(mu, 1e-10)
    return -np.sum(y * np.log(mu) - mu - sp_stats.gammaln(y + 1))


def zip_loglik(lam, y):
    """ZIP negative log-likelihood"""
    n = len(y)
    n0 = np.sum(y == 0)
    nz_y = y[y > 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        pp = (n0 / n - np.exp(-lam)) / (1 - np.exp(-lam))
        pp = np.clip(pp, 1e-10, 1 - 1e-10)

        ll_zero = n0 * np.log(pp + (1 - pp) * np.exp(-lam))
        ll_nonzero = np.sum(
            np.log(1 - pp) + nz_y * np.log(lam) - lam - sp_stats.gammaln(nz_y + 1)
        )
    return -(ll_zero + ll_nonzero)


def zinb_loglik(params, y):
    """ZINB negative log-likelihood: params = [theta, mu, logit_pi0]"""
    theta = max(params[0], 1e-10)
    mu = max(params[1], 1e-10)
    logit_pi = params[2]

    pp = 1.0 / (1.0 + np.exp(-logit_pi))  # sigmoid
    n0 = np.sum(y == 0)
    nz_y = y[y > 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        # numerical stability: for log clip parameters to lower bounds
        log_theta = np.log(np.maximum(theta, 1e-300))
        log_mu = np.log(np.maximum(mu, 1e-300))
        log_theta_mu = np.log(np.maximum(theta + mu, 1e-300))

        nb_zero_prob = np.exp(theta * (log_theta - log_theta_mu))
        ll_zero = n0 * np.log(np.maximum(pp + (1 - pp) * nb_zero_prob, 1e-300))
        ll_nz = np.sum(
            np.log(np.maximum(1 - pp, 1e-300))
            + sp_stats.gammaln(theta + nz_y)
            - sp_stats.gammaln(theta)
            - sp_stats.gammaln(nz_y + 1)
            + theta * log_theta
            + nz_y * log_mu
            - (theta + nz_y) * log_theta_mu
        )
    return -(ll_zero + ll_nz)


# ============================================================
#  genedistributionfit
# ============================================================

def fit_poisson(gene_data, maxiter=500):
    """fit Poisson distribution -> (pi0=0, theta=inf, mu, llk)"""
    m = np.mean(gene_data)
    try:
        res = optimize.minimize_scalar(
            poisson_loglik, args=(gene_data,),
            method="bounded", bounds=(1e-10, max(np.max(gene_data), 1.0)),
            options={"maxiter": maxiter},
        )
        mu = res.x
        llk = -res.fun
        converged = bool(res.success)
    except Exception:
        mu, llk, converged = m, -np.inf, False
    return {"pi0": 0.0, "theta": np.inf, "mu": float(mu),
            "llk": float(llk), "converged": converged}


def fit_zip(gene_data, maxiter=500):
    """fit ZIP distribution -> (pi0, theta=inf, mu, llk)"""
    n = len(gene_data)
    n0 = np.sum(gene_data == 0)
    m = np.mean(gene_data)

    try:
        lower = max(-np.log(n0 / n) if n0 > 0 else 0, 0)
        upper = max(np.max(gene_data), m + 1)
        res = optimize.minimize_scalar(
            zip_loglik, args=(gene_data,),
            method="bounded", bounds=(lower, upper),
            options={"maxiter": maxiter},
        )
        lam = res.x
        pp = (n0 / n - np.exp(-lam)) / (1 - np.exp(-lam))

        if pp <= 0 or pp >= 1 or not res.success:
            p_result = fit_poisson(gene_data, maxiter)
            return {**p_result, "llk": float(p_result["llk"])}

        llk = -res.fun
        return {"pi0": float(pp), "theta": np.inf, "mu": float(lam),
                "llk": float(llk), "converged": True}
    except Exception:
        p_result = fit_poisson(gene_data, maxiter)
        return {**p_result, "llk": float(p_result["llk"])}


def fit_nb(gene_data, maxiter=500):
    """fit NB distribution -> (pi0=0, theta, mu, llk)"""
    m = np.mean(gene_data)
    v = np.var(gene_data, ddof=1) if len(gene_data) > 1 else m

    #  initialize
    if v > m:
        size_init = m ** 2 / (v - m)
    else:
        size_init = 100.0
    size_init = np.clip(size_init, 0.01, 1000.0)

    try:
        if len(gene_data) < 1000:
            res = optimize.minimize(
                nb_loglik, [size_init, m], args=(gene_data,),
                method="L-BFGS-B",
                bounds=[(1e-4, 1e4), (1e-4, 1e4)],
                options={"maxiter": maxiter},
            )
            theta, mu = res.x
        else:
            res = optimize.minimize_scalar(
                nb_loglik_fixed_mu, args=(m, gene_data),
                method="bounded", bounds=(1e-4, 1000),
                options={"maxiter": maxiter},
            )
            theta = res.x
            mu = m

        llk = -nb_loglik([theta, mu], gene_data)
        theta = max(theta, 1e-6)
        mu = max(mu, 1e-10)
        return {"pi0": 0.0, "theta": float(theta), "mu": float(mu),
                "llk": float(llk), "converged": True}
    except Exception:
        theta = max(size_init, 1e-6)
        llk = -nb_loglik([theta, m], gene_data)
        return {"pi0": 0.0, "theta": float(theta), "mu": float(m),
                "llk": float(llk), "converged": False}


def fit_zinb(gene_data, maxiter=500):
    """fit ZINB distribution -> (pi0, theta, mu, llk)"""
    n = len(gene_data)
    n0 = np.sum(gene_data == 0)
    m = np.mean(gene_data)
    v = np.var(gene_data, ddof=1) if n > 1 else m

    if n0 == 0:
        nb_result = fit_nb(gene_data, maxiter)
        return {**nb_result, "llk": float(nb_result["llk"])}

    size_init = m ** 2 / (v - m) if v > m else 1.0
    size_init = np.clip(size_init, 0.01, 1000.0)
    logit_pi_init = float(np.log(n0 / max(n - n0, 1)))

    try:
        res = optimize.minimize(
            zinb_loglik, [size_init, m, logit_pi_init], args=(gene_data,),
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4},
        )
        theta, mu, logit_pi = res.x
        pp = 1.0 / (1.0 + np.exp(-logit_pi))

        if pp <= 0 or pp >= 1 or not res.success:
            nb_result = fit_nb(gene_data, maxiter)
            return {"pi0": 0.0, "theta": float(nb_result["theta"]),
                    "mu": float(nb_result["mu"]),
                    "llk": float(nb_result["llk"]), "converged": False}

        llk = -res.fun
        return {"pi0": float(pp), "theta": float(max(theta, 1e-6)),
                "mu": float(max(mu, 1e-10)),
                "llk": float(llk), "converged": True}
    except Exception:
        nb_result = fit_nb(gene_data, maxiter)
        return {"pi0": 0.0, "theta": float(nb_result["theta"]),
                "mu": float(nb_result["mu"]),
                "llk": float(nb_result["llk"]), "converged": False}


def fit_gaussian(gene_data, **kwargs):
    """fit Gaussian distribution (usein Z-score etc data)"""
    mu = float(np.mean(gene_data))
    sigma = float(np.std(gene_data, ddof=1)) if len(gene_data) > 1 else 1.0
    sigma = max(sigma, 1e-6)
    llk = float(np.sum(sp_stats.norm.logpdf(gene_data, mu, sigma)))
    return {"pi0": 0.0, "theta": sigma, "mu": mu,
            "llk": llk, "converged": True}


def auto_choose_count(gene_data, maxiter=500):
    """
    automaticselect numberdistribution (ZINB / NB / ZIP / Poisson)

      (  R  ):
      1.   mean >= var -> Poisson
      2. fit NB;  ratio < NB expected zero fraction ->  ,checkis requires ZIP
      3.  fit ZINB,LRT   ZINB vs NB
      4. LRT not significant -> attempt ZIP,use AIC   ZIP vs NB
    """
    m = np.mean(gene_data)
    v = np.var(gene_data, ddof=1) if len(gene_data) > 1 else m
    n = len(gene_data)

    # variance <= mean -> Poisson  can
    if m >= v:
        return fit_poisson(gene_data, maxiter)

    # fit NB
    nb_result = fit_nb(gene_data, maxiter)
    theta_nb = nb_result["theta"]
    mu_nb = nb_result["mu"]

    #  ratio
    emp_zero = np.sum(gene_data == 0) / n
    # NB expected zero fraction
    expected_zero = (theta_nb / (theta_nb + mu_nb)) ** theta_nb

    #  NB ratio handling is outside this branch 
    if emp_zero < expected_zero:
        zip_result = fit_zip(gene_data, maxiter)
        aic_diff = 2 * (zip_result["llk"] - nb_result["llk"])  # note: llk is positive
        # AIC(ZIP) - AIC(NB) = 2*(k_zip - k_nb) - 2*(llk_zip - llk_nb)
        # = 2*(2-2) - 2*(zip_llk - nb_llk) = -2*(zip_llk - nb_llk)
        # Using raw log-likelihoods (negative):
        aic_zip_minus_nb = 2 * (zip_result["llk"] - nb_result["llk"])
        if aic_zip_minus_nb > 2:
            return {"pi0": zip_result["pi0"], "theta": np.inf,
                    "mu": zip_result["mu"],
                    "llk": zip_result["llk"], "converged": True}
        return nb_result

    # attempt ZINB
    try:
        zinb_result = fit_zinb(gene_data, maxiter)
        chisq_val = 2 * (zinb_result["llk"] - nb_result["llk"])
        pvalue = 1 - sp_stats.chi2.cdf(max(chisq_val, 0), df=1)

        if pvalue < 0.05:
            return zinb_result

        # ZINB not significant ->   ZIP vs NB
        zip_result = fit_zip(gene_data, maxiter)
        aic_diff = 2 * (zip_result["llk"] - nb_result["llk"])
        if aic_diff > 2:
            return {"pi0": zip_result["pi0"], "theta": np.inf,
                    "mu": zip_result["mu"],
                    "llk": zip_result["llk"], "converged": True}
        return nb_result
    except Exception:
        return nb_result


# ============================================================
# coremodel 
# ============================================================

@dataclass
class SRTsimModel:
    """
    SRTsim spatial transcriptomics statistical simulation model (Python implementation)

    coremethod:
      fit()       - fit a marginal distribution for each gene
      generate()  - sample from fitted distributions + rank preservationgenerate synthetic data

    Attributes:
        params: dict, gene_name -> {pi0, theta, mu, model}
        gene_sel: ndarray, indices of genes with fitted parameters
        gene_names: list, gene 
        data_type: "count" | "continuous"
        sim_scheme: "tissue" | "domain"
        n_spots: int
        domain_params: dict (domain scheme only)
    """

    random_seed: int = 42
    sim_scheme: str = "tissue"
    min_nonzero: int = 2
    maxiter: int = 500
    nn_num: int = 5

    # Fitted state
    params: Optional[Dict] = field(default=None, repr=False)
    gene_sel: Optional[np.ndarray] = field(default=None, repr=False)
    gene_names: Optional[List[str]] = field(default=None, repr=False)
    data_type: Optional[str] = None
    n_spots: int = 0
    domain_params: Optional[Dict] = field(default=None, repr=False)

    # ------ data type  ------

    @staticmethod
    def _detect_data_type(X: np.ndarray) -> str:
        """detect data type: count-valued vs continuous"""
        min_val = X.min()
        if min_val < -0.01:
            return DATA_TYPE_CONTINUOUS
        # checkis as numbervalue ( numberdata )
        sample = X.ravel()[:min(10000, X.size)]
        is_int = np.allclose(sample, np.round(sample), atol=0.01)
        return DATA_TYPE_COUNT if is_int else DATA_TYPE_CONTINUOUS

    # ------ fit ------

    def _fit_gene(self, gene_data: np.ndarray, data_type: str) -> Optional[Dict]:
        """Fit the distribution of one gene."""
        if data_type == DATA_TYPE_CONTINUOUS:
            return fit_gaussian(gene_data)
        # count-valued
        return auto_choose_count(gene_data, self.maxiter)

    def fit(self, X, gene_names: List[str],
            coords: Optional[np.ndarray] = None,
            labels: Optional[np.ndarray] = None):
        """
        fit a marginal distribution for each gene

        Args:
            X: (n_spots, n_genes) expression matrix
            gene_names: gene list
            coords: (n_spots, 2) spatial coordinates (domain mode requires)
            labels: (n_spots,) spatial-domain labels (domain mode requires)
        """
        self.gene_names = list(gene_names)
        rng = np.random.RandomState(self.random_seed)

        #   dense
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)
        n_spots, n_genes = X.shape
        self.n_spots = n_spots

        # detect data type
        self.data_type = self._detect_data_type(X)
        logger.info(f"data type: {self.data_type} ({n_spots} spots x {n_genes} genes)")

        if self.sim_scheme == "domain" and labels is not None:
            self._fit_domain(X, labels, coords)
        else:
            self._fit_tissue(X, rng)

    def _fit_tissue(self, X: np.ndarray, rng: np.random.RandomState):
        """Tissue mode: globalfit"""
        n_genes = X.shape[1]
        self.params = {}
        gene_sel_list = []

        for g in range(n_genes):
            gene_data = X[:, g]
            n_nz = np.sum(gene_data != 0)

            if n_nz < self.min_nonzero:
                self.params[self.gene_names[g]] = {
                    "pi0": 1.0, "theta": np.inf, "mu": 0.0,
                    "model": "Zero", "llk": 0.0,
                }
                continue

            gene_sel_list.append(g)
            try:
                result = self._fit_gene(gene_data, self.data_type)
                model_type = self._get_model_type(result)
                self.params[self.gene_names[g]] = {
                    "pi0": result["pi0"], "theta": result["theta"],
                    "mu": result["mu"], "model": model_type,
                    "llk": result.get("llk", 0.0),
                }
            except Exception as e:
                self.params[self.gene_names[g]] = {
                    "pi0": 0.0, "theta": np.inf, "mu": float(np.mean(gene_data)),
                    "model": "Poisson", "llk": 0.0,
                }

            if (g + 1) % 500 == 0 or (g + 1) == n_genes:
                logger.info(f"  fitting progress: {g + 1}/{n_genes} genes")

        self.gene_sel = np.array(gene_sel_list)
        self._log_summary()

    def _fit_domain(self, X: np.ndarray, labels: np.ndarray,
                    coords: Optional[np.ndarray]):
        """Domain mode: byspatial fit"""
        unique_labels = np.unique(labels)
        self.domain_params = {}

        for lbl in unique_labels:
            mask = labels == lbl
            X_domain = X[mask]
            logger.info(f"Domain '{lbl}': {mask.sum()} spots")

            domain_model = SRTsimModel(
                random_seed=self.random_seed,
                sim_scheme="tissue",
                min_nonzero=self.min_nonzero,
                maxiter=self.maxiter,
            )
            rng = np.random.RandomState(self.random_seed)
            domain_model.gene_names = self.gene_names
            domain_model.data_type = self.data_type
            domain_model.n_spots = X_domain.shape[0]
            domain_model._fit_tissue(X_domain, rng)

            self.domain_params[str(lbl)] = {
                "params": domain_model.params,
                "gene_sel": domain_model.gene_sel,
                "n_spots": int(mask.sum()),
            }

        # global params  items domain ( )
        first_key = list(self.domain_params.keys())[0]
        self.params = self.domain_params[first_key]["params"]
        self.gene_sel = self.domain_params[first_key]["gene_sel"]
        self._log_summary()

    # ------ generate ------

    def generate(self, X_ref, n_generate: Optional[int] = None,
                 labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        generatesyntheticexpressiondata

        Core strategy: sample fitted distributions and assign values by reference ranks.

        Args:
            X_ref: (n_ref_spots, n_genes) reference expression matrix used for rank assignment
            n_generate: generate spot count (default = reference spot count)
            labels: spatial-domain labels (domain mode requires)

        Returns:
            synthetic: (n_generate, n_genes) syntheticexpression matrix
        """
        if self.params is None and self.domain_params is None:
            raise RuntimeError("fit() must run before requesting fitted parameters")

        if sparse.issparse(X_ref):
            X_ref = X_ref.toarray()
        X_ref = np.asarray(X_ref, dtype=np.float64)

        n_ref = X_ref.shape[0]
        n_genes = X_ref.shape[1]

        if n_generate is None:
            n_generate = n_ref

        rng = np.random.RandomState(self.random_seed)

        if self.sim_scheme == "domain" and self.domain_params is not None and labels is not None:
            return self._generate_domain(X_ref, n_generate, labels, rng)

        # Tissue scheme
        synthetic = np.zeros((n_generate, n_genes), dtype=np.float64)

        for g in range(n_genes):
            gene_name = self.gene_names[g]
            param = self.params[gene_name]

            # generatevalue
            values = self._sample_from_params(param, n_generate, rng)

            # rank preservation: by reference data rank order
            ref_gene = X_ref[:, g] if g < X_ref.shape[1] else X_ref[:, 0]
            n_use = min(n_generate, n_ref)

            if n_generate == n_ref:
                synthetic[:, g] = rank_preserving_assign(values, ref_gene)
            else:
                # differentcount: use NN  numberstrategy
                synthetic[:n_use, g] = rank_preserving_assign(
                    values[:n_use], ref_gene[:n_use]
                )
                if n_generate > n_ref:
                    synthetic[n_use:, g] = values[n_use:]

        # post-processing
        if self.data_type == DATA_TYPE_COUNT:
            synthetic = self._postprocess_count(synthetic, rng, self.gene_names, self.params)
            synthetic = np.maximum(np.round(synthetic), 0)
        else:
            # continuous: clip to the reference data range
            ref_min, ref_max = X_ref.min(), X_ref.max()
            synthetic = np.clip(synthetic, ref_min - 1, ref_max + 1)

        return synthetic

    def _generate_domain(self, X_ref, n_generate, labels, rng):
        """Domain modegenerate"""
        n_genes = X_ref.shape[1]
        synthetic = np.zeros((n_generate, n_genes), dtype=np.float64)

        unique_labels = np.unique(labels)
        # byratio number of generated observations
        label_counts = {lbl: int(np.sum(labels == lbl)) for lbl in unique_labels}
        total = sum(label_counts.values())

        offset = 0
        for lbl in unique_labels:
            lbl_str = str(lbl)
            n_lbl = max(1, int(n_generate * label_counts[lbl] / total))
            if lbl_str not in self.domain_params:
                offset += n_lbl
                continue

            dp = self.domain_params[lbl_str]
            label_seed = zlib.crc32(lbl_str.encode("utf-8")) % 1000
            sub_model = SRTsimModel(random_seed=self.random_seed + label_seed)
            sub_model.params = dp["params"]
            sub_model.gene_names = self.gene_names
            sub_model.data_type = self.data_type

            mask = labels == lbl
            X_domain = X_ref[mask] if mask.sum() > 0 else X_ref
            sub_synth = sub_model.generate(X_domain, n_lbl)
            end = min(offset + n_lbl, n_generate)
            synthetic[offset:end] = sub_synth[:end - offset]
            offset = end

        return synthetic

    # ------ sample ------

    def _sample_from_params(self, param: Dict, n: int,
                            rng: np.random.RandomState) -> np.ndarray:
        """sample from fitted parameters"""
        model = param["model"]

        if model == "Gaussian":
            return rng.normal(param["mu"], param["theta"], size=n)

        if model == "Zero":
            return np.zeros(n)

        # count-valueddistribution
        pi0 = param["pi0"]
        theta = param["theta"]
        mu = param["mu"]

        #  partially
        if pi0 > 0 and pi0 < 1:
            is_zero = rng.binomial(1, pi0, size=n).astype(bool)
        else:
            is_zero = np.zeros(n, dtype=bool)

        if model == "Poisson":
            samples = rng.poisson(lam=max(mu, 1e-10), size=n).astype(np.float64)
        elif model == "NB":
            theta_safe = max(theta, 1e-4)
            mu_safe = max(mu, 1e-10)
            p_nb = theta_safe / (theta_safe + mu_safe)
            samples = rng.negative_binomial(
                n=max(theta_safe, 0.01), p=p_nb, size=n
            ).astype(np.float64)
        elif model == "ZIP":
            samples = rng.poisson(lam=max(mu, 1e-10), size=n).astype(np.float64)
        elif model == "ZINB":
            theta_safe = max(theta, 1e-4)
            mu_safe = max(mu, 1e-10)
            p_nb = theta_safe / (theta_safe + mu_safe)
            samples = rng.negative_binomial(
                n=max(theta_safe, 0.01), p=p_nb, size=n
            ).astype(np.float64)
        else:
            samples = rng.poisson(lam=max(mu, 1e-10), size=n).astype(np.float64)

        #  use 
        if pi0 > 0:
            samples[is_zero] = 0

        return samples

    # ------ post-processing ------

    @staticmethod
    def _postprocess_count(synthetic: np.ndarray,
                           rng: np.random.RandomState,
                           gene_names: Optional[List[str]] = None,
                           params: Optional[Dict] = None) -> np.ndarray:
        """post-processing: rescues fitted non-zero genes that sampled all zeros."""
        for g in range(synthetic.shape[1]):
            if synthetic[:, g].sum() == 0:
                if gene_names is not None and params is not None:
                    gene_param = params.get(gene_names[g], {})
                    if gene_param.get("model") == "Zero" or gene_param.get("mu", 0.0) <= 0:
                        continue
                idx = rng.randint(0, synthetic.shape[0])
                synthetic[idx, g] = 1
        return synthetic

    # ------  method ------

    @staticmethod
    def _get_model_type(result: Dict) -> str:
        """fromfitresult model type"""
        pi0 = result["pi0"]
        theta = result["theta"]

        if pi0 == 0 and np.isinf(theta):
            return "Poisson"
        elif pi0 == 0 and not np.isinf(theta):
            return "NB"
        elif pi0 != 0 and np.isinf(theta):
            return "ZIP"
        else:
            return "ZINB"

    def _log_summary(self):
        """ fitresultsummary"""
        if self.params is None:
            return
        model_counts = {}
        for g, p in self.params.items():
            m = p["model"]
            model_counts[m] = model_counts.get(m, 0) + 1

        logger.info(f"\n{'='*40}")
        logger.info(f"SRTsim fitsummary")
        logger.info(f"{'='*40}")
        logger.info(f"  Spots: {self.n_spots}")
        logger.info(f"  Genes: {len(self.params)}")
        logger.info(f"  Data type: {self.data_type}")
        logger.info(f"  Sim scheme: {self.sim_scheme}")
        logger.info(f"  distribution selection:")
        for m, cnt in sorted(model_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {m}: {cnt} ({cnt / len(self.params) * 100:.1f}%)")

        if self.gene_sel is not None:
            logger.info(f"  fittable genes: {len(self.gene_sel)}")
            logger.info(f"  low-expression genes: {len(self.params) - len(self.gene_sel)}")

    # ------  column  ------

    def to_dict(self) -> Dict:
        """return model status as a dictionary for checkpoint storage"""
        # will numpy inf/nan  asstring ensure JSON/pickle  
        state = {
            "random_seed": self.random_seed,
            "sim_scheme": self.sim_scheme,
            "min_nonzero": self.min_nonzero,
            "maxiter": self.maxiter,
            "nn_num": self.nn_num,
            "data_type": self.data_type,
            "n_spots": self.n_spots,
            "gene_names": self.gene_names,
            "params": self.params,
            "gene_sel": self.gene_sel.tolist() if self.gene_sel is not None else None,
        }
        if self.domain_params is not None:
            state["domain_params"] = {}
            for lbl, dp in self.domain_params.items():
                state["domain_params"][lbl] = {
                    "params": dp["params"],
                    "gene_sel": dp["gene_sel"].tolist() if dp["gene_sel"] is not None else None,
                    "n_spots": dp["n_spots"],
                }
        return state

    @classmethod
    def from_dict(cls, state: Dict) -> "SRTsimModel":
        """fromdictionary model"""
        model = cls(
            random_seed=state["random_seed"],
            sim_scheme=state["sim_scheme"],
            min_nonzero=state["min_nonzero"],
            maxiter=state["maxiter"],
            nn_num=state["nn_num"],
        )
        model.data_type = state["data_type"]
        model.n_spots = state["n_spots"]
        model.gene_names = state["gene_names"]
        model.params = state["params"]
        gene_sel = state.get("gene_sel")
        model.gene_sel = np.array(gene_sel) if gene_sel is not None else None

        if "domain_params" in state:
            model.domain_params = {}
            for lbl, dp in state["domain_params"].items():
                gs = dp.get("gene_sel")
                model.domain_params[lbl] = {
                    "params": dp["params"],
                    "gene_sel": np.array(gs) if gs is not None else None,
                    "n_spots": dp["n_spots"],
                }

        return model
