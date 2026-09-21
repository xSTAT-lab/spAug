#!/usr/bin/env python
"""Regression checks for the faithful Python SPARsim GMH core."""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "02_generators" / "_faithful"))

from sparsim_core import SPARsimGMHPython  # noqa: E402


def make_toy_adata(n_spots: int, n_genes: int, n_regions: int, seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    region = np.repeat(np.arange(n_regions), int(np.ceil(n_spots / n_regions)))[:n_spots]
    coords = np.column_stack(
        [
            region.astype(float) * 6.0 + rng.normal(0.0, 0.6, size=n_spots),
            rng.normal(0.0, 1.0, size=n_spots),
        ]
    )
    base = rng.gamma(shape=2.0, scale=2.0, size=(n_regions, n_genes))
    library = rng.lognormal(mean=4.7, sigma=0.25, size=n_spots)
    mu = base[region]
    mu = mu / np.maximum(mu.sum(axis=1, keepdims=True), 1e-8) * library[:, None]
    x = rng.poisson(mu).astype(np.float32)
    obs = pd.DataFrame(
        {"spatial_cluster": [f"region_{int(i)}" for i in region]},
        index=[f"spot_{i}" for i in range(n_spots)],
    )
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])
    adata = ad.AnnData(X=sparse.csr_matrix(x), obs=obs, var=var)
    adata.obsm["spatial"] = coords.astype(np.float64)
    return adata


def dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def assert_true(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def main() -> int:
    adata = make_toy_adata(n_spots=96, n_genes=32, n_regions=3, seed=23)
    x = np.rint(dense(adata.X)).astype(float)
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    model = SPARsimGMHPython(
        n_regions=3,
        min_region_size=10,
        finite_pool_factor=20.0,
        library_size_mode="empirical",
        dropout_match=True,
        coord_sigma_factor=0.10,
        random_seed=23,
    )
    model.fit(x, gene_names=list(map(str, adata.var_names)), coords=coords)

    assert_true("condition params created", len(model.condition_params) >= 2)
    assert_true("condition probabilities sum to 1", np.isclose(float(model.condition_probs.sum()), 1.0))
    assert_true("gene names preserved", model.gene_names == list(map(str, adata.var_names)))
    for params in model.condition_params:
        assert_true(f"{params.condition_id} gamma shape finite", bool(np.isfinite(params.gamma_shape).all()))
        assert_true(f"{params.condition_id} gamma scale finite", bool(np.isfinite(params.gamma_scale).all()))
        assert_true(f"{params.condition_id} mean proportion normalized", np.isclose(float(params.mean_proportion.sum()), 1.0))
        assert_true(f"{params.condition_id} dropout bounded", bool(((params.dropout_prob >= 0) & (params.dropout_prob <= model.dropout_quantile)).all()))
        assert_true(f"{params.condition_id} empirical libraries", len(params.empirical_libraries) == params.n_spots)

    syn_x, syn_coords, syn_conditions = model.generate(128, random_seed=24)
    assert_true("synthetic shape", syn_x.shape == (128, adata.n_vars))
    assert_true("synthetic finite", bool(np.isfinite(syn_x).all()))
    assert_true("synthetic nonnegative", bool(syn_x.min() >= 0))
    assert_true("synthetic integer counts", bool(np.allclose(syn_x, np.rint(syn_x))))
    assert_true("synthetic has counts", bool(syn_x.sum() > 0))
    assert_true("spatial shape", syn_coords.shape == (128, 2))
    assert_true("spatial finite", bool(np.isfinite(syn_coords).all()))
    assert_true("spatial varies", bool(np.any(syn_coords.std(axis=0) > 0)))
    assert_true("conditions recorded", syn_conditions.shape == (128,) and len(set(map(str, syn_conditions))) >= 2)

    # GMH draws are constrained by sampled library sizes. Dropout can only lower
    # the total, so generated library sizes should stay within the reference
    # empirical range up to the small lognormal jitter used in empirical mode.
    ref_lib = x.sum(axis=1)
    syn_lib = syn_x.sum(axis=1)
    assert_true("synthetic library finite", bool(np.isfinite(syn_lib).all()))
    assert_true("synthetic library scale", bool(np.percentile(syn_lib, 95) <= max(ref_lib.max() * 2.5, 1.0)))

    restored = SPARsimGMHPython.from_dict(model.to_dict())
    syn_x2, syn_coords2, syn_conditions2 = restored.generate(40, random_seed=25)
    assert_true("restored synthetic shape", syn_x2.shape == (40, adata.n_vars))
    assert_true("restored finite", bool(np.isfinite(syn_x2).all() and np.isfinite(syn_coords2).all()))
    assert_true("restored integer counts", bool(np.allclose(syn_x2, np.rint(syn_x2))))
    assert_true("restored conditions", syn_conditions2.shape == (40,))

    stats = {
        "n_condition_models": len(model.condition_params),
        "condition_probs": model.condition_probs.round(4).tolist(),
        "synthetic_shape": list(syn_x.shape),
        "library_mean": float(syn_lib.mean()),
        "zero_fraction": float((syn_x == 0).mean()),
        "spatial_sd": syn_coords.std(axis=0).round(4).tolist(),
        "conditions": sorted(set(map(str, syn_conditions))),
    }
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
