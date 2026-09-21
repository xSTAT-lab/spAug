#!/usr/bin/env python
"""Regression checks for the faithful Python Splatter core."""

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

from splatter_core import SpatialSplatterPython  # noqa: E402


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
    adata = make_toy_adata(n_spots=90, n_genes=30, n_regions=3, seed=11)
    x = dense(adata.X)
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    model = SpatialSplatterPython(n_regions=3, min_region_size=10, coord_sigma_factor=0.10, random_seed=11)
    model.fit(x, gene_names=list(map(str, adata.var_names)), coords=coords)

    assert_true("region models created", len(model.region_models) >= 2)
    assert_true("region probabilities sum to 1", np.isclose(float(model.region_probs.sum()), 1.0))
    assert_true("gene names preserved", model.gene_names == list(map(str, adata.var_names)))

    syn_x, syn_coords, syn_regions = model.generate(120, random_seed=12)
    assert_true("synthetic shape", syn_x.shape == (120, adata.n_vars))
    assert_true("synthetic finite", bool(np.isfinite(syn_x).all()))
    assert_true("synthetic nonnegative", bool(syn_x.min() >= 0))
    assert_true("synthetic has counts", bool(syn_x.sum() > 0))
    assert_true("spatial shape", syn_coords.shape == (120, 2))
    assert_true("spatial finite", bool(np.isfinite(syn_coords).all()))
    assert_true("spatial varies", bool(np.any(syn_coords.std(axis=0) > 0)))
    assert_true("regions recorded", syn_regions.shape == (120,) and len(set(map(str, syn_regions))) >= 2)

    restored = SpatialSplatterPython.from_dict(model.to_dict())
    syn_x2, syn_coords2, syn_regions2 = restored.generate(40, random_seed=13)
    assert_true("restored synthetic shape", syn_x2.shape == (40, adata.n_vars))
    assert_true("restored finite", bool(np.isfinite(syn_x2).all() and np.isfinite(syn_coords2).all()))
    assert_true("restored regions", syn_regions2.shape == (40,))

    stats = {
        "n_region_models": len(model.region_models),
        "region_probs": model.region_probs.round(4).tolist(),
        "synthetic_shape": list(syn_x.shape),
        "library_mean": float(syn_x.sum(axis=1).mean()),
        "zero_fraction": float((syn_x == 0).mean()),
        "spatial_sd": syn_coords.std(axis=0).round(4).tolist(),
    }
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
