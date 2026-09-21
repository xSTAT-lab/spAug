"""Weighted real/synthetic sample package.

The weights are used during sample feature aggregation:
real spots have total mass 1-alpha and synthetic spots have total mass alpha.
"""

from __future__ import annotations

import anndata as ad

from .common import concat_packages, package_weights, real_with_source, sample_synthetic


def build_package(real_sample: ad.AnnData, synthetic_sample: ad.AnnData | None, ratio: float, alpha: float, seed: int = 0):
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if alpha == 0.0:
        package = real_with_source(real_sample)
        return package, package_weights(package, alpha=0.0), {"n_real": int(real_sample.n_obs), "n_synthetic": 0}
    n_syn = max(1, int(round(real_sample.n_obs * float(ratio))))
    syn = sample_synthetic(synthetic_sample, n=n_syn, seed=seed)
    syn.obs["augmentation_source"] = "synthetic"
    package = concat_packages([real_with_source(real_sample), syn])
    return package, package_weights(package, alpha=alpha), {"n_real": int(real_sample.n_obs), "n_synthetic": int(syn.n_obs)}
