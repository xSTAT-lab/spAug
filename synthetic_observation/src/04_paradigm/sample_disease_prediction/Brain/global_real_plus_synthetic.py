"""Real plus same-sample synthetic spots."""

from __future__ import annotations

import anndata as ad

from .common import concat_packages, package_weights, real_with_source, sample_synthetic


def build_package(real_sample: ad.AnnData, synthetic_sample: ad.AnnData | None, ratio: float, alpha=None, seed: int = 0):
    n_syn = max(1, int(round(real_sample.n_obs * float(ratio))))
    syn = sample_synthetic(synthetic_sample, n=n_syn, seed=seed)
    syn.obs["augmentation_source"] = "synthetic"
    package = concat_packages([real_with_source(real_sample), syn])
    return package, package_weights(package), {"n_real": int(real_sample.n_obs), "n_synthetic": int(syn.n_obs)}
