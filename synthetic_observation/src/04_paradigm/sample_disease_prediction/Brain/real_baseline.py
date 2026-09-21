"""Real-only sample package."""

from __future__ import annotations

import anndata as ad
import numpy as np

from .common import package_weights, real_with_source


def build_package(real_sample: ad.AnnData, synthetic_sample: ad.AnnData | None = None, ratio=None, alpha=None, seed: int = 0):
    package = real_with_source(real_sample)
    return package, package_weights(package), {"n_real": int(real_sample.n_obs), "n_synthetic": 0}
