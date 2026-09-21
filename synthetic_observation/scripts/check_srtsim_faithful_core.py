#!/usr/bin/env python
"""Mechanism checks for the faithful SRTsim Python core.

These tests exercise the published SRTsim design on tiny synthetic arrays and
cover its core generation behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "02_generators" / "_faithful"))

from srtsim_core import FittedBlock, SRTsimPython  # noqa: E402


def check_rescue_single_zero_gene() -> None:
    block = FittedBlock(
        params=[None],
        gene_sel1=np.array([0], dtype=int),
        gene_sel2=np.array([], dtype=int),
        n_cell=5,
        n_read=0.0,
    )
    out = np.zeros((1, 12), dtype=float)
    SRTsimPython._rescue_all_zero_genes(out, block, np.random.default_rng(1), max_nonzero=4)
    assert int(out.sum()) == 1, "single all-zero fitted gene must rescue exactly one location"


def check_rescue_multiple_zero_genes() -> None:
    block = FittedBlock(
        params=[None, None],
        gene_sel1=np.array([0, 1], dtype=int),
        gene_sel2=np.array([], dtype=int),
        n_cell=5,
        n_read=0.0,
    )
    out = np.zeros((2, 12), dtype=float)
    SRTsimPython._rescue_all_zero_genes(out, block, np.random.default_rng(1), max_nonzero=3)
    assert np.all(out.sum(axis=1) == 3), "multiple all-zero fitted genes must use max_nonzero"


def check_rank_preserving_output_shape() -> None:
    x = np.array(
        [
            [0, 1, 4],
            [1, 0, 3],
            [0, 2, 0],
            [3, 0, 1],
            [2, 1, 0],
        ],
        dtype=float,
    )
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
    model = SRTsimPython(sim_scheme="tissue", random_seed=7, maxiter=20).fit(
        x, gene_names=["g1", "g2", "g3"]
    )
    generated = model.generate(x, ref_coords=coords, new_coords=coords, random_seed=8)
    assert generated.shape == x.shape
    assert np.issubdtype(generated.dtype, np.integer)
    assert np.isfinite(generated).all()


def main() -> None:
    check_rescue_single_zero_gene()
    check_rescue_multiple_zero_genes()
    check_rank_preserving_output_shape()
    print("SRTsim faithful core checks: PASS")


if __name__ == "__main__":
    main()
