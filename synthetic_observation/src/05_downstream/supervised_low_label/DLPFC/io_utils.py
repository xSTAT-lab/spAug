"""I/O helpers for the standalone DLPFC supervised low-label workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

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


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def fraction_tag(label_fraction: float) -> str:
    return f"fraction{int(round(float(label_fraction) * 100))}"


def safe_token(value: str | int | float) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return token.strip("_") or "value"


def ensure_columns(adata: ad.AnnData, columns: list[str], context: str):
    missing = [col for col in columns if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"{context} is missing obs columns: {missing}")


def dense_array(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    arr = np.asarray(x)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def read_split(path: str | Path) -> ad.AnnData:
    path = resolve_path(path)
    adata = ad.read_h5ad(path)
    ensure_columns(adata, ["slice_id", "label", "split"], str(path))
    if "spatial" not in adata.obsm:
        raise ValueError(f"{path} is missing obsm['spatial']")
    return adata


def write_json(path: str | Path, payload: dict[str, Any]):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sanitize_obs(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    for col in adata.obs.columns:
        series = adata.obs[col]
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            adata.obs[col] = series.astype(str)
    return adata


def pool_tag(pool_ratio: float) -> str:
    ratio = float(pool_ratio)
    text = str(int(ratio)) if ratio.is_integer() else f"{ratio:g}"
    return f"pool_{text}x"


def synthetic_pool_path(
    output_root: str | Path,
    model: str,
    label_fraction: float,
    seed: int,
    pool_ratio: float = 40.0,
) -> Path:
    tag = pool_tag(pool_ratio)
    return (
        resolve_path(output_root)
        / "supervised_low_label"
        / "DLPFC"
        / model
        / fraction_tag(label_fraction)
        / f"seed{seed}"
        / tag
        / f"synthetic_{tag}.h5ad"
    )


def downstream_root(results_root: str | Path, model: str, label_fraction: float, seed: int) -> Path:
    return (
        resolve_path(results_root)
        / "supervised_low_label"
        / "DLPFC"
        / model
        / fraction_tag(label_fraction)
        / f"seed{seed}"
    )
