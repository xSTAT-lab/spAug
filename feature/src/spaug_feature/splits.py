"""Stratified within-slice label splits."""
from __future__ import annotations
import numpy as np
import pandas as pd
import anndata as ad


def make_low_label_split(
    adata: ad.AnnData,
    label_fraction: float,
    seed: int,
    slice_col: str = "slice_id",
    label_col: str = "label",
) -> tuple[ad.AnnData, pd.DataFrame]:
    if not 0 < label_fraction < 1:
        raise ValueError("label_fraction must be in (0, 1)")
    if slice_col not in adata.obs.columns:
        raise ValueError(f"AnnData is missing obs[{slice_col!r}]")
    if label_col not in adata.obs.columns:
        raise ValueError(f"AnnData is missing obs[{label_col!r}]")

    rng = np.random.default_rng(seed)
    obs = adata.obs[[slice_col, label_col]].astype(str).copy()
    split = pd.Series("test", index=adata.obs_names.astype(str), dtype="object")
    rows = []

    for (slice_id, label), names in obs.groupby([slice_col, label_col], sort=True).groups.items():
        names = np.asarray(list(names), dtype=object)
        n_total = int(names.size)
        if n_total < 2:
            rows.append(
                {
                    "slice_id": str(slice_id),
                    "label": str(label),
                    "n_total": n_total,
                    "n_train": 0,
                    "n_test": n_total,
                    "status": "too_few_for_train_test",
                }
            )
            continue
        n_train = int(round(n_total * label_fraction))
        n_train = max(1, n_train)
        n_train = min(n_train, n_total - 1)
        chosen = rng.choice(names, size=n_train, replace=False)
        split.loc[chosen.astype(str)] = "train"
        rows.append(
            {
                "slice_id": str(slice_id),
                "label": str(label),
                "n_total": n_total,
                "n_train": int(n_train),
                "n_test": int(n_total - n_train),
                "status": "ok",
            }
        )

    out = adata.copy()
    out.obs["split"] = split.reindex(out.obs_names.astype(str)).to_numpy()
    out.obs["low_label_fraction"] = float(label_fraction)
    out.obs["low_label_seed"] = int(seed)
    out.uns["dataset"] = "DLPFC"
    out.uns["split_mode"] = "supervised"
    out.uns["task_mode"] = "low_label"
    out.uns["label_fraction"] = float(label_fraction)
    out.uns["low_label_seed"] = int(seed)
    out.uns["low_label_split_policy"] = "slice_label_fraction"
    return out, pd.DataFrame(rows)
