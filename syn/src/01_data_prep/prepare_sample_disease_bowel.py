"""Prepare Bowel sample-level disease prediction inputs.

The Bowel HEST subset has sample-level disease labels but no spot-level
manual labels. This script builds one task-specific AnnData with sample
metadata and a separate sample-level cross-validation split table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def disease_label(row: pd.Series) -> str:
    update = str(row.get("disease_update", "") or "").strip()
    oncotree = str(row.get("oncotree_code", "") or "").strip()
    state = str(row.get("disease_state", "") or "").strip()
    cancer_codes = {"COAD", "READ", "COADREAD"}
    if update.upper() in cancer_codes or oncotree.upper() in cancer_codes or state.lower() == "cancer":
        return "Cancer"
    if update.lower() == "healthy" or state.lower() == "healthy":
        return "Healthy"
    return update or state


def cancer_subtype(row: pd.Series) -> str:
    update = str(row.get("disease_update", "") or "").strip()
    oncotree = str(row.get("oncotree_code", "") or "").strip()
    for value in (update, oncotree):
        if value.upper() in {"COAD", "READ", "COADREAD"}:
            return value.upper()
    return ""


def dense_or_sparse_copy(x):
    return x.copy() if sp.issparse(x) else np.asarray(x).copy()


def read_one_sample(path: Path, meta_row: pd.Series) -> ad.AnnData:
    sample = str(meta_row["id"])
    a = ad.read_h5ad(path)
    if "in_tissue" in a.obs.columns:
        mask = a.obs["in_tissue"].astype(str).isin(["1", "True", "true"])
        if bool(mask.any()):
            a = a[mask].copy()
    a.var_names_make_unique()
    obs = a.obs.copy()
    obs["sample_id"] = sample
    obs["slice_id"] = sample
    obs["disease_label"] = disease_label(meta_row)
    obs["disease_update"] = "" if pd.isna(meta_row.get("disease_update")) else str(meta_row.get("disease_update"))
    obs["cancer_subtype"] = cancer_subtype(meta_row)
    obs["disease_state_raw"] = str(meta_row.get("disease_state", ""))
    obs["oncotree_code"] = "" if pd.isna(meta_row.get("oncotree_code")) else str(meta_row.get("oncotree_code"))
    obs["patient"] = "" if pd.isna(meta_row.get("patient")) else str(meta_row.get("patient"))
    obs["dataset_title"] = str(meta_row.get("dataset_title", ""))
    obs["organ"] = str(meta_row.get("organ", "Bowel"))
    obs["st_technology"] = str(meta_row.get("st_technology", ""))
    obs["tissue"] = "" if pd.isna(meta_row.get("tissue")) else str(meta_row.get("tissue"))
    obs["subseries"] = "" if pd.isna(meta_row.get("subseries")) else str(meta_row.get("subseries"))
    obs["source"] = "real"
    obs.index = [f"{sample}:{idx}" for idx in obs.index.astype(str)]
    out = ad.AnnData(X=dense_or_sparse_copy(a.X), obs=obs, var=a.var.copy())
    if "spatial" not in a.obsm:
        raise ValueError(f"{path} has no obsm['spatial']")
    out.obsm["spatial"] = np.asarray(a.obsm["spatial"][:, :2], dtype=np.float64)
    return out


def make_splits(samples: pd.DataFrame, n_splits: int, seed: int, quick: bool = False) -> pd.DataFrame:
    n_splits = max(2, min(int(n_splits), int(samples["disease_label"].value_counts().min())))
    x = samples["sample_id"].to_numpy()
    y = samples["disease_label"].to_numpy()
    patient = samples["patient"].astype("string").fillna("").astype(str).str.strip()
    patient_values = patient.to_numpy()
    groups = np.where((patient_values == "") | (patient_values == "nan"), x, patient_values)
    labels = set(pd.Series(y).astype(str).unique())

    def build_rows(split_iter, splitter_name: str) -> pd.DataFrame:
        rows = []
        for fold, (train_idx, test_idx) in enumerate(split_iter):
            for idx in train_idx:
                rows.append({
                    "fold": fold,
                    "sample_id": x[idx],
                    "split": "train",
                    "disease_label": y[idx],
                    "patient_group": groups[idx],
                    "splitter": splitter_name,
                })
            for idx in test_idx:
                rows.append({
                    "fold": fold,
                    "sample_id": x[idx],
                    "split": "test",
                    "disease_label": y[idx],
                    "patient_group": groups[idx],
                    "splitter": splitter_name,
                })
            if quick and fold >= 1:
                break
        return pd.DataFrame(rows)

    def has_all_labels(df: pd.DataFrame) -> bool:
        for _, fold_df in df.groupby("fold"):
            for split in ("train", "test"):
                observed = set(fold_df.loc[fold_df["split"] == split, "disease_label"].astype(str).unique())
                if observed != labels:
                    return False
        return True

    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        out = build_rows(splitter.split(x, y, groups=groups), "StratifiedGroupKFold")
        if has_all_labels(out):
            return out
    except Exception:
        pass

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = build_rows(splitter.split(x, y), "StratifiedKFold")
    if not has_all_labels(out):
        raise ValueError("Could not create stratified folds with both labels in every train/test split")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/01_raw/Bowel")
    parser.add_argument("--output", default="data/02_interim/sample_disease_prediction/Bowel/bowel_samples.h5ad")
    parser.add_argument("--splits-output", default="data/02_interim/sample_disease_prediction/Bowel/splits.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-genes", type=int, default=2000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    raw_root = resolve_path(args.raw_root)
    meta = pd.read_csv(raw_root / "meta_df_t_update.csv")
    meta["disease_label"] = meta.apply(disease_label, axis=1)
    meta["cancer_subtype"] = meta.apply(cancer_subtype, axis=1)
    meta = meta[meta["disease_label"].isin(["Cancer", "Healthy"])].copy()
    meta = meta.sort_values("id").reset_index(drop=True)
    if args.quick and args.max_samples is None:
        args.max_samples = 4
    if args.max_samples is not None:
        pieces = []
        per_label = max(1, int(np.ceil(args.max_samples / 2)))
        for _, g in meta.groupby("disease_label", sort=True):
            pieces.append(g.head(per_label))
        meta = pd.concat(pieces, axis=0).sort_values("id").head(args.max_samples).reset_index(drop=True)

    adatas = []
    missing = []
    for _, row in meta.iterrows():
        path = raw_root / "slices_h5ad" / f"{row['id']}.h5ad"
        if not path.exists():
            path = raw_root / "slices_h5ad" / f"{row['id']}_adata.h5ad"
        if not path.exists():
            missing.append(str(path))
            continue
        adatas.append(read_one_sample(path, row))
    if missing:
        raise FileNotFoundError(f"Missing h5ad files: {missing[:5]}")
    if not adatas:
        raise ValueError("No Bowel samples were loaded")

    common = adatas[0].var_names
    for a in adatas[1:]:
        common = common.intersection(a.var_names)
    if len(common) == 0:
        raise ValueError("No common genes across selected Bowel samples")
    adatas = [a[:, common].copy() for a in adatas]
    combined = ad.concat(adatas, axis=0, join="inner", merge="same", label=None)
    if args.max_genes is not None and args.max_genes > 0 and combined.n_vars > args.max_genes:
        gene_score = (
            np.asarray((combined.X > 0).sum(axis=0)).ravel()
            if sp.issparse(combined.X)
            else (np.asarray(combined.X) > 0).sum(axis=0)
        )
        keep = np.argsort(gene_score)[::-1][: int(args.max_genes)]
        keep = np.sort(keep)
        combined = combined[:, keep].copy()
    combined.var["highly_variable"] = True
    combined.uns["dataset"] = "Bowel"
    combined.uns["task"] = "sample_disease_prediction"
    combined.uns["label_level"] = "sample"
    combined.uns["disease_labels"] = sorted(combined.obs["disease_label"].astype(str).unique().tolist())
    combined.uns["n_samples"] = int(combined.obs["sample_id"].nunique())

    out = resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(out)

    samples = combined.obs[["sample_id", "disease_label", "patient", "dataset_title", "cancer_subtype"]].drop_duplicates("sample_id")
    splits = make_splits(samples, n_splits=args.n_splits, seed=args.random_seed, quick=args.quick)
    split_out = resolve_path(args.splits_output)
    split_out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(split_out, index=False)
    manifest = {
        "output": str(out),
        "splits_output": str(split_out),
        "n_obs": int(combined.n_obs),
        "n_vars": int(combined.n_vars),
        "n_samples": int(samples.shape[0]),
        "label_counts": samples["disease_label"].value_counts().to_dict(),
    }
    (out.parent / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
