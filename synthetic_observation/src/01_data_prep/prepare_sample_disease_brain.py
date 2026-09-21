"""Prepare Brain sample-level disease prediction inputs.

The Brain HEST subset contains several disease/state sources. For the
sample-level disease prediction benchmark we use a clean full-transcriptome
subset: EPM tumor samples versus spatialLIBD healthy DLPFC samples. GBM and
targeted-panel cerebellum/brain samples are excluded from this first Brain
sample-level task to avoid mixing very small subtypes and targeted panels.
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
    if update.upper() == "EPM" or oncotree.upper() == "EPM":
        return "EPM"
    if update.lower() == "healthy" or state.lower() == "healthy":
        return "Healthy"
    return update or state


def disease_state_binary(row: pd.Series) -> str:
    state = str(row.get("disease_state", "") or "").strip()
    if state.lower() == "cancer":
        return "Cancer"
    if state.lower() == "healthy":
        return "Healthy"
    label = disease_label(row)
    return "Cancer" if label == "EPM" else label


def is_selected_brain_sample(row: pd.Series) -> bool:
    update = str(row.get("disease_update", "") or "").strip()
    title = str(row.get("dataset_title", "") or "").strip()
    if update == "EPM":
        return True
    return update == "Healthy" and title == "spatialLIBD"


def dense_or_sparse_copy(x):
    return x.copy() if sp.issparse(x) else np.asarray(x).copy()


def sample_h5ad_path(raw_root: Path, sample_id: str) -> Path:
    path = raw_root / "slices_h5ad" / f"{sample_id}.h5ad"
    if not path.exists():
        path = raw_root / "slices_h5ad" / f"{sample_id}_adata.h5ad"
    return path


def collect_common_genes(paths: list[Path]) -> pd.Index:
    common: pd.Index | None = None
    for path in paths:
        a = ad.read_h5ad(path, backed="r")
        try:
            genes = pd.Index(a.var_names.astype(str))
            common = genes if common is None else common.intersection(genes)
        finally:
            a.file.close()
    if common is None or len(common) == 0:
        raise ValueError("No common genes across selected Brain samples")
    return common


def select_top_genes(paths: list[Path], common: pd.Index, max_genes: int | None) -> pd.Index:
    if max_genes is None or max_genes <= 0 or len(common) <= max_genes:
        return common
    # Keep data preparation bounded. Computing a detection-frequency ranking
    # over all 31k common genes requires repeatedly materializing large backed
    # slices and is slow on this HEST subset. The downstream task uses
    # a stable common gene space, so we take the first max_genes genes in the
    # common-gene order inherited from the first selected sample.
    return common[: int(max_genes)]


def read_one_sample(path: Path, meta_row: pd.Series, genes: pd.Index | None = None) -> ad.AnnData:
    sample = str(meta_row["id"])
    if genes is None:
        a = ad.read_h5ad(path)
    else:
        backed = ad.read_h5ad(path, backed="r")
        try:
            a = backed[:, genes.tolist()].to_memory()
        finally:
            backed.file.close()
    if "in_tissue" in a.obs.columns:
        mask = a.obs["in_tissue"].astype(str).isin(["1", "True", "true"])
        if bool(mask.any()):
            a = a[mask].copy()
    a.var_names_make_unique()
    obs = a.obs.copy()
    obs["sample_id"] = sample
    obs["slice_id"] = sample
    obs["disease_label"] = disease_label(meta_row)
    obs["disease_state_binary"] = disease_state_binary(meta_row)
    obs["disease_update"] = "" if pd.isna(meta_row.get("disease_update")) else str(meta_row.get("disease_update"))
    obs["disease_state_raw"] = str(meta_row.get("disease_state", ""))
    obs["oncotree_code"] = "" if pd.isna(meta_row.get("oncotree_code")) else str(meta_row.get("oncotree_code"))
    obs["patient"] = "" if pd.isna(meta_row.get("patient")) else str(meta_row.get("patient"))
    obs["dataset_title"] = str(meta_row.get("dataset_title", ""))
    obs["organ"] = str(meta_row.get("organ", "Brain"))
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
    parser.add_argument("--raw-root", default="data/01_raw/Brain")
    parser.add_argument("--output", default="data/02_interim/sample_disease_prediction/Brain/brain_samples.h5ad")
    parser.add_argument("--splits-output", default="data/02_interim/sample_disease_prediction/Brain/splits.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-genes", type=int, default=2000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    raw_root = resolve_path(args.raw_root)
    meta = pd.read_csv(raw_root / "meta_df_t_update.csv")
    meta = meta[meta.apply(is_selected_brain_sample, axis=1)].copy()
    meta["disease_label"] = meta.apply(disease_label, axis=1)
    meta["disease_state_binary"] = meta.apply(disease_state_binary, axis=1)
    meta = meta[meta["disease_label"].isin(["EPM", "Healthy"])].copy()
    meta = meta.sort_values("id").reset_index(drop=True)
    if args.quick and args.max_samples is None:
        args.max_samples = 4
    if args.max_samples is not None:
        pieces = []
        per_label = max(1, int(np.ceil(args.max_samples / 2)))
        for _, g in meta.groupby("disease_label", sort=True):
            pieces.append(g.head(per_label))
        meta = pd.concat(pieces, axis=0).sort_values("id").head(args.max_samples).reset_index(drop=True)

    missing = []
    sample_paths = []
    for _, row in meta.iterrows():
        path = sample_h5ad_path(raw_root, str(row["id"]))
        if not path.exists():
            missing.append(str(path))
            continue
        sample_paths.append(path)
    if missing:
        raise FileNotFoundError(f"Missing h5ad files: {missing[:5]}")
    if not sample_paths:
        raise ValueError("No Brain samples were loaded")

    common = collect_common_genes(sample_paths)
    selected_genes = select_top_genes(sample_paths, common, args.max_genes)
    adatas = []
    for path, (_, row) in zip(sample_paths, meta.iterrows()):
        adatas.append(read_one_sample(path, row, genes=selected_genes))
    combined = ad.concat(adatas, axis=0, join="inner", merge="same", label=None)
    combined.var["highly_variable"] = True
    combined.uns["dataset"] = "Brain"
    combined.uns["task"] = "sample_disease_prediction"
    combined.uns["label_level"] = "sample"
    combined.uns["disease_labels"] = sorted(combined.obs["disease_label"].astype(str).unique().tolist())
    combined.uns["disease_state_binary_labels"] = sorted(
        combined.obs["disease_state_binary"].astype(str).unique().tolist()
    )
    combined.uns["n_samples"] = int(combined.obs["sample_id"].nunique())
    combined.uns["selection_rule"] = "EPM tumor samples plus spatialLIBD healthy full-transcriptome samples"
    combined.uns["excluded_brain_samples"] = "GBM and targeted-panel cerebellum/brain samples"

    out = resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(out)

    samples = combined.obs[
        ["sample_id", "disease_label", "disease_state_binary", "patient", "dataset_title"]
    ].drop_duplicates("sample_id")
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
        "disease_state_binary_counts": samples["disease_state_binary"].value_counts().to_dict(),
        "selection_rule": combined.uns["selection_rule"],
    }
    (out.parent / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
