"""
src/01_data_prep/split_train_test.py
====================================
Train/validation/test split module

The following rules are enforced:
1. supervised learning (Task1, Task2): split at slice/sample level; all spots from one slice stay in the same set
2. unsupervised learning (Task3): use all data and mark split='all'
3. never generate from the test set (data leakage prevention)

input: data/02_interim/{dataset}/processed*.h5ad (output from normalize.py)
output: update AnnData obs['split'] and save split index files

splitmode:
  - slice_split: DLPFC split by slice ID
  - sample_split: Trastuzumab split by sample ID (stratified)
  - all: clustering task; mark all observations as 'all'
"""

import os
import tempfile
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import Counter

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "spaug_numba_cache"))

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.model_selection import StratifiedShuffleSplit
import yaml

# ============================================================
# Logging and helper functions
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("split")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml_config(filename: str) -> dict:
    """Load YAML configuration from configs/."""
    filepath = PROJECT_ROOT / "configs" / filename
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============================================================
# split strategy
# ============================================================

def slice_level_split(
    adata: ad.AnnData,
    slice_col: str = "slice_id",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    stratify_col: Optional[str] = "label",
    random_seed: int = 42,
) -> pd.Series:
    """
    slice split (DLPFC)

    split by slice ID so all spots from each slice stay together.
     preserve class balance in training and test sets using the dominant label per slice.

    Args:
        adata:  processed AnnData
        slice_col: slice identifier column in obs 
        train_ratio / val_ratio / test_ratio: fractions assigned to each split
        stratify_col: stratified column (optional)
        random_seed: random seed

    Returns:
        pd.Series: split label for each observation ('train' / 'val' / 'test')
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "train_ratio + val_ratio + test_ratio must equal 1.0"

    slice_ids = adata.obs[slice_col].unique()
    n_slices = len(slice_ids)
    logger.info(f"slice split: {n_slices} slices, ratio {train_ratio}/{val_ratio}/{test_ratio}")

    # when labels are available, compute the dominant label for each slice
    if stratify_col and stratify_col in adata.obs.columns:
        slice_labels = {}
        for sid in slice_ids:
            mask = adata.obs[slice_col] == sid
            label_counts = adata.obs.loc[mask, stratify_col].value_counts()
            slice_labels[sid] = label_counts.index[0] if len(label_counts) > 0 else "unknown"
        stratify_values = pd.Series(slice_labels)
    else:
        stratify_values = None

    # split 
    rng = np.random.RandomState(random_seed)

    if stratify_values is not None and len(stratify_values.unique()) > 1:
        # stratify slices by their dominant labels and shuffle within each stratum 
        split_assignments = {}
        for label in stratify_values.unique():
            label_slices = stratify_values[stratify_values == label].index.tolist()
            rng.shuffle(label_slices)
            n = len(label_slices)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio)) if n > 2 else 0
            # Allocate slice groups within the current label stratum.
            n_test = n - n_train - n_val

            for s in label_slices[:n_train]:
                split_assignments[s] = "train"
            for s in label_slices[n_train:n_train + n_val]:
                split_assignments[s] = "val"
            for s in label_slices[n_train + n_val:]:
                split_assignments[s] = "test"
    else:
        # Shuffle groups using the configured random seed.
        shuffled = list(slice_ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0

        split_assignments = {}
        for s in shuffled[:n_train]:
            split_assignments[s] = "train"
        for s in shuffled[n_train:n_train + n_val]:
            split_assignments[s] = "val"
        for s in shuffled[n_train + n_val:]:
            split_assignments[s] = "test"

    # Map group assignments to observations.
    splits = adata.obs[slice_col].map(split_assignments)

    # statistics
    _log_split_stats(splits, adata, slice_col)
    return splits


def sample_level_split(
    adata: ad.AnnData,
    sample_col: str = "sample_id",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_col: str = "label",
    random_seed: int = 42,
) -> pd.Series:
    """
    Sample-level split for Trastuzumab.

    Group observations by sample ID and optionally stratify by sample label.
     All observations from one sample share a split assignment.

    Args:
        adata:  processed AnnData
        sample_col: sample identifier column in obs 
        stratify_col: label column used for stratification

    Returns:
        pd.Series: split label for each observation
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "train_ratio + val_ratio + test_ratio must equal 1.0"

    # get one label per sample
    sample_info = adata.obs[[sample_col]].drop_duplicates()
    if stratify_col and stratify_col in adata.obs.columns:
        # use a representative spot label for each sample
        sample_labels = adata.obs.drop_duplicates(subset=[sample_col])[[sample_col, stratify_col]]
        sample_labels = sample_labels.set_index(sample_col)[stratify_col]
    else:
        sample_labels = None

    sample_ids = sample_info[sample_col].unique()
    n_samples = len(sample_ids)
    logger.info(f"sample split: {n_samples} sample records, ratio {train_ratio}/{val_ratio}/{test_ratio}")

    rng = np.random.RandomState(random_seed)

    if sample_labels is not None and len(sample_labels.unique()) > 1:
        # stratifiedsplit: use StratifiedShuffleSplit
        X_dummy = np.zeros((n_samples, 1))
        y = sample_labels.reindex(sample_ids).values

        # first  test
        sss1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_ratio,
            random_state=random_seed,
        )
        train_val_idx, test_idx = next(sss1.split(X_dummy, y))

        #  from train_val in  val
        remaining_ratio = val_ratio / (train_ratio + val_ratio)
        sss2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=remaining_ratio,
            random_state=random_seed,
        )
        train_idx_in_remaining, val_idx_in_remaining = next(
            sss2.split(X_dummy[train_val_idx], y[train_val_idx])
        )

        train_idx = train_val_idx[train_idx_in_remaining]
        val_idx = train_val_idx[val_idx_in_remaining]

        split_map = {}
        for i in train_idx:
            split_map[sample_ids[i]] = "train"
        for i in val_idx:
            split_map[sample_ids[i]] = "val"
        for i in test_idx:
            split_map[sample_ids[i]] = "test"
    else:
        #  randomsplit
        shuffled = list(sample_ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0

        split_map = {}
        for s in shuffled[:n_train]:
            split_map[s] = "train"
        for s in shuffled[n_train:n_train + n_val]:
            split_map[s] = "val"
        for s in shuffled[n_train + n_val:]:
            split_map[s] = "test"

    # Map group assignments to observations.
    splits = adata.obs[sample_col].map(split_map)

    _log_split_stats(splits, adata, sample_col)
    return splits


def all_split(adata: ad.AnnData) -> pd.Series:
    """
    clustering task: use all observations and mark split=all

    Returns:
        pd.Series: all as 'all'
    """
    logger.info(f"clustering mode: all data ({adata.n_obs} spots)")
    return pd.Series(["all"] * adata.n_obs, index=adata.obs_names, name="split")


# ============================================================
# helper functions
# ============================================================

def _log_split_stats(splits: pd.Series, adata: ad.AnnData, group_col: str):
    """ split statistics"""
    logger.info(f"\n--- split statistics ---")
    for split_name in ["train", "val", "test", "all"]:
        mask = splits == split_name
        n_spots = mask.sum()
        if n_spots == 0:
            continue
        n_groups = adata.obs.loc[mask, group_col].nunique()
        logger.info(f"  {split_name}: {n_spots} spots ({n_groups} items {group_col})")

    # summarize the label distribution
    if "label" in adata.obs.columns:
        for split_name in ["train", "val", "test"]:
            mask = splits == split_name
            if mask.sum() == 0:
                continue
            label_dist = adata.obs.loc[mask, "label"].value_counts(normalize=True)
            logger.info(f"  {split_name} label distribution: {dict(label_dist.round(3))}")


LEAKAGE_LABEL_COLUMNS = {
    "label",
    "spatialLIBD",
    "ground_truth",
    "manual_annotation",
    "manual_label",
    "cluster",
    "clusters",
    "refined_label",
    "refined_labels",
    "response",
    "Response",
}

GENERATOR_MODELS = ["SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion"]
MODELS_REQUIRE_COORDS = {"SRTsim", "Splatter", "SPARsim"}
MODELS_COORD_MAPPING = {"scGAN", "scDiffusion"}


def _drop_unsupervised_labels(
    adata: ad.AnnData,
    output_dir: Path,
    dataset_name: str,
) -> ad.AnnData:
    """Remove supervision/evaluation labels from unsupervised training inputs."""
    label_cols = [c for c in LEAKAGE_LABEL_COLUMNS if c in adata.obs.columns]
    if not label_cols:
        return adata

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir.parent / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    labels_path = eval_dir / "labels_for_evaluation.csv"
    labels = adata.obs[label_cols].copy()
    labels.insert(0, "obs_name", adata.obs_names.astype(str))
    labels.to_csv(labels_path, index=False)
    logger.info(
        f"{dataset_name} unsupervised: saved labels to the evaluation-only file: "
        f"{labels_path}"
    )

    clean = adata.copy()
    clean.obs = clean.obs.drop(columns=label_cols)
    logger.info(
        f"{dataset_name} unsupervised: label columns exported for evaluation: {label_cols}"
    )
    return clean


def _minimal_obs_for_generator_view(
    adata: ad.AnnData,
    dataset_name: str,
    split_mode: str,
) -> pd.DataFrame:
    """Build the minimum obs schema that generator inputs are allowed to carry."""
    obs = pd.DataFrame(index=adata.obs_names.copy())

    if dataset_name == "DLPFC":
        if "slice_id" not in adata.obs.columns:
            raise ValueError("DLPFC model-view missing obs['slice_id']")
        obs["slice_id"] = adata.obs["slice_id"].astype(str).values
    elif dataset_name == "Trastuzumab":
        if "sample_id" not in adata.obs.columns:
            raise ValueError("Trastuzumab model-view missing obs['sample_id']")
        obs["sample_id"] = adata.obs["sample_id"].astype(str).values
        if "cohort_id" in adata.obs.columns:
            obs["cohort_id"] = adata.obs["cohort_id"].astype(str).values

    if "split" not in adata.obs.columns:
        raise ValueError("model-view missing obs['split']")
    obs["split"] = adata.obs["split"].astype(str).values

    if split_mode == "supervised":
        if "label" not in adata.obs.columns:
            raise ValueError(f"{dataset_name} supervised model-view missing obs['label']")
        obs["label"] = adata.obs["label"].values

    return obs


def _make_generator_view(
    adata: ad.AnnData,
    dataset_name: str,
    split_mode: str,
    model_name: str,
) -> ad.AnnData:
    """Create a model-specific AnnData input view from canonical split data."""
    view = ad.AnnData(
        X=adata.X.copy(),
        obs=_minimal_obs_for_generator_view(adata, dataset_name, split_mode),
        var=adata.var.copy(),
    )

    # spatial is original observed data. It is mandatory for coordinate-aware models
    # and useful as reference for coordinate assignment even when model training ignores it.
    if "spatial" in adata.obsm:
        view.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float64).copy()
    elif model_name in MODELS_REQUIRE_COORDS or dataset_name == "DLPFC":
        raise ValueError(f"{model_name} model-view missing obsm['spatial']")

    view.uns["dataset"] = dataset_name
    view.uns["split_mode"] = split_mode
    view.uns["generator_model"] = model_name
    view.uns["task"] = adata.uns.get("task", [])
    view.uns["gene_space"] = "highly_variable"
    if dataset_name == "DLPFC" and "slice_ids" in adata.uns:
        view.uns["slice_ids"] = adata.uns["slice_ids"]
    if dataset_name == "Trastuzumab":
        view.uns["normalization"] = adata.uns.get("normalization", "z_score")

    return view


def save_generator_model_views(
    adata: ad.AnnData,
    dataset_name: str,
    split_mode: str,
    interim_dir: Path,
):
    """Export per-generator input AnnData files."""
    models = list(GENERATOR_MODELS)

    if dataset_name == "DLPFC" and split_mode == "unsupervised":
        canonical = interim_dir / split_mode / "processed_with_split.h5ad"
        if not canonical.exists():
            raise FileNotFoundError(f"Missing canonical unsupervised input: {canonical}")
        _validate_generator_ready_adata(adata, dataset_name, split_mode)
        for model_name in models:
            out_dir = interim_dir / "model_inputs" / model_name / split_mode
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "processed_with_split.h5ad"
            if out_path.exists() or out_path.is_symlink():
                out_path.unlink()
            rel_target = os.path.relpath(canonical, start=out_dir)
            out_path.symlink_to(rel_target)
            logger.info(
                f"save {dataset_name}/{model_name}/{split_mode} model-view symbolic link: "
                f"{out_path} -> {rel_target}"
            )
        return

    for model_name in models:
        view = _make_generator_view(adata, dataset_name, split_mode, model_name)
        _validate_generator_ready_adata(view, dataset_name, split_mode)
        out_dir = interim_dir / "model_inputs" / model_name / split_mode
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "processed_with_split.h5ad"
        view.write(out_path)
        logger.info(
            f"save {dataset_name}/{model_name}/{split_mode} model-view: "
            f"{out_path}"
        )


def save_split_info(
    adata: ad.AnnData,
    output_dir: Path,
    dataset_name: str,
    split_mode: str = "supervised",
    split_col: str = "split",
):
    """
    save split information:
    1. save AnnData with the obs['split'] column
    2.   JSON file containing the split index
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if split_mode == "unsupervised":
        adata = _drop_unsupervised_labels(adata, output_dir, dataset_name)

    _validate_generator_ready_adata(adata, dataset_name, split_mode, split_col)

    # Save AnnData with split annotations.
    adata_path = output_dir / f"processed_with_split.h5ad"
    adata.write(adata_path)
    logger.info(f"saved AnnData with split annotations: {adata_path}")

    # Write the split index as JSON.
    split_info = {}
    for split_name in ["train", "val", "test", "all"]:
        mask = adata.obs[split_col] == split_name
        if mask.sum() == 0:
            continue
        split_info[split_name] = {
            "n_spots": int(mask.sum()),
            "spot_indices": adata.obs_names[mask].tolist(),
        }
        # for slice/sample splits, record split information
        if "slice_id" in adata.obs.columns:
            split_info[split_name]["slice_ids"] = (
                adata.obs.loc[mask, "slice_id"].unique().tolist()
            )
        if "sample_id" in adata.obs.columns:
            split_info[split_name]["sample_ids"] = (
                adata.obs.loc[mask, "sample_id"].unique().tolist()
            )

    json_path = output_dir / "split_info.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    logger.info(f"saved split index: {json_path}")

    return adata, adata_path, json_path


def _validate_generator_ready_adata(
    adata: ad.AnnData,
    dataset_name: str,
    split_mode: str = "supervised",
    split_col: str = "split",
):
    """Validate the split AnnData contract expected by src/02_generators."""
    errors = []

    if not adata.obs_names.is_unique:
        errors.append("obs_names must be globally unique spot/sample identifiers")

    required_obs = [split_col]
    if split_mode == "supervised":
        required_obs.append("label")
    else:
        leakage_cols = sorted(c for c in LEAKAGE_LABEL_COLUMNS if c in adata.obs.columns)
        if leakage_cols:
            errors.append(
                "unsupervised contains label/target columns in unsupervised output: "
                f"{leakage_cols}"
            )

    if dataset_name == "DLPFC":
        required_obs.append("slice_id")
    elif dataset_name == "Trastuzumab":
        required_obs.append("sample_id")
    else:
        if "slice_id" not in adata.obs.columns and "sample_id" not in adata.obs.columns:
            errors.append("obs must include slice_id or sample_id")

    for col in required_obs:
        if col not in adata.obs.columns:
            errors.append(f"obs is missing column: {col}")

    if "spatial" not in adata.obsm:
        errors.append("obsm is missing key: spatial")
    else:
        coords = np.asarray(adata.obsm["spatial"])
        if coords.shape != (adata.n_obs, 2):
            errors.append(f"obsm['spatial'] shape ({adata.n_obs}, 2), received {coords.shape}")
        elif not np.isfinite(coords).all():
            errors.append("obsm['spatial'] contains NaN/Inf")

    if "highly_variable" not in adata.var.columns:
        errors.append("var is missing column: highly_variable")
    else:
        hvg = adata.var["highly_variable"].astype(bool)
        if int(hvg.sum()) == 0:
            errors.append("var['highly_variable'] must contain at least one true value")

    if split_col in adata.obs.columns:
        valid_splits = {"train", "val", "test", "all"}
        unknown = set(adata.obs[split_col].dropna().astype(str).unique()) - valid_splits
        if unknown:
            errors.append(f"{split_col} contains unknown value: {sorted(unknown)}")

    if errors:
        msg = ";".join(errors)
        raise ValueError(f"{dataset_name} split output validation details: {msg}")

    logger.info(
        f"generator input contract passed ({split_mode}): obs={adata.n_obs}, "
        f"genes={adata.n_vars}, HVG={int(adata.var['highly_variable'].astype(bool).sum())}"
    )


# ============================================================
#  dataset split configuration defaults
# ============================================================

DEFAULT_SPLIT_CONFIGS = {
    "DLPFC": {
        "supervised": {
            "tasks": ["task1_spot_clf"],
            "method": "slice_level_split",
            "slice_col": "slice_id",
            "stratify_col": "label",
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "input_file": "processed_combined.h5ad",
        },
        "unsupervised": {
            "tasks": ["task3_spatial_cluster"],
            "method": "all_split",
            "input_file": "processed_combined.h5ad",
        },
    },
    "Trastuzumab": {
        "supervised": {
            "tasks": ["task2_wsi_pred"],
            "method": "sample_level_split",
            "sample_col": "sample_id",
            "stratify_col": "label",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "input_file": "processed.h5ad",
        },
    },
}


# ============================================================
# main workflow
# ============================================================

def split_dataset(
    dataset_name: str,
    mode: str = "supervised",
    config: Optional[dict] = None,
    random_seed: int = 42,
    input_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
):
    """
    run a train/test split for each selected dataset

    Args:
        dataset_name: dataset name
        mode: 'supervised' / 'unsupervised' / 'both'
        config:  split configuration that overrides defaults
        random_seed: random seed
    """
    if input_dir is None and dataset_name == "DLPFC" and mode == "unsupervised":
        input_interim_dir = PROJECT_ROOT / "data" / "02_interim" / "common" / "DLPFC"
    else:
        input_interim_dir = Path(input_dir) if input_dir is not None else PROJECT_ROOT / "data" / "02_interim" / dataset_name

    if output_dir is None and dataset_name == "DLPFC" and mode == "unsupervised":
        output_interim_dir = PROJECT_ROOT / "data" / "02_interim" / "spatial_cluster" / "DLPFC"
    else:
        output_interim_dir = Path(output_dir) if output_dir is not None else input_interim_dir
    if not input_interim_dir.is_absolute():
        input_interim_dir = PROJECT_ROOT / input_interim_dir
    if not output_interim_dir.is_absolute():
        output_interim_dir = PROJECT_ROOT / output_interim_dir

    if not input_interim_dir.exists():
        logger.error(f"dataset directory does not exist: {input_interim_dir},run normalize.py first")
        return
    output_interim_dir.mkdir(parents=True, exist_ok=True)

    # getsplitconfiguration
    if config is None:
        dataset_configs = DEFAULT_SPLIT_CONFIGS.get(dataset_name, {})
    else:
        dataset_configs = config

    modes_to_run = []
    if mode in ("supervised", "both") and "supervised" in dataset_configs:
        modes_to_run.append(("supervised", dataset_configs["supervised"]))
    if mode in ("unsupervised", "both") and "unsupervised" in dataset_configs:
        modes_to_run.append(("unsupervised", dataset_configs["unsupervised"]))

    if not modes_to_run:
        # if the mode has a corresponding configuration, use that configuration
        for m in ["supervised", "unsupervised"]:
            if m in dataset_configs:
                modes_to_run.append((m, dataset_configs[m]))

    for split_mode, split_cfg in modes_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"dataset: {dataset_name} | mode: {split_mode} | task: {split_cfg.get('tasks', [])}")
        logger.info(f"{'='*60}")

        # loaddata
        input_file = input_interim_dir / split_cfg.get("input_file", "processed.h5ad")
        if not input_file.exists():
            logger.error(f"input file does not exist: {input_file}")
            continue

        logger.info(f"load: {input_file}")
        adata = sc.read_h5ad(input_file)
        logger.info(f"data: {adata.n_obs} spots x {adata.n_vars} genes")

        # runsplit
        method = split_cfg["method"]
        if method == "slice_level_split":
            splits = slice_level_split(
                adata,
                slice_col=split_cfg.get("slice_col", "slice_id"),
                train_ratio=split_cfg.get("train_ratio", 0.6),
                val_ratio=split_cfg.get("val_ratio", 0.2),
                test_ratio=split_cfg.get("test_ratio", 0.2),
                stratify_col=split_cfg.get("stratify_col", "label"),
                random_seed=random_seed,
            )
        elif method == "sample_level_split":
            splits = sample_level_split(
                adata,
                sample_col=split_cfg.get("sample_col", "sample_id"),
                train_ratio=split_cfg.get("train_ratio", 0.7),
                val_ratio=split_cfg.get("val_ratio", 0.15),
                test_ratio=split_cfg.get("test_ratio", 0.15),
                stratify_col=split_cfg.get("stratify_col", "label"),
                random_seed=random_seed,
            )
        elif method == "all_split":
            splits = all_split(adata)
        else:
            logger.error(f"unknown split method: {method}")
            continue

        #   split column
        adata.obs["split"] = splits.values
        adata.uns["split_mode"] = split_mode
        adata.uns["split_method"] = method
        adata.uns["split_tasks"] = split_cfg.get("tasks", [])

        # save
        split_output_dir = output_interim_dir / split_mode
        clean_adata, _, _ = save_split_info(
            adata, split_output_dir, dataset_name, split_mode=split_mode
        )
        save_generator_model_views(
            clean_adata,
            dataset_name=dataset_name,
            split_mode=split_mode,
            interim_dir=output_interim_dir,
        )

    logger.info(f"\n[ok] {dataset_name} split completed")


def main(
    datasets: Optional[List[str]] = None,
    mode: str = "both",
    random_seed: int = 42,
    input_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
):
    """
    main entry point for dataset train/test splits

    Args:
        datasets: dataset list; None processes all datasets
        mode: 'supervised' / 'unsupervised' / 'both'
        random_seed: random seed
    """
    available = list(DEFAULT_SPLIT_CONFIGS.keys())
    if datasets is None:
        datasets = available
    else:
        for d in datasets:
            if d not in DEFAULT_SPLIT_CONFIGS:
                logger.warning(f"unknown dataset: {d}, skip. optional: {available}")

    logger.info(f"\n{'#'*60}")
    logger.info(f"# starting split: {datasets} | mode: {mode}")
    logger.info(f"{'#'*60}")

    for dataset_name in datasets:
        if dataset_name not in DEFAULT_SPLIT_CONFIGS:
            continue
        try:
            split_dataset(
                dataset_name,
                mode=mode,
                random_seed=random_seed,
                input_dir=input_dir,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.error(f"[error] {dataset_name} split failed: {e}", exc_info=True)

    logger.info(f"\n{'#'*60}")
    logger.info(f"# allsplit completed")
    logger.info(f"{'#'*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training set/test setsplit")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"dataset list (optional: {list(DEFAULT_SPLIT_CONFIGS.keys())}), all by default",
    )
    parser.add_argument(
        "--mode",
        choices=["supervised", "unsupervised", "both"],
        default="both",
        help="splitmode (default: both)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="directory containing processed_combined.h5ad; default data/02_interim/<dataset>",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="  directory for split/model_inputs/evaluation; defaults to the input directory",
    )
    args = parser.parse_args()

    main(
        datasets=args.datasets,
        mode=args.mode,
        random_seed=args.seed,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
