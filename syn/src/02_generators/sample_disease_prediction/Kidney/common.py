"""
Shared generator data-contract helpers.

These helpers enforce the project semantics:
- DLPFC generation is slice-wise.
- Unsupervised generator inputs use expression and spatial features; evaluation
  labels remain in the downstream evaluation layer.
- Synthetic IDs are provenance only, never model features.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import yaml


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
CAPABILITY_CONFIG = PROJECT_ROOT / "configs" / "sample_disease_prediction" / "Kidney" / "model_capabilities.yaml"

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


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_generator_capability(generator: str) -> dict:
    if not CAPABILITY_CONFIG.exists():
        return {}
    with CAPABILITY_CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    capability = cfg.get(generator, {}) or {}
    return dict(capability)


def infer_coord_source(generator: str, syn: Optional[ad.AnnData] = None) -> str:
    capability = load_generator_capability(generator)
    coord_source = str(capability.get("default_coord_source", "") or "")
    if syn is not None and "coord_method" in syn.uns:
        return str(syn.uns["coord_method"])
    return coord_source or "unknown"


def infer_condition_source(
    generator: str,
    label_strategy: Optional[str] = None,
) -> str:
    if label_strategy:
        return str(label_strategy)
    capability = load_generator_capability(generator)
    return str(capability.get("default_condition_source", "") or "none")


def infer_dataset(adata: ad.AnnData, input_path: Optional[str] = None) -> str:
    dataset = str(adata.uns.get("dataset", "") or "")
    if dataset:
        return dataset
    if input_path:
        parts = set(resolve_path(input_path).parts)
        if "DLPFC" in parts:
            return "DLPFC"
        if "Kidney" in parts:
            return "Kidney"
        if "Trastuzumab" in parts:
            return "Trastuzumab"
    return "unknown"


def infer_mode(adata: ad.AnnData, input_path: Optional[str] = None) -> str:
    mode = str(adata.uns.get("split_mode", "") or "").lower()
    if mode in {"supervised", "unsupervised"}:
        return mode
    if input_path:
        parts = set(resolve_path(input_path).parts)
        if "unsupervised" in parts:
            return "unsupervised"
        if "supervised" in parts:
            return "supervised"
    return "supervised" if "label" in adata.obs.columns else "unsupervised"


def label_like_columns(adata: ad.AnnData) -> list[str]:
    return sorted(c for c in LEAKAGE_LABEL_COLUMNS if c in adata.obs.columns)


def validate_no_unsupervised_leakage(adata: ad.AnnData, mode: str):
    if mode != "unsupervised":
        return
    leakage = label_like_columns(adata)
    if leakage:
        raise ValueError(
            "Unsupervised generator input contains manual/evaluation label columns: "
            f"{leakage}. Re-run src/01_data_prep/split_train_test.py after the "
            "unsupervised leakage fix."
        )


def select_hvg(adata: ad.AnnData) -> ad.AnnData:
    if "highly_variable" not in adata.var.columns:
        return adata
    hvg = adata.var["highly_variable"].astype(bool)
    if int(hvg.sum()) == 0:
        return adata
    return adata[:, hvg].copy()


def split_training_mask(adata: ad.AnnData) -> pd.Series:
    if "split" not in adata.obs.columns:
        return pd.Series(True, index=adata.obs_names)
    split = adata.obs["split"].astype(str)
    if (split == "train").any():
        return split == "train"
    if (split == "all").any():
        return split == "all"
    raise ValueError("No train/all observations are available for generator training")


def prepare_training_adata(
    adata: ad.AnnData,
    input_path: Optional[str] = None,
    slice_id: Optional[str] = None,
    label_value: Optional[str] = None,
    require_slice_for_dlpfc: bool = True,
    allow_multiclass_supervised: bool = False,
) -> Tuple[ad.AnnData, str, str, Optional[str]]:
    """Return a leakage-checked, HVG-filtered training AnnData view."""
    dataset = infer_dataset(adata, input_path)
    mode = infer_mode(adata, input_path)
    validate_no_unsupervised_leakage(adata, mode)
    if dataset == "DLPFC" and mode == "supervised" and label_value is None and not allow_multiclass_supervised:
        raise ValueError(
            "DLPFC supervised generation is now label-wise. Pass --label <label> "
            "and train/generate one slice-label condition at a time."
        )

    train_mask = split_training_mask(adata)
    train = adata[train_mask].copy()

    if dataset == "DLPFC":
        if "slice_id" not in train.obs.columns:
            raise ValueError("DLPFC generator input must contain obs['slice_id']")
        train_slices = sorted(train.obs["slice_id"].astype(str).unique().tolist())
        if slice_id is None:
            if require_slice_for_dlpfc and len(train_slices) != 1:
                raise ValueError(
                    "DLPFC generation must be slice-wise. The input contains "
                    f"{len(train_slices)} training slices: {train_slices}. "
                    "Pass --slice-id <slice_id> and train/generate one slice at a time."
                )
            slice_id = train_slices[0]
        train = train[train.obs["slice_id"].astype(str) == str(slice_id)].copy()
        if train.n_obs == 0:
            raise ValueError(f"No training spots found for slice_id={slice_id}")
    elif dataset == "Kidney":
        if "sample_id" not in train.obs.columns:
            raise ValueError("Kidney sample-level generator input must contain obs['sample_id']")
        train_samples = sorted(train.obs["sample_id"].astype(str).unique().tolist())
        if slice_id is None:
            if len(train_samples) != 1:
                raise ValueError(
                    "Kidney sample-level generation must be sample-wise. "
                    f"The input contains {len(train_samples)} samples: {train_samples[:10]}. "
                    "Pass --slice-id <sample_id> and train/generate one sample at a time."
                )
            slice_id = train_samples[0]
        train = train[train.obs["sample_id"].astype(str) == str(slice_id)].copy()
        if train.n_obs == 0:
            raise ValueError(f"No training spots found for sample_id={slice_id}")

    if label_value is not None:
        if mode != "supervised":
            raise ValueError("--label can only be used with supervised generator inputs")
        if "label" not in train.obs.columns:
            raise ValueError("Label-wise supervised generation requires obs['label']")
        train = train[train.obs["label"].astype(str) == str(label_value)].copy()
        if train.n_obs == 0:
            context = f"slice_id={slice_id}, " if slice_id is not None else ""
            raise ValueError(f"No training observations found for {context}label={label_value}")

    train = select_hvg(train)
    if train.n_obs == 0 or train.n_vars == 0:
        raise ValueError("Training data is empty after split/slice/HVG filtering")
    return train, dataset, mode, slice_id


def repeat_reference_obs(ref_obs: pd.DataFrame, n_obs: int) -> pd.DataFrame:
    if ref_obs.shape[0] == 0:
        raise ValueError("Cannot create synthetic metadata from empty reference obs")
    idx = np.arange(n_obs) % ref_obs.shape[0]
    return ref_obs.iloc[idx].copy()


def validate_trusted_label_condition(
    mode: str,
    checkpoint: dict,
    label_value: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Return a trusted supervised label condition or fail loudly.

    A valid supervised label condition comes from a checkpoint trained label-wise;
    generation uses that recorded condition together with the requested label.
    """
    ckpt_label = checkpoint.get("label_condition")
    ckpt_strategy = checkpoint.get("label_strategy")
    if label_value is None and ckpt_label is not None:
        label_value = str(ckpt_label)
    if label_value is not None and ckpt_label is not None and str(label_value) != str(ckpt_label):
        raise ValueError(
            f"Requested label={label_value} does not match checkpoint label_condition={ckpt_label}"
        )
    if mode != "supervised":
        return None, None
    if label_value is None:
        return None, None
    if ckpt_strategy != "label_wise_generation":
        condition_classes = [str(x) for x in checkpoint.get("condition_classes") or []]
        conditional_key = str(checkpoint.get("conditional_key") or "")
        if (
            checkpoint.get("conditional") is True
            and conditional_key == "label"
            and str(label_value) in condition_classes
        ):
            return str(label_value), "internal_cscgan_conditional_bn_projection"
        classifier_classes = [str(x) for x in checkpoint.get("classifier_classes") or []]
        classifier_key = str(checkpoint.get("classifier_key") or "")
        if (
            checkpoint.get("classifier_state") is not None
            and classifier_key == "label"
            and str(label_value) in classifier_classes
        ):
            return str(label_value), "classifier_guidance"
        raise ValueError(
            "Supervised synthetic labels require a trusted conditional checkpoint. "
            "Train this generator with --label <label>, or use a checkpoint "
            "with internal cscGAN conditioning or classifier guidance trained on obs['label']; "
            "the checkpoint records the requested label condition."
        )
    return str(label_value), str(ckpt_strategy)


def minimal_reference_obs(
    adata: ad.AnnData,
    dataset: Optional[str] = None,
    mode: Optional[str] = None,
) -> pd.DataFrame:
    """Keep only provenance/condition columns needed by synthetic obs creation."""
    dataset = dataset or infer_dataset(adata)
    mode = mode or infer_mode(adata)
    obs = pd.DataFrame(index=adata.obs_names.copy())

    if dataset == "DLPFC":
        if "slice_id" in adata.obs.columns:
            obs["slice_id"] = adata.obs["slice_id"].astype(str).values
        if mode == "supervised" and "label" in adata.obs.columns:
            obs["label"] = adata.obs["label"].values
    elif dataset == "Kidney":
        for col in ("sample_id", "slice_id", "disease_label", "disease_state_raw", "oncotree_code", "patient", "dataset_title", "organ", "st_technology"):
            if col in adata.obs.columns:
                obs[col] = adata.obs[col].astype(str).values
    elif dataset == "Trastuzumab":
        if "sample_id" in adata.obs.columns:
            obs["sample_id"] = adata.obs["sample_id"].astype(str).values
        if mode == "supervised" and "label" in adata.obs.columns:
            obs["label"] = adata.obs["label"].values
    else:
        for col in ("slice_id", "sample_id"):
            if col in adata.obs.columns:
                obs[col] = adata.obs[col].astype(str).values
        if mode == "supervised" and "label" in adata.obs.columns:
            obs["label"] = adata.obs["label"].values

    return obs


def make_synthetic_obs(
    ref_obs: pd.DataFrame,
    n_obs: int,
    generator: str,
    dataset: str,
    mode: str,
    slice_id: Optional[str] = None,
    synthetic_labels: Optional[str | Sequence[str] | np.ndarray] = None,
    label_strategy: Optional[str] = None,
) -> pd.DataFrame:
    ref = repeat_reference_obs(ref_obs, n_obs)
    id_parts = [generator]
    if slice_id is not None:
        id_parts.append(str(slice_id))
    if synthetic_labels is not None and isinstance(synthetic_labels, str):
        id_parts.append(str(synthetic_labels))
    prefix = "_".join(part.replace("/", "-").replace(" ", "-") for part in id_parts)
    obs = pd.DataFrame(index=[f"{prefix}_{i}" for i in range(n_obs)])
    obs["source"] = generator
    obs["generator"] = generator
    obs["augmentation_source"] = "synthetic"
    obs["split"] = "synthetic"
    obs["dataset"] = dataset
    obs["synthetic_id"] = obs.index.astype(str)
    obs["source_obs_name"] = ref.index.astype(str).values

    trusted_labels = None
    if synthetic_labels is not None:
        if isinstance(synthetic_labels, str):
            trusted_labels = np.repeat(synthetic_labels, n_obs)
        else:
            trusted_labels = np.asarray(synthetic_labels).astype(str)
            if trusted_labels.shape[0] != n_obs:
                raise ValueError(
                    f"synthetic_labels length {trusted_labels.shape[0]} does not match n_obs={n_obs}"
                )
    elif mode == "supervised":
        raise ValueError(
            "Supervised synthetic observations require trusted synthetic labels. "
            "Use label-wise generation (--label) or pass explicit synthetic_labels; "
            "unconditional synthetic expression receives labels only from trusted generation conditions."
        )

    if dataset == "DLPFC":
        fixed_slice = str(slice_id or ref["slice_id"].astype(str).iloc[0])
        obs["slice_id"] = fixed_slice
        if trusted_labels is not None:
            obs["label"] = trusted_labels
    elif dataset == "Kidney":
        fixed_sample = str(slice_id or ref["sample_id"].astype(str).iloc[0])
        obs["sample_id"] = fixed_sample
        obs["slice_id"] = fixed_sample
        for col in ("disease_label", "disease_state_raw", "oncotree_code", "patient", "dataset_title", "organ", "st_technology"):
            if col in ref.columns:
                obs[col] = ref[col].astype(str).values
    elif dataset == "Trastuzumab":
        obs["sample_id"] = [f"{generator}_synthetic_{i}" for i in range(n_obs)]
        if "sample_id" in ref.columns:
            obs["source_sample_id"] = ref["sample_id"].astype(str).values
        if trusted_labels is not None:
            obs["label"] = trusted_labels
    else:
        for col in ("slice_id", "sample_id"):
            if col in ref.columns:
                obs[col] = ref[col].astype(str).values
        if trusted_labels is not None:
            obs["label"] = trusted_labels

    if trusted_labels is not None:
        obs["label_strategy"] = label_strategy or "explicit_synthetic_labels"

    if mode == "unsupervised":
        drop_cols = [c for c in LEAKAGE_LABEL_COLUMNS if c in obs.columns]
        if drop_cols:
            obs = obs.drop(columns=drop_cols)

    return obs


def make_synthetic_var(gene_names: list[str]) -> pd.DataFrame:
    var = pd.DataFrame(index=gene_names)
    var["highly_variable"] = True
    return var


def repeat_reference_coords(ref_coords: np.ndarray, n_obs: int) -> np.ndarray:
    ref_coords = np.asarray(ref_coords, dtype=np.float64)
    if ref_coords.shape[0] == 0:
        raise ValueError("Cannot create synthetic coordinates from empty reference coords")
    idx = np.arange(n_obs) % ref_coords.shape[0]
    return ref_coords[idx].copy()


def add_common_uns(
    syn: ad.AnnData,
    generator: str,
    dataset: str,
    mode: str,
    n_train: int,
    n_generate_ratio: float,
    slice_id: Optional[str] = None,
    label_strategy: Optional[str] = None,
    label_condition: Optional[str] = None,
):
    capability = load_generator_capability(generator)
    syn.uns["generator"] = generator
    syn.uns["generator_capability"] = capability
    syn.uns["dataset"] = dataset
    syn.uns["generation_mode"] = mode
    syn.uns["n_train_spots"] = int(n_train)
    syn.uns["n_train_condition"] = int(n_train)
    syn.uns["n_generated"] = int(syn.n_obs)
    syn.uns["n_generate_ratio"] = float(n_generate_ratio)
    syn.uns["generate_ratio"] = float(n_generate_ratio)
    if float(n_generate_ratio) == 40.0:
        syn.uns["pool_ratio"] = 40
        syn.uns["pool_name"] = "pool_40x"
    syn.uns["gene_space"] = "hvg"
    syn.uns["n_genes"] = int(syn.n_vars)
    syn.uns["condition_source"] = infer_condition_source(generator, label_strategy) if label_strategy else "none"
    syn.uns["coord_source"] = infer_coord_source(generator, syn)
    if slice_id is not None:
        syn.uns["slice_id"] = str(slice_id)
    if label_strategy is not None:
        syn.uns["label_strategy"] = str(label_strategy)
    if label_condition is not None:
        syn.uns["label_condition"] = str(label_condition)
