"""Fit SRTsim parameters on prepared HVG training data.

The command records fitted distributions, training metadata, and the selected
coordinate policy in a reusable checkpoint.
"""

import os
import sys
import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import anndata as ad
import yaml

def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "02_generators" / "_faithful"))

from srtsim_core import SRTsimPython
from common import infer_dataset, infer_mode, minimal_reference_obs, prepare_training_adata

# ============================================================
# log
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SRTsim.train")


# ============================================================
# helper functions
# ============================================================

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ============================================================
# data loading
# ============================================================

def load_training_data(
    input_path: str,
    slice_id: Optional[str] = None,
    label_value: Optional[str] = None,
) -> tuple:
    """
    load training data

    Returns:
        (X_train, gene_names, coords, labels, adata_train)
    """
    input_path = resolve_path(input_path)
    logger.info(f"loaddata: {input_path}")

    adata = ad.read_h5ad(str(input_path))
    logger.info(f"full data: {adata.n_obs} spots x {adata.n_vars} genes")

    adata_train, dataset, mode, slice_id = prepare_training_adata(
        adata, input_path=input_path, slice_id=slice_id, label_value=label_value
    )
    logger.info(
        f"training set: {adata_train.n_obs} observations | dataset={dataset} | "
        f"mode={mode} | slice_id={slice_id}"
    )
    logger.info(f"HVG subset: {adata_train.n_vars} genes")

    # extract expression matrix
    if hasattr(adata_train.X, "toarray"):
        X = adata_train.X.toarray()
    else:
        X = np.array(adata_train.X)
    X = X.astype(np.float64)

    gene_names = adata_train.var_names.tolist()

    # coordinates
    coords = None
    if "spatial" in adata_train.obsm:
        coords = np.array(adata_train.obsm["spatial"], dtype=np.float64)
        logger.info(f"coordinates: {coords.shape}")

    # label (usein domain mode)
    labels = None
    if "label" in adata_train.obs.columns:
        labels = adata_train.obs["label"].values.astype(str)
        unique_labels = np.unique(labels)
        logger.info(f"label: {len(unique_labels)}   ({unique_labels[:5]}...)")

    logger.info(f"expression matrix: {X.shape}, min={X.min():.4f}, max={X.max():.4f}, "
                f"mean={X.mean():.4f},  ={np.mean(X == 0) * 100:.1f}%")

    return X, gene_names, coords, labels, adata_train


# ============================================================
# training (fit) main number
# ============================================================

def train(
    input_path: str,
    output_path: str,
    config_path: str = "configs/sample_disease_prediction/Bowel/generators.yaml",
    sim_scheme: Optional[str] = None,
    random_seed: Optional[int] = None,
    min_nonzero: Optional[int] = None,
    maxiter: Optional[int] = None,
    slice_id: Optional[str] = None,
    label_value: Optional[str] = None,
):
    """
    Fit SRTsim parameters and save the checkpoint.

    Args:
        input_path:   input h5ad file
        output_path:  model parameter output path (.pkl)
        config_path:  generators.yaml configuration file
        sim_scheme:   "tissue" / "domain" (override YAML)
        random_seed:  random seed (override YAML)
        min_nonzero:  minimumnon- valuenumber (override YAML)
        maxiter:      maximum number (override YAML)
    """
    # Load configuration
    cfg = load_yaml(str(resolve_path(config_path)))
    srtsim_cfg = cfg.get("SRTsim", {})

    sim_scheme = sim_scheme or srtsim_cfg.get("sim_scheme", srtsim_cfg.get("simulation_method", "tissue"))
    # will "mixture" (YAML)  as "tissue" (internal)
    if sim_scheme == "mixture":
        sim_scheme = "tissue"
    if random_seed is None:
        random_seed = srtsim_cfg.get("random_seed", 42)
    if min_nonzero is None:
        min_nonzero = srtsim_cfg.get("min_nonzero", 2)
    if maxiter is None:
        maxiter = srtsim_cfg.get("maxiter", 500)

    logger.info("=" * 60)
    logger.info("SRTsim fitting configuration")
    logger.info("=" * 60)
    logger.info(f"  sim_scheme:   {sim_scheme}")
    logger.info(f"  random_seed:  {random_seed}")
    logger.info(f"  min_nonzero:  {min_nonzero}")
    logger.info(f"  maxiter:      {maxiter}")
    logger.info("=" * 60)

    # loaddata
    X, gene_names, coords, labels, adata_train = load_training_data(
        input_path, slice_id=slice_id, label_value=label_value
    )
    dataset = infer_dataset(adata_train, input_path)
    mode = infer_mode(adata_train, input_path)

    # create canonical Python faithful reimplementation.
    model = SRTsimPython(
        random_seed=random_seed,
        sim_scheme=sim_scheme,
        min_nonzero_num=min_nonzero,
        maxiter=maxiter,
    )

    # fit
    start_time = time.time()
    logger.info(f"\nFitting {X.shape[1]} genes...")

    model.fit(X, gene_names, labels=labels)

    elapsed = time.time() - start_time
    logger.info(f"\nfitcompleted,elapsed time: {elapsed:.1f}s")

    # save checkpoint
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model,
        "model_backend": "python_reimplementation",
        "model_backend_detail": "SRTsim reference-based Python implementation",
        "model_source_reference": "published SRTsim design; published design reference: published project licensing applies",
        "n_genes": len(gene_names),
        "gene_names": gene_names,
        "n_train_spots": X.shape[0],
        "data_shape": list(X.shape),
        # save training coordinates for SRTsim coordinate-aware generation
        "train_coords": coords,
        "train_labels": labels,
        # save the reference expression matrix for rank preservation
        "X_ref": X,
        "dataset": dataset,
        "generation_mode": mode,
        "slice_id": str(slice_id) if slice_id is not None else (
            str(adata_train.obs["slice_id"].iloc[0]) if "slice_id" in adata_train.obs else None
        ),
        "label_condition": str(label_value) if label_value is not None else None,
        "label_strategy": "label_wise_generation" if label_value is not None else None,
        "label_conditioned": bool(label_value is not None or sim_scheme == "domain"),
        "gene_space": "hvg",
        "reference_obs": minimal_reference_obs(adata_train, dataset, mode),
    }

    with open(str(output_path), "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"\nmodel saved: {output_path} ({file_size_mb:.1f} MB)")
    logger.info(f"  gene count: {len(gene_names)}")
    logger.info(f"  training spots: {X.shape[0]}")

    return str(output_path)


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit SRTsim parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="input h5ad filepath",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="model parameter output path (.pkl)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/sample_disease_prediction/Bowel/generators.yaml",
        help="generators.yaml configuration file path",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default=None,
        choices=["tissue", "domain", "mixture"],
        help="simulation scheme (override YAML)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed (override YAML)",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="maximum number (override YAML)",
    )
    parser.add_argument(
        "--slice-id",
        type=str,
        default=None,
        help="DLPFC mode trains on the selected slice",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="supervised generation condition: train on selected observations with this label and record label-wise synthetic labels",
    )

    args = parser.parse_args()

    train(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        sim_scheme=args.scheme,
        random_seed=args.seed,
        maxiter=args.maxiter,
        slice_id=args.slice_id,
        label_value=args.label,
    )
