"""Unified DLPFC spatial-clustering synthetic generation.

This entrypoint is intentionally task-specific. Spatial clustering is an
unsupervised task: generator inputs must not contain manual labels, and
synthetic outputs must not carry labels. Ground-truth labels are supplied to
the downstream evaluation code through ``labels_for_evaluation.csv``.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import subprocess
import sys
from pathlib import Path

import anndata as ad
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        LEAKAGE_LABEL_COLUMNS,
        add_common_uns,
        infer_mode,
        make_synthetic_obs,
        make_synthetic_var,
        prepare_training_adata,
        resolve_path,
    )
else:
    from .common import (
        LEAKAGE_LABEL_COLUMNS,
        add_common_uns,
        infer_mode,
        make_synthetic_obs,
        make_synthetic_var,
        prepare_training_adata,
        resolve_path,
    )


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_MODELS = ("SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion")
NON_SPATIAL_MODELS = {"scGAN", "scDiffusion"}
CHECKPOINT_SUFFIX = {
    "SRTsim": ".pkl",
    "Splatter": ".pkl",
    "SPARsim": ".pkl",
    "scGAN": ".pth",
    "scDiffusion": ".pt",
}


def safe_token(value: str) -> str:
    return (
        str(value)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
        .replace(":", "-")
    )


def run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool = False):
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def load_yaml(path: str | Path) -> dict:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_unsupervised_input(path: str | Path) -> list[str]:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing unsupervised generator input: {path}")
    adata = ad.read_h5ad(path, backed="r")
    try:
        mode = infer_mode(adata, path)
        if mode != "unsupervised":
            raise ValueError(f"Expected unsupervised input, got mode={mode!r}: {path}")
        leakage = sorted(c for c in LEAKAGE_LABEL_COLUMNS if c in adata.obs.columns)
        if leakage:
            raise ValueError(f"Unsupervised input contains label leakage columns {leakage}: {path}")
        if "slice_id" not in adata.obs.columns:
            raise ValueError(f"DLPFC unsupervised input must contain obs['slice_id']: {path}")
        if "split" not in adata.obs.columns:
            raise ValueError(f"DLPFC unsupervised input must contain obs['split']: {path}")
        if "spatial" not in adata.obsm:
            raise ValueError(f"DLPFC unsupervised input must contain obsm['spatial']: {path}")
        split = adata.obs["split"].astype(str)
        train_mask = split == "train"
        if not train_mask.any() and (split == "all").any():
            train_mask = split == "all"
        slices = sorted(adata.obs.loc[train_mask, "slice_id"].astype(str).unique().tolist())
        if not slices:
            raise ValueError(f"No train/all slices found in {path}")
        return slices
    finally:
        adata.file.close()


def synthetic_pool_dir(output_root: str | Path, model: str) -> Path:
    return resolve_path(output_root) / "spatial_cluster" / "DLPFC" / model / "pool_40x"


def checkpoint_path(model_dir: str | Path, model: str, slice_id: str) -> Path:
    return (
        resolve_path(model_dir)
        / "spatial_cluster"
        / "DLPFC"
        / model
        / f"{model}_DLPFC_unsupervised_{safe_token(slice_id)}{CHECKPOINT_SUFFIX[model]}"
    )


def train_and_generate_slice(
    model: str,
    input_path: str | Path,
    output_dir: Path,
    model_dir: str | Path,
    config_path: str | Path,
    mapping_config: str | Path,
    slice_id: str,
    pool_ratio: float,
    python_bin: str,
    device: str,
    coord_method: str,
    dry_run: bool,
):
    model_root = PROJECT_ROOT / "src" / "02_generators" / "spatial_cluster" / "DLPFC" / model
    train_script = model_root / "train.py"
    generate_script = model_root / "generate.py"
    validate_script = PROJECT_ROOT / "src" / "02_generators" / "spatial_cluster" / "DLPFC" / "validate_synthetic_schema.py"
    ckpt = checkpoint_path(model_dir, model, slice_id)
    out = output_dir / safe_token(slice_id) / "synthetic.h5ad"
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spaug_matplotlib"))
    env.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "spaug_numba_cache"))

    train_cmd = [
        python_bin,
        str(train_script),
        "-i",
        str(resolve_path(input_path)),
        "-o",
        str(ckpt),
        "-c",
        str(resolve_path(config_path)),
        "--slice-id",
        str(slice_id),
    ]
    if model in NON_SPATIAL_MODELS:
        train_cmd.extend(["--device", device])
    run_cmd(train_cmd, env=env, dry_run=dry_run)

    gen_cmd = [
        python_bin,
        str(generate_script),
        "-i",
        str(resolve_path(input_path)),
        "--ckpt",
        str(ckpt),
        "-o",
        str(out),
        "-c",
        str(resolve_path(config_path)),
        "--slice-id",
        str(slice_id),
        "--ratio",
        str(float(pool_ratio)),
    ]
    if model in NON_SPATIAL_MODELS:
        gen_cmd.extend(["--device", device])
        if model == "scDiffusion":
            gen_cmd.append("--use-ddim")
    run_cmd(gen_cmd, env=env, dry_run=dry_run)

    if model in NON_SPATIAL_MODELS:
        assign_script = model_root / "assign_coords.py"
        run_cmd(
            [
                python_bin,
                str(assign_script),
                "--synthetic",
                str(out),
                "--reference",
                str(resolve_path(input_path)),
                "--method",
                coord_method,
                "--config",
                str(resolve_path(mapping_config)),
                "--slice-id",
                str(slice_id),
                "--generation-default",
            ],
            env=env,
            dry_run=dry_run,
        )

    run_cmd(
        [
            python_bin,
            str(validate_script),
            "--synthetic",
            str(out),
            "--generator",
            model,
            "--dataset",
            "DLPFC",
            "--mode",
            "unsupervised",
        ],
        env=env,
        dry_run=dry_run,
    )


def generate_model(
    model: str,
    input_path: str | Path,
    output_root: str | Path,
    model_dir: str | Path,
    config_path: str | Path,
    mapping_config: str | Path,
    pool_ratio: float,
    python_bin: str,
    device: str,
    coord_method: str,
    dry_run: bool,
) -> Path:
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {model!r}; expected one of {SUPPORTED_MODELS}")
    slices = validate_unsupervised_input(input_path)
    pool_dir = synthetic_pool_dir(output_root, model)
    if not dry_run:
        pool_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{model}] backend=python unsupervised slices={','.join(slices)} output={pool_dir}", flush=True)
    for slice_id in slices:
        train_and_generate_slice(
            model=model,
            input_path=input_path,
            output_dir=pool_dir,
            model_dir=model_dir,
            config_path=config_path,
            mapping_config=mapping_config,
            slice_id=slice_id,
            pool_ratio=pool_ratio,
            python_bin=python_bin,
            device=device,
            coord_method=coord_method,
            dry_run=dry_run,
        )

    pool_path = pool_dir / "synthetic_pool_40x.h5ad"
    merge_script = PROJECT_ROOT / "src" / "02_generators" / "spatial_cluster" / "DLPFC" / "merge_synthetic.py"
    validate_script = PROJECT_ROOT / "src" / "02_generators" / "spatial_cluster" / "DLPFC" / "validate_synthetic_schema.py"
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spaug_matplotlib"))
    env.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "spaug_numba_cache"))
    run_cmd(
        [
            python_bin,
            str(merge_script),
            "--input-root",
            str(pool_dir),
            "-o",
            str(pool_path),
        ],
        env=env,
        dry_run=dry_run,
    )
    run_cmd(
        [
            python_bin,
            str(validate_script),
            "--synthetic",
            str(pool_path),
            "--generator",
            model,
            "--dataset",
            "DLPFC",
            "--mode",
            "unsupervised",
        ],
        env=env,
        dry_run=dry_run,
    )
    return pool_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DLPFC unsupervised spatial-clustering synthetic pools.")
    parser.add_argument("--models", nargs="*", default=list(SUPPORTED_MODELS))
    parser.add_argument(
        "--input-template",
        default="data/02_interim/spatial_cluster/DLPFC/model_inputs/{model}/unsupervised/processed_with_split.h5ad",
    )
    parser.add_argument("--output-root", default="data/03_synthetic")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("-c", "--config", default="configs/spatial_cluster/DLPFC/generators.yaml")
    parser.add_argument("--mapping-config", default="configs/spatial_cluster/DLPFC/mapping.yaml")
    parser.add_argument("--pool-ratio", type=float, default=40.0)
    parser.add_argument(
        "--coord-method",
        default="spatial_resampling",
        choices=["spatial_resampling", "knn_mapping", "spatial_perturbation", "gmm_sampling"],
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    for model in args.models:
        input_path = args.input_template.format(model=model)
        out = generate_model(
            model=model,
            input_path=input_path,
            output_root=args.output_root,
            model_dir=args.model_dir,
            config_path=args.config,
            mapping_config=args.mapping_config,
            pool_ratio=args.pool_ratio,
            python_bin=args.python,
            device=args.device,
            coord_method=args.coord_method,
            dry_run=args.dry_run,
        )
        action = "dry-run target" if args.dry_run else "wrote"
        print(f"[{model}] {action} {out}", flush=True)


if __name__ == "__main__":
    main()
