"""Generate DLPFC supervised low-label synthetic pools.

This script is intentionally separate from the unsupervised spatial-clustering
pipeline. It trains one slice-level conditional generator per slice. Labels are
used as supervised generation conditions and are written as trusted
synthetic labels for downstream within-slice classification.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from io_utils import (  # type: ignore
        dense_array,
        ensure_columns,
        fraction_tag,
        pool_tag,
        read_split,
        resolve_path,
        safe_token,
        sanitize_obs,
        synthetic_pool_path,
        write_json,
    )
else:
    from .io_utils import (
        dense_array,
        ensure_columns,
        fraction_tag,
        pool_tag,
        read_split,
        resolve_path,
        safe_token,
        sanitize_obs,
        synthetic_pool_path,
        write_json,
    )


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
INTERNAL_CONDITION_MODELS = ("SRTsim", "Splatter", "SPARsim")
EXTERNAL_LABELWISE_MODELS = ("scGAN", "scDiffusion")
SUPPORTED_MODELS = INTERNAL_CONDITION_MODELS + EXTERNAL_LABELWISE_MODELS
CHECKPOINT_SUFFIX = {"scGAN": ".pth", "scDiffusion": ".pt"}


def load_yaml(path: str | Path | None) -> dict:
    if path is None:
        return {}
    path = resolve_path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def model_overrides(config: dict, model_name: str) -> dict:
    return dict((config or {}).get(model_name, {}) or {})


def normalize_slices(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    out: set[str] = set()
    for value in values:
        out.update(str(x) for x in str(value).split(",") if x)
    return out or None


def finalize_pool_schema(adata: ad.AnnData, model_name: str) -> ad.AnnData:
    adata = sanitize_obs(adata)
    adata.obs["source"] = model_name
    adata.obs["augmentation_source"] = "synthetic"
    adata.obs["split"] = "synthetic"
    adata.obs["dataset"] = "DLPFC"
    adata.obs["generator"] = model_name
    adata.obs["synthetic_id"] = adata.obs_names.astype(str)
    adata.uns["generator"] = model_name
    adata.uns["dataset"] = "DLPFC"
    adata.uns["generation_mode"] = "supervised"
    adata.uns["n_generated"] = int(adata.n_obs)
    adata.uns["n_genes"] = int(adata.n_vars)
    adata.uns["gene_space"] = "hvg"
    if "highly_variable" not in adata.var.columns:
        adata.var["highly_variable"] = True
    else:
        adata.var["highly_variable"] = adata.var["highly_variable"].astype(bool)
    return adata


def hvg_mask(var: pd.DataFrame) -> pd.Series | None:
    if "highly_variable" not in var.columns:
        return None
    mask = var["highly_variable"].astype(bool)
    if int(mask.sum()) == 0:
        return None
    return mask


def select_hvg(adata: ad.AnnData) -> ad.AnnData:
    mask = hvg_mask(adata.var)
    if mask is None:
        return adata.copy()
    return adata[:, mask].copy()


def select_hvg_var(adata: ad.AnnData) -> pd.DataFrame:
    mask = hvg_mask(adata.var)
    if mask is None:
        return adata.var.copy()
    return adata.var.loc[mask].copy()


def _load_symbol(file_path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location(f"_spg_{safe_token(file_path.stem)}_{safe_token(symbol)}", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {symbol} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, symbol)


@dataclass
class ConditionFit:
    label: str
    n_real_train: int
    model: Any
    x_ref: np.ndarray
    coords_ref: np.ndarray
    region_key: str | None = None


def _fit_base_model(
    model_name: str,
    x: np.ndarray,
    gene_names: list[str],
    coords: np.ndarray,
    seed: int,
    params: dict[str, Any],
):
    model_root = PROJECT_ROOT / "src" / "02_generators" / "supervised_low_label" / "DLPFC" / model_name / "model.py"
    if model_name == "SRTsim":
        cls = _load_symbol(model_root, "SRTsimModel")
        model = cls(
            random_seed=seed,
            sim_scheme="tissue",
            min_nonzero=int(params.get("min_nonzero", 1)),
            maxiter=int(params.get("maxiter", 120)),
        )
        model.fit(x, gene_names=gene_names, coords=coords)
        return model
    if model_name == "Splatter":
        cls = _load_symbol(model_root, "SplatterModel")
        model = cls(
            n_regions=int(params.get("n_regions", 7)),
            min_region_size=int(params.get("min_region_size", 10)),
            coord_sigma_factor=float(params.get("coord_sigma_factor", 0.15)),
            random_seed=seed,
        )
        model.fit(x, gene_names=gene_names, coords=coords)
        return model
    if model_name == "SPARsim":
        cls = _load_symbol(model_root, "SPARsimModel")
        model = cls(
            n_regions=int(params.get("n_regions", 7)),
            min_region_size=int(params.get("min_region_size", 10)),
            coord_sigma_factor=float(params.get("coord_sigma_factor", 0.12)),
            random_seed=seed,
        )
        model.fit(x, gene_names=gene_names, coords=coords)
        return model
    raise ValueError(f"Unsupported supervised conditional generator: {model_name}")


def _sample_coords(coords_ref: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    coords_ref = np.asarray(coords_ref, dtype=np.float64)
    if coords_ref.shape[0] == 0:
        return np.zeros((n, 2), dtype=np.float64)
    base = coords_ref[rng.integers(0, coords_ref.shape[0], size=n), :2].astype(np.float64, copy=True)
    sd = np.maximum(coords_ref[:, :2].std(axis=0) * 0.08, 1e-6)
    return base + rng.normal(0.0, sd, size=(n, 2))


def fit_slice_conditions(
    adata: ad.AnnData,
    model_name: str,
    slice_id: str,
    seed: int,
    params: dict[str, Any],
) -> list[ConditionFit]:
    ensure_columns(adata, ["slice_id", "label", "split"], "supervised split")
    train = adata[
        (adata.obs["split"].astype(str) == "train")
        & (adata.obs["slice_id"].astype(str) == str(slice_id))
    ].copy()
    if train.n_obs == 0:
        raise ValueError(f"No train spots for slice_id={slice_id}")
    if "spatial" not in train.obsm:
        raise ValueError(f"No obsm['spatial'] found for slice_id={slice_id}")
    train = select_hvg(train)

    gene_names = train.var_names.astype(str).tolist()
    fits: list[ConditionFit] = []
    for label, idx in train.obs.groupby(train.obs["label"].astype(str), sort=True).groups.items():
        cond = train[list(idx)].copy()
        x = np.maximum(dense_array(cond.X).astype(np.float64), 0.0)
        coords = np.asarray(cond.obsm["spatial"][:, :2], dtype=np.float64)
        if x.shape[0] == 0:
            continue
        fitted = _fit_base_model(
            model_name,
            x=x,
            gene_names=gene_names,
            coords=coords,
            seed=seed + len(fits) * 997,
            params=params,
        )
        fits.append(
            ConditionFit(
                label=str(label),
                n_real_train=int(cond.n_obs),
                model=fitted,
                x_ref=x,
                coords_ref=coords,
            )
        )
    if not fits:
        raise ValueError(f"No label conditions could be fitted for slice_id={slice_id}")
    return fits


def generate_condition(
    model_name: str,
    fit: ConditionFit,
    n_generate: int,
    seed: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    if model_name == "SRTsim":
        x = fit.model.generate(fit.x_ref, n_generate=n_generate)
        coords_ref = np.asarray(fit.coords_ref, dtype=np.float64)
        if coords_ref.shape[0] == 0:
            coords = np.zeros((n_generate, 2), dtype=np.float64)
        else:
            base = coords_ref[rng.integers(0, coords_ref.shape[0], size=n_generate), :2].astype(np.float64, copy=True)
            sd = np.maximum(coords_ref[:, :2].std(axis=0) * float(params.get("coord_sigma_factor", 0.08)), 1e-6)
            coords = base + rng.normal(0.0, sd, size=(n_generate, 2))
        regions = None
    elif model_name in {"Splatter", "SPARsim"}:
        x, coords, regions = fit.model.generate(n_generate, random_seed=seed)
    else:
        raise ValueError(f"Unsupported supervised conditional generator: {model_name}")
    return np.maximum(np.asarray(x, dtype=np.float32), 0.0), np.asarray(coords, dtype=np.float64), regions


def generate_slice_pool(
    adata: ad.AnnData,
    model_name: str,
    slice_id: str,
    label_fraction: float,
    seed: int,
    pool_ratio: float,
    params: dict[str, Any],
) -> ad.AnnData:
    fits = fit_slice_conditions(adata, model_name=model_name, slice_id=slice_id, seed=seed, params=params)
    blocks = []
    obs_blocks = []
    coord_blocks = []
    region_blocks = []
    var = select_hvg_var(adata)

    for cond_i, fit in enumerate(fits):
        n_generate = max(1, int(round(fit.n_real_train * pool_ratio)))
        x, coords, regions = generate_condition(
            model_name,
            fit,
            n_generate=n_generate,
            seed=seed + cond_i * 1009 + 17,
            params=params,
        )
        obs = pd.DataFrame(
            {
                "slice_id": str(slice_id),
                "label": fit.label,
                "condition_label": fit.label,
                "source": "synthetic",
                "generator": model_name,
                "n_real_train_condition": fit.n_real_train,
                "label_fraction": float(label_fraction),
                "low_label_seed": int(seed),
                "pool_ratio": float(pool_ratio),
                "generation_scope": "slice_conditional",
                "condition_key": "label",
            },
            index=[f"syn_{model_name}_{safe_token(slice_id)}_{safe_token(fit.label)}_{i}" for i in range(n_generate)],
        )
        blocks.append(x)
        obs_blocks.append(obs)
        coord_blocks.append(coords)
        if regions is not None:
            region_blocks.extend([str(r) for r in np.asarray(regions).astype(str).tolist()])
        else:
            region_blocks.extend(["reference_coord_resampling"] * n_generate)

    x_all = np.vstack(blocks).astype(np.float32)
    obs_all = pd.concat(obs_blocks, axis=0)
    coords_all = np.vstack(coord_blocks)
    obs_all["synthetic_region"] = region_blocks
    out = ad.AnnData(X=x_all, obs=obs_all, var=var)
    out.obsm["spatial"] = coords_all
    out.uns["dataset"] = "DLPFC"
    out.uns["task_family"] = "supervised_low_label"
    out.uns["task_name"] = "dlpfc_slice_low_label_spot_clf"
    out.uns["split_mode"] = "supervised"
    out.uns["generation_mode"] = "supervised"
    out.uns["generation_scope"] = "slice_conditional"
    out.uns["condition_key"] = "label"
    out.uns["label_strategy"] = "slice_conditional_label_generation"
    out.uns["trusted_synthetic_labels"] = True
    out.uns["generator"] = model_name
    out.uns["label_fraction"] = float(label_fraction)
    out.uns["low_label_seed"] = int(seed)
    out.uns["pool_ratio"] = float(pool_ratio)
    out.uns["coord_method"] = "model_region_sampling" if model_name in {"Splatter", "SPARsim"} else "reference_coord_resampling"
    return finalize_pool_schema(out, model_name)


def _run_cmd(cmd: list[str], env: dict[str, str] | None = None, dry_run: bool = False):
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


def _training_conditions(adata: ad.AnnData) -> pd.DataFrame:
    train = adata.obs[adata.obs["split"].astype(str) == "train"].copy()
    if train.empty:
        raise ValueError("No train observations found in supervised split")
    return (
        train[["slice_id", "label"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["slice_id", "label"])
        .reset_index(drop=True)
    )


def generate_external_labelwise_pool(
    model_name: str,
    split_path: str | Path,
    output_root: str | Path,
    model_dir: str | Path,
    config_path: str | Path,
    mapping_config: str | Path,
    label_fraction: float,
    seed: int,
    pool_ratio: float,
    python_bin: str,
    device: str,
    deep_epochs: int | None,
    deep_batch_size: int | None,
    deep_n_timesteps: int | None,
    use_ddim: bool,
    task_config: dict[str, Any],
    slice_filter: set[str] | None,
    write_combined: bool,
    skip_existing: bool,
    dry_run: bool,
) -> Path:
    adata = read_split(split_path)
    params = model_overrides(task_config, model_name)
    train_epochs = deep_epochs if deep_epochs is not None else params.get("epochs")
    train_batch_size = deep_batch_size if deep_batch_size is not None else params.get("batch_size")
    generate_batch_size = params.get("generate_batch_size", train_batch_size)
    diffusion_timesteps = deep_n_timesteps if deep_n_timesteps is not None else params.get("n_timesteps")
    use_ddim = bool(use_ddim or params.get("use_ddim", False))
    conditions = _training_conditions(adata)
    if slice_filter is not None:
        conditions = conditions[conditions["slice_id"].astype(str).isin(slice_filter)].reset_index(drop=True)
    if conditions.empty:
        raise ValueError(f"No train conditions remain after slice filtering: {sorted(slice_filter or [])}")
    if model_name == "scGAN":
        return generate_scgan_slice_conditional_pool(
            adata=adata,
            conditions=conditions,
            split_path=split_path,
            output_root=output_root,
            model_dir=model_dir,
            config_path=config_path,
            mapping_config=mapping_config,
            label_fraction=label_fraction,
            seed=seed,
            pool_ratio=pool_ratio,
            python_bin=python_bin,
            device=device,
            deep_epochs=deep_epochs,
            deep_batch_size=deep_batch_size,
            task_config=task_config,
            write_combined=write_combined,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )
    if model_name == "scDiffusion":
        return generate_scdiffusion_slice_classifier_guided_pool(
            adata=adata,
            conditions=conditions,
            split_path=split_path,
            output_root=output_root,
            model_dir=model_dir,
            config_path=config_path,
            mapping_config=mapping_config,
            label_fraction=label_fraction,
            seed=seed,
            pool_ratio=pool_ratio,
            python_bin=python_bin,
            device=device,
            deep_epochs=deep_epochs,
            deep_batch_size=deep_batch_size,
            deep_n_timesteps=deep_n_timesteps,
            use_ddim=use_ddim,
            task_config=task_config,
            write_combined=write_combined,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )
    base_out = synthetic_pool_path(output_root, model_name, label_fraction, seed, pool_ratio=pool_ratio)
    condition_root = base_out.parent / "conditions"
    ckpt_root = (
        resolve_path(model_dir)
        / "supervised_low_label"
        / "DLPFC"
        / model_name
        / fraction_tag(label_fraction)
        / f"seed{seed}"
    )
    if not dry_run:
        condition_root.mkdir(parents=True, exist_ok=True)
        ckpt_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "spaug_numba_cache"))

    parts = []
    condition_files = []
    suffix = CHECKPOINT_SUFFIX[model_name]
    for _, row in conditions.iterrows():
        slice_id = str(row["slice_id"])
        label = str(row["label"])
        safe_label = safe_token(label)
        ckpt = ckpt_root / f"{model_name}_DLPFC_supervised_low_label_{safe_token(slice_id)}_{safe_label}{suffix}"
        out = condition_root / f"slice_{safe_token(slice_id)}" / safe_label / "synthetic.h5ad"
        train_script = PROJECT_ROOT / "src" / "02_generators" / "supervised_low_label" / "DLPFC" / model_name / "train.py"
        generate_script = PROJECT_ROOT / "src" / "02_generators" / "supervised_low_label" / "DLPFC" / model_name / "generate.py"
        assign_script = PROJECT_ROOT / "src" / "02_generators" / "supervised_low_label" / "DLPFC" / model_name / "assign_coords.py"

        if skip_existing and out.exists():
            condition_files.append(str(out))
            print(f"[{model_name} seed={seed}] skip existing slice={slice_id} label={label}: {out}", flush=True)
            if write_combined:
                parts.append(ad.read_h5ad(out))
            continue

        train_cmd = [
            python_bin,
            str(train_script),
            "-i",
            str(resolve_path(split_path)),
            "-o",
            str(ckpt),
            "-c",
            str(resolve_path(config_path)),
            "--slice-id",
            slice_id,
            "--label",
            label,
            "--device",
            device,
        ]
        if train_epochs is not None:
            train_cmd.extend(["--epochs", str(int(train_epochs))])
        if train_batch_size is not None:
            train_cmd.extend(["--batch-size", str(int(train_batch_size))])
        if model_name == "scGAN":
            for cli_key, cfg_key, caster in [
                ("--latent-dim", "latent_dim", int),
                ("--lr-gen", "lr_gen", float),
                ("--lr-dis", "lr_dis", float),
                ("--n-critic", "n_critic", int),
                ("--lambda-gp", "lambda_gp", float),
                ("--lr-final", "lr_final", float),
            ]:
                if params.get(cfg_key) is not None:
                    train_cmd.extend([cli_key, str(caster(params[cfg_key]))])
            if bool(params.get("use_lsn", False)):
                train_cmd.append("--use-lsn")
            if bool(params.get("lr_decay", False)):
                train_cmd.append("--lr-decay")
        if model_name == "scDiffusion":
            if diffusion_timesteps is not None:
                train_cmd.extend(["--n-timesteps", str(int(diffusion_timesteps))])
            if params.get("lr") is not None:
                train_cmd.extend(["--lr", str(float(params["lr"]))])
            if params.get("use_vae") is True:
                train_cmd.append("--use-vae")
            elif params.get("use_vae") is False:
                train_cmd.append("--no-vae")
            if params.get("vae_epochs") is not None:
                train_cmd.extend(["--vae-epochs", str(int(params["vae_epochs"]))])
            if params.get("vae_latent_dim") is not None:
                train_cmd.extend(["--vae-latent-dim", str(int(params["vae_latent_dim"]))])
        _run_cmd(train_cmd, env=env, dry_run=dry_run)

        gen_cmd = [
            python_bin,
            str(generate_script),
            "-i",
            str(resolve_path(split_path)),
            "--ckpt",
            str(ckpt),
            "-o",
            str(out),
            "-c",
            str(resolve_path(config_path)),
            "--slice-id",
            slice_id,
            "--label",
            label,
            "--ratio",
            str(float(pool_ratio)),
            "--device",
            device,
        ]
        if generate_batch_size is not None:
            gen_cmd.extend(["--batch-size", str(int(generate_batch_size))])
        if model_name == "scDiffusion" and use_ddim:
            gen_cmd.append("--use-ddim")
        _run_cmd(gen_cmd, env=env, dry_run=dry_run)

        _run_cmd(
            [
                python_bin,
                str(assign_script),
                "--synthetic",
                str(out),
                "--reference",
                str(resolve_path(split_path)),
                "--method",
                "label_spatial_perturbation",
                "--config",
                str(resolve_path(mapping_config)),
                "--slice-id",
                slice_id,
                "--generation-default",
            ],
            env=env,
            dry_run=dry_run,
        )
        if dry_run:
            continue

        syn = ad.read_h5ad(out)
        syn.obs["label"] = label
        syn.obs["condition_label"] = label
        syn.obs["source"] = model_name
        syn.obs["augmentation_source"] = "synthetic"
        syn.obs["split"] = "synthetic"
        syn.obs["dataset"] = "DLPFC"
        syn.obs["generator"] = model_name
        syn.obs["synthetic_id"] = syn.obs_names.astype(str)
        syn.obs["generation_scope"] = "slice_labelwise"
        syn.obs["condition_key"] = "label"
        syn.obs["label_fraction"] = float(label_fraction)
        syn.obs["low_label_seed"] = int(seed)
        syn.obs["pool_ratio"] = float(pool_ratio)
        syn.uns["task_family"] = "supervised_low_label"
        syn.uns["task_name"] = "dlpfc_slice_low_label_spot_clf"
        syn.uns["generation_scope"] = "slice_labelwise"
        syn.uns["label_strategy"] = "label_wise_generation"
        syn.uns["trusted_synthetic_labels"] = True
        syn.uns["generator"] = model_name
        syn.uns["dataset"] = "DLPFC"
        syn.uns["generation_mode"] = "supervised"
        syn.write_h5ad(out)
        parts.append(syn)
        condition_files.append(str(out))
        print(f"[{model_name} seed={seed}] generated slice={slice_id} label={label}: {syn.n_obs}", flush=True)

    if dry_run:
        print(f"[{model_name} seed={seed}] dry-run finished; no synthetic pool was written", flush=True)
        return base_out

    if not write_combined:
        write_json(
            condition_root / f"generation_manifest_{'_'.join(sorted(conditions['slice_id'].astype(str).unique()))}.json",
            {
                "model": model_name,
                "dataset": "DLPFC",
                "task_family": "supervised_low_label",
                "task_name": "dlpfc_slice_low_label_spot_clf",
                "label_fraction": float(label_fraction),
                "seed": int(seed),
                "pool_ratio": float(pool_ratio),
                "split_path": str(resolve_path(split_path)),
                "synthetic_pool_path": str(base_out),
                "generation_scope": "slice_labelwise",
                "condition_files": condition_files,
                "slices": sorted(conditions["slice_id"].astype(str).unique().tolist()),
                "merge_required": True,
            },
        )
        print(f"[{model_name} seed={seed}] wrote slice shards under {condition_root}; merge required", flush=True)
        return base_out

    combined = ad.concat(parts, join="inner", merge="same", uns_merge="unique", index_unique=None)
    combined.obs_names_make_unique()
    combined.uns["dataset"] = "DLPFC"
    combined.uns["task_family"] = "supervised_low_label"
    combined.uns["task_name"] = "dlpfc_slice_low_label_spot_clf"
    combined.uns["split_mode"] = "supervised"
    combined.uns["generation_mode"] = "supervised"
    combined.uns["generation_scope"] = "slice_labelwise"
    combined.uns["condition_key"] = "label"
    combined.uns["label_strategy"] = "label_wise_generation"
    combined.uns["trusted_synthetic_labels"] = True
    combined.uns["generator"] = model_name
    combined.uns["label_fraction"] = float(label_fraction)
    combined.uns["low_label_seed"] = int(seed)
    combined.uns["pool_ratio"] = float(pool_ratio)
    combined.uns["pool_name"] = pool_tag(pool_ratio)
    combined.uns["coord_method"] = "label_spatial_perturbation_generation_default"
    combined.uns["condition_files"] = condition_files
    base_out.parent.mkdir(parents=True, exist_ok=True)
    finalize_pool_schema(combined, model_name).write_h5ad(base_out)
    write_json(
        base_out.parent / "generation_manifest.json",
        {
            "model": model_name,
            "dataset": "DLPFC",
            "task_family": "supervised_low_label",
            "task_name": "dlpfc_slice_low_label_spot_clf",
            "label_fraction": float(label_fraction),
            "seed": int(seed),
            "pool_ratio": float(pool_ratio),
            "split_path": str(resolve_path(split_path)),
            "synthetic_pool_path": str(base_out),
            "generation_scope": "slice_labelwise",
            "condition_files": condition_files,
        },
    )
    return base_out


def generate_scdiffusion_slice_classifier_guided_pool(
    adata: ad.AnnData,
    conditions: pd.DataFrame,
    split_path: str | Path,
    output_root: str | Path,
    model_dir: str | Path,
    config_path: str | Path,
    mapping_config: str | Path,
    label_fraction: float,
    seed: int,
    pool_ratio: float,
    python_bin: str,
    device: str,
    deep_epochs: int | None,
    deep_batch_size: int | None,
    deep_n_timesteps: int | None,
    use_ddim: bool,
    task_config: dict[str, Any],
    write_combined: bool,
    skip_existing: bool,
    dry_run: bool,
) -> Path:
    """Train one slice-level scDiffusion checkpoint and sample labels via classifier guidance."""
    model_name = "scDiffusion"
    params = model_overrides(task_config, model_name)
    train_epochs = deep_epochs if deep_epochs is not None else params.get("epochs")
    train_batch_size = deep_batch_size if deep_batch_size is not None else params.get("batch_size")
    generate_batch_size = params.get("generate_batch_size", train_batch_size)
    diffusion_timesteps = deep_n_timesteps if deep_n_timesteps is not None else params.get("n_timesteps")
    use_ddim = bool(use_ddim or params.get("use_ddim", False))
    base_out = synthetic_pool_path(output_root, model_name, label_fraction, seed, pool_ratio=pool_ratio)
    condition_root = base_out.parent / "conditions"
    ckpt_root = (
        resolve_path(model_dir)
        / "supervised_low_label"
        / "DLPFC"
        / model_name
        / fraction_tag(label_fraction)
        / f"seed{seed}"
    )
    if not dry_run:
        condition_root.mkdir(parents=True, exist_ok=True)
        ckpt_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "spaug_numba_cache"))

    train_obs = adata.obs[adata.obs["split"].astype(str) == "train"].copy()
    parts = []
    condition_files = []
    script_root = PROJECT_ROOT / "src" / "02_generators" / "supervised_low_label" / "DLPFC" / model_name
    train_script = script_root / "train.py"
    generate_script = script_root / "generate.py"
    assign_script = script_root / "assign_coords.py"

    for slice_id, slice_conditions in conditions.groupby("slice_id", sort=True):
        slice_id = str(slice_id)
        labels = sorted(slice_conditions["label"].astype(str).unique().tolist())
        label_outs = {
            label: condition_root / f"slice_{safe_token(slice_id)}" / safe_token(label) / "synthetic.h5ad"
            for label in labels
        }
        if skip_existing and all(path.exists() for path in label_outs.values()):
            print(f"[scDiffusion seed={seed}] skip existing classifier-guided slice={slice_id}: {len(labels)} labels", flush=True)
            condition_files.extend(str(path) for path in label_outs.values())
            if write_combined:
                parts.extend(ad.read_h5ad(path) for path in label_outs.values())
            continue

        ckpt = ckpt_root / f"scDiffusion_DLPFC_supervised_low_label_{safe_token(slice_id)}_classifier_guided.pt"
        train_cmd = [
            python_bin,
            str(train_script),
            "-i",
            str(resolve_path(split_path)),
            "-o",
            str(ckpt),
            "-c",
            str(resolve_path(config_path)),
            "--slice-id",
            slice_id,
            "--classifier-key",
            "label",
            "--device",
            device,
        ]
        if train_epochs is not None:
            train_cmd.extend(["--epochs", str(int(train_epochs))])
        if train_batch_size is not None:
            train_cmd.extend(["--batch-size", str(int(train_batch_size))])
        if diffusion_timesteps is not None:
            train_cmd.extend(["--n-timesteps", str(int(diffusion_timesteps))])
        for cli_key, cfg_key, caster in [
            ("--lr", "lr", float),
            ("--classifier-epochs", "classifier_epochs", int),
            ("--classifier-lr", "classifier_lr", float),
            ("--vae-epochs", "vae_epochs", int),
            ("--vae-latent-dim", "vae_latent_dim", int),
        ]:
            if params.get(cfg_key) is not None:
                train_cmd.extend([cli_key, str(caster(params[cfg_key]))])
        if params.get("use_vae") is True:
            train_cmd.append("--use-vae")
        elif params.get("use_vae") is False:
            train_cmd.append("--no-vae")
        _run_cmd(train_cmd, env=env, dry_run=dry_run)

        for label in labels:
            n_real = int(
                (
                    (train_obs["slice_id"].astype(str) == slice_id)
                    & (train_obs["label"].astype(str) == str(label))
                ).sum()
            )
            n_samples = max(1, int(round(n_real * pool_ratio)))
            out = label_outs[label]
            gen_cmd = [
                python_bin,
                str(generate_script),
                "-i",
                str(resolve_path(split_path)),
                "--ckpt",
                str(ckpt),
                "-o",
                str(out),
                "-c",
                str(resolve_path(config_path)),
                "--slice-id",
                slice_id,
                "--label",
                str(label),
                "--guidance-label",
                str(label),
                "--n-samples",
                str(n_samples),
                "--device",
                device,
            ]
            if generate_batch_size is not None:
                gen_cmd.extend(["--batch-size", str(int(generate_batch_size))])
            if use_ddim:
                gen_cmd.append("--use-ddim")
            if params.get("guidance_scale") is not None:
                gen_cmd.extend(["--guidance-scale", str(float(params["guidance_scale"]))])
            _run_cmd(gen_cmd, env=env, dry_run=dry_run)
            _run_cmd(
                [
                    python_bin,
                    str(assign_script),
                    "--synthetic",
                    str(out),
                    "--reference",
                    str(resolve_path(split_path)),
                    "--method",
                    "label_spatial_perturbation",
                    "--config",
                    str(resolve_path(mapping_config)),
                    "--slice-id",
                    slice_id,
                    "--generation-default",
                ],
                env=env,
                dry_run=dry_run,
            )
            if dry_run:
                continue

            syn = ad.read_h5ad(out)
            syn.obs["label"] = str(label)
            syn.obs["condition_label"] = str(label)
            syn.obs["source"] = model_name
            syn.obs["augmentation_source"] = "synthetic"
            syn.obs["split"] = "synthetic"
            syn.obs["dataset"] = "DLPFC"
            syn.obs["generator"] = model_name
            syn.obs["synthetic_id"] = syn.obs_names.astype(str)
            syn.obs["generation_scope"] = "slice_conditional"
            syn.obs["condition_key"] = "label"
            syn.obs["label_fraction"] = float(label_fraction)
            syn.obs["low_label_seed"] = int(seed)
            syn.obs["pool_ratio"] = float(pool_ratio)
            syn.uns["task_family"] = "supervised_low_label"
            syn.uns["task_name"] = "dlpfc_slice_low_label_spot_clf"
            syn.uns["generation_scope"] = "slice_conditional"
            syn.uns["label_strategy"] = "classifier_guidance"
            syn.uns["trusted_synthetic_labels"] = True
            syn.uns["generator"] = model_name
            syn.uns["dataset"] = "DLPFC"
            syn.uns["generation_mode"] = "supervised"
            syn.write_h5ad(out)
            parts.append(syn)
            condition_files.append(str(out))
            print(f"[scDiffusion seed={seed}] generated classifier-guided slice={slice_id} label={label}: {syn.n_obs}", flush=True)

    if dry_run:
        print(f"[scDiffusion seed={seed}] dry-run finished; no synthetic pool was written", flush=True)
        return base_out

    if not write_combined:
        write_json(
            condition_root / f"generation_manifest_{'_'.join(sorted(conditions['slice_id'].astype(str).unique()))}.json",
            {
                "model": model_name,
                "dataset": "DLPFC",
                "task_family": "supervised_low_label",
                "task_name": "dlpfc_slice_low_label_spot_clf",
                "label_fraction": float(label_fraction),
                "seed": int(seed),
                "pool_ratio": float(pool_ratio),
                "split_path": str(resolve_path(split_path)),
                "synthetic_pool_path": str(base_out),
                "generation_scope": "slice_conditional",
                "label_strategy": "classifier_guidance",
                "condition_files": condition_files,
                "slices": sorted(conditions["slice_id"].astype(str).unique().tolist()),
                "merge_required": True,
            },
        )
        return base_out

    combined = ad.concat(parts, join="inner", merge="same", uns_merge="unique", index_unique=None)
    combined.obs_names_make_unique()
    combined.uns["dataset"] = "DLPFC"
    combined.uns["task_family"] = "supervised_low_label"
    combined.uns["task_name"] = "dlpfc_slice_low_label_spot_clf"
    combined.uns["split_mode"] = "supervised"
    combined.uns["generation_mode"] = "supervised"
    combined.uns["generation_scope"] = "slice_conditional"
    combined.uns["condition_key"] = "label"
    combined.uns["label_strategy"] = "classifier_guidance"
    combined.uns["trusted_synthetic_labels"] = True
    combined.uns["generator"] = model_name
    combined.uns["label_fraction"] = float(label_fraction)
    combined.uns["low_label_seed"] = int(seed)
    combined.uns["pool_ratio"] = float(pool_ratio)
    combined.uns["pool_name"] = pool_tag(pool_ratio)
    combined.uns["coord_method"] = "label_spatial_perturbation_generation_default"
    combined.uns["condition_files"] = condition_files
    base_out.parent.mkdir(parents=True, exist_ok=True)
    finalize_pool_schema(combined, model_name).write_h5ad(base_out)
    write_json(
        base_out.parent / "generation_manifest.json",
        {
            "model": model_name,
            "dataset": "DLPFC",
            "task_family": "supervised_low_label",
            "task_name": "dlpfc_slice_low_label_spot_clf",
            "label_fraction": float(label_fraction),
            "seed": int(seed),
            "pool_ratio": float(pool_ratio),
            "split_path": str(resolve_path(split_path)),
            "synthetic_pool_path": str(base_out),
            "generation_scope": "slice_conditional",
            "label_strategy": "classifier_guidance",
            "condition_files": condition_files,
        },
    )
    return base_out


def generate_scgan_slice_conditional_pool(
    adata: ad.AnnData,
    conditions: pd.DataFrame,
    split_path: str | Path,
    output_root: str | Path,
    model_dir: str | Path,
    config_path: str | Path,
    mapping_config: str | Path,
    label_fraction: float,
    seed: int,
    pool_ratio: float,
    python_bin: str,
    device: str,
    deep_epochs: int | None,
    deep_batch_size: int | None,
    task_config: dict[str, Any],
    write_combined: bool,
    skip_existing: bool,
    dry_run: bool,
) -> Path:
    """Train one cscGAN-style scGAN checkpoint per slice, then sample each label condition."""
    model_name = "scGAN"
    params = model_overrides(task_config, model_name)
    train_epochs = deep_epochs if deep_epochs is not None else params.get("epochs")
    train_batch_size = deep_batch_size if deep_batch_size is not None else params.get("batch_size")
    generate_batch_size = params.get("generate_batch_size", train_batch_size)
    base_out = synthetic_pool_path(output_root, model_name, label_fraction, seed, pool_ratio=pool_ratio)
    condition_root = base_out.parent / "conditions"
    ckpt_root = (
        resolve_path(model_dir)
        / "supervised_low_label"
        / "DLPFC"
        / model_name
        / fraction_tag(label_fraction)
        / f"seed{seed}"
    )
    if not dry_run:
        condition_root.mkdir(parents=True, exist_ok=True)
        ckpt_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "spaug_numba_cache"))

    train_obs = adata.obs[adata.obs["split"].astype(str) == "train"].copy()
    parts = []
    condition_files = []
    train_script = PROJECT_ROOT / "src" / "02_generators" / "supervised_low_label" / "DLPFC" / model_name / "train.py"
    generate_script = PROJECT_ROOT / "src" / "02_generators" / "supervised_low_label" / "DLPFC" / model_name / "generate.py"
    assign_script = PROJECT_ROOT / "src" / "02_generators" / "supervised_low_label" / "DLPFC" / model_name / "assign_coords.py"

    for slice_id, slice_conditions in conditions.groupby("slice_id", sort=True):
        slice_id = str(slice_id)
        labels = sorted(slice_conditions["label"].astype(str).unique().tolist())
        label_outs = {
            label: condition_root / f"slice_{safe_token(slice_id)}" / safe_token(label) / "synthetic.h5ad"
            for label in labels
        }
        if skip_existing and all(path.exists() for path in label_outs.values()):
            print(f"[scGAN seed={seed}] skip existing conditional slice={slice_id}: {len(labels)} labels", flush=True)
            condition_files.extend(str(path) for path in label_outs.values())
            if write_combined:
                parts.extend(ad.read_h5ad(path) for path in label_outs.values())
            continue

        ckpt = ckpt_root / f"scGAN_DLPFC_supervised_low_label_{safe_token(slice_id)}_conditional.pth"
        train_cmd = [
            python_bin,
            str(train_script),
            "-i",
            str(resolve_path(split_path)),
            "-o",
            str(ckpt),
            "-c",
            str(resolve_path(config_path)),
            "--slice-id",
            slice_id,
            "--conditional-key",
            "label",
            "--device",
            device,
        ]
        if train_epochs is not None:
            train_cmd.extend(["--epochs", str(int(train_epochs))])
        if train_batch_size is not None:
            train_cmd.extend(["--batch-size", str(int(train_batch_size))])
        for cli_key, cfg_key, caster in [
            ("--latent-dim", "latent_dim", int),
            ("--lr-gen", "lr_gen", float),
            ("--lr-dis", "lr_dis", float),
            ("--n-critic", "n_critic", int),
            ("--lambda-gp", "lambda_gp", float),
            ("--lr-final", "lr_final", float),
            ("--condition-dim", "condition_dim", int),
        ]:
            if params.get(cfg_key) is not None:
                train_cmd.extend([cli_key, str(caster(params[cfg_key]))])
        if bool(params.get("use_lsn", False)):
            train_cmd.append("--use-lsn")
        if bool(params.get("lr_decay", False)):
            train_cmd.append("--lr-decay")
        if params.get("conditional_gan") is False:
            train_cmd.append("--no-conditional-gan")
        _run_cmd(train_cmd, env=env, dry_run=dry_run)

        for label in labels:
            n_real = int(
                (
                    (train_obs["slice_id"].astype(str) == slice_id)
                    & (train_obs["label"].astype(str) == str(label))
                ).sum()
            )
            n_samples = max(1, int(round(n_real * pool_ratio)))
            out = label_outs[label]
            gen_cmd = [
                python_bin,
                str(generate_script),
                "-i",
                str(resolve_path(split_path)),
                "--ckpt",
                str(ckpt),
                "-o",
                str(out),
                "-c",
                str(resolve_path(config_path)),
                "--slice-id",
                slice_id,
                "--label",
                str(label),
                "--condition-label",
                str(label),
                "--n-samples",
                str(n_samples),
                "--device",
                device,
            ]
            if generate_batch_size is not None:
                gen_cmd.extend(["--batch-size", str(int(generate_batch_size))])
            _run_cmd(gen_cmd, env=env, dry_run=dry_run)
            _run_cmd(
                [
                    python_bin,
                    str(assign_script),
                    "--synthetic",
                    str(out),
                    "--reference",
                    str(resolve_path(split_path)),
                    "--method",
                    "label_spatial_perturbation",
                    "--config",
                    str(resolve_path(mapping_config)),
                    "--slice-id",
                    slice_id,
                    "--generation-default",
                ],
                env=env,
                dry_run=dry_run,
            )
            if dry_run:
                continue

            syn = ad.read_h5ad(out)
            syn.obs["label"] = str(label)
            syn.obs["condition_label"] = str(label)
            syn.obs["source"] = model_name
            syn.obs["augmentation_source"] = "synthetic"
            syn.obs["split"] = "synthetic"
            syn.obs["dataset"] = "DLPFC"
            syn.obs["generator"] = model_name
            syn.obs["synthetic_id"] = syn.obs_names.astype(str)
            syn.obs["generation_scope"] = "slice_conditional"
            syn.obs["condition_key"] = "label"
            syn.obs["label_fraction"] = float(label_fraction)
            syn.obs["low_label_seed"] = int(seed)
            syn.obs["pool_ratio"] = float(pool_ratio)
            syn.uns["task_family"] = "supervised_low_label"
            syn.uns["task_name"] = "dlpfc_slice_low_label_spot_clf"
            syn.uns["generation_scope"] = "slice_conditional"
            syn.uns["label_strategy"] = "internal_cscgan_conditional_bn_projection"
            syn.uns["trusted_synthetic_labels"] = True
            syn.uns["generator"] = model_name
            syn.uns["dataset"] = "DLPFC"
            syn.uns["generation_mode"] = "supervised"
            syn.write_h5ad(out)
            parts.append(syn)
            condition_files.append(str(out))
            print(f"[scGAN seed={seed}] generated conditional slice={slice_id} label={label}: {syn.n_obs}", flush=True)

    if dry_run:
        print(f"[scGAN seed={seed}] dry-run finished; no synthetic pool was written", flush=True)
        return base_out

    if not write_combined:
        write_json(
            condition_root / f"generation_manifest_{'_'.join(sorted(conditions['slice_id'].astype(str).unique()))}.json",
            {
                "model": model_name,
                "dataset": "DLPFC",
                "task_family": "supervised_low_label",
                "task_name": "dlpfc_slice_low_label_spot_clf",
                "label_fraction": float(label_fraction),
                "seed": int(seed),
                "pool_ratio": float(pool_ratio),
                "split_path": str(resolve_path(split_path)),
                "synthetic_pool_path": str(base_out),
                "generation_scope": "slice_conditional",
                "label_strategy": "internal_cscgan_conditional_bn_projection",
                "condition_files": condition_files,
                "slices": sorted(conditions["slice_id"].astype(str).unique().tolist()),
                "merge_required": True,
            },
        )
        return base_out

    combined = ad.concat(parts, join="inner", merge="same", uns_merge="unique", index_unique=None)
    combined.obs_names_make_unique()
    combined.uns["dataset"] = "DLPFC"
    combined.uns["task_family"] = "supervised_low_label"
    combined.uns["task_name"] = "dlpfc_slice_low_label_spot_clf"
    combined.uns["split_mode"] = "supervised"
    combined.uns["generation_mode"] = "supervised"
    combined.uns["generation_scope"] = "slice_conditional"
    combined.uns["condition_key"] = "label"
    combined.uns["label_strategy"] = "internal_cscgan_conditional_bn_projection"
    combined.uns["trusted_synthetic_labels"] = True
    combined.uns["generator"] = model_name
    combined.uns["label_fraction"] = float(label_fraction)
    combined.uns["low_label_seed"] = int(seed)
    combined.uns["pool_ratio"] = float(pool_ratio)
    combined.uns["pool_name"] = pool_tag(pool_ratio)
    combined.uns["coord_method"] = "label_spatial_perturbation_generation_default"
    combined.uns["condition_files"] = condition_files
    base_out.parent.mkdir(parents=True, exist_ok=True)
    finalize_pool_schema(combined, model_name).write_h5ad(base_out)
    write_json(
        base_out.parent / "generation_manifest.json",
        {
            "model": model_name,
            "dataset": "DLPFC",
            "task_family": "supervised_low_label",
            "task_name": "dlpfc_slice_low_label_spot_clf",
            "label_fraction": float(label_fraction),
            "seed": int(seed),
            "pool_ratio": float(pool_ratio),
            "split_path": str(resolve_path(split_path)),
            "synthetic_pool_path": str(base_out),
            "generation_scope": "slice_conditional",
            "label_strategy": "internal_cscgan_conditional_bn_projection",
            "condition_files": condition_files,
        },
    )
    return base_out


def generate_model_seed(
    model_name: str,
    split_path: str | Path,
    output_root: str | Path,
    model_dir: str | Path,
    config_path: str | Path,
    mapping_config: str | Path,
    label_fraction: float,
    seed: int,
    pool_ratio: float,
    python_bin: str,
    device: str,
    deep_epochs: int | None,
    deep_batch_size: int | None,
    deep_n_timesteps: int | None,
    use_ddim: bool,
    task_config: dict[str, Any],
    slice_filter: set[str] | None,
    write_combined: bool,
    skip_existing: bool,
    dry_run: bool,
) -> Path:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Supported models are {SUPPORTED_MODELS}; got {model_name}")
    if model_name in EXTERNAL_LABELWISE_MODELS:
        return generate_external_labelwise_pool(
            model_name=model_name,
            split_path=split_path,
            output_root=output_root,
            model_dir=model_dir,
            config_path=config_path,
            mapping_config=mapping_config,
            label_fraction=label_fraction,
            seed=seed,
            pool_ratio=pool_ratio,
            python_bin=python_bin,
            device=device,
            deep_epochs=deep_epochs,
            deep_batch_size=deep_batch_size,
            deep_n_timesteps=deep_n_timesteps,
            use_ddim=use_ddim,
            task_config=task_config,
            slice_filter=slice_filter,
            write_combined=write_combined,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )

    adata = read_split(split_path)
    params = model_overrides(task_config, model_name)
    slices = sorted(adata.obs["slice_id"].astype(str).unique().tolist())
    if slice_filter is not None:
        slices = [slice_id for slice_id in slices if slice_id in slice_filter]
    if not slices:
        raise ValueError(f"No slices remain after slice filtering: {sorted(slice_filter or [])}")
    pools = []
    slice_paths = []
    base_out = synthetic_pool_path(output_root, model_name, label_fraction, seed, pool_ratio=pool_ratio)
    if dry_run:
        print(
            f"[{model_name} seed={seed}] dry-run target {base_out}; "
            f"slices={','.join(slices)}; pool_ratio={float(pool_ratio):g}",
            flush=True,
        )
        return base_out
    slice_root = base_out.parent / "slices"
    slice_root.mkdir(parents=True, exist_ok=True)
    for slice_id in slices:
        out_path = slice_root / f"slice_{safe_token(slice_id)}.h5ad"
        if skip_existing and out_path.exists():
            slice_paths.append(str(out_path))
            print(f"[{model_name} seed={seed}] skip existing slice {slice_id}: {out_path}", flush=True)
            if write_combined:
                pools.append(ad.read_h5ad(out_path))
            continue
        pool = generate_slice_pool(
            adata,
            model_name=model_name,
            slice_id=slice_id,
            label_fraction=label_fraction,
            seed=seed,
            pool_ratio=pool_ratio,
            params=params,
        )
        pool.write_h5ad(out_path)
        pools.append(pool)
        slice_paths.append(str(out_path))
        print(f"[{model_name} seed={seed}] generated slice {slice_id}: {pool.n_obs} synthetic spots", flush=True)

    if not write_combined:
        write_json(
            slice_root / f"generation_manifest_{'_'.join(safe_token(x) for x in slices)}.json",
            {
                "model": model_name,
                "dataset": "DLPFC",
                "task_family": "supervised_low_label",
                "task_name": "dlpfc_slice_low_label_spot_clf",
                "label_fraction": float(label_fraction),
                "seed": int(seed),
                "pool_ratio": float(pool_ratio),
                "split_path": str(resolve_path(split_path)),
                "synthetic_pool_path": str(base_out),
                "slices": slices,
                "slice_files": slice_paths,
                "merge_required": True,
            },
        )
        print(f"[{model_name} seed={seed}] wrote slice shards under {slice_root}; merge required", flush=True)
        return base_out

    combined = ad.concat(pools, join="inner", merge="same", uns_merge="unique", index_unique=None)
    combined.uns["dataset"] = "DLPFC"
    combined.uns["task_family"] = "supervised_low_label"
    combined.uns["task_name"] = "dlpfc_slice_low_label_spot_clf"
    combined.uns["split_mode"] = "supervised"
    combined.uns["generation_mode"] = "supervised"
    combined.uns["generation_scope"] = "slice_conditional"
    combined.uns["condition_key"] = "label"
    combined.uns["label_strategy"] = "slice_conditional_label_generation"
    combined.uns["trusted_synthetic_labels"] = True
    combined.uns["generator"] = model_name
    combined.uns["label_fraction"] = float(label_fraction)
    combined.uns["low_label_seed"] = int(seed)
    combined.uns["pool_ratio"] = float(pool_ratio)
    combined.uns["pool_name"] = pool_tag(pool_ratio)
    combined.uns["slice_files"] = slice_paths
    base_out.parent.mkdir(parents=True, exist_ok=True)
    finalize_pool_schema(combined, model_name).write_h5ad(base_out)
    write_json(
        base_out.parent / "generation_manifest.json",
        {
            "model": model_name,
            "dataset": "DLPFC",
            "task_family": "supervised_low_label",
            "task_name": "dlpfc_slice_low_label_spot_clf",
            "label_fraction": float(label_fraction),
            "seed": int(seed),
            "pool_ratio": float(pool_ratio),
            "split_path": str(resolve_path(split_path)),
            "synthetic_pool_path": str(base_out),
            "slices": slices,
            "slice_files": slice_paths,
        },
    )
    return base_out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DLPFC supervised low-label synthetic pools.")
    parser.add_argument("--models", nargs="*", default=list(SUPPORTED_MODELS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--label-fraction", type=float, default=0.50)
    parser.add_argument("--pool-ratio", type=float, default=40.0)
    parser.add_argument(
        "--split-template",
        default="data/02_interim/supervised_low_label/DLPFC/model_inputs/{model}/{fraction_tag}/seed{seed}/processed_with_split.h5ad",
    )
    parser.add_argument("--output-root", default="data/03_synthetic")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("-c", "--config", default="configs/supervised_low_label/DLPFC/generators.yaml")
    parser.add_argument("--mapping-config", default="configs/supervised_low_label/DLPFC/mapping.yaml")
    parser.add_argument("--low-label-config", default="configs/supervised_low_label/DLPFC/supervised_low_label_generators.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--deep-epochs", type=int, default=None)
    parser.add_argument("--deep-batch-size", type=int, default=None)
    parser.add_argument("--deep-n-timesteps", type=int, default=None)
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--slices", nargs="*", default=None, help="Optional slice_id list for slice-level parallel generation.")
    parser.add_argument("--no-merge", action="store_true", help="Write slice/condition shards and merge them with merge_synthetic.py.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing slice/condition shard files.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    task_config = load_yaml(args.low_label_config)
    tag = fraction_tag(args.label_fraction)
    slice_filter = normalize_slices(args.slices)
    for model_name in args.models:
        for seed in args.seeds:
            split_path = args.split_template.format(fraction_tag=tag, seed=seed, model=model_name)
            out = generate_model_seed(
                model_name=model_name,
                split_path=split_path,
                output_root=args.output_root,
                model_dir=args.model_dir,
                config_path=args.config,
                mapping_config=args.mapping_config,
                label_fraction=args.label_fraction,
                seed=seed,
                pool_ratio=args.pool_ratio,
                python_bin=args.python,
                device=args.device,
                deep_epochs=args.deep_epochs,
                deep_batch_size=args.deep_batch_size,
                deep_n_timesteps=args.deep_n_timesteps,
                use_ddim=args.use_ddim,
                task_config=task_config,
                slice_filter=slice_filter,
                write_combined=not args.no_merge,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
            )
            action = "dry-run target" if args.dry_run else "wrote"
            print(f"[{model_name} seed={seed}] {action} {out}", flush=True)


if __name__ == "__main__":
    main()
