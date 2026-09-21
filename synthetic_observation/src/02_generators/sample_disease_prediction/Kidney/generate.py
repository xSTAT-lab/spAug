"""Unified formal Kidney sample-level synthetic generation.

This entrypoint uses the copied DLPFC formal model implementations. For each
Kidney sample, it trains one generator checkpoint from that sample's real spots
and then generates a same-sample synthetic pool.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import subprocess
import sys
from pathlib import Path

import anndata as ad

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import resolve_path  # type: ignore
else:
    from .common import resolve_path


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
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
    return str(value).replace("/", "-").replace("\\", "-").replace(" ", "_").replace(":", "-")


def run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool = False):
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def sample_ids_from_input(path: str | Path) -> list[str]:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing Kidney generator input: {path}")
    adata = ad.read_h5ad(path, backed="r")
    try:
        if str(adata.uns.get("dataset", "")) != "Kidney":
            raise ValueError(f"Expected Kidney input, got dataset={adata.uns.get('dataset')!r}")
        if "sample_id" not in adata.obs.columns:
            raise ValueError("Kidney generator input must contain obs['sample_id']")
        if "disease_label" not in adata.obs.columns:
            raise ValueError("Kidney generator input must contain obs['disease_label']")
        if "spatial" not in adata.obsm:
            raise ValueError("Kidney generator input must contain obsm['spatial']")
        samples = sorted(adata.obs["sample_id"].astype(str).unique().tolist())
        if not samples:
            raise ValueError("No Kidney samples found for generation")
        return samples
    finally:
        adata.file.close()


def checkpoint_path(model_dir: str | Path, model: str, sample_id: str) -> Path:
    return (
        resolve_path(model_dir)
        / "sample_disease_prediction"
        / "Kidney"
        / model
        / f"{model}_Kidney_{safe_token(sample_id)}{CHECKPOINT_SUFFIX[model]}"
    )


def output_path(output_root: str | Path, model: str, sample_id: str) -> Path:
    return resolve_path(output_root) / "sample_disease_prediction" / "Kidney" / model / f"{safe_token(sample_id)}.h5ad"


def train_and_generate_sample(
    model: str,
    input_path: str | Path,
    output_root: str | Path,
    model_dir: str | Path,
    config_path: str | Path,
    mapping_config: str | Path,
    sample_id: str,
    pool_ratio: float,
    python_bin: str,
    device: str,
    coord_method: str,
    dry_run: bool,
    skip_existing: bool = False,
):
    model_root = PROJECT_ROOT / "src" / "02_generators" / "sample_disease_prediction" / "Kidney" / model
    train_script = model_root / "train.py"
    generate_script = model_root / "generate.py"
    validate_script = PROJECT_ROOT / "src" / "02_generators" / "sample_disease_prediction" / "Kidney" / "validate_synthetic_schema.py"
    ckpt = checkpoint_path(model_dir, model, sample_id)
    out = output_path(output_root, model, sample_id)
    if not dry_run:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        out.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and out.exists():
        print(f"[{model}] skip existing {out}", flush=True)
        return

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
        str(sample_id),
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
        str(sample_id),
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
                str(sample_id),
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
            "Kidney",
            "--mode",
            "unsupervised",
        ],
        env=env,
        dry_run=dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate formal Kidney sample-level synthetic pools.")
    parser.add_argument("--models", nargs="*", default=list(SUPPORTED_MODELS))
    parser.add_argument("--input", default="data/02_interim/sample_disease_prediction/Kidney/kidney_samples.h5ad")
    parser.add_argument("--output-root", default="data/03_synthetic")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("-c", "--config", default="configs/sample_disease_prediction/Kidney/generators.yaml")
    parser.add_argument("--mapping-config", default="configs/sample_disease_prediction/Kidney/mapping.yaml")
    parser.add_argument("--pool-ratio", type=float, default=20.0)
    parser.add_argument(
        "--coord-method",
        default="spatial_resampling",
        choices=["spatial_resampling", "knn_mapping", "spatial_perturbation", "gmm_sampling"],
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--sample-ids-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.sample_id and args.sample_ids_file:
        raise ValueError("Select one input: --sample-id or --sample-ids-file")
    if args.sample_id:
        samples = [args.sample_id]
    elif args.sample_ids_file:
        with resolve_path(args.sample_ids_file).open("r", encoding="utf-8") as f:
            samples = [line.strip() for line in f if line.strip()]
    else:
        samples = sample_ids_from_input(args.input)
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]
    for model in args.models:
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")
        print(f"[{model}] Kidney samples={len(samples)}", flush=True)
        for sample_id in samples:
            train_and_generate_sample(
                model=model,
                input_path=args.input,
                output_root=args.output_root,
                model_dir=args.model_dir,
                config_path=args.config,
                mapping_config=args.mapping_config,
                sample_id=str(sample_id),
                pool_ratio=args.pool_ratio,
                python_bin=args.python,
                device=args.device,
                coord_method=args.coord_method,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )


if __name__ == "__main__":
    main()
