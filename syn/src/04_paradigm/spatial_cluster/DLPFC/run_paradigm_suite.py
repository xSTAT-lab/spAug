"""Build parameterized augmentation packages by dispatching paradigm modules."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import anndata as ad


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path = [p for p in sys.path if Path(p or os.getcwd()).resolve() != PROJECT_ROOT]

try:
    from . import global_concat, local_spatial
    from .common import (
        DEFAULT_FAMILIES,
        DLPFC_ONLY_FAMILIES,
        align_gene_space,
        infer_dataset,
        infer_mode,
        load_capability,
        load_yaml,
        parse_float_list,
        parse_str_list,
        resolve_path,
        subset_fraction,
        subset_real_for_paradigm,
        validate_synthetic_for_paradigm,
    )
except ImportError:
    import global_concat
    import local_spatial
    from common import (
        DEFAULT_FAMILIES,
        DLPFC_ONLY_FAMILIES,
        align_gene_space,
        infer_dataset,
        infer_mode,
        load_capability,
        load_yaml,
        parse_float_list,
        parse_str_list,
        resolve_path,
        subset_fraction,
        subset_real_for_paradigm,
        validate_synthetic_for_paradigm,
    )


def dispatch_family(
    family: str,
    real_subset: ad.AnnData,
    synthetic: ad.AnnData,
    output_root: Path,
    dataset: str,
    mode: str,
    model: str,
    capability: dict,
    synthetic_ratio: float,
    real_fraction: float,
    seed: int,
    pca_fit_policy: str,
    reference_path: str | Path,
    coord_policy: str = "pool_default",
    mapping_config: str | Path = "configs/spatial_cluster/DLPFC/mapping.yaml",
) -> list[Path]:
    common = {
        "real": real_subset,
        "synthetic": synthetic,
        "output_root": output_root,
        "dataset": dataset,
        "mode": mode,
        "model": model,
        "capability": capability,
        "synthetic_ratio": synthetic_ratio,
        "real_fraction": real_fraction,
        "seed": seed,
        "pca_fit_policy": pca_fit_policy,
        "coord_policy": coord_policy,
        "mapping_config": mapping_config,
        "reference_path": reference_path,
    }
    if family == global_concat.FAMILY:
        return [global_concat.build(**common)]
    if family == local_spatial.FAMILY:
        return [local_spatial.build(**common)]
    raise ValueError(f"Unknown paradigm family: {family}")


def infer_task_name(dataset: str, mode: str, task_name: Optional[str] = None) -> str:
    if task_name:
        return str(task_name)
    if dataset == "DLPFC" and mode == "supervised":
        return "spot_clf"
    if dataset == "DLPFC" and mode == "unsupervised":
        return "spatial_cluster"
    if dataset == "Trastuzumab" and mode == "supervised":
        return "response_pred"
    return "unknown_task"


def standard_output_root(base: str | Path, model: str, dataset: str, task_name: str, mode: str) -> Path:
    return resolve_path(base) / task_name / dataset / model / mode


def annotate_manifests(
    paths: list[Path],
    task_name: str,
    standard_layout: bool,
    synthetic_pool_path: str | Path,
) -> list[Path]:
    for output in paths:
        manifest_path = output / "manifest.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["task_name"] = task_name
        payload["standard_layout"] = bool(standard_layout)
        payload["synthetic_pool_path"] = str(resolve_path(synthetic_pool_path))
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def build_packages(
    real_path: str | Path,
    synthetic_path: str | Path,
    output_root: str | Path = "data/04_paradigm_augmented",
    dataset: Optional[str] = None,
    mode: Optional[str] = None,
    model: Optional[str] = None,
    task_name: Optional[str] = None,
    standard_layout: bool = True,
    families: Optional[list[str]] = None,
    synthetic_ratios: Optional[list[float]] = None,
    local_spatial_ratios: Optional[list[float]] = None,
    real_fractions: Optional[list[float]] = None,
    random_seed: int = 42,
    pca_fit_policy: str = "real_train_only",
    coord_policy: str = "pool_default",
    mapping_config: str | Path = "configs/spatial_cluster/DLPFC/mapping.yaml",
) -> list[Path]:
    real_path = resolve_path(real_path)
    synthetic_path = resolve_path(synthetic_path)
    real = ad.read_h5ad(real_path)
    synthetic = ad.read_h5ad(synthetic_path)
    dataset = dataset or infer_dataset(real, real_path)
    mode = mode or infer_mode(real, real_path)
    if model is None:
        model = str(synthetic.uns.get("generator", "") or "")
    if not model:
        model = next(
            (
                name
                for name in ("SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion")
                if name in synthetic_path.parts
            ),
            "unknown",
        )
    task_name = infer_task_name(dataset, mode, task_name=task_name)
    output_root = (
        standard_output_root(output_root, model=model, dataset=dataset, task_name=task_name, mode=mode)
        if standard_layout
        else resolve_path(output_root)
    )
    families = families or DEFAULT_FAMILIES
    synthetic_ratios = synthetic_ratios or [1.0]
    local_spatial_ratios = local_spatial_ratios or synthetic_ratios
    real_fractions = real_fractions or [1.0]
    capability = load_capability(model)

    validate_synthetic_for_paradigm(synthetic, dataset=dataset, mode=mode)
    real_train = subset_real_for_paradigm(real, dataset=dataset, mode=mode)
    real_train, synthetic = align_gene_space(real_train, synthetic, strategy="intersection")

    outputs: list[Path] = []
    for family in families:
        if dataset != "DLPFC" and family in DLPFC_ONLY_FAMILIES:
            continue
        ratio_grid = local_spatial_ratios if family == "local_spatial" else synthetic_ratios
        for real_fraction in real_fractions:
            real_subset = subset_fraction(
                real_train,
                real_fraction,
                seed=random_seed,
                group_cols=("slice_id", "label"),
            )
            for synthetic_ratio in ratio_grid:
                built = dispatch_family(
                    family=family,
                    real_subset=real_subset,
                    synthetic=synthetic,
                    output_root=output_root,
                    dataset=dataset,
                    mode=mode,
                    model=model,
                    capability=capability,
                    synthetic_ratio=synthetic_ratio,
                    real_fraction=real_fraction,
                    seed=random_seed,
                    pca_fit_policy=pca_fit_policy,
                    reference_path=real_path,
                    coord_policy=coord_policy,
                    mapping_config=mapping_config,
                )
                outputs.extend(
                    annotate_manifests(
                        built,
                        task_name=task_name,
                        standard_layout=standard_layout,
                        synthetic_pool_path=synthetic_path,
                    )
                )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build parameterized paradigm packages.")
    parser.add_argument("-r", "--real", required=True)
    parser.add_argument("-s", "--synthetic", required=True)
    parser.add_argument(
        "-o",
        "--output-root",
        default="data/04_paradigm_augmented",
        help="Base output directory. With standard layout, task/dataset/model/mode are appended.",
    )
    parser.add_argument("-c", "--config", default="configs/spatial_cluster/DLPFC/paradigm.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--mode", default=None, choices=[None, "supervised", "unsupervised"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default=None, help="Task name for the standard output layout")
    parser.add_argument(
        "--no-standard-layout",
        action="store_true",
        help="Use --output-root as the complete destination path.",
    )
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--synthetic-ratios", nargs="*", default=None)
    parser.add_argument("--real-fractions", nargs="*", default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--coord-policy",
        default="pool_default",
        choices=["pool_default", "spatial_resampling", "knn_mapping", "spatial_perturbation", "gmm_sampling"],
    )
    parser.add_argument("--mapping-config", default="configs/spatial_cluster/DLPFC/mapping.yaml")
    return parser


def main():
    args = build_parser().parse_args()
    suite_cfg = load_yaml(args.config).get("paradigm_suite", {})
    cli_ratios = parse_float_list(args.synthetic_ratios, [])
    outputs = build_packages(
        real_path=args.real,
        synthetic_path=args.synthetic,
        output_root=args.output_root,
        dataset=args.dataset,
        mode=args.mode,
        model=args.model,
        task_name=args.task,
        standard_layout=not args.no_standard_layout,
        families=parse_str_list(args.families, suite_cfg.get("families", DEFAULT_FAMILIES)),
        synthetic_ratios=cli_ratios or suite_cfg.get("synthetic_ratios", [1.0]),
        local_spatial_ratios=cli_ratios or suite_cfg.get("local_spatial_max_ratios", suite_cfg.get("synthetic_ratios", [1.0])),
        real_fractions=parse_float_list(args.real_fractions, suite_cfg.get("real_fractions", [1.0])),
        random_seed=args.random_seed,
        pca_fit_policy=str(suite_cfg.get("pca_fit_policy", "real_train_only")),
        coord_policy=args.coord_policy,
        mapping_config=args.mapping_config,
    )
    for output in outputs:
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
