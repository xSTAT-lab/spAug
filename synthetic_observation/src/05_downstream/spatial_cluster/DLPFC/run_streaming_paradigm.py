"""Build one paradigm in memory and immediately run downstream prediction.

This runner avoids writing data/04_paradigm_augmented/*.h5ad packages. It reads
the real split AnnData and a 40x synthetic pool once, samples one parameter
variant at a time, runs the downstream task, writes metrics under
data/05_results/spatial_cluster/DLPFC, and then releases the in-memory package.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import anndata as ad


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path = [p for p in sys.path if Path(p or os.getcwd()).resolve() != PROJECT_ROOT]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "04_paradigm" / "spatial_cluster" / "DLPFC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DEFAULT_FAMILIES,
    DEFAULT_REAL_FRACTIONS,
    DEFAULT_SYNTHETIC_RATIOS,
    DLPFC_ONLY_FAMILIES,
    align_gene_space,
    build_memory_package,
    infer_dataset,
    infer_mode,
    infer_task_name,
    load_capability,
    load_yaml,
    parse_float_list,
    parse_str_list,
    resolve_path,
    subset_fraction,
    subset_real_for_paradigm,
    validate_synthetic_for_paradigm,
)
from spagcn_cluster import run_spatial_clustering_many_seeds  # noqa: E402


def parse_int_list(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    out: list[int] = []
    for value in values:
        out.extend(int(x) for x in str(value).split(",") if x)
    return out


def enforce_final_spatial_cluster_contract(
    ablations: list[str],
    downstream_backends: list[str],
    cluster_seeds: list[int],
) -> tuple[list[str], list[str], list[int]]:
    ablations = [str(x) for x in ablations]
    downstream_backends = [str(x) for x in downstream_backends]
    cluster_seeds = [int(x) for x in cluster_seeds]
    if ablations != ["path_B"]:
        raise ValueError("DLPFC spatial_cluster formal run uses ablations=['path_B']")
    if downstream_backends != ["pca_spatial_leiden"]:
        raise ValueError("DLPFC spatial_cluster formal run uses backend=['pca_spatial_leiden']")
    if cluster_seeds != [42]:
        raise ValueError("DLPFC spatial_cluster formal run uses cluster_seeds=[42]")
    return ablations, downstream_backends, cluster_seeds


def result_dir(results_root: Path, manifest: dict, task_name: str, extra: str | None = None) -> Path:
    parts = [
        results_root,
        str(manifest.get("task_name", task_name)),
        str(manifest.get("dataset", "unknown")),
        str(manifest.get("generator", "unknown")),
        str(manifest.get("mode", "unknown")),
        str(manifest.get("paradigm_family", "unknown")),
        str(manifest.get("variant_id", "variant")),
    ]
    if extra is not None:
        parts.append(str(extra))
    return Path(*map(str, parts))


def prepare_inputs(
    real_path: str | Path,
    synthetic_pool_path: str | Path,
    dataset: str | None,
    mode: str | None,
    model: str | None,
):
    real_path = resolve_path(real_path)
    synthetic_pool_path = resolve_path(synthetic_pool_path)
    real = ad.read_h5ad(real_path)
    synthetic = ad.read_h5ad(synthetic_pool_path)
    dataset = dataset or infer_dataset(real, real_path)
    mode = mode or infer_mode(real, real_path)
    if model is None:
        model = str(synthetic.uns.get("generator", "") or "")
    if not model:
        model = next(
            (name for name in ("SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion") if name in synthetic_pool_path.parts),
            "unknown",
        )
    validate_synthetic_for_paradigm(synthetic, dataset=dataset, mode=mode)
    real_train = subset_real_for_paradigm(real, dataset=dataset, mode=mode)
    real_train, synthetic = align_gene_space(real_train, synthetic, strategy="intersection")
    return real_train, synthetic, dataset, mode, model


def run_package(
    package,
    task_name: str,
    reference_path: str | Path,
    results_root: Path,
    config_path: str | Path,
    split: str,
    labels_path: str | Path,
    classifiers: list[str] | None,
    slices: list[str],
    ablations: list[str],
    downstream_backends: list[str],
    cluster_seeds: list[int],
    skip_existing: bool = False,
):
    manifest = dict(package.manifest)
    manifest["task_name"] = task_name
    manifest["standard_layout"] = True
    manifest["input_mode"] = "in_memory"

    dataset = str(manifest.get("dataset", ""))
    mode = str(manifest.get("mode", ""))

    if dataset == "DLPFC" and mode == "unsupervised" and task_name == "spatial_cluster":
        if package.train is None:
            raise ValueError("spatial_cluster requires package.train")
        for sid in slices:
            for ablation in ablations:
                for backend in downstream_backends:
                    output_dirs_by_seed = {}
                    for cluster_seed in cluster_seeds:
                        backend_manifest = dict(manifest)
                        backend_manifest["downstream_backend"] = str(backend)
                        backend_manifest["cluster_seed"] = int(cluster_seed)
                        output_dir = result_dir(
                            results_root,
                            backend_manifest,
                            "spatial_cluster",
                            f"{sid}_{ablation}_{backend}_seed{cluster_seed}",
                        )
                        output_dirs_by_seed[int(cluster_seed)] = output_dir
                    run_spatial_clustering_many_seeds(
                        input_path=None,
                        input_adata=package.train,
                        output_dirs_by_seed=output_dirs_by_seed,
                        labels_path=labels_path,
                        config_path=config_path,
                        slice_id=sid,
                        ablation=ablation,
                        manifest_path=manifest,
                        backend=backend,
                        cluster_seeds=cluster_seeds,
                        skip_existing=skip_existing,
                    )
        return

    raise ValueError(f"Unsupported streaming task: dataset={dataset}, mode={mode}, task={task_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one paradigm directly in memory.")
    parser.add_argument("-r", "--real", required=True)
    parser.add_argument("-s", "--synthetic-pool", required=True)
    parser.add_argument("--family", required=True, choices=[
        "global_real_plus_synthetic",
        "local_spatial",
    ])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--mode", default=None, choices=[None, "supervised", "unsupervised"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--ratios", nargs="*", default=None)
    parser.add_argument("--real-fractions", nargs="*", default=None)
    parser.add_argument(
        "--coord-policy",
        default="pool_default",
        choices=["pool_default", "spatial_resampling", "knn_mapping", "spatial_perturbation", "gmm_sampling"],
    )
    parser.add_argument("--mapping-config", default="configs/spatial_cluster/DLPFC/mapping.yaml")
    parser.add_argument("--results-root", default="data/05_results")
    parser.add_argument("-c", "--config", default="configs/spatial_cluster/DLPFC/downstream.yaml")
    parser.add_argument("--paradigm-config", default="configs/spatial_cluster/DLPFC/paradigm.yaml")
    parser.add_argument("--reference", default=None, help="Reference split AnnData. Defaults to --real.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--labels", default="data/02_interim/spatial_cluster/DLPFC/evaluation/labels_for_evaluation.csv")
    parser.add_argument("--classifiers", nargs="*", default=None)
    parser.add_argument("--slices", nargs="*", default=[
        "151507", "151508", "151509", "151510", "151669", "151670",
        "151671", "151672", "151673", "151674", "151675", "151676",
    ])
    parser.add_argument("--ablations", nargs="*", default=["path_B"])
    parser.add_argument("--downstream-backends", nargs="*", default=None)
    parser.add_argument("--cluster-seeds", nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", "--random-seed", dest="random_seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N parameter packages.")
    return parser


def main():
    args = build_parser().parse_args()
    suite_cfg = load_yaml(args.paradigm_config).get("paradigm_suite", {})
    family_ratio_key = "local_spatial_max_ratios" if args.family == "local_spatial" else "synthetic_ratios"
    ratios = parse_float_list(args.ratios, suite_cfg.get(family_ratio_key, DEFAULT_SYNTHETIC_RATIOS))
    real_fractions = parse_float_list(args.real_fractions, suite_cfg.get("real_fractions", DEFAULT_REAL_FRACTIONS))
    pca_fit_policy = str(suite_cfg.get("pca_fit_policy", "real_train_only"))
    downstream_cfg = load_yaml(args.config).get("task3_spatial_cluster", {}).get("clustering", {})
    default_backend = str(downstream_cfg.get("backend", "pca_spatial_leiden"))
    downstream_backends = parse_str_list(args.downstream_backends, downstream_cfg.get("backends", [default_backend]))
    cluster_seeds = parse_int_list(args.cluster_seeds, downstream_cfg.get("seeds", [args.random_seed]))
    args.ablations, downstream_backends, cluster_seeds = enforce_final_spatial_cluster_contract(
        args.ablations,
        downstream_backends,
        cluster_seeds,
    )

    real_train, synthetic, dataset, mode, model = prepare_inputs(
        args.real, args.synthetic_pool, args.dataset, args.mode, args.model
    )
    task_name = infer_task_name(dataset, mode, args.task)
    if dataset != "DLPFC" and args.family in DLPFC_ONLY_FAMILIES:
        raise ValueError(f"{args.family} is only defined for DLPFC")
    capability = load_capability(model)
    results_root = resolve_path(args.results_root)
    reference_path = args.reference or args.real
    n_done = 0

    for real_fraction in real_fractions:
        real_subset = subset_fraction(
            real_train,
            real_fraction,
            seed=args.random_seed,
            group_cols=("slice_id", "label"),
        )
        for ratio in ratios:
            package = build_memory_package(
                family=args.family,
                real=real_subset,
                synthetic=synthetic,
                dataset=dataset,
                mode=mode,
                model=model,
                capability=capability,
                synthetic_ratio=ratio,
                real_fraction=real_fraction,
                alpha=None,
                seed=args.random_seed,
                pca_fit_policy=pca_fit_policy,
                regional_alpha={},
                coord_policy=args.coord_policy,
                mapping_config=args.mapping_config,
                reference_path=args.real,
            )
            package.manifest["synthetic_pool_path"] = str(resolve_path(args.synthetic_pool))
            run_package(
                package=package,
                task_name=task_name,
                reference_path=reference_path,
                results_root=results_root,
                config_path=args.config,
                split=args.split,
                labels_path=args.labels,
                classifiers=args.classifiers,
                slices=args.slices,
                ablations=args.ablations,
                downstream_backends=downstream_backends,
                cluster_seeds=cluster_seeds,
                skip_existing=args.skip_existing,
            )
            n_done += 1
            print(f"finished {package.manifest['variant_id']}")
            del package
            gc.collect()
            if args.limit is not None and n_done >= args.limit:
                return


if __name__ == "__main__":
    main()
