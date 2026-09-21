"""Global real + synthetic concatenation paradigm."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import argparse

try:
    from .common import (
        add_common_cli_args,
        apply_coord_policy,
        build_manifest,
        build_single_family_packages,
        concat_train,
        limit_synthetic_like_real,
        parse_float_list,
        variant_id,
        write_package,
    )
except ImportError:
    from common import (
        add_common_cli_args,
        apply_coord_policy,
        build_manifest,
        build_single_family_packages,
        concat_train,
        limit_synthetic_like_real,
        parse_float_list,
        variant_id,
        write_package,
    )


FAMILY = "global_real_plus_synthetic"


def build(
    real: ad.AnnData,
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
    coord_policy: str = "pool_default",
    mapping_config: str = "configs/spatial_cluster/DLPFC/mapping.yaml",
    reference_path: str | Path = "",
) -> Path:
    syn_subset = limit_synthetic_like_real(real, synthetic, synthetic_ratio, seed)
    syn_subset = apply_coord_policy(syn_subset, real, coord_policy, mapping_config, reference_path)
    selection_policy = "matched_group_ratio"
    selection_policy_with_coord = f"{selection_policy}_{coord_policy}"
    variant = variant_id(FAMILY, synthetic_ratio, real_fraction, None, selection_policy_with_coord)
    train = concat_train(real, syn_subset)
    manifest = build_manifest(
        family=FAMILY,
        variant=variant,
        dataset=dataset,
        mode=mode,
        model=model,
        synthetic_ratio=synthetic_ratio,
        real_fraction=real_fraction,
        alpha=None,
        selection_policy=selection_policy_with_coord,
        real=real,
        synthetic=syn_subset,
        capability=capability,
        training_regime="concat",
        pca_fit_policy=pca_fit_policy,
        coord_policy=coord_policy,
    )
    return write_package(output_root, FAMILY, variant, manifest, train=train)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build global real + synthetic augmentation packages.")
    add_common_cli_args(parser)
    return parser


def main():
    args = build_parser().parse_args()
    outputs = build_single_family_packages(
        builder=build,
        family=FAMILY,
        real_path=args.real,
        synthetic_pool_path=args.synthetic_pool,
        output_root=args.output_root,
        config_path=args.config,
        dataset=args.dataset,
        mode=args.mode,
        model=args.model,
        task_name=args.task,
        standard_layout=not args.no_standard_layout,
        synthetic_ratios=parse_float_list(args.ratios, []),
        real_fractions=parse_float_list(args.real_fractions, []),
        alphas=None,
        random_seed=args.random_seed,
        coord_policy=args.coord_policy,
        mapping_config=args.mapping_config,
    )
    for output in outputs:
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
