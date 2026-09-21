"""Prepare DLPFC cross-slice generalization split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata as ad
import pandas as pd

from io_utils import DATASET, TASK_NAME, resolve_path


DEFAULT_TRAIN_SLICES = [
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
]
DEFAULT_TEST_SLICES = ["151673", "151674", "151675", "151676"]
ALL_SLICES = [
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
]
DEFAULT_MODELS = ["SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion"]
PRESET_SPLITS = {
    "fixed_8train_4test": (
        DEFAULT_TRAIN_SLICES,
        DEFAULT_TEST_SLICES,
    ),
    "split_seed43_8train_4test": (
        ["151507", "151509", "151669", "151671", "151673", "151674", "151675", "151676"],
        ["151508", "151510", "151670", "151672"],
    ),
    "split_seed44_8train_4test": (
        ["151508", "151510", "151670", "151672", "151673", "151674", "151675", "151676"],
        ["151507", "151509", "151669", "151671"],
    ),
    "all_slices_pool": (
        ALL_SLICES,
        [],
    ),
}


def build_split(
    adata: ad.AnnData,
    train_slices: list[str],
    test_slices: list[str],
    split_tag: str,
) -> tuple[ad.AnnData, pd.DataFrame]:
    for col in ("slice_id", "label"):
        if col not in adata.obs.columns:
            raise ValueError(f"AnnData is missing obs[{col!r}]")
    if "spatial" not in adata.obsm:
        raise ValueError("AnnData is missing obsm['spatial']")
    train_set = {str(x) for x in train_slices}
    test_set = {str(x) for x in test_slices}
    overlap = train_set & test_set
    if overlap:
        raise ValueError(f"Train/test slices overlap: {sorted(overlap)}")

    obs_slices = set(adata.obs["slice_id"].astype(str).unique())
    missing = sorted((train_set | test_set) - obs_slices)
    if missing:
        raise ValueError(f"Requested slices not found in DLPFC data: {missing}")

    split = pd.Series("unused", index=adata.obs_names.astype(str), dtype=object)
    slice_values = adata.obs["slice_id"].astype(str)
    split.loc[slice_values.isin(train_set).to_numpy()] = "train"
    split.loc[slice_values.isin(test_set).to_numpy()] = "test"

    out = adata[split.to_numpy() != "unused"].copy()
    out.obs["split"] = split.loc[out.obs_names.astype(str)].to_numpy()
    out.obs["split_tag"] = split_tag
    out.uns["dataset"] = DATASET
    out.uns["task_family"] = TASK_NAME
    out.uns["task_name"] = "dlpfc_cross_slice_spot_clf"
    out.uns["split_mode"] = "supervised"
    out.uns["split_policy"] = "fixed_cross_slice_train_test"
    out.uns["split_tag"] = split_tag
    out.uns["train_slices"] = sorted(train_set)
    out.uns["test_slices"] = sorted(test_set)

    summary = (
        out.obs.groupby(["split", "slice_id", "label"], observed=False)
        .size()
        .reset_index(name="n_spots")
        .sort_values(["split", "slice_id", "label"])
    )
    return out, summary


def write_split(
    input_path: str | Path,
    output_root: str | Path,
    split_tag: str,
    train_slices: list[str],
    test_slices: list[str],
    models: list[str],
) -> Path:
    input_path = resolve_path(input_path)
    out_dir = resolve_path(output_root) / split_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(input_path)
    split_adata, summary = build_split(adata, train_slices, test_slices, split_tag)
    out_path = out_dir / "processed_with_split.h5ad"
    split_adata.write_h5ad(out_path)
    summary.to_csv(out_dir / "split_summary.csv", index=False)
    index = split_adata.obs[["slice_id", "label", "split", "split_tag"]].copy()
    index.insert(0, "obs_name", split_adata.obs_names.astype(str))
    index.to_csv(out_dir / "split_index.csv", index=False)

    model_root = resolve_path(output_root) / "model_inputs"
    for model in models:
        model_dir = model_root / model / split_tag
        model_dir.mkdir(parents=True, exist_ok=True)
        link = model_dir / "processed_with_split.h5ad"
        if link.exists() or link.is_symlink():
            link.unlink()
        rel = os.path.relpath(out_path, start=model_dir)
        link.symlink_to(rel)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DLPFC cross-slice generalization split.")
    parser.add_argument("--input", default="data/02_interim/common/DLPFC/processed_combined.h5ad")
    parser.add_argument(
        "--preset",
        choices=[*PRESET_SPLITS.keys(), "all"],
        default=None,
        help="Named split preset. Use all to write all train/test presets plus all_slices_pool.",
    )
    parser.add_argument("--split-tag", default="fixed_8train_4test")
    parser.add_argument("--train-slices", nargs="*", default=DEFAULT_TRAIN_SLICES)
    parser.add_argument("--test-slices", nargs="*", default=DEFAULT_TEST_SLICES)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--output-root", default="data/02_interim/cross_slice_generalization/DLPFC")
    return parser


def main():
    args = build_parser().parse_args()
    if args.preset is not None:
        selected = PRESET_SPLITS if args.preset == "all" else {args.preset: PRESET_SPLITS[args.preset]}
        for split_tag, (train_slices, test_slices) in selected.items():
            out = write_split(
                input_path=args.input,
                output_root=args.output_root,
                split_tag=split_tag,
                train_slices=[str(x) for x in train_slices],
                test_slices=[str(x) for x in test_slices],
                models=list(args.models),
            )
            print(f"wrote {out}")
        return

    out = write_split(
        input_path=args.input,
        output_root=args.output_root,
        split_tag=args.split_tag,
        train_slices=[str(x) for x in args.train_slices],
        test_slices=[str(x) for x in args.test_slices],
        models=list(args.models),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
