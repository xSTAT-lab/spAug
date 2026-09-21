"""Prepare standalone DLPFC supervised low-label splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_split(
    adata: ad.AnnData,
    label_fraction: float,
    seed: int,
    slice_col: str = "slice_id",
    label_col: str = "label",
) -> tuple[ad.AnnData, pd.DataFrame]:
    if not 0 < label_fraction < 1:
        raise ValueError("label_fraction must be in (0, 1)")
    for col in (slice_col, label_col):
        if col not in adata.obs.columns:
            raise ValueError(f"AnnData is missing obs[{col!r}]")
    rng = np.random.default_rng(seed)
    split = pd.Series("test", index=adata.obs_names.astype(str), dtype=object)
    rows = []
    obs = adata.obs[[slice_col, label_col]].astype(str)
    for (slice_id, label), idx in obs.groupby([slice_col, label_col], sort=True).groups.items():
        names = np.asarray(list(idx), dtype=object)
        n_total = int(names.size)
        if n_total < 2:
            rows.append(
                {
                    "slice_id": str(slice_id),
                    "label": str(label),
                    "n_total": n_total,
                    "n_train": 0,
                    "n_test": n_total,
                    "status": "too_few_for_train_test",
                }
            )
            continue
        n_train = int(round(n_total * label_fraction))
        n_train = max(1, min(n_train, n_total - 1))
        chosen = rng.choice(names, size=n_train, replace=False)
        split.loc[chosen.astype(str)] = "train"
        rows.append(
            {
                "slice_id": str(slice_id),
                "label": str(label),
                "n_total": n_total,
                "n_train": n_train,
                "n_test": n_total - n_train,
                "status": "ok",
            }
        )
    out = adata.copy()
    out.obs["split"] = split.reindex(out.obs_names.astype(str)).to_numpy()
    out.obs["label_fraction"] = float(label_fraction)
    out.obs["low_label_seed"] = int(seed)
    out.uns["dataset"] = "DLPFC"
    out.uns["task_family"] = "supervised_low_label"
    out.uns["task_name"] = "dlpfc_slice_low_label_spot_clf"
    out.uns["split_mode"] = "supervised"
    out.uns["label_fraction"] = float(label_fraction)
    out.uns["low_label_seed"] = int(seed)
    out.uns["split_policy"] = "within_slice_label_fraction"
    return out, pd.DataFrame(rows)


def write_split(input_path: str | Path, output_dir: str | Path, label_fraction: float, seed: int) -> Path:
    input_path = resolve_path(input_path)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(input_path)
    out, summary = build_split(adata, label_fraction=label_fraction, seed=seed)
    out_path = output_dir / "processed_with_split.h5ad"
    out.write_h5ad(out_path)
    summary.to_csv(output_dir / "split_summary.csv", index=False)
    index = out.obs[["slice_id", "label", "split", "label_fraction", "low_label_seed"]].copy()
    index.insert(0, "obs_name", out.obs_names.astype(str))
    index.to_csv(output_dir / "split_index.csv", index=False)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DLPFC supervised low-label splits.")
    parser.add_argument("--input", default="data/02_interim/common/DLPFC/processed_combined.h5ad")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--label-fraction", type=float, default=0.50)
    parser.add_argument(
        "--output-root",
        default=None,
    )
    return parser


def main():
    args = build_parser().parse_args()
    fraction_root = args.output_root or (
        f"data/02_interim/supervised_low_label/DLPFC/fraction{int(round(args.label_fraction * 100))}"
    )
    for seed in args.seeds:
        out = write_split(
            input_path=args.input,
            output_dir=resolve_path(fraction_root) / f"seed{seed}",
            label_fraction=args.label_fraction,
            seed=seed,
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
