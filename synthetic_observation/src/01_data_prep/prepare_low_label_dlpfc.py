"""Prepare DLPFC low-label spot-classification splits.

The low-label task uses each slice independently: for every slice_id + label
group, a configured fraction of real spots is marked as train and the remaining
spots are marked as test. No val split is produced.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = ["SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion"]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def source_for_model(model: str, source_template: str) -> Path:
    return resolve_path(source_template.format(model=model))


def make_low_label_split(
    adata: ad.AnnData,
    label_fraction: float,
    seed: int,
    slice_col: str = "slice_id",
    label_col: str = "label",
) -> tuple[ad.AnnData, pd.DataFrame]:
    if not 0 < label_fraction < 1:
        raise ValueError("label_fraction must be in (0, 1)")
    if slice_col not in adata.obs.columns:
        raise ValueError(f"AnnData is missing obs[{slice_col!r}]")
    if label_col not in adata.obs.columns:
        raise ValueError(f"AnnData is missing obs[{label_col!r}]")

    rng = np.random.default_rng(seed)
    obs = adata.obs[[slice_col, label_col]].astype(str).copy()
    split = pd.Series("test", index=adata.obs_names.astype(str), dtype="object")
    rows = []

    for (slice_id, label), names in obs.groupby([slice_col, label_col], sort=True).groups.items():
        names = np.asarray(list(names), dtype=object)
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
        n_train = max(1, n_train)
        n_train = min(n_train, n_total - 1)
        chosen = rng.choice(names, size=n_train, replace=False)
        split.loc[chosen.astype(str)] = "train"
        rows.append(
            {
                "slice_id": str(slice_id),
                "label": str(label),
                "n_total": n_total,
                "n_train": int(n_train),
                "n_test": int(n_total - n_train),
                "status": "ok",
            }
        )

    out = adata.copy()
    out.obs["split"] = split.reindex(out.obs_names.astype(str)).to_numpy()
    out.obs["low_label_fraction"] = float(label_fraction)
    out.obs["low_label_seed"] = int(seed)
    out.uns["dataset"] = "DLPFC"
    out.uns["split_mode"] = "supervised"
    out.uns["task_mode"] = "low_label"
    out.uns["label_fraction"] = float(label_fraction)
    out.uns["low_label_seed"] = int(seed)
    out.uns["low_label_split_policy"] = "slice_label_fraction"
    return out, pd.DataFrame(rows)


def write_low_label_outputs(
    model: str,
    seed: int,
    label_fraction: float,
    source_template: str,
    output_template: str,
) -> Path:
    source = source_for_model(model, source_template)
    if not source.exists():
        raise FileNotFoundError(f"Missing source AnnData for {model}: {source}")
    adata = ad.read_h5ad(source)
    out, summary = make_low_label_split(adata, label_fraction=label_fraction, seed=seed)
    output_dir = resolve_path(
        output_template.format(
            model=model,
            seed=seed,
            fraction_tag=f"fraction{int(round(label_fraction * 100))}",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "processed_with_split.h5ad"
    out.write_h5ad(out_path)
    summary.to_csv(output_dir / "low_label_split_summary.csv", index=False)
    index = out.obs[["slice_id", "label", "split", "low_label_fraction", "low_label_seed"]].copy()
    index.insert(0, "obs_name", out.obs_names.astype(str))
    index.to_csv(output_dir / "low_label_split_index.csv", index=False)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DLPFC low-label splits.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--label-fraction", type=float, default=0.50)
    parser.add_argument(
        "--source-template",
        default="data/02_interim/common/DLPFC/processed_combined.h5ad",
    )
    parser.add_argument(
        "--output-template",
        default="data/02_interim/supervised_low_label/DLPFC/model_inputs/{model}/{fraction_tag}/seed{seed}",
    )
    return parser


def main():
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "spaug_numba_cache"))
    args = build_parser().parse_args()
    for model in args.models:
        for seed in args.seeds:
            out = write_low_label_outputs(
                model=model,
                seed=seed,
                label_fraction=args.label_fraction,
                source_template=args.source_template,
                output_template=args.output_template,
            )
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
