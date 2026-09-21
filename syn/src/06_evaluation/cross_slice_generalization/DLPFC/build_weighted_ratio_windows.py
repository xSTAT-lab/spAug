"""Build weighted ERM ratio windows from cross-slice real+synthetic results.

For each model x classifier, this script first finds the best
global_real_plus_synthetic ratio separately in each split, averages those best
ratios across splits, then creates one 10-ratio integer window around the
rounded average. The resulting window represents the rounded average across splits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from io_utils import resolve_path
else:
    from .io_utils import resolve_path


def ten_ratio_window(center: int, min_ratio: int = 1, max_ratio: int = 30) -> tuple[int, int, list[int]]:
    start = int(center) - 4
    end = int(center) + 5
    if start < min_ratio:
        end += min_ratio - start
        start = min_ratio
    if end > max_ratio:
        start -= end - max_ratio
        end = max_ratio
    start = max(min_ratio, start)
    end = min(max_ratio, end)
    if end - start + 1 < 10:
        if start == min_ratio:
            end = min(max_ratio, start + 9)
        elif end == max_ratio:
            start = max(min_ratio, end - 9)
    ratios = list(range(start, end + 1))
    if len(ratios) != 10:
        raise ValueError(f"Expected 10 ratios, got {len(ratios)} from center={center}: {ratios}")
    return start, end, ratios


def build_windows(
    aggregate_path: str | Path,
    output_path: str | Path,
    min_ratio: int = 1,
    max_ratio: int = 30,
) -> pd.DataFrame:
    aggregate_path = resolve_path(aggregate_path)
    output_path = resolve_path(output_path)
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing aggregate metrics: {aggregate_path}")
    aggregate = pd.read_csv(aggregate_path)
    required = {
        "model",
        "classifier",
        "split_tag",
        "paradigm",
        "ratio",
        "macro_f1_mean",
        "delta_macro_f1_mean",
        "accuracy_mean",
        "mcc_mean",
    }
    missing = required.difference(aggregate.columns)
    if missing:
        raise ValueError(f"Aggregate metrics missing columns: {sorted(missing)}")
    data = aggregate[
        (aggregate["model"].astype(str) != "baseline")
        & (aggregate["paradigm"].astype(str) == "global_real_plus_synthetic")
    ].copy()
    if data.empty:
        raise ValueError("No global_real_plus_synthetic rows found in aggregate metrics")
    data["ratio"] = pd.to_numeric(data["ratio"], errors="coerce")
    data = data.dropna(subset=["ratio"])
    best_by_split = (
        data.sort_values(
            ["model", "classifier", "split_tag", "delta_macro_f1_mean", "macro_f1_mean"],
            ascending=[True, True, True, False, False],
        )
        .groupby(["model", "classifier", "split_tag"], dropna=False)
        .head(1)
        .reset_index(drop=True)
    )
    rows = []
    for (model, classifier), group in best_by_split.groupby(["model", "classifier"], sort=True):
        best_ratios = group["ratio"].astype(float).tolist()
        mean_best_ratio = float(sum(best_ratios) / len(best_ratios))
        center = int(round(mean_best_ratio))
        start, end, window = ten_ratio_window(center, min_ratio=min_ratio, max_ratio=max_ratio)
        rows.append(
            {
                "model": model,
                "classifier": classifier,
                "split_best_ratios": ",".join(str(int(x)) for x in best_ratios),
                "split_tags": ",".join(group["split_tag"].astype(str).tolist()),
                "mean_best_ratio": mean_best_ratio,
                "rounded_center_ratio": center,
                "weighted_ratio_start": start,
                "weighted_ratio_end": end,
                "weighted_ratio_window": ",".join(str(x) for x in window),
                "n_splits": int(len(best_ratios)),
                "selection_metric": "delta_macro_f1_mean_then_macro_f1_mean",
                "source_paradigm": "global_real_plus_synthetic",
            }
        )
    out = pd.DataFrame(rows).sort_values(["model", "classifier"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cross-slice weighted ERM ratio windows.")
    parser.add_argument(
        "--aggregate-path",
        default="data/05_results/summary/cross_slice_generalization/DLPFC/metrics_aggregate.csv",
    )
    parser.add_argument(
        "--output-path",
        default="data/05_results/summary/cross_slice_generalization/DLPFC/weighted_erm_ratio_windows_from_global_real_plus_synthetic.csv",
    )
    parser.add_argument("--min-ratio", type=int, default=1)
    parser.add_argument("--max-ratio", type=int, default=30)
    return parser


def main():
    args = build_parser().parse_args()
    out = build_windows(args.aggregate_path, args.output_path, min_ratio=args.min_ratio, max_ratio=args.max_ratio)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
