"""Visualize Bowel sample-level disease prediction evaluation."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spaug_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_yaml(path: str | Path) -> dict:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_bar_delta(delta: pd.DataFrame, out: Path) -> None:
    df = delta[delta["model"].astype(str) != "baseline"].copy()
    if df.empty:
        return
    summary = df.groupby(["model", "paradigm"], dropna=False)["delta_macro_f1"].mean().sort_values()
    plt.figure(figsize=(10, 5))
    summary.plot(kind="barh")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Mean delta macro F1 vs real baseline")
    plt.tight_layout()
    plt.savefig(out / "delta_macro_f1_by_model_paradigm.png", dpi=180)
    plt.close()


def save_ratio_curves(delta: pd.DataFrame, out: Path) -> None:
    df = delta[delta["ratio"].notna() & (delta["model"].astype(str) != "baseline")].copy()
    if df.empty:
        return
    for paradigm, g in df.groupby("paradigm"):
        pivot = g.groupby(["ratio", "model"], dropna=False)["delta_macro_f1"].mean().unstack()
        plt.figure(figsize=(9, 5))
        pivot.plot(ax=plt.gca(), marker="o")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("Mean delta macro F1")
        plt.title(str(paradigm))
        plt.tight_layout()
        plt.savefig(out / f"ratio_curve_{paradigm}.png", dpi=180)
        plt.close()


def save_alpha_heatmap(delta: pd.DataFrame, out: Path) -> None:
    df = delta[(delta["paradigm"] == "weighted_erm_package") & delta["alpha"].notna()].copy()
    if df.empty:
        return
    for model, g in df.groupby("model"):
        pivot = g.groupby(["alpha", "ratio"], dropna=False)["delta_macro_f1"].mean().unstack()
        plt.figure(figsize=(9, 5))
        plt.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
        plt.colorbar(label="Mean delta macro F1")
        plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns], rotation=45)
        plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
        plt.xlabel("ratio")
        plt.ylabel("alpha")
        plt.title(f"weighted ERM: {model}")
        plt.tight_layout()
        plt.savefig(out / f"weighted_alpha_heatmap_{model}.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sample_disease_prediction/Bowel/evaluation.yaml")
    parser.add_argument("--evaluation-root", default=None)
    parser.add_argument("--figures-root", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    eval_root = resolve_path(args.evaluation_root or cfg["evaluation_root"])
    fig_root = resolve_path(args.figures_root or cfg["figures_root"])
    fig_root.mkdir(parents=True, exist_ok=True)
    delta = pd.read_csv(eval_root / "metrics_with_baseline_delta.csv")
    save_bar_delta(delta, fig_root)
    save_ratio_curves(delta, fig_root)
    save_alpha_heatmap(delta, fig_root)
    print(f"wrote figures to {fig_root}")


if __name__ == "__main__":
    main()
