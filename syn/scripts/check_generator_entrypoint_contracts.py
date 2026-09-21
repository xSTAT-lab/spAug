#!/usr/bin/env python
"""Static contract checks for generator entrypoints.

This gate reads source contracts only and keeps model training outside the check.
It verifies that every task-local generator entrypoint follows the implementation
contract used by formal experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


MODELS = ("SRTsim", "Splatter", "SPARsim", "scGAN", "scDiffusion")


@dataclass(frozen=True)
class TokenRule:
    path_suffix: str
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()


MODEL_RULES: dict[str, tuple[TokenRule, ...]] = {
    "SRTsim": (
        TokenRule("train.py", ("SRTsimPython", '"model_backend": "python_reimplementation"', "SRTsim reference-based Python implementation")),
        TokenRule("generate.py", ("SRTsimPython", 'syn_adata.uns["generator_backend"]', 'syn_adata.uns["coord_method"]')),
    ),
    "Splatter": (
        TokenRule("train.py", ("SpatialSplatterPython", '"model_backend": "python_reimplementation"', "Splat single-population Python implementation")),
        TokenRule("generate.py", ("SpatialSplatterPython", 'syn.uns["generator_backend"]', "splat_single_region_simadaptor")),
    ),
    "SPARsim": (
        TokenRule("train.py", ("SPARsimGMHPython", '"model_backend": "python_reimplementation"', "Gamma-Multivariate-Hypergeometric")),
        TokenRule("generate.py", ("SPARsimGMHPython", 'syn.uns["generator_backend"]', "sparsim_gmh_region_simadaptor")),
    ),
    "scGAN": (
        TokenRule(
            "model.py",
            (
                "class ConditionalGenerator",
                "class ConditionalCritic",
                "class ConditionalBatchNorm1d",
                "condition_projection",
            ),
        ),
        TokenRule(
            "train.py",
            (
                '"model_backend": "python_reimplementation"',
                "internal_cscgan_conditional_bn_projection",
                "published imsb-uke/scGAN design",
                '"conditional"',
            ),
        ),
        TokenRule(
            "generate.py",
            (
                "ConditionalGenerator",
                "reference_coord_resampling_with_jitter",
                'syn_adata.uns["generator_backend"]',
                'syn_adata.uns["conditional_generation"]',
            ),
        ),
    ),
    "scDiffusion": (
        TokenRule(
            "model.py",
            (
                "class DiffusionClassifier",
                "norm1",
                "norm2",
                "norm3",
                "hidden_dim = [1024, 1024, 1024]",
            ),
        ),
        TokenRule("diffusion.py", ("model_mean = model_mean + gradient",), ("model_mean = model_mean + out[\"variance\"] * gradient",)),
        TokenRule(
            "train.py",
            (
                'model_backend": "python_reimplementation"',
                "DiffusionClassifier",
                "guidance_t_max",
                "train_classifier_on_0_to_T_over_2",
                "classifier_guidance_t_max",
            ),
        ),
        TokenRule(
            "generate.py",
            (
                "DiffusionClassifier",
                "reference_coord_resampling_with_jitter",
                "guidance_t_max",
                "beta_scaled",
                'syn_adata.uns["classifier_guidance"]',
                'syn_adata.uns["generator_backend"]',
            ),
        ),
    ),
}


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise RuntimeError(f"Cannot find project root from {start}")


def discover_generator_dirs(root: Path) -> list[Path]:
    base = root / "src" / "02_generators"
    dirs: list[Path] = []
    for model in MODELS:
        dirs.extend(path for path in base.glob(f"*/*/{model}") if path.is_dir())
        dirs.extend(path for path in base.glob(f"*/*/*/{model}") if path.is_dir())
    return sorted(path for path in set(dirs) if has_task_local_entrypoint(path))


def has_task_local_entrypoint(path: Path) -> bool:
    """Return True for model folders with runnable entry points."""
    expected = {"train.py", "generate.py", "model.py", "diffusion.py"}
    return any((path / name).exists() for name in expected)


def check_file(path: Path, rule: TokenRule) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing file: {path}"]
    text = path.read_text(encoding="utf-8")
    for token in rule.required:
        if token not in text:
            errors.append(f"missing token {token!r} in {path}")
    for token in rule.forbidden:
        if token in text:
            errors.append(f"forbidden token {token!r} in {path}")
    return errors


def check_generator_dir(generator_dir: Path) -> dict:
    model = generator_dir.name
    rules = MODEL_RULES.get(model)
    if rules is None:
        return {
            "path": str(generator_dir),
            "model": model,
            "status": "skipped_unknown_model",
            "errors": [],
        }

    errors: list[str] = []
    for rule in rules:
        errors.extend(check_file(generator_dir / rule.path_suffix, rule))
    return {
        "path": str(generator_dir),
        "model": model,
        "status": "ok" if not errors else "failed",
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check generator entrypoint contracts.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--fail-on-missing-model", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = find_project_root(Path(__file__).resolve())
    requested = set(args.models)
    dirs = [path for path in discover_generator_dirs(root) if path.name in requested]
    rows = [check_generator_dir(path) for path in dirs]

    found = {row["model"] for row in rows}
    missing_requested = sorted(requested - found)
    if args.fail_on_missing_model and missing_requested:
        for model in missing_requested:
            rows.append({"path": "", "model": model, "status": "failed", "errors": ["model directory not found"]})

    failed = [row for row in rows if row["status"] == "failed"]
    summary = {
        "n_checked": len(rows),
        "n_failed": len(failed),
        "models_found": sorted(found),
        "missing_requested_models": missing_requested,
        "rows": rows,
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, ensure_ascii=False))
    for row in rows:
        marker = "OK" if row["status"] == "ok" else ("SKIP" if row["status"].startswith("skipped") else "FAIL")
        print(f"{marker}\t{row['model']}\t{row['path']}")
        for error in row["errors"]:
            print(f"  - {error}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
