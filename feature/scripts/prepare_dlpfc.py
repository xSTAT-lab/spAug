#!/usr/bin/env python3
"""Prepare DLPFC expression inputs for feature notebooks."""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

FEATURE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FEATURE_ROOT / "src"))
from spaug_feature.paths import load_config, resolve_path
from spaug_feature.preprocessing import DLPFCPreprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/DLPFC", help="Module-relative directory containing st/<slice>_adata.h5ad")
    parser.add_argument("--config", default="configs/common/DLPFC/normalize.yaml")
    parser.add_argument("--output-dir", default=None, help="Module-relative destination for prepared AnnData")
    parser.add_argument("--slices", nargs="+", help="DLPFC section IDs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prep = DLPFCPreprocessor(load_config(args.config))
    prep.raw_dir = resolve_path(args.data_dir)
    if args.output_dir:
        prep.output_dir = resolve_path(args.output_dir)
    if args.slices:
        prep.SLICE_IDS = args.slices
    paths = prep.run()
    if not paths:
        raise SystemExit("Provide DLPFC AnnData files under the selected data directory's st/ folder.")
    print(f"Prepared {len(paths)} DLPFC files")


if __name__ == "__main__":
    main()
