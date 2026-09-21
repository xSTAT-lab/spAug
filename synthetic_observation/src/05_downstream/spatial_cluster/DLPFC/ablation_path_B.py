"""Run spatial clustering ablation path B.

Path B clusters the real + synthetic augmented graph jointly, then evaluates
only real observations after synthetic labels are discarded.
"""

from __future__ import annotations

import argparse

try:
    from .spagcn_cluster import run_spatial_clustering
except ImportError:
    from spagcn_cluster import run_spatial_clustering


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DLPFC clustering ablation path B.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("-c", "--config", default="configs/spatial_cluster/DLPFC/downstream.yaml")
    parser.add_argument("--slice-id", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    run_spatial_clustering(
        input_path=args.input,
        output_dir=args.output_dir,
        labels_path=args.labels,
        config_path=args.config,
        slice_id=args.slice_id,
        ablation="path_B",
    )


if __name__ == "__main__":
    main()
