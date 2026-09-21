#!/usr/bin/env python3
"""Run the adapted SpaGCN graph integration used for PFM feature-level clustering.

This reproduces the "PFM Graph Integration" strategy of the feature-level route. The
frozen PFM embedding of each spot is passed into the construction of the SpaGCN
adjacency matrix, where it scales the histology-derived axis spot by spot. Spot
expression features remain the node features of the graph convolutional network,
followed by deep embedded clustering.

The PFM representation influences graph connectivity, and the expression matrix
provides node features for this integration workflow.

For every DLPFC slice and every requested PFM, the script writes the refined labels the
clustering notebook loads::

    <data-dir>/pkl/spagcn_refined/<tag>_r<radius>_update_adj_multi.pkl

Each file is a pickle holding ``dict`` mapping slice id -> list of refined cluster
labels, aligned with ``adata.obs_names`` of ``<data-dir>/st/<slice>_adata.h5ad``.

Example
-------
    python run_spagcn_integration.py --data-dir data/DLPFC \\
        --models gpfm uni gigapath uni2_h --radius 112
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

from PIL import Image

# Allow the full-resolution H&E images used by this workflow.
Image.MAX_IMAGE_PIXELS = 7793202000

# The SpaGCN implementation shipped next to this script (feature/spagcn/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import spagcn as spg
except ImportError as exc:  # pragma: no cover - depends on the checkout layout
    raise SystemExit(
        "Could not import the bundled SpaGCN package.\n"
        "Expected it at feature/spagcn/ next to feature/scripts/."
    ) from exc

DEFAULT_SLICES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]


def load_wsi(path: Path) -> np.ndarray:
    """Read an H&E slide as an (H, W, 3) uint8 array.

    The reference pipeline loaded the slide through a lazy image container and took the
    first z-plane, which for a single-plane slide is the same array.
    """
    if not path.exists():
        raise SystemExit(f"missing H&E image: {path}")
    return np.array(Image.open(path))


def run_slice(
    sid: str,
    data_dir: Path,
    radius: int,
    tag: str,
    n_clusters: int,
    beta: float,
    alpha: float,
    seed: int,
) -> list:
    import anndata as ad
    import scanpy as sc

    adata_path = data_dir / "st" / f"{sid}_adata.h5ad"
    if not adata_path.exists():
        raise SystemExit(f"missing slice AnnData: {adata_path}")
    embedding_path = data_dir / "pkl" / "embeddings" / f"visium_{sid}_allspot_{tag}_{radius}.pkl"
    if not embedding_path.exists():
        raise SystemExit(
            f"missing PFM embedding: {embedding_path}\n"
            f"run extract_pfm_embeddings.py for tag '{tag}' first"
        )

    adata = sc.read_h5ad(adata_path)
    image = load_wsi(data_dir / "wsis" / f"{sid}_full_image.tif")

    x_array = adata.obs["array_row"].tolist()
    y_array = adata.obs["array_col"].tolist()
    x_pixel = adata.obsm["spatial"][:, 0].tolist()
    y_pixel = adata.obsm["spatial"][:, 1].tolist()

    # Adjacency from pixel coordinates plus a histology-derived axis, whose per-spot
    # scaling incorporates the frozen PFM embedding.
    adj = spg.calculate_adj_matrix(
        x=x_pixel,
        y=y_pixel,
        x_pixel=x_pixel,
        y_pixel=y_pixel,
        image=image,
        beta=beta,
        alpha=alpha,
        histology=True,
        image_feature=str(embedding_path),
    )

    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["MT_gene"] = [gene.startswith("MT-") for gene in adata.var_names]
    adata.obsm["MT"] = adata[:, adata.var["MT_gene"].values].X.toarray()
    adata = adata[:, ~adata.var["MT_gene"].values].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Search the adjacency scale l for a target density, then the Louvain resolution.
    l = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    res = spg.search_res(
        adata,
        adj,
        l,
        n_clusters,
        start=0.7,
        step=0.1,
        tol=5e-3,
        lr=0.05,
        max_epochs=20,
        r_seed=seed,
        t_seed=seed,
        n_seed=seed,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = spg.SpaGCN()
    model.set_l(l)
    # Node features are the expression matrix; the PFM embedding already acted on `adj`.
    model.train(adata, adj, res=res)
    y_pred, _ = model.predict()

    # Post-process with a hexagon neighbourhood over array coordinates only.
    adj_2d = spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    refined = spg.refine(
        sample_id=adata.obs.index.tolist(),
        pred=y_pred,
        dis=adj_2d,
        shape="hexagon",
    )
    return list(refined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="DLPFC directory holding st/, wsis/, pkl/embeddings/ and receiving pkl/spagcn_refined/",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpfm", "uni", "gigapath", "uni2_h"],
        help="PFM tags whose embeddings should drive the graph",
    )
    parser.add_argument("--radius", type=int, default=112, help="patch half-size of the embedding suffix")
    parser.add_argument("--slices", nargs="*", default=None, help="slice ids to process")
    parser.add_argument("--n-clusters", type=int, default=7, help="expected spatial domains per slice")
    parser.add_argument("--beta", type=float, default=55, help="neighbourhood range for the histology axis")
    parser.add_argument("--alpha", type=float, default=1, help="colour-scale weight for the histology axis")
    parser.add_argument("--seed", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir does not exist: {data_dir}")

    slices = args.slices or DEFAULT_SLICES
    out_dir = data_dir / "pkl" / "spagcn_refined"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tag in args.models:
        print(f"\n===== {tag} (radius={args.radius}) =====")
        label_dict = {}
        for sid in slices:
            print(f"  {sid} ...", end=" ", flush=True)
            label_dict[sid] = run_slice(
                sid,
                data_dir,
                args.radius,
                tag,
                args.n_clusters,
                args.beta,
                args.alpha,
                args.seed,
            )
            print("done")

        out_path = out_dir / f"{tag}_r{args.radius}_update_adj_multi.pkl"
        with open(out_path, "wb") as handle:
            pickle.dump(label_dict, handle)
        print(f"  saved {out_path}")

    print("\ndone.")


if __name__ == "__main__":
    main()
