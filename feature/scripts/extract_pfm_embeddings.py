#!/usr/bin/env python3
"""Extract frozen pathology-foundation-model (PFM) embeddings for the feature-level route.

The feature-level experiments concatenate or fuse PFM image embeddings with expression
features. This script reproduces the embedding cache that the notebooks load:

    DLPFC   <data-dir>/pkl/embeddings/visium_<slice>_allspot_<tag>_<radius>.pkl
    HEST    <data-dir>/pkl/<sample>_<tag>_emb.pkl

It runs in two stages.

  Stage 1 (``--stage patches``, DLPFC only)
      Crop a ``2 * radius + 1`` patch centred on every spot from the full-resolution
      H&E image and cache the crops as pickles. Spot coordinates are read from the
      Visium ``tissue_positions_list.txt``. Patches that extend past the image border
      are zero-padded; this matches the crop convention behind the reported results.

  Stage 2 (``--stage embed``)
      Run a frozen PFM over every patch and cache the per-spot embeddings.

The PFM implementations and weights are deliberately *not* redistributed here, because
each checkpoint carries its own licence (several are gated). This script expects a
``models`` package on ``PYTHONPATH`` exposing::

    get_model(name, device, n_gpu) -> torch.nn.Module
    get_custom_transformer(name)   -> torchvision-style transform

with ``name`` in ``{"GPFM", "uni", "gigapath", "uni2_h"}``. See ``README.md`` for the
checkpoints used and how to obtain them.

Examples
--------
DLPFC: crop patches once, then embed with all four PFMs::

    python extract_pfm_embeddings.py --dataset dlpfc --data-dir /path/to/DLPFC --stage patches
    python extract_pfm_embeddings.py --dataset dlpfc --data-dir /path/to/DLPFC \\
        --stage embed --models gpfm uni gigapath uni2_h

HEST: patches are supplied as HDF5 files, so only the embedding stage applies::

    python extract_pfm_embeddings.py --dataset hest --data-dir /path/to/HEST \\
        --stage embed --models gpfm
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from PIL import Image

# Full-resolution H&E slides exceed Pillow's default decompression-bomb threshold.
Image.MAX_IMAGE_PIXELS = 7793202000

# tag -> (model name passed to get_model, embedding dimension)
MODEL_CONFIGS = {
    "gpfm": ("GPFM", 1024),
    "uni": ("uni", 1024),
    "gigapath": ("gigapath", 1536),
    "uni2_h": ("uni2_h", 1536),
}


def load_model_api():
    """Import the external PFM registry, failing with actionable guidance."""
    try:
        from models import get_model, get_custom_transformer
    except ImportError as exc:  # pragma: no cover - depends on the user's environment
        raise SystemExit(
            "Could not import the `models` package required for PFM inference.\n"
            "This release does not redistribute the PFM implementations or weights.\n"
            "See README.md for the four checkpoints and where to obtain them, then put\n"
            "the matching `models` package on PYTHONPATH."
        ) from exc
    return get_model, get_custom_transformer


def pick_device() -> tuple[torch.device, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    return device, n_gpu


def build_model(name, device, n_gpu, get_model):
    model = get_model(name, device, n_gpu)
    # Some registries already wrap the model in DataParallel; only wrap the bare ones.
    if n_gpu > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model.eval()
    return model


def embed_patch_batches(model, transform, patches, dim, batch_size, device):
    """Run a frozen PFM over a list of (H, W, 1, 3) uint8 patches."""
    feats = torch.zeros((len(patches), dim))
    for start in range(0, len(patches), batch_size):
        batch = patches[start : start + batch_size]
        tensors = []
        for patch in batch:
            rgb = patch[:, :, 0, :]
            image = Image.fromarray(np.uint8(rgb)).convert("RGB")
            tensors.append(transform(image))
        with torch.inference_mode():
            out = model(torch.stack(tensors).to(device))
        feats[start : start + len(batch)] = out.detach().cpu()
    return feats


def crop_dlpfc_patches(data_dir: Path, radius: int, slices=None) -> None:
    """Stage 1 for DLPFC: cache one patch per spot as a pickle."""
    import anndata as ad

    combined = data_dir / "st" / "DLPFC_12_slices.h5ad"
    if not combined.exists():
        raise SystemExit(f"missing DLPFC AnnData: {combined}")

    out_dir = data_dir / "pkl" / "patches"
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(combined)
    if slices is None:
        slices = [str(s) for s in adata.obs["sample_id"].unique()]

    patch_size = 2 * radius + 1
    for sid in slices:
        out_path = out_dir / f"visium_{sid}_{radius}.pkl"
        if out_path.exists():
            print(f"{sid}: skip (already exists)")
            continue

        spot = adata[adata.obs["sample_id"].astype(str) == sid]
        positions = pd_read_positions(data_dir / "pilot" / sid / "tissue_positions_list.txt")
        positions = positions.loc[spot.obs_names]
        # columns 4 and 5 hold the pixel (col, row) of each spot
        coords = np.stack((positions[4].values, positions[5].values), axis=1)

        wsi = data_dir / "wsis" / f"{sid}_full_image.tif"
        if not wsi.exists():
            raise SystemExit(f"missing H&E image: {wsi}")
        full_img = np.array(Image.open(wsi))
        height, width = full_img.shape[:2]

        patches = []
        for col, row in coords:
            cx, cy = int(col), int(row)
            x1, x2 = cx - radius, cx + radius + 1
            y1, y2 = cy - radius, cy + radius + 1
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                sx1, sx2 = max(0, x1), min(width, x2)
                sy1, sy2 = max(0, y1), min(height, y2)
                patch[sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1] = full_img[sy1:sy2, sx1:sx2]
            else:
                patch = full_img[y1:y2, x1:x2]
            # keep the (H, W, 1, C) layout used by the cached pickles
            patches.append(patch[:, :, np.newaxis, :])

        with open(out_path, "wb") as handle:
            pickle.dump(patches, handle)
        print(f"{sid}: {len(patches)} patches -> {out_path}")


def pd_read_positions(path: Path):
    import pandas as pd

    if not path.exists():
        raise SystemExit(f"missing spot position file: {path}")
    positions = pd.read_csv(path, header=None)
    positions.index = positions[0]
    return positions


def embed_dlpfc(data_dir: Path, radius: int, tags, batch_size: int, slices=None) -> None:
    """Stage 2 for DLPFC: embed the cached patches for each requested PFM."""
    import anndata as ad

    device, n_gpu = pick_device()
    get_model, get_custom_transformer = load_model_api()
    print(f"device={device} gpus={n_gpu} batch_size={batch_size}")

    combined = data_dir / "st" / "DLPFC_12_slices.h5ad"
    adata = ad.read_h5ad(combined, backed="r")
    if slices is None:
        slices = [str(s) for s in adata.obs["sample_id"].unique()]

    out_dir = data_dir / "pkl" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tag in tags:
        name, declared = MODEL_CONFIGS[tag]
        print(f"\n===== {name} (tag={tag}) =====")
        model = build_model(name, device, n_gpu, get_model)
        transform = get_custom_transformer(name)
        dim = declared

        for sid in slices:
            out_path = out_dir / f"visium_{sid}_allspot_{tag}_{radius}.pkl"
            if out_path.exists():
                print(f"  {sid}: skip (already exists)")
                continue
            patch_path = data_dir / "pkl" / "patches" / f"visium_{sid}_{radius}.pkl"
            if not patch_path.exists():
                raise SystemExit(
                    f"missing cached patches: {patch_path}\n"
                    f"run --stage patches first"
                )
            with open(patch_path, "rb") as handle:
                patches = pickle.load(handle)
            feats = embed_patch_batches(model, transform, patches, dim, batch_size, device)
            torch.save(feats, out_path)
            print(f"  {sid}: {feats.shape[0]} spots -> {out_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def embed_hest(data_dir: Path, tags, batch_size: int) -> None:
    """Stage 2 for HEST: patches arrive as HDF5 files under raw/patches."""
    import h5py

    device, n_gpu = pick_device()
    get_model, get_custom_transformer = load_model_api()
    print(f"device={device} gpus={n_gpu} batch_size={batch_size}")

    patches_root = data_dir / "raw" / "patches"
    if not patches_root.is_dir():
        raise SystemExit(f"missing HEST patch directory: {patches_root}")
    out_dir = data_dir / "pkl"
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(patches_root.glob("*.h5"))
    if not h5_files:
        raise SystemExit(f"no .h5 patch files under {patches_root}")

    for tag in tags:
        name, declared = MODEL_CONFIGS[tag]
        print(f"\n===== {name} (tag={tag}) =====")
        model = build_model(name, device, n_gpu, get_model)
        transform = get_custom_transformer(name)
        dim = declared

        for h5_path in h5_files:
            sid = h5_path.stem
            out_path = out_dir / f"{sid}_{tag}_emb.pkl"
            if out_path.exists():
                print(f"  {sid}: skip (already exists)")
                continue
            with h5py.File(h5_path, "r") as handle:
                raw = handle["img"][:]
            # HEST patches are stored directly as (H, W, C); normalise to (H, W, 1, C)
            patches = [p[:, :, np.newaxis, :] if p.ndim == 3 else p for p in raw]
            feats = embed_patch_batches(model, transform, patches, dim, batch_size, device)
            torch.save(feats, out_path)
            print(f"  {sid}: {feats.shape[0]} patches -> {out_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["dlpfc", "hest"], required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="DLPFC: directory holding st/, wsis/, pilot/, pkl/. "
        "HEST: directory holding raw/patches/ and receiving pkl/.",
    )
    parser.add_argument(
        "--stage",
        choices=["patches", "embed", "both"],
        default="both",
        help="DLPFC supports patches/embed/both; HEST only runs embed.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_CONFIGS),
        default=["gpfm", "uni", "gigapath", "uni2_h"],
    )
    parser.add_argument("--radius", type=int, default=112, help="patch half-size in pixels")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 selects 128 * number of visible GPUs",
    )
    parser.add_argument("--slices", nargs="*", default=None, help="DLPFC slice ids to restrict to")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir does not exist: {data_dir}")

    device, n_gpu = pick_device()
    batch_size = args.batch_size or 128 * max(n_gpu, 1)

    if args.dataset == "dlpfc":
        if args.stage in ("patches", "both"):
            crop_dlpfc_patches(data_dir, args.radius, args.slices)
        if args.stage in ("embed", "both"):
            embed_dlpfc(data_dir, args.radius, args.models, batch_size, args.slices)
    else:
        if args.stage == "patches":
            raise SystemExit("HEST patches are supplied as HDF5 files; use --stage embed")
        embed_hest(data_dir, args.models, batch_size)

    print("\ndone.")


if __name__ == "__main__":
    main()
