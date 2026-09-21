# Feature-level (PFM) route

Usage notes for the feature-level augmentation route. For the release overview see the
repository [`README.md`](../README.md).

## Environment

Python 3.10, installed from the exported environment:

```bash
pip install -r requirements.txt        # run from the repository root
```

`requirements.txt` is the export of the environment in which the reported runs were produced
(Python 3.10.20, PyTorch 2.6.0+cu124, CUDA 12.4). `torch`/`torchvision` are the CUDA 12.4
Linux builds; on a CPU-only or non-Linux machine install the matching wheels instead. Two
packages used by this route, `louvain` (SpaGCN graph integration) and `timm` (PFM model
registry), came from a separate environment and are therefore not listed in the export.

The recorded runs used three NVIDIA RTX 4090 (48 GiB). The cost measurements in the response
letter were taken on a 128-core Intel Xeon Gold 6530 host with 503 GiB RAM.

## Data and path conventions

Raw data are not redistributed. Paths below are relative to the repository root; the
notebooks express them relative to `feature/notebooks/` (i.e. one directory deeper).

| Variable | Default | Contents |
|---|---|---|
| `DATA_DIR` | `data/DLPFC` | per-section Visium AnnData (`st/`), full-resolution H&E slides (`wsis/`), spot position lists (`pilot/`), PFM caches (`pkl/`) |
| `HEST_DATA_DIR` | `data/HEST` | HEST sample AnnData, H&E patches (`raw/patches/*.h5`), PFM caches (`pkl/`) |
| `PROCESSED_DIR` | `synthetic/data/02_interim/common/DLPFC` | preprocessed expression and splits produced by the synthetic-observation route |

The notebooks also reuse downstream utilities of the synthetic-observation route
(classifiers, feature transforms, DLPFC preprocessing, the PCA-spatial Leiden backend) by
prepending `synthetic/src/01_data_prep` and `synthetic/src/05_downstream/...` to `sys.path`.
To run them, place that code as a sibling `synthetic/` directory next to `feature/`, or adjust
those few lines. Only expression, spatial coordinates, labels and splits are actually needed;
any preprocessing that follows the same AnnData contract can be substituted.

## PFM embeddings

The route consumes frozen PFM embeddings of the H&E image around every spot. The four
checkpoints are **not** redistributed: each is published by its own authors under its own
terms, and several are gated. To rebuild the cache, supply a `models` package exposing
`get_model(name, device, n_gpu)` and `get_custom_transformer(name)` for the models below,
then run the extraction script.

| tag | model | dim | patch -> transform |
|---|---|---|---|
| `gpfm` | GPFM (DINOv2 ViT-L/14 + MLP) | 1024 | radius 112 -> Resize(224, bicubic) |
| `uni` | UNI (ViT-L/16) | 1024 | radius 112 -> Resize(224, bilinear) |
| `uni2_h` | UNI2-h (ViT-h/14 + reg8 + SwiGLU) | 1536 | radius 112 -> Resize(224) + CenterCrop(224) |
| `gigapath` | GigaPath (ViT-giant/14 DINOv2) | 1536 | radius 112 -> Resize(256, bicubic) + CenterCrop(224) |

```bash
# DLPFC: crop one 225x225 patch per spot, then embed with all four PFMs
python feature/scripts/extract_pfm_embeddings.py --dataset dlpfc \
    --data-dir /path/to/DLPFC --stage patches
python feature/scripts/extract_pfm_embeddings.py --dataset dlpfc \
    --data-dir /path/to/DLPFC --stage embed --models gpfm uni gigapath uni2_h

# HEST (sample-level tasks): patches are supplied as HDF5 files
python feature/scripts/extract_pfm_embeddings.py --dataset hest \
    --data-dir /path/to/HEST --stage embed --models gpfm uni gigapath uni2_h
```

This writes `pkl/embeddings/visium_<slice>_allspot_<tag>_112.pkl` for DLPFC and
`pkl/<sample>_<tag>_emb.pkl` for HEST, which is where the notebooks look for them. The
embeddings used in the reported runs were produced by this script, following the extraction
recipe of spEMO (Liu et al. 2025); they are not redistributed here.

## SpaGCN graph integration

`feature/spagcn/` is an adapted copy of SpaGCN (Hu et al., *Nature Methods* 2021, MIT
License) carrying a reimplementation of the `image_feature` strategy introduced by spEMO
(Liu et al., *Nature Biomedical Engineering* 2025): the frozen PFM embedding of each spot
scales the
histology-derived axis of the adjacency matrix, so the PFM representation shapes the graph,
while the node features of the graph convolutional network remain the expression matrix. See
[`spagcn/README.md`](spagcn/README.md) for the exact modification and both citations. To
regenerate the refined labels that `feature_spatial_cluster.ipynb` loads for the SpaGCN-based
results:

```bash
python feature/scripts/run_spagcn_integration.py \
    --data-dir /path/to/DLPFC --models gpfm uni gigapath uni2_h --radius 112
```

This writes `pkl/spagcn_refined/<tag>_r112_update_adj_multi.pkl`. See
[`spagcn/README.md`](spagcn/README.md) for the exact modification and both citations.

## Notebooks

The notebooks in `feature/notebooks/` implement the alignment to each task's inputs and
splits, and the fusion/weighting settings:

- `feature_spatial_cluster.ipynb` — PFM concatenation vs. the PCA baseline, plus the
  normalization / modality-weighting / PCA fusion grid.
- `feature_low_label_classification.ipynb` — within-slice low-label prediction.
- `feature_cross_slice_classification.ipynb` — cross-slice prediction.
- `feature_disease_prediction.ipynb` — sample-level disease prediction.

Each notebook is self-contained apart from the shared utilities and data paths described
above; set `DATA_DIR` / `HEST_DATA_DIR` at the top if your layout differs.

## Data splits and parameters

- DLPFC cross-slice: three predefined eight-training / four-test-section splits.
  `151507–151510` belong to Br5292, `151669–151672` to Br5595, `151673–151676` to Br8100.
  The fixed split holds Br8100 out entirely; the other two splits place Br5292 and Br5595 on
  both sides. Exact compositions are defined in the synthetic-observation route's
  `cross_slice_generalization` configuration and in the response letter.
- DLPFC within-slice low-label: 25% / 50% / 75% of the labelled spots per section, seeds
  42/43/44.
- Downstream classifier hyperparameters (LR, MLP, XGBoost) and the evaluation metrics are read
  from the synthetic-observation route's per-task `downstream.yaml` files by the notebooks.

## Outputs

Pipeline outputs (per-configuration results, stability summaries, figures) are written under
`feature/outputs/`. Result-table schemas are documented in the Supplementary Information.
