# spAug — Anonymous Code Release

This repository is the **feature-level augmentation** part of the release. It augments the
measured spot / sample representation with frozen pathology-foundation-model (PFM)
embeddings (GPFM, UNI, UNI2-h, GigaPath) and compares them against the expression-only PCA
baseline on four downstream task families:

1. spatial clustering (DLPFC),
2. within-slice low-label prediction (DLPFC),
3. cross-slice prediction (DLPFC),
4. sample-level disease prediction (Kidney, Bowel, Brain).

The **synthetic-observation** route (generated observations from SRTsim, Splatter, SPARsim,
scGAN and scDiffusion) is not included in this release and is provided separately. A few
downstream utilities that both routes share — the classifiers, the PCA/PCA+embedding feature
transforms, the DLPFC preprocessing and the PCA-spatial Leiden backend — live with that
route, so the notebooks here add them to `sys.path` at the top (see
[`feature/README.md`](feature/README.md)).

> This is an anonymized release prepared for double-blind review. Author names,
> affiliations, and absolute local paths have been removed.

## Layout

```text
.
├── requirements.txt            # exported run environment (Python 3.10, torch 2.6 cu124)
└── feature/                    # feature-level (PFM) route
    ├── README.md               # detailed usage: embeddings, SpaGCN, notebooks
    ├── notebooks/              # task alignment and fusion experiments
    ├── scripts/                # PFM embedding extraction, SpaGCN graph integration
    └── spagcn/                 # adapted SpaGCN (Hu et al. 2021, MIT)
```

## Quick start

```bash
pip install -r requirements.txt
```

Python 3.10 is required; `requirements.txt` is the export of the environment in which the
reported runs were produced. Neither the PFM checkpoints nor the raw data are redistributed —
see [`feature/README.md`](feature/README.md) for the checkpoints, the expected data layout
and the step-by-step instructions.

## Contents

| Path | Role |
|---|---|
| `feature/scripts/extract_pfm_embeddings.py` | crop per-spot H&E patches, cache frozen PFM embeddings |
| `feature/scripts/run_spagcn_integration.py` | PFM-scaled SpaGCN graph integration |
| `feature/spagcn/` | adapted SpaGCN (Hu et al. 2021) with a reimplementation of the spEMO `image_feature` strategy (Liu et al. 2025) |
| `feature/notebooks/feature_spatial_cluster.ipynb` | PFM concatenation vs. PCA baseline, fusion grid |
| `feature/notebooks/feature_low_label_classification.ipynb` | within-slice low-label prediction |
| `feature/notebooks/feature_cross_slice_classification.ipynb` | cross-slice prediction |
| `feature/notebooks/feature_disease_prediction.ipynb` | sample-level disease prediction |

Outputs are written under `feature/outputs/`.

## Data

Raw data are not redistributed. The four task families use:

| Task | Dataset | Source |
|---|---|---|
| spatial clustering, low-label, cross-slice | DLPFC (12 sections; donors Br5292, Br5595, Br8100) | Maynard et al., *Nature Neuroscience* 2021 |
| sample-level disease prediction | Kidney | public spatial-transcriptomics cohort |
| sample-level disease prediction | Bowel | public spatial-transcriptomics cohort |
| sample-level disease prediction | Brain (EPM/Cancer + spatialLIBD/Healthy) | EPM study + spatialLIBD |

## Documentation

- [`feature/README.md`](feature/README.md) — environment, data layout, PFM embedding
  extraction, SpaGCN graph integration, the notebooks, data splits and parameters, outputs.
- [`feature/spagcn/README.md`](feature/spagcn/README.md) — the exact modification applied to
  SpaGCN and the citations (SpaGCN, spEMO).
