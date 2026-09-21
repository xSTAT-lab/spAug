# SpaGCN graph integration

This directory contains adapted SpaGCN code used by the feature augmentation
module. The package retains the original [MIT License](LICENSE).

## Attribution

SpaGCN source: <https://github.com/jianhuupenn/SpaGCN>.

Hu, J., Li, X., Coleman, K. et al. SpaGCN: Integrating gene expression, spatial
location and histology to identify spatial domains and spatially variable
genes by graph convolutional network. *Nature Methods* 18, 1342-1351 (2021).

The `image_feature` strategy is attributed to spEMO:
<https://github.com/HelloWorldLTY/spEMO>.

Liu, T., Huang, T., Ding, T. et al. Leveraging multi-modal foundation models for
analysing spatial multi-omic and histopathology data. *Nature Biomedical
Engineering* (2025). DOI: 10.1038/s41551-025-01602-6.

## Graph mechanism

`calculate_adj_matrix` accepts an `image_feature` embedding cache. Each spot's
scale is computed from its float32 embedding:

```python
spot_scale = embedding.sum(axis=1, dtype=np.float32) / embedding.shape[1]
z = z * spot_scale
```

Here `z` is the histology-derived adjacency axis. The PFM embedding therefore
influences graph connectivity. The supplied integration script uses expression
features for the graph nodes. `SpaGCN.train` also exposes an optional
`image_emb` argument for experiments with concatenated node features.

## Workflow

[run_spagcn_integration.py](../scripts/run_spagcn_integration.py) constructs the
adjacency matrix, searches scale and resolution parameters, trains SpaGCN,
applies hexagonal refinement, and writes per-slice labels for the
[spatial clustering notebook](../notebooks/feature_spatial_cluster.ipynb).

Generate embeddings with
[extract_pfm_embeddings.py](../scripts/extract_pfm_embeddings.py), following
the [feature setup guide](../README.md). Runtime dependencies are included in
[feature/requirements.txt](../requirements.txt).
