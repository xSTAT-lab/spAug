# SpaGCN (adapted)

This directory holds the graph convolutional network behind the **PFM Graph
Integration** strategy of the feature-level route.

It is adapted from SpaGCN by Jian Hu, originally released at
<https://github.com/jianhuupenn/SpaGCN> under the MIT License (see `LICENSE`). It also
carries the `image_feature` strategy introduced by spEMO, reimplemented here (see below).
Please cite both works:

> Hu, J., Li, X., Coleman, K. et al. SpaGCN: Integrating gene expression, spatial
> location and histology to identify spatial domains and spatially variable genes by
> graph convolutional network. *Nature Methods* 18, 1342–1351 (2021).

> Liu, T., Huang, T., Ding, T. et al. Leveraging multi-modal foundation models for
> analysing spatial multi-omic and histopathology data. *Nature Biomedical Engineering*
> (2025). <https://doi.org/10.1038/s41551-025-01602-6>
> Code: <https://github.com/HelloWorldLTY/spEMO>

## What was changed

`calculate_adj.py` gains an `image_feature` argument on `calculate_adj_matrix`. The
argument follows the `image_feature` strategy introduced by spEMO, reimplemented here rather
than copied: the per-spot scale is computed as
`embedding.sum(axis=1, dtype=np.float32) / embedding.shape[1]`, which is numerically
identical to spEMO's `np.mean(data, axis=1)` on the same float32 embeddings. When
`image_feature` is supplied, the histology-derived axis `z` of the adjacency matrix is
scaled spot by spot with that per-spot scale:

```python
z = z * spot_scale
```

That is the only behavioural change: it lets a pathology foundation model take the place
of the image statistics that would otherwise determine the axis on their own. The
original implementation is kept, commented out, at the end of `calculate_adj.py`. The
reimplementation was checked on DLPFC section 151507 (4,226 spots) with the GPFM embedding:
the resulting adjacency matrix is bit-identical to the previous code (`np.array_equal` is
true, with 0 differing elements).

The PFM embeddings loaded through this argument are our own artefacts, produced by the
extraction script in this repository (`../scripts/extract_pfm_embeddings.py`) following the
spEMO recipe; no embeddings are taken from spEMO.

`SpaGCN.train` also accepts an optional `image_emb` argument that would append the
embeddings to the PCA node features. **The reported experiments do not use it**: the node
features remain the expression matrix, and the PFM representation acts through the graph
rather than through the node representation. Any extension that does pass `image_emb`
would change the model class and is not covered by the numbers in the manuscript.

## Usage

`../scripts/run_spagcn_integration.py` drives this package end to end for the DLPFC
slices: it builds the adjacency matrix with the PFM embeddings, searches the `l` and
resolution hyperparameters, trains the model, post-processes with hexagonal refinement,
and writes `pkl/spagcn_refined/<tag>_r<radius>_update_adj_multi.pkl` for
`../notebooks/feature_spatial_cluster.ipynb`.

This package needs `numba`, `scanpy`, `torch`, `scikit-learn` and `matplotlib`.
