# Feature augmentation

The feature module augments spatial transcriptomics representations with frozen
pathology foundation model (PFM) embeddings. It provides patch extraction,
embedding caching, feature fusion notebooks, and SpaGCN graph integration.
It includes its own runtime utilities, task configurations, and preprocessing.

## Environment

From the `feature/` directory, install its environment:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` includes scientific dependencies, notebook
support, and the Louvain dependency used by SpaGCN. A recorded Linux/CUDA 12.4
environment is available in [environments/cuda124.txt](environments/cuda124.txt).
Model registries may require additional model-specific packages.

## Data and path conventions

Provide the data and embedding caches locally. Default paths are relative to
the `feature/` directory:

| Location | Contents |
| --- | --- |
| `data/DLPFC/st/` | Combined `DLPFC_12_slices.h5ad` and per-section `<slice>_adata.h5ad` |
| `data/DLPFC/wsis/` | Full-resolution H&E images named `<slice>_full_image.tif` |
| `data/DLPFC/pilot/<slice>/` | Visium `tissue_positions_list.txt` |
| `data/DLPFC/pkl/` | Cached patches, embeddings, and SpaGCN labels |
| `data/HEST/` | Sample AnnData, `raw/patches/*.h5`, and `pkl/` embeddings |
| `data/02_interim/` | Prepared expression data and task splits |
| `outputs/` | Task results and figures |

The notebooks locate this module and import utilities from
`src/spaug_feature/`. Configuration and prepared-data paths resolve inside
this module. Each module owns its data, configurations, and outputs.

## DLPFC expression preparation

From `feature/`, prepare the expression inputs independently:

```bash
python scripts/prepare_dlpfc.py --data-dir data/DLPFC
```

Supply `data/DLPFC/st/<slice>_adata.h5ad` with expression in `X`, manual labels
in `obs["spatialLIBD"]`, and coordinates in `obsm["spatial"]`. The preprocessing
configuration is `configs/common/DLPFC/normalize.yaml`. Prepared slices and
`processed_combined.h5ad` are written to `data/02_interim/common/DLPFC/`, with
`slice_id`, `label`, and slice-prefixed observation names for cache alignment.

Sample-disease notebooks can load prepared sample AnnData and `splits.csv`
under `data/02_interim/sample_disease_prediction/<dataset>/`, or build their
inputs directly from local HEST data. Prepared sample data use `sample_id`,
`disease_label`, and spatial coordinates. Split tables contain `sample_id`,
`fold`, and `split` columns. The HEST input layout contains
`meta_df_t_update.csv`, `raw/st/<sample>.h5ad`, and `raw/patches/<sample>.h5`.

## Model registry and weights

Embedding extraction uses a locally supplied Python module named `models`
with these callables:

```python
get_model(name, device, n_gpu)        # returns a torch.nn.Module
get_custom_transformer(name)         # returns a PIL-image-to-tensor transform
```

The module must be importable in the extraction environment, for example by
installing the selected registry or adding its parent directory to `PYTHONPATH`.
This interface is an external dependency of the extraction script. Acquire the
implementations and weights from the model providers under their access terms,
and configure the registry to use those local weights. Each model returns a
batch of embedding vectors with the dimensions below.

| Cache tag | Registry name | Dimensions | Image transform |
| --- | --- | --- | --- |
| `gpfm` | `GPFM` | 1024 | Resize to 224 with bicubic interpolation |
| `uni` | `uni` | 1024 | Resize to 224 with bilinear interpolation |
| `uni2_h` | `uni2_h` | 1536 | Resize to 224 and center crop to 224 |
| `gigapath` | `gigapath` | 1536 | Resize to 256 with bicubic interpolation and center crop to 224 |

DLPFC patches use radius 112, producing a 225 x 225 crop before the model
transform. Pixel rows and columns follow the
[Space Ranger position-table specification](https://www.10xgenomics.com/support/software/space-ranger/3.1/tutorials/outputs/spatial-outputs). Embedding caches follow the raw AnnData observation order. Notebooks
align cached rows to prepared observation identifiers, including reordered or
filtered inputs. HEST inputs support plain and sample-prefixed barcodes.
Precomputed caches in the following layout can be used directly by notebooks.

## Embedding extraction

Run from `feature/`:

```bash
python scripts/extract_pfm_embeddings.py --dataset dlpfc \
    --data-dir data/DLPFC --stage patches
python scripts/extract_pfm_embeddings.py --dataset dlpfc \
    --data-dir data/DLPFC --stage embed --models gpfm uni gigapath uni2_h

python scripts/extract_pfm_embeddings.py --dataset hest \
    --data-dir data/HEST --stage embed --models gpfm uni gigapath uni2_h
```

DLPFC embeddings are stored as PyTorch tensors in
`pkl/embeddings/visium_<slice>_allspot_<tag>_112.pkl`. HEST embeddings are stored
in `pkl/<sample>_<tag>_emb.pkl`. These paths are relative to the corresponding
data directory. The extraction workflow follows the patch-based feature
recipe attributed in the [SpaGCN guide](spagcn/README.md).

## SpaGCN graph integration

The bundled [SpaGCN implementation](spagcn/README.md) applies a per-spot PFM
scale to the histology-derived adjacency axis. Expression features are used
as graph node inputs.

```bash
python scripts/run_spagcn_integration.py \
    --data-dir data/DLPFC --models gpfm uni gigapath uni2_h --radius 112
```

The resulting `data/DLPFC/pkl/spagcn_refined/<tag>_r112_update_adj_multi.pkl`
files contain slice-to-label mappings used by the spatial clustering notebook.

## Notebooks

Launch from `feature/`:

```bash
python -m jupyter lab notebooks
```

| Notebook | Task |
| --- | --- |
| [feature_spatial_cluster.ipynb](notebooks/feature_spatial_cluster.ipynb) | Expression and image feature fusion, plus precomputed SpaGCN labels |
| [feature_low_label_classification.ipynb](notebooks/feature_low_label_classification.ipynb) | Within-slice low-label prediction |
| [feature_cross_slice_classification.ipynb](notebooks/feature_cross_slice_classification.ipynb) | Cross-slice prediction |
| [feature_disease_prediction.ipynb](notebooks/feature_disease_prediction.ipynb) | Kidney, Bowel, and Brain sample-level disease prediction |

Use a fresh kernel for each notebook. Run cells in order and adjust the data
locations in the configuration cells as needed. Task configuration files under
`configs/` define classifier parameters and splits. Low-label
notebooks use label fractions 0.25, 0.50, and 0.75. Cross-slice notebooks use
the split definitions in `configs/cross_slice_generalization/DLPFC/downstream.yaml`.

## Validation and outputs

From `feature/`, validate the standalone module:

```bash
python scripts/check_setup.py --isolated
```

The check copies this module to a temporary directory, executes each notebook's
import and configuration cells from module and notebook working directories,
and runs small numerical workflow tests. It uses the installed dependencies
and temporary output directories.

Full notebook execution uses local data and embedding caches. Task tables and
figures are written under `outputs/`. Save notebooks with cleared outputs and execution counts.

Project-owned code uses the [MIT License](LICENSE); see the module's
[third-party notices](THIRD_PARTY_NOTICES.md) and [citation metadata](CITATION.cff).
