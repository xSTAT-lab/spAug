# spAug

> Cheng, M., Wang, R., & Wang, L. (2026). spAug: Understanding Complementary Augmentation Strategies for Spatial Transcriptomics. Statistical Learning and Data Science Section D. https://openreview.net/forum?id=vSsi8P1jEe


spAug provides two complementary routes for spatial transcriptomics data
augmentation: synthetic observations and pathology image features. Both routes
support spatial clustering, within-slice low-label prediction, cross-slice
prediction, and sample-level disease prediction.

## Modules

| Module | Purpose | Guide |
| --- | --- | --- |
| `syn/` | Generate synthetic spots with SRTsim, Splatter, SPARsim, scGAN, and scDiffusion; assign coordinates, construct training paradigms, and evaluate downstream tasks | [Synthetic-observation guide](syn/README.md) |
| `feature/` | Extract frozen pathology foundation model embeddings and evaluate feature fusion or SpaGCN graph integration | [Feature guide](feature/README.md) |

Each module contains its own code, configurations, dependencies, data paths,
and checks. Each directory can be copied and used independently.

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The root requirements combine both modules. For synthetic-observation workflows
alone, use `syn/requirements.txt`. Feature dependencies, including Jupyter and
Louvain clustering, are declared in `feature/requirements.txt`. The optional
Vine-Copula dependency is in `syn/requirements-extra.txt`.

For environment provenance, the feature module also provides a
[recorded CUDA 12.4 environment](feature/environments/cuda124.txt). The root
requirements provide the portable installation path. Select PyTorch and GPU
packages appropriate for your hardware when reproducing a specific GPU run.

## Getting started

Run synthetic-observation commands from `syn/`, following its module guide:

```bash
cd syn
python src/check_environment.py --help
```

From the repository root, launch the feature notebooks:

```bash
python -m jupyter lab feature/notebooks
```

Each notebook locates the feature module and imports its local
`src/spaug_feature/` package. Its configuration and data paths are module-local.

Feature extraction requires locally supplied PFM weights and a Python model
registry providing `get_model` and `get_custom_transformer`. The
[feature guide](feature/README.md#model-registry-and-weights) specifies this
interface and the embedding cache layout. Precomputed embedding caches can be
used directly by the feature notebooks.

## Data and outputs

Provide data locally in accordance with the dataset access terms. The feature
route uses `feature/data/DLPFC/` and `feature/data/HEST/`; synthetic
workflows use `syn/data/`. See the [data contract](syn/docs/data_contract.md)
and [synthetic data layout](syn/data/README.md). The feature notebooks read
prepared data from `feature/data/02_interim/`. The feature guide includes an
independent DLPFC preparation command.

Generated feature results are written to `feature/outputs/`; synthetic results
follow the task-specific layout under `syn/data/`. Git ignore rules cover local
data, model weights, environments, credentials, and generated artifacts.

## Validation

From the repository root:

```bash
python scripts/check_public_repo.py
python syn/scripts/check_generator_entrypoint_contracts.py
python -m unittest discover -s tests -v
```

With runtime dependencies installed:

```bash
python scripts/check_public_repo.py --with-mechanisms
python scripts/check_feature_setup.py
python syn/scripts/check_deep_generator_task_cli_smoke.py
```

The static checks cover both modules, notebook syntax and metadata, local
links, and portable paths. Runtime checks use synthetic fixtures, standalone
feature setup cells, and numerical workflows. Full scientific experiments
require the selected datasets,
embedding caches or model weights, configurations, and recorded runtime versions.

## Repository layout

```text
syn/                    synthetic-observation code, configurations, and module guides
feature/                feature notebooks, extraction scripts, and adapted SpaGCN
scripts/                repository and feature integration checks
tests/                  repository validation
.github/workflows/      automated static checks
requirements.txt        combined runtime dependencies
LICENSE                 license for project-owned code
THIRD_PARTY_NOTICES.md   third-party attribution and license scope
CITATION.cff            software citation metadata
```

## License and citation

Project-owned code is distributed under the [MIT License](LICENSE).
[Third-party notices](THIRD_PARTY_NOTICES.md) describe the scope of bundled
software and model dependencies. Citation metadata is provided in
[CITATION.cff](CITATION.cff).
