# Third-party notices

The root [MIT License](LICENSE) covers project-owned code. Third-party software,
models, datasets, and dependencies retain their respective licenses and terms.

## Synthetic-observation module

The `syn/` module contains project-local implementations of published spatial
and single-cell simulation methods. Model names and source references are
recorded in generator metadata and the
[model provenance guide](syn/docs/model_provenance.md). Consult the relevant
method and dependency terms when redistributing related work.

## Feature module

`feature/spagcn/` contains adapted SpaGCN code. Its original MIT copyright and
license are retained in [feature/spagcn/LICENSE](feature/spagcn/LICENSE).
The [SpaGCN guide](feature/spagcn/README.md) describes the graph integration
mechanism and provides attribution to SpaGCN and spEMO.

GPFM, UNI, UNI2-h, and GigaPath implementations and weights are supplied locally
under their providers' terms. The feature extraction script accesses them
through the model registry interface documented in [feature/README.md](feature/README.md).

