# AnnData data contract

Every task consumes an AnnData object. The following fields are the stable
interface between preparation, generation, paradigm, and evaluation stages.

## Required fields

| Field | Meaning |
| --- | --- |
| `X` | Expression matrix. Generation uses the prepared HVG representation. |
| `obs_names` | Stable spot or cell identifiers. |
| `obs["slice_id"]` or `obs["sample_id"]` | Spatial slice or biological sample identifier. |
| `obsm["spatial"]` | Two-dimensional spatial coordinates. |

DLPFC supervised tasks additionally require `obs["label"]`. Sample-level disease
tasks require `obs["disease_label"]` at the sample level; spot-level copies may
carry the propagated value for aggregation.

## Synthetic metadata

Generated observations should include:

- `source="synthetic"` and the generator name in `generator`;
- `split="synthetic"` and a unique `synthetic_id`;
- the source slice or sample identifier when a pool is generated per domain.

Trusted labels are retained only for supervised paradigms that explicitly use
the generator's conditional label. Unsupervised or leakage-sensitive paths
drop labels before fitting. The test split remains real-only in the standard
configurations.

## Representation rules

Preparation scripts normalize counts, select highly variable genes, and store
the selected feature names in `var`. Downstream code expects matching feature
order between real and synthetic matrices. Coordinate assignment policies are
selected explicitly (`spatial_resampling`, `knn_mapping`,
`spatial_perturbation`, or `gmm_sampling`) and should be recorded with the
output package.

## Adding a new dataset

Add one preparation script, one YAML configuration, and a small CPU smoke test.
Document the label semantics, sample/slice split, feature preprocessing, and
the synthetic-label policy for each paradigm. Keep dataset files in the local
data workspace described in `data/README.md`.
