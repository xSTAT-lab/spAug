# spAug

spAug is a reproducible toolkit for augmenting spatial transcriptomics data
with synthetic spots and evaluating their effect on downstream tasks. It keeps
generation, coordinate assignment, training paradigms, and evaluation as
separate stages so the same synthetic pool can support multiple experiments.

The public tree provides source code, task configurations, toy smoke checks, and
documentation. Research datasets, checkpoints, logs, and result tables are managed through
the documented local data workflow.

spAug is a repository-root research toolkit. Run commands from the repository
root so the numeric stage directories under `src/` and relative `configs/`
paths resolve consistently.

## What is included

The active tasks are:

- DLPFC spatial clustering (unsupervised)
- DLPFC supervised low-label prediction
- DLPFC cross-slice generalization
- Sample-level disease prediction for Kidney, Bowel, and Brain cohorts

The generator layer includes project-local implementations for SRTsim,
Splatter, SPARsim, scGAN, and scDiffusion. The `_faithful` directory contains
shared statistical cores; task directories preserve command-line entry points
and metadata contracts. Model-specific numerical validation is described in
[`docs/model_provenance.md`](docs/model_provenance.md).

## Installation

Use Python 3.10 or newer in a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency file covers the complete research workflow. Some optional
generators use PyTorch, and some datasets use additional readers. Install
`requirements-extra.txt` for the optional Vine-Copula path and
`requirements-dev.txt` when running the checks.

## Quick checks

Run these commands from the repository root:

```bash
python scripts/check_public_repo.py
python scripts/check_generator_entrypoint_contracts.py
python scripts/check_srtsim_faithful_core.py
python scripts/check_splatter_faithful_core.py
python scripts/check_sparsim_faithful_core.py
python scripts/check_scgan_mechanism.py
python scripts/check_scdiffusion_mechanism.py
```

These checks use small, locally generated synthetic arrays or tensors.
For a command-line smoke test across task entry points, run
`python scripts/check_deep_generator_task_cli_smoke.py`.

## Typical workflow

1. Prepare an AnnData object with the fields described in
   [`docs/data_contract.md`](docs/data_contract.md).
2. Run a task-specific preparation script in `src/01_data_prep`.
3. Generate a synthetic pool with the matching entry point under
   `src/02_generators`.
4. Assign coordinates and build a paradigm package under `src/04_paradigm`.
5. Run a downstream predictor under `src/05_downstream` and evaluate intrinsic
   or downstream metrics with `src/06_evaluation`.

For example, the DLPFC spatial-clustering driver accepts a real AnnData file
and a synthetic pool:

```bash
python src/05_downstream/spatial_cluster/DLPFC/run_streaming_paradigm.py \
  --real data/dlpfc/real.h5ad \
  --synthetic-pool data/dlpfc/synthetic_pool.h5ad \
  --family global_real_plus_synthetic \
  --config configs/spatial_cluster/DLPFC/paradigm.yaml \
  --results-root results/dlpfc_spatial_cluster
```

The supervised and cross-slice predictors are documented by their built-in
help (`--help`) and use the corresponding YAML files in `configs/`.

## Reproducibility and scope

Random seeds, split definitions, coordinate policies, and generator metadata
are recorded in AnnData attributes or sidecar files by the task scripts. The
test split for supervised tasks remains real-only unless a task configuration
explicitly enables another policy. Synthetic observations carry `source`,
`generator`, `split`, and `synthetic_id` metadata; see the data contract for
the leakage rules used by each paradigm.

The repository is research software. Reproducing a published number requires
the original dataset, a matching configuration, and the exact dependency
versions and hardware details used for the run.

## Repository layout

```text
src/01_data_prep/       dataset preparation and split definitions
src/02_generators/      generator entry points and shared statistical cores
src/04_paradigm/        synthetic-pool and coordinate-assignment logic
src/05_downstream/      prediction and clustering experiments
src/06_evaluation/      intrinsic and downstream metrics
configs/                task-specific YAML configuration
scripts/                portable checks and example runners
docs/                   data, provenance, and reproducibility notes
```

## License

The original orchestration and project code are released under the MIT
License. Third-party projects and dependencies retain their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`docs/model_provenance.md`](docs/model_provenance.md).

## Citation

Repository metadata is provided in [`CITATION.cff`](CITATION.cff). Add the
named authors and final repository URL to that file before creating a public
release.
