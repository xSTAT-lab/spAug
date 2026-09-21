# Reproducibility guide

Use a fresh environment and record the output of `python --version`,
`pip freeze`, the selected YAML files, and the random seeds. Keep raw data,
intermediate AnnData files, checkpoints, and results outside Git.

For each experiment, record:

1. the preparation script and its input data version;
2. the generator, pool ratio, and generator seed;
3. the coordinate policy and paradigm family;
4. the real/synthetic split and held-out samples or slices;
5. the classifier, PCA dimension, and hardware backend;
6. intrinsic and downstream evaluation commands.

The supplied configurations provide DLPFC and sample-disease split templates
that can be adapted to a dataset-specific methods record. Sample-level
generation uses a pool ratio at least as large as the largest downstream ratio
requested by the selected configuration.
