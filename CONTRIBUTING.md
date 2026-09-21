# Contributing

Use issues for questions and reproducible problem reports, and pull requests
for proposed changes.

Include the command, Python version, relevant configuration, and a compact
synthetic example when reporting a problem. Use English for documentation,
configuration comments, command help, and log messages. Describe current
behavior and validation results in neutral, factual language.

Run from the repository root before submitting changes:

```bash
python scripts/check_public_repo.py
python syn/scripts/check_generator_entrypoint_contracts.py
python -m unittest discover -s tests -v
```

For changes involving module-local notebook imports or model execution, also run
`python scripts/check_feature_setup.py` and the relevant mechanism checks.

Keep dataset files, outputs, model weights, and credentials in local ignored
locations. Save notebooks with cleared outputs and execution counts, and use
generic kernel metadata. Preserve required third-party license notices.
