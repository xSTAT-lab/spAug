# Examples

The full workflows require external spatial transcriptomics data. The
repository-level checks use generated toy arrays and are the quickest way to
verify an installation:

Run from `syn/`:

```bash
python scripts/check_deep_generator_task_cli_smoke.py
python scripts/check_public_repo.py --with-mechanisms
```

For a real experiment, start from the matching YAML under `configs/`, prepare
an AnnData file, and follow the stages in the [module guide](../README.md).
