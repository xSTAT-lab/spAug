# Contributing

Bug reports and pull requests are welcome. Please include the command used,
the Python version, and a small reproducible example when reporting a failure.

Before opening a pull request, run:

```bash
python scripts/check_public_repo.py
python scripts/check_generator_entrypoint_contracts.py
```

Keep downloaded data, checkpoints, logs, and result tables outside Git. New
tasks should document their AnnData contract and provide a CPU-friendly smoke
check. Use English for public documentation, configuration comments, command
help, and log messages. Preserve established field names such as
`obs["slice_id"]` and `obsm["spatial"]` exactly.
