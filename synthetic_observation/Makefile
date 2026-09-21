.PHONY: check compile smoke test

check:
	python scripts/check_public_repo.py
	python scripts/check_generator_entrypoint_contracts.py

compile:
	python -m compileall -q src scripts

smoke:
	python scripts/check_deep_generator_task_cli_smoke.py

test:
	python -m unittest discover -s tests -v
