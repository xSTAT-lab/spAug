PYTHON ?= python

.PHONY: check smoke test

check:
	$(PYTHON) scripts/check_public_repo.py
	$(PYTHON) syn/scripts/check_generator_entrypoint_contracts.py

test:
	$(PYTHON) -m unittest discover -s tests -v

smoke:
	$(PYTHON) scripts/check_public_repo.py --with-mechanisms
	$(PYTHON) scripts/check_feature_setup.py
	$(PYTHON) syn/scripts/check_deep_generator_task_cli_smoke.py
