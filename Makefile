.PHONY: install test check-data evaluate verify-results reproduce audit

install:
	python -m pip install -e '.[dev]'

test:
	pytest

check-data:
	PYTHONPATH=src python -m threshold_queries.inputs

evaluate:
	MPLCONFIGDIR=.cache/matplotlib PYTHONPATH=src python -m threshold_queries.workflow

verify-results:
	PYTHONPATH=src python -m threshold_queries.verification

reproduce: evaluate
	$(MAKE) verify-results

audit:
	black --check src tests experiments data/fetch_uscrn.py
	flake8 src tests experiments data/fetch_uscrn.py
	pytest
	python -m compileall -q src experiments tests
	$(MAKE) verify-results
