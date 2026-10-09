.PHONY: setup data train evaluate demo test ci-smoke lint

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -r requirements.txt
	cd web && npm install && npm run build

data:
	$(PYTHON) download_data.py
	$(PYTHON) -m scripts.build_all_data

train:
	$(PYTHON) -m scripts.train_all_models

evaluate:
	$(PYTHON) -m scripts.evaluate_all_models

demo:
	$(PYTHON) run_app.py

test:
	$(PYTHON) -m pytest tests/ -v

ci-smoke:
	$(PYTHON) scripts/ci_smoke_test.py

