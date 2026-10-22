.PHONY: setup data train evaluate render-readme demo test ci-smoke lint archive check-readme

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
	$(PYTHON) scripts/render_readme.py

render-readme:
	$(PYTHON) scripts/render_readme.py

check-readme:
	$(PYTHON) scripts/render_readme.py --check

demo:
	$(PYTHON) run_app.py

test:
	$(PYTHON) -m pytest tests/ -v

ci-smoke:
	$(PYTHON) scripts/ci_smoke_test.py

# Ship only tracked source: git archive cannot include untracked DB/runtime state.
archive:
	git archive --format=zip --prefix=AETURNUS/ -o AETURNUS_source.zip HEAD
	@$(PYTHON) -c "import sys, zipfile; names = zipfile.ZipFile('AETURNUS_source.zip').namelist(); bad = [n for n in names if n.endswith(('.db', '.db-wal', '.db-shm', '.db-journal'))]; sys.exit(f'Archive contains DB state: {bad}') if bad else print(f'Archive OK: {len(names)} tracked files, no DB state')"


