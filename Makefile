.PHONY: help venv install kernel test clean notebook-main kaggle-submit

SHELL := /bin/bash
PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
JUPYTER := .venv/bin/jupyter
KAGGLE := .venv/bin/kaggle

SEED ?= 42
NOTEBOOK_MAIN ?= notebooks/00_main_pipeline.ipynb
SUBMISSION_FILE ?= output/submission.csv
COMPETITION ?=

help:
	@echo "Target yang tersedia:"
	@echo "  make venv           - Buat virtual environment (.venv)"
	@echo "  make install        - Install dependencies dari requirements.txt"
	@echo "  make kernel         - Daftarkan kernel Jupyter untuk proyek ini"
	@echo "  make test           - Jalankan unit & reproducibility test dengan pytest"
	@echo "  make notebook-main  - Jalankan notebook utama secara headless (SEED=$(SEED))"
	@echo "  make kaggle-submit  - Upload submission ke Kaggle (butuh COMPETITION=...)"
	@echo "  make clean          - Hapus cache, pytest cache, dan file sementara"

venv:
	test -d .venv || python3 -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

kernel: install
	$(PYTHON) -m ipykernel install --user --name=ml-tim-project --display-name="Python (ml-tim-project)"

test:
	$(PYTEST) -v tests/

notebook-main:
	SEED=$(SEED) $(JUPYTER) nbconvert --to notebook --execute --inplace $(NOTEBOOK_MAIN)

kaggle-submit:
	@test -n "$(COMPETITION)" || (echo "Set COMPETITION=<slug-kompetisi>" && exit 1)
	@test -f "$(SUBMISSION_FILE)" || (echo "File $(SUBMISSION_FILE) tidak ditemukan" && exit 1)
	$(KAGGLE) competitions submit -c $(COMPETITION) -f $(SUBMISSION_FILE) -m "seed=$(SEED) baseline submission"

clean:
	rm -rf .pytest_cache __pycache__ notebooks/.ipynb_checkpoints
	find src tests scripts -type d -name "__pycache__" -exec rm -rf {} +
	find src tests scripts -type f -name "*.py[cod]" -delete
