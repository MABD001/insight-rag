.DEFAULT_GOAL := help
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := uv pip

.PHONY: help install run ingest test lint fmt eval eval-compare docker clean

help: ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install dependencies
	uv venv $(VENV)
	VIRTUAL_ENV=$(VENV) $(PIP) install -e ".[dev]"

run: ## Start the API and UI on http://localhost:8000
	$(VENV)/bin/uvicorn app.main:app --reload --port 8000

ingest: ## Index everything in sample_docs/
	$(PY) -m scripts.ingest sample_docs

test: ## Run the test suite (offline, no API key)
	$(PY) -m pytest -q

lint: ## Lint with ruff
	$(VENV)/bin/ruff check app tests eval scripts

fmt: ## Auto-fix lint issues
	$(VENV)/bin/ruff check --fix app tests eval scripts

eval: ## Score the golden set and write eval/report.json
	$(PY) -m eval.run

eval-compare: ## Score the golden set and diff against the previous run
	$(PY) -m eval.run --compare

docker: ## Build and run the full stack with Postgres + pgvector
	docker compose up --build

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ eval/report.json
