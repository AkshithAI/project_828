# ─── Project 828 — Build & Test Automation ──────────────────────────────────
.PHONY: test test-cpu test-all lint docker-build docker-run clean help

PYTHON   ?= python3
PYTEST   ?= $(PYTHON) -m pytest
DOCKER_TAG ?= project-828:latest

# ── Test targets ────────────────────────────────────────────────────────────

## Run the full CPU test suite (used in CI)
test:
	$(PYTEST) tests/ \
		-v --tb=short \
		-m "not requires_cuda" \
		-k "not cuda" \
		--strict-markers \
		-x

## Alias: explicit CPU-only run
test-cpu: test

## Run ALL tests including CUDA (requires GPU)
test-all:
	$(PYTEST) tests/ -v --tb=short --strict-markers

# ── Docker ──────────────────────────────────────────────────────────────────

## Build the Docker image
docker-build:
	docker build -t $(DOCKER_TAG) .

## Run the container (interactive shell)
docker-run:
	docker run --rm -it $(DOCKER_TAG)

## Run tests inside Docker
docker-test:
	docker run --rm $(DOCKER_TAG) make test

# ── Housekeeping ────────────────────────────────────────────────────────────

## Remove Python cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

## Print this help
help:
	@echo ""
	@echo "  Project 828 — Available targets:"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make test          Run full CPU test suite (CI)"
	@echo "  make test-cpu      Alias for 'make test'"
	@echo "  make test-all      Run ALL tests including CUDA"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run container interactively"
	@echo "  make docker-test   Run tests inside Docker"
	@echo "  make clean         Remove __pycache__, .pytest_cache"
	@echo "  make help          Print this help"
	@echo ""
