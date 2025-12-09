# =============================================================================
# NexusMatcher v2.0.0 — Makefile
# =============================================================================
# Convenient commands for development and deployment
#
# Usage:
#   make install    # Create venv and install dependencies
#   make test       # Run test suite
#   make start      # Start the API server
#   make all        # Install, test, and start
#   make clean      # Remove venv and cache files
#   make help       # Show all available commands
# =============================================================================

.PHONY: all install test start clean help lint format benchmark docs docker

# Configuration
PYTHON ?= python3
VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin
PIP := $(VENV_BIN)/pip
PYTEST := $(VENV_BIN)/pytest
UVICORN := $(VENV_BIN)/uvicorn

# Server settings
HOST ?= 0.0.0.0
PORT ?= 8000
WORKERS ?= 4
LOG_LEVEL ?= info

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

# =============================================================================
# MAIN TARGETS
# =============================================================================

## all: Install dependencies, run tests, and start server
all: install test start

## help: Show this help message
help:
	@echo ""
	@echo "$(BLUE)NexusMatcher v2.0.0 — Available Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup & Installation:$(NC)"
	@echo "  make install       Create venv and install dependencies"
	@echo "  make install-dev   Install with development dependencies"
	@echo "  make install-full  Install all optional dependencies"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  make test          Run test suite"
	@echo "  make test-fast     Run tests without coverage"
	@echo "  make test-unit     Run only unit tests"
	@echo "  make lint          Run linting (ruff)"
	@echo "  make format        Format code (black + ruff)"
	@echo "  make typecheck     Run type checking (mypy)"
	@echo ""
	@echo "$(GREEN)Benchmarks:$(NC)"
	@echo "  make benchmark     Run all benchmarks"
	@echo "  make bench-cache   Run cache benchmarks (SUITE-004)"
	@echo "  make bench-colbert Run ColBERT benchmarks (SUITE-003)"
	@echo "  make bench-incr    Run incremental update benchmarks (SUITE-005)"
	@echo ""
	@echo "$(GREEN)Server:$(NC)"
	@echo "  make start         Start the API server"
	@echo "  make start-dev     Start in development mode (reload)"
	@echo ""
	@echo "$(GREEN)Publishing (PyPI):$(NC)"
	@echo "  make build         Build distribution packages"
	@echo "  make check-build   Verify built packages"
	@echo "  make publish-test  Publish to TestPyPI"
	@echo "  make publish       Publish to PyPI (production)"
	@echo "  make release       Full release workflow"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  make clean         Remove venv and cache files"
	@echo "  make clean-cache   Remove only cache files"
	@echo "  make clean-build   Remove build artifacts"
	@echo "  make update        Update all dependencies"
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo "  make docker-up     Start with docker-compose"
	@echo ""
	@echo "$(YELLOW)Environment Variables:$(NC)"
	@echo "  HOST=$(HOST)  PORT=$(PORT)  WORKERS=$(WORKERS)  LOG_LEVEL=$(LOG_LEVEL)"
	@echo ""

# =============================================================================
# SETUP & INSTALLATION
# =============================================================================

## install: Create venv and install dependencies
install: $(VENV_DIR)
	@echo "$(BLUE)[INFO]$(NC) Installing dependencies..."
	@$(PIP) install --upgrade pip --quiet
	@$(PIP) install -r requirements.txt --quiet
	@$(PIP) install -e . --quiet
	@echo "$(GREEN)[✓]$(NC) Installation complete"

## install-dev: Install with development dependencies
install-dev: $(VENV_DIR)
	@echo "$(BLUE)[INFO]$(NC) Installing with dev dependencies..."
	@$(PIP) install --upgrade pip --quiet
	@$(PIP) install -r requirements.txt --quiet
	@$(PIP) install -e ".[dev]" --quiet
	@echo "$(GREEN)[✓]$(NC) Development installation complete"

## install-full: Install all optional dependencies
install-full: $(VENV_DIR)
	@echo "$(BLUE)[INFO]$(NC) Installing all dependencies..."
	@$(PIP) install --upgrade pip --quiet
	@$(PIP) install -r requirements.txt --quiet
	@$(PIP) install -e ".[full]" --quiet
	@echo "$(GREEN)[✓]$(NC) Full installation complete"

$(VENV_DIR):
	@echo "$(BLUE)[INFO]$(NC) Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "$(GREEN)[✓]$(NC) Virtual environment created"

# =============================================================================
# TESTING
# =============================================================================

## test: Run test suite with coverage
test: $(VENV_DIR)
	@echo "$(BLUE)[INFO]$(NC) Running tests..."
	@$(PYTEST) tests/ -v --tb=short
	@echo "$(GREEN)[✓]$(NC) Tests complete"

## test-fast: Run tests without coverage
test-fast: $(VENV_DIR)
	@$(PYTEST) tests/ -v --tb=short --no-cov

## test-unit: Run only unit tests
test-unit: $(VENV_DIR)
	@$(PYTEST) tests/unit/ -v --tb=short --no-cov

## test-integration: Run integration tests
test-integration: $(VENV_DIR)
	@$(PYTEST) tests/integration/ -v --tb=short --no-cov

# =============================================================================
# CODE QUALITY
# =============================================================================

## lint: Run linting
lint: $(VENV_DIR)
	@echo "$(BLUE)[INFO]$(NC) Running ruff..."
	@$(VENV_BIN)/ruff check src/ tests/

## format: Format code
format: $(VENV_DIR)
	@echo "$(BLUE)[INFO]$(NC) Formatting code..."
	@$(VENV_BIN)/ruff format src/ tests/
	@$(VENV_BIN)/ruff check --fix src/ tests/
	@echo "$(GREEN)[✓]$(NC) Formatting complete"

## typecheck: Run type checking
typecheck: $(VENV_DIR)
	@echo "$(BLUE)[INFO]$(NC) Running mypy..."
	@$(VENV_BIN)/mypy src/

# =============================================================================
# BENCHMARKS
# =============================================================================

## benchmark: Run all benchmarks
benchmark: $(VENV_DIR)
	@echo "$(BLUE)[INFO]$(NC) Running all benchmarks..."
	@$(VENV_BIN)/python benchmarks/suite_003_colbert_reranking.py
	@$(VENV_BIN)/python benchmarks/suite_004_cache_performance.py
	@CORPUS_SIZE=10000 $(VENV_BIN)/python benchmarks/suite_005_incremental_updates.py
	@echo "$(GREEN)[✓]$(NC) Benchmarks complete"

## bench-cache: Run cache benchmarks
bench-cache: $(VENV_DIR)
	@$(VENV_BIN)/python benchmarks/suite_004_cache_performance.py
	@$(VENV_BIN)/python benchmarks/suite_004b_semantic_cache.py
	@$(VENV_BIN)/python benchmarks/suite_004c_context_enrichment.py

## bench-colbert: Run ColBERT benchmarks
bench-colbert: $(VENV_DIR)
	@$(VENV_BIN)/python benchmarks/suite_003_colbert_reranking.py

## bench-incr: Run incremental update benchmarks
bench-incr: $(VENV_DIR)
	@CORPUS_SIZE=50000 $(VENV_BIN)/python benchmarks/suite_005_incremental_updates.py

## bench-quant: Run quantization benchmarks
bench-quant: $(VENV_DIR)
	@$(VENV_BIN)/python benchmarks/suite_002_quantization.py

# =============================================================================
# SERVER
# =============================================================================

## start: Start the API server
start: $(VENV_DIR)
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  NexusMatcher API Server$(NC)"
	@echo "$(GREEN)  API Docs: http://$(HOST):$(PORT)/docs$(NC)"
	@echo "$(GREEN)  Health: http://$(HOST):$(PORT)/health$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@$(UVICORN) nexus_matcher.presentation.api.app:app \
		--host $(HOST) \
		--port $(PORT) \
		--workers $(WORKERS) \
		--log-level $(LOG_LEVEL)

## start-dev: Start in development mode with auto-reload
start-dev: $(VENV_DIR)
	@$(UVICORN) nexus_matcher.presentation.api.app:app \
		--host $(HOST) \
		--port $(PORT) \
		--reload \
		--log-level debug

# =============================================================================
# MAINTENANCE
# =============================================================================

## clean: Remove venv and cache files
clean: clean-cache
	@echo "$(BLUE)[INFO]$(NC) Removing virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "$(GREEN)[✓]$(NC) Clean complete"

## clean-cache: Remove only cache files
clean-cache:
	@echo "$(BLUE)[INFO]$(NC) Removing cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -f coverage.xml .coverage 2>/dev/null || true
	@echo "$(GREEN)[✓]$(NC) Cache cleaned"

## update: Update all dependencies
update: $(VENV_DIR)
	@echo "$(BLUE)[INFO]$(NC) Updating dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install --upgrade -r requirements.txt
	@$(PIP) install --upgrade -e .
	@echo "$(GREEN)[✓]$(NC) Dependencies updated"

# =============================================================================
# DOCKER
# =============================================================================

## docker-build: Build Docker image
docker-build:
	@echo "$(BLUE)[INFO]$(NC) Building Docker image..."
	@docker build -t nexus-matcher:latest -f docker/Dockerfile .
	@echo "$(GREEN)[✓]$(NC) Docker image built"

## docker-run: Run Docker container
docker-run:
	@docker run -p $(PORT):$(PORT) \
		-e NEXUS_HOST=$(HOST) \
		-e NEXUS_PORT=$(PORT) \
		nexus-matcher:latest

## docker-up: Start with docker-compose
docker-up:
	@docker-compose -f docker/docker-compose.yml up -d

## docker-down: Stop docker-compose
docker-down:
	@docker-compose -f docker/docker-compose.yml down

# =============================================================================
# DOCUMENTATION
# =============================================================================

## docs: Build documentation
docs: $(VENV_DIR)
	@$(PIP) install -e ".[docs]" --quiet
	@$(VENV_BIN)/mkdocs build

## docs-serve: Serve documentation locally
docs-serve: $(VENV_DIR)
	@$(PIP) install -e ".[docs]" --quiet
	@$(VENV_BIN)/mkdocs serve

# =============================================================================
# PUBLISHING
# =============================================================================

## build: Build distribution packages
build: $(VENV_DIR) clean-build
	@echo "$(BLUE)[INFO]$(NC) Building packages..."
	@$(PIP) install --upgrade build twine --quiet
	@$(VENV_BIN)/python -m build
	@echo "$(GREEN)[✓]$(NC) Build complete. Packages in dist/"

## clean-build: Remove build artifacts
clean-build:
	@rm -rf dist/ build/
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

## check-build: Verify built packages
check-build: $(VENV_DIR)
	@$(VENV_BIN)/twine check dist/*

## publish-test: Publish to TestPyPI
publish-test: build check-build
	@echo "$(YELLOW)[WARN]$(NC) Publishing to TestPyPI..."
	@$(VENV_BIN)/twine upload --repository testpypi dist/*
	@echo "$(GREEN)[✓]$(NC) Published to TestPyPI"
	@echo "Install with: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nexus-matcher"

## publish: Publish to PyPI (PRODUCTION)
publish: build check-build
	@echo "$(RED)[WARN]$(NC) Publishing to PyPI (PRODUCTION)!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	@$(VENV_BIN)/twine upload dist/*
	@echo "$(GREEN)[✓]$(NC) Published to PyPI"
	@echo "Install with: pip install nexus-matcher"

## release: Full release workflow
release: test lint typecheck build check-build
	@echo ""
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  Release Ready!$(NC)"
	@echo "$(GREEN)  Version: $$($(VENV_BIN)/python -c 'import nexus_matcher; print(nexus_matcher.__version__)')$(NC)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. make publish-test  # Test on TestPyPI first"
	@echo "  2. make publish       # Publish to PyPI"
	@echo "  3. git tag -a v$$($(VENV_BIN)/python -c 'import nexus_matcher; print(nexus_matcher.__version__)') -m 'Release'"
	@echo "  4. git push origin --tags"
	@echo ""
