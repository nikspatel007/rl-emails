.PHONY: install lint format type-check test test-unit test-integration coverage check hooks clean run status

# =============================================================================
# Installation
# =============================================================================

install:
	uv sync --all-extras

hooks:
	uv run pre-commit install

# =============================================================================
# Code Quality
# =============================================================================

lint:
	uv run ruff check src/ tests/ scripts/
	uv run ruff format --check src/ tests/ scripts/

format:
	uv run ruff check --fix src/ tests/ scripts/
	uv run ruff format src/ tests/ scripts/

type-check:
	uv run mypy src/ --strict

# For current state (scripts only, before restructure)
type-check-scripts:
	uv run mypy scripts/

# =============================================================================
# Testing
# =============================================================================

test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

coverage:
	uv run pytest tests/ -v --cov=src/rl_emails --cov-report=term-missing --cov-report=html --cov-fail-under=100
	@echo ""
	@echo "Coverage report: htmlcov/index.html"

# For current state (scripts only, before restructure)
coverage-scripts:
	uv run pytest tests/ -v --cov=scripts --cov-report=term-missing --cov-report=html --cov-fail-under=100
	@echo ""
	@echo "Coverage report: htmlcov/index.html"

# =============================================================================
# All Checks (What pre-commit runs)
# =============================================================================

# Full check (after restructure)
check: lint type-check test coverage
	@echo ""
	@echo "All checks passed!"

# Check for current state (scripts only)
check-scripts: type-check-scripts
	@echo ""
	@echo "Type checks passed!"

# =============================================================================
# Pipeline
# =============================================================================

run:
	uv run python scripts/onboard_data.py

status:
	uv run python scripts/onboard_data.py --status

validate:
	uv run python scripts/validate_data.py

# =============================================================================
# Maintenance
# =============================================================================

clean:
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned cache files"

# =============================================================================
# Help
# =============================================================================

help:
	@echo "rl-emails Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install all dependencies"
	@echo "  make hooks            Install pre-commit hooks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Auto-fix formatting"
	@echo "  make type-check       Run mypy strict (src/)"
	@echo "  make type-check-scripts  Run mypy (scripts/)"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make coverage         Run tests with 100% coverage requirement"
	@echo ""
	@echo "All Checks:"
	@echo "  make check            Run all checks (lint + type + test + coverage)"
	@echo "  make check-scripts    Run type checks on scripts/"
	@echo ""
	@echo "Pipeline:"
	@echo "  make run              Run the full pipeline"
	@echo "  make status           Check pipeline status"
	@echo "  make validate         Validate data"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove cache files"
