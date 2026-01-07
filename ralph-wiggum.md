# Ralph Wiggum Production Plan

**Goal**: Transform the codebase into a clean, production-ready system with:
- Proper package structure
- 100% strict typing (mypy strict mode)
- 100% test coverage
- Pre-commit hooks enforcing quality gates

**Status**: COMPLETE - All 8 iterations finished

---

## Current State Analysis

### What Exists
```
rl-emails/
├── scripts/              # 15 Python files (pipeline + utilities)
├── tests/                # Empty (only __init__.py)
├── archive/              # Archived experimental code
├── alembic/              # Database migrations
├── data/                 # Email data
├── pyproject.toml        # Project config (mypy/ruff configured)
└── CLAUDE.md             # Project instructions
```

### Problems
1. **No package structure**: Scripts are standalone files, not importable modules
2. **No type hints**: mypy strict fails with 200+ errors
3. **No tests**: 0% test coverage
4. **No enforcement**: Nothing prevents bad code from being committed

---

## Target State

### Clean Structure
```
rl-emails/
├── src/
│   └── rl_emails/
│       ├── __init__.py
│       ├── pipeline/           # Pipeline stages as modules
│       │   ├── __init__.py
│       │   ├── parse_mbox.py
│       │   ├── import_to_postgres.py
│       │   ├── populate_threads.py
│       │   ├── enrich_emails.py
│       │   ├── compute_features.py
│       │   ├── compute_embeddings.py
│       │   ├── classify_ai.py
│       │   ├── populate_users.py
│       │   ├── cluster_emails.py
│       │   ├── compute_priority.py
│       │   └── classify_llm.py
│       ├── core/               # Shared utilities
│       │   ├── __init__.py
│       │   ├── db.py           # Database connection helpers
│       │   ├── config.py       # Configuration management
│       │   └── types.py        # Shared type definitions
│       └── cli/                # Command-line interface
│           ├── __init__.py
│           └── main.py         # Entry point (onboard_data.py logic)
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures
│   ├── unit/                   # Unit tests
│   │   ├── __init__.py
│   │   └── pipeline/
│   │       ├── test_parse_mbox.py
│   │       └── ...
│   └── integration/            # Integration tests
│       ├── __init__.py
│       └── test_full_pipeline.py
├── scripts/                    # Thin CLI wrappers (optional)
├── alembic/
├── pyproject.toml
├── Makefile
└── .pre-commit-config.yaml
```

### Quality Gates
- `mypy --strict` passes with 0 errors
- `pytest --cov` shows 100% coverage
- `ruff check` shows 0 warnings
- Pre-commit hooks enforce all gates before every commit

---

## Iteration Plan

### Iteration 1: Add Type Hints to Existing Scripts
**Goal**: Make mypy strict pass on current structure before restructuring

**Success Criteria**:
- [ ] `mypy scripts/` exits with 0 errors
- [ ] All 15 scripts have full type annotations
- [ ] Commit passes CI checks

**Tasks**:
1. Add `from __future__ import annotations` to all files
2. Add type hints to all functions:
   - parse_mbox.py
   - import_to_postgres.py
   - populate_threads.py
   - enrich_emails_db.py
   - compute_basic_features.py
   - compute_embeddings.py
   - classify_ai_handleability.py
   - populate_users.py
   - cluster_emails.py
   - compute_priority.py
   - run_llm_classification.py
   - onboard_data.py
   - checkpoint.py
   - query_db.py
   - validate_data.py
3. Fix all mypy errors
4. Commit: "Add type hints to all scripts"

**Verification Contract**:
```bash
# MUST pass with exit code 0
make type-check-scripts

# Expected output:
# Success: no issues found in 15 source files

# MUST show 0 errors
uv run mypy scripts/ 2>&1 | grep -E "^Found [0-9]+ error" || echo "PASS: No errors"
```

**Pass/Fail Criteria**:
- PASS: `make type-check-scripts` exits with code 0
- FAIL: Any mypy error in any script

---

### Iteration 2: Create Package Structure
**Goal**: Reorganize code into proper Python package

**Success Criteria**:
- [ ] `src/rl_emails/` package exists
- [ ] All scripts converted to importable modules
- [ ] `mypy src/` passes
- [ ] Old functionality preserved

**Tasks**:
1. Create directory structure:
   ```bash
   mkdir -p src/rl_emails/{pipeline,core,cli}
   touch src/rl_emails/__init__.py
   touch src/rl_emails/{pipeline,core,cli}/__init__.py
   ```

2. Create `src/rl_emails/core/db.py`:
   - Database connection helpers
   - Connection context managers
   - Shared cursor utilities

3. Create `src/rl_emails/core/config.py`:
   - Environment variable loading
   - Configuration dataclass
   - Validation helpers

4. Create `src/rl_emails/core/types.py`:
   - TypedDict definitions for email data
   - Type aliases for common patterns

5. Move and refactor each script to `src/rl_emails/pipeline/`:
   - Extract shared code to core/
   - Keep module interface clean
   - Maintain backward compatibility via scripts/ wrappers

6. Update pyproject.toml:
   ```toml
   [tool.hatch.build.targets.wheel]
   packages = ["src/rl_emails"]
   ```

7. Create thin wrapper scripts in scripts/:
   ```python
   #!/usr/bin/env python3
   from rl_emails.pipeline.parse_mbox import main
   if __name__ == "__main__":
       main()
   ```

8. Verify all commands still work:
   ```bash
   uv run python -m rl_emails.cli.main --status
   uv run python scripts/onboard_data.py --status
   ```

9. Commit: "Restructure into rl_emails package"

**Verification Contract**:
```bash
# MUST pass with exit code 0
make type-check

# MUST import successfully
uv run python -c "from rl_emails.pipeline import parse_mbox; print('PASS: Import works')"

# MUST run status command
uv run python -m rl_emails.cli.main --status

# Backward compatibility MUST work
uv run python scripts/onboard_data.py --status
```

**Pass/Fail Criteria**:
- PASS: All 4 commands exit with code 0
- FAIL: Any command fails or import error

---

### Iteration 3: Add Test Infrastructure
**Goal**: Set up comprehensive testing framework

**Success Criteria**:
- [ ] `tests/` directory with proper structure
- [ ] conftest.py with shared fixtures
- [ ] Test database setup/teardown working
- [ ] At least 1 test runs successfully

**Tasks**:
1. Create test structure:
   ```bash
   mkdir -p tests/{unit,integration}/pipeline
   touch tests/__init__.py
   touch tests/{unit,integration}/__init__.py
   touch tests/unit/pipeline/__init__.py
   ```

2. Create `tests/conftest.py`:
   ```python
   import pytest
   import psycopg2
   from typing import Generator

   @pytest.fixture(scope="session")
   def test_db_url() -> str:
       """Test database URL (separate from prod)."""
       return "postgresql://postgres:postgres@localhost:5433/test_rl_emails"

   @pytest.fixture
   def db_connection(test_db_url: str) -> Generator[psycopg2.extensions.connection, None, None]:
       """Database connection for tests."""
       conn = psycopg2.connect(test_db_url)
       yield conn
       conn.rollback()
       conn.close()

   @pytest.fixture
   def sample_email() -> dict:
       """Sample email data for testing."""
       return {
           "message_id": "<test@example.com>",
           "from_email": "sender@example.com",
           "to_emails": ["recipient@example.com"],
           "subject": "Test Subject",
           "body_text": "Test body content",
           "date_str": "Mon, 1 Jan 2024 10:00:00 +0000",
       }
   ```

3. Create first unit test `tests/unit/pipeline/test_parse_mbox.py`:
   ```python
   from rl_emails.pipeline.parse_mbox import extract_email_address, clean_body

   def test_extract_email_address():
       assert extract_email_address("John Doe <john@example.com>") == "john@example.com"
       assert extract_email_address("jane@example.com") == "jane@example.com"

   def test_clean_body():
       html = "<html><body>Hello World</body></html>"
       assert "Hello World" in clean_body(html)
   ```

4. Update pyproject.toml pytest config:
   ```toml
   [tool.pytest.ini_options]
   testpaths = ["tests"]
   python_files = ["test_*.py"]
   python_functions = ["test_*"]
   addopts = "-v --tb=short --strict-markers"
   markers = [
       "unit: Unit tests (fast, no external deps)",
       "integration: Integration tests (requires database)",
   ]
   ```

5. Commit: "Add test infrastructure"

**Verification Contract**:
```bash
# MUST have at least 1 test passing
uv run pytest tests/ -v --tb=short

# MUST show test collection
uv run pytest tests/ --collect-only | grep "test session starts"

# MUST have conftest.py
test -f tests/conftest.py && echo "PASS: conftest.py exists"
```

**Pass/Fail Criteria**:
- PASS: `pytest tests/` exits with code 0, at least 1 test passes
- FAIL: No tests found or pytest fails

---

### Iteration 4: Unit Tests (Target 100% Coverage)
**Goal**: Add unit tests for all pure functions

**Success Criteria**:
- [ ] Unit tests for all pipeline modules
- [ ] Coverage = 100% for src/rl_emails/
- [ ] All tests pass

**Tasks**:
1. For each pipeline module, test:
   - Input validation
   - Data transformation functions
   - Error handling
   - Edge cases

2. Test files to create:
   - tests/unit/pipeline/test_parse_mbox.py
   - tests/unit/pipeline/test_import_to_postgres.py
   - tests/unit/pipeline/test_populate_threads.py
   - tests/unit/pipeline/test_enrich_emails.py
   - tests/unit/pipeline/test_compute_features.py
   - tests/unit/pipeline/test_compute_embeddings.py
   - tests/unit/pipeline/test_classify_ai.py
   - tests/unit/pipeline/test_populate_users.py
   - tests/unit/pipeline/test_cluster_emails.py
   - tests/unit/pipeline/test_compute_priority.py
   - tests/unit/pipeline/test_classify_llm.py
   - tests/unit/core/test_db.py
   - tests/unit/core/test_config.py

3. Use mocking for external dependencies:
   ```python
   from unittest.mock import patch, MagicMock

   @patch("rl_emails.pipeline.compute_embeddings.openai")
   def test_get_embedding(mock_openai):
       mock_openai.embeddings.create.return_value = MagicMock(
           data=[MagicMock(embedding=[0.1] * 1536)]
       )
       # Test embedding generation
   ```

4. Commit: "Add unit tests (100% coverage)"

**Verification Contract**:
```bash
# MUST pass with 100% coverage
uv run pytest tests/unit/ -v --cov=src/rl_emails --cov-report=term-missing --cov-fail-under=100

# MUST show 100% in coverage report
uv run pytest tests/unit/ --cov=src/rl_emails --cov-report=term 2>&1 | grep "TOTAL" | grep "100%"

# MUST have test file for each pipeline module
ls tests/unit/pipeline/test_*.py | wc -l  # Should be 11+
```

**Pass/Fail Criteria**:
- PASS: Coverage report shows 100%, all tests pass
- FAIL: Coverage < 100% or any test fails

---

### Iteration 5: Integration Tests
**Goal**: Add integration tests for end-to-end validation

**Success Criteria**:
- [ ] Integration tests for full pipeline
- [ ] Tests use isolated test database
- [ ] All tests pass (unit + integration)

**Tasks**:
1. Create `tests/integration/test_full_pipeline.py`:
   - Test complete pipeline flow
   - Use test database with sample data
   - Verify database state after each stage

2. Create `tests/integration/test_database_operations.py`:
   - Test actual database queries
   - Test transaction handling
   - Test error recovery

3. Create test data fixtures:
   - Sample MBOX file
   - Expected output data
   - Test database schema

4. Commit: "Add integration tests"

**Verification Contract**:
```bash
# MUST pass all tests (unit + integration)
uv run pytest tests/ -v --cov=src/rl_emails --cov-report=term-missing --cov-fail-under=100

# MUST have integration test files
test -f tests/integration/test_full_pipeline.py && echo "PASS: Pipeline test exists"
test -f tests/integration/test_database_operations.py && echo "PASS: DB test exists"

# Integration tests MUST actually hit the database
uv run pytest tests/integration/ -v -k "database" --tb=short
```

**Pass/Fail Criteria**:
- PASS: All tests pass, integration tests exercise real database
- FAIL: Any test fails or integration tests are mocked

---

### Iteration 6: Pre-Commit Hooks
**Goal**: Enforce quality gates before every commit

**Success Criteria**:
- [ ] `.pre-commit-config.yaml` configured
- [ ] Pre-commit hooks installed
- [ ] Commit blocked if checks fail
- [ ] All team members can use hooks

**Tasks**:
1. Add pre-commit to dev dependencies:
   ```toml
   [project.optional-dependencies]
   dev = [
       "pre-commit>=3.5.0",
       # ... existing deps
   ]
   ```

2. Create `.pre-commit-config.yaml`:
   ```yaml
   repos:
     - repo: https://github.com/pre-commit/pre-commit-hooks
       rev: v4.5.0
       hooks:
         - id: trailing-whitespace
         - id: end-of-file-fixer
         - id: check-yaml
         - id: check-added-large-files

     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.1.9
       hooks:
         - id: ruff
           args: [--fix]
         - id: ruff-format

     - repo: local
       hooks:
         - id: mypy
           name: mypy
           entry: uv run mypy src/ --strict
           language: system
           types: [python]
           pass_filenames: false

         - id: pytest
           name: pytest
           entry: uv run pytest tests/ -x --tb=short
           language: system
           types: [python]
           pass_filenames: false
           stages: [commit]
   ```

3. Install hooks:
   ```bash
   uv run pre-commit install
   ```

4. Test hooks work:
   ```bash
   # This should run all checks
   uv run pre-commit run --all-files
   ```

5. Document hook setup in README.md

6. Commit: "Add pre-commit hooks"

**Verification Contract**:
```bash
# MUST have pre-commit config
test -f .pre-commit-config.yaml && echo "PASS: Config exists"

# MUST pass all hooks on clean code
uv run pre-commit run --all-files

# MUST block bad commits (test and revert)
echo "x: str = 123" > /tmp/test_bad_code.py
cp /tmp/test_bad_code.py src/rl_emails/core/test_temp.py
git add src/rl_emails/core/test_temp.py
uv run pre-commit run mypy; RESULT=$?
rm src/rl_emails/core/test_temp.py
git reset HEAD src/rl_emails/core/test_temp.py 2>/dev/null || true
[ $RESULT -ne 0 ] && echo "PASS: Bad code blocked" || echo "FAIL: Bad code not blocked"
```

**Pass/Fail Criteria**:
- PASS: `pre-commit run --all-files` exits 0 on clean code, exits non-0 on bad code
- FAIL: Hooks don't block type errors or test failures

---

### Iteration 7: Makefile & CI
**Goal**: Standardize development workflow

**Success Criteria**:
- [ ] Makefile with all common commands
- [ ] GitHub Actions CI (optional)
- [ ] Documentation complete

**Tasks**:
1. Create `Makefile`:
   ```makefile
   .PHONY: install lint type-check test test-unit test-integration coverage clean hooks

   # Installation
   install:
   	uv sync --all-extras

   # Code quality
   lint:
   	uv run ruff check src/ tests/
   	uv run ruff format --check src/ tests/

   format:
   	uv run ruff check --fix src/ tests/
   	uv run ruff format src/ tests/

   type-check:
   	uv run mypy src/ --strict

   # Testing
   test:
   	uv run pytest tests/ -v

   test-unit:
   	uv run pytest tests/unit/ -v

   test-integration:
   	uv run pytest tests/integration/ -v

   coverage:
   	uv run pytest tests/ --cov=src/rl_emails --cov-report=html --cov-fail-under=100
   	@echo "Coverage report: htmlcov/index.html"

   # All checks (what pre-commit runs)
   check: lint type-check test

   # Development
   hooks:
   	uv run pre-commit install

   clean:
   	rm -rf .mypy_cache .pytest_cache .coverage htmlcov .ruff_cache
   	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

   # Pipeline
   run:
   	uv run python -m rl_emails.cli.main

   status:
   	uv run python -m rl_emails.cli.main --status
   ```

2. (Optional) Create `.github/workflows/ci.yml`:
   ```yaml
   name: CI
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: astral-sh/setup-uv@v1
         - run: uv sync --all-extras
         - run: make check
   ```

3. Update README.md with development instructions

4. Commit: "Add Makefile and CI"

**Verification Contract**:
```bash
# MUST have Makefile
test -f Makefile && echo "PASS: Makefile exists"

# ALL make targets MUST work
make clean
make install
make lint
make type-check
make test
make coverage

# Full check MUST pass
make check

# CI file MUST exist (if created)
test -f .github/workflows/ci.yml && echo "PASS: CI exists" || echo "INFO: CI not created (optional)"
```

**Pass/Fail Criteria**:
- PASS: `make check` exits with code 0
- FAIL: Any make target fails

---

### Iteration 8: Final Cleanup & Documentation
**Goal**: Production-ready state

**Success Criteria**:
- [ ] All archive/ files deleted (or documented why kept)
- [ ] CLAUDE.md updated with new structure
- [ ] README.md complete
- [ ] No dead code

**Tasks**:
1. Review and clean `archive/`:
   - Delete if no longer needed
   - Or document why it's kept

2. Update CLAUDE.md:
   - New project structure
   - Development workflow
   - Testing instructions
   - Pre-commit hook setup

3. Update README.md:
   - Installation instructions
   - Quick start guide
   - API documentation (if applicable)

4. Final verification and commit: "Production ready release"

5. Create git tag: `v2.0.0`

**Verification Contract**:
```bash
# MUST have clean git status (no untracked important files)
git status --porcelain | grep -v "^??" | wc -l  # Should be 0

# MUST pass full check suite
make clean && make install && make check

# MUST have no archive/ directory OR it's documented
test ! -d archive/ && echo "PASS: archive/ deleted" || cat archive/README.md

# MUST have updated documentation
grep -q "100% test coverage" README.md && echo "PASS: README updated"
grep -q "pre-commit" README.md && echo "PASS: Pre-commit documented"

# MUST have version tag
git tag | grep -q "v2.0.0" && echo "PASS: Tag exists" || echo "INFO: Tag not yet created"

# Final sanity check: run the actual pipeline
make status
```

**Pass/Fail Criteria**:
- PASS: `make check` passes, documentation complete, tag created
- FAIL: Any check fails or documentation incomplete

---

## Commands Reference

```bash
# Setup
make install              # Install all dependencies
make hooks                # Install pre-commit hooks

# Development
make lint                 # Run ruff linter
make format               # Auto-fix formatting
make type-check           # Run mypy strict
make test                 # Run all tests
make coverage             # Run tests with coverage report
make check                # Run all checks (lint + type + test)

# Pipeline
make run                  # Run the full pipeline
make status               # Check pipeline status

# Maintenance
make clean                # Remove cache files
```

---

## Progress Log

| Iteration | Status | Date | Commit | Notes |
|-----------|--------|------|--------|-------|
| 1 | COMPLETE | 2026-01-07 | bd3ad72 | Add type hints - 15 scripts pass mypy |
| 2 | COMPLETE | 2026-01-07 | cd62284 | Package structure - src/rl_emails/ created |
| 3 | COMPLETE | 2026-01-07 | 3abfcc9 | Test infrastructure - 8 tests passing |
| 4 | COMPLETE | 2026-01-07 | e1cb7a1 | 100% coverage - 23 tests, lean package |
| 5 | COMPLETE | 2026-01-07 | 3777592 | Integration tests - 63 tests (8 skip w/o DB) |
| 6 | COMPLETE | 2026-01-07 | 8ebb9f9 | Pre-commit hooks installed |
| 7 | COMPLETE | 2026-01-07 | (prior) | Makefile created in earlier iterations |
| 8 | COMPLETE | 2026-01-07 | 1876060 | Documentation updated, production ready |

---

## Definition of Done

Each iteration is complete when:
1. All success criteria checkboxes are checked
2. **Verification Contract passes** (all commands exit 0)
3. Changes are committed with descriptive message
4. Progress log is updated with date and commit hash

**Critical Rule**: An iteration is NOT complete until the Verification Contract passes. No exceptions.

The project is production-ready when:
- `make check` passes (lint + types + tests + 100% coverage)
- Pre-commit hooks block bad commits (tested by verification)
- Documentation is complete
- Git tag `v2.0.0` is created

## Quality Gates Summary

| Gate | Command | Requirement |
|------|---------|-------------|
| Linting | `make lint` | 0 warnings |
| Typing | `make type-check` | 0 errors |
| Unit Tests | `make test-unit` | All pass |
| Integration Tests | `make test-integration` | All pass |
| Coverage | `make coverage` | 100% |
| Full Check | `make check` | All above pass |
