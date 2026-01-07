# Ralph Wiggum Production Plan

**Goal**: Convert the onboarding pipeline into a typed, 100% tested, strict production codebase.

**Status**: IN_PROGRESS
**Current Iteration**: 2

---

## Overview

The onboarding pipeline (`scripts/`) is the core working code. The `src/` folder contains experimental feature extraction and training code that is NOT used by the pipeline.

### What Works (Keep)
- `scripts/` - 15 files: The complete 11-stage onboarding pipeline + 4 utilities
- `alembic/` - Database migrations
- `tests/` - Existing tests (need expansion for scripts/)
- `pyproject.toml` - Project configuration

### What to Clean Up
- `src/` - 30+ files of unused experimental code (no scripts import it)
- `apps/` - Legacy UI code
- `archive/` - Already archived legacy scripts
- Root files - Various unused configs

---

## Iteration Plan

### Iteration 1: Audit & Clean ✅ COMPLETE
**Success Criteria**:
- [x] `src/` moved to `archive/experimental/`
- [x] `apps/` moved to `archive/ui/`
- [x] Root folder cleaned (only essential files)
- [x] `git status` shows clean working tree after commit

**Tasks**:
1. Move `src/` to `archive/experimental/`
2. Move `apps/` to `archive/ui/`
3. Remove unused root files (keep README.md, pyproject.toml, CLAUDE.md, etc.)
4. Update `.gitignore` if needed
5. Commit changes

---

### Iteration 2: Add Type Hints
**Success Criteria**:
- [ ] All 15 scripts have type hints on all functions
- [ ] `mypy scripts/` passes with no errors
- [ ] `pyproject.toml` has mypy configuration

**Tasks**:
1. Add `mypy` to dev dependencies
2. Configure mypy in `pyproject.toml` (strict mode)
3. Add type hints to each script (in order):
   - `parse_mbox.py`
   - `import_to_postgres.py`
   - `populate_threads.py`
   - `enrich_emails_db.py`
   - `compute_basic_features.py`
   - `compute_embeddings.py`
   - `classify_ai_handleability.py`
   - `populate_users.py`
   - `cluster_emails.py`
   - `compute_priority.py`
   - `run_llm_classification.py`
   - `onboard_data.py`
   - `checkpoint.py`
   - `query_db.py`
   - `validate_data.py`
4. Run `mypy scripts/` and fix all errors
5. Commit changes

---

### Iteration 3: Add Tests for Scripts
**Success Criteria**:
- [ ] `tests/scripts/` directory created
- [ ] Each pipeline script has corresponding test file
- [ ] `pytest tests/scripts/ -v` passes
- [ ] Coverage >= 80% for scripts/

**Tasks**:
1. Create `tests/scripts/` directory
2. Write tests for each script:
   - `test_parse_mbox.py`
   - `test_import_to_postgres.py`
   - `test_populate_threads.py`
   - `test_enrich_emails_db.py`
   - `test_compute_basic_features.py`
   - `test_compute_embeddings.py`
   - `test_classify_ai_handleability.py`
   - `test_populate_users.py`
   - `test_cluster_emails.py`
   - `test_compute_priority.py`
   - `test_run_llm_classification.py`
   - `test_onboard_data.py`
   - `test_checkpoint.py`
   - `test_validate_data.py`
3. Add pytest-cov to dev dependencies
4. Run `pytest --cov=scripts tests/scripts/`
5. Commit changes

---

### Iteration 4: Create Makefile
**Success Criteria**:
- [ ] `Makefile` exists with standard targets
- [ ] `make lint` runs mypy + ruff
- [ ] `make test` runs pytest with coverage
- [ ] `make run` runs the onboarding pipeline
- [ ] `make clean` cleans generated files

**Tasks**:
1. Create `Makefile` with targets:
   ```makefile
   .PHONY: install lint test run clean

   install:
       uv sync

   lint:
       uv run mypy scripts/
       uv run ruff check scripts/

   test:
       uv run pytest tests/ -v --cov=scripts --cov-report=term-missing

   run:
       uv run python scripts/onboard_data.py

   clean:
       rm -rf .mypy_cache .pytest_cache .coverage
       find . -name "*.pyc" -delete
   ```
2. Add `ruff` to dev dependencies
3. Configure ruff in `pyproject.toml`
4. Verify all make targets work
5. Commit changes

---

### Iteration 5: Update CLAUDE.md
**Success Criteria**:
- [ ] `CLAUDE.md` documents the production codebase
- [ ] Quick start guide for new developers
- [ ] Pipeline architecture documented
- [ ] Testing instructions included

**Tasks**:
1. Update `CLAUDE.md` with:
   - Project overview
   - Pipeline architecture (11 stages)
   - Development workflow (make targets)
   - Testing instructions
   - Database setup
   - Environment variables
2. Remove outdated sections
3. Commit changes

---

### Iteration 6: Achieve 100% Coverage
**Success Criteria**:
- [ ] `make test` shows 100% coverage for all scripts
- [ ] All edge cases tested
- [ ] CI-ready test suite

**Tasks**:
1. Review coverage report for gaps
2. Add missing tests for uncovered lines
3. Add edge case tests
4. Add integration tests for full pipeline
5. Verify `make test` passes with 100% coverage
6. Commit changes

---

### Iteration 7: Final Cleanup
**Success Criteria**:
- [ ] No unused files in repository
- [ ] All code passes lint + type checks
- [ ] 100% test coverage
- [ ] Documentation complete
- [ ] Ready for production use

**Tasks**:
1. Review `archive/` - delete if not needed
2. Clean up any remaining unused files
3. Final lint pass
4. Final test pass
5. Update README.md
6. Create release tag
7. Commit changes

---

## Commands Reference

```bash
# Install dependencies
uv sync

# Run linting
make lint

# Run tests
make test

# Run pipeline
make run

# Clean
make clean
```

---

## Progress Log

### Iteration 1 - 2026-01-07
- Status: ✅ COMPLETE
- Commit: c6f8dc9
- Notes: Removed 69 files (33,745 lines). src/, apps/, old tests moved to archive/. Repository now contains only production pipeline code.

### Iteration 2 - [IN PROGRESS]
- Status: PENDING
- Notes:

(Continue for each iteration...)
