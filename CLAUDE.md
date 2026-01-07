# rl-emails Project Guide

## Overview

Email ML pipeline for analyzing Gmail exports and predicting email priority/actions.

**Status**: Production-ready with 100% type coverage and 100% test coverage.

---

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Install pre-commit hooks
make hooks

# Set up environment
cp .env.example .env
# Edit .env with your DATABASE_URL, MBOX_PATH, YOUR_EMAIL, OPENAI_API_KEY

# Start PostgreSQL
docker compose up -d

# Run pipeline
make run

# Check status
make status
```

---

## Development

```bash
# Setup
make install              # Install all dependencies
make hooks                # Install pre-commit hooks

# Quality checks
make lint                 # Run ruff linter
make type-check           # Run mypy strict
make test                 # Run all tests
make coverage             # Run tests with 100% coverage

# All checks at once
make check                # lint + type-check + test + coverage

# Formatting
make format               # Auto-fix lint and format issues

# Pipeline
make run                  # Run the full pipeline
make status               # Check pipeline status
make validate             # Validate data
```

---

## Project Structure

```
rl-emails/
├── src/rl_emails/        # Python package (100% typed, 100% tested)
│   ├── core/             # Core utilities
│   │   ├── db.py         # Database connection helpers
│   │   ├── config.py     # Configuration management
│   │   └── types.py      # TypedDict definitions
│   └── __init__.py
├── scripts/              # Pipeline scripts (15 files)
│   ├── onboard_data.py   # Main orchestrator
│   ├── parse_mbox.py     # Stage 1: Parse MBOX
│   ├── import_to_postgres.py  # Stage 2: Import to DB
│   ├── populate_threads.py    # Stage 3: Build threads
│   └── ...               # Stages 4-11
├── tests/                # Test suite
│   ├── unit/             # Unit tests (100% coverage)
│   └── integration/      # Integration tests
├── alembic/              # Database migrations
├── Makefile              # Development commands
├── pyproject.toml        # Project config
├── .pre-commit-config.yaml  # Quality gates
└── ralph-wiggum.md       # Production plan (completed)
```

---

## Quality Gates

All enforced by pre-commit hooks:

| Gate | Command | Status |
|------|---------|--------|
| Linting | `make lint` | ruff check + format |
| Types | `make type-check` | mypy --strict |
| Tests | `make test` | pytest |
| Coverage | `make coverage` | 100% required |

Run `make check` to verify all gates pass.

---

## Pipeline Stages (11 stages)

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | parse_mbox.py | Parse MBOX to JSONL |
| 2 | import_to_postgres.py | Import to PostgreSQL |
| 3 | populate_threads.py | Build thread relationships |
| 4 | enrich_emails_db.py | Compute action labels |
| 5 | compute_basic_features.py | Compute ML features |
| 6 | compute_embeddings.py | Generate embeddings |
| 7 | classify_ai_handleability.py | Rule-based classification |
| 8 | populate_users.py | User profiles |
| 9 | cluster_emails.py | Multi-dimensional clustering |
| 10 | compute_priority.py | Hybrid priority ranking |
| 11 | run_llm_classification.py | LLM classification |

---

## Environment Variables

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/gmail_test_30d
MBOX_PATH=/path/to/your/gmail.mbox
YOUR_EMAIL=your_email@example.com
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=ant-...  # Optional, for LLM classification
```

---

## Database Schema

Key tables populated by the pipeline:

| Table | Purpose |
|-------|---------|
| emails | Raw email data |
| threads | Conversation groupings |
| email_features | ML features |
| email_embeddings | OpenAI embeddings |
| email_ai_classification | Rule-based classification |
| email_llm_classification | LLM classification |
| users | User profiles |
| email_clusters | Clustering results |
| email_priority | Priority scores |
