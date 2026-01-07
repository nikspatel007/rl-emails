# rl-emails Project Guide

## Overview

Email ML pipeline for analyzing Gmail exports and predicting email priority/actions.

## Production Plan

**See `ralph-wiggum.md` for the iteration plan.**

The goal is to transform this codebase into a production-ready system with:
- Proper `src/rl_emails/` package structure
- 100% strict typing (`mypy --strict` passes)
- 100% test coverage (`pytest --cov-fail-under=100`)
- Pre-commit hooks enforcing all quality gates

**Current Status**: Check `ralph-wiggum.md` Progress Log for current iteration.

### How to Continue

1. Open `ralph-wiggum.md`
2. Find the current iteration (first one with PENDING status)
3. Complete all tasks in that iteration
4. Run the verification commands
5. Update the Progress Log
6. Commit and move to next iteration

---

## Current Structure

```
rl-emails/
├── scripts/           # Pipeline scripts (15 files)
├── tests/             # Test suite (to be expanded)
├── alembic/           # Database migrations
├── archive/           # Legacy code (to be deleted)
├── pyproject.toml     # Project config
├── ralph-wiggum.md    # Production iteration plan
└── CLAUDE.md          # This file
```

---

## Quick Start

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your DATABASE_URL, MBOX_PATH, YOUR_EMAIL, OPENAI_API_KEY

# Start PostgreSQL
docker compose up -d

# Run pipeline
uv run python scripts/onboard_data.py

# Check status
uv run python scripts/onboard_data.py --status
```

---

## Environment Variables

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/gmail_test_30d
MBOX_PATH=/path/to/your/gmail.mbox
YOUR_EMAIL=your_email@example.com
OPENAI_API_KEY=sk-...
```

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

### Utilities

| Script | Purpose |
|--------|---------|
| onboard_data.py | Main orchestrator |
| checkpoint.py | Checkpoint/restore |
| query_db.py | Database queries |
| validate_data.py | Data validation |

---

## Development (After Iteration 7)

Once the production plan is complete:

```bash
# Setup
make install              # Install dependencies
make hooks                # Install pre-commit hooks

# Development cycle
make format               # Auto-fix formatting
make check                # Run all checks (lint + type + test)

# Testing
make test                 # Run all tests
make coverage             # Run with coverage report

# Pipeline
make run                  # Run pipeline
make status               # Check status
```

---

## Database Schema

Key tables:

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
