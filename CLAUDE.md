# rl-emails Project Guide

## Overview

Email ML pipeline for analyzing Gmail exports and predicting email priority/actions.

**Status**: Production-ready with 100% type coverage and 100% test coverage.

**Current Phase**: Phase 2 - Gmail API Integration (IN PROGRESS)

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
│   ├── pipeline/         # Pipeline stages and orchestration
│   │   ├── orchestrator.py    # Main pipeline orchestrator
│   │   ├── status.py          # Status utilities
│   │   └── stages/            # All 11 pipeline stages
│   ├── cli.py            # CLI entry point
│   └── __init__.py
├── tests/                # Test suite
│   ├── unit/             # Unit tests (100% coverage)
│   └── integration/      # Integration tests
├── alembic/              # Database migrations
├── Makefile              # Development commands
├── pyproject.toml        # Project config
└── .pre-commit-config.yaml  # Quality gates
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

| Stage | Module | Purpose |
|-------|--------|---------|
| 1 | stage_01_parse_mbox | Parse MBOX to JSONL |
| 2 | stage_02_import_postgres | Import to PostgreSQL |
| 3 | stage_03_populate_threads | Build thread relationships |
| 4 | stage_04_enrich_emails | Compute action labels |
| 5 | stage_05_compute_features | Compute ML features |
| 6 | stage_06_compute_embeddings | Generate embeddings |
| 7 | stage_07_classify_handleability | Rule-based classification |
| 8 | stage_08_populate_users | User profiles |
| 9 | stage_09_cluster_emails | Multi-dimensional clustering |
| 10 | stage_10_compute_priority | Hybrid priority ranking |
| 11 | stage_11_llm_classification | LLM classification |

All stages are located in `src/rl_emails/pipeline/stages/` and follow the pattern:
```python
def run(config: Config, **kwargs) -> StageResult:
    """Execute the stage with config passed in."""
```

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

### Multi-Tenant Tables (Phase 1)

| Table | Purpose |
|-------|---------|
| organizations | Tenant organizations |
| org_users | Users within organizations |
| oauth_tokens | Gmail OAuth2 tokens |
| sync_state | Gmail sync state per user |
| cluster_metadata | Cluster labels and project detection |

---

## Phase 1 Progress

### Architecture Plan
See `docs/ARCHITECTURE_PLAN.md` for full 16-iteration roadmap.

### Iteration Status

| Iter | Name | Status | Notes |
|------|------|--------|-------|
| 1 | Database Schema | ✅ Complete | Multi-tenant tables + migrations |
| 2 | Core Models | ✅ Complete | SQLAlchemy models, Pydantic schemas, repositories |
| 3 | Pipeline Multi-Tenant | ✅ Complete | Config.with_user(), CLI --user/--org flags |

### Iteration 1 Deliverables
- [x] Alembic migration: `20260107144608_add_multi_tenant_tables.py`
- [x] TypedDict definitions for multi-tenant entities
- [x] Tests for new types
- [x] 100% test coverage maintained

### Iteration 2 Deliverables
- [x] SQLAlchemy models: Organization, OrgUser, OAuthToken, SyncState
- [x] Pydantic schemas for create/update/response patterns
- [x] Repository classes with async CRUD operations
- [x] 100% test coverage (610 tests)

### Iteration 3 Deliverables
- [x] Config class with `is_multi_tenant` property and `with_user()` method
- [x] CLI flags: `--user UUID` and `--org UUID` for multi-tenant mode
- [x] UUID validation and error handling
- [x] 100% test coverage (620 tests)

### Documentation
- `docs/ARCHITECTURE_PLAN.md` - Full 16-iteration architecture plan
- `docs/PHASE1_ITERATIONS.md` - Phase 1 detailed iteration specs
- `docs/GMAIL_API_SETUP.md` - Gmail API OAuth setup guide

---

## Phase 2 Progress

### Goal
Add Gmail API as alternative data source alongside existing MBOX pipeline.

### Iteration Status

| Iter | Name | Status | Notes |
|------|------|--------|-------|
| 1 | Auth Module Setup | ✅ Complete | OAuth types, GoogleTokens, OAuthError |
| 2 | GoogleOAuth Implementation | ✅ Complete | Authorization URL, code exchange, token refresh |
| 3 | OAuthToken Repository | ✅ Complete | CRUD operations, Pydantic schemas |
| 4 | AuthService Implementation | ✅ Complete | OAuth flow orchestration, token refresh |
| 5 | CLI Auth Commands | Pending | connect, status, disconnect commands |
| 6 | Gmail Integration Module | Pending | Models, rate limiter |
| 7 | GmailClient Implementation | Pending | list, get, batch operations |
| 8 | Gmail Parser | Pending | Gmail to EmailData conversion |
| 9 | SyncService + Stage 00 | Pending | Initial sync implementation |
| 10 | Incremental Sync | Pending | History API, delta processing |

### Iteration 1 Deliverables
- [x] `src/rl_emails/auth/` module structure
- [x] `GoogleTokens` dataclass with `is_expired()` and `expires_in_seconds()`
- [x] `OAuthError` exception class
- [x] Dependencies: google-auth, google-auth-oauthlib, httpx
- [x] Tests for OAuth types (10 tests)

### Iteration 2 Deliverables
- [x] `GoogleOAuth` class with authorization URL generation
- [x] Code exchange for access/refresh tokens
- [x] Token refresh functionality
- [x] Token revocation support
- [x] Tests for GoogleOAuth (19 tests)

### Iteration 3 Deliverables
- [x] `OAuthTokenRepository` with async CRUD operations
- [x] Pydantic schemas: OAuthTokenCreate, OAuthTokenUpdate, OAuthTokenResponse, OAuthTokenStatus
- [x] Repository and schema exports updated
- [x] Tests for repository and schemas (25 tests)
- [x] 100% test coverage maintained (682 tests)

### Iteration 4 Deliverables
- [x] `src/rl_emails/services/` module structure
- [x] `AuthService` class for OAuth flow orchestration
- [x] `start_auth_flow()`, `complete_auth_flow()`, `get_valid_token()`, `revoke_token()` methods
- [x] Auto-refresh tokens before expiry (5-minute buffer)
- [x] `TokenNotFoundError` and `AuthServiceError` exceptions
- [x] Tests for AuthService (25 tests)
- [x] 100% test coverage maintained (707 tests)

### Documentation
- `docs/PHASE2_ITERATIONS.md` - Phase 2 detailed iteration specs
- `ralph-wiggum-phase2.md` - Phase 2 implementation plan
