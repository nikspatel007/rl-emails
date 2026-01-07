# Phase 1: Multi-Tenant Foundation

## Status: ✅ COMPLETE

**Completed**: 2026-01-07
**Tests**: 620 passing
**Coverage**: 100%

---

## Overview

Phase 1 establishes the multi-tenant data model that enables multiple organizations and users. This is the foundation for Gmail API integration (Phase 2).

**Duration**: 3 iterations
**Goal**: Enable org/user data model while maintaining backward compatibility with existing MBOX pipeline

---

## Iteration 1: Database Schema Updates

### Story
As a developer, I need multi-tenant database schema so that multiple organizations and users can be supported.

### Deliverables
1. Alembic migration for multi-tenant tables
2. Updated type definitions
3. Tests for migrations

### Schema Design

```sql
-- Core tenant tables
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE org_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    name TEXT,
    role TEXT DEFAULT 'member', -- 'admin', 'member'
    gmail_connected BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(org_id, email)
);

CREATE TABLE oauth_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES org_users(id) ON DELETE CASCADE,
    provider TEXT DEFAULT 'google',
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    scopes TEXT[],
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE sync_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES org_users(id) ON DELETE CASCADE,
    last_history_id TEXT,
    last_sync_at TIMESTAMPTZ,
    sync_status TEXT DEFAULT 'idle', -- 'idle', 'syncing', 'error'
    error_message TEXT,
    emails_synced INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes for multi-tenant queries
CREATE INDEX idx_org_users_org_id ON org_users(org_id);
CREATE INDEX idx_org_users_email ON org_users(email);
CREATE INDEX idx_oauth_tokens_user_id ON oauth_tokens(user_id);
CREATE INDEX idx_sync_state_user_id ON sync_state(user_id);
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `alembic/versions/xxx_add_multi_tenant_tables.py` | Create | Migration for new tables |
| `src/rl_emails/core/types.py` | Modify | Add Organization, OrgUser, OAuthToken, SyncState types |
| `tests/unit/core/test_types.py` | Modify | Add tests for new types |

### Acceptance Criteria

- [x] Alembic migration creates `organizations` table with correct columns
- [x] Alembic migration creates `org_users` table with foreign key to organizations
- [x] Alembic migration creates `oauth_tokens` table with foreign key to org_users
- [x] Alembic migration creates `sync_state` table with foreign key to org_users
- [x] All indexes are created for efficient querying
- [x] Migration is reversible (downgrade works)
- [x] TypedDict definitions exist for all new tables
- [x] 100% test coverage for new code

### Test Plan

```python
# tests/unit/test_migrations.py

class TestMultiTenantMigration:
    """Test multi-tenant migration."""

    def test_migration_creates_organizations_table(self, db_session):
        """Verify organizations table structure."""
        # Check table exists
        # Check columns: id, name, slug, settings, created_at, updated_at
        # Check slug unique constraint

    def test_migration_creates_org_users_table(self, db_session):
        """Verify org_users table structure."""
        # Check table exists
        # Check columns: id, org_id, email, name, role, gmail_connected, created_at, updated_at
        # Check foreign key to organizations
        # Check unique constraint on (org_id, email)

    def test_migration_creates_oauth_tokens_table(self, db_session):
        """Verify oauth_tokens table structure."""
        # Check table exists
        # Check columns: id, user_id, provider, access_token, refresh_token, expires_at, scopes
        # Check foreign key to org_users

    def test_migration_creates_sync_state_table(self, db_session):
        """Verify sync_state table structure."""
        # Check table exists
        # Check columns as specified
        # Check foreign key to org_users

    def test_migration_cascade_deletes(self, db_session):
        """Verify cascade delete from org → users → tokens."""
        # Create org, user, tokens
        # Delete org
        # Verify all related records deleted

    def test_migration_rollback(self, db_session):
        """Verify migration is reversible."""
        # Apply migration
        # Run downgrade
        # Verify tables removed
```

### Verification Steps

1. **Run migration**:
   ```bash
   alembic upgrade head
   ```

2. **Verify tables exist**:
   ```sql
   \dt organizations
   \dt org_users
   \dt oauth_tokens
   \dt sync_state
   ```

3. **Verify constraints**:
   ```sql
   \d organizations  -- Check columns
   \d org_users      -- Check FK, unique
   \d oauth_tokens   -- Check FK
   \d sync_state     -- Check FK
   ```

4. **Test rollback**:
   ```bash
   alembic downgrade -1
   # Verify tables removed
   alembic upgrade head
   # Verify tables restored
   ```

5. **Run tests**:
   ```bash
   make test
   make coverage  # Must be 100%
   ```

### Success Criteria

| Metric | Requirement |
|--------|-------------|
| Migration runs | ✓ No errors on fresh DB |
| Migration rollback | ✓ Downgrade succeeds |
| Type safety | ✓ mypy --strict passes |
| Test coverage | ✓ 100% on new code |
| Integration | ✓ Existing pipeline unaffected |

---

## Iteration 2: Core Model Updates

### Story
As a developer, I need Pydantic/SQLAlchemy models for multi-tenant entities so that I can work with them in a type-safe manner.

### Deliverables
1. SQLAlchemy ORM models for new tables
2. Pydantic schemas for validation
3. Repository pattern for data access
4. Config updates for user context

### Architecture

```
src/rl_emails/
├── models/                    # NEW: SQLAlchemy models
│   ├── __init__.py
│   ├── base.py               # Base model class
│   ├── organization.py       # Organization model
│   ├── org_user.py           # OrgUser model
│   ├── oauth_token.py        # OAuthToken model
│   └── sync_state.py         # SyncState model
├── schemas/                   # NEW: Pydantic schemas
│   ├── __init__.py
│   ├── organization.py       # Org create/update/response
│   ├── org_user.py           # User create/update/response
│   └── sync.py               # Sync state schemas
├── repositories/              # NEW: Data access layer
│   ├── __init__.py
│   ├── organization.py       # Org CRUD
│   ├── org_user.py           # User CRUD
│   └── sync_state.py         # Sync state CRUD
└── core/
    └── config.py             # MODIFY: Add user context
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/rl_emails/models/__init__.py` | Create | Barrel exports |
| `src/rl_emails/models/base.py` | Create | Base model with common fields |
| `src/rl_emails/models/organization.py` | Create | Organization ORM model |
| `src/rl_emails/models/org_user.py` | Create | OrgUser ORM model |
| `src/rl_emails/models/oauth_token.py` | Create | OAuthToken ORM model |
| `src/rl_emails/models/sync_state.py` | Create | SyncState ORM model |
| `src/rl_emails/schemas/__init__.py` | Create | Barrel exports |
| `src/rl_emails/schemas/organization.py` | Create | Org Pydantic schemas |
| `src/rl_emails/schemas/org_user.py` | Create | User Pydantic schemas |
| `src/rl_emails/schemas/sync.py` | Create | Sync Pydantic schemas |
| `src/rl_emails/repositories/__init__.py` | Create | Barrel exports |
| `src/rl_emails/repositories/organization.py` | Create | Org repository |
| `src/rl_emails/repositories/org_user.py` | Create | User repository |
| `src/rl_emails/repositories/sync_state.py` | Create | Sync repository |
| `src/rl_emails/core/config.py` | Modify | Add user_id, org_id optional fields |
| `tests/unit/models/` | Create | Model tests |
| `tests/unit/schemas/` | Create | Schema tests |
| `tests/unit/repositories/` | Create | Repository tests |

### Model Design

```python
# src/rl_emails/models/organization.py
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

from rl_emails.models.base import Base

class Organization(Base):
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False)
    settings = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default="now()")
    updated_at = Column(DateTime(timezone=True), server_default="now()")

    users = relationship("OrgUser", back_populates="organization", cascade="all, delete-orphan")
```

```python
# src/rl_emails/schemas/organization.py
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime

class OrganizationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=63, pattern=r"^[a-z0-9-]+$")
    settings: dict = Field(default_factory=dict)

class OrganizationResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    settings: dict
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
```

```python
# src/rl_emails/repositories/organization.py
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from rl_emails.models.organization import Organization
from rl_emails.schemas.organization import OrganizationCreate

class OrganizationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: OrganizationCreate) -> Organization:
        org = Organization(**data.model_dump())
        self.session.add(org)
        await self.session.commit()
        await self.session.refresh(org)
        return org

    async def get_by_id(self, org_id: UUID) -> Organization | None:
        result = await self.session.execute(
            select(Organization).where(Organization.id == org_id)
        )
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Organization | None:
        result = await self.session.execute(
            select(Organization).where(Organization.slug == slug)
        )
        return result.scalar_one_or_none()
```

### Config Updates

```python
# src/rl_emails/core/config.py (modifications)

@dataclass
class Config:
    # Existing fields...
    database_url: str
    mbox_path: Path | None
    your_email: str
    openai_api_key: str | None
    anthropic_api_key: str | None

    # NEW: Multi-tenant context (optional for backward compatibility)
    org_id: UUID | None = None
    user_id: UUID | None = None

    @property
    def is_multi_tenant(self) -> bool:
        """Check if running in multi-tenant mode."""
        return self.user_id is not None
```

### Acceptance Criteria

- [x] SQLAlchemy models defined for all 4 new tables
- [x] Pydantic schemas for create/update/response patterns
- [x] Repository classes with async CRUD methods
- [x] Config class updated with optional org_id, user_id
- [x] All models have proper relationships defined
- [x] All schemas validate input correctly
- [x] 100% test coverage on all new code
- [x] mypy --strict passes

### Test Plan

```python
# tests/unit/models/test_organization.py
class TestOrganizationModel:
    def test_organization_has_required_fields(self):
        """Test Organization model has all required fields."""

    def test_organization_slug_unique(self, db_session):
        """Test slug uniqueness constraint."""

    def test_organization_cascade_delete(self, db_session):
        """Test users deleted when org deleted."""

# tests/unit/schemas/test_organization.py
class TestOrganizationSchemas:
    def test_organization_create_validates_slug(self):
        """Test slug validation pattern."""

    def test_organization_create_requires_name(self):
        """Test name is required."""

    def test_organization_response_from_orm(self):
        """Test from_attributes conversion."""

# tests/unit/repositories/test_organization.py
class TestOrganizationRepository:
    async def test_create_organization(self, db_session):
        """Test organization creation."""

    async def test_get_by_id(self, db_session):
        """Test get by UUID."""

    async def test_get_by_slug(self, db_session):
        """Test get by slug."""

    async def test_get_nonexistent_returns_none(self, db_session):
        """Test missing org returns None."""
```

### Verification Steps

1. **Create test organization via repository**:
   ```python
   repo = OrganizationRepository(session)
   org = await repo.create(OrganizationCreate(name="Test", slug="test"))
   assert org.id is not None
   ```

2. **Verify relationships work**:
   ```python
   user = OrgUser(org_id=org.id, email="test@example.com")
   session.add(user)
   await session.commit()

   # Verify relationship
   await session.refresh(org)
   assert len(org.users) == 1
   ```

3. **Run type checks**:
   ```bash
   make type-check
   ```

4. **Run tests**:
   ```bash
   make test
   make coverage
   ```

### Success Criteria

| Metric | Requirement |
|--------|-------------|
| Models | ✓ All 4 models defined |
| Schemas | ✓ Create/Response patterns for all |
| Repositories | ✓ CRUD operations implemented |
| Type safety | ✓ mypy --strict passes |
| Test coverage | ✓ 100% on new code |
| Backward compat | ✓ Existing pipeline still works |

---

## Iteration 3: Pipeline Multi-Tenant Adaptation

### Story
As a developer, I need all pipeline stages to support user context so that emails can be processed per-user in a multi-tenant environment.

### Deliverables
1. All 11 pipeline stages accept optional user_id
2. All database queries scoped by user_id when provided
3. Orchestrator supports user context
4. CLI supports --user flag

### Architecture Changes

```python
# Before: Stages query all data
def run(config: Config) -> StageResult:
    emails = await get_all_emails()  # No filtering

# After: Stages filter by user_id
def run(config: Config) -> StageResult:
    if config.user_id:
        emails = await get_emails_for_user(config.user_id)
    else:
        emails = await get_all_emails()  # Backward compatible
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/rl_emails/pipeline/stages/stage_01_parse_mbox.py` | Add user_id to output |
| `src/rl_emails/pipeline/stages/stage_02_import_postgres.py` | Add user_id column |
| `src/rl_emails/pipeline/stages/stage_03_populate_threads.py` | Filter by user_id |
| `src/rl_emails/pipeline/stages/stage_04_enrich_emails.py` | Filter by user_id |
| `src/rl_emails/pipeline/stages/stage_05_compute_features.py` | Filter by user_id |
| `src/rl_emails/pipeline/stages/stage_06_compute_embeddings.py` | Filter by user_id |
| `src/rl_emails/pipeline/stages/stage_07_classify_handleability.py` | Filter by user_id |
| `src/rl_emails/pipeline/stages/stage_08_populate_users.py` | Filter by user_id |
| `src/rl_emails/pipeline/stages/stage_09_cluster_emails.py` | Filter by user_id |
| `src/rl_emails/pipeline/stages/stage_10_compute_priority.py` | Filter by user_id |
| `src/rl_emails/pipeline/stages/stage_11_llm_classification.py` | Filter by user_id |
| `src/rl_emails/pipeline/orchestrator.py` | Pass user_id to stages |
| `src/rl_emails/cli.py` | Add --user option |

### Query Scoping Pattern

```python
# src/rl_emails/pipeline/stages/stage_03_populate_threads.py

async def get_emails(conn: Connection, user_id: UUID | None) -> list[dict]:
    """Get emails, optionally filtered by user."""
    if user_id:
        query = """
            SELECT id, message_id, in_reply_to, subject, date_sent
            FROM emails
            WHERE user_id = $1
            ORDER BY date_sent
        """
        return await conn.fetch(query, user_id)
    else:
        # Backward compatible: no user filter
        query = """
            SELECT id, message_id, in_reply_to, subject, date_sent
            FROM emails
            ORDER BY date_sent
        """
        return await conn.fetch(query)


def run(config: Config, batch_size: int = 100) -> StageResult:
    """Run thread population with optional user scope."""
    async def _run() -> StageResult:
        async with get_connection(config.database_url) as conn:
            emails = await get_emails(conn, config.user_id)
            # ... rest of processing

    return asyncio.run(_run())
```

### CLI Updates

```python
# src/rl_emails/cli.py

@click.command()
@click.option("--user", type=click.UUID, help="User ID for multi-tenant mode")
@click.option("--org", type=click.UUID, help="Organization ID for multi-tenant mode")
@click.option("--status", is_flag=True, help="Show pipeline status")
# ... existing options ...
def main(
    user: UUID | None,
    org: UUID | None,
    status: bool,
    # ... existing params ...
) -> None:
    config = Config.from_env()

    # Apply multi-tenant context if provided
    if user:
        config = dataclasses.replace(config, user_id=user)
    if org:
        config = dataclasses.replace(config, org_id=org)

    # ... rest of CLI logic
```

### Acceptance Criteria

- [x] All 11 stages accept user_id from config
- [x] When user_id provided, all queries filtered
- [x] When user_id is None, backward compatible (no filtering)
- [x] Orchestrator passes user context through
- [x] CLI supports --user and --org flags
- [x] Existing tests still pass
- [x] New tests cover user-scoped behavior
- [x] 100% test coverage maintained

### Test Plan

```python
# tests/unit/pipeline/stages/test_stage_03_populate_threads.py

class TestPopulateThreadsMultiTenant:
    """Test thread population with user scoping."""

    async def test_queries_all_emails_without_user_id(self, db_session):
        """Without user_id, should query all emails."""
        config = Config(database_url="...", user_id=None)
        # Insert test emails
        # Run stage
        # Verify all emails processed

    async def test_queries_user_emails_with_user_id(self, db_session):
        """With user_id, should only query that user's emails."""
        user1_id = uuid.uuid4()
        user2_id = uuid.uuid4()
        # Insert emails for both users

        config = Config(database_url="...", user_id=user1_id)
        # Run stage
        # Verify only user1's emails processed

    async def test_does_not_affect_other_users(self, db_session):
        """User scoped processing doesn't modify other users' data."""
        # Process user1's emails
        # Verify user2's data unchanged
```

```python
# tests/unit/test_cli.py

class TestCLIMultiTenant:
    """Test CLI multi-tenant options."""

    def test_cli_accepts_user_flag(self):
        """Test --user flag is parsed."""

    def test_cli_accepts_org_flag(self):
        """Test --org flag is parsed."""

    def test_config_has_user_id_when_provided(self):
        """Test user_id set on config."""
```

### Verification Steps

1. **Test backward compatibility**:
   ```bash
   # Without --user, should work as before
   rl-emails --status
   rl-emails  # Full pipeline
   ```

2. **Test user-scoped mode**:
   ```bash
   # Create test user first
   # Then run with --user
   rl-emails --user <uuid> --status
   ```

3. **Verify SQL queries include user filter**:
   ```sql
   -- Check query logs or add EXPLAIN
   EXPLAIN SELECT * FROM emails WHERE user_id = '...'
   ```

4. **Run full test suite**:
   ```bash
   make check  # lint + type-check + test + coverage
   ```

### Success Criteria

| Metric | Requirement |
|--------|-------------|
| Backward compat | ✓ Pipeline works without --user |
| User scoping | ✓ All stages filter by user_id |
| CLI | ✓ --user and --org flags work |
| Type safety | ✓ mypy --strict passes |
| Test coverage | ✓ 100% maintained |
| Isolation | ✓ Users cannot see each other's data |

---

## Phase 1 Completion Checklist

### Pre-Implementation
- [x] Read and understand existing codebase
- [x] Review gmail_twoyrs schema for reference
- [x] Ensure dev environment set up

### Iteration 1: Database Schema ✅
- [x] Write Alembic migration (`alembic/versions/20260107144608_add_multi_tenant_tables.py`)
- [x] Test migration forward/backward
- [x] Add TypedDict definitions
- [x] Write unit tests
- [x] Verify 100% coverage

### Iteration 2: Core Models ✅
- [x] Create SQLAlchemy models (`src/rl_emails/models/`)
- [x] Create Pydantic schemas (`src/rl_emails/schemas/`)
- [x] Create repository classes (`src/rl_emails/repositories/`)
- [x] Update Config class with `with_user()` method
- [x] Write unit tests for all
- [x] Verify 100% coverage

### Iteration 3: Pipeline Multi-Tenant ✅
- [x] Update Config with `is_multi_tenant` property
- [x] Update orchestrator to pass user context
- [x] Update CLI with `--user` and `--org` flags
- [x] Test backward compatibility
- [x] Test user scoping
- [x] Verify 100% coverage (620 tests)

### Post-Implementation
- [x] Run `make check` (all pass)
- [x] Update CLAUDE.md with progress
- [ ] Create PR if applicable

---

## Technical Notes

### Backward Compatibility Strategy

The multi-tenant changes are **additive**. Key principles:

1. **Optional user_id**: All user_id fields are nullable
2. **Conditional filtering**: Stages only filter when user_id provided
3. **Default behavior**: Without --user flag, behaves exactly as before
4. **No breaking changes**: Existing .env files continue to work

### Database Migration Safety

1. **New tables only**: Iteration 1 creates new tables, doesn't modify existing
2. **Column additions**: user_id on emails added as nullable
3. **No data migration**: Existing emails remain without user_id (single-tenant mode)
4. **Index strategy**: Indexes created for performance, not blocking

### Testing Strategy

1. **Unit tests**: Each component tested in isolation
2. **Integration tests**: End-to-end pipeline tests
3. **Backward compat tests**: Verify existing behavior unchanged
4. **Multi-tenant tests**: Verify user isolation

### Performance Considerations

1. **Indexed queries**: user_id columns indexed for performance
2. **Partial indexes**: Consider for user-specific hot paths
3. **Query plans**: Monitor with EXPLAIN for large datasets
