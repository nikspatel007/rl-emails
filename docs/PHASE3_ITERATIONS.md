# Phase 3: FastAPI + Clerk Authentication & Real-Time Sync

## Status: ✅ COMPLETE

**Prerequisite**: Phase 2 Complete (Gmail API Integration)
**Iterations**: 6 (0-5) + Enhanced Clustering
**Goal**: Production-ready API with Clerk auth, multi-provider email connections, real-time sync, and projects/tasks extraction

### Progress Summary
| Iteration | Status |
|-----------|--------|
| 0. Database Models | ✅ Complete |
| 1. FastAPI Foundation | ✅ Complete |
| 2. Clerk Authentication | ✅ Complete |
| 3. Provider Interface | ✅ Complete |
| 4. Real-Time Sync | ✅ Complete |
| 5. Projects & Tasks API | ✅ Complete |
| Enhanced Clustering | ✅ Complete |

> **Note**: Pipeline Worker, Scheduler, and Integration & Docs have been moved to [FUTURE_PHASES.md](./FUTURE_PHASES.md)

---

## Overview

Phase 3 transforms the CLI-based pipeline into a production API service with:
- **Database Models**: Projects, Tasks, and Priority Context tables populated from pipeline data
- **Clerk Authentication**: Secure user auth with JWT validation
- **Multi-Provider Email**: Gmail now, extensible to Outlook, IMAP later
- **Real-Time Sync**: Webhooks + background workers for fresh data
- **Project/Task Management**: Surface actionable items from emails

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 3 ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ Mobile App   │    │   Web App    │    │  Webhooks    │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             │                                           │
│                    ┌────────▼────────┐                                  │
│                    │   FastAPI       │◄─── Clerk JWT Validation         │
│                    │   Gateway       │                                  │
│                    └────────┬────────┘                                  │
│                             │                                           │
│         ┌───────────────────┼───────────────────┐                       │
│         │                   │                   │                       │
│  ┌──────▼──────┐    ┌───────▼──────┐    ┌──────▼──────┐               │
│  │ Auth/Users  │    │ Email Sync   │    │  Projects   │               │
│  │ Service     │    │ Service      │    │  & Tasks    │               │
│  └─────────────┘    └──────┬───────┘    └─────────────┘               │
│                            │                                           │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    Background Workers                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │  │
│  │  │ Sync Worker │  │ ML Pipeline │  │ Task/Project│             │  │
│  │  │ (Webhooks)  │  │   Worker    │  │  Extractor  │             │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘             │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                      PostgreSQL                                  │  │
│  │  emails | projects | tasks | priority_context | connections     │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Design Principles

1. **Verifiable**: Every feature has clear acceptance criteria and verification steps
2. **Testable**: Unit tests + integration tests with mocked external services
3. **Reliable**: Graceful error handling, retries, circuit breakers
4. **Performant**: Async everywhere, connection pooling, caching where appropriate
5. **Security-driven**: Clerk JWT validation, input sanitization, rate limiting
6. **Well-tested**: 100% coverage on business logic, integration tests for API endpoints

---

## Existing Pydantic Models (to inform database design)

From `models/` directory:

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `Project` | Named project/deal entity | name, code, project_type, owner_email, participants, due_date, email_count |
| `ProjectMention` | Project reference in email | text, pattern_type, confidence, context |
| `ProjectFeatures` | Project-related ML features | project_count, has_deadline, urgency_score, action_score |
| `Task` | Actionable task from email | description, task_type, complexity, deadline, assignee, urgency_score |
| `TaskFeatures` | Task-related ML features | has_deadline, deadline_urgency, is_assigned_to_user, action_items |
| `PriorityContext` | Combined priority context | sender, thread, temporal, user contexts + component scores |
| `SenderContext` | Sender relationship info | email, frequency, importance, reply_rate, org_level |
| `ThreadContext` | Thread structure info | thread_length, thread_depth, user_already_replied |
| `TemporalContext` | Time-based features | is_business_hours, age_hours, day_of_week |
| `UserContext` | User profile/preferences | frequent_contacts, priority_senders, typical_daily_volume |

---

## Iteration 0: Database Models & Population

### Story
As a system, I need database tables for projects, tasks, and priority context so that the pipeline can store extracted information for the API to serve.

### Deliverables
1. SQLAlchemy models for Project, Task, PriorityContext
2. Alembic migration for new tables
3. Population service to extract data from existing pipeline output
4. Integration with LLM classification results
5. Pipeline stage to run extraction

### Architecture

```
src/rl_emails/
├── models/                       # MODIFY: Add SQLAlchemy models
│   ├── project.py               # NEW: SQLAlchemy Project model
│   ├── task.py                  # NEW: SQLAlchemy Task model
│   └── priority_context.py      # NEW: SQLAlchemy PriorityContext model
├── services/
│   ├── project_extractor.py     # NEW: Extract projects from emails
│   └── task_extractor.py        # NEW: Extract tasks from LLM results
├── pipeline/stages/
│   └── stage_12_extract_entities.py  # NEW: Entity extraction stage
└── repositories/
    ├── project.py               # NEW: Project CRUD
    └── task.py                  # NEW: Task CRUD
```

### Database Schema

```sql
-- Projects table
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES org_users(id),
    name VARCHAR(200) NOT NULL,
    code VARCHAR(50),
    project_type VARCHAR(50),  -- 'project', 'deal', 'operation', 'proposal'

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    priority INTEGER CHECK (priority >= 0 AND priority <= 4),

    -- Participants
    owner_email VARCHAR(255),
    participants TEXT[],

    -- Timing
    start_date TIMESTAMPTZ,
    due_date TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Metrics (computed)
    email_count INTEGER DEFAULT 0,
    last_activity TIMESTAMPTZ,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ,

    -- Detection source
    detected_from VARCHAR(50),  -- 'cluster', 'subject_pattern', 'llm', 'manual'
    cluster_id INTEGER REFERENCES email_clusters(id),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1)
);

-- Project-Email association
CREATE TABLE project_emails (
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    email_id INTEGER REFERENCES emails(id) ON DELETE CASCADE,
    mention_text TEXT,
    mention_confidence FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (project_id, email_id)
);

-- Tasks table
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES org_users(id),
    email_id INTEGER REFERENCES emails(id),
    project_id UUID REFERENCES projects(id),

    -- Task details
    description TEXT NOT NULL,
    task_type VARCHAR(50),  -- 'review', 'send', 'schedule', 'decision', etc.
    complexity VARCHAR(20),  -- 'trivial', 'quick', 'medium', 'substantial'

    -- Deadline
    deadline TIMESTAMPTZ,
    deadline_text VARCHAR(200),
    deadline_type VARCHAR(50),  -- 'explicit_date', 'relative_day', 'urgency_keyword'
    urgency_score FLOAT CHECK (urgency_score >= 0 AND urgency_score <= 1),

    -- Assignment
    assigned_to VARCHAR(255),
    assigned_by VARCHAR(255),
    is_assigned_to_user BOOLEAN DEFAULT FALSE,
    assignment_confidence FLOAT,

    -- Source
    source_text TEXT,
    extraction_method VARCHAR(50),  -- 'llm', 'pattern', 'manual'

    -- Status
    status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'in_progress', 'completed', 'dismissed'
    completed_at TIMESTAMPTZ,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

-- Priority context (per email, computed)
CREATE TABLE email_priority_context (
    id SERIAL PRIMARY KEY,
    email_id INTEGER UNIQUE NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES org_users(id),

    -- Sender context
    sender_email VARCHAR(255),
    sender_frequency FLOAT,
    sender_importance FLOAT,
    sender_reply_rate FLOAT,
    sender_org_level INTEGER,
    sender_relationship_strength FLOAT,

    -- Thread context
    thread_id VARCHAR(255),
    is_reply BOOLEAN,
    thread_length INTEGER,
    thread_depth INTEGER,
    user_already_replied BOOLEAN,

    -- Temporal context
    email_timestamp TIMESTAMPTZ,
    is_business_hours BOOLEAN,
    age_hours FLOAT,

    -- Component scores (0-1)
    people_score FLOAT,
    project_score FLOAT,
    topic_score FLOAT,
    task_score FLOAT,
    temporal_score FLOAT,
    relationship_score FLOAT,

    -- Overall
    overall_priority FLOAT,

    -- Metadata
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_projects_user_id ON projects(user_id);
CREATE INDEX idx_projects_is_active ON projects(is_active);
CREATE INDEX idx_tasks_user_id ON tasks(user_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_deadline ON tasks(deadline);
CREATE INDEX idx_email_priority_context_user ON email_priority_context(user_id);
CREATE INDEX idx_email_priority_context_priority ON email_priority_context(overall_priority DESC);
```

### Implementation Design

```python
# src/rl_emails/models/project.py (SQLAlchemy)
from sqlalchemy import Column, String, Boolean, Integer, Float, ForeignKey, ARRAY
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.orm import relationship
import uuid

from rl_emails.models.base import Base


class Project(Base):
    """Project entity extracted from email patterns."""
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("org_users.id"), nullable=False)
    name = Column(String(200), nullable=False)
    code = Column(String(50))
    project_type = Column(String(50))

    is_active = Column(Boolean, default=True)
    priority = Column(Integer)

    owner_email = Column(String(255))
    participants = Column(ARRAY(String))

    start_date = Column(TIMESTAMP(timezone=True))
    due_date = Column(TIMESTAMP(timezone=True))
    completed_at = Column(TIMESTAMP(timezone=True))

    email_count = Column(Integer, default=0)
    last_activity = Column(TIMESTAMP(timezone=True))

    detected_from = Column(String(50))
    cluster_id = Column(Integer, ForeignKey("email_clusters.id"))
    confidence = Column(Float)

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())

    # Relationships
    emails = relationship("Email", secondary="project_emails", back_populates="projects")
    tasks = relationship("Task", back_populates="project")
```

```python
# src/rl_emails/services/project_extractor.py
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.repositories.project import ProjectRepository
from rl_emails.repositories.email_cluster import EmailClusterRepository


class ProjectExtractor:
    """Extract projects from email clusters and patterns."""

    def __init__(
        self,
        session: AsyncSession,
        project_repo: ProjectRepository,
        cluster_repo: EmailClusterRepository,
    ) -> None:
        self._session = session
        self._project_repo = project_repo
        self._cluster_repo = cluster_repo

    async def extract_from_clusters(self, user_id: UUID) -> list[Project]:
        """Extract projects from email clusters.

        Uses cluster metadata to identify project-like groupings:
        - High email count clusters
        - Clusters with common subject patterns
        - Clusters with multiple participants
        """
        # Get clusters that look like projects
        clusters = await self._cluster_repo.get_project_candidates(
            user_id=user_id,
            min_emails=5,
            min_participants=2,
        )

        projects = []
        for cluster in clusters:
            # Check if project already exists for this cluster
            existing = await self._project_repo.get_by_cluster_id(cluster.id)
            if existing:
                continue

            # Extract project info from cluster
            project = Project(
                user_id=user_id,
                name=self._extract_project_name(cluster),
                project_type=self._infer_project_type(cluster),
                owner_email=cluster.primary_sender,
                participants=cluster.participants,
                email_count=cluster.email_count,
                last_activity=cluster.last_email_date,
                detected_from="cluster",
                cluster_id=cluster.id,
                confidence=cluster.coherence_score,
            )

            created = await self._project_repo.create(project)

            # Link emails to project
            await self._link_cluster_emails(created.id, cluster.id)

            projects.append(created)

        return projects

    async def extract_from_llm_results(self, user_id: UUID) -> list[Project]:
        """Extract projects mentioned in LLM classification results."""
        # Get LLM classifications with project mentions
        classifications = await self._get_classifications_with_projects(user_id)

        projects = []
        for classification in classifications:
            if not classification.project_name:
                continue

            # Find or create project
            project = await self._project_repo.get_by_name(
                user_id=user_id,
                name=classification.project_name,
            )

            if not project:
                project = Project(
                    user_id=user_id,
                    name=classification.project_name,
                    detected_from="llm",
                    confidence=0.8,
                )
                project = await self._project_repo.create(project)
                projects.append(project)

            # Link email to project
            await self._link_email_to_project(
                project_id=project.id,
                email_id=classification.email_id,
            )

        return projects

    def _extract_project_name(self, cluster: EmailCluster) -> str:
        """Extract project name from cluster metadata."""
        # Use common subject pattern
        if cluster.common_subject_pattern:
            # Remove RE:, FW:, etc.
            name = re.sub(r'^(RE:|FW:|FWD:)\s*', '', cluster.common_subject_pattern, flags=re.I)
            return name.strip()[:200]

        # Use primary sender domain as fallback
        if cluster.primary_sender:
            domain = cluster.primary_sender.split('@')[-1]
            return f"Project: {domain}"

        return f"Project #{cluster.id}"

    def _infer_project_type(self, cluster: EmailCluster) -> str:
        """Infer project type from cluster characteristics."""
        subject_lower = (cluster.common_subject_pattern or "").lower()

        if any(word in subject_lower for word in ["deal", "proposal", "quote"]):
            return "deal"
        if any(word in subject_lower for word in ["contract", "agreement"]):
            return "contract"
        if any(word in subject_lower for word in ["meeting", "call", "sync"]):
            return "operation"

        return "project"
```

```python
# src/rl_emails/services/task_extractor.py
from uuid import UUID

from rl_emails.repositories.task import TaskRepository


class TaskExtractor:
    """Extract tasks from LLM classification results."""

    async def extract_from_llm(self, user_id: UUID) -> list[Task]:
        """Extract tasks from LLM classification action items."""
        # Get classifications with action_required = True
        classifications = await self._get_actionable_classifications(user_id)

        tasks = []
        for classification in classifications:
            # Check if task already exists for this email
            existing = await self._task_repo.get_by_email_id(classification.email_id)
            if existing:
                continue

            # Create main task from classification
            if classification.action_summary:
                task = Task(
                    user_id=user_id,
                    email_id=classification.email_id,
                    description=classification.action_summary,
                    task_type=self._infer_task_type(classification),
                    complexity=self._infer_complexity(classification),
                    deadline=classification.suggested_deadline,
                    deadline_text=classification.deadline_text,
                    urgency_score=classification.urgency_score or 0.5,
                    is_assigned_to_user=True,
                    extraction_method="llm",
                )
                created = await self._task_repo.create(task)
                tasks.append(created)

            # Create sub-tasks from action items
            for action_item in classification.action_items or []:
                sub_task = Task(
                    user_id=user_id,
                    email_id=classification.email_id,
                    description=action_item,
                    task_type="other",
                    complexity="medium",
                    urgency_score=0.5,
                    extraction_method="llm",
                )
                created = await self._task_repo.create(sub_task)
                tasks.append(created)

        return tasks

    def _infer_task_type(self, classification: LLMClassification) -> str:
        """Infer task type from classification."""
        summary_lower = (classification.action_summary or "").lower()

        if any(word in summary_lower for word in ["review", "check", "approve"]):
            return "review"
        if any(word in summary_lower for word in ["send", "share", "forward"]):
            return "send"
        if any(word in summary_lower for word in ["schedule", "meeting", "call"]):
            return "schedule"
        if any(word in summary_lower for word in ["decide", "decision", "choose"]):
            return "decision"
        if any(word in summary_lower for word in ["follow up", "followup", "remind"]):
            return "follow_up"

        return "other"
```

```python
# src/rl_emails/pipeline/stages/stage_12_extract_entities.py
"""Stage 12: Extract projects and tasks from pipeline results."""

from rl_emails.core.config import Config
from rl_emails.pipeline.stages.base import StageResult


def run(config: Config) -> StageResult:
    """Extract projects and tasks from emails.

    Uses:
    - Cluster metadata to detect projects
    - LLM classification to extract action items
    - Email features for priority context
    """
    import asyncio
    import time

    async def _run() -> StageResult:
        start = time.time()

        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker

        from rl_emails.services.project_extractor import ProjectExtractor
        from rl_emails.services.task_extractor import TaskExtractor
        from rl_emails.services.priority_context_builder import PriorityContextBuilder

        engine = create_async_engine(config.database_url.replace("postgresql://", "postgresql+asyncpg://"))
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session:
            # Extract projects
            project_extractor = ProjectExtractor(session)
            projects = await project_extractor.extract_from_clusters(config.user_id)
            projects += await project_extractor.extract_from_llm_results(config.user_id)

            # Extract tasks
            task_extractor = TaskExtractor(session)
            tasks = await task_extractor.extract_from_llm(config.user_id)

            # Build priority context
            context_builder = PriorityContextBuilder(session)
            contexts = await context_builder.build_all(config.user_id)

            await session.commit()

        duration = time.time() - start

        return StageResult(
            success=True,
            records_processed=len(projects) + len(tasks) + len(contexts),
            duration_seconds=duration,
            message=f"Extracted {len(projects)} projects, {len(tasks)} tasks, {len(contexts)} priority contexts",
        )

    return asyncio.run(_run())
```

### Acceptance Criteria

- [x] SQLAlchemy models for Project, Task, EmailPriorityContext (columns added to existing tables)
- [x] Alembic migration creates all tables with indexes (`fb71ba5949b0`)
- [x] ProjectExtractor detects projects from clusters (`extract_from_content_clusters`)
- [x] ProjectExtractor extracts projects from real person threads (`extract_from_real_person_threads`)
- [x] TaskExtractor extracts tasks from LLM action items (`extract_from_llm_classifications`)
- [x] TaskExtractor extracts tasks from AI classifications (`extract_from_ai_classifications`)
- [x] PriorityContextBuilder computes context for all emails (`build_contexts`)
- [x] Stage 12 runs after LLM classification (`stage_12_entity_extraction.py`)
- [x] Projects linked to clusters via `cluster_id` column
- [x] Task status column with default 'pending'
- [x] Marketing email filtering (MARKETING_PATTERNS, real person detection)
- [x] 100% test coverage on new code (1,178 tests)
- [x] mypy --strict passes

### Test Plan

```python
# tests/unit/services/test_project_extractor.py
class TestProjectExtractor:
    """Tests for project extraction."""

    async def test_extracts_from_cluster(self, session, sample_cluster):
        """Extracts project from email cluster."""
        extractor = ProjectExtractor(session)
        projects = await extractor.extract_from_clusters(user_id)

        assert len(projects) == 1
        assert projects[0].name == "Q1 Marketing Campaign"
        assert projects[0].detected_from == "cluster"

    async def test_links_emails_to_project(self, session, sample_cluster):
        """Links cluster emails to extracted project."""
        extractor = ProjectExtractor(session)
        projects = await extractor.extract_from_clusters(user_id)

        # Verify email associations
        project = projects[0]
        assert len(project.emails) == sample_cluster.email_count

    async def test_deduplicates_projects(self, session):
        """Doesn't create duplicate projects."""
        extractor = ProjectExtractor(session)

        # Run twice
        await extractor.extract_from_clusters(user_id)
        projects = await extractor.extract_from_clusters(user_id)

        # Second run should find existing
        assert len(projects) == 0


# tests/unit/services/test_task_extractor.py
class TestTaskExtractor:
    """Tests for task extraction."""

    async def test_extracts_from_llm(self, session, sample_classification):
        """Extracts task from LLM classification."""
        extractor = TaskExtractor(session)
        tasks = await extractor.extract_from_llm(user_id)

        assert len(tasks) >= 1
        assert tasks[0].description == sample_classification.action_summary
        assert tasks[0].extraction_method == "llm"

    async def test_extracts_action_items(self, session, classification_with_items):
        """Extracts sub-tasks from action items."""
        extractor = TaskExtractor(session)
        tasks = await extractor.extract_from_llm(user_id)

        # Main task + action items
        assert len(tasks) == 1 + len(classification_with_items.action_items)
```

### Verification Steps

1. **Run migration**:
   ```bash
   alembic upgrade head
   ```

2. **Verify tables created**:
   ```sql
   \dt projects
   \dt tasks
   \dt email_priority_context
   ```

3. **Run extraction stage**:
   ```bash
   rl-emails --user <uuid> --start-from 12
   ```

4. **Verify data populated**:
   ```sql
   SELECT COUNT(*) FROM projects WHERE user_id = '<uuid>';
   SELECT COUNT(*) FROM tasks WHERE user_id = '<uuid>';
   SELECT COUNT(*) FROM email_priority_context WHERE user_id = '<uuid>';
   ```

### Success Criteria

| Metric | Requirement |
|--------|-------------|
| Tables created | All with proper indexes |
| Project detection | Projects extracted from clusters |
| Task extraction | Tasks from LLM action items |
| Priority context | Computed for all emails |
| Test coverage | 100% on new code |

---

## Iteration 1: FastAPI Foundation

### Story
As a developer, I need a FastAPI application structure so that I can build secure, documented API endpoints with production-ready logging, rate limiting, and error handling.

### Deliverables
1. FastAPI application with proper project structure
2. Health check and readiness endpoints
3. OpenAPI documentation configuration
4. CORS middleware for mobile/web clients
5. Global error handling with RFC 7807 Problem Details
6. Structured logging with structlog and correlation IDs
7. Rate limiting foundation with slowapi
8. Async database connection pooling

### Architecture

```
src/rl_emails/
├── api/                              # NEW: FastAPI application
│   ├── __init__.py
│   ├── main.py                       # FastAPI app factory with lifespan
│   ├── config.py                     # APIConfig (pydantic-settings)
│   ├── database.py                   # Async engine + session factory
│   ├── dependencies.py               # Dependency injection
│   ├── exceptions.py                 # Custom exceptions + Problem Details
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── cors.py                  # CORS configuration
│   │   ├── error_handler.py         # Global exception middleware
│   │   ├── logging.py               # Request logging with correlation ID
│   │   └── rate_limit.py            # Rate limiting setup (slowapi)
│   └── routes/
│       ├── __init__.py
│       └── health.py                # /health and /ready endpoints
```

### Dependencies

```toml
# FastAPI + Server
"fastapi>=0.115.0",
"uvicorn[standard]>=0.32.0",

# Structured logging
"structlog>=24.0.0",
"asgi-correlation-id>=4.0.0",

# Rate limiting
"slowapi>=0.1.9",

# Async database
"asyncpg>=0.30.0",
```

### Best Practices Applied

1. **Structured Logging (structlog)**
   - JSON format in production, pretty logs in development
   - Correlation IDs via asgi-correlation-id for request tracing
   - Context variables auto-added to all logs
   - Non-blocking async-safe logging

2. **Rate Limiting (slowapi)**
   - Token bucket algorithm for burst control
   - Redis-ready for distributed deployments
   - Per-endpoint configurable limits
   - Standard rate limit headers (X-RateLimit-*)

3. **Async Database (asyncpg + SQLAlchemy)**
   - Connection pooling: pool_size=10, max_overflow=5
   - Pool recycling every 30 minutes
   - Dependency injection for session lifecycle
   - Non-blocking database operations

4. **Error Handling**
   - RFC 7807 Problem Details format
   - Global exception middleware (not just handlers)
   - Starlette HTTPException handling
   - Structured error logging with context

5. **Middleware Ordering**
   - CORS (outermost)
   - Correlation ID
   - Request Logging
   - Rate Limiting
   - Error Handler (innermost)

### Implementation Design

```python
# src/rl_emails/api/config.py
from pydantic_settings import BaseSettings

class APIConfig(BaseSettings):
    """API configuration from environment."""

    # Server
    debug: bool = False
    enable_docs: bool = True

    # Database
    database_url: str
    db_pool_size: int = 10
    db_max_overflow: int = 5
    db_pool_recycle: int = 1800

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # Rate limiting
    rate_limit_per_minute: int = 60

    # Logging
    log_level: str = "INFO"
    log_json: bool = True  # JSON in prod, pretty in dev

    model_config = {"env_prefix": "API_"}
```

```python
# src/rl_emails/api/exceptions.py
from dataclasses import dataclass
from typing import Any

@dataclass
class ProblemDetail:
    """RFC 7807 Problem Details response."""
    type: str = "about:blank"
    title: str = "An error occurred"
    status: int = 500
    detail: str | None = None
    instance: str | None = None
    extensions: dict[str, Any] | None = None
```

```python
# src/rl_emails/api/middleware/logging.py
import structlog
from asgi_correlation_id import CorrelationIdMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing and correlation ID."""

    async def dispatch(self, request, call_next):
        logger = structlog.get_logger()
        start = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000
        await logger.ainfo(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response
```

```python
# src/rl_emails/api/routes/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()

@router.get("/health")
async def health() -> dict:
    """Liveness probe - is the service running?"""
    return {"status": "healthy"}

@router.get("/ready")
async def ready(db: AsyncSession = Depends(get_db_session)) -> dict:
    """Readiness probe - is the service ready to handle requests?"""
    await db.execute(text("SELECT 1"))
    return {"status": "ready", "database": "connected"}
```

### Acceptance Criteria

- [x] FastAPI app factory creates configured application
- [x] Health endpoint returns 200 with status info
- [x] Readiness endpoint validates database connection
- [x] CORS allows configured origins
- [x] Global error handler returns RFC 7807 Problem Details
- [x] Request logging captures method, path, duration, correlation_id
- [x] Correlation ID propagated via X-Request-ID header
- [x] Rate limiting returns 429 with Retry-After header
- [x] OpenAPI docs available at /docs (configurable)
- [x] App starts with `uvicorn rl_emails.api.main:app`
- [x] Structured logs in JSON format (production mode)
- [x] Database connection pooling configured
- [x] 100% test coverage on new code
- [x] mypy --strict passes

### Test Plan

```python
# tests/unit/api/test_health.py
class TestHealthEndpoints:
    async def test_health_returns_healthy(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    async def test_ready_checks_database(self, client, mock_db):
        response = await client.get("/ready")
        assert response.status_code == 200
        assert response.json()["database"] == "connected"

    async def test_ready_fails_without_database(self, client, broken_db):
        response = await client.get("/ready")
        assert response.status_code == 503

# tests/unit/api/test_middleware.py
class TestRequestLogging:
    async def test_logs_request_with_correlation_id(self, client, caplog):
        response = await client.get("/health")
        assert "correlation_id" in caplog.text
        assert "duration_ms" in caplog.text

class TestRateLimiting:
    async def test_rate_limit_returns_429(self, client):
        # Exceed rate limit
        for _ in range(100):
            await client.get("/health")
        response = await client.get("/health")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

class TestErrorHandling:
    async def test_returns_problem_details(self, client):
        response = await client.get("/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert data["type"] == "about:blank"
        assert data["status"] == 404
```

### Verification Steps

1. **Start the server**:
   ```bash
   uvicorn rl_emails.api.main:app --reload
   ```

2. **Test health endpoints**:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/ready
   ```

3. **Verify correlation ID**:
   ```bash
   curl -H "X-Request-ID: test-123" http://localhost:8000/health -v
   # Response should include X-Request-ID: test-123
   ```

4. **Check rate limiting**:
   ```bash
   for i in {1..100}; do curl -s http://localhost:8000/health; done
   # Should see 429 responses after limit exceeded
   ```

5. **Verify structured logs**:
   ```bash
   API_LOG_JSON=true uvicorn rl_emails.api.main:app
   # Logs should be JSON formatted
   ```

---

## Iteration 2: Clerk Authentication

### Story
As a user, I need to authenticate with Clerk so that I can securely access my email data through the API.

### Deliverables
1. Clerk JWT validation middleware
2. User context extraction from JWT
3. Protected route decorator
4. User sync from Clerk to local database
5. API key support for service-to-service calls

### Implementation Design

```python
# src/rl_emails/api/auth/clerk.py
from dataclasses import dataclass
import jwt
from jwt import PyJWKClient


@dataclass
class ClerkUser:
    """User info from Clerk JWT."""
    id: str
    email: str | None
    first_name: str | None
    last_name: str | None


class ClerkJWTValidator:
    """Validates Clerk JWT tokens."""

    def __init__(self, config: ClerkConfig) -> None:
        self.config = config
        self._jwks_client = PyJWKClient(config.jwks_url)

    async def validate_token(self, token: str) -> ClerkUser:
        """Validate JWT and extract user info."""
        signing_key = self._jwks_client.get_signing_key_from_jwt(token)

        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=self.config.issuer,
        )

        return ClerkUser(
            id=payload["sub"],
            email=payload.get("email"),
            first_name=payload.get("first_name"),
            last_name=payload.get("last_name"),
        )
```

### Acceptance Criteria

- [x] ClerkJWTValidator validates RS256 JWTs
- [x] Token expiration is enforced
- [x] Invalid tokens return 401 with proper message
- [x] get_current_user dependency extracts user from request
- [x] User is synced to local database on first auth
- [x] Protected routes require valid JWT
- [x] 100% test coverage on new code (104 auth tests)

### Deliverables Implemented

- `src/rl_emails/api/auth/config.py` - ClerkConfig with pydantic-settings
- `src/rl_emails/api/auth/clerk.py` - ClerkJWTValidator, ClerkUser dataclass
- `src/rl_emails/api/auth/exceptions.py` - Auth exception classes
- `src/rl_emails/api/auth/dependencies.py` - FastAPI dependencies for auth
- `src/rl_emails/api/auth/middleware.py` - AuthenticationMiddleware
- `src/rl_emails/api/auth/user_sync.py` - UserSyncService for Clerk users
- `tests/unit/api/auth/` - 104 comprehensive tests

---

## Iteration 3: Email Connection Provider Interface

### Story
As a developer, I need a provider interface so that I can support multiple email services (Gmail now, others later) with a unified API.

### Deliverables
1. Email provider abstract interface
2. Gmail provider implementation (wrap existing)
3. Connection management service
4. Provider registry pattern
5. Connection status endpoints

### Implementation Design

```python
# src/rl_emails/providers/base.py
from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncIterator


class ProviderType(str, Enum):
    """Supported email providers."""
    GMAIL = "gmail"
    # Future: OUTLOOK, IMAP


class EmailProvider(ABC):
    """Abstract email provider interface."""

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        pass

    @abstractmethod
    async def connect(self, user_id: UUID, credentials: dict) -> ConnectionStatus:
        pass

    @abstractmethod
    async def sync_messages(
        self,
        user_id: UUID,
        since: datetime | None = None,
    ) -> AsyncIterator[EmailMessage]:
        pass

    @abstractmethod
    async def get_auth_url(self, user_id: UUID, redirect_uri: str) -> str:
        pass
```

### Acceptance Criteria

- [ ] EmailProvider abstract class defines clear interface
- [ ] GmailProvider implements all EmailProvider methods
- [ ] Provider registry supports dynamic registration
- [ ] Connection service manages multi-provider connections
- [ ] API endpoints for connect/disconnect/status
- [ ] 100% test coverage on new code

---

## Iteration 4: Real-Time Sync with Gmail Push Notifications

### Story
As a user, I need real-time email updates so that my data is always fresh without manual sync.

### Deliverables
1. Gmail Push Notification (Pub/Sub) integration
2. Webhook endpoint for Gmail notifications
3. Background sync worker
4. Sync state management
5. Notification deduplication

### Implementation Design

```python
# src/rl_emails/api/routes/webhooks.py
from fastapi import APIRouter, Request

router = APIRouter()


@router.post("/webhooks/gmail")
async def gmail_webhook(
    request: Request,
    push_service: PushNotificationService = Depends(get_push_service),
) -> dict:
    """Handle Gmail push notification webhook."""
    body = await request.json()

    message_data = body.get("message", {}).get("data")
    if message_data:
        await push_service.handle_notification(message_data)

    return {"status": "ok"}
```

### Acceptance Criteria

- [ ] Gmail watch setup creates Pub/Sub subscription
- [ ] Webhook endpoint receives and validates notifications
- [ ] Sync worker processes queued sync jobs
- [ ] Incremental sync uses history_id
- [ ] Watch expiration is monitored and renewed
- [ ] 100% test coverage on new code

---

## Iteration 5: Projects & Tasks API

### Story
As a user, I need API endpoints to view and manage my projects and tasks so that I can focus on what needs attention.

### Deliverables
1. Project list/detail endpoints
2. Task list/detail endpoints
3. Task status management (complete, dismiss)
4. Project-email associations
5. Priority inbox endpoint

### Implementation Design

```python
# src/rl_emails/api/routes/projects.py
from fastapi import APIRouter, Depends

router = APIRouter()


@router.get("/projects", response_model=list[ProjectResponse])
async def list_projects(
    user: CurrentUser = Depends(get_current_user),
    is_active: bool = True,
    project_service: ProjectService = Depends(get_project_service),
) -> list[ProjectResponse]:
    """List user's projects."""
    return await project_service.list_projects(
        user_id=user.id,
        is_active=is_active,
    )


@router.get("/projects/{project_id}", response_model=ProjectDetailResponse)
async def get_project(
    project_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    project_service: ProjectService = Depends(get_project_service),
) -> ProjectDetailResponse:
    """Get project details with associated emails."""
    return await project_service.get_project(
        user_id=user.id,
        project_id=project_id,
    )


# src/rl_emails/api/routes/tasks.py
@router.get("/tasks", response_model=list[TaskResponse])
async def list_tasks(
    user: CurrentUser = Depends(get_current_user),
    status: str = "pending",
    task_service: TaskService = Depends(get_task_service),
) -> list[TaskResponse]:
    """List user's tasks."""
    return await task_service.list_tasks(
        user_id=user.id,
        status=status,
    )


@router.post("/tasks/{task_id}/complete")
async def complete_task(
    task_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service),
) -> TaskResponse:
    """Mark task as completed."""
    return await task_service.complete_task(
        user_id=user.id,
        task_id=task_id,
    )


@router.post("/tasks/{task_id}/dismiss")
async def dismiss_task(
    task_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    task_service: TaskService = Depends(get_task_service),
) -> TaskResponse:
    """Dismiss task (not applicable)."""
    return await task_service.dismiss_task(
        user_id=user.id,
        task_id=task_id,
    )


# src/rl_emails/api/routes/inbox.py
@router.get("/inbox/priority", response_model=PriorityInboxResponse)
async def get_priority_inbox(
    user: CurrentUser = Depends(get_current_user),
    limit: int = 20,
    email_service: EmailService = Depends(get_email_service),
) -> PriorityInboxResponse:
    """Get priority-sorted inbox with context."""
    return await email_service.get_priority_inbox(
        user_id=user.id,
        limit=limit,
    )
```

### Acceptance Criteria

- [x] Project list with filtering
- [x] Project detail with associated emails
- [x] Task list with status filtering
- [x] Task complete/dismiss actions
- [x] Priority inbox endpoint
- [x] Pagination support
- [x] 100% test coverage on new code (1954 tests)

### Deliverables Implemented

- `src/rl_emails/api/routes/projects.py` - Project CRUD endpoints
- `src/rl_emails/api/routes/tasks.py` - Task CRUD + complete/dismiss endpoints
- `src/rl_emails/api/routes/inbox.py` - Priority inbox endpoint
- `src/rl_emails/services/project_service.py` - Project business logic
- `src/rl_emails/services/task_service.py` - Task business logic
- `src/rl_emails/services/inbox_service.py` - Priority inbox service
- `src/rl_emails/schemas/project.py` - Project Pydantic schemas
- `src/rl_emails/schemas/task.py` - Task Pydantic schemas
- `src/rl_emails/schemas/inbox.py` - Inbox response schemas
- `src/rl_emails/models/project.py` - Project SQLAlchemy model
- `src/rl_emails/models/task.py` - Task SQLAlchemy model
- `src/rl_emails/repositories/project.py` - Project repository
- `src/rl_emails/repositories/task.py` - Task repository
- `tests/unit/api/routes/test_projects.py` - Project route tests
- `tests/unit/api/routes/test_tasks.py` - Task route tests
- `tests/unit/api/routes/test_inbox.py` - Inbox route tests
- Full test coverage for all services, schemas, models, and repositories

---

## Future Iterations

> **Moved to [FUTURE_PHASES.md](./FUTURE_PHASES.md)**:
> - Background Pipeline Worker
> - Scheduled Sync & Watch Renewal
> - Integration Testing & Documentation

---

## Phase 3 Completion Checklist

### Pre-Implementation
- [x] Clerk account and credentials configured
- [x] Google Cloud Pub/Sub topic created
- [x] Phase 2 complete (Gmail API working)

### Iteration 0: Database Models & Population ✅ COMPLETE
- [x] SQLAlchemy models for Project, Task, PriorityContext (columns added)
- [x] Alembic migration (`fb71ba5949b0_add_entity_extraction_columns.py`)
- [x] ProjectExtractor service (`entity_extraction.py`)
- [x] TaskExtractor service (`entity_extraction.py`)
- [x] PriorityContextBuilder service (`entity_extraction.py`)
- [x] Stage 12 entity extraction (`stage_12_entity_extraction.py`)
- [x] Marketing email filtering
- [x] Write tests (105 new tests, 100% coverage)

### Iteration 1: FastAPI Foundation ✅ COMPLETE
- [x] Create API module structure
- [x] Implement app factory
- [x] Add health endpoints
- [x] Configure middleware (CORS, logging, rate limiting, error handling)
- [x] Write tests (126 tests, 100% coverage)

### Iteration 2: Clerk Authentication ✅ COMPLETE
- [x] Implement JWT validation (`ClerkJWTValidator`)
- [x] Create auth dependencies (`get_current_user`, `require_auth`)
- [x] Add user sync (`UserSyncService`)
- [x] API key auth for service-to-service calls
- [x] Write tests (104 auth tests)

### Iteration 3: Provider Interface ✅ COMPLETE
- [x] Define abstract interface (`EmailProvider`)
- [x] Implement Gmail provider
- [x] Create provider registry
- [x] Connection management service
- [x] Write tests

### Iteration 4: Real-Time Sync ✅ COMPLETE
- [x] Gmail push notifications (Pub/Sub)
- [x] Webhook endpoint (`/webhooks/gmail`)
- [x] Watch subscription management
- [x] Sync worker with history API
- [x] Write tests

### Iteration 5: Projects & Tasks API ✅ COMPLETE
- [x] Project endpoints (list, detail, create, update, delete)
- [x] Task endpoints (list, detail, create, update, complete, dismiss, delete)
- [x] Priority inbox endpoint
- [x] ProjectService, TaskService, InboxService
- [x] Full Pydantic schemas for all responses
- [x] SQLAlchemy models and repositories
- [x] Write tests (1954 total tests, 100% coverage)

### Enhanced Clustering ✅ COMPLETE
- [x] ClusterMetadata model with temporal strength fields
- [x] ClusterLabelerService for LLM-based auto-labeling
- [x] ProjectDetectorService for detecting projects from clusters
- [x] Temporal strength calculation (recency, velocity, trend)
- [x] Stage 13: enhance_clusters pipeline stage
- [x] Migrations for cluster_metadata enhancements
- [x] Write tests (77 new tests)

---

## Future Work

See [FUTURE_PHASES.md](./FUTURE_PHASES.md) for deferred items including:
- Background Pipeline Worker
- Scheduled Sync & Watch Renewal
- Integration Testing & Documentation
- IMAP/Outlook providers
- Mobile push notifications
- ML model training

---

## Success Criteria Summary

| Iteration | Key Deliverable | Status |
|-----------|-----------------|--------|
| 0 | Database models & population | ✅ Complete |
| 1 | FastAPI app | ✅ Complete |
| 2 | Clerk auth | ✅ Complete |
| 3 | Provider interface | ✅ Complete |
| 4 | Real-time sync | ✅ Complete |
| 5 | Projects/Tasks API | ✅ Complete |
| Enhanced | Cluster labeling & project detection | ✅ Complete |

**Phase 3 Complete** ✅:
- [x] Database tables populated with projects/tasks
- [x] API serves authenticated requests
- [x] Gmail sync works in real-time
- [x] Projects and tasks extracted and accessible
- [x] 100% test coverage maintained (1954 tests)
