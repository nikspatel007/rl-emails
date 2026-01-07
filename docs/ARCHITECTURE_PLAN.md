# rl-emails Architecture Plan: Gmail API Integration & Multi-Tenant Support

## Executive Summary

This document outlines a phased plan to evolve rl-emails to support multiple data sources (MBOX + Gmail API), multi-tenant capabilities, and comprehensive clustering with API access.

**Key Objectives:**
1. **ADD Gmail API as a data source** (MBOX remains fully supported)
2. Support multiple organizations and users
3. Restore full clustering capabilities (people, content, behavior, service, temporal)
4. Build APIs for external consumption
5. Maintain 100% test coverage throughout

**Important:** This is an ADDITIVE change. The existing MBOX pipeline continues to work exactly as it does today. Gmail API provides an alternative data ingestion path that shares the same processing pipeline (stages 3-11).

---

## Quick Reference: All 16 Iterations

| Iter | Phase | Name | Deliverable |
|------|-------|------|-------------|
| **1** | Foundation | Database Schema | `organizations`, `org_users`, `oauth_tokens`, `sync_state` tables |
| **2** | Foundation | Core Models | Pydantic/SQLAlchemy models, Config updates |
| **3** | Foundation | Pipeline Multi-Tenant | All stages accept user_id, scoped queries |
| **4** | Gmail API | OAuth2 Flow | Google OAuth consent, token storage/refresh |
| **5** | Gmail API | Gmail Client | API wrapper, batch fetch, rate limiting |
| **6** | Gmail API | Initial Sync | `rl-emails sync --days N`, Gmail import stage |
| **7** | Gmail API | Incremental Sync | History API, delta processing |
| **8** | Clustering | Cluster Metadata | `cluster_metadata` table, auto-labeling, project detection |
| **9** | Clustering | Advanced Algorithms | HDBSCAN, UMAP, semantic content clustering |
| **10** | Clustering | Priority Enhancement | Project-aware scoring, cluster/sender novelty |
| **11** | API | FastAPI Foundation | App structure, auth middleware |
| **12** | API | Core Endpoints | Org/User CRUD, Gmail connect |
| **13** | API | Email & Analytics | Email listing, clusters, priority inbox |
| **14** | API | Background Jobs | Sync scheduling, re-clustering |
| **15** | Polish | Attachments | Metadata storage, S3 option |
| **16** | Polish | Documentation | OpenAPI docs, rate limiting, monitoring |

**Recommended Starting Points:**
- **Iteration 1** → Start here for foundation
- **Iteration 4** → Jump here if multi-tenant not needed yet
- **Iteration 8** → Jump here to enhance existing clustering first

---

## Current State Analysis

### Existing Pipeline (11 Stages)

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  MBOX File      │────▶│  Stage 1-2       │────▶│  PostgreSQL       │
│  (Single User)  │     │  Parse & Import  │     │  emails table     │
└─────────────────┘     └──────────────────┘     └───────────────────┘
                                                          │
         ┌────────────────────────────────────────────────┘
         ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Stages 3-5     │────▶│  Stage 6         │────▶│  Stages 7-8       │
│  Threads,       │     │  Embeddings      │     │  Classification   │
│  Enrich, Feats  │     │  (OpenAI)        │     │  & Users          │
└─────────────────┘     └──────────────────┘     └───────────────────┘
                                                          │
         ┌────────────────────────────────────────────────┘
         ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Stage 9        │────▶│  Stage 10        │────▶│  Stage 11         │
│  Clustering     │     │  Priority        │     │  LLM Class        │
│  (5 dimensions) │     │  Ranking         │     │  (Optional)       │
└─────────────────┘     └──────────────────┘     └───────────────────┘
```

### Current Database Schema (gmail_test_30d)

| Table | Purpose | User-Aware |
|-------|---------|------------|
| emails | Core email data | ❌ No |
| threads | Thread groupings | ❌ No |
| email_features | ML features | ❌ No |
| email_embeddings | Vector embeddings | ❌ No |
| email_ai_classification | Rule-based labels | ❌ No |
| email_llm_classification | LLM labels | ❌ No |
| users | Sender/recipient profiles | ❌ No |
| email_clusters | Clustering results | ❌ No |
| email_priority | Priority scores | ❌ No |

### Target Database Schema (from gmail_twoyrs)

Additional tables/columns needed:
- `raw_emails` - Raw email storage before processing
- `attachments` - Attachment metadata with file storage
- `cluster_metadata` - Cluster labels and statistics

---

## Target Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API Layer (FastAPI)                             │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│  /auth          │  /users         │  /emails        │  /analytics           │
│  OAuth2 flows   │  User CRUD      │  Email queries  │  Clustering, Priority │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                    │
         ▼                 ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Service Layer                                      │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│  AuthService    │  SyncService    │  AnalysisService│  ClusterService       │
│  - OAuth tokens │  - Gmail API    │  - Embeddings   │  - Multi-dim cluster  │
│  - Refresh      │  - Incremental  │  - Features     │  - Priority compute   │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                    │
         ▼                 ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Data Layer                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  PostgreSQL (Multi-tenant with org_id, user_id on all tables)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ organizations│  │ users        │  │ oauth_tokens │  │ sync_state   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ emails       │  │ threads      │  │ email_feats  │  │ embeddings   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ clusters     │  │ cluster_meta │  │ priority     │  │ attachments  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Tenant Data Model

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
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Modified emails table with multi-tenant support
CREATE TABLE emails (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES org_users(id) ON DELETE CASCADE,
    gmail_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    thread_id TEXT,
    -- ... existing email fields ...
    UNIQUE(user_id, gmail_id)
);

-- All other tables get org_id and user_id foreign keys
```

### Gmail API Integration Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Gmail API Sync Flow                               │
└─────────────────────────────────────────────────────────────────────────┘

1. INITIAL SYNC (First Connection)
   ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
   │  OAuth   │───▶│ Get History │───▶│ List Messages│───▶│ Batch Get  │
   │  Connect │    │ ID          │    │ (last N days)│    │ Full Msgs  │
   └──────────┘    └─────────────┘    └──────────────┘    └────────────┘
                                                                │
                            ┌───────────────────────────────────┘
                            ▼
   ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
   │ Parse &      │───▶│ Run Pipeline│───▶│ Store Sync   │
   │ Store Raw    │    │ Stages 3-11 │    │ State        │
   └──────────────┘    └─────────────┘    └──────────────┘

2. INCREMENTAL SYNC (Subsequent)
   ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
   │ Check Sync   │───▶│ Get History │───▶│ Process      │
   │ State        │    │ Changes     │    │ Deltas       │
   └──────────────┘    └─────────────┘    └──────────────┘
                                                │
                            ┌───────────────────┘
                            ▼
   ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
   │ Add New      │───▶│ Update      │───▶│ Recompute    │
   │ Emails       │    │ Labels      │    │ Affected     │
   └──────────────┘    └─────────────┘    └──────────────┘
```

---

## Iteration Plan

### Phase 1: Multi-Tenant Foundation (Iterations 1-3)

**Goal:** Establish org/user data model and migrate existing pipeline

#### Iteration 1: Database Schema Updates
- Create Alembic migrations for multi-tenant tables
  - `organizations`, `org_users`, `oauth_tokens`, `sync_state`
- Add `org_id`, `user_id` columns to existing tables
- Create indexes for tenant-scoped queries
- Update `raw_emails` table to match gmail_twoyrs structure

**Files to modify:**
- `alembic/versions/` - New migration files
- `src/rl_emails/core/types.py` - Add Organization, OrgUser types

**Tests:**
- Migration rollback/forward tests
- Schema validation tests

#### Iteration 2: Core Model Updates
- Create SQLAlchemy/Pydantic models for new entities
- Update Config to support multi-user context
- Add tenant context middleware concept

**Files to create:**
- `src/rl_emails/models/organization.py`
- `src/rl_emails/models/org_user.py`
- `src/rl_emails/models/oauth_token.py`
- `src/rl_emails/models/sync_state.py`

**Files to modify:**
- `src/rl_emails/core/config.py` - Add user context
- `src/rl_emails/pipeline/stages/*.py` - Add user_id filtering

#### Iteration 3: Pipeline Multi-Tenant Adaptation
- Update all 11 pipeline stages to accept user_id
- Add user scoping to all database queries
- Ensure clustering is per-user

**Files to modify:**
- All `src/rl_emails/pipeline/stages/stage_*.py`
- `src/rl_emails/pipeline/orchestrator.py`

---

### Phase 2: Gmail API Integration (Iterations 4-7)

**Goal:** Replace MBOX ingestion with Gmail API sync

#### Iteration 4: OAuth2 Flow
- Implement Google OAuth2 authentication
- Token storage and refresh logic
- Scopes: `gmail.readonly`, `gmail.labels`

**Files to create:**
- `src/rl_emails/auth/oauth.py`
- `src/rl_emails/auth/google.py`
- `src/rl_emails/services/auth_service.py`

**Environment variables:**
```
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_REDIRECT_URI=...
```

#### Iteration 5: Gmail API Client
- Create Gmail API wrapper
- Implement message listing with date range filter
- Implement batch message fetching
- Handle rate limiting and pagination

**Files to create:**
- `src/rl_emails/integrations/gmail/__init__.py`
- `src/rl_emails/integrations/gmail/client.py`
- `src/rl_emails/integrations/gmail/models.py`
- `src/rl_emails/integrations/gmail/parser.py`

**Key functions:**
```python
class GmailClient:
    async def list_messages(
        self,
        user_id: str,
        days: int = 30,
        labels: list[str] | None = None,
    ) -> AsyncIterator[MessageRef]

    async def get_message(self, message_id: str) -> GmailMessage

    async def batch_get_messages(
        self,
        message_ids: list[str],
        batch_size: int = 100,
    ) -> AsyncIterator[GmailMessage]
```

#### Iteration 6: Initial Sync Implementation
- **Keep Stage 1-2 for MBOX** (no changes to existing pipeline)
- **Add new Gmail sync stages** as alternative ingestion path
- Both paths feed into the same stages 3-11

**Files to keep unchanged:**
- `src/rl_emails/pipeline/stages/stage_01_parse_mbox.py` (MBOX still works)
- `src/rl_emails/pipeline/stages/stage_02_import_postgres.py` (MBOX still works)

**Files to create (Gmail path):**
- `src/rl_emails/pipeline/sources/gmail_source.py` (new Gmail-specific ingestion)
- `src/rl_emails/services/sync_service.py` (Gmail sync orchestration)
- `src/rl_emails/pipeline/stages/stage_01_gmail_sync.py` (Gmail equivalent of stage 1-2)

**CLI updates:**
```bash
# New commands
rl-emails sync --user <user_id> --days 30
rl-emails sync --user <user_id> --since "2024-01-01"
rl-emails sync --user <user_id> --incremental
```

#### Iteration 7: Incremental Sync
- Implement Gmail History API integration
- Delta sync for new/modified/deleted emails
- Handle label changes
- Update affected clusters/priority

**Files to create:**
- `src/rl_emails/integrations/gmail/history.py`
- `src/rl_emails/services/delta_processor.py`

---

### Phase 3: Enhanced Clustering & Importance (Iterations 8-10)

**Goal:** Implement full clustering capabilities with project/topic detection and importance scoring

#### Clustering Dimensions Deep Dive

The gmail_twoyrs database shows 5 clustering dimensions that we need to fully implement:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     5-DIMENSIONAL CLUSTERING                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ PEOPLE          │  │ CONTENT         │  │ BEHAVIOR        │
│ (Sender-based)  │  │ (Topic/Project) │  │ (Response)      │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ • Who emails    │  │ • Email topics  │  │ • Reply patterns│
│   you most      │  │ • Projects      │  │ • Response time │
│ • Domains       │  │ • Conversations │  │ • Action taken  │
│ • Relationship  │  │ • Subject       │  │ • Engagement    │
│   strength      │  │   similarity    │  │   level         │
└─────────────────┘  └─────────────────┘  └─────────────────┘

┌─────────────────┐  ┌─────────────────┐
│ SERVICE         │  │ TEMPORAL        │
│ (Automated)     │  │ (Time-based)    │
├─────────────────┤  ├─────────────────┤
│ • Newsletters   │  │ • Morning/Noon  │
│ • Notifications │  │ • Weekday/End   │
│ • Transactional │  │ • Business hrs  │
│ • Marketing     │  │ • Time zones    │
└─────────────────┘  └─────────────────┘
```

#### Content Clustering = Project Detection

**Key Insight:** Content clusters often represent **projects, topics, or conversation threads**. This is critical for:

1. **Project Importance Scoring**: Emails about active projects should be prioritized
2. **Context Grouping**: Show related emails together
3. **Activity Tracking**: Identify which projects are most active

```
Content Cluster Analysis:
┌────────────────────────────────────────────────────────────────────────────┐
│ Cluster | Size | Auto-Label              | Importance Indicators           │
├─────────────────────────────────────────────────────────────────────────────┤
│    0    | 106  | "Q4 Budget Planning"    | High reply rate, recent, urgent │
│    1    |  71  | "Customer Support"      | External senders, needs response│
│    2    |  51  | "Team Standup"          | Recurring, same participants    │
│   ...   | ...  | ...                     | ...                             │
└────────────────────────────────────────────────────────────────────────────┘
```

#### Importance Scoring Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     IMPORTANCE SCORING MODEL                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Importance = f(
    sender_importance,        -- Based on people cluster + relationship strength
    content_importance,       -- Based on content cluster activity + novelty
    urgency_indicators,       -- Keywords, deadline mentions, reply expectations
    behavior_patterns,        -- Historical: did you reply to similar emails?
    temporal_relevance        -- Recency, business hours, time-sensitive
)

Formula (from email_priority table):
┌────────────────────────────────────────────────────────────────────────────┐
│ priority_score =                                                            │
│   0.3 * feature_score +        -- Basic email features                     │
│   0.2 * replied_similarity +    -- Similar to emails you've replied to     │
│   0.2 * cluster_novelty +       -- Novel content in familiar project       │
│   0.3 * sender_novelty          -- Important sender with new topic         │
└────────────────────────────────────────────────────────────────────────────┘

Output: priority_rank (1 = highest priority)
```

#### Iteration 8: Cluster Metadata & Auto-Labeling
- Add `cluster_metadata` table and models
- **Auto-labeling for clusters** (LLM-based or keyword extraction)
- Cluster statistics computation (pct_replied, avg_response_time, etc.)
- **Project detection from content clusters**

**Files to create:**
- `src/rl_emails/models/cluster_metadata.py`
- `src/rl_emails/services/cluster_labeler.py`
- `src/rl_emails/services/project_detector.py`

**Database additions:**
```sql
CREATE TABLE cluster_metadata (
    id UUID PRIMARY KEY,
    org_id UUID REFERENCES organizations(id),
    user_id UUID REFERENCES org_users(id),
    dimension TEXT, -- 'people', 'content', 'behavior', 'service', 'temporal'
    cluster_id INTEGER,
    size INTEGER,
    representative_email_id UUID,
    auto_label TEXT,                    -- LLM-generated or extracted label
    pct_replied FLOAT,                  -- What % of emails got replies
    avg_response_time_hours FLOAT,      -- Average time to respond
    avg_relationship_strength FLOAT,    -- For people clusters
    is_project BOOLEAN DEFAULT false,   -- Content clusters flagged as projects
    project_status TEXT,                -- 'active', 'stale', 'completed'
    last_activity_at TIMESTAMPTZ,       -- Most recent email in cluster
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Index for quick project lookup
CREATE INDEX idx_cluster_metadata_project ON cluster_metadata(user_id, is_project, last_activity_at DESC);
```

#### Iteration 9: Advanced Clustering Algorithms
- Implement **HDBSCAN** for better cluster quality (handles noise)
- **UMAP** for dimensionality reduction before clustering
- Improved content clustering with **semantic similarity**
- **Project extraction** from content clusters

**Files to modify:**
- `src/rl_emails/pipeline/stages/stage_09_cluster_emails.py`

**New clustering approach:**
```python
# Current: KMeans with fixed cluster count
# New: HDBSCAN with automatic cluster detection

def cluster_content_with_projects(embeddings: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Cluster content and identify projects."""
    # 1. Reduce dimensions with UMAP
    reducer = umap.UMAP(n_components=50, metric='cosine')
    reduced = reducer.fit_transform(embeddings)

    # 2. Cluster with HDBSCAN (auto-detects clusters)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
    labels = clusterer.fit_predict(reduced)

    # 3. Identify projects (content clusters with high coherence)
    projects = []
    for cluster_id in set(labels):
        if cluster_id == -1:  # Noise
            continue
        coherence = compute_cluster_coherence(embeddings, labels, cluster_id)
        if coherence > 0.7:
            projects.append({
                'cluster_id': cluster_id,
                'is_project': True,
                'coherence': coherence
            })

    return labels, projects
```

**Dependencies:**
- hdbscan
- umap-learn

#### Iteration 10: Priority Ranking with Project Importance
- Add **cluster novelty scoring** (new email in active project = important)
- Add **sender novelty scoring** (new topic from important sender)
- **Project-aware priority** (active projects get boosted)
- Enhanced priority computation using all clustering dimensions

**Files to modify:**
- `src/rl_emails/pipeline/stages/stage_10_compute_priority.py`

**Priority enhancement:**
```python
def compute_project_importance(email_id: int, clusters: EmailClusters) -> float:
    """Compute importance based on project membership."""
    content_cluster = clusters.content_cluster_id

    # Get project metadata
    project = get_cluster_metadata('content', content_cluster)

    if project.is_project:
        # Boost for active projects
        recency_boost = 1.0 if project.project_status == 'active' else 0.5
        # Boost for high-engagement projects
        engagement_boost = project.pct_replied / 100.0
        return 0.5 + (recency_boost * 0.3) + (engagement_boost * 0.2)

    return 0.5  # Neutral importance for non-project emails
```

---

### Phase 4: API Layer (Iterations 11-14)

**Goal:** Build REST APIs for external consumption

#### Iteration 11: FastAPI Foundation
- Set up FastAPI application structure
- Authentication middleware (OAuth2 + JWT)
- Error handling and logging

**Files to create:**
- `src/rl_emails/api/__init__.py`
- `src/rl_emails/api/app.py`
- `src/rl_emails/api/deps.py`
- `src/rl_emails/api/middleware/auth.py`
- `src/rl_emails/api/middleware/tenant.py`

#### Iteration 12: Core API Endpoints
- Organizations CRUD
- Users CRUD
- Gmail connection flow

**Files to create:**
- `src/rl_emails/api/routes/organizations.py`
- `src/rl_emails/api/routes/users.py`
- `src/rl_emails/api/routes/auth.py`

**Endpoints:**
```
POST   /api/v1/organizations
GET    /api/v1/organizations/{org_id}
POST   /api/v1/organizations/{org_id}/users
GET    /api/v1/organizations/{org_id}/users
POST   /api/v1/auth/google/connect
POST   /api/v1/auth/google/callback
```

#### Iteration 13: Email & Analytics APIs
- Email listing and search
- Thread view
- Cluster analytics
- Priority inbox

**Files to create:**
- `src/rl_emails/api/routes/emails.py`
- `src/rl_emails/api/routes/threads.py`
- `src/rl_emails/api/routes/analytics.py`

**Endpoints:**
```
GET    /api/v1/users/{user_id}/emails
GET    /api/v1/users/{user_id}/emails/{email_id}
GET    /api/v1/users/{user_id}/threads
GET    /api/v1/users/{user_id}/threads/{thread_id}
GET    /api/v1/users/{user_id}/priority-inbox
GET    /api/v1/users/{user_id}/clusters
GET    /api/v1/users/{user_id}/clusters/{dimension}
GET    /api/v1/users/{user_id}/analytics/summary
```

#### Iteration 14: Background Jobs
- Sync scheduling
- Re-clustering triggers
- Webhook notifications

**Files to create:**
- `src/rl_emails/jobs/__init__.py`
- `src/rl_emails/jobs/scheduler.py`
- `src/rl_emails/jobs/sync_job.py`
- `src/rl_emails/jobs/cluster_job.py`

---

### Phase 5: Attachments & Refinements (Iterations 15-16)

#### Iteration 15: Attachment Handling
- Store attachment metadata
- Optional attachment content storage (S3/local)
- Attachment search capabilities

**Files to create:**
- `src/rl_emails/models/attachment.py`
- `src/rl_emails/services/attachment_service.py`
- `src/rl_emails/storage/s3.py` (optional)

#### Iteration 16: Polish & Documentation
- API documentation (OpenAPI)
- Performance optimization
- Rate limiting
- Monitoring hooks

---

## Detailed Iteration Cards

### Iteration 1: Database Schema Updates

**Story:** As a developer, I need multi-tenant database schema so that multiple organizations and users can be supported.

**Acceptance Criteria:**
- [ ] Alembic migration creates `organizations` table
- [ ] Alembic migration creates `org_users` table
- [ ] Alembic migration creates `oauth_tokens` table
- [ ] Alembic migration creates `sync_state` table
- [ ] Alembic migration adds `org_id`, `user_id` to `emails`
- [ ] Alembic migration adds `org_id`, `user_id` to all related tables
- [ ] Indexes created for tenant queries
- [ ] Migration is reversible
- [ ] 100% test coverage maintained

**Technical Notes:**
- Use UUID for all new IDs
- Add `ON DELETE CASCADE` for tenant cleanup
- Consider partial indexes for common queries

---

### Iteration 4: OAuth2 Flow

**Story:** As a user, I want to connect my Gmail account so that my emails can be analyzed.

**Acceptance Criteria:**
- [ ] Google OAuth2 client credentials configured
- [ ] Authorization URL generation
- [ ] Token exchange on callback
- [ ] Token storage encrypted at rest
- [ ] Token refresh before expiry
- [ ] Scope: `gmail.readonly`, `gmail.labels`
- [ ] Error handling for denied/expired tokens
- [ ] 100% test coverage

**Technical Notes:**
```python
# Example OAuth flow
@router.get("/auth/google/connect")
async def google_connect(user_id: UUID):
    state = generate_state(user_id)
    return RedirectResponse(
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={settings.google_client_id}&"
        f"redirect_uri={settings.google_redirect_uri}&"
        f"scope=https://www.googleapis.com/auth/gmail.readonly&"
        f"response_type=code&"
        f"state={state}&"
        f"access_type=offline&"
        f"prompt=consent"
    )

@router.get("/auth/google/callback")
async def google_callback(code: str, state: str):
    user_id = validate_state(state)
    tokens = await exchange_code(code)
    await store_tokens(user_id, tokens)
    return {"status": "connected"}
```

---

### Iteration 6: Initial Sync Implementation

**Story:** As a user, I want to sync my last N days of Gmail so that I can analyze my email patterns.

**Acceptance Criteria:**
- [ ] CLI command: `rl-emails sync --user <id> --days 30`
- [ ] Fetches all messages from last N days
- [ ] Parses message content (to, from, subject, body)
- [ ] Stores in multi-tenant emails table
- [ ] Updates sync_state with progress
- [ ] Handles pagination (>500 messages)
- [ ] Rate limiting respected
- [ ] Runs pipeline stages 3-11 after sync
- [ ] 100% test coverage

**CLI Interface:**
```bash
# Sync last 30 days
rl-emails sync --user abc123 --days 30

# Sync from specific date
rl-emails sync --user abc123 --since 2024-01-01

# Check sync status
rl-emails sync --user abc123 --status

# Force full re-sync
rl-emails sync --user abc123 --full
```

---

## Data Flow Comparison

### Dual Data Source Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DUAL DATA SOURCE ARCHITECTURE                            │
│                     (MBOX + Gmail API both supported)                        │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │           DATA SOURCES                   │
                    │        (Choose One Per User)             │
                    └─────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┴───────────────────────────┐
          │                                                       │
          ▼                                                       ▼
┌─────────────────────┐                               ┌─────────────────────┐
│   MBOX PATH         │                               │   GMAIL API PATH    │
│   (Existing)        │                               │   (New)             │
├─────────────────────┤                               ├─────────────────────┤
│ Stage 1: Parse MBOX │                               │ Gmail Sync:         │
│ Stage 2: Import DB  │                               │ - OAuth             │
│                     │                               │ - Fetch messages    │
│ CLI:                │                               │ - Import to DB      │
│ rl-emails           │                               │                     │
│ (uses MBOX_PATH)    │                               │ CLI:                │
│                     │                               │ rl-emails sync      │
│                     │                               │ --user X --days 30  │
└─────────────────────┘                               └─────────────────────┘
          │                                                       │
          │                                                       │
          └───────────────────────────┬───────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SHARED PROCESSING PIPELINE                               │
│                     (Stages 3-11, identical for both paths)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stage 3:  Populate Threads    │ Build thread relationships                 │
│  Stage 4:  Enrich Emails       │ Compute actions, timing                    │
│  Stage 5:  Compute Features    │ ML features, sender stats                  │
│  Stage 6:  Compute Embeddings  │ OpenAI embeddings                          │
│  Stage 7:  Classify AI Handle  │ Rule-based classification                  │
│  Stage 8:  Populate Users      │ Build user profiles                        │
│  Stage 9:  Cluster Emails      │ 5-dimensional clustering                   │
│  Stage 10: Compute Priority    │ Priority ranking                           │
│  Stage 11: LLM Classification  │ Optional LLM enhancement                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     OUTPUT: Analyzed Email Database                          │
│                     (Same schema regardless of data source)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MBOX Path (Existing - No Changes)
```
MBOX File → Stage 1 (Parse) → Stage 2 (Import) → [Stages 3-11]
```

### Gmail API Path (New - Added)
```
Gmail API → OAuth → Sync (with --days N) → [Stages 3-11]
              │
              └── Incremental sync available via History API
```

---

## Technical Dependencies

### New Python Packages
```toml
[project.dependencies]
# Gmail API
google-api-python-client = "^2.0"
google-auth-httplib2 = "^0.2"
google-auth-oauthlib = "^1.0"

# API Layer
fastapi = "^0.110"
uvicorn = "^0.27"
python-jose = "^3.3"  # JWT
passlib = "^1.7"       # Password hashing

# Background Jobs
celery = "^5.3"
redis = "^5.0"

# Enhanced Clustering
hdbscan = "^0.8"
umap-learn = "^0.5"
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Data Sources | MBOX only | Gmail API + MBOX |
| Multi-user | No | Yes (org/user model) |
| Clustering Dimensions | 5 | 5 (with metadata) |
| Incremental Sync | No | Yes (Gmail History) |
| API Available | No | Yes (REST) |
| Test Coverage | 100% | 100% |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Gmail API quotas | Implement rate limiting, backoff |
| Token expiry during sync | Proactive refresh, checkpoint resume |
| Large mailbox handling | Pagination, batch processing |
| Multi-tenant data isolation | Row-level security, org_id checks |
| Schema migration failures | Thorough testing, rollback support |

---

## Timeline Estimate

| Phase | Iterations | Complexity |
|-------|------------|------------|
| Phase 1: Multi-Tenant Foundation | 1-3 | Medium |
| Phase 2: Gmail API Integration | 4-7 | High |
| Phase 3: Enhanced Clustering | 8-10 | Medium |
| Phase 4: API Layer | 11-14 | Medium |
| Phase 5: Attachments & Polish | 15-16 | Low |

**Total: 16 iterations**

---

## Detailed Process Flow: Gmail API vs MBOX

### Current MBOX Pipeline Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CURRENT: MBOX FILE PROCESSING                            │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Parse MBOX (stage_01_parse_mbox.py)
┌──────────────────────────────────────────────────────────────────────────────┐
│  Input: /path/to/gmail.mbox (Google Takeout export)                          │
│                                                                              │
│  ┌────────────┐     ┌────────────────┐     ┌──────────────────┐              │
│  │ Read MBOX  │────▶│ Parse Headers  │────▶│ Extract Body     │              │
│  │ mailbox.   │     │ From, To, Date │     │ Text/HTML decode │              │
│  │ mbox()     │     │ Subject, MsgID │     │ Attachment detect│              │
│  └────────────┘     └────────────────┘     └──────────────────┘              │
│                                                     │                        │
│                                                     ▼                        │
│                                            ┌──────────────────┐              │
│                                            │ Write JSONL      │              │
│                                            │ data/parsed.jsonl│              │
│                                            └──────────────────┘              │
│  Output: JSONL file with all emails                                          │
└──────────────────────────────────────────────────────────────────────────────┘

Step 2: Import to PostgreSQL (stage_02_import_postgres.py)
┌──────────────────────────────────────────────────────────────────────────────┐
│  Input: data/parsed.jsonl                                                    │
│                                                                              │
│  ┌────────────┐     ┌────────────────┐     ┌──────────────────┐              │
│  │ Read JSONL │────▶│ Sanitize Text  │────▶│ Parse Dates      │              │
│  │ line by    │     │ Remove nulls   │     │ Multiple formats │              │
│  │ line       │     │ Normalize CRLF │     │ RFC2822, ISO8601 │              │
│  └────────────┘     └────────────────┘     └──────────────────┘              │
│                                                     │                        │
│                                                     ▼                        │
│  ┌────────────┐     ┌────────────────┐     ┌──────────────────┐              │
│  │ Batch      │◀────│ Generate       │◀────│ Parse Addresses  │              │
│  │ Insert     │     │ Preview        │     │ Extract all      │              │
│  │ asyncpg    │     │ Count words    │     │ recipients       │              │
│  └────────────┘     └────────────────┘     └──────────────────┘              │
│                                                                              │
│  Output: emails table populated                                              │
└──────────────────────────────────────────────────────────────────────────────┘

Steps 3-11: Analysis Pipeline (same for MBOX and Gmail)
┌──────────────────────────────────────────────────────────────────────────────┐
│  Stage 3:  Populate Threads    │ Build thread_id → email relationships       │
│  Stage 4:  Enrich Emails       │ Compute: is_sent, action, timing            │
│  Stage 5:  Compute Features    │ Sender stats, service detection, urgency    │
│  Stage 6:  Compute Embeddings  │ OpenAI text-embedding-3-small               │
│  Stage 7:  Classify AI Handle  │ Rule-based: newsletter, auto, needs-human   │
│  Stage 8:  Populate Users      │ Build user profiles from email addresses    │
│  Stage 9:  Cluster Emails      │ 5 dimensions: people, content, behavior...  │
│  Stage 10: Compute Priority    │ Hybrid scoring: features + clusters         │
│  Stage 11: LLM Classification  │ Optional: GPT/Claude for edge cases         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Target: Gmail API Pipeline Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TARGET: GMAIL API PROCESSING                             │
└─────────────────────────────────────────────────────────────────────────────┘

Step 0: OAuth Authentication (one-time per user)
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌────────────┐     ┌────────────────┐     ┌──────────────────┐              │
│  │ User clicks│────▶│ Redirect to    │────▶│ User grants      │              │
│  │ "Connect   │     │ Google OAuth   │     │ gmail.readonly   │              │
│  │ Gmail"     │     │ consent screen │     │ permission       │              │
│  └────────────┘     └────────────────┘     └──────────────────┘              │
│                                                     │                        │
│                                                     ▼                        │
│  ┌────────────┐     ┌────────────────┐     ┌──────────────────┐              │
│  │ Store      │◀────│ Exchange code  │◀────│ Callback with    │              │
│  │ refresh &  │     │ for tokens     │     │ auth code        │              │
│  │ access     │     │ (Google API)   │     │                  │              │
│  └────────────┘     └────────────────┘     └──────────────────┘              │
│                                                                              │
│  Output: oauth_tokens record with refresh_token                              │
└──────────────────────────────────────────────────────────────────────────────┘

Step 1: Initial Sync with Days Parameter
┌──────────────────────────────────────────────────────────────────────────────┐
│  CLI: rl-emails sync --user <user_id> --days 30                              │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Date Range Calculation                                                │  │
│  │  ─────────────────────                                                 │  │
│  │  --days 30    → after:2025/12/08 before:2026/01/08                     │  │
│  │  --days 90    → after:2025/10/09 before:2026/01/08                     │  │
│  │  --days 365   → after:2025/01/08 before:2026/01/08                     │  │
│  │  --since 2024-01-01 → after:2024/01/01                                 │  │
│  │  --all        → no date filter (full mailbox)                          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────┐     ┌────────────────┐     ┌──────────────────┐              │
│  │ Validate   │────▶│ Build Gmail    │────▶│ List Messages    │              │
│  │ OAuth      │     │ Query String   │     │ with pagination  │              │
│  │ Token      │     │ "after:DATE"   │     │ (100 per page)   │              │
│  └────────────┘     └────────────────┘     └──────────────────┘              │
│                                                     │                        │
│                                 For each batch of message IDs                │
│                                                     ▼                        │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Gmail API Batch Request (up to 100 messages per batch)                │  │
│  │                                                                        │  │
│  │  POST https://www.googleapis.com/batch/gmail/v1                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │ GET /gmail/v1/users/me/messages/{id}?format=full                 │  │  │
│  │  │ GET /gmail/v1/users/me/messages/{id}?format=full                 │  │  │
│  │  │ ... (up to 100)                                                  │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                     │                        │
│                                                     ▼                        │
│  ┌────────────┐     ┌────────────────┐     ┌──────────────────┐              │
│  │ Parse      │────▶│ Decode Body    │────▶│ Insert to        │              │
│  │ Headers    │     │ Base64 parts   │     │ emails table     │              │
│  │ (same as   │     │ Handle MIME    │     │ with user_id     │              │
│  │ MBOX)      │     │ multipart      │     │                  │              │
│  └────────────┘     └────────────────┘     └──────────────────┘              │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Progress Tracking                                                     │  │
│  │  ─────────────────                                                     │  │
│  │  sync_state.emails_synced = 1523                                       │  │
│  │  sync_state.last_history_id = "12345678"                               │  │
│  │  sync_state.last_sync_at = now()                                       │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Output: emails table populated, sync_state updated                          │
└──────────────────────────────────────────────────────────────────────────────┘

Step 2: Run Analysis Pipeline (Stages 3-11)
┌──────────────────────────────────────────────────────────────────────────────┐
│  After sync completes, automatically run stages 3-11                         │
│  (Same as MBOX pipeline, but scoped to user_id)                              │
│                                                                              │
│  Note: All queries include WHERE user_id = <current_user>                    │
└──────────────────────────────────────────────────────────────────────────────┘

Step 3: Incremental Sync (subsequent runs)
┌──────────────────────────────────────────────────────────────────────────────┐
│  CLI: rl-emails sync --user <user_id> --incremental                          │
│                                                                              │
│  ┌────────────┐     ┌────────────────┐     ┌──────────────────┐              │
│  │ Load last  │────▶│ Call History   │────▶│ Process changes  │              │
│  │ history_id │     │ API with       │     │ - messagesAdded  │              │
│  │ from       │     │ startHistoryId │     │ - labelsAdded    │              │
│  │ sync_state │     │                │     │ - labelsRemoved  │              │
│  └────────────┘     └────────────────┘     └──────────────────┘              │
│                                                     │                        │
│                                                     ▼                        │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Delta Processing                                                      │  │
│  │  ────────────────                                                      │  │
│  │  New emails      → Fetch full message → Insert to DB → Update features │  │
│  │  Label changes   → Update labels array → Recalc action field           │  │
│  │  Deleted emails  → Mark as deleted (soft delete)                       │  │
│  │                                                                        │  │
│  │  After processing: Re-run clustering for affected emails only          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Output: Incremental updates, sync_state.last_history_id updated             │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data Validation: Ensuring Consistency

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DATA VALIDATION PROCESS                                  │
└─────────────────────────────────────────────────────────────────────────────┘

The same validation that happens with MBOX must happen with Gmail API:

┌──────────────────────────────────────────────────────────────────────────────┐
│  1. Email Parsing Validation                                                 │
│  ───────────────────────────                                                 │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐            │
│  │ Header decode   │   │ Date parsing    │   │ Address parsing │            │
│  │ - UTF-8         │   │ - RFC2822       │   │ - Name extract  │            │
│  │ - Base64        │   │ - ISO8601       │   │ - Email extract │            │
│  │ - Q-encoding    │   │ - Fallbacks     │   │ - Validation    │            │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘            │
│                                                                              │
│  Validation checks:                                                          │
│  ✓ message_id is unique                                                      │
│  ✓ from_email is valid email format                                          │
│  ✓ date_parsed is parseable                                                  │
│  ✓ body_text is decodeable                                                   │
│  ✓ No null bytes in text fields                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  2. Data Completeness Validation                                             │
│  ───────────────────────────────                                             │
│  After sync, validate:                                                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  SELECT                                                                 │ │
│  │    COUNT(*) as total_emails,                                            │ │
│  │    COUNT(*) FILTER (WHERE from_email IS NOT NULL) as has_from,          │ │
│  │    COUNT(*) FILTER (WHERE date_parsed IS NOT NULL) as has_date,         │ │
│  │    COUNT(*) FILTER (WHERE body_text IS NOT NULL) as has_body,           │ │
│  │    COUNT(*) FILTER (WHERE thread_id IS NOT NULL) as has_thread          │ │
│  │  FROM emails                                                            │ │
│  │  WHERE user_id = <user_id>                                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Expected: 100% coverage for from, date, body                                │
│  If < 95%: Log warning, investigate parsing issues                           │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  3. Pipeline Stage Validation                                                │
│  ────────────────────────────                                                │
│  After each stage, validate:                                                 │
│                                                                              │
│  Stage 3 (Threads):                                                          │
│    ✓ All emails have thread_id                                               │
│    ✓ Threads table populated                                                 │
│                                                                              │
│  Stage 4 (Enrich):                                                           │
│    ✓ All emails have action field                                            │
│    ✓ Sent emails marked correctly                                            │
│                                                                              │
│  Stage 5 (Features):                                                         │
│    ✓ email_features has 1:1 with emails                                      │
│    ✓ No NULL urgency_score                                                   │
│                                                                              │
│  Stage 6 (Embeddings):                                                       │
│    ✓ email_embeddings has 1:1 with emails                                    │
│    ✓ All embeddings are 1536 dimensions                                      │
│                                                                              │
│  Stage 9 (Clusters):                                                         │
│    ✓ email_clusters has 1:1 with emails                                      │
│    ✓ All 5 cluster dimensions populated                                      │
│                                                                              │
│  Stage 10 (Priority):                                                        │
│    ✓ email_priority has 1:1 with emails                                      │
│    ✓ priority_rank is sequential                                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

### CLI Interface Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CLI COMMANDS                                             │
└─────────────────────────────────────────────────────────────────────────────┘

# Authentication
rl-emails auth connect --email user@example.com
  → Opens browser for Google OAuth
  → Stores tokens in database

rl-emails auth status --email user@example.com
  → Shows: Connected, token expires in 45 minutes

rl-emails auth disconnect --email user@example.com
  → Revokes tokens, clears local storage

# Sync Commands
rl-emails sync --user <user_id> --days <N>
  → Initial sync: Last N days of email
  → Example: --days 30 (1 month), --days 90 (3 months), --days 365 (1 year)

rl-emails sync --user <user_id> --since <DATE>
  → Sync from specific date
  → Example: --since 2024-01-01

rl-emails sync --user <user_id> --incremental
  → Only fetch new emails since last sync
  → Uses Gmail History API

rl-emails sync --user <user_id> --full
  → Force complete re-sync (clears existing data)

rl-emails sync --user <user_id> --status
  → Shows: Last sync, emails count, next sync due

# Pipeline Commands (existing, now with user scope)
rl-emails run --user <user_id>
  → Run full pipeline stages 3-11

rl-emails run --user <user_id> --stage 9
  → Run specific stage

rl-emails status --user <user_id>
  → Show pipeline status for user

# Organization Management
rl-emails org create --name "Acme Corp" --slug acme
rl-emails org add-user --org acme --email user@acme.com
rl-emails org list-users --org acme
```

---

## Comparison: MBOX vs Gmail API Data Sources

Both data sources are fully supported. Choose based on your use case:

| Aspect | MBOX (Existing) | Gmail API (New) |
|--------|-----------------|-----------------|
| **Status** | ✅ Fully supported | ✅ New addition |
| **Data Source** | Local file export | Live API connection |
| **Freshness** | Point-in-time snapshot | Real-time + incremental |
| **Setup** | Download from Takeout | OAuth consent flow |
| **Date Range** | All data in file | Configurable (--days N) |
| **Attachments** | Embedded in MBOX | Separate API calls |
| **Labels** | X-Gmail-Labels header | labels[] array |
| **Rate Limits** | None (local file) | 250 quota units/user/sec |
| **Multi-user** | Separate files per user | Same codebase, different tokens |
| **Incremental** | Re-process entire file | History API deltas |
| **Best For** | One-time analysis, privacy-first | Continuous monitoring, live access |

---

## Gmail API Specifics

### Rate Limiting Strategy

```python
# Gmail API quotas:
# - 250 quota units per user per second
# - List messages = 5 units
# - Get message = 5 units
# - Batch request = 1 unit per sub-request

RATE_LIMIT_CONFIG = {
    "requests_per_second": 10,  # Conservative: 50 units/sec
    "batch_size": 100,          # Max messages per batch
    "retry_delays": [1, 2, 4, 8, 16],  # Exponential backoff
    "max_retries": 5,
}

async def fetch_with_rate_limit(client, message_ids):
    """Fetch messages respecting rate limits."""
    for batch in chunks(message_ids, 100):
        try:
            messages = await client.batch_get(batch)
            yield messages
        except RateLimitError:
            await exponential_backoff()
            yield await client.batch_get(batch)

        await asyncio.sleep(0.1)  # 10 req/sec
```

### Gmail Message Format

```json
{
  "id": "18d1234567890abc",
  "threadId": "18d1234567890abc",
  "labelIds": ["INBOX", "UNREAD", "CATEGORY_PRIMARY"],
  "snippet": "Hey, just wanted to follow up on...",
  "historyId": "12345678",
  "internalDate": "1704067200000",
  "payload": {
    "mimeType": "multipart/alternative",
    "headers": [
      {"name": "From", "value": "sender@example.com"},
      {"name": "To", "value": "recipient@example.com"},
      {"name": "Subject", "value": "Follow up"},
      {"name": "Date", "value": "Mon, 1 Jan 2024 00:00:00 +0000"},
      {"name": "Message-ID", "value": "<abc123@mail.example.com>"}
    ],
    "parts": [
      {
        "mimeType": "text/plain",
        "body": {"data": "base64-encoded-text"}
      },
      {
        "mimeType": "text/html",
        "body": {"data": "base64-encoded-html"}
      }
    ]
  }
}
```

---

## Next Steps

1. Review this plan with stakeholders
2. Prioritize phases based on business needs
3. Begin Iteration 1: Database Schema Updates
4. Set up Google Cloud project for OAuth credentials
