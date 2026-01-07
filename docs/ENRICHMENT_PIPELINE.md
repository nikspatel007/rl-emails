# Email Enrichment Pipeline

## Overview

This document describes the data enrichment process for transforming raw Gmail MBOX data into ML-ready features. The pipeline builds on the base data pipeline (see `DATA_PIPELINE.md`) and adds action labels, computed features, and LLM-powered classifications.

**Goal:** Predict what action a user will take on an email (REPLY, ARCHIVE, IGNORE, etc.)

---

## Current Database State

After running the base pipeline, you should have:

| Table | Description | Status |
|-------|-------------|--------|
| `raw_emails` | Immutable source data from MBOX | Required |
| `emails` | Parsed email data with headers | Required |
| `threads` | Thread-level aggregations | Required |
| `users` | User/sender statistics | Required |
| `attachments` | Attachment metadata | Required |
| `email_features` | Pre-computed ML features | Created by this pipeline |
| `email_llm_features` | LLM-powered classifications | Created by this pipeline |

---

## Enrichment Phases

### Phase 1: Action Labels (Foundation)

**Purpose:** Compute what action the user took on each email.

**Script:** `scripts/enrich_emails.py`

**Updates to `emails` table:**
- `is_sent` - TRUE if user sent this email
- `action` - REPLIED, FORWARDED, STARRED, ARCHIVED, IGNORED
- `timing` - IMMEDIATE (<1hr), SAME_DAY, NEXT_DAY, LATER, NEVER
- `response_time_seconds` - Time between receiving and replying

**How it works:**
1. Mark emails as `is_sent=TRUE` based on user's email addresses
2. Match sent emails to received via `in_reply_to → message_id`
3. Compute action based on labels and reply matching
4. Compute timing based on response_time_seconds

**Run:**
```bash
source .venv/bin/activate
python scripts/enrich_emails.py --db-url $DB_URL --your-email me@nik-patel.com
```

**Verification:**
```sql
SELECT action, COUNT(*) FROM emails GROUP BY action;
SELECT timing, COUNT(*) FROM emails WHERE action = 'REPLIED' GROUP BY timing;
```

---

### Phase 2: Basic Feature Extraction

**Purpose:** Compute ML features from email metadata and relationships.

**Script:** `scripts/compute_features.py`

**Creates `email_features` table with:**

| Feature Category | Dimensions | Description |
|-----------------|------------|-------------|
| Relationship | 11 | Sender frequency, response rate, thread depth, CC patterns |
| Service detection | 6 | is_service_email, service_type, unsubscribe signals |
| Temporal | 8 | Hour, day of week, business hours, time zone |
| Content basic | 5 | Word count, has_attachments, recipient count |

**Run:**
```bash
python scripts/compute_features.py --db-url $DB_URL
```

**Verification:**
```sql
SELECT COUNT(*) FROM email_features;
SELECT AVG(relationship_strength), AVG(urgency_score) FROM email_features;
```

---

### Phase 3: Content Embeddings

**Purpose:** Generate semantic embeddings for email content.

**Script:** `scripts/generate_embeddings.py`

**Adds to `email_features`:**
- `content_embedding` - 384-dim sentence-transformer embedding
- `embedding_model` - Model name used
- `embedding_dim` - Dimension of embedding

**Run:**
```bash
python scripts/generate_embeddings.py --db-url $DB_URL --model all-MiniLM-L6-v2
```

**Note:** Requires `sentence-transformers` package. Can run on CPU but GPU recommended for speed.

---

### Phase 4: LLM-Powered Features

**Purpose:** Use LLM for high-quality classification of ambiguous cases.

**Script:** `scripts/extract_llm_features.py`

**Creates `email_llm_features` table with:**

| Feature | Description | LLM Used |
|---------|-------------|----------|
| `service_type` | TRANSACTIONAL, FYI, NEWSLETTER, FINANCIAL, etc. | Haiku |
| `conversation_type` | LONG_RUNNING, SHORT_EXCHANGE, SINGLE_MESSAGE, BROADCAST | Haiku |
| `tasks` | Array of extracted tasks with deadlines | Haiku |
| `urgency_assessment` | LLM-assessed urgency with reasoning | Haiku |

**Run:**
```bash
export ANTHROPIC_API_KEY=your_key
python scripts/extract_llm_features.py --db-url $DB_URL --batch-size 100
```

**Cost estimate:** ~$0.01 per 100 emails with Haiku

---

## Feature Categories Detail

### Relationship Features (11 dims)

From sender/recipient communication patterns:

```python
@dataclass
class RelationshipFeatures:
    emails_from_sender_7d: int      # Recent activity
    emails_from_sender_30d: int
    emails_from_sender_90d: int
    user_replied_to_sender_rate: float  # How often user replies
    avg_response_time_hours: float
    user_initiated_ratio: float     # Who starts conversations
    cc_affinity_score: float        # CC co-occurrence
    avg_thread_depth: float
    days_since_last_interaction: int
    reciprocity_score: float        # Balance of sent/received
    relationship_momentum: float    # Trend (30d vs 90d)
```

### Service Classification (6 dims)

Detect automated/service emails:

```python
@dataclass
class ServiceFeatures:
    is_service_email: bool
    service_confidence: float
    service_type: str  # newsletter, notification, billing, etc.
    has_unsubscribe_link: bool
    has_tracking_pixels: bool
    template_score: float  # 0=personal, 1=fully templated
```

**Service types:**
- TRANSACTIONAL - Receipts, confirmations, shipping
- FYI - Notifications, alerts, status updates
- NEWSLETTER - Content digests, daily/weekly
- FINANCIAL - Statements, bills, investment alerts
- SOCIAL - LinkedIn, Twitter notifications
- MARKETING - Promotions, sales
- SYSTEM - Password resets, security alerts
- CALENDAR - Invites, reminders

### Task Extraction

Extract action items from email content:

```python
@dataclass
class ExtractedTask:
    task_id: str
    description: str
    deadline: Optional[datetime]
    deadline_text: str  # "next Tuesday"
    assignee_hint: str  # "you", "John"
    complexity: str     # trivial, quick, medium, substantial
    task_type: str      # review, send, schedule, decision
    urgency_score: float
```

### Urgency Scoring

Combined urgency from multiple signals:

```python
@dataclass
class UrgencyFeatures:
    urgency_score: float        # 0.0-1.0 overall
    urgency_signals: list[str]  # What triggered score
    deadline_driven: bool
    sender_driven: bool         # From important sender
    content_driven: bool        # Language signals
    thread_activity_driven: bool  # Rapid back-and-forth
```

---

## Full Pipeline Commands

```bash
# Set environment
export DB_URL="postgresql://postgres:postgres@localhost:5433/gmail_twoyrs"
export ANTHROPIC_API_KEY="your_key"
source .venv/bin/activate

# Phase 1: Action labels
python scripts/enrich_emails.py --db-url $DB_URL --your-email me@nik-patel.com

# Phase 2: Basic features
python scripts/compute_features.py --db-url $DB_URL

# Phase 3: Embeddings
python scripts/generate_embeddings.py --db-url $DB_URL

# Phase 4: LLM features (optional, costs money)
python scripts/extract_llm_features.py --db-url $DB_URL --batch-size 100

# Validate
python scripts/validate_features.py --db-url $DB_URL
```

---

## Verification Queries

### Check action label coverage
```sql
SELECT
    action,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct
FROM emails
WHERE is_sent = FALSE
GROUP BY action
ORDER BY count DESC;
```

### Check feature completeness
```sql
SELECT
    COUNT(*) as total_emails,
    COUNT(ef.id) as with_features,
    COUNT(ef.content_embedding) as with_embeddings,
    COUNT(llm.id) as with_llm_features
FROM emails e
LEFT JOIN email_features ef ON ef.email_id = e.id
LEFT JOIN email_llm_features llm ON llm.email_id = e.id
WHERE e.is_sent = FALSE;
```

### Sample feature values
```sql
SELECT
    e.subject,
    ef.relationship_strength,
    ef.urgency_score,
    ef.is_service_email,
    ef.service_type
FROM emails e
JOIN email_features ef ON ef.email_id = e.id
WHERE e.is_sent = FALSE
LIMIT 10;
```

---

## Script Dependencies

All scripts should use environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_URL` | PostgreSQL connection URL | `postgresql://postgres:postgres@localhost:5433/rl_emails` |
| `MBOX_PATH` | Path to MBOX file | (none) |
| `ANTHROPIC_API_KEY` | For LLM features | (required for Phase 4) |

Scripts support `--db-url` CLI override.

---

## Troubleshooting

### "No emails marked as is_sent"
- Check `--your-email` matches your actual email addresses
- Run: `SELECT DISTINCT from_email FROM emails WHERE from_email LIKE '%nik%'`

### "No action labels computed"
- Ensure `in_reply_to` column is populated in raw_emails
- Check: `SELECT COUNT(*) FROM raw_emails WHERE in_reply_to IS NOT NULL`

### "LLM features failing"
- Verify ANTHROPIC_API_KEY is set
- Check API rate limits
- Use `--batch-size 50` for smaller batches

---

## Next Steps After Enrichment

1. **Train ML model** - Use email_features for action prediction
2. **Build preference pairs** - Create training data from REPLIED vs IGNORED
3. **Add temporal features** - Sliding windows, relationship decay
4. **Fine-tune embeddings** - Domain-specific embeddings for email

See `00-feature-extraction-plan.md` for the full feature roadmap.
