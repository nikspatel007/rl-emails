# Database Schema

## Overview

Multi-table design to support:
- **Raw data preservation** - Never lose original data, replay anytime
- **Parallel processing** - Independent workers for 45k+ emails
- **Thread reconstruction** - Reply chains and response patterns
- **Attachment tracking** - At email and thread level

---

## Design Principles

1. **`raw_emails` is immutable** - Direct MBOX parse, never modified
2. **Derived tables rebuilt from raw** - Can drop and recreate anytime
3. **Parallel-safe imports** - Batch inserts with workers, no conflicts
4. **Idempotent enrichment** - Can re-run without duplicates

---

## Tables

### `raw_emails` (Source of Truth)
Immutable raw data from MBOX parse. Never modified after initial import.

```sql
CREATE TABLE raw_emails (
    id SERIAL PRIMARY KEY,
    message_id TEXT UNIQUE NOT NULL,

    -- Raw headers (exactly as parsed from MBOX)
    thread_id TEXT,                    -- X-GM-THRID
    in_reply_to TEXT,
    references_raw TEXT,               -- Full References header

    date_raw TEXT,
    from_raw TEXT,
    to_raw TEXT,
    cc_raw TEXT,
    bcc_raw TEXT,
    subject_raw TEXT,

    -- Raw content
    body_text TEXT,
    body_html TEXT,

    -- Gmail metadata
    labels_raw TEXT,                   -- Unparsed X-Gmail-Labels

    -- MBOX metadata (for re-parsing if needed)
    mbox_offset BIGINT,                -- Byte offset in MBOX file
    raw_size_bytes INTEGER,

    imported_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_raw_emails_message_id ON raw_emails(message_id);
CREATE INDEX idx_raw_emails_thread_id ON raw_emails(thread_id);
```

### `emails` (Enriched/Derived)
Derived from raw_emails with parsed and computed fields. Can be dropped and rebuilt.

```sql
CREATE TABLE emails (
    id SERIAL PRIMARY KEY,
    raw_email_id INTEGER REFERENCES raw_emails(id),
    message_id TEXT UNIQUE NOT NULL,
    thread_id TEXT,
    in_reply_to TEXT,

    -- Parsed headers
    date_parsed TIMESTAMPTZ,
    from_email TEXT,
    from_name TEXT,
    to_emails TEXT[],
    cc_emails TEXT[],
    subject TEXT,

    -- Content
    body_text TEXT,
    body_preview TEXT,                 -- First 500 chars for display
    word_count INTEGER,

    -- Parsed Gmail metadata
    labels TEXT[],                     -- Parsed array from labels_raw

    -- Attachment summary (aggregated from attachments table)
    has_attachments BOOLEAN DEFAULT FALSE,
    attachment_count INTEGER DEFAULT 0,
    attachment_types TEXT[],           -- ['pdf', 'xlsx', 'jpg']
    total_attachment_bytes BIGINT DEFAULT 0,

    -- Ownership
    is_sent BOOLEAN DEFAULT FALSE,     -- TRUE if you sent this email

    -- Computed enrichment (Stage 5)
    action TEXT,                       -- REPLIED, FORWARDED, STARRED, IGNORED
    timing TEXT,                       -- IMMEDIATE, SAME_DAY, NEXT_DAY, LATER, NEVER
    response_time_seconds INTEGER,     -- Seconds to your reply
    priority_score FLOAT,              -- Computed priority 0-1

    -- Processing metadata
    enriched_at TIMESTAMPTZ,
    enrichment_version INTEGER DEFAULT 1,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_emails_thread_id ON emails(thread_id);
CREATE INDEX idx_emails_from_email ON emails(from_email);
CREATE INDEX idx_emails_date ON emails(date_parsed);
CREATE INDEX idx_emails_in_reply_to ON emails(in_reply_to);
CREATE INDEX idx_emails_is_sent ON emails(is_sent);
CREATE INDEX idx_emails_labels ON emails USING GIN(labels);
```

### `attachments`
Individual attachment metadata. Links to both raw_emails and emails.

```sql
CREATE TABLE attachments (
    id SERIAL PRIMARY KEY,
    raw_email_id INTEGER REFERENCES raw_emails(id),
    email_id INTEGER REFERENCES emails(id),  -- Set after enrichment

    -- Attachment metadata
    filename TEXT,
    content_type TEXT,                 -- MIME type
    size_bytes INTEGER,
    content_disposition TEXT,          -- inline/attachment

    -- Hashing for dedup
    content_hash TEXT,                 -- SHA256 of content (if stored)

    -- Storage (optional - may just track metadata)
    stored_path TEXT,                  -- If saving to disk

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_attachments_raw_email_id ON attachments(raw_email_id);
CREATE INDEX idx_attachments_email_id ON attachments(email_id);
CREATE INDEX idx_attachments_content_type ON attachments(content_type);
```

### `threads` (Aggregated)
Thread-level aggregations with attachment summary.

```sql
CREATE TABLE threads (
    id SERIAL PRIMARY KEY,
    thread_id TEXT UNIQUE NOT NULL,    -- X-GM-THRID
    subject TEXT,                      -- From first email

    -- Participants
    participants TEXT[],               -- All unique email addresses
    your_role TEXT,                    -- 'sender', 'recipient', 'cc', 'none'

    -- Email stats
    email_count INTEGER DEFAULT 0,
    your_email_count INTEGER DEFAULT 0,
    your_reply_count INTEGER DEFAULT 0,

    -- Attachment stats (thread-level)
    has_attachments BOOLEAN DEFAULT FALSE,
    total_attachment_count INTEGER DEFAULT 0,
    attachment_types TEXT[],           -- All types in thread

    -- Timing
    started_at TIMESTAMPTZ,
    last_activity TIMESTAMPTZ,
    your_first_reply_at TIMESTAMPTZ,

    -- Computed
    avg_response_time_seconds INTEGER,
    thread_duration_seconds INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_threads_thread_id ON threads(thread_id);
CREATE INDEX idx_threads_has_attachments ON threads(has_attachments);
```

### `users` (Aggregated)
Email addresses with aggregated communication stats.

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT,                         -- Most common display name

    -- Relationship to you
    is_you BOOLEAN DEFAULT FALSE,      -- Your own email addresses

    -- Communication stats
    emails_from INTEGER DEFAULT 0,     -- Emails they sent you
    emails_to INTEGER DEFAULT 0,       -- Emails you sent them
    threads_with INTEGER DEFAULT 0,    -- Threads involving them
    reply_count INTEGER DEFAULT 0,     -- Times you replied to them

    -- Response patterns
    avg_response_time_seconds INTEGER, -- Your avg response to them
    reply_rate FLOAT,                  -- % of their emails you reply to

    -- Labels
    is_important_sender BOOLEAN DEFAULT FALSE,
    first_contact TIMESTAMPTZ,
    last_contact TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_is_you ON users(is_you);
```

---

## Parallel Processing Design

### MBOX Import (Stage 2)
```python
# Split MBOX into chunks by byte offset
# Each worker processes a chunk independently

WORKERS = 8
CHUNK_SIZE = 5000  # emails per batch

async def parallel_import(mbox_path):
    # Pre-scan to get message offsets
    offsets = scan_mbox_offsets(mbox_path)

    # Split into chunks
    chunks = [offsets[i:i+CHUNK_SIZE] for i in range(0, len(offsets), CHUNK_SIZE)]

    # Process chunks in parallel
    async with asyncio.TaskGroup() as tg:
        for chunk in chunks:
            tg.create_task(import_chunk(mbox_path, chunk))
```

### Enrichment (Stage 5)
```python
# Process emails in parallel batches
# Each batch is independent - no cross-batch dependencies

async def parallel_enrich():
    # Get all raw_email IDs
    ids = await db.fetch("SELECT id FROM raw_emails")

    # Split into batches
    batches = chunk(ids, 1000)

    # Process with worker pool
    async with asyncio.TaskGroup() as tg:
        for batch in batches:
            tg.create_task(enrich_batch(batch))
```

### Worker Configuration
```python
# config.py
PARALLEL_CONFIG = {
    'import_workers': 8,
    'import_batch_size': 1000,
    'enrich_workers': 4,
    'enrich_batch_size': 500,
    'db_pool_size': 20,
}
```

---

## Rebuilding Derived Tables

Since `raw_emails` is immutable, you can always rebuild:

```sql
-- Drop derived tables
DROP TABLE IF EXISTS emails CASCADE;
DROP TABLE IF EXISTS threads CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Recreate schema
\i schema.sql

-- Re-run enrichment pipeline
-- (scripts/enrich_data.py)
```

---

## Enrichment Flow (Stage 5)

```
raw_emails
    │
    ├─→ Parse headers → emails (basic fields)
    │
    ├─→ Extract attachments → attachments table
    │
    ├─→ Update emails.attachment_* fields
    │
    ├─→ Mark is_sent = TRUE for your emails
    │
    ├─→ Match replies (in_reply_to → message_id)
    │   └─→ Set action, timing, response_time
    │
    ├─→ Aggregate → threads table
    │   └─→ Include thread-level attachment stats
    │
    └─→ Aggregate → users table
```

---

## Queries for Training

### Get emails with computed labels
```sql
SELECT
    e.message_id,
    e.subject,
    e.body_preview,
    e.labels,
    e.action,
    e.timing,
    e.has_attachments,
    e.attachment_types
FROM emails e
WHERE e.action IS NOT NULL
  AND e.is_sent = FALSE;
```

### Get preference pairs (replied vs ignored)
```sql
SELECT
    replied.message_id AS chosen_id,
    ignored.message_id AS rejected_id,
    'reply_signal' AS signal_type,
    replied.response_time_seconds
FROM emails replied
JOIN emails ignored
    ON replied.from_email = ignored.from_email
WHERE replied.action = 'REPLIED'
  AND ignored.action = 'IGNORED'
  AND ABS(EXTRACT(EPOCH FROM (replied.date_parsed - ignored.date_parsed))) < 86400;
```

### Get threads with attachments
```sql
SELECT
    t.thread_id,
    t.subject,
    t.email_count,
    t.total_attachment_count,
    t.attachment_types,
    array_agg(a.filename) AS attachment_files
FROM threads t
JOIN emails e ON e.thread_id = t.thread_id
JOIN attachments a ON a.email_id = e.id
WHERE t.has_attachments = TRUE
GROUP BY t.id;
```
