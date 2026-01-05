# Gmail Data Pipeline v2

## Overview

A systematic, reproducible pipeline for processing Gmail MBOX exports into a PostgreSQL database ready for ML training.

**Design Principle:** Scripts are reusable. Point them at any MBOX file and they work.

---

## Architecture

```
data/                          # Not committed - local data only
├── nik_gmail/                 # Your Gmail export
│   ├── takeout/
│   │   └── extracted/
│   │       └── All mail.mbox  # 7.7GB, 41k emails
│   ├── parsed_emails.jsonl    # Stage 2 output
│   ├── quality_report.json    # Stage 3 output
│   ├── cleaned_emails.jsonl   # Stage 4 output
│   └── enriched_emails.jsonl  # Stage 5 output

scripts/                       # Committed - reusable tools
├── parse_mbox.py             # MBOX → JSONL
├── analyze_data.py           # Generate quality report
├── clean_data.py             # Normalize and dedupe
├── enrich_data.py            # Compute action labels
├── import_to_postgres.py     # Load to database
└── validate_import.py        # End-to-end validation

db/
├── schema.sql                # PostgreSQL schema
└── queries/                  # Common queries
```

---

## Environment Setup

Copy `.env.example` to `.env` and configure your environment:

```bash
cp .env.example .env
# Edit .env with your values
```

Key environment variables:
- `DATABASE_URL` - PostgreSQL connection string
- `SURREALDB_GMAIL_URL` - SurrealDB Gmail database URL
- `OPENAI_API_KEY` - OpenAI API key (for embeddings/LLM features)
- `ANTHROPIC_API_KEY` - Anthropic API key (optional)

Scripts automatically load `.env` via `python-dotenv`.

---

## Database

**Connection:** Set via `DATABASE_URL` environment variable (default: `postgresql://postgres:postgres@localhost:5433/rl_emails`)

Using existing PostgreSQL 16 instance. pgvector will be added later for embeddings.

### Schema (4 tables)

See [SCHEMA.md](./SCHEMA.md) for full details.

| Table | Purpose |
|-------|---------|
| `emails` | Core email data (41k records) |
| `threads` | Thread aggregations, response patterns |
| `users` | Sender/recipient stats, reply rates |
| `attachments` | Attachment metadata |

### Key Fields for Enrichment

```sql
-- emails table enrichment fields
is_sent BOOLEAN         -- TRUE if you sent this email
action TEXT             -- REPLIED, FORWARDED, STARRED, IGNORED
timing TEXT             -- IMMEDIATE, SAME_DAY, LATER, NEVER
response_time_seconds   -- How fast you replied
in_reply_to TEXT        -- Links your reply to original email
```

### Enrichment Flow

1. Import all emails
2. Mark `is_sent = TRUE` for your email addresses
3. Match your sent emails to received via `in_reply_to → message_id`
4. Compute `action`, `timing`, `response_time_seconds`
5. Aggregate into `threads` and `users` tables

---

## Pipeline Stages

### Stage 1: Export (Manual)
**Status:** Complete

Export Gmail via Google Takeout. Ensure you get:
- All mail (inbox, sent, archive, trash)
- MBOX format with X-Gmail-Labels headers

### Stage 2: Parse MBOX
**Script:** `scripts/parse_mbox.py`

```bash
uv run python scripts/parse_mbox.py \
    --input data/nik_gmail/takeout/extracted/All\ mail.mbox \
    --output data/nik_gmail/parsed_emails.jsonl
```

Extracts:
- `message_id`, `thread_id` (X-GM-THRID)
- `labels` (X-Gmail-Labels)
- `from`, `to`, `cc`, `subject`, `date`
- `body` (text and HTML parts)
- `has_attachments`

### Stage 3: Analyze Quality
**Script:** `scripts/analyze_data.py`

```bash
uv run python scripts/analyze_data.py \
    --input data/nik_gmail/parsed_emails.jsonl \
    --output data/nik_gmail/quality_report.json
```

Generates:
- Total count, date range
- Label distribution
- Field coverage (% with each field)
- Red flags (missing sent folder, no labels, etc.)

### Stage 4: Clean & Normalize
**Script:** `scripts/clean_data.py`

```bash
uv run python scripts/clean_data.py \
    --input data/nik_gmail/parsed_emails.jsonl \
    --output data/nik_gmail/cleaned_emails.jsonl
```

Operations:
- Normalize dates to ISO 8601 UTC
- Deduplicate by message_id
- Fix encoding issues
- Extract email addresses from headers
- Filter spam/trash if needed

### Stage 5: Enrich with Actions
**Script:** `scripts/enrich_data.py`

```bash
uv run python scripts/enrich_data.py \
    --input data/nik_gmail/cleaned_emails.jsonl \
    --output data/nik_gmail/enriched_emails.jsonl \
    --your-email me@nik-patel.com
```

Computes:
- `action`: REPLIED (you sent reply), FORWARDED, STARRED, IGNORED
- `timing`: Based on response time
- `priority_score`: Weighted combination of signals

### Stage 6: Load to PostgreSQL
**Script:** `scripts/import_to_postgres.py`

```bash
uv run python scripts/import_to_postgres.py \
    --input data/nik_gmail/enriched_emails.jsonl \
    --db-url postgresql://postgres:postgres@localhost:5433/rl_emails \
    --create-schema
```

### Stage 7: Validate
**Script:** `scripts/validate_import.py`

```bash
uv run python scripts/validate_import.py \
    --source data/nik_gmail/enriched_emails.jsonl \
    --db-url postgresql://postgres:postgres@localhost:5433/rl_emails
```

Checks:
- Count matches
- Sample verification
- No null required fields
- Output: `READY_FOR_TRAINING: true/false`

---

## Processing Another MBOX

To process a different Gmail export:

```bash
# 1. Extract your takeout zip
unzip takeout-*.zip -d data/other_gmail/

# 2. Run the pipeline
export MBOX_FILE="data/other_gmail/Takeout/Mail/All mail.mbox"
export OUTPUT_DIR="data/other_gmail"
export YOUR_EMAIL="your@email.com"

uv run python scripts/parse_mbox.py --input "$MBOX_FILE" --output "$OUTPUT_DIR/parsed.jsonl"
uv run python scripts/analyze_data.py --input "$OUTPUT_DIR/parsed.jsonl" --output "$OUTPUT_DIR/quality.json"
uv run python scripts/clean_data.py --input "$OUTPUT_DIR/parsed.jsonl" --output "$OUTPUT_DIR/cleaned.jsonl"
uv run python scripts/enrich_data.py --input "$OUTPUT_DIR/cleaned.jsonl" --output "$OUTPUT_DIR/enriched.jsonl" --your-email "$YOUR_EMAIL"
uv run python scripts/import_to_postgres.py --input "$OUTPUT_DIR/enriched.jsonl" --db-url "$DB_URL"
uv run python scripts/validate_import.py --source "$OUTPUT_DIR/enriched.jsonl" --db-url "$DB_URL"
```

---

## Dependencies

```bash
uv pip install asyncpg mailbox python-dateutil tqdm alembic psycopg2-binary
```

---

## Database Migrations (Alembic)

We use [Alembic](https://alembic.sqlalchemy.org/) for versioned database schema changes.

### Setup

Alembic is already configured in this project:
- `alembic.ini` - Configuration (database URL, logging)
- `alembic/` - Migrations directory
- `alembic/versions/` - Migration scripts

### Common Commands

```bash
# Check current migration state
alembic current

# View migration history
alembic history

# Create a new migration
alembic revision -m "add_new_column"

# Apply all pending migrations
alembic upgrade head

# Rollback last migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade 92f0657c25ef
```

### For Existing Databases

If you have an existing database with the schema already created, stamp it to mark the initial migration as complete:

```bash
alembic stamp 92f0657c25ef
```

### For Fresh Databases

Run all migrations to create the schema:

```bash
alembic upgrade head
```

### Creating New Migrations

1. Make changes to the schema
2. Create a migration:
   ```bash
   alembic revision -m "descriptive_name"
   ```
3. Edit the generated file in `alembic/versions/`
4. Test the upgrade and downgrade
5. Commit the migration file

### Environment Variable Override

You can override the database URL with:
```bash
DATABASE_URL=postgresql://user:pass@host:port/db alembic upgrade head
```

---

## Current Data

| Metric | Value |
|--------|-------|
| Source | Google Takeout (Jan 2026) |
| File | `All mail Including Spam and Trash.mbox` |
| Size | 7.7 GB |
| Emails | ~41,377 |
| Has Labels | Yes (X-Gmail-Labels header) |
| Has Thread IDs | Yes (X-GM-THRID header) |
