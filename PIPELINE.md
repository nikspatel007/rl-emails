# Email ML Pipeline

This document describes how to run the complete email analysis pipeline from a Gmail MBOX export.

## Prerequisites

### 1. PostgreSQL with pgvector

Start the database:

```bash
docker compose up -d postgres
# Or use the helper script:
./scripts/start_db.sh
```

The database should be accessible at `localhost:5433` with credentials `postgres:postgres`.

### 2. Python Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. OpenAI API Key (Optional)

For embedding generation, set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

If not set, the pipeline will skip embedding-based features.

## Quick Start

```bash
# Run the complete pipeline
python run_pipeline.py /path/to/your.mbox

# With custom data directory
python run_pipeline.py /path/to/your.mbox --data-dir ./my_data

# Skip embedding generation (if no OpenAI key)
python run_pipeline.py /path/to/your.mbox --skip-embeddings

# Resume from a specific step (if pipeline failed)
python run_pipeline.py /path/to/your.mbox --start-from 5
```

## Pipeline Stages

### Stage 1: Parse MBOX (`scripts/parse_mbox.py`)

Parses the Gmail MBOX export into a normalized JSONL format.

**Input:** Gmail MBOX file from Google Takeout
**Output:** `data/gmail/parsed_emails.jsonl`

```bash
# Manual run with custom paths
python scripts/parse_mbox.py \
    --mbox /path/to/your.mbox \
    --output ./data/gmail/parsed_emails.jsonl
```

### Stage 2: Import to PostgreSQL (`scripts/import_to_postgres.py`)

Imports parsed emails into PostgreSQL tables:
- `raw_emails` - Immutable source data
- `emails` - Enriched/derived data with indexes

**Input:** `parsed_emails.jsonl`
**Output:** Database tables populated

```bash
python scripts/import_to_postgres.py \
    --input ./data/gmail/parsed_emails.jsonl
```

### Stage 3: Populate Threads (`scripts/populate_threads.py`)

Builds thread relationships using In-Reply-To and References headers.

**Input:** Emails in database
**Output:** `threads` table, email `thread_id` populated

```bash
python scripts/populate_threads.py
```

### Stage 4: Generate Embeddings (`scripts/generate_embeddings.py`)

Creates OpenAI embeddings (text-embedding-3-small, 1536 dims) for semantic search.

**Requires:** `OPENAI_API_KEY` environment variable
**Input:** Emails in database
**Output:** `email_embeddings` table with pgvector

```bash
python scripts/generate_embeddings.py
```

### Stage 5: Mine Gmail Labels (`scripts/mine_gmail_labels.py`)

Extracts projects from Gmail labels (user-created and Superhuman AI labels).

**Input:** Emails with labels in database
**Output:** Projects in `projects` table with `source='gmail_label'`

```bash
python scripts/mine_gmail_labels.py [--dry-run]
```

### Stage 6: Discover Participant Projects (`scripts/discover_participant_projects.py`)

Finds recurring participant groups (people who email together frequently).

**Input:** Emails in database
**Output:** Projects in `projects` table with `source='participant'`

```bash
python scripts/discover_participant_projects.py [--dry-run] [--min-emails 5]
```

### Stage 7: Cluster Embeddings (`scripts/cluster_embeddings.py`)

Clusters emails by semantic similarity using KMeans on embeddings.

**Requires:** Embeddings generated in Stage 4
**Input:** Embeddings in database
**Output:** Projects in `projects` table with `source='cluster'`

```bash
python scripts/cluster_embeddings.py [--n-clusters 30] [--dry-run]
```

### Stage 8: Dedupe Projects (`scripts/dedupe_projects.py`)

Merges duplicate projects based on participant overlap and name similarity.

**Input:** Projects from Stages 5-7
**Output:** `merged_into` field set on duplicate projects

```bash
python scripts/dedupe_projects.py [--dry-run]
```

### Stage 9: Detect Priority Contexts (`scripts/detect_priority_contexts.py`)

Analyzes response times to find periods of heightened engagement.

**Input:** Emails with response times
**Output:** `priority_contexts` table populated

```bash
python scripts/detect_priority_contexts.py [--dry-run]
```

## Post-Pipeline: Labeling UI

After the pipeline completes, start the labeling UI to manually classify tasks:

```bash
streamlit run apps/labeling_ui.py
```

The UI allows you to:
- View emails with extracted tasks
- See auto-detected project associations
- Confirm or correct project assignments
- Rate task triage categories (FYI, Quick Win, AI Doable, etc.)
- Store labels for model training

## Database Schema

Key tables created by the pipeline:

| Table | Description |
|-------|-------------|
| `raw_emails` | Immutable source data from MBOX |
| `emails` | Enriched email data with indexes |
| `threads` | Conversation threading |
| `email_embeddings` | OpenAI embeddings (pgvector) |
| `projects` | Discovered projects/topics |
| `email_project_links` | Email-to-project associations |
| `priority_contexts` | Detected high-engagement periods |
| `human_task_labels` | Manual labels for training |

## Configuration

The pipeline uses these default settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `DB_URL` | `postgresql://postgres:postgres@localhost:5433/rl_emails` | Database connection |
| `BATCH_SIZE` | 1000 | Records per database batch |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI model |
| `N_CLUSTERS` | 30 | Number of semantic clusters |
| `MIN_EMAILS_PER_PROJECT` | 3 | Minimum emails for a participant project |

Override via environment variables or script arguments.

## Troubleshooting

### Database connection failed

```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check connection
psql -h localhost -p 5433 -U postgres -d rl_emails -c "SELECT 1"
```

### Out of memory during embedding generation

Reduce batch size:

```bash
python scripts/generate_embeddings.py --batch-size 50
```

### Pipeline failed mid-run

Resume from the failed step:

```bash
python run_pipeline.py /path/to/your.mbox --start-from 5
```

### No embeddings / clustering skipped

Ensure `OPENAI_API_KEY` is set:

```bash
export OPENAI_API_KEY="sk-..."
python run_pipeline.py /path/to/your.mbox
```

## Development

To run individual scripts during development:

```bash
# Activate environment
source .venv/bin/activate

# Run any script
python scripts/mine_gmail_labels.py --dry-run
```

## License

MIT License - See LICENSE file.
