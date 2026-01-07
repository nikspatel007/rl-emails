# Repository Cleanup Plan

**Status: COMPLETED** - 2026-01-07

The onboarding pipeline (`onboard_data.py`) is production-ready. Cleanup has been executed.

---

## Summary of Changes

### Files Removed (Phase 1)
- `run_pipeline.py` - Legacy pipeline (superseded by `onboard_data.py`)
- `inference.py` - Orphaned inference script
- `scripts/compute_features.py` - Duplicate of `compute_basic_features.py`
- `scripts/extract_features.py` - Duplicate functionality
- `scripts/labeling_ui.py`, `scripts/labeling_ui_v2.py`, `apps/labeling_ui.py` - Duplicates
- `scripts/enrich_emails.py`, `scripts/enrich_emails_db.py` - Old enrichment
- `scripts/cluster_embeddings.py`, `scripts/cluster_emails.py`, `scripts/dedupe_projects.py` - Legacy clustering
- `scripts/update_in_reply_to.py`, `scripts/filter_mbox_by_date.py`, `scripts/restore_embeddings.py` - One-off utilities
- `haiku_features.json`, `haiku_100_features.json`, `ollama_features.json`, `phi4_features.json`, `test_tasks.json` - Benchmark files

### Files Archived (Phase 2)
Moved to `archive/` subdirectories:

**archive/benchmarks/**
- `test_phase1_llm.py`, `test_haiku_parallel.py`, `test_ollama_parallel.py`, `test_lm_studio_tasks.py`, `test_phi4_oneshot.py`

**archive/training/**
- `train_ensemble.py`, `evaluate_ensemble.py`, `infer.py`

**archive/verification/**
- `validate_e2e.py`, `verify_enrichment.py`, `verify_raw_import.py`, `verify_training_ready.py`

**archive/mining/**
- `mine_gmail_labels.py`, `discover_participant_projects.py`, `download_mailing_list.py`

**archive/analysis/**
- `analyze_emails.py`, `analyze_data_quality.py`, `clean_emails.py`

**archive/detection/**
- `detect_priority_contexts.py`, `detect_participant_bursts.py`, `detect_recurring_patterns.py`

**archive/populate/**
- `populate_tasks.py`, `populate_users.py`

**archive/legacy/**
- `generate_embeddings.py`, `compute_priority.py`, `compute_priority_contexts.py`
- `create_splits.py`, `create_splits_gmail.py`
- `extract_attachments.py`, `extract_implicit_preferences.py`, `extract_llm_features.py`
- `generate_sample_gmail_data.py`, `run_rf_predictions.py`, `validate_features.py`

### Documentation Consolidated (Phase 3)
- Moved `docs/01-06*.md` (v1 docs) to `docs/archive/`
- Moved `docs/v2/BEADS.md`, `docs/v2/PHASE*_PROPOSAL.md` to `docs/archive/`
- Moved remaining `docs/v2/*` up to `docs/`

---

## Current Scripts Directory

After cleanup, `scripts/` contains only **10 essential files**:

```
scripts/
├── onboard_data.py          # Main orchestrator
├── parse_mbox.py            # Stage 1: MBOX → JSONL
├── import_to_postgres.py    # Stage 2: Import to DB
├── populate_threads.py      # Stage 3: Thread relationships
├── compute_basic_features.py # Stage 4: ML features
├── compute_embeddings.py    # Stage 5: Embeddings
├── classify_ai_handleability.py # Stage 6: Rule-based
├── run_llm_classification.py # Stage 7: LLM classification
├── checkpoint.py            # Utility: Checkpoint/restore
└── query_db.py              # Utility: Database queries
```

---

## Validation Results

Pipeline executed successfully on 1-month MBOX (1,925 emails):

```
============================================================
PIPELINE COMPLETED in 0:00:13.380684

Emails imported:        1,795
ML features computed:   1,795
Embeddings generated:   1,795
Rule-based classified:  1,795
LLM classified:         781
============================================================
```

Report generated at `data/onboarding/onboarding_report.md`.

---

## What's Still Kept

### Production Code
- `src/` - All 18 feature modules, policy network, reward, metrics
- `tests/` - All 18 test files
- `alembic/` - Database migrations
- `apps/labeling_ui_v2.py` - Canonical labeling UI
- `db/` - SurrealDB (optional)

### Documentation
- `README.md`
- `PIPELINE.md`
- `docs/onboarding.md`
- `docs/SCHEMA.md`, `docs/DATA_PIPELINE.md`, etc.
- `CLAUDE.md`

### Data/Models
- `data/` - Email datasets
- `models/` - Trained models
- `checkpoints/` - Database snapshots
- `eval_results/` - Evaluation history

---

## Future Cleanup (Optional)

### Large File Cleanup (~10-12GB)
If disk space is needed:
1. Review `checkpoints/` - keep only most recent
2. Review `backups/` - archive to external storage
3. Document active models in `models/`

### Archive Cleanup
The `archive/` directory can be deleted entirely if those scripts are confirmed unnecessary after a few weeks of production use.
