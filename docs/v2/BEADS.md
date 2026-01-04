# Work Tracking - Beads Structure

## Pipeline Overview

```
Setup → Import → Gate1 → Analyze → Clean → Enrich → Gate2 → Gate3 → Validate
  │        │       │        │        │       │        │       │        │
  │        │       │        │        │       │        │       │        └─ Human review
  │        │       │        │        │       │        │       └─ Training ready?
  │        │       │        │        │       │        └─ Enrichment quality
  │        │       │        │        │       └─ Compute actions
  │        │       │        │        └─ Normalize data
  │        │       │        └─ Quality report
  │        │       └─ Verify raw import
  │        └─ Parse MBOX → raw_emails
  └─ Create database
```

---

## Bead Dependency Graph

```
rle-0uj: Setup PostgreSQL
    │
    └──→ rle-mp8: Parse MBOX → raw_emails
              │
              └──→ rle-zoa: [GATE 1] Verify raw import ⛔
                        │
                        └──→ rle-zbv: Analyze Data Quality
                                  │
                                  └──→ rle-pjd: Clean & Normalize
                                            │
                                            └──→ rle-8v3: Enrich with Actions
                                                      │
                                                      └──→ rle-5aw: [GATE 2] Verify enrichment ⛔
                                                                │
                                                                └──→ rle-zfh: [GATE 3] Training ready? ⛔
                                                                          │
                                                                          └──→ rle-eac: Final Validation
```

**⛔ Gates = Automated verification points. Must PASS before proceeding.**

---

## All Pipeline Beads

| Order | Bead | Type | Title | Blocked By |
|-------|------|------|-------|------------|
| 1 | `rle-0uj` | task | Setup PostgreSQL | - |
| 2 | `rle-mp8` | task | Parse MBOX → raw_emails | rle-0uj |
| 3 | `rle-zoa` | gate | **GATE: Verify raw import** | rle-mp8 |
| 4 | `rle-zbv` | task | Analyze Data Quality | rle-zoa |
| 5 | `rle-pjd` | task | Clean & Normalize | rle-zbv |
| 6 | `rle-8v3` | task | Enrich with Actions | rle-pjd |
| 7 | `rle-5aw` | gate | **GATE: Verify enrichment** | rle-8v3 |
| 8 | `rle-zfh` | gate | **GATE: Training readiness** | rle-5aw |
| 9 | `rle-eac` | task | Final Validation | rle-zfh |

---

## Gate Details

### Gate 1: Verify Raw Import (`rle-zoa`)
**After:** MBOX parsing
**Checks:**
- Count matches (41,377 expected)
- No duplicate message_ids
- Required fields present
- Labels coverage >90%
- Thread IDs coverage >80%

### Gate 2: Verify Enrichment (`rle-5aw`)
**After:** Action computation
**Checks:**
- All raw_emails have enriched record
- is_sent correctly marked
- Reply chains resolve
- Actions computed for >70% emails
- Response times plausible
- Threads/users tables populated

### Gate 3: Training Readiness (`rle-zfh`)
**After:** Enrichment verification
**Checks:**
- Sufficient labeled data (>10k)
- Balanced action distribution
- Can generate >5k preference pairs
- Human review sample generated

---

## Epic Structure

### `rle-7er`: Data Pipeline (Active)
All pipeline beads roll up to this epic.

### `rle-cj7`: Human Preference Pipeline
Blocked until data pipeline complete. Will use training-ready data.

### `rle-8pb`: Email RL Training System
End goal. Depends on quality data from pipeline.

---

## Scripts by Stage

| Stage | Script | Purpose |
|-------|--------|---------|
| 2 | `scripts/parse_mbox.py` | MBOX → raw_emails (parallel) |
| 3 | `scripts/verify_raw_import.py` | Gate 1 checks |
| 4 | `scripts/analyze_data.py` | Quality report |
| 5 | `scripts/clean_data.py` | Normalize |
| 6 | `scripts/enrich_data.py` | Compute actions (parallel) |
| 7 | `scripts/verify_enrichment.py` | Gate 2 checks |
| 8 | `scripts/verify_training_ready.py` | Gate 3 checks |
| 9 | `scripts/validate_final.py` | E2E validation |

---

## Quality Assurance Flow

```
Polecat completes stage
        │
        ▼
Witness reviews output
        │
        ▼
Automated gate runs ──→ FAIL? ──→ Polecat fixes
        │
        ▼ PASS
Refinery merges
        │
        ▼
Next stage unblocked
```

**Mayor coordinates:** Ensures polecats are assigned, gates pass, work progresses.
**Witness monitors:** Checks polecat work quality before gate runs.
**Refinery merges:** Only after gate PASS.

---

## Commands

```bash
# View pipeline status
bd list --status=open | grep rle-

# Check what's blocked
bd blocked

# View gate details
bd show rle-zoa   # Gate 1
bd show rle-5aw   # Gate 2
bd show rle-zfh   # Gate 3

# Check ready work
bd ready
```

---

## Execution Plan

### Phase 1: Setup + Import
1. `rle-0uj` - Create rl_emails database
2. `rle-mp8` - Parse 41k emails (parallel workers)
3. `rle-zoa` - **GATE 1**: Verify import

### Phase 2: Analyze + Clean
4. `rle-zbv` - Generate quality report
5. `rle-pjd` - Normalize data

**CHECKPOINT: Review quality report**

### Phase 3: Enrich + Verify
6. `rle-8v3` - Compute actions (parallel workers)
7. `rle-5aw` - **GATE 2**: Verify enrichment
8. `rle-zfh` - **GATE 3**: Training readiness

**CHECKPOINT: Review human sample**

### Phase 4: Final Validation
9. `rle-eac` - End-to-end validation

**READY FOR ML TRAINING**
