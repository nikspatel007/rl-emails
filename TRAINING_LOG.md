# Training Log

## Run: 2026-01-04 (Enron Full Pipeline)

### Data Pipeline
- **Source**: Enron email dataset (CMU mirror)
- **Raw emails**: 517,401
- **Labeled (actionable)**: 199,336
- **Splits**:
  - Train: 140,000 (112 users)
  - Val: 32,000 (19 users)
  - Test: 27,169 (19 users)

### Action Distribution (5-class)
| Action | Train | Val | Test |
|--------|-------|-----|------|
| archive | 63% | 35% | 37% |
| reply_later | 25% | 33% | 33% |
| delete | 9% | 23% | 23% |
| forward | 3% | 9% | 7% |
| reply_now | 0% | 0% | 0% |

**Note**: `reply_now` is 0% because response-time matching failed for most replies
(Enron has sparse In-Reply-To/References headers).

### Results
| Stage | Val Accuracy | Test Accuracy | Notes |
|-------|--------------|---------------|-------|
| Baseline | - | 1.59% | Random init, predicts reply_now |
| SFT (vanilla) | 96.8% | 37.48% | Mode collapse → archive |
| SFT (class weights) | 48.6% | 53.14% | Balanced predictions, +15.7pp |

#### Per-class F1 Scores (Class Weighted SFT)
| Action | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| reply_later | 0.609 | 0.653 | 0.630 |
| forward | 0.204 | 0.696 | 0.316 |
| archive | 0.727 | 0.347 | 0.469 |
| delete | 0.596 | 0.611 | 0.603 |

### Artifacts (gitignored, local only)
```
data/
├── maildir/           # Raw Enron emails (517K files)
├── emails.json        # Parsed emails (517K)
├── emails_labeled.json # Labeled (199K)
├── train.json         # Training split (140K)
├── val.json           # Validation split (32K)
├── test.json          # Test split (27K)
└── split_metadata.json

checkpoints/
├── stage_1.pt         # SFT final checkpoint
├── best_sft.pt        # Best validation checkpoint
└── sft_epoch_*.pt     # Per-epoch checkpoints
```

### Issues Identified (Resolved)
1. **Class imbalance**: archive dominates training (63%) -> Fixed with class weighting
2. **Distribution shift**: train vs test distributions differ significantly -> Mitigated by class weights
3. **Mode collapse**: model learns to predict majority class -> Fixed with inverse frequency weighting
4. **Label mapping bug**: ACTION_TO_IDX was missing REPLY_LATER, ARCHIVE, DELETE -> Fixed
5. **No reply_now**: response-time matching failed (data issue, not fixable)

### Fixes Applied
- [x] Add class weighting to SFT loss (inverse frequency weighting)
- [x] Implement focal loss option (alternative to class weighting)
- [x] Add balanced sampling option (WeightedRandomSampler)
- [x] Fix ACTION_TO_IDX label mapping for training data

### Next Steps
- [ ] Run GRPO with reward shaping (Stage 3)
- [ ] Try Gmail data (better threading headers for reply_now)
- [ ] Experiment with focal loss + balanced sampling combination
- [ ] Stage 4: DPO training
