# Experiment Automation Design

**Status:** Design Complete - Ready for Implementation
**Date:** 2025-11-05
**Priority:** High (eliminates manual intervention in future runs)

## Problem Statement

### What Happened in Nov 2025 Run

The experiment that validated exp_p3 (860 evaluations) required **manual intervention**:

1. ✅ Ran `run_paper_complete_end_to_end.sh` → created baselines + ensembles + selection tokenizers
2. ✅ Evaluated on TEST, OOS, FineWeb domains → **automatic**
3. ❌ The Stack evaluation ran → **only found baseline/selection tokenizers** (merge tokenizers missing)
4. ❌ **Manual step:** Created `eval_thestack_merge.sh` to evaluate 146 merge tokenizers
5. ❌ **Manual step:** Ran separate Stack evaluation for merge tokenizers
6. ❌ **Manual step:** Combined results from multiple files

### Root Cause

**Merge tokenizers (exp_p2, exp_p3, sequential, etc.) were NOT created automatically**

Current pipeline only creates:
- ✅ Baselines (automatic)
- ✅ Ensembles (automatic)
- ✅ Selection tokenizer (automatic)
- ❌ Merge majority (missing)
- ❌ Weighted θ=0.3, 0.7 (missing)
- ❌ Sequential (missing)
- ❌ Exponential p=2, p=3 (missing)

**Missing phase:** Automatic merge creation between training and evaluation

---

## Solution Architecture

### 5-Phase Automated Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: SETUP                                             │
│  - Data preparation                                         │
│  - Create manifest.json (registry)                          │
│  - Validate configuration                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│  PHASE 2: TRAINING                                          │
│  - Train baselines (1 per seed/vocab)                       │
│  - Train ensembles (K members per config)                   │
│  - Update manifest with paths                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│  PHASE 3: MERGE CREATION ⭐ NEW AUTOMATED PHASE             │
│  - For each ensemble, create ALL merge variants:            │
│    1. Selection (already exists)                            │
│    2. Merge majority                                        │
│    3. Weighted θ=0.3                                        │
│    4. Weighted θ=0.7                                        │
│    5. Sequential voting                                     │
│    6. Exponential p=2                                       │
│    7. Exponential p=3                                       │
│  - Update manifest                                          │
│  - Skip if exists (idempotent)                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│  PHASE 4: EVALUATION                                        │
│  - Discover ALL tokenizers from manifest                    │
│  - Evaluate on ALL domains:                                 │
│    * TEST                                                   │
│    * OOS                                                    │
│    * FineWeb                                                │
│    * The Stack                                              │
│  - Parallel workers (4+)                                    │
│  - Skip completed (idempotent)                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│  PHASE 5: AGGREGATION & VERIFICATION                        │
│  - Collect all results                                      │
│  - Verify completeness                                      │
│  - Generate unified tables                                  │
│  - Report missing evaluations                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Innovation: Manifest-Based Registry

**File:** `<experiment_dir>/manifest.json`

**Purpose:**
- Track every tokenizer created (baseline, ensemble, merged)
- Track every evaluation completed (tokenizer × domain)
- Enable idempotency (skip completed work)
- Enable verification (detect missing pieces)
- Enable resumption (continue after failure)

**Schema:**
```json
{
  "experiment_id": "paper_run_20251105_143022",
  "status": "in_progress",
  "phases": {
    "training": {
      "status": "completed",
      "tokenizers": {
        "baselines": ["s13_v16k", "s17_v16k", ...],
        "ensembles": ["s13_v16k_k4", "s17_v16k_k8", ...]
      }
    },
    "merge_creation": {
      "status": "in_progress",
      "merged": {
        "s13_v16k_k4_exp_p3": {
          "status": "completed",
          "path": "tokenizers/merged/s13_v16k_k4_exp_p3",
          "method": "exponential_p3"
        }
      }
    },
    "evaluation": {
      "status": "pending",
      "completed": {
        "test": ["s13_v16k_baseline", "s13_v16k_k4_exp_p3"],
        "oos": ["s13_v16k_baseline"],
        "fineweb": [],
        "thestack": []
      }
    }
  },
  "expected_counts": {
    "baselines": 10,
    "ensembles": 10,
    "merged": 70,
    "evaluations": 320
  }
}
```

---

## Implementation Plan

### Minimal Viable Automation (Quick Win)

**Goal:** 80% benefit with 20% effort
**Time:** 1-2 days

**Three critical scripts:**

#### 1. `scripts/create_all_merges.py`
```python
"""
Batch create all merge tokenizers for all ensembles.

Usage:
  uv run python -m scripts.create_all_merges \\
    --out /nas4/data/experiments/ensemble-bpe \\
    --methods exp_p2 exp_p3 sequential merge_majority weighted_30 weighted_70
"""
```

**Features:**
- Discovers all ensemble directories
- Creates all 7 merge methods for each
- Skips if tokenizer.json already exists
- Updates manifest (if exists)

#### 2. `scripts/evaluate_all_domains.py`
```python
"""
Evaluate all tokenizers on all domains with parallelization.

Usage:
  uv run python -m scripts.evaluate_all_domains \\
    --out /nas4/data/experiments/ensemble-bpe \\
    --domains test oos fineweb thestack \\
    --workers 4
"""
```

**Features:**
- Discovers all tokenizer directories
- Evaluates on each domain
- Parallel worker pool
- Skips completed evaluations
- Progress tracking

#### 3. `scripts/verify_experiment.py`
```python
"""
Verify experiment completeness and report missing evaluations.

Usage:
  uv run python -m scripts.verify_experiment \\
    --out /nas4/data/experiments/ensemble-bpe
"""
```

**Features:**
- Counts expected vs actual tokenizers
- Lists missing merge tokenizers
- Lists missing evaluations
- Reports failures
- Actionable output

**Usage Flow:**
```bash
# After training completes:

# 1. Create all merge variants (NEW - eliminates manual step)
uv run python -m scripts.create_all_merges --out <experiment_dir>

# 2. Evaluate everything on all domains (automatic)
uv run python -m scripts.evaluate_all_domains \\
  --out <experiment_dir> \\
  --domains test oos fineweb thestack \\
  --workers 4

# 3. Verify nothing was missed
uv run python -m scripts.verify_experiment --out <experiment_dir>
```

### Full Manifest-Based System (Complete Solution)

**Time:** 1-2 weeks
**Includes:**

#### 4. `scripts/run_paper_automated.py`
Master orchestrator with manifest management:
```bash
# Single command for complete automation
uv run python -m scripts.run_paper_automated \\
  --seeds 13 17 19 \\
  --vocab 16384 32768 \\
  --k 4 8 \\
  --out artifacts/experiments/paper_final

# Resume after interruption
uv run python -m scripts.run_paper_automated \\
  --resume artifacts/experiments/paper_final
```

#### 5. `scripts/aggregate_experiment.py`
Unified result aggregation:
```bash
uv run python -m scripts.aggregate_experiment \\
  --experiment-dir <dir> \\
  --output results/
```

#### 6. Manifest utilities (`src/ebpe/manifest.py`)
- Schema definition
- Load/save functions
- Status update utilities
- Verification logic

---

## Benefits

### ✅ Zero Manual Intervention
- Single command starts complete experiment
- All merge tokenizers created automatically
- All domains evaluated automatically
- Results automatically aggregated

### ✅ Idempotent Execution
- Can re-run without duplicating work
- Skips completed tokenizers
- Skips completed evaluations
- Safe to interrupt and resume

### ✅ Verification Built-In
- Detects missing tokenizers
- Detects missing evaluations
- Reports incomplete work
- Actionable output

### ✅ Robust Error Handling
- Continues on individual failures
- Marks failed items in manifest
- Retry mechanism with backoff
- Checkpoint/resume system

### ✅ Parallel Execution
- Multiple evaluation workers
- Configurable parallelism
- Progress tracking
- Estimated time remaining

---

## Expected Counts (Reference)

For configuration: 5 seeds × 2 vocab × 2 K = 20 configs

### Tokenizers Created
- **Baselines:** 10 (5 seeds × 2 vocab)
- **Ensembles:** 20 (5 seeds × 2 vocab × 2 K)
- **Merged per ensemble:** 7 methods
- **Total merged:** 20 × 7 = 140
- **TOTAL TOKENIZERS:** 10 + 20 + 140 = **170**

### Evaluations Performed
- **Tokenizers:** 170
- **Domains:** 4 (TEST, OOS, FineWeb, The Stack)
- **TOTAL EVALUATIONS:** 170 × 4 = **680**

**Note:** Current Nov 2025 run was:
- 20 configs
- Only baselines (20) + ensembles (20) + some merged (100) = 140 tokenizers
- 4 domains
- 140 × 4 = 560 evaluations (but had to run Stack separately)

---

## Testing Strategy

### Unit Tests
- Manifest creation/loading
- Tokenizer discovery
- Skip logic (idempotency)
- Verification counts

### Integration Tests
```python
def test_minimal_automation():
    """Test create_all_merges + evaluate_all_domains + verify"""
    # 1 seed, 1 vocab, K=2
    # Expect: 1 baseline + 1 ensemble + 7 merged = 9 tokenizers
    # Expect: 9 × 2 domains = 18 evaluations
    pass

def test_resume_after_interruption():
    """Test resumption works correctly"""
    # Create half the merges
    # Interrupt
    # Resume
    # Verify all created
    pass
```

### Scale Test
```python
def test_20_config_automation():
    """Test full 20-config run with automation"""
    # 5 seeds × 2 vocab × 2 K
    # Verify 170 tokenizers created
    # Verify 680 evaluations completed
    pass
```

---

## Migration Path

### Current State (checkpoint-20251105)
- ✅ Training pipeline works
- ✅ Evaluation works for discovered tokenizers
- ❌ Merge creation requires manual step
- ❌ The Stack evaluation required separate run
- ❌ No verification of completeness

### After Minimal Automation (1-2 days)
- ✅ Merge creation automated
- ✅ All domains evaluated automatically
- ✅ Completeness verification available
- ⚠️  Still requires running 3 commands
- ⚠️  No manifest system yet

### After Full System (1-2 weeks)
- ✅ Single command for everything
- ✅ Manifest-based tracking
- ✅ Automatic verification
- ✅ Resume after failures
- ✅ Production-ready automation

---

## File Structure

### Minimal Automation Scripts
```
scripts/
├── create_all_merges.py          # NEW - batch merge creation
├── evaluate_all_domains.py       # NEW - parallel domain evaluation
└── verify_experiment.py          # NEW - completeness checking
```

### Full Automation (Future)
```
src/ebpe/manifest.py               # NEW - manifest utilities
scripts/
├── run_paper_automated.py         # NEW - master orchestrator
├── aggregate_experiment.py        # NEW - unified aggregation
├── create_all_merges.py          # (as above)
├── evaluate_all_domains.py       # (as above)
└── verify_experiment.py          # (as above)
```

---

## Priority & Timing

### For Current Research (Nov 2025)
- ✅ Results validated and documented in RESULTS.md
- ✅ Checkpoint saved to GitHub (checkpoint-20251105)
- ✅ Paper writing can proceed
- ℹ️  Automation can wait until next experiment run

### For Future Experiments
- **High priority:** Implement minimal automation (1-2 days)
- **Medium priority:** Full manifest system (when running larger scale)

### Recommendation
- Complete minimal automation **before next experiment run**
- Benefits: No manual intervention, verified completeness
- Cost: 1-2 days development + testing

---

## Success Criteria

### Minimal Automation Success
- ✅ `create_all_merges.py` creates all 7 merge variants
- ✅ `evaluate_all_domains.py` evaluates all tokenizers on all domains
- ✅ `verify_experiment.py` reports missing evaluations
- ✅ No manual intervention required
- ✅ Integration test passes for 1 seed × 1 vocab × K=2

### Full System Success
- ✅ Single command runs complete experiment
- ✅ Resumes after interruption
- ✅ Handles 1000+ tokenizers
- ✅ Parallel evaluation (4+ workers)
- ✅ Automatic verification
- ✅ Scale test passes for 20 configs

---

## References

- **Design Document:** This file (AUTOMATION.md)
- **Current Implementation:** `scripts/run_paper_complete_end_to_end.sh`
- **Merge Creation:** `scripts/merge_ensemble.py` (needs batch wrapper)
- **Evaluation Examples:** `scripts/eval_thestack.py`, `scripts/eval_fineweb.py`
- **Manual Fix (Nov 2025):** `eval_thestack_merge.sh` (temporary script created to fill gap)

---

## Related Documentation

- **AGENTS.md:** Agent instructions referencing this automation design
- **README.md:** User-facing documentation for running experiments
- **TODO.md:** Implementation tasks tracked in project TODO
- **RESULTS.md:** Validated experimental results from Nov 2025 run
