#!/bin/bash
################################################################################
# COMPLETE END-TO-END PAPER EXPERIMENT RUNNER
################################################################################
#
# This script generates 100% of the data needed for the paper from scratch:
# 1. Train baseline BPE on full corpus
# 2. Train ensemble members on shards
# 3. Generate ALL merge methods (selection, majority, weighted, sequential, exponential)
# 4. Evaluate ALL on TEST + OOS + FineWeb
# 5. Aggregate results to CSV/JSON tables
#
# Parameters:
#   --seeds: Space-separated seeds (default: 13)
#   --vocab: Space-separated vocab sizes (default: 16384 32768)
#   --k: Space-separated K values (default: 4 16)
#   --data-dir: Path to prepared data (default: data/processed/pg_paper_full)
#   --out: Output base directory (default: artifacts)
#   --skip-fineweb: Skip FineWeb evaluation (faster testing)
#
# Usage:
#   # Quick test (1 seed, 1 config)
#   ./scripts/run_paper_complete_end_to_end.sh --seeds 13 --vocab 16384 --k 4
#
#   # Full paper run
#   ./scripts/run_paper_complete_end_to_end.sh --seeds 13 17 19
#
################################################################################

set -euo pipefail

# Default configuration
SEEDS=(13)
VOCAB_SIZES=(16384 32768)
K_VALUES=(4 16)
DATA_DIR="data/processed/pg_paper_full"
OUT_DIR="artifacts"
SKIP_FINEWEB=false
FINEWEB_SAMPLES=1000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            shift
            SEEDS=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                SEEDS+=("$1")
                shift
            done
            ;;
        --vocab)
            shift
            VOCAB_SIZES=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                VOCAB_SIZES+=("$1")
                shift
            done
            ;;
        --k)
            shift
            K_VALUES=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                K_VALUES+=("$1")
                shift
            done
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --out)
            OUT_DIR="$2"
            shift 2
            ;;
        --skip-fineweb)
            SKIP_FINEWEB=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Verify data directory exists
if [[ ! -d "$DATA_DIR" ]]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    echo "Please prepare data first with:"
    echo "  uv run python -m scripts.prepare_hf_books ..."
    exit 1
fi

# Verify required files exist
for file in train.txt test.txt test_oos.txt; do
    if [[ ! -f "$DATA_DIR/$file" ]]; then
        echo "❌ Error: Required file not found: $DATA_DIR/$file"
        exit 1
    fi
done

echo "================================================================================"
echo "ENSEMBLE BPE PAPER: COMPLETE END-TO-END RUN"
echo "================================================================================"
echo "Seeds: ${SEEDS[*]}"
echo "Vocab sizes: ${VOCAB_SIZES[*]}"
echo "K values: ${K_VALUES[*]}"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUT_DIR"
echo "Skip FineWeb: $SKIP_FINEWEB"
echo ""
echo "Total configurations: $((${#SEEDS[@]} * ${#VOCAB_SIZES[@]} * ${#K_VALUES[@]}))"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUT_DIR"

# Track all experiment directories for later aggregation
BASELINE_DIRS=()
ENSEMBLE_DIRS=()
MERGE_DIRS=()

################################################################################
# PHASE 1: Train baselines and ensembles
################################################################################

echo ""
echo "================================================================================"
echo "PHASE 1: Training Baselines and Ensembles"
echo "================================================================================"
echo ""

for seed in "${SEEDS[@]}"; do
    for vocab in "${VOCAB_SIZES[@]}"; do
        for k in "${K_VALUES[@]}"; do
            echo "────────────────────────────────────────────────────────────────"
            echo "Config: seed=$seed, vocab=$vocab, K=$k"
            echo "────────────────────────────────────────────────────────────────"

            # 1. Train baseline (single BPE on full corpus)
            echo "  [1/3] Training baseline..."
            BASELINE_RUN=$(uv run python -m scripts.train_tokenizer \
                --exp "paper_e2e_s${seed}_v${vocab}_baseline" \
                --train "$DATA_DIR/train.txt" \
                --vocab-size "$vocab" \
                --out "$OUT_DIR" 2>&1 | tail -1)
            BASELINE_DIRS+=("$BASELINE_RUN")
            echo "    ✓ $BASELINE_RUN"

            # 2. Train ensemble
            echo "  [2/3] Training ensemble (K=$k)..."
            ENSEMBLE_RUN=$(uv run python -m scripts.train_ensemble \
                --exp "paper_e2e_s${seed}_v${vocab}_k${k}" \
                --train "$DATA_DIR/train.txt" \
                --eval-file "$DATA_DIR/test.txt" \
                --num-shards "$k" \
                --vocab-size "$vocab" \
                --out "$OUT_DIR" 2>&1 | tail -1)
            ENSEMBLE_DIRS+=("$ENSEMBLE_RUN")
            echo "    ✓ $ENSEMBLE_RUN"

            # 3. Selection is automatic (done by train_ensemble)
            echo "  [3/3] Selection completed (automatic)"
            echo ""
        done
    done
done

echo "✅ Phase 1 complete: ${#BASELINE_DIRS[@]} baselines, ${#ENSEMBLE_DIRS[@]} ensembles"
echo ""

################################################################################
# PHASE 2: Generate merge methods
################################################################################

echo ""
echo "================================================================================"
echo "PHASE 2: Generating Merge Methods"
echo "================================================================================"
echo ""

for ens_dir in "${ENSEMBLE_DIRS[@]}"; do
    ENS_NAME=$(basename "$ens_dir")
    echo "Processing: $ENS_NAME"

    # Extract config from name
    if [[ "$ENS_NAME" =~ _s([0-9]+)_v([0-9]+)_k([0-9]+) ]]; then
        SEED="${BASH_REMATCH[1]}"
        VOCAB="${BASH_REMATCH[2]}"
        K="${BASH_REMATCH[3]}"
    else
        echo "  ⚠️  Cannot parse config from name, skipping"
        continue
    fi

    # 1. Merge majority
    echo "  [1/6] Merge majority..."
    MERGE_DIR=$(uv run python -m scripts.merge_ensemble \
        --exp "paper_e2e_s${SEED}_v${VOCAB}_k${K}_merge_majority" \
        --ensemble-run "$ens_dir" \
        --method weighted \
        --k $(( (K + 1) / 2 )) \
        --out "$OUT_DIR" 2>&1 | tail -1)
    MERGE_DIRS+=("$MERGE_DIR")

    # 2. Weighted entropy θ=0.3
    echo "  [2/6] Weighted entropy θ=0.3..."
    MERGE_DIR=$(uv run python -m scripts.merge_ensemble \
        --exp "paper_e2e_s${SEED}_v${VOCAB}_k${K}_weighted_30" \
        --ensemble-run "$ens_dir" \
        --method weighted \
        --weight-by quality \
        --theta 0.3 \
        --out "$OUT_DIR" 2>&1 | tail -1)
    MERGE_DIRS+=("$MERGE_DIR")

    # 3. Weighted entropy θ=0.7
    echo "  [3/6] Weighted entropy θ=0.7..."
    MERGE_DIR=$(uv run python -m scripts.merge_ensemble \
        --exp "paper_e2e_s${SEED}_v${VOCAB}_k${K}_weighted_70" \
        --ensemble-run "$ens_dir" \
        --method weighted \
        --weight-by quality \
        --theta 0.7 \
        --out "$OUT_DIR" 2>&1 | tail -1)
    MERGE_DIRS+=("$MERGE_DIR")

    # 4. Sequential voting ⭐
    echo "  [4/6] Sequential voting ⭐..."
    MERGE_DIR=$(uv run python -m scripts.merge_ensemble \
        --exp "paper_e2e_s${SEED}_v${VOCAB}_k${K}_sequential" \
        --ensemble-run "$ens_dir" \
        --method sequential \
        --weight-by quality \
        --out "$OUT_DIR" 2>&1 | tail -1)
    MERGE_DIRS+=("$MERGE_DIR")

    # 5. Exponential p=2 ⭐
    echo "  [5/6] Exponential p=2 ⭐..."
    MERGE_DIR=$(uv run python -m scripts.merge_ensemble \
        --exp "paper_e2e_s${SEED}_v${VOCAB}_k${K}_exp_p2" \
        --ensemble-run "$ens_dir" \
        --method exponential \
        --weight-by quality \
        --theta 0.3 \
        --power 2.0 \
        --out "$OUT_DIR" 2>&1 | tail -1)
    MERGE_DIRS+=("$MERGE_DIR")

    # 6. Exponential p=3 ⭐
    echo "  [6/6] Exponential p=3 ⭐..."
    MERGE_DIR=$(uv run python -m scripts.merge_ensemble \
        --exp "paper_e2e_s${SEED}_v${VOCAB}_k${K}_exp_p3" \
        --ensemble-run "$ens_dir" \
        --method exponential \
        --weight-by quality \
        --theta 0.3 \
        --power 3.0 \
        --out "$OUT_DIR" 2>&1 | tail -1)
    MERGE_DIRS+=("$MERGE_DIR")

    echo ""
done

echo "✅ Phase 2 complete: ${#MERGE_DIRS[@]} merge tokenizers generated"
echo ""

################################################################################
# PHASE 3: Evaluate all tokenizers
################################################################################

echo ""
echo "================================================================================"
echo "PHASE 3: Evaluating All Tokenizers (TEST + OOS + FineWeb)"
echo "================================================================================"
echo ""

ALL_TOKENIZERS=()

# Collect all tokenizer directories
for dir in "${BASELINE_DIRS[@]}" "${MERGE_DIRS[@]}"; do
    if [[ -f "$dir/tokenizer.json" ]]; then
        ALL_TOKENIZERS+=("$dir")
    fi
done

# Also add selected tokenizers from ensembles
for ens_dir in "${ENSEMBLE_DIRS[@]}"; do
    # Find selected tokenizer
    if [[ -f "$ens_dir/ensemble.json" ]]; then
        SELECTED_PATH=$(jq -r '.best.train_run' "$ens_dir/ensemble.json" 2>/dev/null || echo "")
        if [[ -n "$SELECTED_PATH" ]] && [[ -f "$SELECTED_PATH/tokenizer.json" ]]; then
            ALL_TOKENIZERS+=("$SELECTED_PATH")
        fi
    fi
done

echo "Found ${#ALL_TOKENIZERS[@]} tokenizers to evaluate"
echo ""

EVAL_COUNT=0
EVAL_DIR="$OUT_DIR/paper_e2e_evaluations"
mkdir -p "$EVAL_DIR"

for tok_dir in "${ALL_TOKENIZERS[@]}"; do
    TOK_NAME=$(basename "$tok_dir")
    echo "Evaluating: $TOK_NAME"

    # TEST
    if [[ ! -f "$EVAL_DIR/${TOK_NAME}_test/metrics.json" ]]; then
        uv run python -m scripts.eval_compression \
            --exp "${TOK_NAME}_test" \
            --tokenizer "$tok_dir/tokenizer.json" \
            --eval-file "$DATA_DIR/test.txt" \
            --out "$EVAL_DIR" > /dev/null 2>&1 && echo "  ✓ TEST" || echo "  ⚠️  TEST failed"
        EVAL_COUNT=$((EVAL_COUNT + 1))
    fi

    # OOS
    if [[ ! -f "$EVAL_DIR/${TOK_NAME}_oos/metrics.json" ]]; then
        uv run python -m scripts.eval_compression \
            --exp "${TOK_NAME}_oos" \
            --tokenizer "$tok_dir/tokenizer.json" \
            --eval-file "$DATA_DIR/test_oos.txt" \
            --out "$EVAL_DIR" > /dev/null 2>&1 && echo "  ✓ OOS" || echo "  ⚠️  OOS failed"
        EVAL_COUNT=$((EVAL_COUNT + 1))
    fi

    # FineWeb
    if [[ "$SKIP_FINEWEB" == "false" ]]; then
        if [[ ! -f "$EVAL_DIR/${TOK_NAME}_fineweb.json" ]]; then
            uv run --with datasets python -m scripts.eval_fineweb \
                --tokenizer "$tok_dir/tokenizer.json" \
                --num-samples "$FINEWEB_SAMPLES" \
                --out "$EVAL_DIR/${TOK_NAME}_fineweb.json" > /dev/null 2>&1 && echo "  ✓ FineWeb" || echo "  ⚠️  FineWeb failed"
            EVAL_COUNT=$((EVAL_COUNT + 1))
        fi
    fi
    echo ""
done

echo "✅ Phase 3 complete: $EVAL_COUNT evaluations performed"
echo ""

################################################################################
# PHASE 4: Aggregate results
################################################################################

echo ""
echo "================================================================================"
echo "PHASE 4: Aggregating Results"
echo "================================================================================"
echo ""

# Use the aggregation script from run_complete_evaluation.sh
python3 - "$EVAL_DIR" << 'PYTHON_EOF'
import json
import csv
from pathlib import Path
import sys

eval_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/paper_e2e_evaluations")

# Collect all results
results = []
for json_file in eval_dir.rglob("*.json"):
    # Skip non-metrics files
    if json_file.name.endswith("_fineweb.json"):
        with open(json_file) as f:
            data = json.load(f)

        result = {
            "name": json_file.stem.replace("_fineweb", ""),
            "dataset": "fineweb",
            "tokens_per_byte": data.get("tokens_per_byte"),
            "total_bytes": data.get("total_bytes"),
            "total_tokens": data.get("total_tokens"),
        }
    elif json_file.name == "metrics.json":
        with open(json_file) as f:
            data = json.load(f)

        parent_name = json_file.parent.name
        if "_test" in parent_name and "_oos" not in parent_name:
            dataset = "test"
        elif "_oos" in parent_name:
            dataset = "oos"
        else:
            dataset = "unknown"

        result = {
            "name": parent_name.replace("_test", "").replace("_oos", ""),
            "dataset": dataset,
            "tokens_per_byte": data.get("tokens_per_byte"),
            "bytes_per_token": data.get("bytes_per_token"),
            "total_bytes": data.get("total_bytes"),
            "total_tokens": data.get("total_tokens"),
        }
    else:
        continue

    # Extract metadata from name
    name = result["name"]

    # Method
    if "sequential" in name:
        result["method"] = "sequential"
    elif "exp_p3" in name:
        result["method"] = "exp_p3"
    elif "exp_p2" in name:
        result["method"] = "exp_p2"
    elif "weighted_70" in name:
        result["method"] = "weighted_70"
    elif "weighted_30" in name:
        result["method"] = "weighted_30"
    elif "merge_majority" in name:
        result["method"] = "merge_majority"
    elif "baseline" in name:
        result["method"] = "baseline"
    elif "_m" in name and name.split("_m")[-1].split("_")[0].isdigit():
        result["method"] = "selection"
    else:
        result["method"] = "unknown"

    # Config
    import re
    match = re.search(r'_s(\d+)_v(\d+)_k(\d+)', name)
    if match:
        result["seed"] = int(match.group(1))
        result["vocab"] = int(match.group(2))
        result["k"] = int(match.group(3))
    else:
        result["seed"], result["vocab"], result["k"] = None, None, None

    results.append(result)

# Sort
results.sort(key=lambda x: (
    x.get("seed") or 0,
    x.get("vocab") or 0,
    x.get("k") or 0,
    x.get("method", ""),
    x.get("dataset", "")
))

# Write CSV
csv_path = eval_dir / "results_aggregated.csv"
if results:
    fieldnames = ["seed", "vocab", "k", "method", "dataset", "tokens_per_byte", "bytes_per_token", "total_bytes", "total_tokens", "name"]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"✓ Written {len(results)} results to {csv_path}")

# Write JSON
json_path = eval_dir / "results_aggregated.json"
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Written {len(results)} results to {json_path}")

# Print summary
print("\n" + "="*80)
print("RESULTS SUMMARY - OOS Performance (tokens_per_byte, lower is better)")
print("="*80)

configs = {}
for r in results:
    if r["dataset"] != "oos":
        continue
    key = (r.get("seed"), r.get("vocab"), r.get("k"))
    if key not in configs:
        configs[key] = []
    configs[key].append(r)

for (seed, vocab, k), config_results in sorted(configs.items(), key=lambda x: (x[0][0] or 0, x[0][1] or 0, x[0][2] or 0)):
    if vocab is None:
        continue
    print(f"\nSeed={seed}, Vocab={vocab//1024}K, K={k}")
    print("-" * 70)
    config_results.sort(key=lambda x: x.get("tokens_per_byte") or 999)
    for r in config_results:
        method = r.get("method", "unknown")
        tpb = r.get("tokens_per_byte")
        if tpb:
            marker = "⭐" if method in ["sequential", "exp_p2", "exp_p3"] else "  "
            print(f"{marker} {method:20s}: {tpb:.6f}")

print("\n" + "="*80)
PYTHON_EOF

################################################################################
# DONE
################################################################################

echo ""
echo "================================================================================"
echo "✅ END-TO-END RUN COMPLETE!"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  Baselines trained: ${#BASELINE_DIRS[@]}"
echo "  Ensembles trained: ${#ENSEMBLE_DIRS[@]}"
echo "  Merge tokenizers: ${#MERGE_DIRS[@]}"
echo "  Total tokenizers: ${#ALL_TOKENIZERS[@]}"
echo "  Evaluations: $EVAL_COUNT"
echo ""
echo "Results:"
echo "  Aggregated data: $EVAL_DIR/results_aggregated.csv"
echo "  JSON data: $EVAL_DIR/results_aggregated.json"
echo ""
echo "Next steps:"
echo "  1. Review: cat $EVAL_DIR/results_aggregated.csv"
echo "  2. Analyze: Check which methods perform best on OOS"
echo "  3. Generate paper tables/figures from aggregated data"
echo ""
echo "================================================================================"
