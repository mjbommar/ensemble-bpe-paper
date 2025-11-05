#!/bin/bash
################################################################################
# COMPLETE ENSEMBLE BPE EVALUATION SCRIPT
################################################################################
#
# This script evaluates ALL ensemble methods (baseline + breakthrough) on
# ALL evaluation datasets (TEST, OOS, FineWeb) to generate complete paper data.
#
# What it does:
# 1. Evaluates baseline methods (selection, merge majority, weighted voting)
# 2. Evaluates breakthrough methods (sequential, exponential p=2/p=3)
# 3. Evaluates on 3 datasets: TEST, OOS (PG), FineWeb
# 4. Aggregates results to CSV/JSON
# 5. Creates comparison tables
#
# Ensemble configurations tested:
# - 16k vocab, K=4
# - 16k vocab, K=16
# - 32k vocab, K=4
# - 32k vocab, K=16
#
# Total evaluations: ~100+ (all methods × all ensembles × 3 datasets)
#
# Estimated runtime: 2-3 hours
#
# Usage:
#   ./scripts/run_complete_evaluation.sh
#   ./scripts/run_complete_evaluation.sh --quick  # Skip FineWeb (faster)
#
################################################################################

set -euo pipefail

# Configuration
PAPER_FAST_DIR="artifacts/paper_fast"
VALIDATION_DIR="artifacts/validation"
EVAL_OUTPUT_DIR="artifacts/complete_evaluation"
DATA_DIR="data/processed/pg_paper_full"
FINEWEB_SAMPLES=10

# Command line flags
SKIP_FINEWEB=false
if [[ "${1:-}" == "--quick" ]]; then
    SKIP_FINEWEB=true
    echo "⚡ Quick mode: Skipping FineWeb evaluation"
fi

# Create output directory
mkdir -p "$EVAL_OUTPUT_DIR"

echo "================================================================================"
echo "ENSEMBLE BPE COMPLETE EVALUATION"
echo "================================================================================"
echo "Output directory: $EVAL_OUTPUT_DIR"
echo "Data directory: $DATA_DIR"
echo "Skip FineWeb: $SKIP_FINEWEB"
echo ""

################################################################################
# STEP 1: Find all ensembles
################################################################################

echo "Step 1: Finding ensembles..."

ENSEMBLES=()
for ens_dir in "$PAPER_FAST_DIR"/*/; do
    if [[ -f "$ens_dir/ensemble.json" ]]; then
        ENSEMBLES+=("$ens_dir")
    fi
done

echo "Found ${#ENSEMBLES[@]} ensembles:"
for ens in "${ENSEMBLES[@]}"; do
    basename "$ens"
done
echo ""

################################################################################
# STEP 2: Evaluate baseline methods (from paper_fast)
################################################################################

echo "================================================================================"
echo "Step 2: Evaluating BASELINE methods"
echo "================================================================================"
echo ""

BASELINE_COUNT=0

for ens_dir in "${ENSEMBLES[@]}"; do
    ENS_NAME=$(basename "$ens_dir")
    echo "Processing ensemble: $ENS_NAME"

    # Find all baseline tokenizers (selection, merge, weighted)
    BASELINE_TOKS=()

    # Selection
    for tok_dir in "$PAPER_FAST_DIR"/*selected*/; do
        if [[ -f "$tok_dir/tokenizer.json" ]] && [[ "$tok_dir" == *"$ENS_NAME"* ]]; then
            BASELINE_TOKS+=("$tok_dir|selection")
        fi
    done

    # Merge majority
    for tok_dir in "$PAPER_FAST_DIR"/*merge_majority*/; do
        if [[ -f "$tok_dir/tokenizer.json" ]] && [[ "$tok_dir" == *"$ENS_NAME"* ]]; then
            BASELINE_TOKS+=("$tok_dir|merge_majority")
        fi
    done

    # Weighted entropy
    for tok_dir in "$PAPER_FAST_DIR"/*weighted_entropy*/; do
        if [[ -f "$tok_dir/tokenizer.json" ]] && [[ "$tok_dir" == *"$ENS_NAME"* ]]; then
            BASELINE_TOKS+=("$tok_dir|weighted_entropy")
        fi
    done

    echo "  Found ${#BASELINE_TOKS[@]} baseline tokenizers"

    # Evaluate each baseline tokenizer
    for tok_info in "${BASELINE_TOKS[@]}"; do
        TOK_DIR="${tok_info%|*}"
        METHOD_NAME="${tok_info#*|}"
        TOK_NAME=$(basename "$TOK_DIR")

        echo "    Evaluating $METHOD_NAME: $TOK_NAME"

        # TEST evaluation
        if [[ ! -f "$EVAL_OUTPUT_DIR/${TOK_NAME}_test.json" ]]; then
            uv run python -m scripts.eval_compression \
                --exp "${TOK_NAME}_test" \
                --tokenizer "$TOK_DIR/tokenizer.json" \
                --eval-file "$DATA_DIR/test.txt" \
                --out "$EVAL_OUTPUT_DIR" \
                > /dev/null 2>&1 || echo "      ⚠️  TEST failed"
            BASELINE_COUNT=$((BASELINE_COUNT + 1))
        else
            echo "      ✓ TEST (cached)"
        fi

        # OOS evaluation
        if [[ ! -f "$EVAL_OUTPUT_DIR/${TOK_NAME}_oos.json" ]]; then
            uv run python -m scripts.eval_compression \
                --exp "${TOK_NAME}_oos" \
                --tokenizer "$TOK_DIR/tokenizer.json" \
                --eval-file "$DATA_DIR/test_oos.txt" \
                --out "$EVAL_OUTPUT_DIR" \
                > /dev/null 2>&1 || echo "      ⚠️  OOS failed"
            BASELINE_COUNT=$((BASELINE_COUNT + 1))
        else
            echo "      ✓ OOS (cached)"
        fi

        # FineWeb evaluation
        if [[ "$SKIP_FINEWEB" == "false" ]]; then
            if [[ ! -f "$EVAL_OUTPUT_DIR/${TOK_NAME}_fineweb.json" ]]; then
                uv run --with datasets python -m scripts.eval_fineweb \
                    --tokenizer "$TOK_DIR/tokenizer.json" \
                    --num-samples "$FINEWEB_SAMPLES" \
                    --out "$EVAL_OUTPUT_DIR/${TOK_NAME}_fineweb.json" \
                    > /dev/null 2>&1 || echo "      ⚠️  FineWeb failed"
                BASELINE_COUNT=$((BASELINE_COUNT + 1))
            else
                echo "      ✓ FineWeb (cached)"
            fi
        fi
    done
    echo ""
done

echo "Baseline evaluations completed: $BASELINE_COUNT"
echo ""

################################################################################
# STEP 3: Evaluate breakthrough methods (from validation)
################################################################################

echo "================================================================================"
echo "Step 3: Evaluating BREAKTHROUGH methods (sequential, exponential)"
echo "================================================================================"
echo ""

BREAKTHROUGH_COUNT=0

for tok_dir in "$VALIDATION_DIR"/*/; do
    if [[ ! -f "$tok_dir/tokenizer.json" ]]; then
        continue
    fi

    TOK_NAME=$(basename "$tok_dir")

    # Determine method type
    if [[ "$TOK_NAME" == *"sequential"* ]]; then
        METHOD="sequential"
    elif [[ "$TOK_NAME" == *"exponential_p2_t7"* ]]; then
        METHOD="exp_p2_t7"
    elif [[ "$TOK_NAME" == *"exponential_p2"* ]]; then
        METHOD="exp_p2"
    elif [[ "$TOK_NAME" == *"exponential_p3"* ]]; then
        METHOD="exp_p3"
    else
        METHOD="unknown"
    fi

    echo "Evaluating $METHOD: $TOK_NAME"

    # TEST evaluation
    if [[ ! -f "$EVAL_OUTPUT_DIR/${TOK_NAME}_test.json" ]]; then
        uv run python -m scripts.eval_compression \
            --exp "${TOK_NAME}_test" \
            --tokenizer "$tok_dir/tokenizer.json" \
            --eval-file "$DATA_DIR/test.txt" \
            --out "$EVAL_OUTPUT_DIR" \
            > /dev/null 2>&1 || echo "  ⚠️  TEST failed"
        BREAKTHROUGH_COUNT=$((BREAKTHROUGH_COUNT + 1))
    else
        echo "  ✓ TEST (cached)"
    fi

    # OOS evaluation
    if [[ ! -f "$EVAL_OUTPUT_DIR/${TOK_NAME}_oos.json" ]]; then
        uv run python -m scripts.eval_compression \
            --exp "${TOK_NAME}_oos" \
            --tokenizer "$tok_dir/tokenizer.json" \
            --eval-file "$DATA_DIR/test_oos.txt" \
            --out "$EVAL_OUTPUT_DIR" \
            > /dev/null 2>&1 || echo "  ⚠️  OOS failed"
        BREAKTHROUGH_COUNT=$((BREAKTHROUGH_COUNT + 1))
    else
        echo "  ✓ OOS (cached)"
    fi

    # FineWeb evaluation
    if [[ "$SKIP_FINEWEB" == "false" ]]; then
        if [[ ! -f "$EVAL_OUTPUT_DIR/${TOK_NAME}_fineweb.json" ]]; then
            uv run --with datasets python -m scripts.eval_fineweb \
                --tokenizer "$tok_dir/tokenizer.json" \
                --num-samples "$FINEWEB_SAMPLES" \
                --out "$EVAL_OUTPUT_DIR/${TOK_NAME}_fineweb.json" \
                > /dev/null 2>&1 || echo "  ⚠️  FineWeb failed"
            BREAKTHROUGH_COUNT=$((BREAKTHROUGH_COUNT + 1))
        else
            echo "  ✓ FineWeb (cached)"
        fi
    fi
    echo ""
done

echo "Breakthrough evaluations completed: $BREAKTHROUGH_COUNT"
echo ""

################################################################################
# STEP 4: Aggregate results
################################################################################

echo "================================================================================"
echo "Step 4: Aggregating results to tables"
echo "================================================================================"
echo ""

# Create aggregation script on-the-fly
cat > "$EVAL_OUTPUT_DIR/aggregate.py" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Aggregate evaluation results to CSV/JSON tables."""
import json
import csv
from pathlib import Path
from typing import List, Dict
import sys

eval_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

# Collect all results
results = []
for json_file in eval_dir.glob("*/*.json"):
    # Skip non-metrics files
    if "fineweb" in str(json_file):
        with open(json_file) as f:
            data = json.load(f)

        # Parse name to extract metadata
        name_parts = json_file.stem.replace("_fineweb", "").split("_")

        result = {
            "name": json_file.stem,
            "dataset": "fineweb",
            "tokens_per_byte": data.get("tokens_per_byte"),
            "total_bytes": data.get("total_bytes"),
            "total_tokens": data.get("total_tokens"),
        }
    elif json_file.name == "metrics.json":
        with open(json_file) as f:
            data = json.load(f)

        # Determine dataset from parent dir name
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

    # Extract method and config from name
    name = result["name"]
    if "sequential" in name:
        result["method"] = "sequential"
    elif "exponential_p2_t7" in name:
        result["method"] = "exp_p2_t7"
    elif "exponential_p2" in name:
        result["method"] = "exp_p2"
    elif "exponential_p3" in name:
        result["method"] = "exp_p3"
    elif "selected" in name:
        result["method"] = "selection"
    elif "merge_majority" in name:
        result["method"] = "merge_majority"
    elif "weighted_entropy_30" in name:
        result["method"] = "weighted_30"
    elif "weighted_entropy_70" in name:
        result["method"] = "weighted_70"
    else:
        result["method"] = "unknown"

    # Extract config (vocab, K)
    if "v16384_k4" in name:
        result["vocab"], result["k"] = 16384, 4
    elif "v16384_k16" in name:
        result["vocab"], result["k"] = 16384, 16
    elif "v32768_k4" in name:
        result["vocab"], result["k"] = 32768, 4
    elif "v32768_k16" in name:
        result["vocab"], result["k"] = 32768, 16
    else:
        result["vocab"], result["k"] = None, None

    results.append(result)

# Sort by vocab, k, method, dataset
results.sort(key=lambda x: (
    x.get("vocab") or 0,
    x.get("k") or 0,
    x.get("method", ""),
    x.get("dataset", "")
))

# Write CSV
csv_path = eval_dir / "results_aggregated.csv"
if results:
    fieldnames = list(results[0].keys())
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

# Print summary table
print("\n" + "="*80)
print("RESULTS SUMMARY (OOS tokens_per_byte, lower is better)")
print("="*80)

# Group by config
configs = {}
for r in results:
    if r["dataset"] != "oos":
        continue
    key = (r.get("vocab"), r.get("k"))
    if key not in configs:
        configs[key] = []
    configs[key].append(r)

for (vocab, k), config_results in sorted(configs.items()):
    if vocab is None:
        continue
    print(f"\nConfig: {vocab//1024}K vocab, K={k}")
    print("-" * 60)
    config_results.sort(key=lambda x: x.get("tokens_per_byte") or 999)
    for r in config_results:
        method = r.get("method", "unknown")
        tpb = r.get("tokens_per_byte")
        if tpb:
            print(f"  {method:20s}: {tpb:.6f}")

print("\n" + "="*80)
PYTHON_EOF

# Run aggregation
chmod +x "$EVAL_OUTPUT_DIR/aggregate.py"
python3 "$EVAL_OUTPUT_DIR/aggregate.py" "$EVAL_OUTPUT_DIR"

################################################################################
# STEP 5: Summary
################################################################################

echo ""
echo "================================================================================"
echo "EVALUATION COMPLETE"
echo "================================================================================"
echo ""
echo "Total evaluations:"
echo "  Baseline methods: $BASELINE_COUNT"
echo "  Breakthrough methods: $BREAKTHROUGH_COUNT"
echo "  Total: $((BASELINE_COUNT + BREAKTHROUGH_COUNT))"
echo ""
echo "Results saved to:"
echo "  Directory: $EVAL_OUTPUT_DIR"
echo "  CSV: $EVAL_OUTPUT_DIR/results_aggregated.csv"
echo "  JSON: $EVAL_OUTPUT_DIR/results_aggregated.json"
echo ""
echo "Next steps:"
echo "  1. Review results: cat $EVAL_OUTPUT_DIR/results_aggregated.csv"
echo "  2. Check summary above for method rankings"
echo "  3. Generate plots/tables for paper"
echo ""
echo "✅ All done!"
echo "================================================================================"
