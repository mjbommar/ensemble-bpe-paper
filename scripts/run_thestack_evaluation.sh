#!/usr/bin/env bash
#
# Run The Stack evaluation on all 216 tokenizers from paper experiment
# Usage: ./run_thestack_evaluation.sh

set -euo pipefail

# Configuration
BASE_DIR="/nas4/data/experiments/ensemble-bpe"
EVAL_DIR="${BASE_DIR}/paper_e2e_evaluations"
THESTACK_DIR="${EVAL_DIR}/thestack_results"
NUM_SAMPLES=1000
LANGUAGE="python"

# Create output directory
mkdir -p "$THESTACK_DIR"

echo "================================================================================"
echo "THE STACK EVALUATION - Code (Python) Out-of-Sample Test"
echo "================================================================================"
echo ""
echo "Base directory: $BASE_DIR"
echo "Output directory: $THESTACK_DIR"
echo "Language: $LANGUAGE"
echo "Samples per tokenizer: $NUM_SAMPLES"
echo ""

# Count total tokenizers
TOTAL_TOKENIZERS=0
for TOK_DIR in "$BASE_DIR"/20251103_* "$BASE_DIR"/20251104_*; do
    if [[ -d "$TOK_DIR" && -f "$TOK_DIR/tokenizer.json" ]]; then
        TOTAL_TOKENIZERS=$((TOTAL_TOKENIZERS + 1))
    fi
done

echo "Found $TOTAL_TOKENIZERS tokenizers to evaluate"
echo ""
echo "Starting evaluations..."
echo "================================================================================"

# Track progress
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

# Evaluate each tokenizer
for TOK_DIR in "$BASE_DIR"/20251103_* "$BASE_DIR"/20251104_*; do
    if [[ ! -d "$TOK_DIR" || ! -f "$TOK_DIR/tokenizer.json" ]]; then
        continue
    fi

    TOK_NAME=$(basename "$TOK_DIR")
    TOK_PATH="$TOK_DIR/tokenizer.json"
    OUT_PATH="$THESTACK_DIR/${TOK_NAME}_thestack.json"

    # Check if already evaluated
    if [[ -f "$OUT_PATH" ]]; then
        echo "  [SKIP] $TOK_NAME - already evaluated"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    echo ""
    echo "[$((COMPLETED + 1))/$TOTAL_TOKENIZERS] Evaluating: $TOK_NAME"

    # Run evaluation
    if uv run --with datasets --with tokenizers python scripts/eval_thestack.py \
        --tokenizer "$TOK_PATH" \
        --num-samples "$NUM_SAMPLES" \
        --language "$LANGUAGE" \
        --out "$OUT_PATH" 2>/dev/null; then

        # Extract tokens per byte from result
        TPB=$(grep "tokens_per_byte" "$OUT_PATH" | grep -oE "[0-9]+\.[0-9]+")
        echo "  ✓ Success - TPB: $TPB"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "  ✗ Failed"
        FAILED=$((FAILED + 1))
    fi

    # Show progress and time estimate
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [[ $COMPLETED -gt 0 ]]; then
        AVG_TIME=$((ELAPSED / COMPLETED))
        REMAINING=$((TOTAL_TOKENIZERS - COMPLETED - FAILED))
        ETA=$((AVG_TIME * REMAINING))
        echo "  Progress: $COMPLETED complete, $FAILED failed, $REMAINING remaining"
        echo "  ETA: $(date -d "@$((CURRENT_TIME + ETA))" +"%H:%M:%S")"
    fi
done

# Final summary
echo ""
echo "================================================================================"
echo "EVALUATION COMPLETE"
echo "================================================================================"
echo ""
echo "Total tokenizers: $TOTAL_TOKENIZERS"
echo "Successfully evaluated: $COMPLETED"
echo "Failed: $FAILED"
echo ""
echo "Results saved to: $THESTACK_DIR"
echo ""

# Generate aggregated summary
echo "Generating summary..."
python3 << 'PYTHON_EOF'
import json
import csv
from pathlib import Path

thestack_dir = Path("/nas4/data/experiments/ensemble-bpe/paper_e2e_evaluations/thestack_results")
results = []

for json_file in thestack_dir.glob("*.json"):
    with open(json_file) as f:
        data = json.load(f)

    # Extract name and metadata
    name = json_file.stem.replace("_thestack", "")

    result = {
        "name": name,
        "dataset": "thestack",
        "language": data.get("language", "python"),
        "num_samples": data.get("num_samples"),
        "total_bytes": data.get("total_bytes"),
        "total_tokens": data.get("total_tokens"),
        "tokens_per_byte": data.get("tokens_per_byte"),
    }

    # Extract seed, vocab, k from name
    import re
    match = re.search(r'_s(\d+)_v(\d+)', name)
    if match:
        result["seed"] = int(match.group(1))
        result["vocab"] = int(match.group(2))

    if "_k" in name:
        k_match = re.search(r'_k(\d+)', name)
        if k_match:
            result["k"] = int(k_match.group(1))
    else:
        result["k"] = None

    # Identify method
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

    results.append(result)

# Sort results
results.sort(key=lambda x: (
    x.get("seed", 0),
    x.get("vocab", 0),
    x.get("k", 0) if x.get("k") is not None else -1,
    x.get("method", ""),
))

# Save aggregated results
output_path = thestack_dir / "thestack_aggregated.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

# Save CSV
csv_path = thestack_dir / "thestack_aggregated.csv"
if results:
    fieldnames = ["name", "seed", "vocab", "k", "method", "dataset", "language",
                  "num_samples", "total_bytes", "total_tokens", "tokens_per_byte"]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

print(f"✓ Aggregated {len(results)} results")
print(f"  JSON: {output_path}")
print(f"  CSV: {csv_path}")

# Show summary statistics
if results:
    avg_tpb = sum(r["tokens_per_byte"] for r in results) / len(results)
    min_tpb = min(r["tokens_per_byte"] for r in results)
    max_tpb = max(r["tokens_per_byte"] for r in results)
    print(f"\nTokens per byte statistics:")
    print(f"  Average: {avg_tpb:.6f}")
    print(f"  Min: {min_tpb:.6f}")
    print(f"  Max: {max_tpb:.6f}")
PYTHON_EOF

echo ""
echo "================================================================================"
echo "Done! $(date)"
echo "================================================================================"