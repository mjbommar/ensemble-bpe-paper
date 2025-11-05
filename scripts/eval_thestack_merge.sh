#!/usr/bin/env bash
# Evaluate merge tokenizers on The Stack

cd /nas4/data/experiments/ensemble-bpe

count=0
for TOK_DIR in 20251105_00*/; do
    if [ ! -f "$TOK_DIR/tokenizer.json" ]; then
        continue
    fi

    TOK_NAME=$(basename "$TOK_DIR")
    OUT_PATH="paper_e2e_evaluations/thestack_results/${TOK_NAME}_thestack.json"

    if [ -f "$OUT_PATH" ]; then
        echo "Skip $TOK_NAME - already done"
        continue
    fi

    echo "[$count] Evaluating $TOK_NAME..."
    uv run --with datasets --with tokenizers python /home/mjbommar/src/ensemble-bpe-paper/scripts/eval_thestack.py \
        --tokenizer "$TOK_DIR/tokenizer.json" \
        --num-samples 1000 \
        --language python \
        --out "$OUT_PATH" 2>&1 | grep -E "Tokens per byte:" || true

    count=$((count + 1))
done

echo "Done! Evaluated $count tokenizers"
