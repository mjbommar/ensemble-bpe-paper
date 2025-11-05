"""Evaluate tokenizer(s) on FineWeb out-of-sample data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate tokenizer on FineWeb OOS data")
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    ap.add_argument("--num-samples", type=int, default=10, help="number of fineweb samples")
    ap.add_argument("--out", help="output JSON path for results")
    args = ap.parse_args(argv)

    # Import here to avoid hard dependency on datasets
    try:
        from datasets import load_dataset
        from tokenizers import Tokenizer
    except ImportError as e:
        raise SystemExit(
            f"Missing required dependencies: {e}\n"
            "Install with: uv pip install datasets tokenizers"
        )

    # Load FineWeb dataset
    print(f"Loading FineWeb sample-10BT dataset (streaming={True})...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # Collect samples
    print(f"Collecting {args.num_samples} samples...")
    texts = []
    total_bytes = 0
    for i, sample in enumerate(ds):
        if i >= args.num_samples:
            break
        text = sample["text"]
        texts.append(text)
        total_bytes += len(text.encode("utf-8"))
        print(f"  Sample {i+1}/{args.num_samples}: {len(text.encode('utf-8'))} bytes")

    combined = "\n".join(texts)

    # Tokenize
    print(f"Loading tokenizer from {args.tokenizer}...")
    tok = Tokenizer.from_file(str(args.tokenizer))
    print("Tokenizing combined text...")
    encoding = tok.encode(combined)
    total_tokens = len(encoding.tokens)
    tpb = total_tokens / total_bytes

    result = {
        "tokenizer_path": str(args.tokenizer),
        "num_samples": args.num_samples,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "tokens_per_byte": tpb,
    }

    # Save result if requested
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved results to {out_path}")

    # Print result
    print("\nResults:")
    print(json.dumps(result, indent=2))
    print(f"\nTokens per byte: {tpb:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
