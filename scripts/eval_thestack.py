"""Evaluate tokenizer(s) on The Stack (code) out-of-sample data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate tokenizer on The Stack OOS data")
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    ap.add_argument("--num-samples", type=int, default=1000, help="number of code samples")
    ap.add_argument("--language", default="python", help="programming language to sample (default: python)")
    ap.add_argument("--out", help="output JSON path for results")
    ap.add_argument("--verbose", action="store_true", help="show progress for each sample")
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

    # Load The Stack dataset for specified language
    print(f"Loading The Stack dataset for language: {args.language} (streaming=True)...")

    # The Stack uses data_dir to specify language
    data_dir = f"data/{args.language}"

    try:
        ds = load_dataset(
            "bigcode/the-stack",
            data_dir=data_dir,
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Note: The Stack dataset requires authentication. Please:")
        print("1. Create a HuggingFace account")
        print("2. Accept the terms at https://huggingface.co/datasets/bigcode/the-stack")
        print("3. Login with: huggingface-cli login")
        raise SystemExit(1)

    # Collect samples
    print(f"Collecting {args.num_samples} {args.language} code samples...")
    texts = []
    total_bytes = 0
    skipped = 0

    for i, sample in enumerate(ds):
        if len(texts) >= args.num_samples:
            break

        # Extract content field
        content = sample.get("content", "")

        # Skip empty or very short files
        if len(content) < 10:
            skipped += 1
            continue

        texts.append(content)
        byte_count = len(content.encode("utf-8"))
        total_bytes += byte_count

        if args.verbose or (len(texts) % 100 == 0):
            print(f"  Sample {len(texts)}/{args.num_samples}: {byte_count} bytes (skipped {skipped} empty)")

    if len(texts) < args.num_samples:
        print(f"Warning: Only collected {len(texts)} samples (requested {args.num_samples})")

    # Combine with newlines (same as FineWeb)
    combined = "\n".join(texts)

    # Verify actual byte count
    actual_bytes = len(combined.encode("utf-8"))
    print(f"Total content: {actual_bytes} bytes from {len(texts)} samples")

    # Tokenize
    print(f"Loading tokenizer from {args.tokenizer}...")
    tok = Tokenizer.from_file(str(args.tokenizer))
    print("Tokenizing combined code...")
    encoding = tok.encode(combined)
    total_tokens = len(encoding.tokens)
    tpb = total_tokens / actual_bytes if actual_bytes > 0 else 0

    result = {
        "tokenizer_path": str(args.tokenizer),
        "dataset": "the-stack",
        "language": args.language,
        "num_samples": len(texts),
        "samples_requested": args.num_samples,
        "samples_skipped": skipped,
        "total_bytes": actual_bytes,
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
    print(f"Language: {args.language}")
    print(f"Samples used: {len(texts)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())