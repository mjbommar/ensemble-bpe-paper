"""
Evaluate compression metrics on held-out text using a tiktoken encoding.

Metrics computed (corpus-level):
  - total_chars, total_bytes, total_tokens, doc_count
  - chars_per_token, bytes_per_token, tokens_per_char, tokens_per_byte, bytes_per_char

Examples:
  uv run --with tiktoken python -m scripts.eval_compression_tiktoken \
      --exp gpt2_eval --encoding gpt2 --eval-file data/processed/test.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from ebpe.artifact_store import create_run_dir, default_artifact_paths, write_json, write_text, write_yaml_like
import csv


def _get_encoding(name: str):
    import tiktoken  # type: ignore

    # Allow both canonical names (e.g., "gpt2", "cl100k_base") and aliases
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        # Fallback: load via encoding_for_model if a model-like alias is passed
        return tiktoken.encoding_for_model(name)


def _file_lines(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    for line in txt.splitlines():
        yield line.rstrip("\n")


def compute_metrics(encoding, eval_path: Path) -> Dict[str, float | int]:
    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    doc_count = 0

    for line in _file_lines(eval_path):
        doc_count += 1
        total_chars += len(line)
        b = line.encode("utf-8")
        total_bytes += len(b)
        ids = encoding.encode(line)
        total_tokens += len(ids)

    def safe_div(a: int, b: int) -> float:
        return float(a) / float(b) if b else 0.0

    return {
        "doc_count": doc_count,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "chars_per_token": safe_div(total_chars, total_tokens),
        "bytes_per_token": safe_div(total_bytes, total_tokens),
        "tokens_per_char": safe_div(total_tokens, total_chars),
        "tokens_per_byte": safe_div(total_tokens, total_bytes),
        "bytes_per_char": safe_div(total_bytes, total_chars),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment/run name for artifacts")
    ap.add_argument("--encoding", default="gpt2", help="tiktoken encoding name (e.g., gpt2, cl100k_base)")
    ap.add_argument("--eval-file", required=True, help="path to evaluation text file")
    ap.add_argument("--out", default="artifacts", help="artifacts base directory")
    args = ap.parse_args(argv)

    run_dir = create_run_dir(args.exp, base_dir=args.out)
    paths = default_artifact_paths(run_dir)

    cfg = {
        "eval": "compression_tiktoken",
        "encoding": args.encoding,
        "eval_file": str(Path(args.eval_file).resolve()),
    }
    write_yaml_like(paths["config"], cfg)

    # record a minimal env
    write_text(paths["env"], f"evaluation: tiktoken encoding={args.encoding}\n")

    enc = _get_encoding(args.encoding)
    metrics = compute_metrics(enc, Path(args.eval_file))
    write_json(paths["metrics"], metrics)

    csv_path = Path(run_dir) / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)
    write_text(paths["logs"], "computed tiktoken compression metrics\n")

    print(str(run_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

