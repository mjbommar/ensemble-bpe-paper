"""
Evaluate compression metrics on held-out text using a saved HF tokenizer.

Metrics computed (corpus-level):
  - total_chars, total_bytes, total_tokens, doc_count
  - chars_per_token, bytes_per_token, tokens_per_char, tokens_per_byte, bytes_per_char

Examples:
  uv run --with tokenizers python -m scripts.eval_compression \
      --exp hf_bpe_eval --tokenizer artifacts/<run>/tokenizer.json \
      --eval-file data/processed/test.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from ebpe.artifact_store import create_run_dir, default_artifact_paths, write_json, write_text, write_yaml_like
import csv


def _load_tokenizer(tokenizer_path: str):
    from tokenizers import Tokenizer  # type: ignore

    return Tokenizer.from_file(tokenizer_path)


def _file_lines(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    for line in txt.splitlines():
        yield line.rstrip("\n")


def compute_metrics(tokenizer, eval_path: Path) -> Dict[str, float | int]:
    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    doc_count = 0

    for line in _file_lines(eval_path):
        doc_count += 1
        total_chars += len(line)
        b = line.encode("utf-8")
        total_bytes += len(b)
        enc = tokenizer.encode(line)
        total_tokens += len(enc.ids)

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
    ap.add_argument("--tokenizer", help="path to tokenizer.json (if --run-dir not given)")
    ap.add_argument("--run-dir", help="optional existing run dir containing tokenizer.json")
    ap.add_argument("--eval-file", required=True, help="path to evaluation text file")
    ap.add_argument("--out", default="artifacts", help="artifacts base directory")
    args = ap.parse_args(argv)

    if args.run_dir:
        tok_path = Path(args.run_dir) / "tokenizer.json"
    else:
        if not args.tokenizer:
            raise SystemExit("Provide --tokenizer or --run-dir")
        tok_path = Path(args.tokenizer)

    run_dir = create_run_dir(args.exp, base_dir=args.out)
    paths = default_artifact_paths(run_dir)

    cfg = {
        "eval": "compression",
        "tokenizer": str(tok_path.resolve()),
        "eval_file": str(Path(args.eval_file).resolve()),
    }
    write_yaml_like(paths["config"], cfg)

    # record a minimal env
    write_text(paths["env"], "evaluation: compression metrics\n")

    tokenizer = _load_tokenizer(str(tok_path))
    metrics = compute_metrics(tokenizer, Path(args.eval_file))
    write_json(paths["metrics"], metrics)
    # Also write CSV for quick spreadsheet imports
    csv_path = Path(run_dir) / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)
    write_text(paths["logs"], "computed compression metrics\n")

    # also write a convenience file name
    (run_dir / "compression.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(str(run_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
