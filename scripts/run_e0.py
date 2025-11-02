"""
Run the E0 experiment: compare a single-model tokenizer vs ensemble selection
on out-of-sample compression, then aggregate results.

Inputs are two TOML configs compatible with scripts.run_experiment:
  --single <path>   # baseline config (no ensemble)
  --ensemble <path> # ensemble-enabled config

Outputs:
  - Aggregated table under --out (default: artifacts/e0)
  - Writes e0_summary.csv with rows for single and ensemble (IS + OOS when present)

Example:
  uv run --with tokenizers --with psutil python -m scripts.run_e0 \
      --single experiments/pg19_bpe_tiny.toml \
      --ensemble experiments/pg19_bpe_tiny_ensemble.toml
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict


def _run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace")


def _last_line(s: str) -> str:
    return [ln.strip() for ln in s.splitlines() if ln.strip()][-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True, help="baseline config TOML path")
    ap.add_argument("--ensemble", required=True, help="ensemble config TOML path")
    ap.add_argument("--out", default="artifacts/e0", help="output dir for aggregated E0 results")
    args = ap.parse_args(argv)

    # Run single-model pipeline
    s_out = _run([
        "uv", "run", "--with", "tokenizers", "--with", "psutil",
        "python", "-m", "scripts.run_experiment", "--config", args.single,
    ])
    s_dir = Path(_last_line(s_out)).resolve()

    # Run ensemble pipeline
    e_out = _run([
        "uv", "run", "--with", "tokenizers", "--with", "psutil",
        "python", "-m", "scripts.run_experiment", "--config", args.ensemble,
    ])
    e_dir = Path(_last_line(e_out)).resolve()

    # Aggregate across both
    agg_out = _run([
        "uv", "run", "python", "-m", "scripts.aggregate_metrics",
        "--inputs", str(s_dir.parent), str(e_dir.parent),
        "--out", args.out,
    ])
    agg_csv = Path(_last_line(agg_out))

    # Produce a minimal summary filtered to the two pipelines
    agg_json = Path(args.out) / "aggregated.json"
    rows: List[Dict] = json.loads(agg_json.read_text(encoding="utf-8")) if agg_json.exists() else []
    # Keep only rows that belong to either pipeline dir
    keep = [
        r
        for r in rows
        if str(r.get("pipeline_dir", "")).startswith(str(s_dir))
        or str(r.get("pipeline_dir", "")).startswith(str(e_dir))
    ]
    # Write concise summary
    import csv

    summary_csv = Path(args.out) / "e0_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pipeline_dir", "name", "kind", "run_dir", "tokens_per_byte"])
        w.writeheader()
        for r in keep:
            w.writerow({k: r.get(k) for k in ["pipeline_dir", "name", "kind", "run_dir", "tokens_per_byte"]})

    print(str(summary_csv.resolve()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
