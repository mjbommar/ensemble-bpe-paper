"""
Aggregate metrics across multiple pipeline runs into a single CSV/JSON table.

By default, scans one or more artifact roots recursively for
`pipeline_summary.csv` files emitted by scripts.run_experiment, and merges their
rows with the pipeline directory path as `pipeline_dir`.

Examples:
  uv run python -m scripts.aggregate_metrics --inputs artifacts --out artifacts/aggregated
  uv run python -m scripts.aggregate_metrics --glob 'artifacts/*_pipeline/pipeline_summary.csv' --out artifacts/aggregated
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Dict


def _find_summaries(inputs: List[str], glob: str | None) -> List[Path]:
    paths: List[Path] = []
    if glob:
        for p in Path().glob(glob):
            if p.name == "pipeline_summary.csv":
                paths.append(p)
        return paths
    for root in inputs:
        r = Path(root)
        if r.is_file() and r.name == "pipeline_summary.csv":
            paths.append(r)
        elif r.exists():
            paths.extend(r.rglob("pipeline_summary.csv"))
    return sorted(set(paths))


def _read_summary(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["pipeline_dir"] = str(csv_path.parent.resolve())
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=["artifacts"], help="artifact roots to scan (dirs or files)")
    ap.add_argument("--glob", help="optional glob to match pipeline_summary.csv files explicitly")
    ap.add_argument("--out", default="artifacts/aggregated", help="output directory for aggregated files")
    ap.add_argument("--sort", default="tokens_per_byte", help="column to sort ascending by (default: tokens_per_byte)")
    args = ap.parse_args(argv)

    summaries = _find_summaries(args.inputs, args.glob)
    all_rows: List[Dict[str, str]] = []
    for p in summaries:
        all_rows.extend(_read_summary(p))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort if numeric column exists, fallback to string
    key = args.sort
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return float("inf")

    if all_rows and key in all_rows[0]:
        if key in {"tokens_per_byte", "bytes_per_token", "chars_per_token"}:
            all_rows.sort(key=lambda r: _to_float(r.get(key)))
        else:
            all_rows.sort(key=lambda r: str(r.get(key)))

    # Write CSV
    fieldnames = sorted(set(k for r in all_rows for k in r.keys())) if all_rows else ["pipeline_dir", "name", "kind", "run_dir", "tokens_per_byte"]
    with (out_dir / "aggregated.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # Write JSON
    (out_dir / "aggregated.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")

    print(str((out_dir / "aggregated.csv").resolve()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

