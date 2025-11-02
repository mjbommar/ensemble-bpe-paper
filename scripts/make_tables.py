"""
Generate Markdown and LaTeX tables from experiment artifacts.

Inputs:
  - Either pass --summary-csv (aggregated.csv from scripts.aggregate_metrics)
    or --inputs (one or more artifact roots to scan for pipeline_summary.csv).
  - Optionally pass --seeds-stats (JSON from scripts.run_seeds) to include
    mean±std rows by kind.

Outputs:
  - results/table.md
  - results/table.tex

Example:
  uv run python -m scripts.make_tables --inputs artifacts --out results
  uv run python -m scripts.make_tables --summary-csv artifacts/aggregated/aggregated.csv --out results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _find_summaries(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for root in inputs:
        p = Path(root)
        if p.is_file() and p.name == "pipeline_summary.csv":
            paths.append(p)
        elif p.exists():
            paths.extend(p.rglob("pipeline_summary.csv"))
    return sorted(set(paths))


def _load_rows_from_summary(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows.extend(rdr)
    return rows


def _load_all_rows(summary_csv: str | None, inputs: List[str] | None) -> List[Dict[str, str]]:
    if summary_csv:
        return _load_rows_from_summary(Path(summary_csv))
    rows: List[Dict[str, str]] = []
    for p in _find_summaries(inputs or ["artifacts"]):
        rows.extend(_load_rows_from_summary(p))
    return rows


def _format_md_table(rows: List[Dict[str, str]]) -> str:
    # Keep key columns
    cols = ["name", "kind", "tokens_per_byte", "run_dir"]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def _format_latex_table(rows: List[Dict[str, str]]) -> str:
    cols = ["name", "kind", "tokens_per_byte"]
    header = "\\begin{tabular}{l l r}\n\\toprule\nname & kind & tokens/byte\\\\\n\\midrule\n"
    body = "\n".join([f"{r.get('name','')} & {r.get('kind','')} & {r.get('tokens_per_byte','')}\\\\" for r in rows])
    footer = "\n\\bottomrule\n\\end{tabular}\n"
    return header + body + footer


def _maybe_include_seeds(stats_path: str | None, by_kind: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    if not stats_path:
        # Return one representative row per kind (min tokens_per_byte)
        rows: List[Dict[str, str]] = []
        for kind, rs in by_kind.items():
            best = min(rs, key=lambda r: float(r.get("tokens_per_byte", 0.0) or 0.0))
            rows.append(best)
        return rows

    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    out: List[Dict[str, str]] = []
    for kind, meta in stats.items():
        row = {
            "name": f"{kind} (mean±std)",
            "kind": kind,
            "tokens_per_byte": f"{meta.get('mean', 0.0):.6f} ± {meta.get('std', 0.0):.6f}",
            "run_dir": "",
        }
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", help="aggregated.csv from scripts.aggregate_metrics")
    ap.add_argument("--inputs", nargs="*", help="artifact roots to scan for pipeline_summary.csv")
    ap.add_argument("--seeds-stats", help="seeds_stats.json from scripts.run_seeds (optional)")
    ap.add_argument("--out", default="results", help="output directory for tables")
    args = ap.parse_args(argv)

    rows = _load_all_rows(args.summary_csv, args.inputs)
    # Group by kind for concise tables
    by_kind: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        k = str(r.get("kind", ""))
        if not k:
            continue
        by_kind.setdefault(k, []).append(r)

    table_rows = _maybe_include_seeds(args.seeds_stats, by_kind)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table.md").write_text(_format_md_table(table_rows), encoding="utf-8")
    (out_dir / "table.tex").write_text(_format_latex_table(table_rows), encoding="utf-8")
    print(str((out_dir / "table.md").resolve()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

