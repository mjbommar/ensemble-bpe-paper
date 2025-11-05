"""
Run E1 (algo comparison) across multiple vocabulary sizes and aggregate results.

Example:
  uv run --with tokenizers --with psutil python -m scripts.run_e1_sizes \
      --config experiments/gutenberg_e1_32k.toml --sizes 8192 16384 32768 65536
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import tomllib


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"ERROR: Command failed with exit code {e.returncode}", file=sys.stderr)
        print(f"Command: {' '.join(cmd)}", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        if e.output:
            print("Output:", file=sys.stderr)
            print(e.output.decode("utf-8", errors="replace"), file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        raise


def _last_line(s: str) -> str:
    return [ln.strip() for ln in s.splitlines() if ln.strip()][-1]


def _write_toml_like(path: Path, cfg: Dict[str, Any]) -> None:
    lines: List[str] = []
    for sec in ["data", "oos_data", "train", "ensemble", "eval", "baselines", "artifacts"]:
        if sec in cfg and cfg[sec] is not None:
            val = cfg[sec]
            if isinstance(val, dict) and val:
                lines.append(f"[{sec}]")
                for k, v in val.items():
                    if isinstance(v, bool):
                        lines.append(f"{k} = {'true' if v else 'false'}")
                    elif isinstance(v, (int, float)):
                        lines.append(f"{k} = {v}")
                    elif isinstance(v, list):
                        def _fmt(x):
                            if isinstance(x, (int, float)):
                                return str(x)
                            return f'"{x}"'
                        lines.append(f"{k} = [{', '.join(_fmt(x) for x in v)}]")
                    elif v is None:
                        continue
                    else:
                        lines.append(f"{k} = \"{v}\"")
                lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="base TOML config (data paths, etc.)")
    ap.add_argument("--sizes", nargs="+", type=int, required=True, help="vocab sizes (e.g., 8192 16384 32768 65536)")
    ap.add_argument("--algos", nargs="*", default=["hf_bpe", "hf_wordpiece", "hf_unigram"], help="algorithms to run in E1")
    ap.add_argument("--out", default="artifacts/e1_sizes", help="output directory for aggregated results")
    args = ap.parse_args(argv)

    base_cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Path] = []
    with tempfile.TemporaryDirectory() as td:
        for vs in args.sizes:
            cfg = dict(base_cfg)
            train = dict(cfg.get("train", {}))
            train["vocab_size"] = int(vs)
            base_name = train.get("exp_name", "e1_pg")
            train["exp_name"] = f"{base_name}_{vs}"
            cfg["train"] = train

            tmp_toml = Path(td) / f"cfg_{vs}.toml"
            _write_toml_like(tmp_toml, cfg)

            out = _run([
                "uv", "run", "--with", "tokenizers", "--with", "psutil",
                "python", "-m", "scripts.run_e1", "--config", str(tmp_toml), "--algos", *args.algos,
            ])
            summary_csv = Path(_last_line(out))  # run_e1 prints CSV path
            summaries.append(summary_csv)

            # Read rows and tag with vocab_size
            with summary_csv.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    row["vocab_size"] = int(vs)
                    all_rows.append(row)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_csv = out_dir / "e1_sizes_summary.csv"
    fieldnames = [
        "vocab_size",
        "algo",
        "train_run",
        "eval_run",
        "train_wall_time_s",
        "train_cpu_time_s",
        "peak_rss_kb",
        "tokens_per_byte",
    ]
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k) for k in fieldnames})

    (out_dir / "e1_sizes_summary.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
    print(str(agg_csv.resolve()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

