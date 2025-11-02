"""
Run a seed sweep over a base experiment config, aggregating results.

Modifies [data].seed and [train].exp_name per seed, runs the pipeline for each,
then writes a summary CSV/JSON with per-seed tokens_per_byte and simple stats.

Example:
  uv run --with tokenizers --with psutil python -m scripts.run_seeds \
    --config experiments/local_books_bpe_tiny.toml --seeds 13 17 19
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import tomllib


def _run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace")


def _last_line(s: str) -> str:
    return [ln.strip() for ln in s.splitlines() if ln.strip()][-1]


def _write_toml_like(path: Path, cfg: Dict[str, Any]) -> None:
    lines: List[str] = []
    for sec in ["data", "oos_data", "train", "ensemble", "eval", "baselines", "artifacts"]:
        val = cfg.get(sec)
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
    ap.add_argument("--config", required=True, help="base TOML config path")
    ap.add_argument("--seeds", nargs="+", type=int, help="list of seeds to run")
    ap.add_argument("--out", default="artifacts/seeds", help="output dir for seeds summary")
    args = ap.parse_args(argv)

    base_cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds else [13]

    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        for sd in seeds:
            cfg = json.loads(json.dumps(base_cfg))
            data = dict(cfg.get("data", {}))
            data["seed"] = int(sd)
            cfg["data"] = data
            tr = dict(cfg.get("train", {}))
            name = tr.get("exp_name", "exp")
            tr["exp_name"] = f"{name}_s{sd}"
            cfg["train"] = tr

            tmp = Path(td) / f"cfg_s{sd}.toml"
            _write_toml_like(tmp, cfg)
            out = _run([
                "uv", "run", "--with", "tokenizers", "--with", "psutil",
                "python", "-m", "scripts.run_experiment", "--config", str(tmp),
            ])
            pipe = Path(_last_line(out))
            pipe_csv = pipe / "pipeline_summary.csv"
            if pipe_csv.exists():
                with pipe_csv.open("r", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    for r in rdr:
                        # Collect all result kinds (single algo, OOS, ensemble selection/merge, baselines)
                        rows.append({
                            "seed": sd,
                            "name": r.get("name"),
                            "kind": r.get("kind"),
                            "run_dir": r.get("run_dir"),
                            "tokens_per_byte": float(r.get("tokens_per_byte", 0.0) or 0.0),
                            "pipeline_dir": str(pipe.resolve()),
                        })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "seeds_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "name", "kind", "run_dir", "tokens_per_byte", "pipeline_dir"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Simple stats per kind
    stats: Dict[str, Dict[str, float]] = {}
    by_kind: Dict[str, List[float]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["tokens_per_byte"])  # type: ignore
    for k, vals in by_kind.items():
        n = len(vals)
        mean = sum(vals) / n if n else 0.0
        var = sum((v - mean) ** 2 for v in vals) / n if n else 0.0
        stats[k] = {"n": n, "mean": mean, "std": math.sqrt(var)}

    (out_dir / "seeds_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(str(csv_path.resolve()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
