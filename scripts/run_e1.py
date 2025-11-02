"""
Run the E1 experiment: compare training performance (time, peak RSS) across
several tokenization algorithms on the same dataset, and report compression on
the same eval split.

By default runs: hf_bpe, hf_wordpiece, hf_unigram.

Inputs:
  --config <path>  # base TOML for data/artifacts; train fields overridden per algo
  --algos hf_bpe hf_wordpiece hf_unigram  # optional list to override
  --out artifacts/e1  # output directory for E1 summary

Example:
  uv run --with tokenizers --with psutil python -m scripts.run_e1 \
      --config experiments/pg19_bpe_tiny.toml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import tomllib


def _run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace")


def _last_line(s: str) -> str:
    return [ln.strip() for ln in s.splitlines() if ln.strip()][-1]


def _write_toml_like(path: Path, cfg: Dict[str, Any]) -> None:
    # minimal TOML writer for our known schema
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
                        # flat lists (strings or numbers)
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
    ap.add_argument("--algos", nargs="*", default=["hf_bpe", "hf_wordpiece", "hf_unigram"], help="algorithms to run")
    ap.add_argument("--out", default="artifacts/e1", help="output dir for E1 results")
    args = ap.parse_args(argv)

    base_cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))

    results: List[Dict[str, Any]] = []
    pipes: List[Path] = []

    with tempfile.TemporaryDirectory() as td:
        for algo in args.algos:
            cfg = dict(base_cfg)
            train = dict(cfg.get("train", {}))
            train["algo"] = algo
            # Make exp_name distinct per algo for clarity
            base_name = train.get("exp_name", "e1_exp")
            train["exp_name"] = f"{base_name}_{algo}"
            cfg["train"] = train
            tmp_toml = Path(td) / f"cfg_{algo}.toml"
            _write_toml_like(tmp_toml, cfg)

            out = _run([
                "uv", "run", "--with", "tokenizers", "--with", "psutil",
                "python", "-m", "scripts.run_experiment", "--config", str(tmp_toml),
            ])
            pipe_dir = Path(_last_line(out))
            pipes.append(pipe_dir)

            # Read pipeline manifest to locate train/eval runs
            manifest = json.loads((pipe_dir / "pipeline.json").read_text(encoding="utf-8"))
            train_run = Path(manifest["train_run_dir"])  # type: ignore
            eval_run = Path(manifest["eval_run_dir"])  # type: ignore
            train_metrics = json.loads((train_run / "metrics.json").read_text(encoding="utf-8"))
            eval_metrics = json.loads((eval_run / "metrics.json").read_text(encoding="utf-8"))
            results.append(
                {
                    "algo": algo,
                    "train_run": str(train_run.resolve()),
                    "eval_run": str(eval_run.resolve()),
                    "train_wall_time_s": train_metrics.get("train_wall_time_s"),
                    "train_cpu_time_s": train_metrics.get("train_cpu_time_s"),
                    "peak_rss_kb": train_metrics.get("peak_rss_kb"),
                    "tokens_per_byte": eval_metrics.get("tokens_per_byte"),
                    "vocab_size": train_metrics.get("vocab_size"),
                }
            )

    # Write E1 summary
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv

    csv_path = out_dir / "e1_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "algo",
                "train_run",
                "eval_run",
                "train_wall_time_s",
                "train_cpu_time_s",
                "peak_rss_kb",
                "tokens_per_byte",
                "vocab_size",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(r)

    (out_dir / "e1_summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(str(csv_path.resolve()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
