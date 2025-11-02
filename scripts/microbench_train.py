"""
Microbenchmark tokenizer training throughput and memory.

Runs one or more algorithms on a given training file, optionally repeating the
contents to scale up size, and records wall/cpu time, peak RSS, and bytes/sec.

Example:
  uv run --with tokenizers --with psutil python -m scripts.microbench_train \
    --train data/processed/train.txt --algos hf_bpe hf_wordpiece hf_unigram \
    --repeats 1 2 4 --vocab-size 2000 --out artifacts/microbench
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import List


def _with_src_env(extra: dict | None = None) -> dict:
    import os, sys
    root = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    if str(root) not in (env.get("PYTHONPATH") or ""):
        env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(root)
    if extra:
        env.update(extra)
    return env


def _run(cmd: list[str]) -> str:
    import subprocess

    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=_with_src_env())
    return out.decode("utf-8", errors="replace")


def _last_line(s: str) -> str:
    return [ln.strip() for ln in s.splitlines() if ln.strip()][-1]


def _repeat_file(in_path: Path, out_path: Path, k: int) -> None:
    txt = in_path.read_text(encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(txt * max(1, k), encoding="utf-8")


def _algo_module(algo: str) -> str:
    if algo == "hf_bpe":
        return "scripts.train_tokenizer"
    if algo == "hf_wordpiece":
        return "scripts.train_wordpiece"
    if algo == "hf_unigram":
        return "scripts.train_unigram"
    raise SystemExit(f"unknown algo: {algo}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--algos", nargs="+", default=["hf_bpe"])
    ap.add_argument("--repeats", nargs="+", type=int, default=[1])
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--out", default="artifacts/microbench")
    args = ap.parse_args(argv)

    rows: List[dict] = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for rep in args.repeats:
            bench_train = td / f"train_x{rep}.txt"
            _repeat_file(Path(args.train), bench_train, rep)
            size_bytes = bench_train.stat().st_size

            for algo in args.algos:
                module = _algo_module(algo)
                exp = f"mb_{algo}_x{rep}"
                out = _run([
                    "uv", "run", "--with", "tokenizers", "--with", "psutil",
                    "python", "-m", module,
                    "--exp", exp,
                    "--train", str(bench_train),
                    "--vocab-size", str(args.vocab_size),
                    "--out", str(args.out),
                ])
                run_dir = Path(_last_line(out))
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                wall = float(metrics.get("train_wall_time_s", 0.0) or 0.0)
                cpu = float(metrics.get("train_cpu_time_s", 0.0) or 0.0)
                rss = float(metrics.get("peak_rss_kb", 0.0) or 0.0)
                bytes_per_sec = (size_bytes / wall) if wall > 0 else 0.0
                rows.append({
                    "algo": algo,
                    "repeat": rep,
                    "size_bytes": size_bytes,
                    "train_wall_time_s": wall,
                    "train_cpu_time_s": cpu,
                    "peak_rss_kb": rss,
                    "bytes_per_sec": bytes_per_sec,
                    "run_dir": str(run_dir.resolve()),
                })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "microbench.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "algo", "repeat", "size_bytes", "train_wall_time_s", "train_cpu_time_s", "peak_rss_kb", "bytes_per_sec", "run_dir",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(str(csv_path.resolve()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
