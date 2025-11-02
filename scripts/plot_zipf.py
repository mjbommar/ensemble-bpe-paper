"""
Compute and visualize Zipf's law for a tokenizer on an evaluation file.

Outputs:
  - zipf.json with R^2 of log(rank) vs log(freq)
  - zipf.png (optional; requires matplotlib) showing the log-log plot

Example:
  uv run --with tokenizers --with matplotlib python -m scripts.plot_zipf \
    --tokenizer artifacts/<run>/tokenizer.json --eval-file data/processed/test.txt \
    --out artifacts/plots
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict


def _load_tokenizer(tokenizer_path: str):
    from tokenizers import Tokenizer  # type: ignore

    return Tokenizer.from_file(tokenizer_path)


def _file_lines(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    for line in txt.splitlines():
        yield line.rstrip("\n")


def compute_zipf_r2(tokenizer, eval_path: Path) -> Dict[str, float]:
    from collections import Counter

    freq = Counter()
    for line in _file_lines(eval_path):
        enc = tokenizer.encode(line)
        freq.update(enc.ids)

    if not freq:
        return {"r2": 0.0, "n_tokens": 0}

    # Sort by frequency desc, compute log-rank and log-freq
    counts = [c for _, c in freq.most_common()]
    xs = [math.log(i + 1) for i in range(len(counts))]
    ys = [math.log(c) for c in counts]

    # Simple linear regression for R^2
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    slope = ss_xy / ss_xx if ss_xx else 0.0
    intercept = mean_y - slope * mean_x
    y_hat = [slope * x + intercept for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, y_hat))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 0.0
    return {"r2": float(r2), "n_tokens": int(sum(counts))}


def maybe_plot(tokenizer_name: str, eval_name: str, xs, ys, out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.figure(figsize=(5, 4))
    plt.scatter(xs, ys, s=6, alpha=0.6)
    plt.title(f"Zipf log-log: {tokenizer_name} on {eval_name}")
    plt.xlabel("log(rank)")
    plt.ylabel("log(freq)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--out", default="artifacts/plots")
    args = ap.parse_args(argv)

    tok = _load_tokenizer(args.tokenizer)
    eval_path = Path(args.eval_file)

    # Compute stats and write JSON
    stats = compute_zipf_r2(tok, eval_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "zipf.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    # Optionally plot
    # Recompute xs, ys to avoid storing large arrays in JSON
    from collections import Counter

    freq = Counter()
    for line in _file_lines(eval_path):
        enc = tok.encode(line)
        freq.update(enc.ids)
    counts = [c for _, c in freq.most_common()]
    xs = [math.log(i + 1) for i in range(len(counts))]
    ys = [math.log(c) for c in counts]
    maybe_plot("hf_tokenizer", eval_path.name, xs, ys, out_dir / "zipf.png")

    print(str((out_dir / "zipf.json").resolve()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

