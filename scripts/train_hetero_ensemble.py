"""
Train a heterogeneous ensemble: multiple algorithms on the same train file,
evaluate each on an eval file, and select the best (min tokens_per_byte).

Example:
  uv run --with tokenizers --with psutil python -m scripts.train_hetero_ensemble \
    --exp het_ens --train data/train.txt --eval-file data/test.txt \
    --algos hf_bpe hf_wordpiece hf_unigram --vocab-size 2000
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from ebpe.artifact_store import create_run_dir, default_artifact_paths, write_json, write_text


def _last_line(s: str) -> str:
    return [ln.strip() for ln in s.splitlines() if ln.strip()][-1]


def _run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--algos", nargs="+", default=["hf_bpe", "hf_wordpiece", "hf_unigram"],
                    help="list of algos: hf_bpe hf_wordpiece hf_unigram")
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args(argv)

    run_dir = create_run_dir(args.exp, base_dir=args.out)
    paths = default_artifact_paths(run_dir)
    write_text(paths["env"], f"hetero ensemble algos={args.algos}\n")

    mod = {
        "hf_bpe": "scripts.train_tokenizer",
        "hf_wordpiece": "scripts.train_wordpiece",
        "hf_unigram": "scripts.train_unigram",
    }

    members: List[Dict] = []
    for algo in args.algos:
        if algo not in mod:
            continue
        member_exp = f"{args.exp}_{algo}"
        train_cmd = [
            "uv", "run", "--with", "tokenizers", "--with", "psutil",
            "python", "-m", mod[algo],
            "--exp", member_exp,
            "--train", str(args.train),
            "--vocab-size", str(args.vocab_size),
            "--out", str(args.out),
        ]
        train_out = _run(train_cmd)
        train_run = Path(_last_line(train_out))

        eval_cmd = [
            "uv", "run", "--with", "tokenizers",
            "python", "-m", "scripts.eval_compression",
            "--exp", f"{member_exp}_eval",
            "--tokenizer", str(train_run / "tokenizer.json"),
            "--eval-file", str(args.eval_file),
            "--out", str(args.out),
        ]
        eval_out = _run(eval_cmd)
        eval_run = Path(_last_line(eval_out))
        metrics = json.loads((eval_run / "metrics.json").read_text(encoding="utf-8"))
        members.append({
            "algo": algo,
            "train_run": str(train_run),
            "eval_run": str(eval_run),
            "tokens_per_byte": float(metrics.get("tokens_per_byte", 0.0)),
        })

    best = min(members, key=lambda m: m["tokens_per_byte"]) if members else None
    write_json(run_dir / "hetero_ensemble.json", {"members": members, "best": best, "criterion": "min_tokens_per_byte"})
    if best:
        src = Path(best["train_run"]) / "tokenizer.json"
        dst = run_dir / "selected_tokenizer.json"
        shutil.copy2(src, dst)
    write_text(paths["logs"], "trained hetero ensemble and selected best member\n")
    print(str(run_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

