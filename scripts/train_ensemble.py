"""
Train an ensemble of HF BPE tokenizers on shards of the training data,
evaluate each on a held-out file, and select the best-performing member.

Selection criterion (default): minimum tokens_per_byte on eval file.

Example:
  uv run --with tokenizers --with psutil python -m scripts.train_ensemble \
      --exp hf_bpe_ens --train data/processed/train.txt --eval-file data/processed/test.txt \
      --num-shards 3 --vocab-size 2000
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List
from collections import Counter
import math

from ebpe.artifact_store import create_run_dir, default_artifact_paths, write_json, write_text


def _last_line(s: str) -> str:
    return [ln.strip() for ln in s.splitlines() if ln.strip()][-1]


def _run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace")


def shard_lines(in_path: Path, k: int, out_dir: Path) -> List[Path]:
    lines = in_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = len(lines)
    if k <= 0:
        raise ValueError("num-shards must be >= 1")
    size = max(1, math.ceil(n / k))
    out_paths: List[Path] = []
    for i in range(k):
        chunk = lines[i * size : (i + 1) * size]
        if not chunk:
            break
        p = out_dir / f"train_shard_{i}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        out_paths.append(p)
    return out_paths


def _compute_shard_stats(path: Path) -> Dict[str, float]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    data = txt.encode("utf-8", errors="ignore")
    lines = len(txt.splitlines())
    by = len(data)
    ch = len(txt)
    cnt = Counter(data)
    uniq = len(cnt)
    if by > 0:
        probs = [c / by for c in cnt.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    else:
        entropy = 0.0
    # Character-level stats (Unicode code points)
    cc = Counter(txt)
    uniq_chars = len(cc)
    if ch > 0:
        cprobs = [c / ch for c in cc.values()]
        char_entropy = -sum(p * math.log2(p) for p in cprobs if p > 0)
    else:
        char_entropy = 0.0
    return {
        "lines": float(lines),
        "bytes": float(by),
        "uniq_bytes": float(uniq),
        "entropy_bits": float(entropy),
        "chars": float(ch),
        "uniq_chars": float(uniq_chars),
        "char_entropy_bits": float(char_entropy),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--num-shards", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args(argv)

    ens_dir = create_run_dir(args.exp, base_dir=args.out)
    paths = default_artifact_paths(ens_dir)
    write_text(paths["env"], "ensemble trainer v0\n")

    # Prepare shards
    shards_dir = ens_dir / "shards"
    shards = shard_lines(Path(args.train), args.num_shards, shards_dir)

    members: List[Dict] = []
    for i, shard in enumerate(shards):
        member_exp = f"{args.exp}_m{i}"
        train_cmd = [
            "uv",
            "run",
            "--with",
            "tokenizers",
            "--with",
            "psutil",
            "python",
            "-m",
            "scripts.train_tokenizer",
            "--exp",
            member_exp,
            "--train",
            str(shard),
            "--vocab-size",
            str(args.vocab_size),
            "--out",
            str(args.out),
        ]
        train_out = _run(train_cmd)
        train_run = Path(_last_line(train_out))

        eval_cmd = [
            "uv",
            "run",
            "--with",
            "tokenizers",
            "python",
            "-m",
            "scripts.eval_compression",
            "--exp",
            f"{member_exp}_eval",
            "--tokenizer",
            str(train_run / "tokenizer.json"),
            "--eval-file",
            str(args.eval_file),
            "--out",
            str(args.out),
        ]
        eval_out = _run(eval_cmd)
        eval_run = Path(_last_line(eval_out))
        metrics = json.loads((eval_run / "metrics.json").read_text(encoding="utf-8"))
        members.append(
            {
                "shard": str(shard),
                "shard_stats": _compute_shard_stats(shard),
                "train_run": str(train_run),
                "eval_run": str(eval_run),
                "tokens_per_byte": float(metrics.get("tokens_per_byte", 0.0)),
            }
        )

    # Select best (min tokens_per_byte)
    best = min(members, key=lambda m: m["tokens_per_byte"]) if members else None
    selection = {
        "criterion": "min_tokens_per_byte",
        "members": members,
        "best": best,
    }
    write_json(ens_dir / "ensemble.json", selection)
    if best:
        src = Path(best["train_run"]) / "tokenizer.json"
        dst = ens_dir / "selected_tokenizer.json"
        shutil.copy2(src, dst)

    write_text(paths["logs"], "trained ensemble and selected best member\n")
    print(str(ens_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
