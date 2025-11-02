"""
Train a domain-specific ensemble: one tokenizer per provided domain file,
evaluate on a held-out eval file, and select the best.

Optionally, if using HF BPE, build a merge‑voted tokenizer across domains.

Example:
  uv run --with tokenizers --with psutil python -m scripts.train_domain_ensemble \
    --exp ds_ens --train-files data/domain_a.txt data/domain_b.txt \
    --eval-file data/test.txt --algo hf_bpe --vocab-size 2000
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
    ap.add_argument("--train-files", nargs="+", required=True, help="one file per domain")
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--algo", choices=["hf_bpe", "hf_wordpiece", "hf_unigram"], default="hf_bpe")
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--merge-k", type=int, help="if set and algo=hf_bpe, build k-of-n merged tokenizer")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args(argv)

    run_dir = create_run_dir(args.exp, base_dir=args.out)
    paths = default_artifact_paths(run_dir)
    write_text(paths["env"], f"domain ensemble algo={args.algo}\n")

    module = {
        "hf_bpe": "scripts.train_tokenizer",
        "hf_wordpiece": "scripts.train_wordpiece",
        "hf_unigram": "scripts.train_unigram",
    }[args.algo]

    members: List[Dict] = []
    for i, train_path in enumerate(args.train_files):
        member_exp = f"{args.exp}_d{i}"
        train_cmd = [
            "uv", "run", "--with", "tokenizers", "--with", "psutil",
            "python", "-m", module,
            "--exp", member_exp,
            "--train", str(train_path),
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
            "domain_train": str(train_path),
            "train_run": str(train_run),
            "eval_run": str(eval_run),
            "tokens_per_byte": float(metrics.get("tokens_per_byte", 0.0)),
        })

    best = min(members, key=lambda m: m["tokens_per_byte"]) if members else None
    selection = {"criterion": "min_tokens_per_byte", "members": members, "best": best}
    write_json(run_dir / "domain_ensemble.json", selection)
    if best:
        src = Path(best["train_run"]) / "tokenizer.json"
        dst = run_dir / "selected_tokenizer.json"
        shutil.copy2(src, dst)

    # Optional merge for BPE
    if args.algo == "hf_bpe" and args.merge_k:
        ens_cmd = [
            "uv", "run", "--with", "tokenizers",
            "python", "-m", "scripts.merge_ensemble",
            "--exp", f"{args.exp}_merge",
            "--members",
            ",".join([str(Path(m["train_run"]) / "tokenizer.json") for m in members]),
            "--k", str(args.merge_k),
            "--out", str(args.out),
        ]
        m_out = _run(ens_cmd)
        write_text(paths["logs"], f"merge_dir={_last_line(m_out)}\n")

    write_text(paths["logs"], "trained domain ensemble and selected best member\n")
    print(str(run_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

