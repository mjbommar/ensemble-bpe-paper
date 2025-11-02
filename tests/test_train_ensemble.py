from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_train_ensemble_selects_best(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    # Build a small training file intentionally with repeated patterns
    train = tmp_path / "train.txt"
    eval_file = tmp_path / "eval.txt"
    _write(train, "\n".join(["apple banana", "banana apple", "cherry cherry", "hello world"]) + "\n")
    _write(eval_file, "hello world\napple banana\n")

    out = subprocess.check_output(
        [
            "uv",
            "run",
            "--with",
            "tokenizers",
            "--with",
            "psutil",
            "python",
            "-m",
            "scripts.train_ensemble",
            "--exp",
            "hf_bpe_ens_pytest",
            "--train",
            str(train),
            "--eval-file",
            str(eval_file),
            "--num-shards",
            "2",
            "--vocab-size",
            "64",
            "--out",
            str(tmp_path / "artifacts"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    ens_dir = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    assert (ens_dir / "ensemble.json").exists()
    assert (ens_dir / "selected_tokenizer.json").exists()

    sel = json.loads((ens_dir / "ensemble.json").read_text(encoding="utf-8"))
    toks = [m["tokens_per_byte"] for m in sel["members"]]
    assert abs(min(toks) - sel["best"]["tokens_per_byte"]) < 1e-9

    # Merge-voting: build a merged tokenizer and ensure it loads
    m_out = subprocess.check_output(
        [
            "uv",
            "run",
            "--with",
            "tokenizers",
            "python",
            "-m",
            "scripts.merge_ensemble",
            "--exp",
            "hf_bpe_ens_merged_pytest",
            "--ensemble-run",
            str(ens_dir),
            "--k",
            "1",
            "--out",
            str(tmp_path / "artifacts"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")
    merged_dir = Path([ln.strip() for ln in m_out.splitlines() if ln.strip()][-1])
    merged_tok = merged_dir / "tokenizer.json"
    assert merged_tok.exists()
    from tokenizers import Tokenizer  # type: ignore

    tok = Tokenizer.from_file(str(merged_tok))
    s = "hello world"
    assert tok.decode(tok.encode(s).ids)  # decodes to a non-empty string
