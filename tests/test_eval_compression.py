from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _train_tiny_tokenizer(tmp_path: Path, env: dict) -> Path:
    train = tmp_path / "train.txt"
    _write(train, "apple\nbanana\nhello world\néclair fromage\n")

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
            "scripts.train_tokenizer",
            "--exp",
            "hf_bpe_for_eval",
            "--train",
            str(train),
            "--vocab-size",
            "128",
            "--out",
            str(tmp_path / "artifacts"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")
    run_dir = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    assert (run_dir / "tokenizer.json").exists()
    return run_dir / "tokenizer.json"


def test_eval_compression_outputs_metrics(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    tok_path = _train_tiny_tokenizer(tmp_path, env)

    eval_file = tmp_path / "eval.txt"
    _write(eval_file, "hello world\napple banana\nfromage!\n")

    out = subprocess.check_output(
        [
            "uv",
            "run",
            "--with",
            "tokenizers",
            "python",
            "-m",
            "scripts.eval_compression",
            "--exp",
            "hf_bpe_eval_smoke",
            "--tokenizer",
            str(tok_path),
            "--eval-file",
            str(eval_file),
            "--out",
            str(tmp_path / "artifacts"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    run_dir = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    metrics_path = run_dir / "metrics.json"
    metrics_csv = run_dir / "metrics.csv"
    comp_path = run_dir / "compression.json"
    assert metrics_path.exists(), "metrics.json should be written"
    assert metrics_csv.exists(), "metrics.csv should be written"
    assert comp_path.exists(), "compression.json should be written"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["doc_count"] == 3
    assert metrics["total_tokens"] > 0
    # On words, typically tokens_per_char < 1
    assert metrics["tokens_per_char"] < 1.0
