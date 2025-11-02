from __future__ import annotations

import os
from pathlib import Path
import subprocess
import json


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_plot_zipf_produces_json_and_png(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    train = tmp_path / "train.txt"
    eval_file = tmp_path / "eval.txt"
    _write(train, "hello world\nthis is a test\n")
    _write(eval_file, "hello world\nthis is a test\n")

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
            "zipf_tok",
            "--train",
            str(train),
            "--vocab-size",
            "64",
            "--out",
            str(tmp_path / "artifacts"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    run_dir = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    tok_path = run_dir / "tokenizer.json"
    assert tok_path.exists()

    # Run plotter (with matplotlib for PNG)
    out2 = subprocess.check_output(
        [
            "uv",
            "run",
            "--with",
            "tokenizers",
            "--with",
            "matplotlib",
            "python",
            "-m",
            "scripts.plot_zipf",
            "--tokenizer",
            str(tok_path),
            "--eval-file",
            str(eval_file),
            "--out",
            str(tmp_path / "plots"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    json_path = Path([ln.strip() for ln in out2.splitlines() if ln.strip()][-1])
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "r2" in data
    assert (json_path.parent / "zipf.png").exists()

