from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_eval_compression_tiktoken(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    eval_file = tmp_path / "eval.txt"
    _write(eval_file, "hello world\napple banana\n")

    out = subprocess.check_output(
        [
            "uv",
            "run",
            "--with",
            "tiktoken",
            "python",
            "-m",
            "scripts.eval_compression_tiktoken",
            "--exp",
            "tiktoken_eval_pytest",
            "--encoding",
            "gpt2",
            "--eval-file",
            str(eval_file),
            "--out",
            str(tmp_path / "artifacts"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    run_dir = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    assert (run_dir / "metrics.json").exists()
    # Simple sanity: tokens_per_byte should be > 0 for non-empty content
    import json

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["tokens_per_byte"] > 0

