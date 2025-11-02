from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap
import json


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_experiment_hf_books_local_provider(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    # Local text corpus for offline HF provider
    local_dir = tmp_path / "local"
    _write(local_dir / "a.txt", "This is one doc.\nAnother line.\n")
    _write(local_dir / "b.txt", "Third line here.\nFourth line too.\n")

    cfg = tmp_path / "exp_hf_local.toml"
    _write(
        cfg,
        textwrap.dedent(
            f"""
            [data]
            provider = "hf_books"
            local_files = "{local_dir}"
            out_dir = "{tmp_path / 'processed'}"
            train_size = 2
            valid_size = 1
            test_size = 1
            seed = 13

            [train]
            exp_name = "hf_bpe_pytest_pipeline_hf_local"
            vocab_size = 64

            [eval]
            eval_split = "test"

            [artifacts]
            out_dir = "{tmp_path / 'artifacts'}"
            """
        ).strip()
        + "\n",
    )

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
            "scripts.run_experiment",
            "--config",
            str(cfg),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    pipe_dir = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    manifest = json.loads((pipe_dir / "pipeline.json").read_text(encoding="utf-8"))
    train_run = Path(manifest["train_run_dir"])  # type: ignore
    eval_run = Path(manifest["eval_run_dir"])  # type: ignore
    assert (train_run / "tokenizer.json").exists()
    assert (eval_run / "metrics.json").exists()
    # Confirm summary exists
    summ = pipe_dir / "pipeline_summary.csv"
    assert summ.exists() and len(summ.read_text(encoding="utf-8").splitlines()) >= 2

