from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap
import json


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_experiment_with_merge_voting(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    # Tiny dict data
    dict_dir = tmp_path / "dict"
    _write(dict_dir / "en", "apple\nbanana\ncherry\nhello\nworld\n")

    cfg = tmp_path / "exp_merge.toml"
    _write(
        cfg,
        textwrap.dedent(
            f"""
            [data]
            dict_path = "{dict_dir}"
            out_dir = "{tmp_path / 'processed'}"
            train_size = 5
            valid_size = 2
            test_size = 2
            seed = 13

            [train]
            exp_name = "hf_bpe_pytest_pipeline_merge"
            vocab_size = 128

            [ensemble]
            enabled = true
            num_shards = 2
            vocab_size = 128
            [ensemble.merge]
            enabled = true
            k = 1

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
    summ = (pipe_dir / "pipeline_summary.csv").read_text(encoding="utf-8").splitlines()
    assert any("ensemble_merge" in ln for ln in summ)

