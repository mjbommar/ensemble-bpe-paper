from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap
import json


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_experiment_with_ensemble(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    # Tiny dictionary corpus
    dict_dir = tmp_path / "dict"
    _write(dict_dir / "en", "apple\nbanana\ncherry\nhello\nworld\n")

    cfg = tmp_path / "exp_ens.toml"
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
            lower = true
            seed = 13

            [train]
            exp_name = "hf_bpe_pytest_pipeline_ens"
            vocab_size = 128

            [eval]
            eval_split = "test"

            [ensemble]
            enabled = true
            num_shards = 2
            vocab_size = 128

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
    assert pipe_dir.exists()
    manifest = json.loads((pipe_dir / "pipeline.json").read_text(encoding="utf-8"))
    ens_dir = Path(manifest["ensemble_run_dir"])  # type: ignore
    ens_eval_dir = Path(manifest["ensemble_eval_run_dir"])  # type: ignore

    assert ens_dir.exists()
    assert (ens_dir / "ensemble.json").exists()
    assert (ens_dir / "selected_tokenizer.json").exists()
    # summary CSV should exist and have at least header + one row
    summary = ens_dir / "ensemble_summary.csv"
    assert summary.exists()
    assert len(summary.read_text(encoding="utf-8").splitlines()) >= 2

    # Selected evaluation run should exist with metrics
    assert ens_eval_dir.exists()
    assert (ens_eval_dir / "metrics.json").exists()

