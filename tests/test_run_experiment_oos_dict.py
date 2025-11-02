from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap
import json


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_experiment_oos_dict_provider(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    # Build tiny in-sample and OOS dictionaries
    dict_dir = tmp_path / "dict_in"
    oos_dir = tmp_path / "dict_oos"
    _write(dict_dir / "en", "apple\nbanana\ncherry\nhello\nworld\n")
    _write(oos_dir / "fr", "bonjour\nmonde\nfromage\npoire\n")

    cfg = tmp_path / "exp_oos.toml"
    _write(
        cfg,
        textwrap.dedent(
            f"""
            [data]
            provider = "dict"
            dict_path = "{dict_dir}"
            oos_dict_path = "{oos_dir}"
            out_dir = "{tmp_path / 'processed'}"
            train_size = 5
            valid_size = 2
            test_size = 2
            oos_size = 3
            lower = true
            seed = 13

            [train]
            exp_name = "hf_bpe_pytest_pipeline_oos"
            vocab_size = 128

            [eval]
            eval_split = "test"
            run_oos = true

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
    oos_dir_path = manifest.get("oos_eval_run_dir")
    assert oos_dir_path, "Expected OOS eval run in manifest"
    oos_run = Path(oos_dir_path)
    assert (oos_run / "metrics.json").exists()

    # Summary includes OOS row
    summ = (pipe_dir / "pipeline_summary.csv").read_text(encoding="utf-8").splitlines()
    assert any("_oos" in ln for ln in summ)

