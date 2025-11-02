from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap
import json


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_experiment_oos_hf_local_provider(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    # In-sample local texts
    in_local = tmp_path / "local_in"
    _write(in_local / "a.txt", "This is one doc.\nAnother line.\n")
    # OOS local texts
    oos_local = tmp_path / "local_oos"
    _write(oos_local / "b.txt", "Completely different domain.\nYet another line.\n")

    cfg = tmp_path / "exp_hf_oos_local.toml"
    _write(
        cfg,
        textwrap.dedent(
            f"""
            [data]
            provider = "hf_books"
            local_files = "{in_local}"
            out_dir = "{tmp_path / 'processed_in'}"
            train_size = 2
            valid_size = 1
            test_size = 1
            seed = 13

            [oos_data]
            provider = "hf_books"
            local_files = "{oos_local}"
            out_dir = "{tmp_path / 'processed_oos'}"
            test_size = 1
            seed = 7

            [train]
            exp_name = "hf_bpe_pytest_pipeline_hf_oos_local"
            vocab_size = 64

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
    assert any(ln.endswith("_oos") or "_oos," in ln for ln in summ)

