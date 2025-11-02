from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap
import json


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_experiment_oos_hf_local_by_file_auto_detect(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    # Local titles: three files; OOS will be a fraction of files held out
    local_dir = tmp_path / "local_titles"
    _write(local_dir / "t1.txt", "alpha one\nalpha two\n")
    _write(local_dir / "t2.txt", "beta one\nbeta two\n")
    _write(local_dir / "t3.txt", "gamma one\ngamma two\n")

    cfg = tmp_path / "exp_hf_local_by_file.toml"
    _write(
        cfg,
        textwrap.dedent(
            f"""
            [data]
            provider = "hf_books"
            local_files = "{local_dir}"
            out_dir = "{tmp_path / 'processed_hf_local'}"
            train_size = 3
            valid_size = 1
            test_size = 1
            seed = 13
            split_strategy = "by_file"
            oos_frac = 0.34

            [train]
            exp_name = "hf_bpe_pytest_pipeline_hf_local_by_file"
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
            "--with",
            "datasets",
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

    # Verify that the data prep wrote test_oos.txt and the pipeline evaluated OOS automatically
    out_dir = Path((manifest["data"]["local_files"])) if manifest["data"].get("local_files") else None  # type: ignore
    processed = tmp_path / "processed_hf_local"
    assert (processed / "test_oos.txt").exists()

    oos_dir_path = manifest.get("oos_eval_run_dir")
    assert oos_dir_path, "Expected OOS eval run in manifest"
    oos_run = Path(oos_dir_path)
    assert (oos_run / "metrics.json").exists()

    # Summary includes OOS row
    summ = (pipe_dir / "pipeline_summary.csv").read_text(encoding="utf-8").splitlines()
    assert any("_oos" in ln for ln in summ)

