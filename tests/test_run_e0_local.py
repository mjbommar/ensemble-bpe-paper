from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_e0_local_dict_provider(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    # Build tiny in-sample and OOS dictionaries
    dict_dir = tmp_path / "dict_in"
    oos_dir = tmp_path / "dict_oos"
    _write(dict_dir / "en", "apple\nbanana\ncherry\nhello\nworld\n")
    _write(oos_dir / "fr", "bonjour\nmonde\nfromage\npoire\n")

    single = tmp_path / "single.toml"
    _write(
        single,
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
            exp_name = "e0_single"
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

    ensemble = tmp_path / "ensemble.toml"
    _write(
        ensemble,
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
            exp_name = "e0_ensemble"
            vocab_size = 64

            [ensemble]
            enabled = true
            num_shards = 2
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
            "scripts.run_e0",
            "--single",
            str(single),
            "--ensemble",
            str(ensemble),
            "--out",
            str(tmp_path / "e0"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    summary_csv = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    assert summary_csv.exists()
    # Ensure the file contains rows from both pipelines
    lines = summary_csv.read_text(encoding="utf-8").splitlines()
    assert any("e0_single" in ln for ln in lines)
    assert any("ensemble_selected" in ln or "e0_ensemble" in ln for ln in lines)

