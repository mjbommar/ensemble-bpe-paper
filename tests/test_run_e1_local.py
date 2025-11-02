from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap
import csv


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_e1_local_dict_provider(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    # Minimal dictionary config
    dict_dir = tmp_path / "dict"
    _write(dict_dir / "en", "hello\nworld\napple\nbanana\n")

    cfg = tmp_path / "e1.toml"
    _write(
        cfg,
        textwrap.dedent(
            f"""
            [data]
            provider = "dict"
            dict_path = "{dict_dir}"
            out_dir = "{tmp_path / 'processed'}"
            train_size = 4
            valid_size = 2
            test_size = 2
            lower = true
            seed = 13

            [train]
            exp_name = "e1_base"
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
            "scripts.run_e1",
            "--config",
            str(cfg),
            "--out",
            str(tmp_path / "e1"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    csv_path = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    # Should have entries for at least BPE and WordPiece
    algos = {r["algo"] for r in rows}
    assert {"hf_bpe", "hf_wordpiece"}.issubset(algos)
    # All rows must have non-empty numbers for metrics
    for r in rows:
        assert float(r["train_wall_time_s"]) >= 0
        assert int(float(r["peak_rss_kb"])) >= 0

