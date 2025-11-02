from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_e2_local_dicts(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    dict_in = tmp_path / "dict_in"
    dict_oos = tmp_path / "dict_oos"
    _write(dict_in / "en", "hello\nworld\napple\nbanana\ncherry\n")
    _write(dict_oos / "fr", "bonjour\nmonde\nfromage\npoire\npeche\n")

    out = subprocess.check_output(
        [
            "uv", "run",
            "--with", "tokenizers",
            "--with", "psutil",
            "python", "-m", "scripts.run_e2",
            "--dict-path", str(dict_in),
            "--oos-dict-path", str(dict_oos),
            "--out", str(tmp_path / "e2"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    # e2_summary should exist via E0 aggregator
    summary = Path(tmp_path / "e2" / "e0_summary.csv")
    assert summary.exists()
    lines = summary.read_text(encoding="utf-8").splitlines()
    assert any("_oos" in ln for ln in lines)

