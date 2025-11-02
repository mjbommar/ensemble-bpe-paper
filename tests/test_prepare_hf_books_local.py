from __future__ import annotations

import os
from pathlib import Path
import subprocess
import json


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_prepare_hf_books_local_mode(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    local_dir = tmp_path / "local"
    _write(local_dir / "a.txt", "This is one doc.\nAnother line.\n")
    _write(local_dir / "b.txt", "Third line here.\nFourth line too.\n")

    out = subprocess.check_output(
        [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.prepare_hf_books",
            "--local-files",
            str(local_dir),
            "--out",
            str(tmp_path / "processed"),
            "--sample-train",
            "2",
            "--sample-valid",
            "1",
            "--sample-test",
            "1",
            "--seed",
            "7",
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    # Last JSON line contains the summary
    line = [ln.strip() for ln in out.splitlines() if ln.strip()][-1]
    summary = json.loads(line)
    out_dir = Path(summary["out_dir"])  # type: ignore
    assert (out_dir / "train.txt").exists()
    assert (out_dir / "valid.txt").exists()
    assert (out_dir / "test.txt").exists()
    # Counts match our sampling
    assert summary["counts"]["train"] == 2
    assert summary["counts"]["valid"] == 1
    assert summary["counts"]["test"] == 1
