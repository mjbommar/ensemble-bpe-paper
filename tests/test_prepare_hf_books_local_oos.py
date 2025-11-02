from __future__ import annotations

import os
from pathlib import Path
import subprocess
import json


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_prepare_hf_books_local_mode_oos_by_file(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    local_dir = tmp_path / "local"
    _write(local_dir / "a.txt", "a1\na2\n")
    _write(local_dir / "b.txt", "b1\nb2\n")
    _write(local_dir / "c.txt", "c1\nc2\n")

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
            "3",
            "--sample-valid",
            "1",
            "--sample-test",
            "1",
            "--seed",
            "13",
            "--split-strategy",
            "by_file",
            "--oos-frac",
            "0.34",
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
    # OOS file should exist and contain complete-file documents from the held-out file(s)
    assert (out_dir / "test_oos.txt").exists()
    # Ensure counts are non-zero and deterministic
    with (out_dir / "test_oos.txt").open("r", encoding="utf-8") as fh:
        oos_lines = [ln.strip() for ln in fh if ln.strip()]
    assert len(oos_lines) >= 2  # at least one file worth of lines
    # Manifest counts include test_oos
    assert summary["counts"]["test_oos"] == len(oos_lines)

