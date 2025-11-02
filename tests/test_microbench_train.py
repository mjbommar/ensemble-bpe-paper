from __future__ import annotations

import os
from pathlib import Path
import subprocess
import csv


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_microbench_train_runs(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    train = tmp_path / "train.txt"
    _write(train, "hello world\nthis is a test\n")

    out = subprocess.check_output(
        [
            "uv", "run",
            "--with", "tokenizers",
            "--with", "psutil",
            "python", "-m", "scripts.microbench_train",
            "--train", str(train),
            "--algos", "hf_bpe",
            "--repeats", "1", "2",
            "--vocab-size", "64",
            "--out", str(tmp_path / "artifacts"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    csv_path = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert len(rows) >= 2
    # bytes_per_sec should be non-negative
    for r in rows:
        assert float(r["bytes_per_sec"]) >= 0

