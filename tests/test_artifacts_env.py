from __future__ import annotations

import os
from pathlib import Path
import os
import subprocess


def test_record_env_creates_run_dir_and_env(tmp_path: Path):
    # Run the recorder as a module so import path is consistent
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    out = subprocess.check_output(
        [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.record_env",
            "--exp",
            "pytest_smoke",
            "--out",
            str(tmp_path),
        ],
        stderr=subprocess.STDOUT,
        env=env,
    ).decode("utf-8")

    # uv may emit warnings on stdout; take the last non-empty line as the path
    last_line = [ln.strip() for ln in out.splitlines() if ln.strip()][-1]
    run_dir = Path(last_line)
    assert run_dir.exists(), "run directory should exist"
    env_file = run_dir / "env.txt"
    logs_file = run_dir / "logs.txt"
    assert env_file.exists(), "env.txt must be written"
    assert logs_file.exists(), "logs.txt must be written"

    # Validate minimum content
    txt = env_file.read_text(encoding="utf-8")
    assert "Python:" in txt
    assert "uv:" in txt  # we invoked via uv
