from __future__ import annotations

from pathlib import Path
import os
import subprocess


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_aggregate_metrics_collects_multiple_summaries(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    # Create two pipeline dirs with small summaries
    p1 = tmp_path / "artifacts" / "2025-11-01_exp1_pipeline"
    p2 = tmp_path / "artifacts" / "2025-11-01_exp2_pipeline"
    _write(p1 / "pipeline_summary.csv", "name,kind,run_dir,tokens_per_byte\nexp1,hf_bpe,/tmp/x,0.50\n")
    _write(p2 / "pipeline_summary.csv", "name,kind,run_dir,tokens_per_byte\nexp2,hf_wordpiece,/tmp/y,0.55\n")

    out = subprocess.check_output(
        [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.aggregate_metrics",
            "--inputs",
            str(tmp_path / "artifacts"),
            "--out",
            str(tmp_path / "agg"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    csv_path = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    assert csv_path.exists()
    # CSV should have header + 2 rows
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "exp1" in lines[1] and "exp2" in lines[2]

