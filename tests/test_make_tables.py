from __future__ import annotations

from pathlib import Path
import os
import subprocess


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_make_tables_from_aggregated(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    agg = tmp_path / "aggregated.csv"
    _write(
        agg,
        "name,kind,run_dir,tokens_per_byte\na,hf_bpe,/tmp/x,0.9\nb,ensemble_selected,/tmp/y,0.88\n",
    )

    out = subprocess.check_output(
        [
            "uv", "run",
            "python", "-m", "scripts.make_tables",
            "--summary-csv", str(agg),
            "--out", str(tmp_path / "results"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    md = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    assert md.exists()
    txt = md.read_text(encoding="utf-8")
    assert "hf_bpe" in txt and "ensemble_selected" in txt
    assert (md.parent / "table.tex").exists()

