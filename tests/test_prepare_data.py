from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_prepare_data_splits_and_manifest(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # Create fake dictionary directories
    dict_in = tmp_path / "dict_in"
    dict_out = tmp_path / "dict_out"
    _write(dict_in / "en", "apple\nbanana\nCherry\nalpha\nbanana\nBeta\n")
    _write(dict_in / "fr", "éclair\nfromage\nbanane\n")
    _write(dict_out / "misc", "kiwi\nmelon\nnectarine\napple\n")  # apple overlaps, should be filtered

    out_dir = tmp_path / "processed"
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "scripts.prepare_data",
        "--dict-path",
        str(dict_in),
        "--oos-dict-path",
        str(dict_out),
        "--out",
        str(out_dir),
        "--train-size",
        "3",
        "--valid-size",
        "2",
        "--test-size",
        "2",
        "--oos-size",
        "2",
        "--lower",
        "--seed",
        "42",
    ]
    out = subprocess.check_output(cmd, env=env, stderr=subprocess.STDOUT).decode("utf-8")

    # Check files
    for fname in ["train.txt", "valid.txt", "test.txt", "test_oos.txt", "manifest.json"]:
        assert (out_dir / fname).exists(), f"missing {fname}"

    # Check counts
    train = (out_dir / "train.txt").read_text(encoding="utf-8").strip().splitlines()
    valid = (out_dir / "valid.txt").read_text(encoding="utf-8").strip().splitlines()
    test = (out_dir / "test.txt").read_text(encoding="utf-8").strip().splitlines()
    oos = (out_dir / "test_oos.txt").read_text(encoding="utf-8").strip().splitlines()

    assert len(train) == 3
    assert len(valid) == 2
    assert len(test) == 2
    assert len(oos) == 2

    # Verify OOS has no overlap with in-sample pool
    in_pool = set(train + valid + test)
    assert all(w not in in_pool for w in oos)

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["train"] == 3
    assert manifest["seed"] == 42
    assert manifest["lower"] is True

