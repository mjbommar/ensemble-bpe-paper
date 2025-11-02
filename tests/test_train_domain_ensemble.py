from __future__ import annotations

import os
from pathlib import Path
import subprocess
import json


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_train_domain_ensemble_selects_best(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    d1 = tmp_path / "d1.txt"
    d2 = tmp_path / "d2.txt"
    eval_file = tmp_path / "eval.txt"
    _write(d1, "hello world\nhello banana\n")
    _write(d2, "cherry cherry\nbanana banana\n")
    _write(eval_file, "hello world\nbanana\n")

    out = subprocess.check_output(
        [
            "uv", "run", "--with", "tokenizers", "--with", "psutil",
            "python", "-m", "scripts.train_domain_ensemble",
            "--exp", "ds_ens_pytest",
            "--train-files", str(d1), str(d2),
            "--eval-file", str(eval_file),
            "--algo", "hf_bpe",
            "--vocab-size", "64",
            "--out", str(tmp_path / "artifacts"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    ens_dir = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    assert (ens_dir / "domain_ensemble.json").exists()
    assert (ens_dir / "selected_tokenizer.json").exists()
    sel = json.loads((ens_dir / "domain_ensemble.json").read_text(encoding="utf-8"))
    toks = [m["tokens_per_byte"] for m in sel["members"]]
    assert abs(min(toks) - sel["best"]["tokens_per_byte"]) < 1e-9

