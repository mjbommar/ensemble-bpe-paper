from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_train_wordpiece_roundtrip_basic(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    train = tmp_path / "train.txt"
    _write(train, "hello world\napple banana\n")

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
            "scripts.train_wordpiece",
            "--exp",
            "hf_wp_pytest",
            "--train",
            str(train),
            "--vocab-size",
            "64",
            "--out",
            str(tmp_path / "artifacts"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    run_dir = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    tok_path = run_dir / "tokenizer.json"
    assert tok_path.exists()

    # HF-compat: load via tokenizers and ensure simple round-trip on lowercase inputs
    from tokenizers import Tokenizer  # type: ignore

    tok = Tokenizer.from_file(str(tok_path))
    sample = "hello world"
    enc = tok.encode(sample)
    dec = tok.decode(enc.ids)
    assert isinstance(enc.ids, list) and len(enc.ids) > 0
    assert dec == sample

