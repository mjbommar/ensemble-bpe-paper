from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_transformers_fast_compat_with_hf_bpe(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    train = tmp_path / "train.txt"
    _write(train, "hello world\nthis is a test\n")

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
            "scripts.train_tokenizer",
            "--exp",
            "hf_bpe_tf_compat",
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

    # Load via transformers fast tokenizer
    code = f"""
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(tokenizer_file=r"{tok_path}", unk_token="[UNK]")
s = "hello world"
ids = tok.encode(s)
dec = tok.decode(ids)
print(len(ids) > 0 and isinstance(dec, str))
"""
    ok = subprocess.check_output(
        [
            "uv",
            "run",
            "--with",
            "transformers",
            "python",
            "-c",
            code,
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")
    # The first non-empty 'word' on the last line should be 'True'
    last = [ln.strip() for ln in ok.splitlines() if ln.strip()][-1]
    assert last.endswith("True")
