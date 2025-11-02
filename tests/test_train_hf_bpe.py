from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_train_hf_bpe_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}" + (":" + env.get("PYTHONPATH", ""))

    # Create a tiny training corpus
    train = tmp_path / "train.txt"
    _write(
        train,
        "\n".join(
            [
                "apple",
                "banana",
                "cherry",
                "durian",
                "éclair",
                "fromage",
                "hello",
                "world",
                "punctuation,;:!",
            ]
        )
        + "\n",
    )

    # Train
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
            "hf_bpe_pytest",
            "--train",
            str(train),
            "--vocab-size",
            "128",
            "--out",
            str(tmp_path / "artifacts"),
        ],
        env=env,
        stderr=subprocess.STDOUT,
    ).decode("utf-8")

    run_dir = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    tok_path = run_dir / "tokenizer.json"
    metrics_path = run_dir / "metrics.json"
    assert tok_path.exists(), "tokenizer.json must exist"
    assert metrics_path.exists(), "metrics.json must exist"
    import json
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert m["train_wall_time_s"] >= 0
    assert m["train_cpu_time_s"] >= 0
    assert m["peak_rss_kb"] >= 0

    # HF compatibility check: load and round-trip a few strings
    # (Import within uv is not needed for reading the JSON file)
    from tokenizers import Tokenizer  # type: ignore

    tok = Tokenizer.from_file(str(tok_path))
    samples = [
        "apple banana",
        "hello, world!",
        "éclair fromage",
        "punctuation,;:!",
    ]
    for s in samples:
        out = tok.decode(tok.encode(s).ids)
        assert out == s, f"round-trip failed for: {s!r} -> {out!r}"
