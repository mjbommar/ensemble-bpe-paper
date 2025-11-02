from pathlib import Path
import os
import json
import subprocess
import sys


def write_lines(p: Path, n: int, prefix: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(f"{prefix} line {i}" for i in range(n)) + "\n", encoding="utf-8")


def test_train_ensemble_records_shard_stats(tmp_path: Path):
    train = tmp_path / "train.txt"
    evalf = tmp_path / "eval.txt"
    write_lines(train, 40, "train")
    write_lines(evalf, 10, "eval")

    env = dict(**os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    out = subprocess.check_output([
        sys.executable,
        "-m",
        "scripts.train_ensemble",
        "--exp", "ens_stats",
        "--train", str(train),
        "--eval-file", str(evalf),
        "--num-shards", "2",
        "--vocab-size", "64",
        "--out", str(tmp_path / "arts"),
    ], stderr=subprocess.STDOUT, env=env).decode("utf-8")
    ens_dir = Path([ln.strip() for ln in out.splitlines() if ln.strip()][-1])
    j = json.loads((ens_dir / "ensemble.json").read_text(encoding="utf-8"))
    assert "members" in j and len(j["members"]) == 2
    for m in j["members"]:
        st = m.get("shard_stats")
        assert st is not None
        # basic fields present and non-negative
        assert set(st.keys()) >= {"lines", "bytes", "uniq_bytes", "entropy_bits", "chars", "uniq_chars", "char_entropy_bits"}
        assert st["lines"] > 0 and st["bytes"] > 0
