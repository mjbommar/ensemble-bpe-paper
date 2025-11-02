from pathlib import Path
import json


def write_lines(p: Path, lines: list[str]):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_prepare_hf_books_local_by_hash_oos(tmp_path: Path):
    # Create small local corpus (by_line)
    docs = [f"doc {i} line" for i in range(20)]
    local_dir = tmp_path / "local"
    write_lines(local_dir / "a.txt", docs[:10])
    write_lines(local_dir / "b.txt", docs[10:])

    out_dir = tmp_path / "out"
    # Use by_line mode with oos-by-hash over the test pool
    import subprocess, sys

    cmd = [
        sys.executable,
        "-m",
        "scripts.prepare_hf_books",
        "--local-files",
        str(local_dir),
        "--out",
        str(out_dir),
        "--sample-train",
        "10",
        "--sample-valid",
        "5",
        "--sample-test",
        "5",
        "--oos-by-hash",
        "--oos-frac",
        "0.4",
        "--seed",
        "13",
    ]
    subprocess.check_call(cmd)

    # Validate files and determinism
    train = (out_dir / "train.txt").read_text(encoding="utf-8").splitlines()
    test = (out_dir / "test.txt").read_text(encoding="utf-8").splitlines()
    oos = (out_dir / "test_oos.txt").read_text(encoding="utf-8").splitlines()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

    assert len(train) == 10
    assert len(test) + len(oos) == 5  # test pool was split into IS+OOS
    assert manifest["counts"]["test_oos"] == len(oos)
    # No overlap between test and OOS
    assert set(test).isdisjoint(set(oos))

    # Rerun with same seed: outputs deterministic
    out_dir2 = tmp_path / "out2"
    cmd[cmd.index(str(out_dir))] = str(out_dir2)
    subprocess.check_call(cmd)
    oos2 = (out_dir2 / "test_oos.txt").read_text(encoding="utf-8").splitlines()
    assert oos2 == oos

