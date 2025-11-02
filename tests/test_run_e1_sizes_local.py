from pathlib import Path
import csv
import subprocess
import sys


def write_local_corpus(root: Path, n_files: int = 2, lines_per_file: int = 50) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        lines = [f"file{i} line {j} some text" for j in range(lines_per_file)]
        (root / f"f{i}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_run_e1_sizes_local_aggregate(tmp_path: Path):
    # Prepare small local corpus
    local = tmp_path / "local"
    write_local_corpus(local, n_files=2, lines_per_file=40)

    # Create a minimal base TOML using local files provider via prepare_hf_books
    base_toml = tmp_path / "base.toml"
    base_toml.write_text(
        """
[data]
provider = "hf_books"
local_files = "{local}"
out_dir = "{out}"
train_size = 60
valid_size = 10
test_size = 10
seed = 13

[train]
exp_name = "e1_sizes_local"
vocab_size = 128

[eval]
run_oos = true

[oos_data]
provider = "hf_books"
local_files = "{local}"
out_dir = "{oos}"
train_size = 0
valid_size = 0
test_size = 10
seed = 23

[artifacts]
out_dir = "{artifacts}"
""".format(local=str(local), out=str(tmp_path / "proc"), oos=str(tmp_path / "proc_oos"), artifacts=str(tmp_path / "arts")),
        encoding="utf-8",
    )

    # Run the size sweep with two small sizes and only hf_bpe to keep runtime small
    out_dir = tmp_path / "agg"
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_e1_sizes",
        "--config",
        str(base_toml),
        "--sizes",
        "64",
        "128",
        "--algos",
        "hf_bpe",
        "--out",
        str(out_dir),
    ]
    subprocess.check_call(cmd)

    agg_csv = out_dir / "e1_sizes_summary.csv"
    assert agg_csv.exists()
    # Basic schema checks
    rows = list(csv.DictReader(agg_csv.open("r", encoding="utf-8")))
    assert len(rows) == 2  # two sizes * one algo
    for r in rows:
        assert r["algo"] == "hf_bpe"
        assert r["vocab_size"] in {"64", "128"}
        assert float(r["tokens_per_byte"]) >= 0.0

