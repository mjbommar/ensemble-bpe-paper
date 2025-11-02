"""
Run the E2 experiment: multilingual dictionary stress test.

Compares a single HF-BPE model vs an ensemble selection on an English-like
in-sample dictionary with an OOS dictionary (e.g., French). Uses the pipeline's
dictionary provider and OOS path, then aggregates results like E0.

Example:
  uv run --with tokenizers --with psutil python -m scripts.run_e2 \
      --dict-path /usr/share/dict/american-english \
      --oos-dict-path /usr/share/dict/french
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def _run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _cfg(dict_path: str, oos_dict_path: str, out_dir: str, exp: str, ensemble: bool) -> str:
    lines = []
    lines.append("[data]")
    lines.append(f"provider = \"dict\"")
    lines.append(f"dict_path = \"{dict_path}\"")
    lines.append(f"oos_dict_path = \"{oos_dict_path}\"")
    lines.append(f"out_dir = \"{out_dir}\"")
    lines.append("train_size = 5000")
    lines.append("valid_size = 500")
    lines.append("test_size = 500")
    lines.append("oos_size = 500")
    lines.append("lower = true")
    lines.append("seed = 13\n")

    lines.append("[train]")
    lines.append(f"exp_name = \"{exp}\"")
    lines.append("vocab_size = 2000\n")

    if ensemble:
        lines.append("[ensemble]")
        lines.append("enabled = true")
        lines.append("num_shards = 3")
        lines.append("vocab_size = 2000\n")

    lines.append("[eval]")
    lines.append("eval_split = \"test\"")
    lines.append("run_oos = true\n")

    lines.append("[artifacts]")
    lines.append("out_dir = \"artifacts\"\n")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict-path", required=True)
    ap.add_argument("--oos-dict-path", required=True)
    ap.add_argument("--out", default="artifacts/e2")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as td:
        td = str(td)
        single = Path(td) / "single.toml"
        ens = Path(td) / "ens.toml"
        _write(single, _cfg(args.dict_path, args.oos_dict_path, f"{td}/processed", "e2_single", ensemble=False))
        _write(ens, _cfg(args.dict_path, args.oos_dict_path, f"{td}/processed", "e2_ensemble", ensemble=True))

        # Reuse E0 runner to aggregate
        out = _run([
            "uv", "run", "--with", "tokenizers", "--with", "psutil",
            "python", "-m", "scripts.run_e0",
            "--single", str(single),
            "--ensemble", str(ens),
            "--out", args.out,
        ])
        print(out.strip())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

