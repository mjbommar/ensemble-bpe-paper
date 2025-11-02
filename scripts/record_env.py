"""
Record environment details for reproducibility into artifacts/<run>/env.txt.

Usage:
  uv run python -m scripts.record_env --exp EXP_NAME [--out artifacts]
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from ebpe.artifact_store import create_run_dir, default_artifact_paths, write_text


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace").strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return f"<failed: {' '.join(cmd)}>\n{e}"


def gather_env_text() -> str:
    lines: list[str] = []
    lines.append(f"Python: {sys.version}")
    lines.append(f"Platform: {platform.platform()}")
    uv_path = shutil.which("uv")
    if uv_path:
        lines.append(f"uv: {_run(['uv', '--version'])}")
        # best-effort: dependency snapshot
        lines.append("\n# uv pip freeze\n" + _run(["uv", "pip", "freeze"]))
    else:
        lines.append("uv: <not found>")
        lines.append("\n# pip freeze\n" + _run([sys.executable, "-m", "pip", "freeze"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, help="experiment name, e.g., hf_bpe_smoke")
    p.add_argument("--out", default="artifacts", help="artifacts base dir")
    args = p.parse_args(argv)

    run_dir = create_run_dir(args.exp, base_dir=args.out)
    paths = default_artifact_paths(run_dir)
    env_text = gather_env_text()
    write_text(paths["env"], env_text)

    # Also write a small log note for discoverability
    write_text(paths["logs"], f"env recorded to {paths['env']}\n")
    print(str(run_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

