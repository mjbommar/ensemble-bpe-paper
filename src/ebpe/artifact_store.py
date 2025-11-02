from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Dict


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(exp_name: str, base_dir: str | Path = "artifacts") -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    run = base / f"{timestamp()}_{exp_name}"
    run.mkdir(parents=True, exist_ok=False)
    return run


def write_text(path: str | Path, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_yaml_like(path: str | Path, obj: Dict[str, Any]) -> None:
    """
    Minimal dependency-free YAML writer suitable for simple dicts. Falls back to JSON-like.
    This is intentionally simple to avoid adding dependencies; replace with PyYAML later.
    """
    lines = []
    for k, v in obj.items():
        if isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        elif isinstance(v, (list, tuple)):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            s = str(v).replace("\n", " ")
            lines.append(f"{k}: {s}")
    write_text(path, "\n".join(lines) + "\n")


def default_artifact_paths(run_dir: str | Path) -> Dict[str, Path]:
    run = Path(run_dir)
    return {
        "config": run / "config.yaml",
        "metrics": run / "metrics.json",
        "tokenizer": run / "tokenizer.json",
        "logs": run / "logs.txt",
        "env": run / "env.txt",
    }

