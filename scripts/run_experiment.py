"""
End-to-end experiment runner driven by a TOML config.

Stages:
  1) prepare_data (dictionary-based)
  2) train_tokenizer (HF BPE)
  3) eval_compression (held-out split)
  Optional: ensemble flow when [ensemble] is enabled in the TOML.

Example:
  uv run --with tokenizers --with psutil python -m scripts.run_experiment \
      --config experiments/hf_bpe.toml
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict
import json
import csv

import tomllib

# Ensure `src` is on sys.path for local runs (no installation step required)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ebpe.artifact_store import create_run_dir, default_artifact_paths, write_json, write_text, write_yaml_like


def _last_line(s: str) -> str:
    return [ln.strip() for ln in s.splitlines() if ln.strip()][-1]


def _run(cmd: list[str], env: dict | None = None) -> str:
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
    return out.decode("utf-8", errors="replace")


def _with_src_env(extra: dict | None = None) -> dict:
    env = os.environ.copy()
    py = env.get("PYTHONPATH", "")
    prefix = str(SRC_PATH)
    if py:
        env["PYTHONPATH"] = f"{prefix}:{py}"
    else:
        env["PYTHONPATH"] = prefix
    if extra:
        env.update(extra)
    return env


def run_pipeline(cfg_path: Path) -> Path:
    cfg_raw = cfg_path.read_bytes()
    cfg = tomllib.loads(cfg_raw.decode("utf-8"))

    data = cfg.get("data", {})
    train = cfg.get("train", {})
    evalc = cfg.get("eval", {})
    art = cfg.get("artifacts", {})
    ens = cfg.get("ensemble", {})
    baselines = cfg.get("baselines", {})

    artifacts_base = Path(art.get("out_dir", "artifacts"))
    exp_name = str(train.get("exp_name", "hf_bpe"))
    # Create a pipeline run dir
    pipe_dir = create_run_dir(f"{exp_name}_pipeline", base_dir=artifacts_base)
    pipe_paths = default_artifact_paths(pipe_dir)

    # 1) data preparation (provider switch)
    provider = str(data.get("provider", "dict"))
    out_dir = str(data.get("out_dir", "data/processed"))
    if provider == "hf_books":
        # Hugging Face books/text provider (or local offline via --local-files)
        prep_cmd = [
            "uv",
            "run",
            "--with",
            "datasets",
            "python",
            "-m",
            "scripts.prepare_hf_books",
            "--out",
            out_dir,
            "--sample-train",
            str(data.get("train_size", 50000)),
            "--sample-valid",
            str(data.get("valid_size", 5000)),
            "--sample-test",
            str(data.get("test_size", 5000)),
            "--seed",
            str(data.get("seed", 13)),
        ]
        # Either dataset-based or local offline source
        if data.get("local_files"):
            prep_cmd.extend(["--local-files", str(data.get("local_files"))])
            # Local mode enhancements: split strategy and OOS controls
            if data.get("split_strategy"):
                prep_cmd.extend(["--split-strategy", str(data.get("split_strategy"))])
            if data.get("oos_local_files"):
                prep_cmd.extend(["--oos-local-files", str(data.get("oos_local_files"))])
            if data.get("oos_frac"):
                prep_cmd.extend(["--oos-frac", str(data.get("oos_frac"))])
            if data.get("oos_count"):
                prep_cmd.extend(["--oos-count", str(data.get("oos_count"))])
        else:
            if data.get("dataset"):
                prep_cmd.extend(["--dataset", str(data.get("dataset"))])
            if data.get("config"):
                prep_cmd.extend(["--config", str(data.get("config"))])
            if data.get("split"):
                prep_cmd.extend(["--split", str(data.get("split"))])
            if data.get("revision"):
                prep_cmd.extend(["--revision", str(data.get("revision"))])
            if data.get("text_field"):
                prep_cmd.extend(["--text-field", str(data.get("text_field"))])
            if data.get("streaming"):
                prep_cmd.append("--streaming")
        # Pass OOS-by-hash when requested at data level
        if data.get("oos_by_hash"):
            prep_cmd.append("--oos-by-hash")
        if data.get("oos_frac"):
            prep_cmd.extend(["--oos-frac", str(data.get("oos_frac"))])
        if data.get("oos_count"):
            prep_cmd.extend(["--oos-count", str(data.get("oos_count"))])
        _run(prep_cmd, env=_with_src_env())
    else:
        # Dictionary-based provider
        dict_path = str(data.get("dict_path", "/usr/share/dict"))
        prep_cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.prepare_data",
            "--dict-path",
            dict_path,
            "--out",
            out_dir,
            "--train-size",
            str(data.get("train_size", 50000)),
            "--valid-size",
            str(data.get("valid_size", 5000)),
            "--test-size",
            str(data.get("test_size", 5000)),
            "--seed",
            str(data.get("seed", 13)),
        ]
        if data.get("lower", False):
            prep_cmd.append("--lower")
        if data.get("oos_dict_path"):
            prep_cmd.extend(["--oos-dict-path", str(data["oos_dict_path"])])
            if data.get("oos_size"):
                prep_cmd.extend(["--oos-size", str(data["oos_size"])])
        _run(prep_cmd, env=_with_src_env())

    train_file = Path(out_dir) / "train.txt"
    valid_file = Path(out_dir) / "valid.txt"
    test_file = Path(out_dir) / "test.txt"

    # 2) train tokenizer (algo selectable)
    algo = str(train.get("algo", "hf_bpe"))
    if algo == "hf_wordpiece":
        module = "scripts.train_wordpiece"
    elif algo == "hf_unigram":
        module = "scripts.train_unigram"
    else:
        module = "scripts.train_tokenizer"

    train_cmd = [
        "uv",
        "run",
        "--with",
        "tokenizers",
        "--with",
        "psutil",
        "python",
        "-m",
        module,
        "--exp",
        exp_name,
        "--train",
        str(train_file),
        "--vocab-size",
        str(train.get("vocab_size", 2000)),
        "--out",
        str(artifacts_base),
    ]
    if algo == "hf_wordpiece" and train.get("lowercase", False):
        train_cmd.append("--lowercase")
    if valid_file.exists():
        train_cmd.extend(["--valid", str(valid_file)])
    if test_file.exists():
        train_cmd.extend(["--test", str(test_file)])
    train_out = _run(train_cmd, env=_with_src_env())
    train_run = Path(_last_line(train_out))

    # 3) eval_compression
    eval_exp = f"{exp_name}_eval"
    eval_file = str(test_file if evalc.get("eval_split", "test") == "test" else valid_file)
    eval_cmd = [
        "uv",
        "run",
        "--with",
        "tokenizers",
        "python",
        "-m",
        "scripts.eval_compression",
        "--exp",
        eval_exp,
        "--tokenizer",
        str(train_run / "tokenizer.json"),
        "--eval-file",
        eval_file,
        "--out",
        str(artifacts_base),
    ]
    eval_out = _run(eval_cmd, env=_with_src_env())
    eval_run = Path(_last_line(eval_out))

    # Optional: out-of-sample evaluation
    oos_eval_run: Path | None = None
    run_oos = bool(evalc.get("run_oos", False))
    oos_path = None
    if run_oos:
        # Option A: user-provided path
        if evalc.get("oos_eval_file"):
            oos_path = str(evalc.get("oos_eval_file"))
        else:
            # Option B: run an OOS data preparation step if [oos_data] provided
            oos_data = cfg.get("oos_data", {})
            if isinstance(oos_data, dict) and oos_data:
                oos_provider = str(oos_data.get("provider", provider))
                oos_out_dir = str(oos_data.get("out_dir", str(Path(out_dir).with_name(Path(out_dir).name + "_oos"))))
                if oos_provider == "hf_books":
                    prep_cmd = [
                        "uv",
                        "run",
                        "--with",
                        "datasets",
                        "python",
                        "-m",
                        "scripts.prepare_hf_books",
                        "--out",
                        oos_out_dir,
                        "--sample-train",
                        str(oos_data.get("train_size", 0)),
                        "--sample-valid",
                        str(oos_data.get("valid_size", 0)),
                        "--sample-test",
                        str(oos_data.get("test_size", 5000)),
                        "--seed",
                        str(oos_data.get("seed", data.get("seed", 13))),
                    ]
                    if oos_data.get("local_files"):
                        prep_cmd.extend(["--local-files", str(oos_data.get("local_files"))])
                    else:
                        if oos_data.get("dataset"):
                            prep_cmd.extend(["--dataset", str(oos_data.get("dataset"))])
                        if oos_data.get("config"):
                            prep_cmd.extend(["--config", str(oos_data.get("config"))])
                        if oos_data.get("split"):
                            prep_cmd.extend(["--split", str(oos_data.get("split"))])
                        if oos_data.get("revision"):
                            prep_cmd.extend(["--revision", str(oos_data.get("revision"))])
                        if oos_data.get("text_field"):
                            prep_cmd.extend(["--text-field", str(oos_data.get("text_field"))])
                    _run(prep_cmd)
                    oos_path = str(Path(oos_out_dir) / "test.txt")
                else:
                    # dictionary provider for OOS
                    dict_path_oos = str(oos_data.get("dict_path", "/usr/share/dict"))
                    prep_cmd = [
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "scripts.prepare_data",
                        "--dict-path",
                        dict_path_oos,
                        "--out",
                        oos_out_dir,
                        "--train-size",
                        str(oos_data.get("train_size", 0)),
                        "--valid-size",
                        str(oos_data.get("valid_size", 0)),
                        "--test-size",
                        str(oos_data.get("test_size", 5000)),
                        "--seed",
                        str(oos_data.get("seed", data.get("seed", 13))),
                    ]
                    if oos_data.get("lower", False):
                        prep_cmd.append("--lower")
                    _run(prep_cmd)
                    oos_path = str(Path(oos_out_dir) / "test.txt")
            # Option C: auto-detect OOS file produced during data prep
            if not oos_path:
                maybe = Path(out_dir) / "test_oos.txt"
                if maybe.exists():
                    oos_path = str(maybe)
        if oos_path:
            oos_exp = f"{exp_name}_eval_oos"
            oos_cmd = [
                "uv",
                "run",
                "--with",
                "tokenizers",
                "python",
                "-m",
                "scripts.eval_compression",
                "--exp",
                oos_exp,
                "--tokenizer",
                str(train_run / "tokenizer.json"),
                "--eval-file",
                oos_path,
                "--out",
                str(artifacts_base),
            ]
            oos_out = _run(oos_cmd, env=_with_src_env())
            oos_eval_run = Path(_last_line(oos_out))

    # Gather summary rows
    def _read_tokens_per_byte(run_dir: Path) -> float:
        try:
            m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            return float(m.get("tokens_per_byte", 0.0))
        except Exception:
            return 0.0

    # Read training metrics for reporting
    train_vocab_size: int | None = None
    train_wall_time_s: float | None = None
    train_cpu_time_s: float | None = None
    train_peak_rss_kb: float | None = None
    try:
        _train_metrics = json.loads((train_run / "metrics.json").read_text(encoding="utf-8"))
        tv = _train_metrics.get("vocab_size")
        if isinstance(tv, int):
            train_vocab_size = tv
        if isinstance(_train_metrics.get("train_wall_time_s"), (int, float)):
            train_wall_time_s = float(_train_metrics["train_wall_time_s"])  # type: ignore[index]
        if isinstance(_train_metrics.get("train_cpu_time_s"), (int, float)):
            train_cpu_time_s = float(_train_metrics["train_cpu_time_s"])  # type: ignore[index]
        if isinstance(_train_metrics.get("peak_rss_kb"), (int, float)):
            train_peak_rss_kb = float(_train_metrics["peak_rss_kb"])  # type: ignore[index]
    except Exception:
        pass

    summary_rows: list[dict[str, str | float | int | None]] = []
    summary_rows.append(
        {
            "name": exp_name,
            "kind": algo,
            "run_dir": str(eval_run.resolve()),
            "tokens_per_byte": _read_tokens_per_byte(eval_run),
            "vocab_size": train_vocab_size,
            "train_wall_time_s": train_wall_time_s,
            "train_cpu_time_s": train_cpu_time_s,
            "peak_rss_kb": train_peak_rss_kb,
        }
    )
    if oos_eval_run is not None:
        summary_rows.append(
            {
                "name": f"{exp_name}_oos",
                "kind": f"{algo}_oos",
                "run_dir": str(oos_eval_run.resolve()),
                "tokens_per_byte": _read_tokens_per_byte(oos_eval_run),
                "vocab_size": train_vocab_size,
                "train_wall_time_s": train_wall_time_s,
                "train_cpu_time_s": train_cpu_time_s,
                "peak_rss_kb": train_peak_rss_kb,
            }
        )

    # Optional: Ensemble flow
    ensemble_enabled = bool(ens.get("enabled", False))
    ensemble_dir: Path | None = None
    ensemble_eval_run: Path | None = None
    if ensemble_enabled:
        num_shards = int(ens.get("num_shards", 2))
        ens_vocab_size = int(ens.get("vocab_size", train.get("vocab_size", 2000)))
        ens_exp = f"{exp_name}_ens"
        ens_cmd = [
            "uv",
            "run",
            "--with",
            "tokenizers",
            "--with",
            "psutil",
            "python",
            "-m",
            "scripts.train_ensemble",
            "--exp",
            ens_exp,
            "--train",
            str(train_file),
            "--eval-file",
            eval_file,
            "--num-shards",
            str(num_shards),
            "--vocab-size",
            str(ens_vocab_size),
            "--out",
            str(artifacts_base),
        ]
        ens_out = _run(ens_cmd, env=_with_src_env())
        ensemble_dir = Path(_last_line(ens_out))

        # Evaluate the selected ensemble member on eval_file for a clean top-line
        sel_tok = ensemble_dir / "selected_tokenizer.json"
        if sel_tok.exists():
            ens_eval_exp = f"{ens_exp}_selected_eval"
            ens_eval_cmd = [
                "uv",
                "run",
                "--with",
                "tokenizers",
                "python",
                "-m",
                "scripts.eval_compression",
                "--exp",
                ens_eval_exp,
                "--tokenizer",
                str(sel_tok),
                "--eval-file",
                eval_file,
                "--out",
                str(artifacts_base),
            ]
            ens_eval_out = _run(ens_eval_cmd, env=_with_src_env())
            ensemble_eval_run = Path(_last_line(ens_eval_out))
            # Attempt to read the selected member's training metrics
            ens_train_wall: float | None = None
            ens_train_cpu: float | None = None
            ens_peak_kb: float | None = None
            try:
                ens_meta = json.loads((ensemble_dir / "ensemble.json").read_text(encoding="utf-8"))
                best = ens_meta.get("best", {})
                best_train = Path(best.get("train_run", ""))
                if best_train.exists():
                    _m = json.loads((best_train / "metrics.json").read_text(encoding="utf-8"))
                    if isinstance(_m.get("train_wall_time_s"), (int, float)):
                        ens_train_wall = float(_m["train_wall_time_s"])  # type: ignore[index]
                    if isinstance(_m.get("train_cpu_time_s"), (int, float)):
                        ens_train_cpu = float(_m["train_cpu_time_s"])  # type: ignore[index]
                    if isinstance(_m.get("peak_rss_kb"), (int, float)):
                        ens_peak_kb = float(_m["peak_rss_kb"])  # type: ignore[index]
            except Exception:
                pass

            summary_rows.append(
                {
                    "name": ens_exp,
                    "kind": "ensemble_selected",
                    "run_dir": str(ensemble_eval_run.resolve()),
                    "tokens_per_byte": _read_tokens_per_byte(ensemble_eval_run),
                    "vocab_size": ens_vocab_size,
                    "train_wall_time_s": ens_train_wall,
                    "train_cpu_time_s": ens_train_cpu,
                    "peak_rss_kb": ens_peak_kb,
                }
            )

        # Emit ensemble summary CSV from ensemble.json
        try:
            ens_json = json.loads((ensemble_dir / "ensemble.json").read_text(encoding="utf-8"))
            summary_csv = ensemble_dir / "ensemble_summary.csv"
            with summary_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["kind", "index", "train_run", "eval_run", "tokens_per_byte"])
                for idx, m in enumerate(ens_json.get("members", [])):
                    w.writerow([
                        "member",
                        idx,
                        m.get("train_run", ""),
                        m.get("eval_run", ""),
                        m.get("tokens_per_byte", ""),
                    ])
                if ens_json.get("best"):
                    b = ens_json["best"]
                    w.writerow(["selected", "-", b.get("train_run", ""), b.get("eval_run", ""), b.get("tokens_per_byte", "")])
        except Exception as e:  # defensive; do not fail pipeline on CSV writing
            write_text(pipe_paths["logs"], f"warning: failed to write ensemble summary: {e}\n")

        # Optional merge-voted tokenizer
        merge_cfg = ens.get("merge", {}) if isinstance(ens, dict) else {}
        if merge_cfg and merge_cfg.get("enabled", False):
            k = int(merge_cfg.get("k", max(1, (num_shards + 1) // 2)))
            merge_exp = f"{ens_exp}_merge"
            merge_cmd = [
                "uv",
                "run",
                "--with",
                "tokenizers",
                "python",
                "-m",
                "scripts.merge_ensemble",
                "--exp",
                merge_exp,
                "--ensemble-run",
                str(ensemble_dir),
                "--k",
                str(k),
                "--out",
                str(artifacts_base),
            ]
            merge_out = _run(merge_cmd, env=_with_src_env())
            merge_dir = Path(_last_line(merge_out))

            # Evaluate merged tokenizer
            m_eval_exp = f"{merge_exp}_eval"
            m_eval_cmd = [
                "uv",
                "run",
                "--with",
                "tokenizers",
                "python",
                "-m",
                "scripts.eval_compression",
                "--exp",
                m_eval_exp,
                "--tokenizer",
                str(merge_dir / "tokenizer.json"),
                "--eval-file",
                eval_file,
                "--out",
                str(artifacts_base),
            ]
            m_eval_out = _run(m_eval_cmd, env=_with_src_env())
            m_eval_run = Path(_last_line(m_eval_out))
            summary_rows.append(
                {
                    "name": f"{ens_exp}_merge",
                    "kind": "ensemble_merge",
                    "run_dir": str(m_eval_run.resolve()),
                    "tokens_per_byte": _read_tokens_per_byte(m_eval_run),
                    "vocab_size": ens_vocab_size,
                    "train_wall_time_s": None,
                    "train_cpu_time_s": None,
                    "peak_rss_kb": None,
                }
            )

    # Optional: tiktoken baselines
    tke = baselines.get("tiktoken", {}) if isinstance(baselines, dict) else {}
    encs = tke.get("encodings", []) if isinstance(tke, dict) else []
    for enc in encs:
        enc = str(enc)
        bt_exp = f"{exp_name}_tiktoken_{enc}"
        bt_cmd = [
            "uv",
            "run",
            "--with",
            "tiktoken",
            "python",
            "-m",
            "scripts.eval_compression_tiktoken",
            "--exp",
            bt_exp,
            "--encoding",
            enc,
            "--eval-file",
            eval_file,
            "--out",
            str(artifacts_base),
        ]
        bt_out = _run(bt_cmd, env=_with_src_env())
        bt_run = Path(_last_line(bt_out))
        summary_rows.append(
            {
                "name": bt_exp,
                "kind": "baseline_tiktoken",
                "run_dir": str(bt_run.resolve()),
                "tokens_per_byte": _read_tokens_per_byte(bt_run),
                "vocab_size": None,
            }
        )

    # Write pipeline manifest
    # Data provenance info
    data_info: dict[str, str | None] = {"provider": provider, "out_dir": str(Path(out_dir).resolve())}
    if provider == "hf_books":
        data_info.update(
            {
                "dataset": str(data.get("dataset")) if data.get("dataset") else None,
                "config": str(data.get("config")) if data.get("config") else None,
                "split": str(data.get("split", "train")),
                "revision": str(data.get("revision")) if data.get("revision") else None,
                "local_files": str(data.get("local_files")) if data.get("local_files") else None,
            }
        )
    else:
        data_info.update({"dict_path": str(data.get("dict_path", "/usr/share/dict"))})

    manifest = {
        "config_path": str(cfg_path.resolve()),
        "data": data_info,
        "train_run_dir": str(train_run.resolve()),
        "eval_run_dir": str(eval_run.resolve()),
        "ensemble_run_dir": str(ensemble_dir.resolve()) if ensemble_dir else None,
        "ensemble_eval_run_dir": str(ensemble_eval_run.resolve()) if ensemble_eval_run else None,
        "oos_eval_run_dir": str(oos_eval_run.resolve()) if oos_eval_run else None,
    }
    write_yaml_like(pipe_paths["config"], {"pipeline_config": str(cfg_path.resolve())})
    write_json(pipe_paths["metrics"], {"status": "completed"})
    write_text(pipe_paths["env"], "pipeline: prepare_data -> train_tokenizer -> eval_compression\n")
    # Write a simple summary CSV for quick comparisons
    try:
        import csv as _csv

        with (pipe_dir / "pipeline_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "kind",
                    "run_dir",
                    "tokens_per_byte",
                    "vocab_size",
                    "train_wall_time_s",
                    "train_cpu_time_s",
                    "peak_rss_kb",
                ],
            )
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
        (pipe_dir / "pipeline_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    except Exception as e:
        write_text(pipe_paths["logs"], f"warning: failed to write pipeline summary: {e}\n")

    write_text(pipe_paths["logs"], f"train={train_run}\neval={eval_run}\n")
    (pipe_dir / "pipeline.json").write_text(
        __import__("json").dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(str(pipe_dir))
    return pipe_dir


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to TOML experiment config")
    args = ap.parse_args(argv)
    run_pipeline(Path(args.config))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
