"""
One-click runner for the paper's experiments.

Profiles
- small (default): laptop-safe probes with streaming + small samples
- full: server-scale runs with non-streaming snapshot (pass --revision to pin)

What it runs
1) Prepare dataset (Project Gutenberg via HF or local files)
2) E1: algo table across vocab sizes {8k,16k,32k,64k}, multiple seeds
3) E0: ensemble selection vs single at 16k and 32k, K-scaling (K in {1,2,4,8} and optional 16)
4) Weighted-merge ablation for one K (char_entropy, theta=0.6)
5) Aggregate summaries and generate tables

Examples
  uv run python -m scripts.run_paper --profile small
  uv run python -m scripts.run_paper --profile full --revision <hf_rev_hash> --seeds 13 17 19 23 29
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import tomllib


def _run(cmd: list[str], env: dict | None = None) -> str:
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
    return out.decode("utf-8", errors="replace")


def _last_line(s: str) -> str:
    return [ln.strip() for ln in s.splitlines() if ln.strip()][-1]


def _write_toml_like(path: Path, cfg: Dict[str, Any]) -> None:
    lines: List[str] = []
    for sec in ["data", "oos_data", "train", "ensemble", "eval", "baselines", "artifacts"]:
        val = cfg.get(sec)
        if isinstance(val, dict) and val:
            lines.append(f"[{sec}]")
            for k, v in val.items():
                if isinstance(v, bool):
                    lines.append(f"{k} = {'true' if v else 'false'}")
                elif isinstance(v, (int, float)):
                    lines.append(f"{k} = {v}")
                elif isinstance(v, list):
                    def _fmt(x):
                        if isinstance(x, (int, float)):
                            return str(x)
                        return f'"{x}"'
                    lines.append(f"{k} = [{', '.join(_fmt(x) for x in v)}]")
                elif v is None:
                    continue
                else:
                    lines.append(f"{k} = \"{v}\"")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _base_cfg(out_dir: str, e1_name: str, dataset: str, streaming: bool, revision: str | None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "data": {
            "provider": "hf_books",
            "dataset": dataset,
            "split": "train",
            "text_field": "text",
            "out_dir": out_dir,
            # sizes filled by profile
            # seed filled by profile
            **({"streaming": True} if streaming else {}),
            **({"revision": revision} if (revision and not streaming) else {}),
            "oos_by_hash": True,
            "oos_frac": 0.1,
        },
        "train": {"exp_name": e1_name, "vocab_size": 32768},
        "eval": {"run_oos": True},
        "baselines": {"tiktoken": {"encodings": ["gpt2"]}},
        "artifacts": {"out_dir": "artifacts"},
    }
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["small", "full"], default="small")
    ap.add_argument("--dataset", default="common-pile/project_gutenberg")
    ap.add_argument("--revision", help="HF dataset revision to pin (full profile)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 17, 19])
    ap.add_argument("--sizes", nargs="+", type=int, default=[8192, 16384, 32768, 65536])
    ap.add_argument("--k", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--include-k16", action="store_true", help="also run K=16 where feasible")
    ap.add_argument("--out", default="artifacts/paper_run", help="top-level output dir for this run")
    args = ap.parse_args(argv)

    # Profile sizes
    if args.profile == "small":
        train_sz, valid_sz, test_sz = 2000, 200, 200
        streaming = True
        revision = None
        e1_name = "e1_pg_small"
        data_out = "data/processed/pg_paper_small"
    else:
        train_sz, valid_sz, test_sz = 8000, 800, 800
        streaming = False
        revision = args.revision
        e1_name = "e1_pg_full"
        data_out = "data/processed/pg_paper_full"

    # 1) Prepare dataset
    base = _base_cfg(data_out, e1_name, args.dataset, streaming=streaming, revision=revision)
    base["data"].update({"train_size": train_sz, "valid_size": valid_sz, "test_size": test_sz, "seed": 13})

    # write and run a prepare-only pipeline via run_experiment (it will call prepare script)
    # We could call prepare directly, but using run_experiment ensures uniform env capture
    with tempfile.TemporaryDirectory() as td:
        tmp_base = Path(td) / "base.toml"
        _write_toml_like(tmp_base, base)
        # run a cheap BPE train to materialize data + env; we'll reuse paths for later runs anyway
        _run([
            "uv", "run", "--with", "tokenizers", "--with", "psutil",
            "python", "-m", "scripts.run_experiment", "--config", str(tmp_base),
        ])

    # 2) E1 across sizes with seeds
    # We reuse experiments/gutenberg_e1_32k.toml layout but override sizes at runtime
    e1_cfg = {
        "data": base["data"],
        "train": {"exp_name": e1_name, "vocab_size": 32768},
        "eval": {"run_oos": True},
        "artifacts": {"out_dir": "artifacts"},
    }

    # Per-seed E1 runs
    e1_summaries: List[Path] = []
    for sd in args.seeds:
        with tempfile.TemporaryDirectory() as td:
            cfg = json.loads(json.dumps(e1_cfg))
            cfg["data"]["seed"] = int(sd)
            cfg["train"]["exp_name"] = f"{e1_name}_s{sd}"
            tmp = Path(td) / f"e1_s{sd}.toml"
            _write_toml_like(tmp, cfg)
            out = _run([
                "uv", "run", "--with", "tokenizers", "--with", "psutil",
                "python", "-m", "scripts.run_e1_sizes",
                "--config", str(tmp), "--sizes", *[str(s) for s in args.sizes],
            ])
            e1_summaries.append(Path(_last_line(out)))

    # 3) E0: selection vs single at 16k and 32k with K-scaling
    # Build base single and ensemble configs in-memory and override K
    def _cfg_single(exp_name: str, vocab: int) -> Dict[str, Any]:
        return {
            "data": base["data"],
            "train": {"exp_name": exp_name, "algo": "hf_bpe", "vocab_size": vocab},
            "eval": {"run_oos": True},
            "baselines": {"tiktoken": {"encodings": ["gpt2"]}},
            "artifacts": {"out_dir": "artifacts"},
        }

    def _cfg_ensemble(exp_name: str, vocab: int, k: int) -> Dict[str, Any]:
        return {
            "data": base["data"],
            "train": {"exp_name": exp_name, "algo": "hf_bpe", "vocab_size": vocab},
            "ensemble": {"enabled": True, "num_shards": int(k), "vocab_size": vocab, "exp_name": f"{exp_name}_ens"},
            "eval": {"run_oos": True},
            "baselines": {"tiktoken": {"encodings": ["gpt2"]}},
            "artifacts": {"out_dir": "artifacts"},
        }

    k_list = list(args.k)
    if args.include_k16 and 16 not in k_list:
        k_list.append(16)
    k_list = sorted(set(k_list))

    e0_out_root = Path(args.out) / "e0_k_scaling"
    e0_out_root.mkdir(parents=True, exist_ok=True)
    for vocab in [16384, 32768]:
        for k in k_list:
            # K=1 uses single config; K>1 uses ensemble config
            label = f"e0_pg_v{vocab}_k{k}"
            with tempfile.TemporaryDirectory() as td:
                if k == 1:
                    s_cfg = _cfg_single(label + "_single", vocab)
                    e_cfg = _cfg_ensemble(label + "_ens", vocab, 2)  # still run an ensemble to compare against single
                else:
                    s_cfg = _cfg_single(label + "_single", vocab)
                    e_cfg = _cfg_ensemble(label + "_ens", vocab, k)
                s_toml = Path(td) / "single.toml"
                e_toml = Path(td) / "ens.toml"
                _write_toml_like(s_toml, s_cfg)
                _write_toml_like(e_toml, e_cfg)
                out = _run([
                    "uv", "run", "--with", "tokenizers", "--with", "psutil",
                    "python", "-m", "scripts.run_e0", "--single", str(s_toml), "--ensemble", str(e_toml),
                    "--out", str(e0_out_root / f"v{vocab}_k{k}")
                ])
                _ = Path(_last_line(out))

    # 4) Weighted merge ablation (single example on 16k, K=4) if such ensemble exists
    # Find latest K=4 ensemble pipeline and run weighted merge
    # This is best-effort; failure here should not block the rest
    try:
        e4_glob = list(Path(args.out).rglob("*pg_v16384_k4*_pipeline/pipeline.json"))
        if e4_glob:
            pipe_json = e4_glob[-1]
            pipe = json.loads(pipe_json.read_text(encoding="utf-8"))
            ens_dir = pipe.get("ensemble_run_dir")
            if ens_dir:
                _run([
                    "uv", "run", "--with", "tokenizers",
                    "python", "-m", "scripts.merge_ensemble",
                    "--ensemble-run", ens_dir,
                    "--weight-by", "char_entropy", "--theta", "0.6",
                    "--exp", "pg16k_k4_weighted_merge", "--out", "artifacts",
                ])
    except Exception:
        pass

    # 5) Aggregate and make tables (scope to this run's out dir)
    agg_out = _run([
        "uv", "run", "python", "-m", "scripts.aggregate_metrics",
        "--inputs", str(Path(args.out)),
        "--out", str(Path(args.out) / "aggregated"),
    ])
    summary_csv = _last_line(agg_out)
    # Include one seeds stats file if exists (first found)
    seeds_stats = None
    for p in Path(args.out).rglob("seeds_stats.json"):
        seeds_stats = str(p)
        break
    _ = _run([
        "uv", "run", "python", "-m", "scripts.make_tables",
        "--summary-csv", summary_csv,
        *( ["--seeds-stats", seeds_stats] if seeds_stats else [] ),
        "--out", "results",
    ])

    print(str(Path("results/table.md").resolve()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
