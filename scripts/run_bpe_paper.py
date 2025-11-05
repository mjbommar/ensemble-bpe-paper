"""
BPE-only runner for paper experiments - avoids Unigram/WordPiece issues.

Simplified experiment runner that:
1. Trains BPE tokenizers across vocab sizes and seeds
2. Evaluates compression performance
3. Optionally runs ensemble experiments with K-scaling
4. Aggregates results with seed statistics

Examples:
  # Small test run
  uv run python -m scripts.run_bpe_paper --profile small --seeds 13 17

  # Full paper run
  uv run python -m scripts.run_bpe_paper --profile full \
    --seeds 13 17 19 --sizes 16384 32768 65536 \
    --k 1 2 4 8 --out artifacts/bpe_paper_run
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import tomllib


def _run(cmd: list[str], env: dict | None = None) -> str:
    """Run command and capture output, with error logging."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"ERROR: Command failed with exit code {e.returncode}", file=sys.stderr)
        print(f"Command: {' '.join(cmd)}", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        if e.output:
            print("Full output from failed command:", file=sys.stderr)
            print(e.output.decode("utf-8", errors="replace"), file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        
        # Write to error log
        error_log = Path("artifacts/bpe_paper_error.log")
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with error_log.open("a", encoding="utf-8") as f:
            import datetime
            f.write(f"\n{'='*80}\n")
            f.write(f"ERROR at {datetime.datetime.now().isoformat()}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Exit code: {e.returncode}\n")
            if e.output:
                f.write("Output:\n")
                f.write(e.output.decode("utf-8", errors="replace"))
            f.write(f"\n{'='*80}\n")
        raise


def _last_line(s: str) -> str:
    """Extract last non-empty line from output."""
    return [ln.strip() for ln in s.splitlines() if ln.strip()][-1]


def _write_toml_like(path: Path, cfg: Dict[str, Any]) -> None:
    """Write config as TOML-like format."""
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
                        return str(x) if isinstance(x, (int, float)) else f'"{x}"'
                    lines.append(f"{k} = [{', '.join(_fmt(x) for x in v)}]")
                elif v is None:
                    continue
                else:
                    lines.append(f"{k} = \"{v}\"")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run BPE-only paper experiments")
    ap.add_argument("--profile", choices=["small", "full"], default="small")
    ap.add_argument("--dataset", default="common-pile/project_gutenberg")
    ap.add_argument("--revision", help="HF dataset revision to pin (full profile)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 17, 19])
    ap.add_argument("--sizes", nargs="+", type=int, default=[16384, 32768, 65536])
    ap.add_argument("--k", nargs="+", type=int, default=[2, 4, 8, 16], help="Ensemble K values (num shards)")
    ap.add_argument("--run-ensembles", action="store_true", help="Run ensemble experiments")
    ap.add_argument("--out", default="artifacts/bpe_paper_run", help="Output directory")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Profile configuration
    if args.profile == "small":
        train_sz, valid_sz, test_sz = 2000, 200, 200
        streaming = True
        revision = None
        exp_name = "bpe_pg_small"
        data_out = "data/processed/pg_bpe_small"
    else:
        train_sz, valid_sz, test_sz = 8000, 800, 800
        streaming = False
        revision = args.revision
        exp_name = "bpe_pg_full"
        data_out = "data/processed/pg_bpe_full"

    # Base configuration
    base_cfg: Dict[str, Any] = {
        "data": {
            "provider": "hf_books",
            "dataset": args.dataset,
            "split": "train",
            "text_field": "text",
            "out_dir": data_out,
            "train_size": train_sz,
            "valid_size": valid_sz,
            "test_size": test_sz,
            "seed": 13,
            **({"streaming": True} if streaming else {}),
            **({"revision": revision} if (revision and not streaming) else {}),
            "oos_by_hash": True,
            "oos_frac": 0.1,
        },
        "train": {"exp_name": exp_name, "algo": "hf_bpe", "vocab_size": 32768},
        "eval": {"run_oos": True},
        "baselines": {"tiktoken": {"encodings": ["gpt2"]}},
        "artifacts": {"out_dir": "artifacts"},
    }

    # 1) Prepare dataset once
    print(f"[1/4] Preparing dataset: {args.dataset}")
    with tempfile.TemporaryDirectory() as td:
        tmp_base = Path(td) / "base.toml"
        _write_toml_like(tmp_base, base_cfg)
        _run([
            "uv", "run", "--with", "tokenizers", "--with", "psutil",
            "python", "-m", "scripts.run_experiment", "--config", str(tmp_base),
        ])

    # 2) Train BPE across sizes and seeds
    print(f"[2/4] Training BPE tokenizers: {len(args.seeds)} seeds × {len(args.sizes)} vocab sizes")
    import csv as csv_module
    
    bpe_results = []
    results_by_config = defaultdict(list)  # (vocab, seed) -> metrics
    
    for seed in args.seeds:
        for vocab_size in args.sizes:
            print(f"  Training: seed={seed}, vocab={vocab_size}")
            
            cfg = json.loads(json.dumps(base_cfg))
            cfg["data"]["seed"] = seed
            cfg["train"]["exp_name"] = f"{exp_name}_s{seed}_v{vocab_size}"
            cfg["train"]["vocab_size"] = vocab_size
            
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "config.toml"
                _write_toml_like(tmp, cfg)
                out = _run([
                    "uv", "run", "--with", "tokenizers", "--with", "psutil",
                    "python", "-m", "scripts.run_experiment", "--config", str(tmp),
                ])
                
                # Parse results
                pipeline_dir = Path(_last_line(out))
                manifest = json.loads((pipeline_dir / "pipeline.json").read_text(encoding="utf-8"))
                
                train_run = Path(manifest["train_run_dir"])
                eval_run = Path(manifest["eval_run_dir"])
                oos_eval_run = Path(manifest.get("oos_eval_run_dir", ""))
                
                train_metrics = json.loads((train_run / "metrics.json").read_text(encoding="utf-8"))
                eval_metrics = json.loads((eval_run / "metrics.json").read_text(encoding="utf-8"))
                
                result = {
                    "seed": seed,
                    "vocab_size": vocab_size,
                    "algo": "hf_bpe",
                    "train_run": str(train_run),
                    "eval_run": str(eval_run),
                    "pipeline_dir": str(pipeline_dir),
                    "train_wall_time_s": train_metrics.get("train_wall_time_s"),
                    "train_cpu_time_s": train_metrics.get("train_cpu_time_s"),
                    "peak_rss_kb": train_metrics.get("peak_rss_kb"),
                    "tokens_per_byte": eval_metrics.get("tokens_per_byte"),
                }
                bpe_results.append(result)
                results_by_config[(vocab_size, seed)].append(result)

    # Write BPE results
    bpe_csv = out_dir / "bpe_training_results.csv"
    if bpe_results:
        fieldnames = list(bpe_results[0].keys())
        with bpe_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv_module.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(bpe_results)
        
        (out_dir / "bpe_training_results.json").write_text(
            json.dumps(bpe_results, indent=2) + "\n", encoding="utf-8"
        )

    # 3) Compute seed statistics
    print(f"[3/4] Computing statistics across seeds")
    seed_stats = []
    for vocab_size in args.sizes:
        values = [r["tokens_per_byte"] for r in bpe_results if r["vocab_size"] == vocab_size]
        if len(values) > 1:
            seed_stats.append({
                "vocab_size": vocab_size,
                "mean_tokens_per_byte": statistics.mean(values),
                "std_tokens_per_byte": statistics.stdev(values),
                "min_tokens_per_byte": min(values),
                "max_tokens_per_byte": max(values),
                "num_seeds": len(values),
            })
    
    if seed_stats:
        (out_dir / "seed_statistics.json").write_text(
            json.dumps(seed_stats, indent=2) + "\n", encoding="utf-8"
        )

    # 4) Ensemble experiments (optional)
    if args.run_ensembles:
        total_ens = len(args.seeds) * len(args.sizes) * len(args.k)
        print(f"[4/4] Running ensemble experiments: {total_ens} total ({len(args.seeds)} seeds × {len(args.sizes)} vocab sizes × {len(args.k)} K values)")

        ens_count = 0
        for seed in args.seeds:
            for vocab_size in args.sizes:
                for k in args.k:
                    ens_count += 1
                    print(f"  [{ens_count}/{total_ens}] Ensemble: seed={seed}, vocab={vocab_size}, K={k}")

                    # Update base config with current seed
                    ens_cfg = json.loads(json.dumps(base_cfg))
                    ens_cfg["data"]["seed"] = seed

                    # Single tokenizer config
                    single_cfg = json.loads(json.dumps(ens_cfg))
                    single_cfg["train"]["exp_name"] = f"{exp_name}_ens_s{seed}_v{vocab_size}_k{k}_single"
                    single_cfg["train"]["vocab_size"] = vocab_size

                    # Ensemble config
                    ensemble_cfg = json.loads(json.dumps(ens_cfg))
                    ensemble_cfg["train"]["exp_name"] = f"{exp_name}_ens_s{seed}_v{vocab_size}_k{k}_base"
                    ensemble_cfg["train"]["vocab_size"] = vocab_size
                    ensemble_cfg["ensemble"] = {
                        "enabled": True,
                        "num_shards": k,
                        "vocab_size": vocab_size,
                        "exp_name": f"{exp_name}_ens_s{seed}_v{vocab_size}_k{k}_ens"
                    }

                    with tempfile.TemporaryDirectory() as td:
                        s_toml = Path(td) / "single.toml"
                        e_toml = Path(td) / "ens.toml"
                        _write_toml_like(s_toml, single_cfg)
                        _write_toml_like(e_toml, ensemble_cfg)

                        _run([
                            "uv", "run", "--with", "tokenizers", "--with", "psutil",
                            "python", "-m", "scripts.run_e0",
                            "--single", str(s_toml),
                            "--ensemble", str(e_toml),
                            "--out", str(out_dir / "ensembles" / f"s{seed}_v{vocab_size}_k{k}")
                        ])
    else:
        print("[4/4] Skipping ensemble experiments (use --run-ensembles to enable)")

    # Final summary
    print(f"\n{'='*80}")
    print(f"BPE Paper Run Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"BPE results: {len(bpe_results)} experiments")
    print(f"  - Seeds: {args.seeds}")
    print(f"  - Vocab sizes: {args.sizes}")
    if seed_stats:
        print(f"Seed statistics: {len(seed_stats)} vocab sizes")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
