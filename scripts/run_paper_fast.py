"""
================================================================================
FAST ENSEMBLE TOKENIZATION EXPERIMENT RUNNER
================================================================================

⚠️  CRITICAL: This is a REDUCED parameter sweep for rapid testing!

DIFFERENCES FROM run_paper_complete.py:
- Vocabulary sizes: 16k, 32k only (drop 64k)
- K values: 4, 16 only (drop 2, 8)
- Merge voting: Only majority (k=ceil(n/2)), skip 25% and 75%
- Weighted voting: Only entropy weight, only θ={0.3, 0.7}, skip char_entropy and θ=0.5

SPEEDUP: ~6.6x faster (60 experiments vs 396)

ENSEMBLE METHODS IMPLEMENTED:
1. ✅ TOP-1 SELECTION: Pick best member from K candidates
2. ✅ K-OF-N MERGE VOTING: Majority voting only (k=ceil(n/2))
3. ✅ WEIGHTED MERGE VOTING: Entropy weight, θ ∈ {0.3, 0.7}

DO NOT MODIFY THIS SCRIPT TO REMOVE ANY ENSEMBLE METHODS!

================================================================================

Parameter Sweep:

| Parameter | Values | Count |
|-----------|--------|-------|
| Seeds | 13, 17, 19 | 3 |
| Vocab Sizes | 16384, 32768 | 2 |
| K Values | 4, 16 | 2 |
| Base Configs | 3 × 2 × 2 | 12 |

Methods per config:
- Baseline (1)
- Selection (1)
- Merge majority (1)
- Weighted entropy θ=0.3 (1)
- Weighted entropy θ=0.7 (1)
= 5 methods × 2 evals (TEST + OOS) = 10 evaluations

Total: 12 configs × 5 methods = 60 experiments (vs 396 in full version)

Estimated runtime:
- Small profile: 4-8 hours (vs 12-24 hours)
- Full profile: 8-18 hours (vs 24-54 hours)

================================================================================

What this script does:

1. **Baseline Training**
   - Train single BPE on full training data
   - Evaluate on TEST and OOS

2. **Ensemble Member Training**
   - Shard training data into K pieces
   - Train K independent BPE tokenizers
   - Compute shard statistics for weighting

3. **Ensemble Method 1: Selection**
   - Pick the single best member (min tokens_per_byte on TEST)
   - Evaluate on TEST and OOS

4. **Ensemble Method 2: Merge Voting (Majority)**
   - Build tokenizer with merges appearing in ≥ceil(n/2) members
   - Evaluate on TEST and OOS

5. **Ensemble Method 3: Weighted Voting (Entropy)**
   - Weight members by byte-level entropy
   - Test θ ∈ {0.3, 0.7} for coverage
   - Evaluate each on TEST and OOS

6. **Comparison & Analysis**
   - Generate method comparison tables
   - Rank methods by OOS performance

================================================================================

Usage:

  # Quick test (fastest)
  uv run python -m scripts.run_paper_fast \\
      --profile small \\
      --data-dir data/processed/pg_paper_full

  # Full fast sweep
  uv run python -m scripts.run_paper_fast \\
      --profile full \\
      --seeds 13 17 19 \\
      --data-dir data/processed/pg_paper_full

  # Single seed test
  uv run python -m scripts.run_paper_fast \\
      --profile small \\
      --seeds 13 \\
      --data-dir data/processed/pg_paper_full

================================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Reuse helper functions from run_paper_complete.py
# (In production, these would be in a shared module)


def _run(cmd: list[str], env: dict | None = None) -> str:
    """Run command and return output, with error logging."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
        return out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"ERROR: Command failed with exit code {e.returncode}", file=sys.stderr)
        print(f"Command: {' '.join(cmd)}", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        if e.output:
            print("Output:", file=sys.stderr)
            print(e.output.decode("utf-8", errors="replace"), file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        raise


def _last_line(s: str) -> str:
    """Extract last non-empty line from command output."""
    return [ln.strip() for ln in s.splitlines() if ln.strip()][-1]


def _read_tokens_per_byte(run_dir: Path) -> float:
    """Read tokens_per_byte from metrics.json."""
    try:
        m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        return float(m.get("tokens_per_byte", 0.0))
    except Exception:
        return 0.0


def run_baseline(
    train_file: Path,
    test_file: Path,
    oos_file: Path,
    vocab: int,
    seed: int,
    exp_name: str,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """Run single baseline BPE training and evaluation."""
    print(f"\n{'='*80}")
    print(f"BASELINE: seed={seed}, vocab={vocab}")
    print(f"{'='*80}")

    # Train
    train_cmd = [
        "uv", "run", "--with", "tokenizers", "--with", "psutil",
        "python", "-m", "scripts.train_tokenizer",
        "--exp", exp_name,
        "--train", str(train_file),
        "--vocab-size", str(vocab),
        "--out", str(artifacts_dir),
    ]
    train_out = _run(train_cmd)
    train_run = Path(_last_line(train_out))
    print(f"  ✓ Training complete: {train_run.name}")

    # Eval on TEST
    eval_cmd = [
        "uv", "run", "--with", "tokenizers",
        "python", "-m", "scripts.eval_compression",
        "--exp", f"{exp_name}_eval",
        "--tokenizer", str(train_run / "tokenizer.json"),
        "--eval-file", str(test_file),
        "--out", str(artifacts_dir),
    ]
    eval_out = _run(eval_cmd)
    eval_run = Path(_last_line(eval_out))
    test_tpb = _read_tokens_per_byte(eval_run)
    print(f"  ✓ TEST eval: {test_tpb:.6f} tokens/byte")

    # Eval on OOS
    oos_eval_cmd = [
        "uv", "run", "--with", "tokenizers",
        "python", "-m", "scripts.eval_compression",
        "--exp", f"{exp_name}_eval_oos",
        "--tokenizer", str(train_run / "tokenizer.json"),
        "--eval-file", str(oos_file),
        "--out", str(artifacts_dir),
    ]
    oos_eval_out = _run(oos_eval_cmd)
    oos_eval_run = Path(_last_line(oos_eval_out))
    oos_tpb = _read_tokens_per_byte(oos_eval_run)
    print(f"  ✓ OOS eval: {oos_tpb:.6f} tokens/byte")

    # Read training metrics
    train_metrics = json.loads((train_run / "metrics.json").read_text(encoding="utf-8"))

    return {
        "train_run": train_run,
        "eval_run": eval_run,
        "oos_eval_run": oos_eval_run,
        "metrics": {
            "test_tpb": test_tpb,
            "oos_tpb": oos_tpb,
            "train_wall_time_s": train_metrics.get("train_wall_time_s"),
            "peak_rss_kb": train_metrics.get("peak_rss_kb"),
        },
    }


def run_ensemble_training(
    train_file: Path,
    test_file: Path,
    num_shards: int,
    vocab: int,
    exp_name: str,
    artifacts_dir: Path,
) -> Path:
    """Train ensemble of K tokenizers on data shards."""
    print(f"\n{'='*80}")
    print(f"ENSEMBLE TRAINING: K={num_shards}, vocab={vocab}")
    print(f"{'='*80}")

    ens_cmd = [
        "uv", "run", "--with", "tokenizers", "--with", "psutil",
        "python", "-m", "scripts.train_ensemble",
        "--exp", exp_name,
        "--train", str(train_file),
        "--eval-file", str(test_file),
        "--num-shards", str(num_shards),
        "--vocab-size", str(vocab),
        "--out", str(artifacts_dir),
    ]
    ens_out = _run(ens_cmd)
    ens_dir = Path(_last_line(ens_out))
    print(f"  ✓ Ensemble training complete: {ens_dir.name}")

    # Load ensemble metadata
    ens_meta = json.loads((ens_dir / "ensemble.json").read_text(encoding="utf-8"))
    best = ens_meta.get("best", {})
    print(f"  ✓ Best member: {best.get('tokens_per_byte', 0):.6f} tokens/byte on TEST")

    return ens_dir


def eval_method_on_oos(
    tokenizer_path: Path,
    oos_file: Path,
    exp_name: str,
    artifacts_dir: Path,
) -> Path:
    """Evaluate a tokenizer on OOS dataset."""
    eval_cmd = [
        "uv", "run", "--with", "tokenizers",
        "python", "-m", "scripts.eval_compression",
        "--exp", exp_name,
        "--tokenizer", str(tokenizer_path),
        "--eval-file", str(oos_file),
        "--out", str(artifacts_dir),
    ]
    eval_out = _run(eval_cmd)
    eval_run = Path(_last_line(eval_out))
    return eval_run


def run_ensemble_method_selection(
    ens_dir: Path,
    test_file: Path,
    oos_file: Path,
    exp_name: str,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """METHOD 1: TOP-1 SELECTION"""
    print(f"\n{'='*80}")
    print(f"METHOD 1: TOP-1 SELECTION")
    print(f"{'='*80}")

    selected_tok = ens_dir / "selected_tokenizer.json"
    if not selected_tok.exists():
        print("  ✗ No selected tokenizer found!")
        return {}

    # Eval on TEST
    test_eval_cmd = [
        "uv", "run", "--with", "tokenizers",
        "python", "-m", "scripts.eval_compression",
        "--exp", f"{exp_name}_selected_eval",
        "--tokenizer", str(selected_tok),
        "--eval-file", str(test_file),
        "--out", str(artifacts_dir),
    ]
    test_eval_out = _run(test_eval_cmd)
    test_eval_run = Path(_last_line(test_eval_out))
    test_tpb = _read_tokens_per_byte(test_eval_run)
    print(f"  ✓ TEST: {test_tpb:.6f} tokens/byte")

    # Eval on OOS
    oos_eval_run = eval_method_on_oos(selected_tok, oos_file, f"{exp_name}_selected_eval_oos", artifacts_dir)
    oos_tpb = _read_tokens_per_byte(oos_eval_run)
    print(f"  ✓ OOS: {oos_tpb:.6f} tokens/byte")

    return {
        "method": "selection",
        "test_eval_run": test_eval_run,
        "oos_eval_run": oos_eval_run,
        "test_tpb": test_tpb,
        "oos_tpb": oos_tpb,
    }


def run_ensemble_method_merge_majority(
    ens_dir: Path,
    test_file: Path,
    oos_file: Path,
    num_shards: int,
    exp_base: str,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """METHOD 2: MAJORITY MERGE VOTING (k=ceil(n/2))"""
    print(f"\n{'='*80}")
    print(f"METHOD 2: MAJORITY MERGE VOTING")
    print(f"{'='*80}")

    k = max(1, math.ceil(num_shards / 2))
    print(f"  Testing k={k} of n={num_shards} (majority)")

    # Build merged tokenizer
    merge_exp = f"{exp_base}_merge_majority"
    merge_cmd = [
        "uv", "run", "--with", "tokenizers",
        "python", "-m", "scripts.merge_ensemble",
        "--exp", merge_exp,
        "--ensemble-run", str(ens_dir),
        "--k", str(k),
        "--out", str(artifacts_dir),
    ]
    merge_out = _run(merge_cmd)
    merge_dir = Path(_last_line(merge_out))
    merge_tok = merge_dir / "tokenizer.json"
    print(f"  ✓ Merge tokenizer created: {merge_dir.name}")

    # Eval on TEST
    test_eval_cmd = [
        "uv", "run", "--with", "tokenizers",
        "python", "-m", "scripts.eval_compression",
        "--exp", f"{merge_exp}_eval",
        "--tokenizer", str(merge_tok),
        "--eval-file", str(test_file),
        "--out", str(artifacts_dir),
    ]
    test_eval_out = _run(test_eval_cmd)
    test_eval_run = Path(_last_line(test_eval_out))
    test_tpb = _read_tokens_per_byte(test_eval_run)
    print(f"  ✓ TEST: {test_tpb:.6f} tokens/byte")

    # Eval on OOS
    oos_eval_run = eval_method_on_oos(merge_tok, oos_file, f"{merge_exp}_eval_oos", artifacts_dir)
    oos_tpb = _read_tokens_per_byte(oos_eval_run)
    print(f"  ✓ OOS: {oos_tpb:.6f} tokens/byte")

    return {
        "method": "merge_majority",
        "k": k,
        "n": num_shards,
        "merge_dir": merge_dir,
        "test_eval_run": test_eval_run,
        "oos_eval_run": oos_eval_run,
        "test_tpb": test_tpb,
        "oos_tpb": oos_tpb,
    }


def run_ensemble_method_weighted_entropy(
    ens_dir: Path,
    test_file: Path,
    oos_file: Path,
    exp_base: str,
    artifacts_dir: Path,
) -> List[Dict[str, Any]]:
    """METHOD 3: ENTROPY-WEIGHTED VOTING with θ ∈ {0.3, 0.7}"""
    print(f"\n{'='*80}")
    print(f"METHOD 3: ENTROPY-WEIGHTED VOTING")
    print(f"{'='*80}")

    results = []
    thetas = [0.3, 0.7]  # Test extremes only

    for theta in thetas:
        print(f"\n  Testing entropy weighting, theta={theta}")

        # Build weighted tokenizer
        merge_exp = f"{exp_base}_weighted_entropy_{int(theta*100)}"
        merge_cmd = [
            "uv", "run", "--with", "tokenizers",
            "python", "-m", "scripts.merge_ensemble",
            "--exp", merge_exp,
            "--ensemble-run", str(ens_dir),
            "--weight-by", "entropy",
            "--theta", str(theta),
            "--out", str(artifacts_dir),
        ]
        try:
            merge_out = _run(merge_cmd)
            merge_dir = Path(_last_line(merge_out))
            merge_tok = merge_dir / "tokenizer.json"
            print(f"    ✓ Weighted tokenizer created: {merge_dir.name}")

            # Eval on TEST
            test_eval_cmd = [
                "uv", "run", "--with", "tokenizers",
                "python", "-m", "scripts.eval_compression",
                "--exp", f"{merge_exp}_eval",
                "--tokenizer", str(merge_tok),
                "--eval-file", str(test_file),
                "--out", str(artifacts_dir),
            ]
            test_eval_out = _run(test_eval_cmd)
            test_eval_run = Path(_last_line(test_eval_out))
            test_tpb = _read_tokens_per_byte(test_eval_run)
            print(f"    ✓ TEST: {test_tpb:.6f} tokens/byte")

            # Eval on OOS
            oos_eval_run = eval_method_on_oos(merge_tok, oos_file, f"{merge_exp}_eval_oos", artifacts_dir)
            oos_tpb = _read_tokens_per_byte(oos_eval_run)
            print(f"    ✓ OOS: {oos_tpb:.6f} tokens/byte")

            results.append({
                "method": f"weighted_entropy_{int(theta*100)}",
                "weight_by": "entropy",
                "theta": theta,
                "merge_dir": merge_dir,
                "test_eval_run": test_eval_run,
                "oos_eval_run": oos_eval_run,
                "test_tpb": test_tpb,
                "oos_tpb": oos_tpb,
            })
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            continue

    return results


def run_single_experiment(
    seed: int,
    vocab: int,
    k: int,
    train_file: Path,
    test_file: Path,
    oos_file: Path,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """Run complete experiment for one (seed, vocab, K) configuration."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: seed={seed}, vocab={vocab}, K={k}")
    print(f"{'='*80}")

    results = {
        "seed": seed,
        "vocab": vocab,
        "k": k,
        "baseline": None,
        "ensemble_dir": None,
        "methods": {},
    }

    # Always run baseline
    baseline_exp = f"baseline_fast_s{seed}_v{vocab}"
    baseline_results = run_baseline(
        train_file, test_file, oos_file, vocab, seed, baseline_exp, artifacts_dir
    )
    results["baseline"] = baseline_results

    # Train ensemble members
    ens_exp = f"ens_fast_s{seed}_v{vocab}_k{k}"
    ens_dir = run_ensemble_training(
        train_file, test_file, k, vocab, ens_exp, artifacts_dir
    )
    results["ensemble_dir"] = ens_dir

    # Method 1: Selection
    sel_results = run_ensemble_method_selection(
        ens_dir, test_file, oos_file, ens_exp, artifacts_dir
    )
    if sel_results:
        results["methods"]["selection"] = sel_results

    # Method 2: Merge majority
    merge_result = run_ensemble_method_merge_majority(
        ens_dir, test_file, oos_file, k, ens_exp, artifacts_dir
    )
    if merge_result:
        results["methods"]["merge_majority"] = merge_result

    # Method 3: Weighted entropy
    weighted_results = run_ensemble_method_weighted_entropy(
        ens_dir, test_file, oos_file, ens_exp, artifacts_dir
    )
    for wr in weighted_results:
        results["methods"][wr["method"]] = wr

    return results


def generate_comparison_tables(all_results: List[Dict[str, Any]], out_dir: Path) -> None:
    """Generate comparison tables across all methods and configurations."""
    print(f"\n{'='*80}")
    print(f"GENERATING COMPARISON TABLES")
    print(f"{'='*80}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Flatten results for CSV
    rows = []
    for exp in all_results:
        seed = exp["seed"]
        vocab = exp["vocab"]
        k = exp["k"]

        # Baseline row
        if exp["baseline"]:
            bl = exp["baseline"]["metrics"]
            rows.append({
                "seed": seed,
                "vocab": vocab,
                "k": k,
                "method": "baseline",
                "test_tpb": bl["test_tpb"],
                "oos_tpb": bl["oos_tpb"],
                "train_time_s": bl["train_wall_time_s"],
                "peak_rss_kb": bl["peak_rss_kb"],
            })

        # Ensemble method rows
        for method_name, method_data in exp.get("methods", {}).items():
            rows.append({
                "seed": seed,
                "vocab": vocab,
                "k": k,
                "method": method_name,
                "test_tpb": method_data["test_tpb"],
                "oos_tpb": method_data["oos_tpb"],
                "train_time_s": None,
                "peak_rss_kb": None,
            })

    # Write CSV
    csv_path = out_dir / "method_comparison_fast.csv"
    if rows:
        fieldnames = ["seed", "vocab", "k", "method", "test_tpb", "oos_tpb", "train_time_s", "peak_rss_kb"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  ✓ Wrote {csv_path}")

    # Find best method per config
    rankings = []
    for exp in all_results:
        if not exp.get("methods"):
            continue

        seed = exp["seed"]
        vocab = exp["vocab"]
        k = exp["k"]
        baseline_oos = exp["baseline"]["metrics"]["oos_tpb"]

        # Rank methods by OOS performance
        method_scores = []
        for method_name, method_data in exp["methods"].items():
            oos_tpb = method_data["oos_tpb"]
            improvement = (baseline_oos - oos_tpb) / baseline_oos * 100
            method_scores.append({
                "method": method_name,
                "oos_tpb": oos_tpb,
                "improvement_pct": improvement,
            })

        method_scores.sort(key=lambda x: x["oos_tpb"])
        best = method_scores[0] if method_scores else None

        if best:
            rankings.append({
                "seed": seed,
                "vocab": vocab,
                "k": k,
                "best_method": best["method"],
                "best_oos_tpb": best["oos_tpb"],
                "baseline_oos_tpb": baseline_oos,
                "improvement_pct": best["improvement_pct"],
            })

    # Write rankings CSV
    rankings_path = out_dir / "method_rankings_fast.csv"
    if rankings:
        fieldnames = ["seed", "vocab", "k", "best_method", "best_oos_tpb", "baseline_oos_tpb", "improvement_pct"]
        with rankings_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rankings)
        print(f"  ✓ Wrote {rankings_path}")

    print(f"\n{'='*80}")
    print(f"COMPARISON TABLES COMPLETE")
    print(f"{'='*80}")
    print(f"Results: {out_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fast ensemble tokenization experiments (reduced parameter sweep)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--profile", choices=["small", "full"], default="small",
                    help="Experiment scale: small (laptop) or full (server)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 17, 19],
                    help="Random seeds for experiments")
    ap.add_argument("--out", default="artifacts/paper_fast",
                    help="Output directory")
    ap.add_argument("--data-dir", required=True,
                    help="Pre-prepared data directory")
    args = ap.parse_args(argv)

    # Fixed reduced parameters
    vocab_sizes = [16384, 32768]  # Only 16k and 32k
    k_values = [4, 16]  # Only K=4 and K=16

    artifacts_dir = Path(args.out)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_file = data_dir / "train.txt"
    test_file = data_dir / "test.txt"
    oos_file = data_dir / "test_oos.txt"

    # Validate data files exist
    for f in [train_file, test_file, oos_file]:
        if not f.exists():
            print(f"ERROR: Data file not found: {f}", file=sys.stderr)
            return 1

    print(f"\n{'='*80}")
    print(f"FAST ENSEMBLE EXPERIMENT RUNNER")
    print(f"{'='*80}")
    print(f"Profile: {args.profile}")
    print(f"Seeds: {args.seeds}")
    print(f"Vocabularies: {vocab_sizes}")
    print(f"K values: {k_values}")
    print(f"Methods: Selection + Merge Majority + Weighted Entropy (θ=0.3, 0.7)")
    print(f"Data: {data_dir}")
    print(f"Output: {artifacts_dir}")
    print(f"{'='*80}")

    # Calculate totals
    total = len(args.seeds) * len(vocab_sizes) * len(k_values)
    print(f"\nTotal configs: {total}")
    print(f"Methods per config: 5 (baseline + selection + merge + 2 weighted)")
    print(f"Total experiments: {total * 5} (vs 396 in full version)")
    print(f"Speedup: ~6.6x faster\n")

    # Run all experiments
    all_results = []
    current = 0

    for seed in args.seeds:
        for vocab in vocab_sizes:
            for k_val in k_values:
                current += 1
                print(f"\n{'='*80}")
                print(f"PROGRESS: {current}/{total}")
                print(f"{'='*80}")

                exp_results = run_single_experiment(
                    seed, vocab, k_val,
                    train_file, test_file, oos_file,
                    artifacts_dir
                )
                all_results.append(exp_results)

    # Generate comparison tables
    results_dir = artifacts_dir / "results"
    generate_comparison_tables(all_results, results_dir)

    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*80}")
    print(f"Total configurations: {total}")
    print(f"Results: {results_dir}")
    print(f"  - method_comparison_fast.csv")
    print(f"  - method_rankings_fast.csv")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
