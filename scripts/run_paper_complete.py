"""
================================================================================
COMPLETE ENSEMBLE TOKENIZATION EXPERIMENT RUNNER
================================================================================

⚠️  CRITICAL: This script tests ALL ensemble methods, not just selection!

ENSEMBLE METHODS IMPLEMENTED:
1. ✅ TOP-1 SELECTION: Pick best member from K candidates (min tokens_per_byte)
2. ✅ K-OF-N MERGE VOTING: Include merges appearing in ≥k of n members
3. ✅ WEIGHTED MERGE VOTING: Vote with weights based on shard statistics

DO NOT MODIFY THIS SCRIPT TO REMOVE ANY ENSEMBLE METHODS WITHOUT EXPLICIT APPROVAL!

================================================================================

What this script does:

1. **Baseline Training**
   - Train single BPE on full training data
   - Evaluate on TEST and OOS
   - Establishes performance ceiling

2. **Ensemble Member Training**
   - Shard training data into K pieces
   - Train K independent BPE tokenizers
   - Evaluate each member on TEST set
   - Compute shard statistics for weighting

3. **Ensemble Method 1: Selection**
   - Pick the single best member (min tokens_per_byte on TEST)
   - Evaluate selected member on TEST and OOS
   - This is the SIMPLEST ensemble approach

4. **Ensemble Method 2: K-of-N Merge Voting**
   - Build tokenizer with merges appearing in ≥k members
   - Vocabulary: union of all member vocabs
   - Test multiple k thresholds:
     * k = ceil(n/4): 25% threshold (lenient)
     * k = ceil(n/2): 50% threshold (majority)
     * k = ceil(3n/4): 75% threshold (strict)
   - Evaluate each variant on TEST and OOS

5. **Ensemble Method 3: Weighted Merge Voting**
   - Weight members by shard characteristics
   - Weighting schemes:
     * entropy: byte-level entropy (high entropy = more informative)
     * char_entropy: character-level entropy
   - Thresholds (theta):
     * 0.3: lenient (30% weighted support)
     * 0.5: moderate (50% weighted support)
     * 0.7: strict (70% weighted support)
   - Evaluate each variant on TEST and OOS

6. **Comparison & Analysis**
   - Generate tables comparing all methods
   - Rank methods by OOS performance
   - Identify which method wins for which (vocab, K) config

================================================================================

Usage:

  # Small-scale test (laptop-safe)
  uv run python -m scripts.run_paper_complete --profile small

  # Full paper experiments (server-scale)
  uv run python -m scripts.run_paper_complete --profile full --seeds 13 17 19

  # Test specific methods only
  uv run python -m scripts.run_paper_complete --methods selection merge

  # Custom configuration
  uv run python -m scripts.run_paper_complete \\
      --seeds 13 17 \\
      --sizes 16384 32768 \\
      --k 2 4 8 \\
      --methods all

================================================================================

Output Structure:

  artifacts/paper_complete/
    baseline/
      bpe_training_results.csv        # Single baseline results
    ensembles/
      s{seed}_v{vocab}_k{K}/
        e0_summary_complete.csv        # All methods for this config
        ensemble.json                  # Member metadata + stats
        selected_tokenizer.json        # Method 1: Selection
        merge_k{X}_tokenizer.json      # Method 2: Voting variants
        weighted_{w}_{t}_tokenizer.json # Method 3: Weighted variants
    results/
      method_comparison.csv            # Cross-method comparison
      method_rankings_by_config.csv    # Winners per (vocab, K)
      oos_performance_summary.csv      # OOS metrics for all methods

================================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import tomllib


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
    """
    Run single baseline BPE training and evaluation.

    Returns dict with:
        - train_run: Path to training artifacts
        - eval_run: Path to test evaluation
        - oos_eval_run: Path to OOS evaluation
        - metrics: Dict of performance metrics
    """
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
    """
    Train ensemble of K tokenizers on data shards.

    Returns path to ensemble directory containing:
        - ensemble.json: member metadata with shard_stats
        - selected_tokenizer.json: best member (selection method)
    """
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
    """
    METHOD 1: TOP-1 SELECTION

    Evaluate the selected tokenizer (best member) on TEST and OOS.
    """
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


def run_ensemble_method_merge_voting(
    ens_dir: Path,
    test_file: Path,
    oos_file: Path,
    num_shards: int,
    exp_base: str,
    artifacts_dir: Path,
) -> List[Dict[str, Any]]:
    """
    METHOD 2: K-OF-N MERGE VOTING

    Build tokenizers with merges appearing in ≥k of n members.
    Test multiple k thresholds.
    """
    print(f"\n{'='*80}")
    print(f"METHOD 2: K-OF-N MERGE VOTING")
    print(f"{'='*80}")

    results = []

    # Test multiple k thresholds
    k_values = [
        max(1, math.ceil(num_shards / 4)),  # 25% threshold
        max(1, math.ceil(num_shards / 2)),  # 50% threshold (majority)
        max(1, math.ceil(3 * num_shards / 4)),  # 75% threshold
    ]
    k_values = sorted(set(k_values))  # Remove duplicates

    for k in k_values:
        print(f"\n  Testing k={k} of n={num_shards} ({100*k/num_shards:.0f}% threshold)")

        # Build merged tokenizer
        merge_exp = f"{exp_base}_merge_k{k}"
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
        print(f"    ✓ Merge tokenizer created: {merge_dir.name}")

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
            "method": f"merge_k{k}",
            "k": k,
            "n": num_shards,
            "threshold_pct": 100 * k / num_shards,
            "merge_dir": merge_dir,
            "test_eval_run": test_eval_run,
            "oos_eval_run": oos_eval_run,
            "test_tpb": test_tpb,
            "oos_tpb": oos_tpb,
        })

    return results


def run_ensemble_method_weighted_voting(
    ens_dir: Path,
    test_file: Path,
    oos_file: Path,
    exp_base: str,
    artifacts_dir: Path,
) -> List[Dict[str, Any]]:
    """
    METHOD 3: WEIGHTED MERGE VOTING

    Build tokenizers with weighted voting based on shard statistics.
    Test multiple weighting schemes and thresholds.
    """
    print(f"\n{'='*80}")
    print(f"METHOD 3: WEIGHTED MERGE VOTING")
    print(f"{'='*80}")

    results = []

    # Weighting schemes to test
    weight_schemes = ["entropy", "char_entropy"]
    thetas = [0.3, 0.5, 0.7]

    for weight_by in weight_schemes:
        for theta in thetas:
            print(f"\n  Testing weight_by={weight_by}, theta={theta}")

            # Build weighted tokenizer
            merge_exp = f"{exp_base}_weighted_{weight_by}_{int(theta*100)}"
            merge_cmd = [
                "uv", "run", "--with", "tokenizers",
                "python", "-m", "scripts.merge_ensemble",
                "--exp", merge_exp,
                "--ensemble-run", str(ens_dir),
                "--weight-by", weight_by,
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
                    "method": f"weighted_{weight_by}_{int(theta*100)}",
                    "weight_by": weight_by,
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
    methods: List[str],
) -> Dict[str, Any]:
    """
    Run complete experiment for one (seed, vocab, K) configuration.

    Tests all ensemble methods if K > 1, otherwise just baseline.
    """
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
    baseline_exp = f"baseline_s{seed}_v{vocab}"
    baseline_results = run_baseline(
        train_file, test_file, oos_file, vocab, seed, baseline_exp, artifacts_dir
    )
    results["baseline"] = baseline_results

    # If K=1, no ensemble
    if k == 1:
        print(f"\n  K=1: Skipping ensemble methods (same as baseline)")
        return results

    # Train ensemble members
    ens_exp = f"ens_s{seed}_v{vocab}_k{k}"
    ens_dir = run_ensemble_training(
        train_file, test_file, k, vocab, ens_exp, artifacts_dir
    )
    results["ensemble_dir"] = ens_dir

    # Method 1: Selection
    if "selection" in methods or "all" in methods:
        sel_results = run_ensemble_method_selection(
            ens_dir, test_file, oos_file, ens_exp, artifacts_dir
        )
        if sel_results:
            results["methods"]["selection"] = sel_results

    # Method 2: Merge voting
    if "merge" in methods or "all" in methods:
        merge_results = run_ensemble_method_merge_voting(
            ens_dir, test_file, oos_file, k, ens_exp, artifacts_dir
        )
        for mr in merge_results:
            results["methods"][mr["method"]] = mr

    # Method 3: Weighted voting
    if "weighted" in methods or "all" in methods:
        weighted_results = run_ensemble_method_weighted_voting(
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
                "train_time_s": None,  # TODO: aggregate member times
                "peak_rss_kb": None,
            })

    # Write CSV
    csv_path = out_dir / "method_comparison_complete.csv"
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
    rankings_path = out_dir / "method_rankings_by_config.csv"
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
        description="Complete ensemble tokenization experiments with ALL methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--profile", choices=["small", "full"], default="small",
                    help="Experiment scale: small (laptop) or full (server)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 17, 19],
                    help="Random seeds for experiments")
    ap.add_argument("--sizes", nargs="+", type=int, default=[16384, 32768, 65536],
                    help="Vocabulary sizes to test")
    ap.add_argument("--k", nargs="+", type=int, default=[2, 4, 8],
                    help="Ensemble K values (number of shards)")
    ap.add_argument("--methods", nargs="+", default=["all"],
                    choices=["all", "selection", "merge", "weighted"],
                    help="Which ensemble methods to test")
    ap.add_argument("--out", default="artifacts/paper_complete",
                    help="Output directory")
    ap.add_argument("--data-dir", help="Pre-prepared data directory (skip data prep)")
    args = ap.parse_args(argv)

    artifacts_dir = Path(args.out)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Determine data paths
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Use profile-specific data directory
        if args.profile == "small":
            data_dir = Path("data/processed/pg_paper_small")
        else:
            data_dir = Path("data/processed/pg_paper_full")

    train_file = data_dir / "train.txt"
    test_file = data_dir / "test.txt"
    oos_file = data_dir / "test_oos.txt"

    # Validate data files exist
    for f in [train_file, test_file, oos_file]:
        if not f.exists():
            print(f"ERROR: Data file not found: {f}", file=sys.stderr)
            print(f"Run data preparation first or specify --data-dir", file=sys.stderr)
            return 1

    print(f"\n{'='*80}")
    print(f"COMPLETE ENSEMBLE EXPERIMENT RUNNER")
    print(f"{'='*80}")
    print(f"Profile: {args.profile}")
    print(f"Seeds: {args.seeds}")
    print(f"Vocabularies: {args.sizes}")
    print(f"K values: {args.k}")
    print(f"Methods: {args.methods}")
    print(f"Data: {data_dir}")
    print(f"Output: {artifacts_dir}")
    print(f"{'='*80}")

    # Run all experiments
    all_results = []
    total = len(args.seeds) * len(args.sizes) * len(args.k)
    current = 0

    for seed in args.seeds:
        for vocab in args.sizes:
            for k_val in args.k:
                current += 1
                print(f"\n{'='*80}")
                print(f"PROGRESS: {current}/{total}")
                print(f"{'='*80}")

                exp_results = run_single_experiment(
                    seed, vocab, k_val,
                    train_file, test_file, oos_file,
                    artifacts_dir, args.methods
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
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
