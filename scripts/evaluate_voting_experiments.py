"""
Evaluate all voting experiment tokenizers on test_oos.txt
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def evaluate_tokenizer(tokenizer_path: Path, test_file: Path, exp_name: str) -> dict:
    """Run eval_compression on a tokenizer and return metrics"""
    print(f"Evaluating {exp_name}...")

    # Run eval_compression
    result = subprocess.run(
        [
            "uv", "run", "--with", "tokenizers",
            "python", "-m", "scripts.eval_compression",
            "--exp", f"voting_eval_{exp_name}",
            "--tokenizer", str(tokenizer_path),
            "--eval-file", str(test_file),
            "--out", "artifacts/voting_experiments/evals",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return {}

    # Parse output to get run directory
    run_dir = Path(result.stdout.strip())
    metrics_file = run_dir / "metrics.json"

    if not metrics_file.exists():
        print(f"  ERROR: metrics.json not found in {run_dir}")
        return {}

    metrics = json.loads(metrics_file.read_text())
    tpb = metrics.get("tokens_per_byte", 0.0)
    print(f"  TPB: {tpb:.6f}")

    return {
        "name": exp_name,
        "tokenizer_path": str(tokenizer_path),
        "eval_dir": str(run_dir),
        "tokens_per_byte": tpb,
        **metrics,
    }


def main():
    # Test file
    test_file = Path("data/processed/pg_paper_full/test_oos.txt")

    # Get baseline: best member (selection)
    ensemble_dir = Path("artifacts/paper_fast/20251103_083104_ens_fast_s13_v16384_k4")
    selected_tokenizer = ensemble_dir / "selected_tokenizer.json"

    # Get baseline: quality-weighted voting theta=0.3
    # We need to check if this exists or generate it
    baseline_voting = None
    # Check existing merge runs
    import glob
    merge_runs = list(Path("artifacts").glob("*/merge_meta.json"))
    for run_meta_path in merge_runs:
        meta = json.loads(run_meta_path.read_text())
        if meta.get("theta") == 0.3:
            # Check if it used quality weighting
            merge_dir = run_meta_path.parent
            if (merge_dir / "tokenizer.json").exists():
                baseline_voting = merge_dir / "tokenizer.json"
                break

    results = []

    # Evaluate selection baseline
    if selected_tokenizer.exists():
        result = evaluate_tokenizer(
            selected_tokenizer,
            test_file,
            "selection_baseline"
        )
        if result:
            results.append(result)
            baseline_tpb = result["tokens_per_byte"]
        else:
            baseline_tpb = 0.275  # Fallback from user spec
    else:
        baseline_tpb = 0.275

    # Evaluate baseline quality voting if found
    if baseline_voting and baseline_voting.exists():
        result = evaluate_tokenizer(
            baseline_voting,
            test_file,
            "quality_voting_baseline"
        )
        if result:
            results.append(result)

    # Evaluate all voting experiments
    voting_exp_dir = Path("artifacts/voting_experiments")
    experiments = [
        ("exp_power_2.0", "Exponential p=2.0"),
        ("exp_power_3.0", "Exponential p=3.0"),
        ("sequential", "Sequential"),
        ("backbone_thresh_1.5", "Backbone thresh=1.5"),
        ("backbone_thresh_2.0", "Backbone thresh=2.0"),
        ("backbone_thresh_3.0", "Backbone thresh=3.0"),
        ("position_primary", "Position-Primary"),
    ]

    for exp_dir, display_name in experiments:
        tokenizer_path = voting_exp_dir / exp_dir / "tokenizer.json"
        if tokenizer_path.exists():
            result = evaluate_tokenizer(tokenizer_path, test_file, exp_dir)
            if result:
                result["display_name"] = display_name
                results.append(result)

    # Save all results
    results_file = voting_exp_dir / "evaluation_results.json"
    results_file.write_text(json.dumps(results, indent=2) + "\n")

    # Generate markdown table
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print()
    print("| Approach | Configuration | OOS TPB | Δ vs Selection | Notes |")
    print("|----------|--------------|---------|----------------|-------|")

    for result in results:
        name = result.get("display_name", result["name"])
        tpb = result["tokens_per_byte"]
        delta = ((tpb - baseline_tpb) / baseline_tpb * 100) if baseline_tpb > 0 else 0.0

        # Skip baseline from delta comparison
        if "baseline" in result["name"]:
            if "selection" in result["name"]:
                print(f"| BASELINE: Selection | best member | {tpb:.4f} | 0.0% | |")
            else:
                print(f"| BASELINE: Quality θ=0.3 | quality voting | {tpb:.4f} | {delta:+.2f}% | |")
        else:
            print(f"| {name} | - | {tpb:.4f} | {delta:+.2f}% | |")

    print()
    print(f"Baseline (Selection): {baseline_tpb:.4f} OOS TPB")
    print()
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
