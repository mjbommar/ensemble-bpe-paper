"""
Retrospective OOS evaluation for ensemble selected tokenizers.

Problem: Ensemble experiments evaluate selected tokenizer on TEST set only,
not OOS. This script evaluates all selected tokenizers on OOS data and
updates the e0_summary.csv files.

Usage:
  uv run python -m scripts.evaluate_ensemble_oos \
    --ensembles-dir artifacts/paper_final/ensembles \
    --oos-data data/processed/pg_bpe_full/test_oos.txt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from tokenizers import Tokenizer


def find_ensemble_experiments(ensembles_dir: Path) -> List[Dict[str, Path]]:
    """
    Find all completed ensemble experiments.

    Returns list of dicts with:
      - experiment_dir: the s{seed}_v{vocab}_k{K} directory
      - ensemble_base_dir: the actual ensemble run directory with selected_tokenizer.json
      - e0_summary: path to e0_summary.csv
    """
    experiments = []

    for exp_dir in ensembles_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        # Parse seed, vocab, k from experiment dir name
        parts = exp_dir.name.split("_")
        if len(parts) < 3:
            continue

        seed = parts[0][1:]  # s13 -> 13
        vocab = parts[1][1:]  # v16384 -> 16384
        k = parts[2][1:]  # k2 -> 2

        # Look for e0_summary.csv
        summary_csv = exp_dir / "e0_summary.csv"
        if not summary_csv.exists():
            continue

        # Find the ensemble base directory by searching artifacts
        # Pattern: {timestamp}_bpe_pg_full_ens_s{seed}_v{vocab}_k{k}_base_ens
        pattern = f"*_ens_s{seed}_v{vocab}_k{k}_base_ens"

        matching_dirs = list(Path("artifacts").glob(pattern))

        if not matching_dirs:
            continue

        # Take the most recent one (in case of multiple runs)
        ensemble_base_dir = sorted(matching_dirs)[-1]

        selected_tok = ensemble_base_dir / "selected_tokenizer.json"

        if not selected_tok.exists():
            continue

        experiments.append({
            "experiment_dir": exp_dir,
            "ensemble_base_dir": ensemble_base_dir,
            "e0_summary": summary_csv,
            "selected_tokenizer": selected_tok,
            "seed": int(seed),
            "vocab": int(vocab),
            "k": int(k),
        })

    return sorted(experiments, key=lambda x: (x["vocab"], x["k"], x["seed"]))


def evaluate_tokenizer_oos(tokenizer_path: Path, oos_file: Path) -> Dict[str, float]:
    """
    Evaluate a tokenizer on OOS data.

    Returns metrics dict with tokens_per_byte, bytes_per_token, etc.
    """
    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Read OOS data
    with oos_file.open("r", encoding="utf-8") as f:
        text = f.read()

    # Encode
    encoding = tokenizer.encode(text)

    # Calculate metrics
    total_bytes = len(text.encode("utf-8"))
    total_chars = len(text)
    total_tokens = len(encoding.tokens)

    metrics = {
        "total_bytes": total_bytes,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "tokens_per_byte": total_tokens / total_bytes if total_bytes > 0 else 0,
        "bytes_per_token": total_bytes / total_tokens if total_tokens > 0 else 0,
        "chars_per_token": total_chars / total_tokens if total_tokens > 0 else 0,
    }

    return metrics


def update_e0_summary_with_oos(experiment: Dict, oos_metrics: Dict[str, float]) -> None:
    """
    Add ensemble_selected_oos row to e0_summary.csv
    """
    summary_csv = experiment["e0_summary"]

    # Read existing rows
    with summary_csv.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check if OOS row already exists
    has_oos = any(r.get("kind") == "ensemble_selected_oos" for r in rows)
    if has_oos:
        print(f"  ⚠️  OOS row already exists, skipping")
        return

    # Find the ensemble_selected row to get metadata
    selected_row = None
    for row in rows:
        if row.get("kind") == "ensemble_selected":
            selected_row = row
            break

    if not selected_row:
        print(f"  ❌ No ensemble_selected row found")
        return

    # Create new OOS row
    oos_row = {
        "pipeline_dir": selected_row["pipeline_dir"],
        "name": selected_row["name"].replace("_ens", "_ens_oos"),
        "kind": "ensemble_selected_oos",
        "run_dir": str(experiment["ensemble_base_dir"]),  # Point to ensemble dir
        "tokens_per_byte": f"{oos_metrics['tokens_per_byte']:.16f}",
    }

    rows.append(oos_row)

    # Write back
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pipeline_dir", "name", "kind", "run_dir", "tokens_per_byte"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  ✅ Added OOS row: tokens_per_byte={oos_metrics['tokens_per_byte']:.4f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate ensemble selected tokenizers on OOS data")
    ap.add_argument("--ensembles-dir", default="artifacts/paper_final/ensembles",
                   help="Directory containing ensemble experiment subdirectories")
    ap.add_argument("--oos-data", default="data/processed/pg_bpe_full/test_oos.txt",
                   help="OOS test file")
    ap.add_argument("--force", action="store_true",
                   help="Re-evaluate even if OOS metrics already exist")
    args = ap.parse_args(argv)

    ensembles_dir = Path(args.ensembles_dir)
    oos_file = Path(args.oos_data)

    if not ensembles_dir.exists():
        print(f"❌ Ensembles directory not found: {ensembles_dir}")
        return 1

    if not oos_file.exists():
        print(f"❌ OOS data file not found: {oos_file}")
        return 1

    print(f"Finding ensemble experiments in: {ensembles_dir}")
    experiments = find_ensemble_experiments(ensembles_dir)

    print(f"Found {len(experiments)} ensemble experiments")

    if not experiments:
        print("No experiments found. Check that e0_summary.csv files exist.")
        return 0

    print(f"\n{'='*80}")
    print("Evaluating selected tokenizers on OOS data")
    print(f"{'='*80}\n")

    for i, exp in enumerate(experiments, 1):
        seed = exp["seed"]
        vocab = exp["vocab"]
        k = exp["k"]

        print(f"[{i}/{len(experiments)}] Seed {seed}, Vocab {vocab}, K={k}")

        # Check if already has OOS metrics
        if not args.force:
            with exp["e0_summary"].open("r") as f:
                reader = csv.DictReader(f)
                if any(r.get("kind") == "ensemble_selected_oos" for r in reader):
                    print(f"  ⏭️  OOS metrics already exist (use --force to re-evaluate)")
                    continue

        # Evaluate on OOS
        print(f"  Evaluating: {exp['selected_tokenizer'].name}")
        oos_metrics = evaluate_tokenizer_oos(exp["selected_tokenizer"], oos_file)

        # Save metrics to ensemble directory
        metrics_file = exp["ensemble_base_dir"] / "selected_oos_metrics.json"
        with metrics_file.open("w") as f:
            json.dump(oos_metrics, f, indent=2)

        # Update e0_summary.csv
        update_e0_summary_with_oos(exp, oos_metrics)

    print(f"\n{'='*80}")
    print("✅ OOS evaluation complete!")
    print(f"{'='*80}\n")

    print("All e0_summary.csv files have been updated with ensemble_selected_oos rows.")
    print("Re-run your analysis scripts to see the OOS comparison.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
