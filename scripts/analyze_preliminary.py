"""
Quick preliminary analysis for seeds 13 & 17 while waiting for seed 19.
Generates tables showing N=2 results with mean ± std.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def load_baseline_results(csv_path: Path, seeds: list[int]) -> list[dict]:
    """Load baseline BPE results and add OOS metrics."""
    results = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["seed"]) not in seeds:
                continue

            # Load OOS metrics from eval_oos run
            eval_run = Path(row["eval_run"])
            oos_dir = Path(str(eval_run).replace("_eval", "_eval_oos"))

            if oos_dir.exists():
                oos_metrics_file = oos_dir / "metrics.json"
                if oos_metrics_file.exists():
                    oos_metrics = json.loads(oos_metrics_file.read_text())
                    row["tokens_per_byte_oos"] = oos_metrics.get("tokens_per_byte")

            results.append(row)

    return results


def load_ensemble_results(ensembles_dir: Path, seeds: list[int]) -> list[dict]:
    """Load all ensemble experiment results."""
    results = []

    for ens_dir in ensembles_dir.iterdir():
        if not ens_dir.is_dir():
            continue

        # Parse directory name: s{seed}_v{vocab}_k{K}
        parts = ens_dir.name.split("_")
        if len(parts) < 3:
            continue

        seed = int(parts[0][1:])  # s13 -> 13
        vocab = int(parts[1][1:])  # v16384 -> 16384
        k = int(parts[2][1:])  # k2 -> 2

        if seed not in seeds:
            continue

        # Load e0_summary.csv
        summary_csv = ens_dir / "e0_summary.csv"
        if not summary_csv.exists():
            continue

        with summary_csv.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Extract metrics
        single_test = single_oos = ens_test = ens_oos = None

        for row in rows:
            kind = row["kind"]
            tpb = float(row["tokens_per_byte"])

            if kind == "hf_bpe" and single_test is None:
                single_test = tpb
            elif kind == "hf_bpe_oos" and single_oos is None:
                single_oos = tpb
            elif kind == "ensemble_selected" and ens_test is None:
                ens_test = tpb
            elif kind == "ensemble_selected_oos" and ens_oos is None:
                ens_oos = tpb

        # Load ensemble.json for member stats and training metrics
        ens_json_path = None
        for p in Path("artifacts").glob(f"*_ens_s{seed}_v{vocab}_k{k}_base_ens/ensemble.json"):
            ens_json_path = p
            break

        if ens_json_path:
            ens_data = json.loads(ens_json_path.read_text())

            # Get member metrics
            member_compressions = [m["tokens_per_byte"] for m in ens_data["members"]]

            # Get training metrics for members
            member_times = []
            member_rss = []

            for member in ens_data["members"]:
                train_run = Path(member["train_run"])
                metrics_file = train_run / "metrics.json"
                if metrics_file.exists():
                    metrics = json.loads(metrics_file.read_text())
                    member_times.append(metrics.get("train_wall_time_s", 0))
                    member_rss.append(metrics.get("peak_rss_kb", 0) / 1024)  # Convert to MB

            # Get single baseline training metrics
            single_train_run = None
            for row in rows:
                if row["kind"] == "hf_bpe":
                    single_train_run = Path(row["run_dir"])
                    break

            single_time = single_rss = None
            if single_train_run and single_train_run.exists():
                metrics_file = single_train_run / "metrics.json"
                if metrics_file.exists():
                    metrics = json.loads(metrics_file.read_text())
                    single_time = metrics.get("train_wall_time_s")
                    single_rss = metrics.get("peak_rss_kb", 0) / 1024

            results.append({
                "seed": seed,
                "vocab": vocab,
                "k": k,
                "single_test": single_test,
                "single_oos": single_oos,
                "ensemble_test": ens_test,
                "ensemble_oos": ens_oos,
                "member_compression_mean": statistics.mean(member_compressions),
                "member_compression_std": statistics.stdev(member_compressions) if len(member_compressions) > 1 else 0,
                "member_compression_min": min(member_compressions),
                "member_compression_max": max(member_compressions),
                "single_time_s": single_time,
                "single_rss_mb": single_rss,
                "member_time_mean": statistics.mean(member_times) if member_times else None,
                "ensemble_total_time_s": sum(member_times) if member_times else None,
                "member_rss_mean": statistics.mean(member_rss) if member_rss else None,
            })

    return results


def compute_aggregated_stats(results: list[dict], group_by: list[str]) -> list[dict]:
    """Aggregate results by grouping keys, compute mean±std."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        key = tuple(r[k] for k in group_by)
        groups[key].append(r)

    agg = []
    for key, group in groups.items():
        row = {k: v for k, v in zip(group_by, key)}
        row["n_seeds"] = len(group)

        # Compute stats for numeric columns
        for col in ["single_test", "single_oos", "ensemble_test", "ensemble_oos",
                   "single_time_s", "single_rss_mb", "ensemble_total_time_s", "member_rss_mean"]:
            vals = [r[col] for r in group if r.get(col) is not None]
            if vals:
                row[f"{col}_mean"] = statistics.mean(vals)
                row[f"{col}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0

        # Compute improvements
        if row.get("single_oos_mean") and row.get("ensemble_oos_mean"):
            improvement = (row["single_oos_mean"] - row["ensemble_oos_mean"]) / row["single_oos_mean"] * 100
            row["compression_improvement_pct"] = improvement

        if row.get("single_rss_mb_mean") and row.get("member_rss_mean_mean"):
            reduction = (row["single_rss_mb_mean"] - row["member_rss_mean_mean"]) / row["single_rss_mb_mean"] * 100
            row["memory_reduction_pct"] = reduction

        if row.get("single_time_s_mean") and row.get("ensemble_total_time_s_mean"):
            overhead = (row["ensemble_total_time_s_mean"] - row["single_time_s_mean"]) / row["single_time_s_mean"] * 100
            row["time_overhead_pct"] = overhead

        agg.append(row)

    return sorted(agg, key=lambda r: (r["vocab"], r["k"]))


def format_table(data: list[dict], title: str) -> str:
    """Format data as Markdown table."""
    if not data:
        return f"# {title}\n\nNo data available.\n"

    lines = [f"# {title}\n"]

    # Get all keys
    keys = list(data[0].keys())

    # Header
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join(["---"] * len(keys)) + "|")

    # Rows
    for row in data:
        vals = []
        for k in keys:
            v = row.get(k)
            if v is None:
                vals.append("N/A")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines) + "\n\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preliminary analysis for seeds 13 & 17")
    ap.add_argument("--baseline", default="artifacts/paper_final/bpe_training_results.csv")
    ap.add_argument("--ensembles", default="artifacts/paper_final/ensembles")
    ap.add_argument("--out", default="results/preliminary_n2")
    args = ap.parse_args(argv)

    seeds = [13, 17]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline results for seeds {seeds}...")
    baseline = load_baseline_results(Path(args.baseline), seeds)

    print(f"Loading ensemble results for seeds {seeds}...")
    ensemble_results = load_ensemble_results(Path(args.ensembles), seeds)

    print(f"Found {len(baseline)} baseline runs, {len(ensemble_results)} ensemble runs")

    # Aggregate baseline by vocab
    baseline_by_vocab = {}
    for vocab in [16384, 32768, 65536]:
        rows = [r for r in baseline if int(r["vocab_size"]) == vocab]
        if rows:
            test_vals = [float(r["tokens_per_byte"]) for r in rows]
            oos_vals = [float(r["tokens_per_byte_oos"]) for r in rows if r.get("tokens_per_byte_oos")]
            time_vals = [float(r["train_wall_time_s"]) for r in rows]
            rss_vals = [float(r["peak_rss_kb"]) / 1024 for r in rows]

            baseline_by_vocab[vocab] = {
                "vocab": vocab,
                "n_seeds": len(rows),
                "test_mean": statistics.mean(test_vals),
                "test_std": statistics.stdev(test_vals) if len(test_vals) > 1 else 0,
                "oos_mean": statistics.mean(oos_vals) if oos_vals else None,
                "oos_std": statistics.stdev(oos_vals) if len(oos_vals) > 1 else 0,
                "time_mean": statistics.mean(time_vals),
                "time_std": statistics.stdev(time_vals) if len(time_vals) > 1 else 0,
                "rss_mb_mean": statistics.mean(rss_vals),
                "rss_mb_std": statistics.stdev(rss_vals) if len(rss_vals) > 1 else 0,
            }

    # Aggregate ensemble by (vocab, K)
    ensemble_agg = compute_aggregated_stats(ensemble_results, ["vocab", "k"])

    # Write outputs
    with (out_dir / "baseline_summary.md").open("w") as f:
        f.write(format_table(list(baseline_by_vocab.values()), "Baseline BPE Performance (N=2 seeds)"))

    with (out_dir / "ensemble_summary.md").open("w") as f:
        f.write(format_table(ensemble_agg, "Ensemble vs Single Comparison (N=2 seeds)"))

    # Write CSVs
    if baseline_by_vocab:
        keys = list(baseline_by_vocab.values())[0].keys()
        with (out_dir / "baseline_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(baseline_by_vocab.values())

    if ensemble_agg:
        keys = list(ensemble_agg[0].keys())
        with (out_dir / "ensemble_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(ensemble_agg)

    # Print key findings
    print("\n" + "="*80)
    print("PRELIMINARY RESULTS (N=2 seeds: 13, 17)")
    print("="*80)

    print("\n## Baseline BPE Performance")
    for vocab, stats in sorted(baseline_by_vocab.items()):
        print(f"\nVocab {vocab//1024}k:")
        print(f"  Test:    {stats['test_mean']:.4f} ± {stats['test_std']:.4f} tokens/byte")
        if stats['oos_mean']:
            print(f"  OOS:     {stats['oos_mean']:.4f} ± {stats['oos_std']:.4f} tokens/byte")
        print(f"  Time:    {stats['time_mean']:.1f} ± {stats['time_std']:.1f} seconds")
        print(f"  Memory:  {stats['rss_mb_mean']:.0f} ± {stats['rss_mb_std']:.0f} MB")

    print("\n## Ensemble Results (Key Findings)")
    for row in ensemble_agg:
        if row["k"] == 8 and row["vocab"] == 16384:  # Highlight one config
            print(f"\nExample: Vocab 16k, K=8")
            print(f"  Single OOS:       {row.get('single_oos_mean', 0):.4f} ± {row.get('single_oos_std', 0):.4f}")
            print(f"  Ensemble OOS:     {row.get('ensemble_oos_mean', 0):.4f} ± {row.get('ensemble_oos_std', 0):.4f}")
            print(f"  Improvement:      {row.get('compression_improvement_pct', 0):+.2f}%")
            print(f"  Memory reduction: {row.get('memory_reduction_pct', 0):+.1f}%")
            print(f"  Time overhead:    {row.get('time_overhead_pct', 0):+.1f}%")

    print(f"\n{'='*80}")
    print(f"Output written to: {out_dir.resolve()}")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
