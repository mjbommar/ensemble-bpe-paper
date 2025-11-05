#!/usr/bin/env python3
"""
Analyze full paper experiment results and generate statistics.

This script:
1. Loads aggregated results from Phase 4
2. Computes mean ± std across seeds for each (vocab, k, method, dataset)
3. Calculates percent improvement vs baseline and selection
4. Generates paper-ready result tables
5. Exports summary statistics for publication
"""

import json
import csv
import re
from pathlib import Path
from collections import defaultdict
import statistics
import argparse


def load_results(results_path):
    """Load and preprocess aggregated results."""
    with open(results_path) as f:
        results = json.load(f)

    # Fix metadata extraction for all results
    for r in results:
        name = r["name"]

        # Try to extract seed, vocab, k from name if not already set
        if r.get("seed") is None:
            match = re.search(r'_s(\d+)_v(\d+)', name)
            if match:
                r["seed"] = int(match.group(1))
                r["vocab"] = int(match.group(2))

            # For baseline, there's no k
            if "baseline" in name and "_k" not in name:
                r["k"] = None
            elif "_k" in name:
                k_match = re.search(r'_k(\d+)', name)
                if k_match:
                    r["k"] = int(k_match.group(1))

    return results


def compute_statistics(results):
    """
    Compute mean ± std across seeds for each configuration.

    Returns:
        dict: Nested dict structure: stats[vocab][k][dataset][method] = (mean, std, n)
    """
    # Group results by (seed, vocab, k, dataset, method)
    grouped = defaultdict(list)

    for r in results:
        seed = r.get("seed")
        vocab = r.get("vocab")
        k = r.get("k")
        dataset = r.get("dataset")
        method = r.get("method")
        tpb = r.get("tokens_per_byte")

        if None in [seed, vocab, dataset, method, tpb]:
            continue

        # Ensure tpb is a float
        if isinstance(tpb, str):
            tpb = float(tpb)

        # Group by config (averaging across multiple evals per seed)
        key = (seed, vocab, k, dataset, method)
        grouped[key].append(tpb)

    # Average within each seed, then compute stats across seeds
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for (seed, vocab, k, dataset, method), tpb_list in grouped.items():
        # Average multiple evaluations within this seed
        avg_tpb = statistics.mean(tpb_list)

        # Store in structure for cross-seed aggregation
        config_key = (vocab, k, dataset, method)
        if config_key not in stats:
            stats[config_key] = []
        stats[config_key].append(avg_tpb)

    # Compute final statistics across seeds
    final_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for (vocab, k, dataset, method), tpb_list in stats.items():
        mean_tpb = statistics.mean(tpb_list)
        std_tpb = statistics.stdev(tpb_list) if len(tpb_list) > 1 else 0.0
        n = len(tpb_list)

        final_stats[vocab][k][dataset][method] = (mean_tpb, std_tpb, n)

    return final_stats


def compute_improvements(stats):
    """
    Compute percent improvement vs baseline and selection.

    Returns:
        dict: improvements[vocab][k][dataset][method] = {
            'vs_baseline': percent,
            'vs_selection': percent,
            'mean': value,
            'std': value,
            'n': count
        }
    """
    improvements = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for vocab in stats:
        for k in stats[vocab]:
            for dataset in stats[vocab][k]:
                methods = stats[vocab][k][dataset]

                # Get baseline (from k=None for this vocab)
                baseline_tpb = None
                if None in stats[vocab] and dataset in stats[vocab][None]:
                    if "baseline" in stats[vocab][None][dataset]:
                        baseline_tpb = stats[vocab][None][dataset]["baseline"][0]

                # Get selection
                selection_tpb = None
                if "selection" in methods:
                    selection_tpb = methods["selection"][0]

                # Compute improvements for each method
                for method, (mean_tpb, std_tpb, n) in methods.items():
                    improvements[vocab][k][dataset][method] = {
                        'mean': mean_tpb,
                        'std': std_tpb,
                        'n': n,
                        'vs_baseline': None,
                        'vs_selection': None
                    }

                    if baseline_tpb is not None:
                        # Percent improvement (negative because lower is better)
                        pct = ((baseline_tpb - mean_tpb) / baseline_tpb) * 100
                        improvements[vocab][k][dataset][method]['vs_baseline'] = pct

                    if selection_tpb is not None and method != "selection":
                        pct = ((selection_tpb - mean_tpb) / selection_tpb) * 100
                        improvements[vocab][k][dataset][method]['vs_selection'] = pct

    return improvements


def generate_summary_table(improvements, dataset="oos"):
    """Generate a formatted summary table for a specific dataset."""

    print(f"\n{'='*100}")
    print(f"SUMMARY TABLE: {dataset.upper()} Dataset")
    print(f"{'='*100}\n")

    # Header
    print(f"{'Config':<20} {'Method':<20} {'TPB':<15} {'vs Baseline':<15} {'vs Selection':<15}")
    print("-" * 100)

    # Sort by vocab, then k
    for vocab in sorted(improvements.keys()):
        for k in sorted([k for k in improvements[vocab].keys() if k is not None]):
            if dataset not in improvements[vocab][k]:
                continue

            methods = improvements[vocab][k][dataset]

            # Sort methods by performance
            sorted_methods = sorted(
                methods.items(),
                key=lambda x: x[1]['mean']
            )

            config_str = f"V={vocab//1024}K, K={k}"

            for i, (method, data) in enumerate(sorted_methods):
                # Highlight breakthrough methods
                marker = "⭐" if method in ["sequential", "exp_p2", "exp_p3"] else "  "

                mean_str = f"{data['mean']:.6f}±{data['std']:.6f}"

                vs_baseline = f"{data['vs_baseline']:+.2f}%" if data['vs_baseline'] is not None else "N/A"
                vs_selection = f"{data['vs_selection']:+.2f}%" if data['vs_selection'] is not None else "N/A"

                if i == 0:
                    print(f"{config_str:<20} {marker}{method:<18} {mean_str:<15} {vs_baseline:<15} {vs_selection:<15}")
                else:
                    print(f"{'':20} {marker}{method:<18} {mean_str:<15} {vs_baseline:<15} {vs_selection:<15}")

            print()


def generate_latex_table(improvements, dataset="oos", output_path=None):
    """Generate a LaTeX table for the paper."""

    lines = []
    lines.append("% Paper results table - Auto-generated")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Ensemble BPE Performance on " + dataset.upper() + " Dataset}")
    lines.append("\\label{tab:results_" + dataset + "}")
    lines.append("\\begin{tabular}{llrrr}")
    lines.append("\\toprule")
    lines.append("Config & Method & TPB ($\\downarrow$) & vs Baseline & vs Selection \\\\")
    lines.append("\\midrule")

    for vocab in sorted(improvements.keys()):
        for k in sorted([k for k in improvements[vocab].keys() if k is not None]):
            if dataset not in improvements[vocab][k]:
                continue

            methods = improvements[vocab][k][dataset]
            sorted_methods = sorted(methods.items(), key=lambda x: x[1]['mean'])

            # Show only top 3 + selection for space
            top_methods = []
            selection_included = False
            for method, data in sorted_methods[:5]:
                if method == "selection":
                    selection_included = True
                top_methods.append((method, data))

            # Ensure selection is included
            if not selection_included and "selection" in methods:
                top_methods.insert(1, ("selection", methods["selection"]))

            config_str = f"V={vocab//1024}K, K={k}"

            for i, (method, data) in enumerate(top_methods):
                marker = "\\textbf{" if method in ["sequential", "exp_p2", "exp_p3"] else ""
                marker_end = "}" if method in ["sequential", "exp_p2", "exp_p3"] else ""

                mean_str = f"{marker}{data['mean']:.4f}{marker_end}"

                vs_baseline = f"{data['vs_baseline']:+.1f}\\%" if data['vs_baseline'] is not None else "---"
                vs_selection = f"{data['vs_selection']:+.1f}\\%" if data['vs_selection'] is not None else "---"

                if i == 0:
                    lines.append(f"{config_str} & {marker}{method}{marker_end} & {mean_str} & {vs_baseline} & {vs_selection} \\\\")
                else:
                    lines.append(f" & {marker}{method}{marker_end} & {mean_str} & {vs_baseline} & {vs_selection} \\\\")

            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    latex_content = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_content)
        print(f"\nLaTeX table written to: {output_path}")

    return latex_content


def export_csv_summary(improvements, output_path):
    """Export summary statistics to CSV."""

    rows = []
    for vocab in sorted(improvements.keys()):
        for k in sorted([k for k in improvements[vocab].keys() if k is not None]):
            for dataset in sorted(improvements[vocab][k].keys()):
                methods = improvements[vocab][k][dataset]

                for method, data in sorted(methods.items(), key=lambda x: x[1]['mean']):
                    rows.append({
                        'vocab': vocab,
                        'k': k,
                        'dataset': dataset,
                        'method': method,
                        'mean_tpb': data['mean'],
                        'std_tpb': data['std'],
                        'n_seeds': data['n'],
                        'improvement_vs_baseline_pct': data['vs_baseline'],
                        'improvement_vs_selection_pct': data['vs_selection']
                    })

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary statistics exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze paper experiment results")
    parser.add_argument("--results", type=str,
                       default="/nas4/data/experiments/ensemble-bpe/paper_e2e_evaluations/results_aggregated.json",
                       help="Path to aggregated results JSON")
    parser.add_argument("--out", type=str,
                       default="/nas4/data/experiments/ensemble-bpe/paper_e2e_evaluations",
                       help="Output directory for analysis results")

    args = parser.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out)

    print("="*100)
    print("PAPER RESULTS ANALYSIS")
    print("="*100)
    print(f"\nLoading results from: {results_path}")

    # Load and process results
    results = load_results(results_path)
    print(f"Loaded {len(results)} evaluation records")

    # Compute statistics
    print("\nComputing statistics across seeds...")
    stats = compute_statistics(results)

    # Compute improvements
    print("Computing improvements vs baseline and selection...")
    improvements = compute_improvements(stats)

    # Generate summaries for each dataset
    for dataset in ["test", "oos", "fineweb"]:
        generate_summary_table(improvements, dataset=dataset)

    # Export results
    csv_path = out_dir / "results_summary_statistics.csv"
    export_csv_summary(improvements, csv_path)

    # Generate LaTeX tables
    for dataset in ["test", "oos", "fineweb"]:
        latex_path = out_dir / f"table_{dataset}.tex"
        generate_latex_table(improvements, dataset=dataset, output_path=latex_path)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"\nResults saved to: {out_dir}")
    print(f"  - results_summary_statistics.csv")
    print(f"  - table_test.tex")
    print(f"  - table_oos.tex")
    print(f"  - table_fineweb.tex")


if __name__ == "__main__":
    main()
