"""Validate sequential and exponential methods on existing ensembles.

This script tests new merge methods (sequential voting, exponential weighting)
on pre-trained ensembles without requiring full retraining. It helps validate
that the new methods work correctly before running expensive full experiments.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


# Method configurations to test
METHODS = [
    {
        "name": "sequential",
        "method": "sequential",
        "weight_by": "quality",
        "description": "Sequential step-by-step voting (quality-weighted)",
    },
    {
        "name": "exponential_p2",
        "method": "exponential",
        "weight_by": "quality",
        "theta": 0.3,
        "power": 2.0,
        "description": "Exponential weighting with power=2.0, theta=0.3",
    },
    {
        "name": "exponential_p3",
        "method": "exponential",
        "weight_by": "quality",
        "theta": 0.3,
        "power": 3.0,
        "description": "Exponential weighting with power=3.0, theta=0.3",
    },
    {
        "name": "exponential_p2_t7",
        "method": "exponential",
        "weight_by": "quality",
        "theta": 0.7,
        "power": 2.0,
        "description": "Exponential weighting with power=2.0, theta=0.7",
    },
]


def find_ensemble_dirs(base_dir: Path) -> List[Path]:
    """Find all ensemble directories in the given base directory."""
    ens_dirs = []
    if not base_dir.exists():
        return ens_dirs

    for item in base_dir.iterdir():
        if item.is_dir() and (item / "ensemble.json").exists():
            ens_dirs.append(item)

    return sorted(ens_dirs)


def get_ensemble_info(ens_dir: Path) -> Dict[str, Any]:
    """Extract key information from ensemble.json."""
    ensemble_json = ens_dir / "ensemble.json"
    if not ensemble_json.exists():
        return {}

    data = json.loads(ensemble_json.read_text(encoding="utf-8"))
    members = data.get("members", [])

    # Extract vocab size and K from directory name if possible
    name = ens_dir.name
    vocab_size = None
    k = None

    # Parse patterns like: 20251103_083104_ens_fast_s13_v16384_k4
    parts = name.split("_")
    for part in parts:
        if part.startswith("v") and part[1:].isdigit():
            vocab_size = int(part[1:])
        elif part.startswith("k") and part[1:].isdigit():
            k = int(part[1:])

    return {
        "path": str(ens_dir),
        "name": ens_dir.name,
        "num_members": len(members),
        "vocab_size": vocab_size,
        "k": k,
    }


def run_merge(
    ens_dir: Path,
    method_cfg: Dict[str, Any],
    out_dir: Path,
) -> Path | None:
    """Run merge_ensemble for a given method configuration."""
    exp_name = f"validate_{method_cfg['name']}_{ens_dir.name}"

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "scripts.merge_ensemble",
        "--exp",
        exp_name,
        "--ensemble-run",
        str(ens_dir),
        "--method",
        method_cfg["method"],
        "--weight-by",
        method_cfg["weight_by"],
        "--out",
        str(out_dir),
    ]

    if "theta" in method_cfg:
        cmd.extend(["--theta", str(method_cfg["theta"])])
    if "power" in method_cfg:
        cmd.extend(["--power", str(method_cfg["power"])])

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"Method: {method_cfg['description']}")
    print(f"{'='*80}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output_dir = result.stdout.strip()
        print(f"Success! Output: {output_dir}")
        return Path(output_dir)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Merge failed!")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None


def get_tokenizer_stats(tokenizer_path: Path) -> Dict[str, Any]:
    """Extract basic stats from a tokenizer.json file."""
    if not tokenizer_path.exists():
        return {}

    try:
        data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        model = data.get("model", {})
        vocab = model.get("vocab", {})
        merges = model.get("merges", [])

        return {
            "vocab_size": len(vocab),
            "num_merges": len(merges),
            "tokenizer_type": model.get("type", "unknown"),
        }
    except Exception as e:
        return {"error": str(e)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ensemble-dir",
        help="specific ensemble directory to validate (default: scan artifacts/paper_fast/)",
    )
    ap.add_argument(
        "--out",
        default="artifacts/validation",
        help="output directory for validation results",
    )
    ap.add_argument(
        "--method",
        help="specific method name to test (default: test all)",
    )
    args = ap.parse_args(argv)

    # Find ensembles to validate
    if args.ensemble_dir:
        ensemble_dirs = [Path(args.ensemble_dir)]
    else:
        ensemble_dirs = find_ensemble_dirs(Path("artifacts/paper_fast"))

    if not ensemble_dirs:
        print("ERROR: No ensemble directories found!")
        return 1

    print(f"Found {len(ensemble_dirs)} ensemble(s) to validate:")
    for ens_dir in ensemble_dirs:
        info = get_ensemble_info(ens_dir)
        print(f"  - {info['name']}: K={info['k']}, vocab={info['vocab_size']}, members={info['num_members']}")

    # Filter methods if specific method requested
    methods_to_test = METHODS
    if args.method:
        methods_to_test = [m for m in METHODS if m["name"] == args.method]
        if not methods_to_test:
            print(f"ERROR: Method '{args.method}' not found!")
            print(f"Available methods: {', '.join(m['name'] for m in METHODS)}")
            return 1

    print(f"\nTesting {len(methods_to_test)} method(s):")
    for method_cfg in methods_to_test:
        print(f"  - {method_cfg['name']}: {method_cfg['description']}")

    # Run validation
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(ensemble_dirs) * len(methods_to_test)
    completed = 0

    for ens_dir in ensemble_dirs:
        ens_info = get_ensemble_info(ens_dir)

        for method_cfg in methods_to_test:
            completed += 1
            print(f"\n[{completed}/{total}] Testing {method_cfg['name']} on {ens_info['name']}...")

            # Run merge
            merge_dir = run_merge(ens_dir, method_cfg, out_dir)

            if merge_dir:
                # Get tokenizer stats
                tokenizer_path = merge_dir / "tokenizer.json"
                tok_stats = get_tokenizer_stats(tokenizer_path)

                result = {
                    "ensemble": ens_info,
                    "method": method_cfg,
                    "merge_output": str(merge_dir),
                    "tokenizer_stats": tok_stats,
                    "status": "success",
                }
            else:
                result = {
                    "ensemble": ens_info,
                    "method": method_cfg,
                    "status": "failed",
                }

            results.append(result)

    # Save summary
    summary_path = out_dir / "validation_summary.json"
    summary = {
        "ensembles_tested": len(ensemble_dirs),
        "methods_tested": len(methods_to_test),
        "total_runs": total,
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Print summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Ensembles tested: {summary['ensembles_tested']}")
    print(f"Methods tested: {summary['methods_tested']}")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"\nSummary saved to: {summary_path}")

    if summary["failed"] > 0:
        print("\nWARNING: Some validations failed!")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
