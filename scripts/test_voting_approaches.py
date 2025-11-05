"""
Test different voting approaches for BPE ensemble tokenizers.

This script tests 4 new ensemble voting approaches:
1. Exponential Quality Weighting (power parameter)
2. Sequential Voting
3. Best-Member Backbone
4. Position-Primary Ranking
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src is on path
root = Path(__file__).resolve().parents[1] / "src"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from ebpe.ensemble import (
    build_merge_weighted_bpe_json,
    build_sequential_voted_bpe_json,
    build_backbone_voted_bpe_json,
)


def get_ensemble_info(ensemble_dir: Path):
    """Extract member paths and quality weights from ensemble.json"""
    ens_json = json.loads((ensemble_dir / "ensemble.json").read_text(encoding="utf-8"))

    members = []
    tpb_values = []

    for m in ens_json.get("members", []):
        train_run = Path(m["train_run"])
        members.append(train_run / "tokenizer.json")
        tpb_values.append(float(m["tokens_per_byte"]))

    # Quality weights: inverse of tokens_per_byte (lower is better)
    quality_weights = [1.0 / tpb for tpb in tpb_values]

    return members, quality_weights


def save_tokenizer(tok_json: dict, output_path: Path):
    """Save tokenizer JSON to file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(tok_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )


def main():
    # Configuration
    ensemble_dir = Path("artifacts/paper_fast/20251103_083104_ens_fast_s13_v16384_k4")
    output_base = Path("artifacts/voting_experiments")

    # Get ensemble info
    print("Loading ensemble information...")
    members, quality_weights = get_ensemble_info(ensemble_dir)
    print(f"Found {len(members)} members")
    print(f"Quality weights: {quality_weights}")

    # Calculate weight ratios for verification
    max_weight = max(quality_weights)
    normalized = [w / max_weight for w in quality_weights]
    print(f"Weight ratios (best=1.0): {normalized}")

    configs = []

    # Approach 1: Exponential Quality Weighting
    print("\n=== Approach 1: Exponential Quality Weighting ===")
    for power in [2.0, 3.0]:
        print(f"Building with power={power}...")
        tok = build_merge_weighted_bpe_json(
            members,
            weights=quality_weights,
            theta=0.3,
            power=power,
        )
        output_path = output_base / f"exp_power_{power:.1f}" / "tokenizer.json"
        save_tokenizer(tok, output_path)
        configs.append({
            "name": f"Exponential p={power}",
            "path": output_path,
            "config": f"quality, theta=0.3, power={power}",
        })
        print(f"  Saved to {output_path}")
        print(f"  Merges: {len(tok['model']['merges'])}")

    # Approach 2: Sequential Voting
    print("\n=== Approach 2: Sequential Voting ===")
    # Use max_merges based on the voted result to match similar size
    tok = build_sequential_voted_bpe_json(
        members,
        weights=quality_weights,
        max_merges=16384,  # Use large number, it will stop when no more votes
    )
    output_path = output_base / "sequential" / "tokenizer.json"
    save_tokenizer(tok, output_path)
    configs.append({
        "name": "Sequential",
        "path": output_path,
        "config": f"quality weights, max_merges={len(tok['model']['merges'])}",
    })
    print(f"Saved to {output_path}")
    print(f"Merges: {len(tok['model']['merges'])}")

    # Approach 3: Best-Member Backbone
    print("\n=== Approach 3: Best-Member Backbone ===")
    for threshold in [1.5, 2.0, 3.0]:
        print(f"Building with override_threshold={threshold}...")
        tok = build_backbone_voted_bpe_json(
            members,
            weights=quality_weights,
            override_threshold=threshold,
        )
        output_path = output_base / f"backbone_thresh_{threshold:.1f}" / "tokenizer.json"
        save_tokenizer(tok, output_path)
        configs.append({
            "name": f"Backbone thresh={threshold}",
            "path": output_path,
            "config": f"override_threshold={threshold}",
        })
        print(f"  Saved to {output_path}")
        print(f"  Merges: {len(tok['model']['merges'])}")

    # Approach 4: Position-Primary Ranking
    print("\n=== Approach 4: Position-Primary Ranking ===")
    tok = build_merge_weighted_bpe_json(
        members,
        weights=quality_weights,
        theta=0.3,
        position_primary=True,
    )
    output_path = output_base / "position_primary" / "tokenizer.json"
    save_tokenizer(tok, output_path)
    configs.append({
        "name": "Position-Primary",
        "path": output_path,
        "config": "quality, theta=0.3, position_primary=True",
    })
    print(f"Saved to {output_path}")
    print(f"Merges: {len(tok['model']['merges'])}")

    # Save configuration summary (convert Path to str)
    summary_path = output_base / "experiment_summary.json"
    configs_serializable = [
        {**c, "path": str(c["path"])} for c in configs
    ]
    summary_path.write_text(
        json.dumps(configs_serializable, indent=2) + "\n",
        encoding="utf-8"
    )

    print(f"\n=== Summary ===")
    print(f"Generated {len(configs)} tokenizer configurations")
    print(f"Summary saved to {summary_path}")
    print("\nNext steps:")
    print("1. Run eval_compression.py on each tokenizer with test_oos.txt")
    print("2. Compare tokens_per_byte results")


if __name__ == "__main__":
    main()
