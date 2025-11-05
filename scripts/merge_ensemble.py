"""
Merge-voting ensemble for BPE tokenizers.

Given member tokenizers (JSON) trained on shards, construct a merged BPE
tokenizer whose merges are the k-of-n voted merges across members and whose
vocab is the union (preserving base member ids and appending new tokens).

Examples:
  Using an ensemble run dir (from scripts.train_ensemble):
    uv run --with tokenizers python -m scripts.merge_ensemble \
      --exp hf_bpe_merge --ensemble-run artifacts/<ens_run_dir>

  Using explicit members:
    uv run --with tokenizers python -m scripts.merge_ensemble \
      --exp hf_bpe_merge --members artifacts/m1/tokenizer.json,artifacts/m2/tokenizer.json

  Weighting by compression quality (recommended):
    uv run --with tokenizers python -m scripts.merge_ensemble \
      --exp hf_bpe_quality_merge --ensemble-run artifacts/<ens_run_dir> \
      --weight-by quality --theta 0.5

  Sequential voting (step-by-step):
    uv run --with tokenizers python -m scripts.merge_ensemble \
      --exp hf_bpe_sequential --ensemble-run artifacts/<ens_run_dir> \
      --method sequential --weight-by quality

  Exponential weighting (amplified):
    uv run --with tokenizers python -m scripts.merge_ensemble \
      --exp hf_bpe_exponential --ensemble-run artifacts/<ens_run_dir> \
      --method exponential --weight-by quality --theta 0.3 --power 2.0

Merge Methods:
  - weighted: Traditional parallel voting (default)
  - sequential: Step-by-step voting preserving merge order
  - exponential: Amplified weighting using power transform

Weight Schemes:
  - none: Unweighted k-of-n voting
  - quality: Weight by inverse compression (1/tokens_per_byte) - better models have more influence
  - quality_norm: Normalized quality weights (best model = 1.0)
  - entropy: Weight by shard byte-level entropy
  - char_entropy: Weight by shard character-level entropy
  - bytes, lines, chars: Weight by shard size metrics
  - uniq_bytes, uniq_chars: Weight by shard uniqueness
  - custom: User-provided weights via --weights
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from ebpe.artifact_store import create_run_dir, default_artifact_paths, write_json, write_text
from ebpe.ensemble import build_merge_voted_bpe_json, build_merge_weighted_bpe_json, build_sequential_voted_bpe_json


def _ensure_src_on_path() -> None:
    import sys

    root = Path(__file__).resolve().parents[1] / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _member_paths_from_ensemble(ens_dir: Path) -> List[Path]:
    j = json.loads((ens_dir / "ensemble.json").read_text(encoding="utf-8"))
    toks: List[Path] = []
    for m in j.get("members", []):
        run = Path(m["train_run"])  # type: ignore
        toks.append(run / "tokenizer.json")
    if not toks:
        raise SystemExit("No members found in ensemble.json")
    return toks


def main(argv: list[str] | None = None) -> int:
    _ensure_src_on_path()
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--members", help="comma-separated paths to member tokenizer.json files")
    ap.add_argument("--ensemble-run", help="path to an ensemble run dir (containing ensemble.json)")
    ap.add_argument("--k", type=int, help="k for k-of-n voting; default ceil(n/2)")
    ap.add_argument("--theta", type=float, help="weighted support threshold in [0,1]; select merges with support >= theta * total_weight")
    ap.add_argument("--method", choices=["weighted", "sequential", "exponential"], default="weighted",
                    help="merge method: weighted (traditional), sequential (step-by-step), exponential (amplified weights)")
    ap.add_argument("--power", type=float, default=1.0,
                    help="power for exponential weighting (only with --method=exponential)")
    ap.add_argument("--weight-by", choices=["none", "bytes", "lines", "entropy", "uniq_bytes", "chars", "uniq_chars", "char_entropy", "quality", "quality_norm", "custom"], default="none",
                    help="how to weight members when voting (requires shard_stats in ensemble.json unless 'custom' or 'quality*')")
    ap.add_argument("--weights", help="comma-separated floats for --weight-by=custom")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args(argv)

    members: List[Path]
    ens_stats: Optional[list[dict]] = None
    ens_members: Optional[list[dict]] = None
    if args.ensemble_run:
        ens_dir = Path(args.ensemble_run)
        members = _member_paths_from_ensemble(ens_dir)
        # Try to read shard_stats and full member data from ensemble.json for weighting
        try:
            j = json.loads((ens_dir / "ensemble.json").read_text(encoding="utf-8"))
            ens_members = j.get("members", [])
            ens_stats = [m.get("shard_stats", {}) for m in ens_members]
        except Exception:
            ens_stats = None
            ens_members = None
    elif args.members:
        members = [Path(p.strip()) for p in args.members.split(",") if p.strip()]
    else:
        raise SystemExit("Provide --ensemble-run or --members")

    # Build weights if requested
    weights: Optional[List[float]] = None
    if args.weight_by != "none":
        if args.weight_by == "custom":
            if not args.weights:
                raise SystemExit("--weights required for --weight-by=custom")
            weights = [float(x) for x in args.weights.split(",") if x.strip()]
            if len(weights) != len(members):
                raise SystemExit("--weights length must match number of members")
        elif args.weight_by in ["quality", "quality_norm"]:
            # Weight by compression quality (inverse of tokens_per_byte)
            if ens_members is None or len(ens_members) != len(members):
                raise SystemExit(f"--weight-by={args.weight_by} requires ensemble.json with member tokens_per_byte")

            # Extract tokens_per_byte for each member
            raw_vals = [float(m.get("tokens_per_byte", 1.0)) for m in ens_members]

            if args.weight_by == "quality":
                # Inverse: lower compression = higher quality = higher weight
                # Use 1/tpb so that better compressors get higher weight
                weights = [1.0 / max(v, 0.001) for v in raw_vals]
            else:  # quality_norm
                # Max-normalized: best model gets weight 1.0
                # Divide max by each value so best gets 1.0, others get <1.0
                max_val = max(raw_vals) if raw_vals else 1.0
                weights = [max_val / max(v, 0.001) for v in raw_vals]
        else:
            # Shard-stats based weighting (existing schemes)
            if ens_stats is None or len(ens_stats) != len(members):
                raise SystemExit("weighting requested but ensemble.json lacks shard_stats")
            key_map = {
                "bytes": "bytes",
                "lines": "lines",
                "entropy": "entropy_bits",
                "uniq_bytes": "uniq_bytes",
                "chars": "chars",
                "uniq_chars": "uniq_chars",
                "char_entropy": "char_entropy_bits",
            }
            key = key_map[args.weight_by]
            weights = [float(s.get(key, 0.0)) for s in ens_stats]

    theta = args.theta if args.theta is not None else None

    # Select merge method based on --method argument
    if args.method == "sequential":
        # Sequential voting: step-by-step weighted voting
        merged_json = build_sequential_voted_bpe_json(members, weights=weights, base_idx=0)
        k_used = None
    elif args.method == "exponential":
        # Exponential weighting: amplify weights by power
        merged_json = build_merge_weighted_bpe_json(members, weights=weights, theta=theta, power=args.power, base_idx=0)
        k_used = args.k if args.k else None
    else:  # weighted (traditional)
        if weights is not None or theta is not None:
            merged_json = build_merge_weighted_bpe_json(members, weights=weights, k=args.k, theta=theta, base_idx=0)
            k_used = args.k if args.k else None
        else:
            k_used = args.k if args.k else max(1, (len(members) + 1) // 2)
            merged_json = build_merge_voted_bpe_json(members, k=int(k_used), base_idx=0)

    run_dir = create_run_dir(args.exp, base_dir=args.out)
    paths = default_artifact_paths(run_dir)
    # Save merged tokenizer.json and record env/logs
    Path(paths["tokenizer"]).write_text(json.dumps(merged_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta = {"members": [str(p) for p in members], "method": args.method}
    if weights is not None:
        meta["weights"] = weights
    if theta is not None:
        meta["theta"] = theta
    if k_used is not None:
        meta["k"] = k_used
    if args.method == "exponential":
        meta["power"] = args.power
    write_text(paths["env"], f"merge_ensemble members={len(members)} method={args.method} k={k_used} theta={theta} power={args.power if args.method == 'exponential' else 'N/A'} weight_by={args.weight_by}\n")
    write_json(run_dir / "merge_meta.json", meta)
    write_text(paths["logs"], "wrote merged tokenizer\n")
    print(str(run_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
