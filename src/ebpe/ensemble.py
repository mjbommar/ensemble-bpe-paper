from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _load_json(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_merges(model: Dict) -> List[Tuple[str, str]]:
    merges = model.get("merges") or []
    out: List[Tuple[str, str]] = []
    for m in merges:
        if isinstance(m, list) and len(m) == 2:
            out.append((m[0], m[1]))
        elif isinstance(m, str) and " " in m:
            a, b = m.split(" ", 1)
            out.append((a, b))
    return out


def build_merge_voted_bpe_json(tokenizer_json_paths: List[Path], k: int = 2, base_idx: int = 0) -> Dict:
    if not tokenizer_json_paths:
        raise ValueError("No tokenizer paths provided")
    n = len(tokenizer_json_paths)
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, {n}]")

    # Load all models
    toks = [_load_json(Path(p)) for p in tokenizer_json_paths]
    base = toks[base_idx]
    base_model = base.get("model", {})
    if base_model.get("type") != "BPE":
        raise ValueError("base model must be BPE")

    # Collect votes for merges and union of vocabs
    vote: Dict[Tuple[str, str], int] = {}
    rank_sum: Dict[Tuple[str, str], float] = {}
    union_vocab_tokens: Dict[str, int] = dict(base_model.get("vocab", {}))

    for t in toks:
        model = t.get("model", {})
        if model.get("type") != "BPE":
            continue
        merges = _extract_merges(model)
        for idx, m in enumerate(merges):
            vote[m] = vote.get(m, 0) + 1
            rank_sum[m] = rank_sum.get(m, 0.0) + float(idx)
        for tok, _id in model.get("vocab", {}).items():
            if tok not in union_vocab_tokens:
                union_vocab_tokens[tok] = -1  # placeholder, we'll assign ids later

    # Select merges by k-of-n, sort by (votes desc, avg rank asc)
    selected = [m for m, c in vote.items() if c >= k]
    selected.sort(key=lambda m: (-vote[m], rank_sum[m] / vote[m]))

    # Build new vocab mapping: keep base ids, append new tokens sequentially
    new_vocab: Dict[str, int] = dict(base_model.get("vocab", {}))
    next_id = (max(new_vocab.values()) + 1) if new_vocab else 0
    for tok in sorted(union_vocab_tokens.keys()):
        if tok not in new_vocab:
            new_vocab[tok] = next_id
            next_id += 1

    # Construct merged tokenizer JSON using base as template
    merged = json.loads(json.dumps(base))  # deep copy
    merged_model = merged.setdefault("model", {})
    merged_model["vocab"] = new_vocab
    merged_model["merges"] = [[a, b] for (a, b) in selected]
    return merged


def build_sequential_voted_bpe_json(
    tokenizer_json_paths: List[Path],
    weights: Optional[List[float]] = None,
    max_merges: int = 16384,
    base_idx: int = 0,
) -> Dict:
    """Build tokenizer by sequential step-by-step weighted voting.

    At each step, vote on what merge to do NEXT based on what each member
    does at that step in THEIR sequence. Preserves sequential structure.

    Args:
        tokenizer_json_paths: List of paths to member tokenizer JSONs
        weights: Optional weights for each member (defaults to equal weights)
        max_merges: Maximum number of merge steps to perform
        base_idx: Index of base tokenizer to use as template

    Returns:
        Merged tokenizer JSON dictionary
    """
    if not tokenizer_json_paths:
        raise ValueError("No tokenizer paths provided")
    n = len(tokenizer_json_paths)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError("weights length must equal number of tokenizers")

    # Load all models
    toks = [_load_json(Path(p)) for p in tokenizer_json_paths]
    base = toks[base_idx]
    base_model = base.get("model", {})
    if base_model.get("type") != "BPE":
        raise ValueError("base model must be BPE")

    # Extract all member merge sequences
    member_merges = []
    for t in toks:
        model = t.get("model", {})
        if model.get("type") == "BPE":
            member_merges.append(_extract_merges(model))
        else:
            member_merges.append([])

    # Build union vocab
    union_vocab_tokens: Dict[str, int] = dict(base_model.get("vocab", {}))
    for t in toks:
        model = t.get("model", {})
        if model.get("type") == "BPE":
            for tok in model.get("vocab", {}).keys():
                if tok not in union_vocab_tokens:
                    union_vocab_tokens[tok] = -1

    # Sequential voting: vote on next merge, advancing each member's pointer
    # when their merge is selected
    result_merges: List[Tuple[str, str]] = []
    used_merges: set[Tuple[str, str]] = set()
    member_positions = [0] * len(member_merges)  # Track position in each member

    for _output_step in range(max_merges):
        votes: Dict[Tuple[str, str], float] = {}

        # Each member votes for their NEXT unused merge
        for member_idx, merges in enumerate(member_merges):
            pos = member_positions[member_idx]
            # Skip ahead to find next unused merge for this member
            while pos < len(merges) and merges[pos] in used_merges:
                pos += 1

            if pos < len(merges):
                merge = merges[pos]
                votes[merge] = votes.get(merge, 0.0) + weights[member_idx]

        # If no votes, we're done (all members exhausted or all merges used)
        if not votes:
            break

        # Pick merge with highest vote
        best_merge = max(votes.items(), key=lambda x: x[1])[0]
        result_merges.append(best_merge)
        used_merges.add(best_merge)

        # Advance positions for members who had this merge next
        for member_idx, merges in enumerate(member_merges):
            pos = member_positions[member_idx]
            if pos < len(merges) and merges[pos] == best_merge:
                member_positions[member_idx] = pos + 1

    # Build final vocab
    new_vocab: Dict[str, int] = dict(base_model.get("vocab", {}))
    next_id = (max(new_vocab.values()) + 1) if new_vocab else 0
    for tok in sorted(union_vocab_tokens.keys()):
        if tok not in new_vocab:
            new_vocab[tok] = next_id
            next_id += 1

    # Construct result
    merged = json.loads(json.dumps(base))
    merged_model = merged.setdefault("model", {})
    merged_model["vocab"] = new_vocab
    merged_model["merges"] = [[a, b] for (a, b) in result_merges]
    return merged


def build_backbone_voted_bpe_json(
    tokenizer_json_paths: List[Path],
    weights: Optional[List[float]] = None,
    override_threshold: float = 2.0,
    base_idx: int = 0,
) -> Dict:
    """Start with best member's sequence, only override with strong consensus.

    For each step in best member's sequence:
        - Check what other members do at this step
        - Only override if alternative has override_threshold * best_weight support

    Args:
        tokenizer_json_paths: List of paths to member tokenizer JSONs
        weights: Optional weights for each member (defaults to equal weights)
        override_threshold: Multiplier for best member weight to override
        base_idx: Index of base/best tokenizer (typically 0)

    Returns:
        Merged tokenizer JSON dictionary
    """
    if not tokenizer_json_paths:
        raise ValueError("No tokenizer paths provided")
    n = len(tokenizer_json_paths)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError("weights length must equal number of tokenizers")

    # Load all models
    toks = [_load_json(Path(p)) for p in tokenizer_json_paths]
    base = toks[base_idx]
    base_model = base.get("model", {})
    if base_model.get("type") != "BPE":
        raise ValueError("base model must be BPE")

    # Extract all member merge sequences
    member_merges = []
    for t in toks:
        model = t.get("model", {})
        if model.get("type") == "BPE":
            member_merges.append(_extract_merges(model))
        else:
            member_merges.append([])

    # Build union vocab
    union_vocab_tokens: Dict[str, int] = dict(base_model.get("vocab", {}))
    for t in toks:
        model = t.get("model", {})
        if model.get("type") == "BPE":
            for tok in model.get("vocab", {}).keys():
                if tok not in union_vocab_tokens:
                    union_vocab_tokens[tok] = -1

    # Start with best member's sequence
    best_merges = member_merges[base_idx]
    result_merges: List[Tuple[str, str]] = []
    best_weight = weights[base_idx]
    threshold = override_threshold * best_weight

    for step, backbone_merge in enumerate(best_merges):
        votes: Dict[Tuple[str, str], float] = {}

        # Collect votes from all members at this step
        for member_idx, merges in enumerate(member_merges):
            if step < len(merges):
                merge = merges[step]
                votes[merge] = votes.get(merge, 0.0) + weights[member_idx]

        # Check if any alternative beats threshold
        if votes:
            best_vote_merge = max(votes.items(), key=lambda x: x[1])
            if best_vote_merge[0] != backbone_merge and best_vote_merge[1] >= threshold:
                # Override with strong consensus
                result_merges.append(best_vote_merge[0])
            else:
                # Keep backbone merge
                result_merges.append(backbone_merge)
        else:
            result_merges.append(backbone_merge)

    # Build final vocab
    new_vocab: Dict[str, int] = dict(base_model.get("vocab", {}))
    next_id = (max(new_vocab.values()) + 1) if new_vocab else 0
    for tok in sorted(union_vocab_tokens.keys()):
        if tok not in new_vocab:
            new_vocab[tok] = next_id
            next_id += 1

    # Construct result
    merged = json.loads(json.dumps(base))
    merged_model = merged.setdefault("model", {})
    merged_model["vocab"] = new_vocab
    merged_model["merges"] = [[a, b] for (a, b) in result_merges]
    return merged


def build_merge_weighted_bpe_json(
    tokenizer_json_paths: List[Path],
    *,
    weights: Optional[List[float]] = None,
    k: Optional[int] = None,
    theta: Optional[float] = None,
    base_idx: int = 0,
    power: float = 1.0,
    position_primary: bool = False,
) -> Dict:
    """Weighted merge construction for BPE ensembles.

    - If `theta` is provided (0..1), select merges whose supporting weight sum >= theta * total_weight.
    - Else if `k` is provided, select merges that appear in at least k members (unweighted),
      but ranking still uses weighted average rank when `weights` are given.
    - If neither is provided, defaults to simple majority (`k = ceil(n/2)`).
    """
    if not tokenizer_json_paths:
        raise ValueError("No tokenizer paths provided")
    n = len(tokenizer_json_paths)
    if weights is not None and len(weights) != n:
        raise ValueError("weights length must equal number of tokenizers")
    if weights is None:
        weights = [1.0] * n
    total_w = sum(max(0.0, float(w)) for w in weights)
    if total_w <= 0:
        weights = [1.0] * n
        total_w = float(n)
    if theta is None:
        # default to k-of-n (majority if k not provided)
        if k is None:
            from math import ceil

            k = ceil(n / 2)

    toks = [_load_json(Path(p)) for p in tokenizer_json_paths]
    base = toks[base_idx]
    base_model = base.get("model", {})
    if base_model.get("type") != "BPE":
        raise ValueError("base model must be BPE")

    # Collect weighted votes and weighted rank sums
    vote_w: Dict[Tuple[str, str], float] = {}
    rank_wsum: Dict[Tuple[str, str], float] = {}
    rank_wtot: Dict[Tuple[str, str], float] = {}
    union_vocab_tokens: Dict[str, int] = dict(base_model.get("vocab", {}))

    for ti, t in enumerate(toks):
        model = t.get("model", {})
        if model.get("type") != "BPE":
            continue
        merges = _extract_merges(model)
        w = max(0.0, float(weights[ti])) ** power
        for idx, m in enumerate(merges):
            vote_w[m] = vote_w.get(m, 0.0) + w
            rank_wsum[m] = rank_wsum.get(m, 0.0) + (w * float(idx))
            rank_wtot[m] = rank_wtot.get(m, 0.0) + w
        for tok, _id in model.get("vocab", {}).items():
            if tok not in union_vocab_tokens:
                union_vocab_tokens[tok] = -1

    # Selection condition
    selected: List[Tuple[str, str]]
    if theta is not None:
        thr = float(theta) * total_w
        selected = [m for m, vw in vote_w.items() if vw >= thr]
    else:
        # fall back to unweighted k-of-n using presence count
        present: Dict[Tuple[str, str], int] = {}
        for ti, t in enumerate(toks):
            model = t.get("model", {})
            if model.get("type") != "BPE":
                continue
            merges = _extract_merges(model)
            for m in merges:
                present[m] = present.get(m, 0) + 1
        kk = int(k) if k is not None else 1
        selected = [m for m, c in present.items() if c >= kk]

    # Sort selected merges: by weighted support desc, then weighted avg rank asc
    def _avg_rank(m: Tuple[str, str]) -> float:
        if rank_wtot.get(m, 0.0) <= 0:
            return 1e12
        return rank_wsum[m] / rank_wtot[m]

    if position_primary:
        selected.sort(key=lambda m: (_avg_rank(m), -vote_w.get(m, 0.0)))
    else:
        selected.sort(key=lambda m: (-vote_w.get(m, 0.0), _avg_rank(m)))

    # Build vocab union
    new_vocab: Dict[str, int] = dict(base_model.get("vocab", {}))
    next_id = (max(new_vocab.values()) + 1) if new_vocab else 0
    for tok in sorted(union_vocab_tokens.keys()):
        if tok not in new_vocab:
            new_vocab[tok] = next_id
            next_id += 1

    merged = json.loads(json.dumps(base))
    merged_model = merged.setdefault("model", {})
    merged_model["vocab"] = new_vocab
    merged_model["merges"] = [[a, b] for (a, b) in selected]
    return merged
