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


def build_merge_weighted_bpe_json(
    tokenizer_json_paths: List[Path],
    *,
    weights: Optional[List[float]] = None,
    k: Optional[int] = None,
    theta: Optional[float] = None,
    base_idx: int = 0,
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
        w = max(0.0, float(weights[ti]))
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
