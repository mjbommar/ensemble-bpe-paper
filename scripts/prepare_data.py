"""
Prepare dictionary-based datasets for tokenizer experiments.

Reads one or more dictionary files (or directories) and writes shuffled,
deduplicated splits to `data/processed` (or a custom output dir):
  - train.txt, valid.txt, test.txt
  - optional: test_oos.txt (from a separate out-of-sample dictionary path)

Examples:
  uv run python -m scripts.prepare_data \
      --dict-path /usr/share/dict \
      --out data/processed \
      --train-size 50000 --valid-size 5000 --test-size 5000

  uv run python -m scripts.prepare_data \
      --dict-path /usr/share/dict \
      --oos-dict-path /some/other/dict \
      --oos-size 5000

Outputs a manifest.json with counts and sources for replication.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable, List, Set


def _iter_files(p: Path) -> Iterable[Path]:
    if p.is_file():
        yield p
        return
    if p.is_dir():
        for child in sorted(p.iterdir()):
            if child.is_file():
                yield child
        return
    raise FileNotFoundError(str(p))


def _read_words(path: Path) -> List[str]:
    words: List[str] = []
    for f in _iter_files(path):
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # Skip unreadable files
            continue
        for line in txt.splitlines():
            s = line.strip()
            if not s:
                continue
            # Filter out obvious non-words like comments or multi-space phrases.
            if s.startswith("#"):
                continue
            if "\t" in s:
                s = s.replace("\t", " ")
            # Keep single tokens (allow hyphens and apostrophes), but avoid spaces
            if " " in s:
                continue
            words.append(s)
    return words


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gen_sequences(
    vocab: List[str],
    n_lines: int,
    seq_min: int,
    seq_max: int,
    punct_prob: float,
    rng: random.Random,
) -> List[str]:
    """Generate synthetic text by sampling runs of words.

    - Chooses a random length L ~ Uniform[seq_min, seq_max] per line.
    - Samples words with replacement from `vocab`.
    - Injects light punctuation between some tokens with probability `punct_prob`.
    """
    if not vocab:
        return []
    punct = [".", ",", ";", ":", "?", "!"]
    out: List[str] = []
    for _ in range(max(0, n_lines)):
        L = rng.randint(max(1, seq_min), max(seq_min, seq_max))
        toks = [rng.choice(vocab) for _ in range(L)]
        pieces: List[str] = []
        for i, t in enumerate(toks):
            pieces.append(t)
            if i < L - 1 and rng.random() < punct_prob:
                pieces.append(rng.choice(punct))
        out.append(" ".join(pieces))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dict-path", required=True, help="path to dictionary file or directory")
    p.add_argument("--oos-dict-path", help="optional out-of-sample dictionary path")
    p.add_argument("--out", dest="out_dir", default="data/processed", help="output dir for splits")
    p.add_argument("--train-size", type=int, default=50000, help="number of lines (sequence mode) or tokens (word mode)")
    p.add_argument("--valid-size", type=int, default=5000, help="number of lines (sequence mode) or tokens (word mode)")
    p.add_argument("--test-size", type=int, default=5000, help="number of lines (sequence mode) or tokens (word mode)")
    p.add_argument("--oos-size", type=int, default=0)
    p.add_argument("--lower", action="store_true", help="lowercase all tokens")
    # Synthetic sequence generation
    p.add_argument("--sequence", action="store_true", help="generate sequences of words instead of one word per line")
    p.add_argument("--seq-min", type=int, default=5, help="min words per sequence (when --sequence)")
    p.add_argument("--seq-max", type=int, default=20, help="max words per sequence (when --sequence)")
    p.add_argument("--punct-prob", type=float, default=0.1, help="probability of punctuation between words (when --sequence)")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)

    in_words = _read_words(Path(args.dict_path))
    if args.lower:
        in_words = [w.lower() for w in in_words]
    in_words = _dedupe_preserve_order(in_words)
    rng.shuffle(in_words)

    if args.sequence:
        # Sequence mode: sizes denote number of lines; sample with replacement
        train = _gen_sequences(in_words, args.train_size, args.seq_min, args.seq_max, args.punct_prob, rng)
        valid = _gen_sequences(in_words, args.valid_size, args.seq_min, args.seq_max, args.punct_prob, rng)
        test = _gen_sequences(in_words, args.test_size, args.seq_min, args.seq_max, args.punct_prob, rng)
    else:
        # Word mode: sizes denote counts of unique words, no replacement
        t_size = min(args.train_size, len(in_words))
        v_size = min(args.valid_size, max(0, len(in_words) - t_size))
        s_rem = max(0, len(in_words) - t_size - v_size)
        te_size = min(args.test_size, s_rem)

        train = in_words[:t_size]
        valid = in_words[t_size : t_size + v_size]
        test = in_words[t_size + v_size : t_size + v_size + te_size]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_lines(out_dir / "train.txt", train)
    write_lines(out_dir / "valid.txt", valid)
    write_lines(out_dir / "test.txt", test)

    oos_count = 0
    if args.oos_dict_path:
        oos_words = _read_words(Path(args.oos_dict_path))
        if args.lower:
            oos_words = [w.lower() for w in oos_words]
        oos_words = _dedupe_preserve_order(oos_words)
        # Prefer non-overlapping words for OOS in word mode
        if not args.sequence:
            in_pool = set(in_words)
            oos_words = [w for w in oos_words if w not in in_pool]
        rng.shuffle(oos_words)
        oos_size = args.oos_size if args.oos_size > 0 else min(5000, len(oos_words))
        if args.sequence:
            oos_lines = _gen_sequences(oos_words, oos_size, args.seq_min, args.seq_max, args.punct_prob, rng)
            write_lines(out_dir / "test_oos.txt", oos_lines)
            oos_count = len(oos_lines)
        else:
            oos = oos_words[:oos_size]
            write_lines(out_dir / "test_oos.txt", oos)
            oos_count = len(oos)

    manifest = {
        "seed": args.seed,
        "lower": bool(args.lower),
        "sources": {
            "in_sample": str(Path(args.dict_path).resolve()),
            "out_of_sample": str(Path(args.oos_dict_path).resolve()) if args.oos_dict_path else None,
        },
        "counts": {
            "in_total": len(in_words),
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
            "test_oos": oos_count,
        },
        "sequence_mode": bool(args.sequence),
        "sequence_params": {
            "seq_min": int(args.seq_min),
            "seq_max": int(args.seq_max),
            "punct_prob": float(args.punct_prob),
        } if args.sequence else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir.resolve()), **manifest}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
