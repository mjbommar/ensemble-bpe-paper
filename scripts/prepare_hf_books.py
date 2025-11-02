"""
Prepare text splits from Hugging Face datasets (e.g., Project Gutenberg, PG19)
or from local text files, writing train/valid/test to an output directory.

This complements scripts.prepare_data (dictionary-based) with book/corpus text.

Examples (HF):
  uv run --with datasets python -m scripts.prepare_hf_books \
      --dataset pg19 --split train --text-field text \
      --sample-train 5000 --sample-valid 500 --sample-test 500 \
      --out data/processed/hf_pg19

Examples (local fallback used in tests, no network):
  uv run python -m scripts.prepare_hf_books \
      --local-files path/to/texts --out data/processed/local_books

Outputs:
  - train.txt, valid.txt, test.txt
  - manifest.json capturing dataset id/config/revision or local paths
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple
import hashlib


def _iter_files(p: Path) -> Iterable[Path]:
    if p.is_file():
        yield p
    elif p.is_dir():
        for child in sorted(p.iterdir()):
            if child.is_file():
                yield child
    else:
        raise FileNotFoundError(str(p))


def _read_local_texts_lines(files: Iterable[Path]) -> List[str]:
    docs: List[str] = []
    for f in files:
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not txt:
            continue
        # Split by lines; treat each line as a document for simplicity and speed
        docs.extend([ln for ln in txt.splitlines() if ln.strip()])
    return docs


def _split_local_files_by_file(
    local_path: Path, rng: random.Random, oos_frac: float | None, oos_count: int | None
) -> Tuple[List[Path], List[Path]]:
    files = list(_iter_files(local_path))
    if not files:
        return [], []
    # Deterministic shuffle, then take head for OOS
    files = sorted(files)
    rng.shuffle(files)
    k = 0
    if oos_count is not None and oos_count > 0:
        k = min(oos_count, len(files))
    elif oos_frac is not None and oos_frac > 0:
        k = min(len(files), max(1, int(round(oos_frac * len(files)))))
    oos_files = files[:k]
    is_files = files[k:]
    return is_files, oos_files


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hash_u32(s: str, seed: int) -> int:
    h = hashlib.sha1()
    h.update(seed.to_bytes(4, "little"))
    h.update(s.encode("utf-8", errors="ignore"))
    return int.from_bytes(h.digest()[:4], "little")


def _oos_by_hash(docs: List[str], seed: int, oos_frac: float | None, oos_count: int | None) -> Tuple[List[str], List[str]]:
    if not docs:
        return [], []
    keyed = [(doc, _hash_u32(doc, seed)) for doc in docs]
    keyed.sort(key=lambda x: x[1])
    n = len(keyed)
    k = 0
    if oos_count is not None and oos_count > 0:
        k = min(oos_count, n)
    elif oos_frac is not None and oos_frac > 0:
        k = min(n, max(1, int(round(oos_frac * n))))
    oos = [d for d, _ in keyed[:k]]
    is_docs = [d for d, _ in keyed[k:]]
    return is_docs, oos


def _prepare_from_hf(
    dataset: str,
    config: str | None,
    split: str,
    text_field: str,
    sample_train: int,
    sample_valid: int,
    sample_test: int,
    seed: int,
    revision: str | None,
    streaming: bool,
) -> tuple[List[str], List[str], List[str], dict]:
    import datasets  # type: ignore

    if streaming:
        # Stream without downloading the full dataset; take head slices
        ds_iter = datasets.load_dataset(
            dataset,
            config,
            split=split,
            revision=revision,
            streaming=True,
            trust_remote_code=True,
        )
        def take_n(n: int) -> List[str]:
            out: List[str] = []
            if n <= 0:
                return out
            for ex in ds_iter:  # type: ignore[assignment]
                if text_field in ex:
                    s = str(ex[text_field]).strip()
                    if s:
                        out.append(s)
                        if len(out) >= n:
                            break
            return out
        train_docs = take_n(sample_train)
        valid_docs = take_n(sample_valid)
        test_docs = take_n(sample_test)
        total = None
        resolved_rev = None
    else:
        ds = datasets.load_dataset(
            dataset,
            config,
            split=split,
            revision=revision,
            trust_remote_code=True,
        )
        # Ensure we have the text field
        if text_field not in ds.features:
            raise SystemExit(f"text field '{text_field}' not in dataset features: {list(ds.features)}")
        # Shuffle deterministically and slice
        ds = ds.shuffle(seed=seed)
        total = len(ds)
        t = min(sample_train, total)
        v = min(sample_valid, max(0, total - t))
        rem = max(0, total - t - v)
        te = min(sample_test, rem)

        def take_text(start: int, n: int) -> List[str]:
            if n <= 0:
                return []
            subset = ds.select(range(start, start + n))
            return [str(x[text_field]).strip() for x in subset if str(x[text_field]).strip()]

        train_docs = take_text(0, t)
        valid_docs = take_text(t, v)
        test_docs = take_text(t + v, te)

        # Try to capture the resolved dataset revision (commit hash) if available
        resolved_rev = None
        try:
            info = getattr(ds, "info", None)
            resolved_rev = getattr(info, "dataset_revision", None)
        except Exception:
            resolved_rev = None

    meta = {
        "provider": "huggingface",
        "dataset": dataset,
        "config": config,
        "split": split,
        "revision": revision,
        "resolved_revision": resolved_rev,
        "text_field": text_field,
        "total": total,
        "streaming": bool(streaming),
    }
    return train_docs, valid_docs, test_docs, meta


def _prepare_from_local(
    local_path: Path,
    sample_train: int,
    sample_valid: int,
    sample_test: int,
    seed: int,
    split_strategy: str,
    oos_frac: float | None,
    oos_count: int | None,
    oos_local_path: Path | None,
) -> tuple[List[str], List[str], List[str], List[str], dict]:
    rng = random.Random(seed)
    # Decide which files feed IS vs OOS
    if split_strategy == "by_file":
        is_files, oos_files = _split_local_files_by_file(local_path, rng, oos_frac, oos_count)
        if oos_local_path is not None:
            # If a separate OOS directory is provided, take all files from there as OOS instead
            oos_files = list(_iter_files(oos_local_path))
        is_docs = _read_local_texts_lines(is_files)
        oos_docs = _read_local_texts_lines(oos_files)
    else:
        # Default: by_line (previous behavior), no implicit OOS
        is_docs = _read_local_texts_lines(_iter_files(local_path))
        oos_docs = []
        if oos_local_path is not None:
            oos_docs = _read_local_texts_lines(_iter_files(oos_local_path))

    rng.shuffle(is_docs)
    t = min(sample_train, len(is_docs))
    v = min(sample_valid, max(0, len(is_docs) - t))
    rem = max(0, len(is_docs) - t - v)
    te = min(sample_test, rem)
    train_docs = is_docs[:t]
    valid_docs = is_docs[t : t + v]
    test_docs = is_docs[t + v : t + v + te]
    meta = {
        "provider": "local",
        "path": str(local_path.resolve()),
        "total": len(is_docs),
        "split_strategy": split_strategy,
        "oos_from": str(oos_local_path.resolve()) if oos_local_path is not None else ("by_file" if split_strategy == "by_file" and (oos_frac or oos_count) else None),
        "oos_count": len(oos_docs),
    }
    return train_docs, valid_docs, test_docs, oos_docs, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="HF dataset name (e.g., pg19, project_gutenberg)")
    ap.add_argument("--config", help="HF dataset config/subset (optional)")
    ap.add_argument("--split", default="train", help="HF split to read (default: train)")
    ap.add_argument("--revision", help="HF dataset git revision/hash (optional)")
    ap.add_argument("--text-field", default="text", help="column containing text (default: text)")
    ap.add_argument("--local-files", help="fallback local text file/dir (no network)")
    ap.add_argument("--oos-local-files", help="optional OOS text file/dir (local mode only)")
    ap.add_argument(
        "--split-strategy",
        choices=["by_line", "by_file"],
        default="by_line",
        help="local mode only: treat each line as a doc (by_line) or allocate entire files to IS/OOS (by_file)",
    )
    ap.add_argument("--oos-frac", type=float, default=0.0, help="fraction of items to reserve for OOS (by_file for local; by_hash for HF/by_line)")
    ap.add_argument("--oos-count", type=int, default=0, help="number of items to reserve for OOS (by_file for local; by_hash for HF/by_line)")
    ap.add_argument("--oos-by-hash", action="store_true", help="deterministic OOS via hashing text; works for HF and local by_line")
    ap.add_argument("--out", dest="out_dir", default="data/processed/hf_books", help="output dir")
    ap.add_argument("--sample-train", type=int, default=50000)
    ap.add_argument("--sample-valid", type=int, default=5000)
    ap.add_argument("--sample-test", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--streaming", action="store_true", help="use HF streaming to avoid full downloads; takes head samples only")
    args = ap.parse_args(argv)

    if not args.dataset and not args.local_files:
        raise SystemExit("Provide --dataset (uses Hugging Face) or --local-files for offline mode")

    if args.dataset:
        train_docs, valid_docs, test_docs, meta = _prepare_from_hf(
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            text_field=args.text_field,
            sample_train=args.sample_train,
            sample_valid=args.sample_valid,
            sample_test=args.sample_test,
            seed=args.seed,
            revision=args.revision,
            streaming=bool(args.streaming),
        )
    else:
        oos_local_path = Path(args.oos_local_files) if args.oos_local_files else None
        train_docs, valid_docs, test_docs, oos_docs, meta = _prepare_from_local(
            local_path=Path(args.local_files),
            sample_train=args.sample_train,
            sample_valid=args.sample_valid,
            sample_test=args.sample_test,
            seed=args.seed,
            split_strategy=args.split_strategy,
            oos_frac=(args.oos_frac if args.oos_frac > 0 else None),
            oos_count=(args.oos_count if args.oos_count > 0 else None),
            oos_local_path=oos_local_path,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_lines(out_dir / "train.txt", train_docs)
    _write_lines(out_dir / "valid.txt", valid_docs)
    _write_lines(out_dir / "test.txt", test_docs)
    # Optional OOS set
    oos_written = False
    # Case A: local by_file already produced OOS docs
    if 'oos_count' in meta and meta.get("oos_count") and not (out_dir / "test_oos.txt").exists():
        try:
            oos_docs  # type: ignore[name-defined]
        except NameError:
            oos_docs = []
        if oos_docs:
            _write_lines(out_dir / "test_oos.txt", oos_docs)
            oos_written = True
    # Case B: by-hash selection for HF or local by_line when requested or when oos_* provided with HF
    if (args.oos_by_hash or (args.dataset and (args.oos_frac > 0 or args.oos_count > 0))) and not oos_written:
        used_pool = "test" if test_docs else ("valid" if valid_docs else "train")
        pool = list(test_docs) if test_docs else (list(valid_docs) if valid_docs else list(train_docs))
        is_docs, oos_h = _oos_by_hash(
            pool,
            args.seed,
            (args.oos_frac if args.oos_frac > 0 else None),
            (args.oos_count if args.oos_count > 0 else None),
        )
        if oos_h:
            _write_lines(out_dir / "test_oos.txt", oos_h)
            # If we split from test/valid/train, rewrite that file to IS remainder to avoid leakage
            if used_pool == "test":
                test_docs = is_docs
                _write_lines(out_dir / "test.txt", test_docs)
            elif used_pool == "valid":
                valid_docs = is_docs
                _write_lines(out_dir / "valid.txt", valid_docs)
            else:
                train_docs = is_docs
                _write_lines(out_dir / "train.txt", train_docs)
            oos_written = True

    manifest = {
        "seed": args.seed,
        "source": meta,
        "counts": {
            "train": len(train_docs),
            "valid": len(valid_docs),
            "test": len(test_docs),
            "test_oos": (
                len((out_dir / "test_oos.txt").read_text(encoding="utf-8").splitlines())
                if (out_dir / "test_oos.txt").exists()
                else int(meta.get("oos_count", 0))
            ),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Print a machine-friendly line for callers
    print(json.dumps({"out_dir": str(out_dir.resolve()), **manifest}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
