"""
Train a Hugging Face Unigram tokenizer on prepared text files and save artifacts.

Design choices:
- Use Metaspace pre-tokenization/decoding to preserve exact whitespace round-trips.
- Only add [UNK] as a special token to avoid per-line overhead in compression.

Example:
  uv run --with tokenizers --with psutil python -m scripts.train_unigram \
      --exp hf_uni_smoke --train data/processed/train.txt --vocab-size 2000
"""
from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import psutil  # type: ignore

from ebpe.artifact_store import create_run_dir, default_artifact_paths, write_json, write_text, write_yaml_like


def _gather_env_text_fallback() -> str:
    try:
        from scripts.record_env import gather_env_text  # type: ignore

        return gather_env_text()
    except Exception:
        import platform
        import sys

        return f"Python: {sys.version}\nPlatform: {platform.platform()}\n"


def train_hf_unigram(train_files: list[str], vocab_size: int = 2000):
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram
    from tokenizers.trainers import UnigramTrainer
    from tokenizers.pre_tokenizers import Metaspace
    from tokenizers.decoders import Metaspace as MetaspaceDecoder

    tokenizer = Tokenizer(Unigram())
    tokenizer.pre_tokenizer = Metaspace(replacement="▁")
    tokenizer.decoder = MetaspaceDecoder(replacement="▁")

    trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
    tokenizer.train(files=train_files, trainer=trainer)
    return tokenizer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, help="experiment/run name for artifacts")
    p.add_argument("--train", required=True, help="path to training text file")
    p.add_argument("--valid", help="optional path to validation text file")
    p.add_argument("--test", help="optional path to test text file")
    p.add_argument("--vocab-size", type=int, default=2000)
    p.add_argument("--out", default="artifacts", help="artifacts base directory")
    args = p.parse_args(argv)

    run_dir = create_run_dir(args.exp, base_dir=args.out)
    paths = default_artifact_paths(run_dir)

    cfg = {
        "algo": "hf_unigram",
        "vocab_size": int(args.vocab_size),
        "train_file": str(Path(args.train).resolve()),
        "valid_file": str(Path(args.valid).resolve()) if args.valid else None,
        "test_file": str(Path(args.test).resolve()) if args.test else None,
    }
    write_yaml_like(paths["config"], cfg)

    # Record env
    write_text(paths["env"], _gather_env_text_fallback())

    # Measure timing and memory
    t0 = time.perf_counter()
    c0 = time.process_time()
    proc = psutil.Process()

    tokenizer = train_hf_unigram([cfg["train_file"]], vocab_size=args.vocab_size)

    wall_s = time.perf_counter() - t0
    cpu_s = time.process_time() - c0
    ru = resource.getrusage(resource.RUSAGE_SELF)
    mem_kb = getattr(ru, "ru_maxrss", 0)

    # Save tokenizer and metrics
    tokenizer.save(str(paths["tokenizer"]))

    metrics = {
        "train_wall_time_s": wall_s,
        "train_cpu_time_s": cpu_s,
        "peak_rss_kb": mem_kb,
        "vocab_size": args.vocab_size,
        "num_threads": len(proc.cpu_affinity()) if hasattr(proc, "cpu_affinity") else None,
    }
    write_json(paths["metrics"], metrics)
    write_text(paths["logs"], f"trained hf_unigram in {wall_s:.3f}s; peak_rss_kb={mem_kb}\n")

    print(str(run_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

