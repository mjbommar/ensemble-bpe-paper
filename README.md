**Overview**
- Research framework for ensemble tokenization (BPE and variants) with strict Hugging Face compatibility. Focuses on measurable metrics first: compression on held‑out data, peak RAM during training, and CPU training time. All experiment artifacts are saved to disk for reproducibility and paper figures.

**Goals**
- Measure out‑of‑sample compression gains vs strong baselines.
- Keep training RAM/time competitive; profile and document trade‑offs.
- Preserve 100% HF `tokenizers`/`transformers` compatibility for exported tokenizers.

**Key Constraints**
- Use `uv` for all Python execution: `uv run python ...` or `uv run --with <pkg> python ...`.
- Start in Python; move hotspots to Rust via `pyo3` only if benchmarks demand it.
- Save artifacts for every run under `artifacts/` (configs, metrics, tokenizer.json, logs, env).

**Project Structure**
- `src/` — Python package code (framework, tokenizers, metrics, CLI).
- `scripts/` — Entry points for data prep, training, and evaluation.
- `experiments/` — Versioned configs (TOML/YAML) describing datasets and runs.
- `data/` — Local data area:
  - `data/raw/` — Original datasets (HF cache or copies as permitted).
  - `data/processed/` — Preprocessed text splits (train/val/test; in/out‑of‑sample).
- `artifacts/` — Outputs per run: `config.yaml`, `metrics.json`, `tokenizer.json`, `logs.txt`, `env.txt`, optional `profile/`, `plots/`.
- `paper/` — Writing (NeurIPS/ACM/ICAIL/openreview style), figures generated from artifacts.
- `references/`, `notes/` — Background materials and literature.

**Python Setup (uv)**
- Use Python `3.13` and install `uv`.
- Dependencies are declared in `pyproject.toml` (tokenizers, datasets, transformers, tiktoken, psutil, rich, matplotlib). No `--with` flags needed in normal usage.

**Repository Scope**
- This repo contains a research paper plus related source code and experiment artifacts. Prioritize replication over novelty; every result must be reproducible from `experiments/` configs and `artifacts/`.

**Datasets**
- Synthetic multilingual word lists from `/usr/share/dict/*` for controlled tests.
- Local books (offline, default for replication): place small `.txt` files under `data/raw/local_books/` and optional OOS files under `data/raw/local_books_oos/`.
- Hugging Face (full experiments): use `common-pile/project_gutenberg` (auto‑converted to Parquet). It exposes a `train` split with a `text` column; the UI’s “First 5GB” view is ~14.1k docs (~3.1GB), while the full train split is ~73.5k docs (~26GB). Use sampling flags to cap sizes.

**Dataset Setup (IS/OOS, shuffling, limiting)**
- In‑sample (IS) vs Out‑of‑sample (OOS):
  - IS splits are created as `train.txt`, `valid.txt`, `test.txt` in a single output directory.
  - OOS splits use a second provider block or path (e.g., `data/raw/local_books_oos/`) evaluated separately.
- Shuffling: all providers shuffle deterministically with `--seed` before slicing.
- Limiting for small runs: pass `--sample-train/--sample-valid/--sample-test` to cap document counts while preserving deterministic order.
- Streaming: not required for tiny local corpora; for large HF parquet datasets, prefer parquet‑backed sources (e.g., `common-pile/project_gutenberg`). Use sampling flags to cap sizes or pre‑materialize text and point `--local-files` at the directory.
- Manifests: each prepared dataset writes `manifest.json` with counts, seed, and source metadata (including HF `resolved_revision` when available).

Local books IS/OOS by file (titles) and HF hash-based OOS
- Use `--split-strategy by_file` to allocate entire files (titles) to IS or OOS.
- Hold out OOS by fraction/count: `--oos-frac 0.2` or `--oos-count 5` (deterministic w.r.t. `--seed`).
- Or provide a separate OOS directory: `--oos-local-files data/raw/local_books_oos`.
- For HF datasets or local by_line, use deterministic hash-based OOS with `--oos-by-hash` and `--oos-frac/--oos-count`. The script writes `test_oos.txt` and updates counts in `manifest.json`.

**System Packages (/usr/share/dict)**
- On Debian/Ubuntu, install wordlists that populate `/usr/share/dict`.
- Minimum for English: `wamerican` (also provides the `words` symlink via `dictionaries-common`).
- Optional larger English list: `wamerican-huge`.
- Examples for multilingual OOS checks used here: `wfrench` (french), `wspanish` (spanish), `wdanish` (danish), `wesperanto` (esperanto).
- Install what you need (example):
  - `sudo apt-get update && sudo apt-get install -y dictionaries-common wamerican wfrench wspanish wdanish wesperanto`
- Verify available files: `ls -1 /usr/share/dict` (look for `american-english`, `french`, `spanish`, etc.).

**Baselines**
- HF GPT‑2 BPE and BERT WordPiece (`bert-base-uncased`).
- OpenAI `tiktoken` GPT encodings.

**Replication First**
- Pin seeds and config files; commit configs to `experiments/`.
- Snapshot dataset versions/hashes; record in artifact `env.txt`.
- Save all outputs to `artifacts/<date>_<expname>/`; plots/tables are generated from these.

**Initial Metrics**
- Compression ratio on held‑out documents.
- Peak RAM during training (psutil/resource).
- CPU training time (wall and CPU process time).

**Running**
- One‑shot pipeline (prepare → train → eval) using TOML config:
  - `uv run python -m scripts.run_experiment --config experiments/hf_bpe.toml`
- Local books (offline, recommended for replication):
  - Prepare: `uv run python -m scripts.prepare_hf_books --local-files data/raw/local_books --out data/processed/local_books_tiny --sample-train 200 --sample-valid 20 --sample-test 20 --seed 13`
  - Prepare with OOS by file: `uv run python -m scripts.prepare_hf_books --local-files data/raw/local_books --out data/processed/local_books_tiny --split-strategy by_file --oos-frac 0.25 --seed 13`
  - Pipeline: `uv run python -m scripts.run_experiment --config experiments/local_books_bpe_tiny.toml`
- HF books (optional; only datasets that expose data files directly):
  - Project Gutenberg (Common Pile):
    - `uv run python -m scripts.prepare_hf_books --dataset common-pile/project_gutenberg --split train --text-field text --sample-train 500000 --sample-valid 10000 --sample-test 10000 --oos-by-hash --oos-frac 0.1 --out data/processed/pg_books`  
    - For a small offline replica, first pre‑materialize a subset under `data/raw/local_books/` and use `--local-files` instead.
- Offline local-books tiny examples (no network):
  - E0 (single vs ensemble):
    - `uv run python -m scripts.run_e0 --single experiments/local_books_bpe_tiny.toml --ensemble experiments/local_books_bpe_tiny_ensemble.toml`
  - E1 (algo performance table on same data):
    - `uv run python -m scripts.run_e1 --config experiments/local_books_bpe_tiny.toml`
  - E2 (multilingual dict stress test):
    - `uv run python -m scripts.run_e2 --dict-path /usr/share/dict/american-english --oos-dict-path /usr/share/dict/french`
  - Individual steps (advanced):
  - Data prep: `uv run python -m scripts.prepare_data --dict-path /usr/share/dict --out data/processed`
  - Train HF BPE: `uv run python -m scripts.train_tokenizer --exp hf_bpe --train data/processed/train.txt --vocab-size 2000`
    - Eval compression: `uv run python -m scripts.eval_compression --exp hf_bpe_eval --tokenizer artifacts/<train_run>/tokenizer.json --eval-file data/processed/test.txt`
  - Weighted merge (advanced):
    - `uv run python -m scripts.merge_ensemble --exp merge_run --ensemble-run artifacts/<ens_dir> --weight-by char_entropy --theta 0.6`
    - Weighting options: `--weight-by {bytes,lines,entropy,uniq_bytes,chars,uniq_chars,char_entropy,custom}` with `--weights` for custom values; `--theta` applies a weighted support threshold (fraction of total weight).

**Full Paper Run (one command)**
- Small laptop-safe profile (streaming; tiny samples):
  - `uv run python -m scripts.run_paper --profile small` or `./scripts/run_paper.sh --profile small`
- Full server profile (non-streaming; larger samples; pass HF revision to pin snapshot):
  - `uv run python -m scripts.run_paper --profile full --revision <hf_revision_hash> --seeds 13 17 19`  
    or `./scripts/run_paper.sh --profile full --revision <hf_revision_hash> --seeds 13 17 19 --k 1 2 4 8 --include-k16`
- Outputs:
  - Aggregated CSV/JSON under `artifacts/paper_run/aggregated/`
  - Markdown/LaTeX tables under `results/table.md` and `results/table.tex`
  - E0 K-scaling summaries under `artifacts/paper_run/e0_k_scaling/`

**Bash Entry Point**
- Convenience wrapper: `scripts/run_paper.sh` mirrors the Python runner flags and adds `--dry-run`.
- Examples:
  - `./scripts/run_paper.sh --profile small`
  - `./scripts/run_paper.sh --profile full --revision <hf_rev> --seeds 13 17 19 --k 1 2 4 8 --include-k16 --sizes 8192 16384 32768 65536`

**Python ↔ Rust Plan**
- Prototype all functionality in Python. If microbenchmarks show unacceptable runtime or memory, implement the hotspot in Rust and expose via `pyo3`, keeping exported `tokenizer.json` fully HF‑compatible.

**Paper Style**
- Write in a typical NeurIPS/ACM/ICAIL/openreview style. Figures and tables should be generated from `artifacts/` via deterministic scripts.

**Next Steps**
- Run experiments at vocab sizes 8k, 16k, 32k, 64k. Configs are provided under `experiments/gutenberg_*`.
- Sweep E1 across sizes with:
  - `uv run python -m scripts.run_e1_sizes --config experiments/gutenberg_e1_32k.toml --sizes 8192 16384 32768 65536`
