# Ensemble BPE Research: Project TODO

This TODO reflects the current project goals and constraints as discussed on 2025-11-01. It emphasizes practical, measurable metrics; compatibility with Hugging Face tokenizers/transformers; reproducibility via uv; and saving all artifacts for later paper analysis. The repository is a research paper plus related source and experiment data; prioritize replication.

Key guardrails and decisions:
- Success metrics (initial):
  - Compression ratio on held-out/new documents (out-of-sample).
  - Peak RAM usage during tokenizer training (measure via psutil).
  - CPU training time (wall/time.process_time, stdlib `time` + `resource`).
- Baselines: HF `tokenizers` GPT-2 BPE, BERT WordPiece (bert-base-uncased), and OpenAI `tiktoken` GPT encodings.
- Paper writing style: NeurIPS/ACM/ICAIL/openreview; figures/tables reproducible from saved artifacts.
- Implementation strategy: start fully in Python; run microbenchmarks; if hotspots are identified, move the tight loops to Rust and expose via `pyo3`, preserving 100% HF compatibility.
- HF compatibility is a must: ensure our vocab/merges JSON can be loaded by `tokenizers.Tokenizer.from_file` and by `transformers` tokenizers.
  - [x] Added test to load `tokenizer.json` with `transformers.PreTrainedTokenizerFast` and round-trip a sample string (`tests/test_transformers_compat.py`). (2025-11-01)
- Datasets: (a) synthetic from `/usr/share/dict/*` (multi-language word lists), (b) public-domain books via Hugging Face (Common Pile / Project Gutenberg) with explicit in-sample vs out-of-sample splits.
- Focus: mostly English; include small multilingual tests using the word files.
- Python execution: always run via `uv` (e.g., `uv run python ...` or `uv run --with tokenizers python ...`). Target Python 3.13.
- Artifacts: persist every run (configs, logs, metrics.json, profiles, tokenizer files) under `artifacts/` for downstream analysis and paper figures.

## Phase 1: Core Infrastructure

- [ ] Review and finalize experiment framework design document
- [ ] Set up Rust project structure with HuggingFace tokenizers dependency
- [ ] Implement TokenizerComponent trait and base abstractions
- [ ] Implement Corpus management with partitioning strategies
- [ ] Implement Vocabulary and MergeRules data structures
- [x] Implement artifact run directory + env recorder (writes `env.txt` and `logs.txt`)

## Phase 2: Component Tokenizer Implementations

- [x] Create wrappers for HuggingFace BPE, Unigram, WordPiece — implemented `scripts.train_tokenizer` (BPE), `scripts.train_unigram`, and `scripts.train_wordpiece` with tests and pipeline integration (`train.algo = "hf_bpe"|"hf_unigram"|"hf_wordpiece"`).
- [ ] Implement Picky BPE with IoS (Intersection over Self) filtering
- [ ] Implement Scaffold-BPE with scaffold token detection
- [ ] Implement morphological-aware BPE variant

## Phase 3: Evaluation Metrics

- [x] Compression ratio on held-out data (bytes/token counts over documents) — implemented `scripts.eval_compression` with tests
- [x] CPU training time measurements (per run) — recorded in `metrics.json` from `scripts.train_tokenizer` and validated by test
- [x] Peak RAM during training (psutil + resource.getrusage) — recorded as `peak_rss_kb` and validated by test
- [x] Metrics aggregation and reporting (corpus-level JSON + CSV) — `scripts.eval_compression` writes `metrics.json` and `metrics.csv`
  - [x] Include `vocab_size` in pipeline summaries and E1 results to ensure fair, matched-size comparisons across algorithms. (2025-11-02)
- [ ] Scale evaluations to realistic vocab sizes (8k, 16k, 32k, 64k) and ensure all algo comparisons are matched-size.
  - [x] Initial 16k sweep (3 seeds, K=2) complete on Project Gutenberg (streaming, 2k/200/200). Ensemble selection improves IS/OOS compression and lowers per-member peak RSS/time; total ensemble time ~+10% vs single. (2025-11-02)
- [ ] NICE-TO-HAVE/Backlog: CPT/BPT/fertility/NSL; Zipf alignment (R²); vocab-quality (IoS, scaffold ratio); statistical consistency; cross-lingual fairness

## Phase 4: Ensemble Construction & Combination

- [x] Implement bootstrap ensemble builder (bagging) — `scripts.train_ensemble` shards the corpus, trains K tokenizers, evaluates on held‑out, and selects best (min tokens/byte).
- [x] Implement domain-specific ensemble builder — `scripts.train_domain_ensemble.py` (one member per domain file, selection + optional merge for BPE); test `tests/test_train_domain_ensemble.py`. (2025-11-01)
- [x] Implement heterogeneous algorithm ensemble builder — `scripts.train_hetero_ensemble.py` (members across hf_bpe/wordpiece/unigram, selection); test `tests/test_train_hetero_ensemble.py`. (2025-11-01)
- [x] Implement vocabulary merging and merge-rule voting (k-of-n) — added `src/ebpe/ensemble.py` and `scripts/merge_ensemble.py`; integrates with pipeline via `[ensemble.merge]` to produce and evaluate a merged tokenizer; covered by tests `tests/test_train_ensemble.py` (merge flow) and `tests/test_run_experiment_merge.py`. (2025-11-01)
- [x] Record shard statistics for weighting (lines, bytes, uniq_bytes, entropy_bits) in `ensemble.json` from `scripts.train_ensemble` and use them for merge weighting. (2025-11-02)
- [x] Add weighted merge voting with `--weight-by {bytes,lines,entropy,uniq_bytes,chars,uniq_chars,char_entropy,custom}` and optional `--theta` in `scripts.merge_ensemble`; core logic in `build_merge_weighted_bpe_json`. Tests updated. (2025-11-02)
- [ ] Implement byte-level ensemble framework
- [ ] Implement parallel inference with consensus voting

### Basic Ensemble Definition (Agreed)

- Given N-document corpus and chunk size M, form K ≈ ceil(N/M) shards.
- Train K tokenizers (same hyperparameters) — one per shard.
- Combine via ensemble:
  - [x] Selection baseline: choose the single best member by out-of-sample tokens/byte; export `selected_tokenizer.json` (HF-compatible). Implemented in `scripts.train_ensemble`.
  - [x] Merge-voting: construct a combined BPE by k‑of‑n voting over merges; export HF-compatible `tokenizer.json`.
  - [ ] Token-level consensus (backlog): parallel encode across members, consensus vote on tokens; measure overhead and quality.

### Pipeline Support

- [x] Extend `scripts.run_experiment` to optionally run the ensemble path from TOML (train_ensemble → eval_compression on selected/merged model) and emit a summary CSV across members and the combined model. Implemented `ensemble_summary.csv` and manifest fields `ensemble_run_dir`, `ensemble_eval_run_dir`; covered by `tests/test_run_experiment_ensemble.py` (2025-11-01).

## Phase 5: Evaluation Harness & Data Preparation

- [x] Create evaluation harness (runs all success metrics, saves artifacts) — `scripts.run_experiment` + `experiments/hf_bpe.toml`
 - [x] Baseline tokenizer experiments: add `tiktoken` compression evaluator (no training)
 - [x] Baseline tokenizer experiments: add WordPiece training CLI + test
 - [x] Baseline tokenizer experiments: integrate WordPiece and tiktoken baselines into pipeline configs — extended `scripts.run_experiment` with `train.algo = "hf_wordpiece"` and `[baselines.tiktoken].encodings=[...]`, added pipeline summary CSV/JSON; covered by tests `tests/test_run_experiment_wordpiece.py` and `tests/test_run_experiment_tiktoken_baseline.py` (2025-11-01).
- [ ] Acquire and prepare datasets:
  - [x] Synthetic multilingual word lists from `/usr/share/dict/*` (CLI `scripts.prepare_data` + tests)
  - [x] Add HF books preparation CLI with local offline mode for tests — `scripts.prepare_hf_books.py` with `--dataset/--config/--split/--revision` and `--local-files`; covered by `tests/test_prepare_hf_books_local.py` (2025-11-01).
  - [x] Integrate HF provider into pipeline (`scripts.run_experiment` supports `[data].provider="hf_books"` with dataset or `local_files`); covered by `tests/test_run_experiment_hf_local.py`.
  - [x] Pin concrete HF dataset IDs in example configs and update README with commands (added `experiments/pg19_bpe_tiny.toml`, `experiments/pg19_bpe_tiny_with_oos_wikitext.toml`, and `experiments/pg19_bpe_tiny_ensemble.toml`; documented `pg19` and `wikitext-2-v1`).
  - [x] Record resolved dataset revision in HF manifest (if available) via `scripts.prepare_hf_books` (`resolved_revision` field).
  - [ ] Pin dataset revisions (commit hashes) for long-term reproducibility once we settle on exact subsets — deferred: HF loading scripts (e.g., `pg19.py`) are not supported in this runtime; favor local text or parquet-backed datasets and revisit pinning after selecting a compatible HF source. (2025-11-02)
  - [x] Document dataset workflow (IS/OOS splits, deterministic shuffling, and `--sample-*` limiting) in README; include offline local_books path and HF limitations. (2025-11-02)
  - [x] Define explicit IS/OOS title-level splits for local books via `scripts.prepare_hf_books --split-strategy by_file` with `--oos-frac/--oos-count` and optional `--oos-local-files`; writes `test_oos.txt` and records counts in `manifest.json`. (2025-11-02)
  - [ ] Document licenses and snapshot hashes for any selected HF parquet-backed datasets (pending dataset choice)
  - [x] Add pipeline OOS evaluation support (`[eval].run_oos=true` and optional `[eval].oos_eval_file`; dictionary provider auto-detects `test_oos.txt`). Covered by `tests/test_run_experiment_oos_dict.py` (2025-11-01).
- [x] Document required apt packages for `/usr/share/dict` wordlists in README

  - [x] Auto-detect OOS for hf_books local provider via `test_oos.txt` in pipeline (mirrors dict provider); pass `--split-strategy/--oos-*` from TOML to `prepare_hf_books`. Added test `tests/test_run_experiment_oos_hf_local_by_file.py`. (2025-11-02)

## Baseline HF BPE (initial)

- [x] Implement `scripts/train_tokenizer.py` (HF BPE) writing `tokenizer.json`, `metrics.json`, `config.yaml`, `env.txt`, `logs.txt`
- [x] Add pytest that trains on a tiny corpus and verifies HF compatibility (encode→decode round-trip)
 - [x] Add `scripts/eval_compression_tiktoken.py` (no training) + tests to benchmark `tiktoken` encodings for compression

## Phase 6: Core Experiments

- [ ] Run Experiment 1: APX-completeness bound testing
- [ ] Run Experiment 2: Statistical consistency verification
- [ ] Run Experiment 3: Zipf alignment optimization
- [ ] Run Experiment 4: Domain generalization testing
- [ ] Run Experiment 5: Multilingual fairness testing
- [ ] Run Experiment 6: Computational cost-benefit analysis
- [ ] Run Experiment 7: Component diversity impact study
- [ ] Run Experiment 8: Ensemble size ablation (M = 2 to 50)
- [ ] EARLY EXPERIMENTS (aligned to success metrics):
  - [x] E0: Compare single HF BPE vs ensemble selection (best-of-K) on out-of-sample compression — added `scripts/run_e0.py` to run baseline + ensemble and aggregate; includes tiny HF configs (`experiments/pg19_bpe_tiny*.toml`). (2025-11-01)
  - [x] E0 (tiny) executed on local-books offline dataset — artifacts under `artifacts/*local_books*` and summary at `artifacts/e0/e0_summary.csv`. (2025-11-01)
  - [x] E1: Peak RAM and CPU time across tokenizers on same corpus slice — added `scripts/run_e1.py` to run hf_bpe, hf_wordpiece, hf_unigram on the same config and write `e1_summary.{csv,json}`. (2025-11-01)
  - [x] E1 (tiny) executed on local-books offline dataset — summary at `artifacts/e1/e1_summary.csv`. (2025-11-01)
  - [x] E2: Small multilingual dictionary-based stress test (single vs ensemble selection) — added `scripts/run_e2.py` (dictionary provider wrapper over E0); validated with offline local dicts; documented in README. (2025-11-01)

## Phase 7: Downstream Task Integration

- [ ] Implement language modeling evaluation integration
- [ ] Implement machine translation evaluation integration

## Phase 8: Analysis & Visualization

- [x] Add metrics aggregator across runs — `scripts/aggregate_metrics.py` collects all `pipeline_summary.csv` files into a single CSV/JSON table for figures; covered by `tests/test_aggregate_metrics.py` (2025-11-01).
- [x] Add seed sweep runner and stats — `scripts/run_seeds.py` runs multiple seeds for a base config and writes per‑seed results + mean/std; covered by `tests/test_run_seeds_local.py` (2025-11-01).
- [x] Create visualization tools (Zipf R²) — `scripts/plot_zipf.py` computes log‑log Zipf fit (R²) and writes PNG/JSON; covered by `tests/test_plot_zipf.py`. (2025-11-01)
- [ ] Create additional visualizations (fertility distributions, tables-to-figures)
  - [x] Produce initial small-scale aggregated results and microbench numbers (local-books offline):
    - E0/E1/E2 summaries under `artifacts/e0`, `artifacts/e1`, plus seed sweep at `artifacts/seeds_local_books`.
    - Microbench throughput at `artifacts/microbench_local_books/microbench.csv`.
- [ ] Analyze experimental results and identify patterns

## Phase 9: Paper Writing

- [ ] Write paper: Related Work section with 35+ citations
- [ ] Write paper: Methodology section describing framework
- [ ] Write paper: Results section with tables and figures
- [ ] Write paper: Discussion and ablation analysis
- [ ] Prepare supplementary materials and code release
- [x] Ensure figures/tables are auto-produced from saved artifacts — added `scripts/make_tables.py` for Markdown/LaTeX tables and `scripts/plot_zipf.py` for Zipf plots; validated by tests and example outputs under `results/`. (2025-11-01)

## Phase 10: Repository Management

- [ ] Stage and commit framework design and theoretical analysis docs
- [x] Enforce uv usage in docs and examples; avoid bare `python`
 - [x] CI: intentionally skipped for now; validate locally with `uv run --with pytest --with tokenizers --with psutil --with tiktoken pytest -q`. (2025-11-02)

### Dependency Management
- [x] Pin runtime deps in `pyproject.toml` (tokenizers, datasets, transformers, tiktoken, psutil, rich, matplotlib); drop `--with` usage in docs. (2025-11-02)
- [x] Add dev deps (`pytest`, `ruff`) for local testing/lint without flags. (2025-11-02)

## Python vs Rust decision gates

- [x] Implement microbenchmarks for training throughput and memory — `scripts/microbench_train.py` writes per-run wall/cpu time, peak RSS, and bytes/sec; covered by `tests/test_microbench_train.py`. (2025-11-01)
- [x] Clarify scope: HF `tokenizers` training is already implemented in Rust. We do not plan to re‑implement tokenizer training in Rust. Our Rust targets are ensemble/merge logic and consensus inference only. (2025-11-02)
- [ ] If ensemble/merge/consensus or evaluation loops exceed thresholds, move those hotspots to Rust via `pyo3`, keeping exported artifacts HF‑compatible (same `tokenizer.json`).

## Reproducibility & Artifacts

- Artifacts directory layout per run:
  - `artifacts/<date>_<expname>/config.yaml`
  - `artifacts/<date>_<expname>/metrics.json`
  - `artifacts/<date>_<expname>/tokenizer.json` (HF-compatible)
  - `artifacts/<date>_<expname>/logs.txt`
  - `artifacts/<date>_<expname>/env.txt` (uv, Python, package versions)
  - optional: `profile/`, `plots/`
  - note: pin seeds and dataset snapshot hashes for replication

## Execution notes

- Always invoke Python via uv, e.g.:
  - `uv run --with tokenizers --with datasets --with transformers --with tiktoken --with psutil --with rich python -m scripts.train_tokenizer ...`
  - `uv run python -m scripts.eval_compression ...`

## Paper Abstract (Target) & Results Blueprint

This section works backwards from the abstract we want to claim and enumerates the concrete results required to support it. Keep all comparisons vocabulary-size matched.

- [x] Abstract skeleton (target): “We introduce ensemble tokenization, training K shard‑specialized tokenizers on a corpus partition and combining them via selection or merge‑voting. On in‑sample and held‑out (title‑level) book corpora, ensemble methods achieve equal or better out‑of‑sample compression at matched vocabulary sizes (e.g., 256), with small training RAM/time overheads. The pipeline is fully reproducible; all outputs are HF‑compatible.”
- [ ] Main table (E1): Matched‑vocab (128/256/512) HF‑BPE vs WordPiece vs Unigram on IS/OOS; report tokens_per_byte, train wall/cpu time, peak_rss_kb; N=3 seeds with mean±std.
- [ ] Ensemble study (E0/E3): Single vs selection vs merge‑voting across K∈{2,4,8} and shard size M; plot tokens_per_byte vs K and a bar for overheads.
- [ ] Domain/OOS robustness (E2): Multilingual dict IS English, OOS non‑English; show OOS compression change for ensemble vs single.
- [ ] Ablations: vocabulary size sensitivity; shard strategy (by_file vs random); effect of merge vote threshold k.
- [ ] Performance microbench: throughput (bytes/s) vs vocab size and K; decide Python↔Rust gates.
- [ ] Figures/tables: generate via `scripts/make_tables.py` and `scripts/plot_zipf.py`; archive under `results/` with artifact pointers.

Datasets for the paper (replicable):
- [x] Local books (offline) with by‑file IS/OOS (already supported; `test_oos.txt` auto‑detected).
- [x] `/usr/share/dict` wordlists for multilingual stress tests (apt: `dictionaries-common wamerican [wamerican-huge] wfrench wspanish wdanish wesperanto`).
- [x] Choose HF dataset for full experiments: `common-pile/project_gutenberg` (Parquet-backed; `text` column). Use sampling flags to cap local runs; manifests record `resolved_revision`. (2025-11-02)
- [ ] Pin snapshot revision for camera-ready runs and document license notes in README.

Next actions to reach camera‑ready results:
- [x] Finalize dataset choice for main table (Project Gutenberg via Common Pile) and add deterministic OOS by hash. (2025-11-02)
- [ ] Pin seeds and dataset snapshot revisions (`resolved_revision`) for all full runs; document license notes in README.
- [ ] Run 3‑seed E0/E1/E2 sweeps at 256 vocab and K∈{2,4}; aggregate to CSV/JSON; regenerate tables/plots.
- [ ] Calibrate microbench thresholds and, if exceeded, identify hotspots to migrate to Rust via `pyo3` while keeping HF JSON stable.

## Synthetic Data Details (Dictionary-based)

- [x] Default mode writes one word per line after dedup + deterministic shuffle.
- [x] Optional sequence mode (`--sequence`) generates random-length runs of words per line with light punctuation. Controls: `--seq-min/--seq-max` (default 5–20), `--punct-prob` (default 0.1). OOS words come from a separate list (non-overlapping in word mode).

## Planned “Realistic Vocab” Experiments

- [x] E1 main table (partial): 8k and 16k completed on Project Gutenberg (streaming 2k/200/200), 3 seeds, matched across HF‑BPE/WordPiece/Unigram. Aggregates saved to `artifacts/e1_sizes_pg_step1_agg/` (IS + OOS). (2025-11-02)
- [ ] E1 main table (remaining): 32k and 64k on Project Gutenberg, 3 seeds, matched across HF‑BPE/WordPiece/Unigram.
- [ ] E0 ensemble vs single at 32k with K∈{2,4} on the same corpus; add merge‑voting ablation.
- [x] Add initial TOMLs under `experiments/` for 8k, 16k, 32k, 64k (`gutenberg_*`).
- [x] Add convenience sweep script `scripts/run_e1_sizes.py` to aggregate across sizes.
- [x] Smoke sweep (HF streaming): E1 at 8k/16k with tiny samples (200/20/20), hf_bpe only — artifacts under `artifacts/e1_sizes_smoke`. (2025-11-02)
- [x] Step 1 sweep (HF streaming): E1 at 8k/16k with 2000/200/200, hf_bpe only — artifacts under `artifacts/e1_sizes_step1`. (2025-11-02)
- [x] Step 2 probe (HF streaming): single-size 32k with 500/50/50, hf_bpe — artifacts under `artifacts/e1_sizes_step2_32k_small`. (2025-11-02)
- [x] E0 ensemble smoke (HF streaming): 16k vocab, K=2 shards, selection + merge voting; artifacts under latest `*_pg_bpe_16k_ens_smoke_pipeline`. (2025-11-02)
- [ ] Gate larger runs: set memory/time thresholds for 32k/64k; consider Rust/pyo3 kernel if peak_rss_kb remains >2GB at 32k.
  - [x] 32k small probe (500/50/50) completed: peak_rss_kb ≈ 1.78e6; feasible to scale with modest caps. Full 32k/64k gated by time/memory; proceed incrementally. (2025-11-02)

## Empirical Findings (Interim, to replicate at scale)

- [x] Demonstrate reduced peak RAM and improved OOS compression for ensemble selection vs single (Gutenberg streaming, 2k/200/200, 16k vocab, K=2):
  - Single training peak_rss_kb ≈ 4,676,432; tokens/byte (IS) ≈ 0.2600; (OOS) ≈ 0.2534.
  - Ensemble selected member peak_rss_kb ≈ 2,925,071; tokens/byte (IS) ≈ 0.2565.
  - Artifacts: `artifacts/20251102_044933_pg_bpe_16k_smoke_single_pipeline/pipeline_summary.csv`, `artifacts/20251102_042839_pg_bpe_16k_ens_smoke_pipeline/pipeline_summary.csv` and ensemble best member metrics under its `ensemble.json` → `best.train_run/metrics.json`. (2025-11-02)
  - 3-seed aggregates @16k (K=2), streaming (2k/200/200):
    - Single hf_bpe: IS tokens/byte mean±std ≈ 0.259579±0.000878; peak_rss_kb mean ≈ 4,697,244; train_wall_time_s mean ≈ 82.52.
    - Ensemble selection (selected member only): IS tokens/byte mean±std ≈ 0.255964±0.000888; peak_rss_kb mean ≈ 2,925,071; train_wall_time_s mean ≈ 46.72.
    - Ensemble total training (sum across members): wall_time_s mean ≈ 90.76 (~+10% vs single); member peak RSS unaffected (~2.93e6 KB). OOS tokens/byte for selected tokenizer ≈ 0.2462 (vs single ~0.2573 mean), i.e., improved OOS compression. Artifacts under `artifacts/20251102_051114_*`, `.../051920_*`, `.../052700_*`.
  - Merge‑voting (unweighted) degraded compression; weighted merge with `--weight-by char_entropy --theta 0.6` improved but remained worse than selection (IS ~0.2918; OOS ~0.2803 at 16k). Keep selection as primary method; continue tuning merge hyperparameters. Artifacts: `artifacts/20251102_053545_pg_bpe_16k_ens_smoke_s13_merge_w*`.
  - E1 aggregates (IS/OOS; 3 seeds; streaming 2k/200/200) at 8k & 16k vocab (means ± std):
    - 8k IS: hf_bpe 0.28469±0.00091; hf_wordpiece 0.28177±0.00074; hf_unigram 0.29220±0.00120. 8k OOS: bpe 0.28267±0.00960; wordpiece 0.27857±0.00823; unigram 0.29046±0.01255.
    - 16k IS: hf_bpe 0.25958±0.00088; hf_wordpiece 0.26031±0.00062; hf_unigram 0.26852±0.00114. 16k OOS: bpe 0.25730±0.00918; wordpiece 0.25600±0.00696; unigram 0.26680±0.01194.
    - Training costs (means): 8k wall_time_s ≈ 75.21 (bpe), 38.13 (wordpiece), 509.06 (unigram); peak_rss_kb ≈ 4.62e6, 1.37e6, 2.86e6. 16k wall_time_s ≈ 81.11 (bpe), 40.37 (wordpiece), 479.17 (unigram); peak_rss_kb ≈ 4.58e6, 1.38e6, 2.87e6. Artifacts: `artifacts/e1_sizes_pg_step1_agg/*`.
- [x] Add test for size sweep aggregation on local corpus (`tests/test_run_e1_sizes_local.py`). (2025-11-02)

## Publication-Backward Plan (What the paper must deliver)

- [ ] Final claims (camera-ready):
  - Ensemble selection reduces per-process peak RAM vs single-corpus BPE at matched vocab (8k/16k/32k/64k).
  - Equal-or-better OOS compression on Project Gutenberg; similar trends on a second corpus.
  - Compute/time trade-off: small total overhead for K members; favorable wall/GB saved.
- [ ] Reproducible tables and figures generated from artifacts only:
  - Main table: IS/OOS tokens-per-byte ±95% CI across 3–5 seeds; train wall/CPU; peak_rss_kb; vocab_size.
  - K-scaling plot: K∈{1,2,4,8,16} vs memory/time vs OOS compression.
  - Size sweep plot: vocab∈{8k,16k,32k,64k} comparing BPE/WordPiece/Unigram and ensemble selection.
  - Ablation: merge voting (unweighted vs weighted θ) vs selection.
- [ ] Replication pack: pinned dataset revisions, seeds, env.txt, `uv.lock`, and one-click scripts to regenerate tables/plots.
  - [x] Add bash one-click wrapper `scripts/run_paper.sh` mirroring Python runner flags (small/full profiles, seeds, K, sizes). (2025-11-02)

## K-Scaling and Shard Ablations (Ensemble Strength)

- [ ] E0-K: Run K∈{1,2,4,8,16} at 16k vocab on Gutenberg with constant total docs (e.g., 8k train), shard size M=N/K.
  - [ ] Aggregate mean±std and 95% CI (doc-level bootstrap) for IS/OOS tokens/byte; record train time and peak_rss_kb per member and total.
  - [ ] Add artifacts and CSV: `artifacts/e0_k_scaling_16k/` with `k_scaling_summary.csv`.
- [ ] E0-K (size sensitivity): repeat at 32k vocab with smaller caps; enforce memory/time gates.
- [ ] E0-K-shardsize: Fix K=8 and vary shard size M (documents per shard) to separate “more shards” from “smaller shards”.
- [ ] Domain K: per-title/domain shards (by file) to test heterogeneous shards; compare to random shards at same K.

## Statistical Validation & Significance

- [ ] Implement `scripts/aggregate_metrics.py` enhancements:
  - [ ] Paired tests on the same test set: paired t-test and bootstrap CIs for Δ tokens/byte (ensemble vs single).
  - [ ] Effect sizes (Cohen’s d) and percent relative improvement.
  - [ ] Seed aggregation utilities (n≥3, target 5 where feasible) with deterministic RNG.
- [ ] Generate `artifacts/*/stats.json` alongside CSV with CI and p-values for inclusion in paper tables.

## Cross-Corpus Robustness

- [ ] Pin a second reproducible corpus (parquet-backed HF or local mirror), e.g., a curated subset of The Pile or Wikipedia dumps converted to text.
- [ ] Re-run E0/E1 at 16k/32k on the second corpus; report IS/OOS and resource metrics.
- [ ] Multilingual stress (wordlists): reproduce small E2 runs with K∈{4,8} and measure OOS across languages.

## Dataset Revisions, Licensing, and Reproducibility

- [ ] Switch Gutenberg runs to non-streaming snapshot; record `dataset_revision` in manifest and README (license note).
- [x] Document APT packages for `/usr/share/dict` in README (dictionaries-common, wamerican, optional multilingual lists). (2025-11-02)
- [ ] Add `scripts/print_env.py` to emit python/uv versions and `pip freeze` into `env.txt` for every pipeline.

## Merge/Consensus Ablations

- [x] Weighted merge with entropy/bytes/lines/char-entropy and θ implemented; selection remains primary. (2025-11-02)
- [ ] Extend tests for merge edge cases: θ∈{0,1}, ties, K>8, mixed weights; confirm HF round-trip after merge.
- [ ] Explore consensus at inference (parallel voting over merges) for byte-level variants; measure decode speed and quality.

## Resource Scaling & Gates (when to move Rust)

- [ ] Define gates from microbench CSV: if 32k K=8 exceeds peak_rss_kb > 3GB per member or wall_time > 2× single, move the merge/consensus path to Rust via pyo3 (training already uses Rust inside HF `tokenizers`).
- [ ] Track memory/time vs K and vocab; plot scaling curves and annotate gates in paper.

## Paper Assembly

- [ ] Draft Results section with references to artifact-backed tables/figures and CIs.
- [ ] Write Methods (data, splits, shard construction, weighting, baselines, metrics); include exact seeds and revisions.
- [ ] Finalize Abstract with 1–2 sentence claims and numeric effects from 32k runs.
