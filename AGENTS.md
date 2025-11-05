# Repository Guidelines

## Project Structure & Module Organization
- `src/` Python package (framework, tokenizers, metrics, CLI helpers).
- `scripts/` CLI entry points (e.g., `prepare_data`, `train_tokenizer`, `eval_compression`).
- `experiments/` versioned configs (TOML/YAML) driving runs.
- `data/raw/`, `data/processed/` local datasets and splits (do not commit large files).
- `artifacts/` per‑run outputs: `config.yaml`, `metrics.json`, `tokenizer.json`, `logs.txt`, `env.txt`.
- `paper/`, `references/`, `notes/` writing and background.

## Build, Test, and Development Commands
- Python 3.13 is required. Run via uv only. Project deps are pinned; no `--with` needed.
  - `uv run python -V` (verify env)
  - `uv run python -m scripts.prepare_data --out data/processed`
  - `uv run python -m scripts.train_tokenizer --config experiments/hf_bpe.toml`
  - `uv run python -m scripts.eval_compression --config experiments/hf_bpe.toml`
  - `uv run pytest -q` (tests)

## Coding Style & Naming Conventions
- Python 3.13, 4‑space indent, PEP8.
- Names: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Prefer type hints; keep modules focused and small.
- Formatting/lint: `uv run ruff format .` and `uv run ruff check .`.

## Testing Guidelines
- Framework: `pytest`. Place tests under `tests/` named `test_*.py`.
- Add unit tests for new features plus HF‑compat invariants:
  - Export `tokenizer.json`, reload with `tokenizers.Tokenizer.from_file`, and assert round‑trip `detokenize(tokenize(x)) == x` for sample texts.
- Keep tests fast; prefer synthetic inputs and small corpus slices.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject; include rationale in body when non‑trivial.
- PRs: describe purpose, linked issues, how to run, and artifact path (e.g., `artifacts/2025-11-01_expname/`). Include before/after metrics when relevant.

## Security & Configuration Tips
- Do not commit large datasets or generated artifacts; keep only `.gitkeep` placeholders.
- Preserve 100% HF compatibility for any exported tokenizer (`tokenizer.json`).
- Record environment for every run (write `env.txt` via uv and `pip freeze`/`python -V`).

## Replication Focus
- This repository hosts a research paper plus related source and experiment data. Aim for strict replication:
  - Commit configs to `experiments/` and pin random seeds.
  - Snapshot dataset versions and document sources/hashes.
  - Save all run artifacts under `artifacts/<date>_<expname>/`.

## Architecture Overview & Agent Notes
- Python‑first design; promote hotspots to Rust via `pyo3` only if benchmarks require, keeping HF format compatibility.
- Agents editing this repo should use `apply_patch`, keep diffs minimal, and prefer `uv run` in any example or automation.

## Current Research Direction (2025-11-05)
**Breakthrough**: Exponential quality weighting (p=3) achieves 1.8-3.3% improvement, validated at scale.
- Core insight: Traditional voting underperforms due to insufficient quality differentiation; merge order preservation is irrelevant
- Solution: `power` parameter in `build_merge_weighted_bpe_json()` with p=3 amplifies quality exponentially
  - Mechanism: weight_i = (quality_i)^3, where quality = 1/TPB on held-out data
  - A tokenizer with 1.2× better quality gets 1.7× more voting influence with p=3
  - Higher exponent enables leveraging larger ensembles (K=16 > K=8)
- Results across 860 evaluations (5 seeds × 2 vocab × 2 K × 4 domains):
  - exp_p3: -1.83% to -3.31% improvement across all domains (literary, web, code)
  - exp_p2: -1.37% to -2.25% improvement (works but weaker, degrades with K↑)
  - sequential: +4-5% DEGRADATION (failed hypothesis - order preservation doesn't help)
  - baseline: 0% (reference)
- Cross-domain validation:
  - Literary (PG TEST/OOS): -1.83% to -2.60%
  - Web (FineWeb): -2.54%
  - Code (The Stack): -3.31% (strongest - code has sharper quality signals)
- K-scaling: exp_p3 improves with larger K (K=16: -0.37% to -0.92% better than K=8); exp_p2 degrades with K↑
- Status: COMPLETE - ready for paper writing
- Location: `src/ebpe/ensemble.py`, `/nas4/data/experiments/ensemble-bpe/paper_e2e_evaluations/`
- Agents should focus on paper writing, figure generation, and documenting the quality amplification mechanism

## Experiment Automation Roadmap (2025-11-05)
**Problem Identified**: Current pipeline requires manual intervention for complete automation
- Nov 2025 run required manual creation of merge tokenizers and separate Stack evaluation
- Root cause: Merge tokenizer creation (exp_p2, exp_p3, sequential, etc.) not automated
- Gap: Missing phase between training and evaluation to create all 7 merge variants

**Solution Designed**: See `AUTOMATION.md` for complete architecture
- **Minimal viable (1-2 days)**: 3 scripts eliminate manual intervention
  - `create_all_merges.py` - batch create all merge tokenizers
  - `evaluate_all_domains.py` - parallel evaluation on all domains
  - `verify_experiment.py` - completeness checking
- **Full system (1-2 weeks)**: Manifest-based orchestrator with resumption
  - `run_paper_automated.py` - single-command complete automation
  - Manifest.json registry for tracking all artifacts
  - Idempotent execution, automatic verification, error recovery

**Priority**: Implement before next large-scale experiment run
- Current checkpoint (checkpoint-20251105) is production-ready for paper
- Automation improves future reproducibility and eliminates manual steps
- Agents should reference AUTOMATION.md when implementing experiment runners
