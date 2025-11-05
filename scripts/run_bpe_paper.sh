#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
run_bpe_paper.sh — BPE-only experiment runner (avoids Unigram issues)

Examples
  # Quick test
  ./scripts/run_bpe_paper.sh --profile small --seeds 13 17

  # Full paper run
  ./scripts/run_bpe_paper.sh --profile full \
    --seeds 13 17 19 \
    --sizes 16384 32768 65536 \
    --k 1 2 4 8 \
    --run-ensembles

Flags
  --profile {small,full}     Profile size
  --revision <hash>          HF dataset revision to pin (recommended for full)
  --dataset <id>             HF dataset id (default: common-pile/project_gutenberg)
  --seeds <s1> [s2 ...]      Seed list (default: 13 17 19)
  --sizes <v1> [v2 ...]      Vocab sizes (default: 16384 32768 65536)
  --k <k1> [k2 ...]          K values for ensembles (default: 1 2 4 8)
  --run-ensembles            Run ensemble experiments
  --out <dir>                Output dir (default: artifacts/bpe_paper_run)
  --dry-run                  Print command and exit
  -h, --help                 Show this help
USAGE
}

require() { command -v "$1" >/dev/null 2>&1 || { echo "error: missing dependency '$1'" >&2; exit 1; }; }

PROFILE=small
REVISION=""
DATASET="common-pile/project_gutenberg"
OUTDIR="artifacts/bpe_paper_run"
RUN_ENSEMBLES=0
DRYRUN=0
SEEDS=(13 17 19)
KLIST=(1 2 4 8)
SIZES=(16384 32768 65536)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2;;
    --revision) REVISION="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --out) OUTDIR="$2"; shift 2;;
    --seeds) shift; SEEDS=(); while [[ $# -gt 0 ]] && [[ ${1:0:1} != '-' ]]; do SEEDS+=("$1"); shift; done;;
    --k) shift; KLIST=(); while [[ $# -gt 0 ]] && [[ ${1:0:1} != '-' ]]; do KLIST+=("$1"); shift; done;;
    --sizes) shift; SIZES=(); while [[ $# -gt 0 ]] && [[ ${1:0:1} != '-' ]]; do SIZES+=("$1"); shift; done;;
    --run-ensembles) RUN_ENSEMBLES=1; shift;;
    --dry-run) DRYRUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 2;;
  esac
done

require uv

# Build command
CMD=(uv run python -m scripts.run_bpe_paper --profile "$PROFILE" --dataset "$DATASET" --out "$OUTDIR")
if [[ -n "$REVISION" ]]; then CMD+=(--revision "$REVISION"); fi
CMD+=(--seeds "${SEEDS[@]}")
CMD+=(--k "${KLIST[@]}")
CMD+=(--sizes "${SIZES[@]}")
if [[ $RUN_ENSEMBLES -eq 1 ]]; then CMD+=(--run-ensembles); fi

echo "[run_bpe_paper.sh] running: ${CMD[*]}" >&2
if [[ $DRYRUN -eq 1 ]]; then exit 0; fi

"${CMD[@]}"
