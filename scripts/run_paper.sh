#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
run_paper.sh — one-command entry point for all experiments

Examples
  # Laptop-safe probe (streaming; tiny samples)
  ./scripts/run_paper.sh --profile small

  # Server run (non-streaming; pin snapshot; include K=16)
  ./scripts/run_paper.sh --profile full \
    --revision <hf_dataset_revision> \
    --seeds 13 17 19 \
    --k 1 2 4 8 --include-k16 \
    --sizes 8192 16384 32768 65536

Flags
  --profile {small,full}     Profile size; small for laptops, full for servers
  --revision <hash>          HF dataset revision to pin (recommended for full)
  --dataset <id>             HF dataset id (default: common-pile/project_gutenberg)
  --seeds <s1> [s2 ...]      Seed list (default: 13 17 19)
  --k <k1> [k2 ...]          K values for ensemble scaling (default: 1 2 4 8)
  --include-k16              Append K=16 to the list (safety opt-in)
  --sizes <v1> [v2 ...]      Vocab sizes (default: 8192 16384 32768 65536)
  --out <dir>                Top-level output dir (default: artifacts/paper_run)
  --dry-run                  Print the uv command and exit
  -h, --help                 Show this help
USAGE
}

require() { command -v "$1" >/dev/null 2>&1 || { echo "error: missing dependency '$1'" >&2; exit 1; }; }

PROFILE=small
REVISION=""
DATASET="common-pile/project_gutenberg"
OUTDIR="artifacts/paper_run"
INCLUDE_K16=0
DRYRUN=0
SEEDS=(13 17 19)
KLIST=(1 2 4 8)
SIZES=(8192 16384 32768 65536)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2;;
    --revision) REVISION="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --out) OUTDIR="$2"; shift 2;;
    --seeds) shift; SEEDS=(); while [[ $# -gt 0 ]] && [[ ${1:0:1} != '-' ]]; do SEEDS+=("$1"); shift; done;;
    --k) shift; KLIST=(); while [[ $# -gt 0 ]] && [[ ${1:0:1} != '-' ]]; do KLIST+=("$1"); shift; done;;
    --include-k16) INCLUDE_K16=1; shift;;
    --sizes) shift; SIZES=(); while [[ $# -gt 0 ]] && [[ ${1:0:1} != '-' ]]; do SIZES+=("$1"); shift; done;;
    --dry-run) DRYRUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 2;;
  esac
done

require uv

# Build uv command
CMD=(uv run python -m scripts.run_paper --profile "$PROFILE" --dataset "$DATASET" --out "$OUTDIR")
if [[ -n "$REVISION" ]]; then CMD+=(--revision "$REVISION"); fi
for s in "${SEEDS[@]}"; do CMD+=(--seeds "$s"); done
for k in "${KLIST[@]}"; do CMD+=(--k "$k"); done
if [[ $INCLUDE_K16 -eq 1 ]]; then CMD+=(--include-k16); fi
for v in "${SIZES[@]}"; do CMD+=(--sizes "$v"); done

echo "[run_paper.sh] running: ${CMD[*]}" >&2
if [[ $DRYRUN -eq 1 ]]; then exit 0; fi

"${CMD[@]}"

