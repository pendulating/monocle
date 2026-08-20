#!/usr/bin/env bash
# Launch the Integrative Complexity ingredient corpus over the registered
# thinking runs.
#
# The source of every run is the canonical registry
# (`notebooks/cvpr/canonical_data/trace/<case>__<model>/results.parquet`), thus
# the corpus can be tested against the registry afterwards. The stage follows
# the symlink before it reads the case name and the judge model from the path.
#
# Cost, measured 2026-08-14: about 1,400 traces in 1 GPU-hour. A case holds
# 11,000 traces, thus about 8 GPU-hours. 7 cases = about 59 GPU-hours.
#
# Usage:
#   bash scripts/launch_ic_corpus.sh --dry-run
#   bash scripts/launch_ic_corpus.sh              # preemptable, 7 cases, gemma
#   bash scripts/launch_ic_corpus.sh --pierson    # the 8 GPUs of klara instead
#   bash scripts/launch_ic_corpus.sh --cases schools subway_safety
#   bash scripts/launch_ic_corpus.sh --judge qwen3.5-9b --parallel 8
#
# After it lands:
#   python scripts/merge_trace_extractions.py <sweep_dir> --schema ic
#
# See `vlm-narratives-docs/ic-ingredient-extraction.md`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv-mllmsci-vllm025cu129/bin/python"
REG="${ROOT}/notebooks/cvpr/canonical_data/trace"

ALL_CASES=(subway_safety libraries schools road_quality parks_plazas restaurants street_photography)
JUDGE="gemma-4-12b"
DRY=0
CASES=()

# PREEMPTABLE BY DEFAULT. The `gpu` partition holds 143 nodes and takes a job
# back when an owner wants the node; the `pierson` partition holds 8 GPUs on 1
# node and never preempts. The stage resumes from its chunks, thus the wide and
# interruptible side is the right one, and it runs about 5x wider.
#
# The 3 numbers move together with the partition:
#   shards      more, so a task is short enough to backfill
#   parallel    more, because the partition is wide
#   shard_rows  fewer, because a chunk is the unit of loss on a preemption
PREEMPT=1
SHARDS=""
PARALLEL=""
SHARD_ROWS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --cases)      shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do CASES+=("$1"); shift; done ;;
    --judge)      JUDGE="$2"; shift 2 ;;
    --shards)     SHARDS="$2"; shift 2 ;;
    --parallel)   PARALLEL="$2"; shift 2 ;;
    --shard-rows) SHARD_ROWS="$2"; shift 2 ;;
    --preempt)    PREEMPT=1; shift ;;
    --pierson)    PREEMPT=0; shift ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[ ${#CASES[@]} -eq 0 ] && CASES=("${ALL_CASES[@]}")

if [ "${PREEMPT}" -eq 1 ]; then
  LAUNCHER="slurm_gpu_preempt"
  SHARDS="${SHARDS:-8}"
  PARALLEL="${PARALLEL:-24}"
  # About 11 minutes of work. A preemption loses at most 1 chunk.
  SHARD_ROWS="${SHARD_ROWS:-250}"
else
  LAUNCHER="slurm_gpu_1x"
  SHARDS="${SHARDS:-6}"
  PARALLEL="${PARALLEL:-4}"
  SHARD_ROWS="${SHARD_ROWS:-2000}"
fi

cd "${ROOT}"

# The gate. A corpus built from a run that moved is not reproducible.
echo "[ic] gate: the canonical registry"
"${PY}" scripts/register_canonical_runs.py verify --quick

paths=""
for c in "${CASES[@]}"; do
  p="${REG}/${c}__${JUDGE}/results.parquet"
  if [ ! -e "${p}" ]; then
    echo "error: no registered trace run at ${p}" >&2
    exit 1
  fi
  paths="${paths:+${paths},}${p}"
done

shard_list="$(seq -s, 0 $((SHARDS - 1)))"
STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="multirun/ic_corpus_${STAMP}"
LOG="${ROOT}/${SWEEP_DIR}/launch.log"
n_jobs=$(( ${#CASES[@]} * SHARDS ))

echo "[ic] judge    : ${JUDGE}"
echo "[ic] cases    : ${CASES[*]}"
echo "[ic] launcher : ${LAUNCHER}$([ "${PREEMPT}" -eq 1 ] && echo '  (preemptable, partition gpu)')"
echo "[ic] shards   : ${SHARDS} for each case, ${n_jobs} jobs, ${PARALLEL} at a time"
echo "[ic] chunk    : ${SHARD_ROWS} rows, the unit a preemption loses"
echo "[ic] sweep    : ${SWEEP_DIR}"

cmd=("${PY}" -m dagspaces.urbanpairvqa.cli -m pipeline=ic_extract
     "ic_extract.results_path=${paths}"
     "ic_extract.shard_count=${SHARDS}"
     "ic_extract.shard_index=${shard_list}"
     "ic_extract.shard_rows=${SHARD_ROWS}"
     "pipeline.graph.nodes.ic.launcher=${LAUNCHER}"
     "hydra.launcher.array_parallelism=${PARALLEL}")

if [ "${DRY}" -eq 1 ]; then
  echo
  echo "HYDRA_SWEEP_DIR=${SWEEP_DIR} ${cmd[*]}"
  exit 0
fi

mkdir -p "${ROOT}/${SWEEP_DIR}"
HYDRA_SWEEP_DIR="${SWEEP_DIR}" nohup "${cmd[@]}" > "${LOG}" 2>&1 &
echo "[ic] started pid=$! log=${LOG}"
cat <<TXT

To watch:
  squeue -u \$USER -o "%.10i %.20j %.9T %.11M %R"
  tail -f ${LOG}

When it lands:
  ${PY} scripts/merge_trace_extractions.py ${SWEEP_DIR}/... --schema ic
TXT
