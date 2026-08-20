#!/usr/bin/env bash
# Launch the 1,000,000-pair validation-by-proxy battery.
#
# 2 sweeps, 1 for each rater, 7 cases in each, and every case cut into shards
# that run on the PREEMPTABLE `gpu` partition.
#
#   million_proxy_qwen9b    92 shards for each case, 644 jobs, ~1,110 GPU-hours
#   million_proxy_gemma12b  46 shards for each case, 322 jobs, ~300 GPU-hours
#
# The prompt, the seed and the sampling match the "looks like" battery of
# 2026-08-14 exactly. Only the number of pairs changes, 100,000 → 1,000,000.
#
# ┌─ NO MONITOR JOB ───────────────────────────────────────────────────────────┐
# │ A pairwise sweep is a graph of 1 node with no dependency. The normal Hydra │
# │ multirun path puts a CPU monitor job above each stage job, and the monitor │
# │ only blocks on `job.result()`. For 966 shards of about 2 hours that holds  │
# │ about 1,900 CPU-hours of lisbeth for nothing.                             │
# │                                                                            │
# │ `dagspaces.urbanpairvqa.submit_shards` composes every shard config in 1    │
# │ process on the login node, submits them as 1 SLURM ARRAY, and exits. The   │
# │ GPU job writes its own `pipeline_manifest.json`.                          │
# └────────────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   bash scripts/launch_million_battery.sh --dry-run
#   bash scripts/launch_million_battery.sh --smoke       # 1 case, 2 shards
#   bash scripts/launch_million_battery.sh               # both raters
#   bash scripts/launch_million_battery.sh --rater qwen
#   bash scripts/launch_million_battery.sh --parallel-qwen 160 --parallel-gemma 80
#   bash scripts/launch_million_battery.sh --no-prebuild   # draw pairs in each job
#
# When it lands:
#   python scripts/merge_pairwise_shards.py --sweep-dir <base>/qwen/runs --dry-run
#
# See `vlm-narratives-docs/million-pair-battery.md`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv-mllmsci-vllm025cu129/bin/python"

DRY=0
SMOKE=0
LAUNCHER="slurm_gpu_preempt"
RATERS=()
# The work splits about 2 to 1 between the raters, thus so does the width.
# SLURM starts only what is free, so a high cap costs nothing when the
# partition is busy. 62 nodes match the constraint.
P_QWEN=160
P_GEMMA=80
# Draw the pair table of each case once, before the array goes in. A job that
# draws 1,100,000 pairs itself spends about 195 seconds on it, and a shard that
# reads the parquet spends about 1 second. Over 966 jobs that is about 51
# GPU-hours, and a preemption makes a job pay it again. The table does not
# depend on the model, thus 1 file for each case serves BOTH raters.
PREBUILD=1
PAIRS_DIR="${MILLION_PAIRS_DIR:-${ROOT}/multirun/pair_tables_1m}"

while [ $# -gt 0 ]; do
  case "$1" in
    --rater)          RATERS+=("$2"); shift 2 ;;
    --parallel-qwen)  P_QWEN="$2"; shift 2 ;;
    --parallel-gemma) P_GEMMA="$2"; shift 2 ;;
    --parallel)       P_QWEN="$2"; P_GEMMA="$2"; shift 2 ;;
    --pierson)        LAUNCHER="slurm_gpu_1x"; shift ;;
    --pairs-dir)      PAIRS_DIR="$2"; shift 2 ;;
    --no-prebuild)    PREBUILD=0; shift ;;
    --smoke)          SMOKE=1; shift ;;
    --dry-run)        DRY=1; shift ;;
    -h|--help)        sed -n '2,35p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[ ${#RATERS[@]} -eq 0 ] && RATERS=(qwen gemma)

cd "${ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="${ROOT}/multirun/million_battery_${STAMP}"
mkdir -p "${BASE}"

# A smoke run overrides max_pairs, thus the prebuilt tables do not match it.
[ "${SMOKE}" -eq 1 ] && PREBUILD=0

if [ "${PREBUILD}" -eq 1 ]; then
  echo "[million] pairs   : ${PAIRS_DIR}"
  # Idempotent: a table whose sidecar already matches is kept, not drawn again.
  "${PY}" scripts/prebuild_pair_tables.py \
      --sweep million_proxy_qwen9b --out-dir "${PAIRS_DIR}" \
      | grep -E "^\[prebuild\] (built|kept|done|FAILED)"
  echo
fi

echo "[million] launcher: ${LAUNCHER}"
echo "[million] base    : ${BASE}"
[ "${SMOKE}" -eq 1 ] && echo "[million] SMOKE   : 1 case, 2 shards, 2,000 pairs"
echo

launch_one() {
  local rater="$1" sweep="$2" parallel="$3"
  local -a cmd=("${PY}" -m dagspaces.urbanpairvqa.submit_shards
                "--sweep=${sweep}"
                "--base-dir=${BASE}/${rater}"
                "--launcher=${LAUNCHER}"
                "--parallel=${parallel}")

  [ "${PREBUILD}" -eq 1 ] && cmd+=("--pairs-dir=${PAIRS_DIR}")

  if [ "${SMOKE}" -eq 1 ]; then
    cmd+=("--cases" "pairwise_schools_mvp" "--shards" "2"
          "--set" "pair_sampler.max_pairs=2000")
  fi
  [ "${DRY}" -eq 1 ] && cmd+=("--dry-run")

  echo "=== ${rater} (${sweep}) ==="
  "${cmd[@]}"
  echo
}

for rater in "${RATERS[@]}"; do
  case "${rater}" in
    qwen)  launch_one qwen  million_proxy_qwen9b   "${P_QWEN}" ;;
    gemma) launch_one gemma million_proxy_gemma12b "${P_GEMMA}" ;;
    *) echo "unknown rater: ${rater} (use qwen or gemma)" >&2; exit 1 ;;
  esac
done

[ "${DRY}" -eq 1 ] && exit 0

cat <<TXT
To watch:
  squeue -u \$USER -h -t RUNNING -n matt-PAIRVQA-shard | wc -l
  squeue -u \$USER -o "%.12i %.20j %.9T %.11M %R" | head -30

Preemptions leave a trace in the stage logs:
  grep -l "RESTARTS=[1-9]" ${BASE}/*/.slurm_jobs/*log.out | wc -l

Finished shards:
  ls -d ${BASE}/*/runs/*/outputs/pairwise/*_mvp_*.parquet 2>/dev/null | wc -l

When it lands:
  ${PY} scripts/merge_pairwise_shards.py --sweep-dir ${BASE}/qwen/runs --dry-run
  ${PY} scripts/merge_pairwise_shards.py --sweep-dir ${BASE}/qwen/runs \\
      --out-dir ${BASE}/merged_qwen
TXT
