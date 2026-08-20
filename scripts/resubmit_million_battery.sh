#!/usr/bin/env bash
# Run the shards of the 1,000,000-pair battery that hold no final parquet.
#
# ┌─ WHY A SHARD NEEDS A SECOND JOB ───────────────────────────────────────────┐
# │ `slurm_gpu_preempt` gives 180 minutes. A shard of this battery needs 3.2 to │
# │ 6.4 hours, thus most shards hit the wall and end in TIMEOUT. SLURM does NOT │
# │ requeue a TIMEOUT: `--requeue` covers a preemption and a node failure only. │
# │                                                                            │
# │ The work is not lost. Each shard keeps its resume chunks, thus a new job    │
# │ continues from where the first one stopped. This script asks for a longer   │
# │ walltime and submits only the shards that are not finished.                 │
# └────────────────────────────────────────────────────────────────────────────┘
#
# Measured on 2026-08-19 from the partial shards of the first pass:
#
#   qwen   3.2 to 5.3 hours for each shard  →  360 minutes
#   gemma  3.2 to 6.4 hours for each shard  →  480 minutes (its shards are 2x)
#
# Warning: cancel the old array FIRST. A shard that runs now and a shard that
# this script submits write to the SAME directory. The cancel costs 1 chunk for
# each running job, and no more, because a chunk lands with an atomic rename.
#
# Usage:
#   bash scripts/resubmit_million_battery.sh --dry-run
#   bash scripts/resubmit_million_battery.sh --cancel-only
#   bash scripts/resubmit_million_battery.sh          # cancel, then submit

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv-mllmsci-vllm025cu129/bin/python"
BASE="${MILLION_BASE:-${ROOT}/multirun/million_battery_20260818_173049}"
PAIRS="${MILLION_PAIRS_DIR:-${ROOT}/multirun/pair_tables_1m}"
OLD_ARRAYS="${MILLION_OLD_ARRAYS:-189418 190121}"

DRY=0
CANCEL_ONLY=0
P_QWEN=160
P_GEMMA=80
T_QWEN=360
T_GEMMA=480

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)      DRY=1; shift ;;
    --cancel-only)  CANCEL_ONLY=1; shift ;;
    --base)         BASE="$2"; shift 2 ;;
    -h|--help)      sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

cd "${ROOT}"
[ -d "${BASE}" ] || { echo "no base dir at ${BASE}" >&2; exit 1; }

echo "[resubmit] base   : ${BASE}"
echo "[resubmit] pairs  : ${PAIRS}"
echo "[resubmit] old    : ${OLD_ARRAYS}"
echo

# ---- 1. Stop the old array -------------------------------------------------
for a in ${OLD_ARRAYS}; do
  n=$(squeue -h -j "${a}" -o "%i" 2>/dev/null | wc -l || true)
  echo "[resubmit] array ${a} holds ${n} tasks that wait or run"
  if [ "${DRY}" -eq 0 ]; then
    scancel "${a}" 2>/dev/null || true
  fi
done
if [ "${DRY}" -eq 0 ]; then
  echo "[resubmit] cancelled. Waiting for the tasks to leave the queue."
  for _ in $(seq 1 60); do
    left=0
    for a in ${OLD_ARRAYS}; do
      left=$((left + $(squeue -h -j "${a}" -o "%i" 2>/dev/null | wc -l || echo 0)))
    done
    [ "${left}" -eq 0 ] && break
    sleep 5
  done
  echo "[resubmit] the queue is clear of the old arrays"
fi
echo
[ "${CANCEL_ONLY}" -eq 1 ] && exit 0

# ---- 2. Submit what is missing ---------------------------------------------
submit_one() {
  local rater="$1" sweep="$2" parallel="$3" timeout="$4"
  local -a cmd=("${PY}" -m dagspaces.urbanpairvqa.submit_shards
                "--sweep=${sweep}"
                "--base-dir=${BASE}/${rater}"
                "--pairs-dir=${PAIRS}"
                "--launcher=slurm_gpu_preempt"
                "--parallel=${parallel}"
                "--timeout-min=${timeout}"
                "--only-missing")
  [ "${DRY}" -eq 1 ] && cmd+=("--dry-run")
  echo "=== ${rater} (${sweep}) ==="
  "${cmd[@]}"
  echo
}

submit_one qwen  million_proxy_qwen9b   "${P_QWEN}"  "${T_QWEN}"
submit_one gemma million_proxy_gemma12b "${P_GEMMA}" "${T_GEMMA}"

[ "${DRY}" -eq 1 ] && exit 0

cat <<TXT
To watch:
  squeue -u \$USER -h -t RUNNING -n matt-PAIRVQA-shard | wc -l

Shards still missing (run again any time, it is idempotent):
  ${PY} -m dagspaces.urbanpairvqa.submit_shards --sweep million_proxy_qwen9b \\
      --base-dir ${BASE}/qwen --pairs-dir ${PAIRS} --only-missing --dry-run
TXT
