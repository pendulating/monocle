#!/usr/bin/env bash
# Launch the 100,000-pair MATCHED-CONTROL battery, gemma-4-12b only.
#
# The same 7 cases, the CANONICAL prompts, abstention ON, the required
# `images_then_labels` layout, and the SAME 100,000 pairs as the no-city and
# text-first batteries. Nothing is ablated. This run is the control those 2
# arms lacked.
#
#   million_matched_gemma12b   5 shards for each case, 35 jobs
#
# ┌─ WHY THE 1,000,000-PAIR CONTROL DID NOT SERVE ─────────────────────────────┐
# │ `million_proxy_gemma12b` ran `interleaved_labels`, and it shares only      │
# │ 91.7% of the `pair_id` rows of the 100,000-pair arms. Thus a gemma delta   │
# │ of `nocity` or `textfirst` held the named change AND the layout, on 92%    │
# │ of the pairs. This battery reads the SAME `pair_tables_100k` directory as  │
# │ those 2 arms, thus it joins them pair for pair, on `pair_id`.              │
# │                                                                            │
# │ qwen3.5-9b needs no matched control: that rater ran 1 layout in both the   │
# │ arms and the 1,000,000-pair control.                                       │
# └────────────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   bash research/scripts/launch_matched_battery.sh --dry-run
#   bash research/scripts/launch_matched_battery.sh --smoke        # 1 case, 2 shards
#   bash research/scripts/launch_matched_battery.sh
#
# When it lands:
#   python scripts/merge_pairwise_shards.py --sweep-dir <base>/gemma/runs \
#       --sweep million_matched_gemma12b --out-dir multirun/matched_merged
#
# See `research/docs/prompt-ablations.md`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${ROOT}/.venv-mllmsci-vllm025cu129/bin/python"

DRY=0
SMOKE=0
LAUNCHER="slurm_gpu_preempt"
# 35 jobs in all, thus a cap above that costs nothing.
P_GEMMA=40
# A gemma shard of 22,000 rows needs about 100 minutes. The no-city battery
# asked for 480 and lost none to the wall. Keep it: the wall is terminal.
T_GEMMA=480
PREBUILD=1
# The SAME directory as the no-city and text-first batteries, on purpose. The
# prebuild keeps a table whose fingerprint already matches, thus this battery
# reads the tables those 2 arms read.
PAIRS_DIR="${MATCHED_PAIRS_DIR:-${ROOT}/multirun/pair_tables_100k}"

while [ $# -gt 0 ]; do
  case "$1" in
    --parallel)       P_GEMMA="$2"; shift 2 ;;
    --timeout)        T_GEMMA="$2"; shift 2 ;;
    --pierson)        LAUNCHER="slurm_gpu_1x"; shift ;;
    --pairs-dir)      PAIRS_DIR="$2"; shift 2 ;;
    --no-prebuild)    PREBUILD=0; shift ;;
    --smoke)          SMOKE=1; shift ;;
    --dry-run)        DRY=1; shift ;;
    -h|--help)        sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

cd "${ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="${ROOT}/multirun/matched_battery_${STAMP}"
mkdir -p "${BASE}"

# A smoke run overrides max_pairs, thus the prebuilt tables do not match it.
[ "${SMOKE}" -eq 1 ] && PREBUILD=0

if [ "${PREBUILD}" -eq 1 ]; then
  echo "[matched] pairs   : ${PAIRS_DIR}"
  # Idempotent. Every table must report `kept`: a `built` line means the draw
  # does NOT match the no-city and text-first arms, and the run must not go in.
  "${PY}" scripts/prebuild_pair_tables.py \
      --sweep million_matched_gemma12b --out-dir "${PAIRS_DIR}" \
      | grep -E "^\[prebuild\] (built|kept|done|FAILED)"
  echo
fi

echo "[matched] launcher: ${LAUNCHER}"
echo "[matched] base    : ${BASE}"
[ "${SMOKE}" -eq 1 ] && echo "[matched] SMOKE   : 1 case, 2 shards, 2,000 pairs"
echo

cmd=("${PY}" -m dagspaces.pairwisevqa.submit_shards
     "--sweep=million_matched_gemma12b"
     "--base-dir=${BASE}/gemma"
     "--launcher=${LAUNCHER}"
     "--parallel=${P_GEMMA}"
     "--timeout-min=${T_GEMMA}")
[ "${PREBUILD}" -eq 1 ] && cmd+=("--pairs-dir=${PAIRS_DIR}")
if [ "${SMOKE}" -eq 1 ]; then
  cmd+=("--cases" "pairwise_schools_mvp_matched" "--shards" "2"
        "--set" "pair_sampler.max_pairs=2000")
fi
[ "${DRY}" -eq 1 ] && cmd+=("--dry-run")

echo "=== gemma (million_matched_gemma12b) ==="
"${cmd[@]}"
echo

[ "${DRY}" -eq 1 ] && exit 0

cat <<TXT
To watch:
  squeue -u \$USER -r -h -t RUNNING -n matt-PAIRVQA-shard | wc -l

Finished shards:
  ls -d ${BASE}/gemma/runs/*/outputs/pairwise/*_mvp_*.parquet 2>/dev/null | wc -l

Run the missing shards again (the wall is terminal, SLURM does not requeue it):
  ${PY} -m dagspaces.pairwisevqa.submit_shards --sweep million_matched_gemma12b \\
      --base-dir ${BASE}/gemma --pairs-dir ${PAIRS_DIR} --only-missing --timeout-min ${T_GEMMA}

When it lands:
  ${PY} scripts/merge_pairwise_shards.py --sweep-dir ${BASE}/gemma/runs \\
      --sweep million_matched_gemma12b --out-dir ${ROOT}/multirun/matched_merged
TXT
