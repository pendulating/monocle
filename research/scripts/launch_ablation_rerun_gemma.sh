#!/usr/bin/env bash
# Re-run the gemma-4-12b halves of the NO-CITY and TEXT-FIRST arms on the
# canonical `interleaved_labels` layout (decision of 2026-09-02), over rows cut
# out of the 1,000,000-pair table so that they join the canonical battery on
# `pair_id` at 100%.
#
#   million_nocity_gemma12b_il      5 shards for each case, 35 jobs
#   million_textfirst_gemma12b_il   5 shards for each case, 35 jobs
#
# The pair tables come from `scripts/subset_pair_tables.py`. This script does
# NOT run the prebuild: a prebuild over a subset table is refused on purpose.
#
# Usage:
#   bash research/scripts/launch_ablation_rerun_gemma.sh --dry-run
#   bash research/scripts/launch_ablation_rerun_gemma.sh --arm nocity
#   bash research/scripts/launch_ablation_rerun_gemma.sh
#
# When it lands:
#   python scripts/merge_pairwise_shards.py --sweep-dir <base>/nocity/runs \
#       --sweep million_nocity_gemma12b_il --out-dir multirun/nocity_merged_il
#   python scripts/merge_pairwise_shards.py --sweep-dir <base>/textfirst/runs \
#       --sweep million_textfirst_gemma12b_il --out-dir multirun/textfirst_merged_il

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${ROOT}/.venv-mllmsci-vllm025cu129/bin/python"

DRY=0
LAUNCHER="slurm_gpu_preempt"
ARMS=()
PARALLEL=40
TIMEOUT=480
PAIRS_DIR="${RERUN_PAIRS_DIR:-${ROOT}/multirun/pair_tables_100k_of_1m}"

while [ $# -gt 0 ]; do
  case "$1" in
    --arm)       ARMS+=("$2"); shift 2 ;;
    --parallel)  PARALLEL="$2"; shift 2 ;;
    --timeout)   TIMEOUT="$2"; shift 2 ;;
    --pairs-dir) PAIRS_DIR="$2"; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    -h|--help)   sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[ ${#ARMS[@]} -eq 0 ] && ARMS=(nocity textfirst)

cd "${ROOT}"
for c in subway_safety libraries schools road_quality parks_plazas restaurants street_photography; do
  [ -f "${PAIRS_DIR}/${c}_pairs.json" ] || { echo "no subset table for ${c} in ${PAIRS_DIR}; run scripts/subset_pair_tables.py" >&2; exit 1; }
  grep -q '"subset"' "${PAIRS_DIR}/${c}_pairs.json" || { echo "${PAIRS_DIR}/${c}_pairs.json is not a subset sidecar" >&2; exit 1; }
done

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="${ROOT}/multirun/ablation_rerun_il_${STAMP}"
mkdir -p "${BASE}"
echo "[rerun] pairs   : ${PAIRS_DIR}"
echo "[rerun] launcher: ${LAUNCHER}"
echo "[rerun] base    : ${BASE}"
echo

for arm in "${ARMS[@]}"; do
  case "${arm}" in
    nocity)    sweep=million_nocity_gemma12b_il ;;
    textfirst) sweep=million_textfirst_gemma12b_il ;;
    *) echo "unknown arm: ${arm} (use nocity or textfirst)" >&2; exit 1 ;;
  esac
  cmd=("${PY}" -m dagspaces.pairwisevqa.submit_shards
       "--sweep=${sweep}" "--base-dir=${BASE}/${arm}" "--launcher=${LAUNCHER}"
       "--parallel=${PARALLEL}" "--timeout-min=${TIMEOUT}" "--pairs-dir=${PAIRS_DIR}")
  [ "${DRY}" -eq 1 ] && cmd+=("--dry-run")
  echo "=== ${arm} (${sweep}) ==="
  "${cmd[@]}"
  echo
  # The sbatch shim refuses a burst of ssh connections. Two arrays are not a
  # burst, but a pause costs nothing.
  [ "${DRY}" -eq 0 ] && sleep 20
done

[ "${DRY}" -eq 1 ] && exit 0
cat <<TXT
Finished shards:
  ls -d ${BASE}/*/runs/*/outputs/pairwise/*_mvp_*.parquet 2>/dev/null | wc -l

Run the missing shards again (the wall is terminal):
  ${PY} -m dagspaces.pairwisevqa.submit_shards --sweep <sweep> \\
      --base-dir ${BASE}/<arm> --pairs-dir ${PAIRS_DIR} --only-missing --timeout-min ${TIMEOUT}
TXT
