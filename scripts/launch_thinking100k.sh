#!/usr/bin/env bash
# Launch the 100,000-pair thinking runs of both raters on the `gpu` partition.
#
#   bash scripts/launch_thinking100k.sh [--dry-run] [--rater gemma|qwen] [--parallel N] [--timeout MIN]
#
# One SLURM array for each rater, submitted by `submit_shards` from the login
# node with no monitor jobs. The sweeps are `thinking100k_gemma12b` (7 cases
# x 32 shards) and `thinking100k_qwen9b` (7 cases x 40 shards). Both read the
# 100,000-pair cut of the 1,000,000-pair table and the image pack, thus no
# shard touches /share/ju.
#
# A shard that hits the walltime keeps its resume chunks. Run the missing
# shards again on the SAME base dir:
#   python -m dagspaces.urbanpairvqa.submit_shards --sweep <sweep> \
#       --base-dir <base>/<rater> --pairs-dir <pairs> --only-missing \
#       --timeout-min 480 --set runtime.image_pack=<pack>
#
# See dagspaces/urbanpairvqa/conf/sweep/thinking100k_gemma12b.yaml.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv-mllmsci-vllm025cu129/bin/python"
DRY=0
RATERS=()
PARALLEL=128
TIMEOUT=480
LAUNCHER="slurm_gpu_preempt"
PAIRS_DIR="${ROOT}/multirun/pair_tables_100k_of_1m"
IMAGE_PACK="${ROOT}/multirun/pair_tables_1m/image_pack"

while [ $# -gt 0 ]; do
  case "$1" in
    --rater)    RATERS+=("$2"); shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --timeout)  TIMEOUT="$2"; shift 2 ;;
    --dry-run)  DRY=1; shift ;;
    -h|--help)  sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[ ${#RATERS[@]} -eq 0 ] && RATERS=(gemma qwen)

cd "${ROOT}"
for c in subway_safety libraries schools road_quality parks_plazas restaurants street_photography; do
  [ -f "${PAIRS_DIR}/${c}_pairs.json" ] || { echo "no cut table for ${c} in ${PAIRS_DIR}" >&2; exit 1; }
  grep -q '"subset"' "${PAIRS_DIR}/${c}_pairs.json" || { echo "${PAIRS_DIR}/${c}_pairs.json is not a subset sidecar" >&2; exit 1; }
done
[ -f "${IMAGE_PACK}/manifest.json" ] || { echo "no image pack manifest at ${IMAGE_PACK}" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="${ROOT}/multirun/thinking100k_${STAMP}"
mkdir -p "${BASE}"
echo "[thinking100k] pairs    : ${PAIRS_DIR}"
echo "[thinking100k] pack     : ${IMAGE_PACK}"
echo "[thinking100k] launcher : ${LAUNCHER}, walltime ${TIMEOUT} min, throttle ${PARALLEL}"
echo "[thinking100k] base     : ${BASE}"
echo

for rater in "${RATERS[@]}"; do
  case "${rater}" in
    gemma) sweep=thinking100k_gemma12b ;;
    qwen)  sweep=thinking100k_qwen9b ;;
    *) echo "unknown rater: ${rater} (use gemma or qwen)" >&2; exit 1 ;;
  esac
  cmd=("${PY}" -m dagspaces.urbanpairvqa.submit_shards
       "--sweep=${sweep}" "--base-dir=${BASE}/${rater}" "--launcher=${LAUNCHER}"
       "--parallel=${PARALLEL}" "--timeout-min=${TIMEOUT}" "--pairs-dir=${PAIRS_DIR}"
       "--set=runtime.image_pack=${IMAGE_PACK}")
  [ "${DRY}" -eq 1 ] && cmd+=("--dry-run")
  echo "=== ${rater} (${sweep}) ==="
  "${cmd[@]}" 2>&1 | tee "${BASE}/${rater}_submit.log" | grep -v "^\[submit\] \+#"
  echo
  [ "${DRY}" -eq 0 ] && sleep 30
done
[ "${DRY}" -eq 1 ] && exit 0
cat <<TXT
Finished shards:
  ls ${BASE}/*/runs/*/outputs/pairwise/*_mvp_*.parquet 2>/dev/null | wc -l   # of 504
Run the missing shards again (the walltime is terminal, a preemption is not):
  ${PY} -m dagspaces.urbanpairvqa.submit_shards --sweep <sweep> \\
      --base-dir ${BASE}/<rater> --pairs-dir ${PAIRS_DIR} --only-missing \\
      --timeout-min ${TIMEOUT} --set runtime.image_pack=${IMAGE_PACK}
TXT
