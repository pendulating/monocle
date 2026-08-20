#!/usr/bin/env bash
# Launch the "looks like" re-run of the 7-case pairwise battery.
#
# On 2026-08-14 the canonical prompts moved to "looks like" wording, thus every
# proxy run and every trace run must run again. This script starts the 4 sweeps
# that do it:
#
#   | Sweep                    | What it gives          | Pairs  | GPU-hours |
#   |--------------------------|------------------------|--------|-----------|
#   | looks_proxy_qwen9b       | validation by proxy    | 100k   | about 115 |
#   | looks_proxy_gemma12b     | validation by proxy    | 100k   | about 54  |
#   | looks_thinking_qwen9b    | reasoning traces       | 10k    | about 85  |
#   | looks_thinking_gemma12b  | reasoning traces       | 10k    | about 63  |
#
# Each sweep holds 7 jobs (1 for each case) and runs 4 of them at a time.
# klara holds 8 GPUs and 56 CPUs, and a stage job takes 1 GPU and 8 CPUs, thus
# 7 stage jobs run together and the rest wait in the queue. All 4 sweeps
# together need about 2 days of wall clock.
#
# Usage:
#   bash scripts/launch_looks_battery.sh --smoke     # 16 pairs, 1 case, each sweep
#   bash scripts/launch_looks_battery.sh --dry-run   # print the commands only
#   bash scripts/launch_looks_battery.sh             # start all 4 sweeps
#   bash scripts/launch_looks_battery.sh looks_proxy_qwen9b looks_proxy_gemma12b
#
# A launch process must live until its jobs end, because the submitit launcher
# waits for the results. This script starts each launch with `nohup`, thus the
# battery survives a lost terminal.
#
# Warning: run the smoke test first when the venv, the node, or a model config
# changed. A bad launch wastes a night of GPUs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv-mllmsci-vllm025cu129/bin/python"

# The canonical venv. It is the only 1 with a node-local /scratch mirror, so do
# not point this at another venv: the stage then reads torch and vLLM over NFS,
# which costs about 13 minutes for each process spawn.
if [ ! -x "${PY}" ]; then
  echo "error: no interpreter at ${PY}" >&2
  exit 1
fi

ALL_SWEEPS=(
  looks_proxy_qwen9b
  looks_proxy_gemma12b
  looks_thinking_qwen9b
  looks_thinking_gemma12b
)

SMOKE=0
DRY=0
SWEEPS=()
for arg in "$@"; do
  case "${arg}" in
    --smoke)   SMOKE=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)         SWEEPS+=("${arg}") ;;
  esac
done
if [ ${#SWEEPS[@]} -eq 0 ]; then
  SWEEPS=("${ALL_SWEEPS[@]}")
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BATTERY_DIR="multirun/looks_battery_${STAMP}"
LOG_DIR="${ROOT}/${BATTERY_DIR}/launch_logs"
mkdir -p "${LOG_DIR}"

echo "root       : ${ROOT}"
echo "python     : ${PY}"
echo "battery dir: ${BATTERY_DIR}"
echo "sweeps     : ${SWEEPS[*]}"
echo "mode       : $([ "${SMOKE}" -eq 1 ] && echo smoke || echo full)"
echo

cd "${ROOT}"

for sweep in "${SWEEPS[@]}"; do
  conf="${ROOT}/dagspaces/urbanpairvqa/conf/sweep/${sweep}.yaml"
  if [ ! -f "${conf}" ]; then
    echo "error: no sweep config at ${conf}" >&2
    exit 1
  fi

  extra=()
  if [ "${SMOKE}" -eq 1 ]; then
    # 1 case, 16 pairs. Schools is the check case: it is unit mode, it has a
    # proxy, and it holds the newest prompt.
    extra=(pipeline=pairwise_schools_mvp pair_sampler.max_pairs=16)
  fi

  cmd=("${PY}" -m dagspaces.urbanpairvqa.cli --multirun "+sweep=${sweep}" "${extra[@]}")

  if [ "${DRY}" -eq 1 ]; then
    echo "HYDRA_SWEEP_DIR=${BATTERY_DIR}/${sweep} ${cmd[*]}"
    continue
  fi

  if [ "${SMOKE}" -eq 1 ]; then
    # The smoke test runs in front of you, 1 sweep after the other, so you read
    # the label distribution before the next one starts.
    echo "=== smoke: ${sweep}"
    HYDRA_SWEEP_DIR="${BATTERY_DIR}/${sweep}" "${cmd[@]}" 2>&1 \
      | tee "${LOG_DIR}/${sweep}.smoke.out"
  else
    log="${LOG_DIR}/${sweep}.out"
    HYDRA_SWEEP_DIR="${BATTERY_DIR}/${sweep}" nohup "${cmd[@]}" > "${log}" 2>&1 &
    echo "started ${sweep}  pid=$!  log=${log}"
    # Stagger the launches. 2 stage jobs that start in the same second write
    # the same `pairs.parquet` path, which is cosmetic but confuses a reader of
    # the run directory.
    sleep 20
  fi
done

if [ "${DRY}" -eq 0 ] && [ "${SMOKE}" -eq 0 ]; then
  cat <<EOF

All launches run in the background. To watch them:
  squeue -u \$USER -o "%.10i %.24j %.8T %.11M %R"
  tail -f ${LOG_DIR}/*.out

To find the stage logs of a case:
  ls ${BATTERY_DIR}/*/*/*/*/.slurm_jobs/pairwise/

To slow a running array down, do not cancel it:
  scontrol update JobId=<arrayid> ArrayTaskThrottle=<n>
EOF
fi
