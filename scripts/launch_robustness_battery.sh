#!/usr/bin/env bash
# Launch the robustness battery on the canonical setup: 4 raters x 11 arms
# over 1,000 canonical pairs cut from the 1,000,000-pair table (tier A), or
# 2 raters x 2 wording arms over 25,000 pairs (tier B).
# Plan: vlm-narratives-docs/robustness-battery-rerun-plan.md
#
# Each arm is ONE change against the rater's own canonical base. The base is
# the sweep file robustness_1k_<rater>.yaml; the arm is a --set override.
#
# Usage:
#   bash scripts/launch_robustness_battery.sh --dry-run
#   bash scripts/launch_robustness_battery.sh --rater gemma12b --arm R0 --arm L2
#   bash scripts/launch_robustness_battery.sh                     # tier A, all
#   bash scripts/launch_robustness_battery.sh --tier b            # tier B
#   bash scripts/launch_robustness_battery.sh --smoke --rater e2b # fresh 20-pair draw
#
# The runs land in multirun/robustness_<tier>_<stamp>/<rater>/<arm>/runs/subway_safety__0000.
# The .hydra/config.yaml of a run is the proof of its arm; the dir name is the label.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv-mllmsci-vllm025cu129/bin/python"
DRY=0; SMOKE=0; TIER=a
LAUNCHER="slurm_gpu_preempt"
RATERS=(); ARMS=(); BASE=""
TIMEOUT_A=90; TIMEOUT_B=480
while [ $# -gt 0 ]; do
  case "$1" in
    --rater)   RATERS+=("$2"); shift 2 ;;
    --arm)     ARMS+=("$2"); shift 2 ;;
    --tier)    TIER="$2"; shift 2 ;;
    --base)    BASE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --smoke)   SMOKE=1; shift ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

# rater tag -> its layout arms. L1 = the other binding layout, L2 = prompt first.
# Both gemma raters need text BETWEEN the images (probe of 2026-09-04: e2b gives
# Same on 100% of rows under the default layout), thus e2b takes the 12b arms.
declare -A L1=( [gemma12b]=images_then_labels [qwen9b]=interleaved_labels [qwen2b]=interleaved_labels [e2b]=images_then_labels )
declare -A L2=( [gemma12b]=text_first_interleaved [qwen9b]=text_first [qwen2b]=text_first [e2b]=text_first_interleaved )
# F1 = half the image budget. Qwen: a quarter of the pixels (half the side).
# Gemma: max_soft_tokens 280 -> 140 (vLLM takes 70/140/280/560/1120 only).
declare -A F1=( [gemma12b]="++model.engine_kwargs.mm_processor_kwargs.max_soft_tokens=140"
                [e2b]="++model.engine_kwargs.mm_processor_kwargs.max_soft_tokens=140"
                [qwen9b]="model.engine_kwargs.mm_processor_kwargs.max_pixels=262144"
                [qwen2b]="model.engine_kwargs.mm_processor_kwargs.max_pixels=262144" )

arm_override() {  # rater arm -> the --set string ("" for R0)
  local r="$1" a="$2"
  case "$a" in
    R0) echo "" ;;
    W1) echo "prompt=pairwise_subway_safety_paraphrase" ;;
    W2) echo "prompt=pairwise_subway_safety_persona" ;;
    W3) echo "prompt=pairwise_subway_safety_enumrev" ;;
    W4) echo "prompt=pairwise_subway_safety_flipped" ;;
    W5) echo "prompt=pairwise_subway_safety_nocity" ;;
    F1) echo "${F1[$r]}" ;;
    F2) echo "prompt.structured_output.enabled=false" ;;
    F3) echo "sampling_params_vqa.temperature=0.6" ;;
    L1) echo "++prompt.image_layout=${L1[$r]}" ;;
    L2) echo "++prompt.image_layout=${L2[$r]}" ;;
    *) echo "unknown arm: $a" >&2; exit 1 ;;
  esac
}

if [ "${TIER}" = "a" ]; then
  [ ${#RATERS[@]} -eq 0 ] && RATERS=(gemma12b qwen9b qwen2b e2b)
  [ ${#ARMS[@]} -eq 0 ] && ARMS=(R0 W1 W2 W3 W4 W5 F1 F2 F3 L1 L2)
  PAIRS_DIR="${ROOT}/multirun/pair_tables_1k_of_1m"; SHARDS=1; TIMEOUT=${TIMEOUT_A}; EXTRA=()
else
  [ ${#RATERS[@]} -eq 0 ] && RATERS=(gemma12b qwen9b)
  [ ${#ARMS[@]} -eq 0 ] && ARMS=(W1 W2)
  PAIRS_DIR="${ROOT}/multirun/pair_tables_25k_of_1m"; SHARDS=2; TIMEOUT=${TIMEOUT_B}
  EXTRA=(--set pair_sampler.max_pairs=25000)
fi
if [ "${SMOKE}" -eq 1 ]; then
  # A fresh 20-pair draw, no pair table: the cut refuses any other max_pairs.
  PAIRS_DIR=""; EXTRA=(--set pair_sampler.max_pairs=20); TIMEOUT=60; TIER="smoke"
fi
[ -n "${PAIRS_DIR}" ] && [ ! -f "${PAIRS_DIR}/subway_safety_pairs.json" ] && { echo "no pair table in ${PAIRS_DIR}" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
[ -z "${BASE}" ] && BASE="${ROOT}/multirun/robustness_${TIER}_${STAMP}"
mkdir -p "${BASE}"
echo "[battery] tier=${TIER} base=${BASE} pairs=${PAIRS_DIR:-fresh draw} shards=${SHARDS} timeout=${TIMEOUT}"
n=0
for r in "${RATERS[@]}"; do
  for a in "${ARMS[@]}"; do
    ov="$(arm_override "$r" "$a")"
    cmd=("${PY}" -m dagspaces.pairwisevqa.submit_shards "--sweep=robustness_1k_${r}"
         "--base-dir=${BASE}/${r}/${a}" "--launcher=${LAUNCHER}" "--parallel=${SHARDS}"
         "--shards=${SHARDS}" "--timeout-min=${TIMEOUT}")
    [ -n "${PAIRS_DIR}" ] && cmd+=("--pairs-dir=${PAIRS_DIR}")
    [ -n "${ov}" ] && cmd+=(--set "${ov}")
    cmd+=("${EXTRA[@]}")
    [ "${DRY}" -eq 1 ] && cmd+=(--dry-run)
    echo "=== ${r} ${a} : ${ov:-<base>}"
    "${cmd[@]}" | grep -E 'jobs|array|composed|error|Error|→' || true
    echo "${r}\t${a}\t${ov}" >> "${BASE}/arms.tsv"
    n=$((n+1))
    # The sbatch ssh shim refuses a burst of connections: pause between submits.
    [ "${DRY}" -eq 0 ] && sleep 20
  done
done
echo "[battery] ${n} submits. Finished outputs:"
echo "  ls ${BASE}/*/*/runs/*/outputs/pairwise/subway_safety_mvp_*.parquet | wc -l"
