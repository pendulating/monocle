#!/usr/bin/env bash
# F4 x2 round (D130): run a plan of extraction steps, 1 after another, inside
# 1 GPU allocation. QoS nissenbaum gives the group 1 GPU, and an array task
# that ends gives the GPU back to the queue. 1 allocation for the whole plan
# waits in the queue 1 time only.
#
# The plan is a text file, 1 step for each line:
#   <arm> <case> <shard> <shards> <base-dir>
# A line that starts with "#" is a comment. A line "run <command>" runs a
# CPU command in the project root (for example the flag step of the retry
# pass) every time the queue reaches it. A step writes
# <base-dir>/runs/f4_<case>__<shard:04d>/X2_DONE when it ends with exit 0;
# the queue skips a step with that file, thus a requeue continues the plan.
# A step that fails is logged, and the queue goes on to the next step.
#
# Submit from the project root:
#   sbatch --exclude=g3-compute-17 --export=ALL,F4X2_PLAN=research/scripts/f4/plans/x2_smoke.txt research/scripts/f4/x2_queue.sbatch
#
# Warning: g3-compute-17 has B200 GPUs (180 GB). DeepSeek-V4-Flash needs a
# B300; always pass --exclude=g3-compute-17.
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=nissenbaum
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=f4x2
#SBATCH --output=/share/ju/matt/mllmsci/logs/f4/%x-%j.out
set -uo pipefail
ROOT=/share/ju/matt/mllmsci
cd $ROOT
PLAN=${F4X2_PLAN:?set F4X2_PLAN to a plan file}

# ── the environment (as in research/scripts/f4/extract.sbatch) ─────────────────
export PATH=/share/ju/matt/bin:$HOME/.local/bin:$PATH
export TMPDIR=/scratch/$USER/tmp
export HF_HOME=/scratch/$USER/hf-home
F4X_CACHE=$ROOT/.cache/f4x-dsv4-b300
export VLLM_CACHE_ROOT=$F4X_CACHE/vllm-cache
export VLLM_CONFIG_ROOT=/scratch/$USER/vllm-config
export XDG_CACHE_HOME=/scratch/$USER/xdg-cache
export TRITON_CACHE_DIR=$F4X_CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$F4X_CACHE/inductor
export FLASHINFER_WORKSPACE_BASE=$F4X_CACHE/flashinfer
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR" logs/f4 "$F4X_CACHE"

echo "host=$(hostname -s) job=${SLURM_JOB_ID} plan=$PLAN commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
nvidia-smi -L

VENV=$ROOT/.venv-mllmsci-vllm025cu129
export MLLMSCI_DRIVER_VENV=$VENV
source scripts/activate_stage_venv.sh
PY="${MLLMSCI_STAGE_PYTHON:-$VENV/bin/python}"
_site="$(dirname "$PY")/../lib/python3.12/site-packages"
export FLASHINFER_EXTRA_CUDAFLAGS="-I$(readlink -f "$_site/nvidia/cublas/include")"
export FLASHINFER_EXTRA_CFLAGS="$FLASHINFER_EXTRA_CUDAFLAGS"
export F4X_PRELOAD_CUDART=1
export F4X_TOPK_FALLBACK=1
export F4X_HOOK_VERBOSE=1
export PYTHONPATH=$ROOT/research/scripts/f4/preload_cudart${PYTHONPATH:+:$PYTHONPATH}

while read -r ARM CASE SHARD SHARDS BASE; do
  [ -z "${ARM:-}" ] && continue
  case "$ARM" in \#*) continue ;; esac
  if [ "$ARM" = run ]; then
    CMD="$CASE $SHARD $SHARDS $BASE"
    echo "[x2] run $CMD $(date -u +%FT%TZ)"
    bash -c "$CMD" < /dev/null || echo "[x2] FAILED run (exit $?)"
    continue
  fi
  RUN=$BASE/runs/f4_${CASE}__$(printf %04d "$SHARD")
  # A retry arm with no copy suspect has no input file (x2_retry.py flag).
  if [ ! -f "data/f4_traces/extract_input/f4_${CASE}.parquet" ]; then
    echo "[x2] skip $ARM $CASE $SHARD/$SHARDS (no input file)"
    continue
  fi
  if [ -f "$RUN/X2_DONE" ]; then
    echo "[x2] skip $ARM $CASE $SHARD/$SHARDS (done)"
    continue
  fi
  echo "[x2] start $ARM $CASE $SHARD/$SHARDS -> $BASE $(date -u +%FT%TZ)"
  if "$PY" research/scripts/f4/extract_shard.py --arm "$ARM" --case "$CASE" --shard "$SHARD" --shards "$SHARDS" \
       --base-dir "$BASE" < /dev/null; then
    date -u +%FT%TZ > "$RUN/X2_DONE"
    echo "[x2] done $ARM $CASE $SHARD/$SHARDS $(date -u +%FT%TZ)"
  else
    echo "[x2] FAILED $ARM $CASE $SHARD/$SHARDS (exit $?) $(date -u +%FT%TZ)"
  fi
done < "$PLAN"
echo "[x2] plan finished $(date -u +%FT%TZ)"
