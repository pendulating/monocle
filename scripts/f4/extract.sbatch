#!/usr/bin/env bash
# F4: span extraction with DeepSeek-V4-Flash-0731 on 1 B300 (d-0012, D93-D96).
#
# 1 array task is 1 shard. The first F4X_SHARDS tasks are the restaurant
# shards, the next F4X_SHARDS are the caption shards. QoS nissenbaum gives the
# group 1 GPU, thus the array runs 1 task at a time (%1).
#
# Submit from the project root (4 shards for each case gives 8 tasks):
#   sbatch --array=0-7%1 scripts/f4/extract.sbatch
# Smoke test, 50 texts of shard 0 of the restaurant case:
#   sbatch --array=0 --export=ALL,F4X_TAG=smoke,F4X_SET="runtime.sample_n=50" scripts/f4/extract.sbatch
#
# The caption job (D101 to D104), 1 shard of 6,000 answers:
#   sbatch --array=0 --export=ALL,F4X_ARM=captions,F4X_SHARDS=1 scripts/f4/extract.sbatch
#
# The retry pass (D105): F4X_ARM=captions_retry, the same command.
#
# The descriptor job (D109 to D112) has the 2 cases of the cues arm:
#   sbatch --array=0-7%1 --export=ALL,F4X_ARM=descriptors scripts/f4/extract.sbatch
# Its smoke test reads 1 named case in 1 shard:
#   sbatch --array=0 --export=ALL,F4X_ARM=descriptors,F4X_CASE=descsmoke,F4X_SHARDS=1,F4X_TAG=smoke scripts/f4/extract.sbatch
#
# The descriptor retry pass (D121): F4X_ARM=descriptors_retry,F4X_SHARDS=1.
#
# The street descriptors (D120, D122), 1 named case in 4 shards:
#   sbatch --array=0-3%1 --export=ALL,F4X_ARM=descriptors_street,F4X_CASE=street scripts/f4/extract.sbatch
#
# The street cue spans (D125), the smoke and then the full run:
#   sbatch --array=0 --export=ALL,F4X_ARM=cues_street,F4X_CASE=streetsmoke,F4X_SHARDS=1,F4X_TAG=smoke scripts/f4/extract.sbatch
#   sbatch --array=0-3%1 --export=ALL,F4X_ARM=cues_street,F4X_CASE=street scripts/f4/extract.sbatch
#
# Optional env: F4X_ARM (cues, cues_street, captions, captions_retry,
# descriptors, descriptors_retry, or descriptors_street;
# default cues), F4X_CASE (1 case; the task id is then the shard),
# F4X_SHARDS (default 4), F4X_TAG (a suffix for the base dir),
# F4X_SET (more Hydra overrides, separated by spaces).
#
# A requeue skips each chunk of 250 rows that exists. The weights load from
# NFS (/share/ju): 167 GB is too large for the free space that /scratch
# usually has.
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
#SBATCH --job-name=f4x
#SBATCH --output=/share/ju/matt/mllmsci/logs/f4/%x-%A_%a.out
set -euo pipefail
ROOT=/share/ju/matt/mllmsci
cd $ROOT

SHARDS=${F4X_SHARDS:-4}
TASK=${SLURM_ARRAY_TASK_ID:-0}
ARM=${F4X_ARM:-cues}
if [ -n "${F4X_CASE:-}" ]; then
  # 1 named case; the task id is the shard.
  CASE=$F4X_CASE; SHARD=$TASK
elif [ "$ARM" = cues ] || [ "$ARM" = descriptors ]; then
  if [ "$TASK" -lt "$SHARDS" ]; then CASE=restaurant; SHARD=$TASK; else CASE=caption; SHARD=$((TASK - SHARDS)); fi
else
  # captions or captions_retry: 1 case with the name of the arm; the task id
  # is the shard.
  CASE=$ARM; SHARD=$TASK
fi
if [ "$ARM" = cues ]; then
  BASE=$ROOT/multirun/f4_extract_dsv4${F4X_TAG:+_$F4X_TAG}
else
  BASE=$ROOT/multirun/f4_${ARM}_dsv4${F4X_TAG:+_$F4X_TAG}
fi
read -r -a SETS <<< "${F4X_SET:-}"

# ── the environment (as in scripts/f4/serve_and_query.sbatch) ─────────
export PATH=/share/ju/matt/bin:$HOME/.local/bin:$PATH
export TMPDIR=/scratch/$USER/tmp
export HF_HOME=/scratch/$USER/hf-home
# The compile caches live on NFS, not on /scratch. A task on a new node then
# reuses the torch.compile graphs and the FlashInfer JIT kernels (about 20 min
# on the first node, smoke job 92682). The array runs 1 task at a time, thus
# no 2 jobs write the caches at once.
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

echo "host=$(hostname -s) job=${SLURM_JOB_ID} task=$TASK arm=$ARM case=$CASE shard=$SHARD/$SHARDS base=$BASE commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
nvidia-smi -L

VENV=$ROOT/.venv-mllmsci-vllm025cu129
export MLLMSCI_DRIVER_VENV=$VENV
source scripts/activate_stage_venv.sh
PY="${MLLMSCI_STAGE_PYTHON:-$VENV/bin/python}"
_site="$(dirname "$PY")/../lib/python3.12/site-packages"
export FLASHINFER_EXTRA_CUDAFLAGS="-I$(readlink -f "$_site/nvidia/cublas/include")"
export FLASHINFER_EXTRA_CFLAGS="$FLASHINFER_EXTRA_CUDAFLAGS"
echo "python=$PY"

# Bind FlashInfer to the real libcudart before TileLang maps its stub. Without
# this, vLLM 0.25 stops in kernel_warmup on DeepSeek-V4 ("undefined symbol:
# cudaDeviceReset", smoke job 92495). See preload_cudart/sitecustomize.py.
export F4X_PRELOAD_CUDART=1
# The cooperative top-k kernel has no sm_103 code; decode uses persistent_topk
# (smoke job 92682). See the same file.
export F4X_TOPK_FALLBACK=1
export F4X_HOOK_VERBOSE=1
export PYTHONPATH=$ROOT/scripts/f4/preload_cudart${PYTHONPATH:+:$PYTHONPATH}

"$PY" scripts/f4/extract_shard.py --arm "$ARM" --case "$CASE" --shard "$SHARD" --shards "$SHARDS" \
  --base-dir "$BASE" --set "${SETS[@]}"
echo "done $(date -u +%FT%TZ)"
