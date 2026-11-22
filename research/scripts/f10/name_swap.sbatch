#!/usr/bin/env bash
# F10.1: the text swap of the place name. 1 model, 1 GPU. A requeue resumes.
# Rules: research/docs/facct-runbook.md, F10.1.
#
# Submit from the project root, 1 job for each model:
#   sbatch --export=ALL,F10_MODEL=gemma-4-12b --job-name=f10-gemma-4-12b research/scripts/f10/name_swap.sbatch
# Optional env: F10_LIMIT (first N items, smoke test), F10_TAG (a suffix for the output dir), F10_ITEMS, F10_DTYPE.
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g3-compute-17
#SBATCH --output=/share/ju/matt/mllmsci/logs/f10/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
ROOT=$PWD
MODEL="${F10_MODEL:?set F10_MODEL (gemma-4-12b, gemma-4-31b, qwen3.5-9b, qwen3.8-27b)}"
LIMIT="${F10_LIMIT:-0}"
TAG="${F10_TAG:-}"
ZOO=/share/ju/matt/zoo/models
VENV=$ROOT/.venv-mllmsci-vllm025cu129
case "$MODEL" in
  gemma-4-12b) SRC=$ZOO/gemma-4-12B-it ;;
  gemma-4-31b) SRC=$ZOO/Gemma-4-31B-it ;;
  qwen3.5-9b)  SRC=$ZOO/Qwen3.5-9B ;;
  qwen3.8-27b) SRC=$ZOO/Qwen3.8-27B ;;
  *) echo "unknown F10_MODEL=$MODEL" >&2; exit 2 ;;
esac

# The environment: as research/scripts/f8/cue_job.sbatch.
export PATH=/share/ju/matt/bin:$HOME/.local/bin:$PATH
export TMPDIR=/scratch/$USER/tmp
export HF_HOME=/scratch/$USER/hf-home
export VLLM_CACHE_ROOT=/scratch/$USER/vllm-cache
export VLLM_CONFIG_ROOT=/scratch/$USER/vllm-config
export XDG_CACHE_HOME=/scratch/$USER/xdg-cache
export TRITON_CACHE_DIR=$TMPDIR/triton
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export FLASHINFER_WORKSPACE_BASE=/scratch/$USER/flashinfer
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR" logs/f10
echo "host=$(hostname -s) job=${SLURM_JOB_ID} model=$MODEL commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
nvidia-smi -L

export IMAGE_FORWARD_DRIVER_VENV=$VENV
source scripts/activate_stage_venv.sh
PY="${IMAGE_FORWARD_STAGE_PYTHON:-$VENV/bin/python}"
_site="$(dirname "$PY")/../lib/python3.12/site-packages"
export FLASHINFER_EXTRA_CUDAFLAGS="-I$(readlink -f "$_site/nvidia/cublas/include")"
export FLASHINFER_EXTRA_CFLAGS="$FLASHINFER_EXTRA_CUDAFLAGS"
eval "$(bash research/scripts/whatami/stage_inputs.sh "$SRC" data/f8/images_all.parquet "$PY")"
echo "python=$PY MODEL_LOCAL=$MODEL_LOCAL IMAGE_ROOT=$IMAGE_ROOT"

# The offline engine needs the system libstdc++ (job 103145).
env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "$PY" research/scripts/f10/name_swap_readout.py \
    --model-key "$MODEL" --model-path "$MODEL_LOCAL" --image-root "$IMAGE_ROOT" \
    --items "${F10_ITEMS:-data/f10/name_items.parquet}" --out-dir "data/f10/runs${TAG:+-$TAG}/$MODEL" --limit "$LIMIT" --dtype "${F10_DTYPE:-auto}"
echo "done $(date -u +%FT%TZ)"
