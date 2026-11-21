#!/usr/bin/env bash
# F9: label the valence of the social inference spans on 1 B300 (the environment
# of scripts/f4/x2_queue.sbatch). F9V_ARGS holds the arguments of the `label` step.
#
# Warning: g3-compute-17 has B200 GPUs. Pass --exclude=g3-compute-17.
#   F9V_ARGS="--tag smoke --limit 600" sbatch --export=ALL --exclude=g3-compute-17 scripts/f9/label_inference_valence.sbatch
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=nissenbaum
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=f9v
#SBATCH --output=/share/ju/matt/mllmsci/logs/f4/%x-%j.out
set -uo pipefail
ROOT=/share/ju/matt/mllmsci
cd $ROOT

# ── the environment (as in scripts/f4/extract.sbatch) ─────────────────
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

echo "host=$(hostname -s) job=${SLURM_JOB_ID} commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
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
export PYTHONPATH=$ROOT/scripts/f4/preload_cudart${PYTHONPATH:+:$PYTHONPATH}

"$PY" scripts/f9/label_inference_valence.py label ${F9V_ARGS:-}
