#!/usr/bin/env bash
# F10.29: steering with the shared aesthetic direction (items 5 and 7). 1 model, 1 GPU. A requeue resumes.
#
#   sbatch --export=ALL,F10_MODEL=gemma-4-31b --job-name=f1029-gemma-4-31b research/scripts/f10/hop_steer.sbatch
# Optional env: F10_ARGS (more stage arguments), F10_LIMIT (the first images only), F10_STAGES (default `words run`; the follow-up is `words_model dose`).
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g3-compute-17
#SBATCH --output=/share/ju/matt/mllmsci/logs/f10/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
ROOT=$PWD
MODEL="${F10_MODEL:?set F10_MODEL (gemma-4-31b, qwen3.5-9b, qwen3.6-27b)}"
ZOO=/share/ju/matt/zoo/models
VENV=$ROOT/.venv-mllmsci-vllm025cu129
case "$MODEL" in
  gemma-4-31b) SRC=$ZOO/Gemma-4-31B-it ;;
  qwen3.5-9b)  SRC=$ZOO/Qwen3.5-9B ;;
  qwen3.6-27b) SRC=$ZOO/Qwen3.6-27B ;;
  *) echo "unknown F10_MODEL=$MODEL" >&2; exit 2 ;;
esac
set -a; source server.env; set +a
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/scratch/$USER/tmp
mkdir -p "$TMPDIR" logs/f10
echo "host=$(hostname -s) job=${SLURM_JOB_ID} model=$MODEL $(date -u +%FT%TZ)"
nvidia-smi -L
export IMAGE_FORWARD_DRIVER_VENV=$VENV
source scripts/activate_stage_venv.sh
PY="${IMAGE_FORWARD_STAGE_PYTHON:-$VENV/bin/python}"
eval "$(bash research/scripts/whatami/stage_inputs.sh "$SRC" data/f8/images_all.parquet "$PY")"
for STAGE in ${F10_STAGES:-words run}; do
  "$PY" research/scripts/f10/hop_steer.py "$STAGE" --model-key "$MODEL" --model-dir "$MODEL_LOCAL" --image-root "$IMAGE_ROOT" \
      --lens "outputs/_monocle/jlens/jpp/$MODEL/lens.pt" --limit "${F10_LIMIT:-0}" ${F10_ARGS:-}
done
echo "done $(date -u +%FT%TZ)"
