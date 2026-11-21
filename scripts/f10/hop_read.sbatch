#!/usr/bin/env bash
# F10.4: read the whole vocabulary at the text positions after the image. 1 model, 1 GPU.
# Rules: vlm-narratives-docs/facct-runbook.md, F10.4.
#
#   sbatch --export=ALL,F10_MODEL=gemma-4-31b --job-name=f104-gemma-4-31b scripts/f10/hop_read.sbatch
# Optional env: F10_HELDOUT (the pilot output dir of this model), F10_N (images for each case and borough), F10_TAG (a suffix for the output dir).
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g3-compute-17
#SBATCH --output=/share/ju/matt/mllmsci/logs/f10/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
ROOT=$PWD
MODEL="${F10_MODEL:?set F10_MODEL (gemma-4-31b, qwen3.5-9b)}"
TAG="${F10_TAG:-}"
ZOO=/share/ju/matt/zoo/models
VENV=$ROOT/.venv-mllmsci-vllm025cu129
case "$MODEL" in
  gemma-4-31b) SRC=$ZOO/Gemma-4-31B-it ;;
  qwen3.5-9b)  SRC=$ZOO/Qwen3.5-9B ;;
  *) echo "unknown F10_MODEL=$MODEL" >&2; exit 2 ;;
esac
set -a; source server.env; set +a
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/scratch/$USER/tmp
mkdir -p "$TMPDIR" logs/f10
echo "host=$(hostname -s) job=${SLURM_JOB_ID} model=$MODEL $(date -u +%FT%TZ)"
nvidia-smi -L
export MLLMSCI_DRIVER_VENV=$VENV
source scripts/activate_stage_venv.sh
PY="${MLLMSCI_STAGE_PYTHON:-$VENV/bin/python}"
eval "$(bash scripts/whatami/stage_inputs.sh "$SRC" data/f8/images_all.parquet "$PY")"
"$PY" scripts/f10/hop_read.py --model-key "$MODEL" --model-dir "$MODEL_LOCAL" \
    --lens "outputs/_monocle/jlens/jpp/$MODEL/lens.pt" --image-root "$IMAGE_ROOT" \
    --out-dir "data/f10/hops${TAG:+-$TAG}/$MODEL" --n-per-cell "${F10_N:-50}" ${F10_HELDOUT:+--heldout "$F10_HELDOUT"}
echo "done $(date -u +%FT%TZ)"
