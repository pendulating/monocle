#!/usr/bin/env bash
# F10.3: the coordinate swap of the place names. 1 model, 1 GPU. A requeue resumes.
# Rules: vlm-narratives-docs/facct-runbook.md, F10.3.
#
# Submit from the project root, 1 job for each model:
#   sbatch --export=ALL,F10_MODEL=gemma-4-31b --job-name=f103-gemma-4-31b research/scripts/f10/name_coord_swap.sbatch
# Optional env: F10_N (images for each case and borough), F10_TAG (a suffix for the output dir),
# F10_HELDOUT (the F10.3 output dir of this model: runs the held-out test).
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
N="${F10_N:-100}"
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
echo "host=$(hostname -s) job=${SLURM_JOB_ID} model=$MODEL commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
nvidia-smi -L
export MLLMSCI_DRIVER_VENV=$VENV
source scripts/activate_stage_venv.sh
PY="${MLLMSCI_STAGE_PYTHON:-$VENV/bin/python}"
eval "$(bash research/scripts/whatami/stage_inputs.sh "$SRC" data/f8/images_all.parquet "$PY")"
echo "python=$PY MODEL_LOCAL=$MODEL_LOCAL IMAGE_ROOT=$IMAGE_ROOT"
"$PY" research/scripts/f10/name_coord_swap.py --model-key "$MODEL" --model-dir "$MODEL_LOCAL" \
    --lens "outputs/_monocle/jlens/jpp/$MODEL/lens.pt" --image-root "$IMAGE_ROOT" \
    --out-dir "data/f10/swap${TAG:+-$TAG}/$MODEL" --n-per-cell "$N" ${F10_HELDOUT:+--heldout "$F10_HELDOUT"}
echo "done $(date -u +%FT%TZ)"
