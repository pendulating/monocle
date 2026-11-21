#!/usr/bin/env bash
# F10.1 in float32 with the HF model. 1 model, 1 shard, 1 GPU. A requeue resumes.
#
#   sbatch --export=ALL,F10_MODEL=gemma-4-31b,F10_SHARD=0/3 --job-name=f101-31b-0 scripts/f10/name_readout_hf.sbatch
# Optional env: F10_LIMIT (first N items), F10_TAG (a suffix for the output dir).
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
MODEL="${F10_MODEL:?set F10_MODEL (gemma-4-12b, gemma-4-31b)}"
ZOO=/share/ju/matt/zoo/models
VENV=$ROOT/.venv-mllmsci-vllm025cu129
case "$MODEL" in
  gemma-4-12b) SRC=$ZOO/gemma-4-12B-it ;;
  gemma-4-31b) SRC=$ZOO/Gemma-4-31B-it ;;
  *) echo "unknown F10_MODEL=$MODEL" >&2; exit 2 ;;
esac
set -a; source server.env; set +a
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/scratch/$USER/tmp
mkdir -p "$TMPDIR" logs/f10
echo "host=$(hostname -s) job=${SLURM_JOB_ID} model=$MODEL shard=${F10_SHARD:-0/1} $(date -u +%FT%TZ)"
export MLLMSCI_DRIVER_VENV=$VENV
source scripts/activate_stage_venv.sh
PY="${MLLMSCI_STAGE_PYTHON:-$VENV/bin/python}"
eval "$(bash scripts/whatami/stage_inputs.sh "$SRC" data/f8/images_all.parquet "$PY")"
"$PY" scripts/f10/name_readout_hf.py --model-key "$MODEL" --model-dir "$MODEL_LOCAL" --image-root "$IMAGE_ROOT" \
    --out-dir "data/f10/runs-fp32${F10_TAG:+-$F10_TAG}/$MODEL" --shard "${F10_SHARD:-0/1}" --limit "${F10_LIMIT:-0}"
echo "done $(date -u +%FT%TZ)"
