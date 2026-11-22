#!/usr/bin/env bash
# F10.30: the lean capture of the 14 prompts on 10 images for each NTA. 1 model, 1 GPU. Many workers
# can fill the same stores (shard claims), and a requeue resumes.
#
#   sbatch --export=ALL,F10_MODEL=gemma-4-31b --job-name=f1030-lean-gemma-4-31b research/scripts/f10/hop_lean.sbatch
# Optional env: F10_LIMIT (the first images only), F10_PROMPTS (adjective:noun pairs, comma-separated),
# F10_SAMPLE (default data/f10/hop_more/images_k10.parquet).
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g3-compute-17
#SBATCH --output=/share/ju/matt/mllmsci/logs/f10/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
ROOT=$PWD
MODEL="${F10_MODEL:?set F10_MODEL (gemma-4-31b, qwen3.5-9b, qwen3.6-27b)}"
SAMPLE="${F10_SAMPLE:-data/f10/hop_more/images_k10.parquet}"
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
export MLLMSCI_DRIVER_VENV=$VENV
source scripts/activate_stage_venv.sh
PY="${MLLMSCI_STAGE_PYTHON:-$VENV/bin/python}"
eval "$(bash research/scripts/whatami/stage_inputs.sh "$SRC" "$SAMPLE" "$PY")"
"$PY" research/scripts/f10/hop_lean.py --model-key "$MODEL" --model-dir "$MODEL_LOCAL" --lens "outputs/_monocle/jlens/jpp/$MODEL/lens.pt" \
    --image-root "$IMAGE_ROOT" --sample "$SAMPLE" --limit "${F10_LIMIT:-0}" ${F10_PROMPTS:+--prompts "$F10_PROMPTS"}
echo "done $(date -u +%FT%TZ)"
