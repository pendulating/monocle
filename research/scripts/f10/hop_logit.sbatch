#!/usr/bin/env bash
# F10.7: logit metrics from the saved residuals. 1 model, 1 GPU. A requeue resumes.
# Rules: research/docs/facct-runbook.md, F10.7.
#
#   sbatch --export=ALL,F10_MODEL=gemma-4-31b --job-name=f107-gemma-4-31b research/scripts/f10/hop_logit.sbatch
# Optional env: F10_LIMIT (the first images only), F10_TAG (a suffix for the output dir),
# F10_ADJ and F10_NOUN (the words of the question). An array task id picks 1 item of PLAN.
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
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  PLAN=()
  for PROMPT in pretty:place ugly:place fancy:place urban:place cool:scene nice:scene ordinary:scene pretty:scene ugly:scene fancy:scene urban:scene; do
    for M in gemma-4-31b qwen3.5-9b qwen3.6-27b; do PLAN+=("$M:$PROMPT"); done
  done
  ITEM=${PLAN[$SLURM_ARRAY_TASK_ID]}
  IFS=: read -r F10_MODEL F10_ADJ F10_NOUN <<< "$ITEM"
fi
MODEL="${F10_MODEL:?set F10_MODEL (gemma-4-31b, qwen3.5-9b, qwen3.6-27b)}"
ADJ="${F10_ADJ:-}"
NOUN="${F10_NOUN:-place}"
if [ "$NOUN" = place ]; then TAG="${F10_TAG:-$ADJ}"; else TAG="${F10_TAG:-$NOUN-$ADJ}"; fi
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
"$PY" research/scripts/f10/hop_logit.py --model-key "$MODEL${TAG:+-$TAG}" --model-dir "$MODEL_LOCAL" ${ADJ:+--adjective "$ADJ" --noun "$NOUN"} \
    --lens "outputs/_monocle/jlens/jpp/$MODEL/lens.pt" --image-root "$IMAGE_ROOT" \
    --out-dir "data/f10/hop_logit/$MODEL${TAG:+-$TAG}" --limit "${F10_LIMIT:-0}" --shards "${F10_SHARDS:-0}"
echo "done $(date -u +%FT%TZ)"
