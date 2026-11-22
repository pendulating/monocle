#!/usr/bin/env bash
# F10.6: the read of every token of the prompt. 1 model, 1 GPU. A requeue resumes.
# Rules: vlm-narratives-docs/facct-runbook.md, F10.6.
#
#   sbatch --export=ALL,F10_MODEL=gemma-4-31b --job-name=f106-gemma-4-31b research/scripts/f10/hop_all.sbatch
# Optional env: F10_LIMIT (the first images only), F10_TAG (a suffix for the output dir),
# F10_ADJ (the adjective of the question; this skips the patches and sets the tag),
# F10_NOUN (the noun of the question; a noun other than `place` goes in the tag).
#
# An array task is 1 worker. The task id picks the model, the adjective and the noun from PLAN, and
# many workers can fill 1 store. See vlm-narratives-docs/f10-judgment-mechanism.md.
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
  ITEM=${PLAN[$((SLURM_ARRAY_TASK_ID % ${#PLAN[@]}))]}
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
export MLLMSCI_DRIVER_VENV=$VENV
source scripts/activate_stage_venv.sh
PY="${MLLMSCI_STAGE_PYTHON:-$VENV/bin/python}"
eval "$(bash research/scripts/whatami/stage_inputs.sh "$SRC" data/f8/images_all.parquet "$PY")"
"$PY" research/scripts/f10/hop_all.py --model-key "$MODEL" --model-dir "$MODEL_LOCAL" \
    --lens "outputs/_monocle/jlens/jpp/$MODEL/lens.pt" --image-root "$IMAGE_ROOT" \
    --out-dir "data/f10/hop_all/$MODEL${TAG:+-$TAG}" --limit "${F10_LIMIT:-0}" \
    ${ADJ:+--adjective "$ADJ" --noun "$NOUN" --no-patches}
echo "done $(date -u +%FT%TZ)"
