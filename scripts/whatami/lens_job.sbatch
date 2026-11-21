#!/usr/bin/env bash
# One whatami lens step on one B300: a fit shard, a smoke, or a replay.
#
# The job copies the model weights and the probe images to node-local /scratch
# (stage_inputs.sh), takes the vLLM 0.25 venv from its /scratch mirror (HF
# transformers 5.13 is in it), and runs one python command from
# scripts/whatami/. A requeue resumes: the fit from its checkpoint, the replay
# from its finished chunks.
#
# Submit from the project root:
#   sbatch --export=ALL,WHATAMI_MODEL=gemma-4-12b,WHATAMI_CMD="lens_fit.py --corpus image --smoke" \
#       scripts/whatami/lens_job.sbatch
#   sbatch --export=ALL,WHATAMI_MODEL=gemma-4-12b,WHATAMI_CMD="lens_fit.py --corpus text --shard 0/1" ...
#   sbatch --export=ALL,WHATAMI_MODEL=gemma-4-12b,WHATAMI_CMD="lens_replay.py --shard 0/1" ...
# The script adds --model, and --image-root for the replay.
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=nissenbaum
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=whatami-lens
#SBATCH --output=/share/ju/matt/mllmsci/logs/whatami/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci

MODEL="${WHATAMI_MODEL:?set WHATAMI_MODEL}"
CMD="${WHATAMI_CMD:?set WHATAMI_CMD, e.g. lens_fit.py --corpus text --smoke}"
SAMPLE=data/whatami/sample_1k.parquet
ZOO=/share/ju/matt/zoo/models
case "$MODEL" in
  gemma-4-12b) SRC=$ZOO/gemma-4-12B-it ;;
  qwen3.5-9b) SRC=$ZOO/Qwen3.5-9B ;;
  llama-3.2-11b) SRC=$ZOO/Llama-3.2-11B-Vision-Instruct ;;
  llama-3.2-90b) SRC=$ZOO/Llama-3.2-90B-Vision-Instruct ;;
  *) echo "unknown WHATAMI_MODEL=$MODEL" >&2; exit 2 ;;
esac

export PATH=/share/ju/matt/bin:$HOME/.local/bin:$PATH
export TMPDIR=/scratch/$USER/tmp
export HF_HOME=/scratch/$USER/hf-home
export XDG_CACHE_HOME=/scratch/$USER/xdg-cache
export TRITON_CACHE_DIR=$TMPDIR/triton
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR" logs/whatami

echo "host=$(hostname -s) job=${SLURM_JOB_ID} model=$MODEL cmd='$CMD' commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
nvidia-smi -L

export MLLMSCI_DRIVER_VENV=/share/ju/matt/mllmsci/.venv-mllmsci-vllm025cu129
source scripts/activate_stage_venv.sh
PY="${MLLMSCI_STAGE_PYTHON:-$MLLMSCI_DRIVER_VENV/bin/python}"
# The anaconda python loads an old libstdc++ first (see serve_and_query.sbatch).
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# qwen3.5: without flash-linear-attention, transformers runs the gated delta
# rule as a python loop. The image fit then ran out of memory at dim_batch 16,
# and the text fit took 2.4x the gemma time (2026-10-21). The overlay holds
# fla-core and flash-linear-attention 0.5.2 (pure python and Triton), outside
# the shared venv. causal-conv1d is not in it; its torch path stays.
FLA_OVERLAY=/share/ju/matt/mllmsci/.overlays/fla-0.5.2
if [[ "$MODEL" == qwen3.5-9b ]]; then
  [[ -f "$FLA_OVERLAY/.sync_complete" ]] || { echo "no complete overlay at $FLA_OVERLAY" >&2; exit 1; }
  export PYTHONPATH="$FLA_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
  echo "fla overlay: $FLA_OVERLAY"
fi

eval "$(bash scripts/whatami/stage_inputs.sh "$SRC" "$SAMPLE" "$PY")"
export WHATAMI_MODEL_DIR=$MODEL_LOCAL
echo "MODEL_LOCAL=$MODEL_LOCAL IMAGE_ROOT=$IMAGE_ROOT"

read -r -a ARGV <<< "$CMD"
SCRIPT="${ARGV[0]}"
EXTRA=()
[[ "$SCRIPT" == lens_replay.py ]] && EXTRA=(--image-root "$IMAGE_ROOT")
trap 'echo "SIGTERM: a requeue resumes"; exit 143' TERM
"$PY" "scripts/whatami/$SCRIPT" --model "$MODEL" "${ARGV[@]:1}" "${EXTRA[@]}" &
wait $!
echo "done $(date -u +%FT%TZ)"
