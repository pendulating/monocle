#!/usr/bin/env bash
# One F8.4 lens replay on one B300: the smoke, or the figure run.
#
# The environment block is the block of research/scripts/whatami/lens_job.sbatch. The
# job copies the gemma-4-31b weights and the selected images to node-local
# /scratch (research/scripts/whatami/stage_inputs.sh), takes the vLLM 0.25 venv from its
# /scratch mirror (HF transformers 5.13 is in it), and runs
# research/scripts/f8/lens_replay_f4.py. A requeue resumes from the finished chunks.
#
# Submit from the project root:
#   sbatch --export=ALL,F8_ARGS="--run smoke --limit 20" research/scripts/f8/lens_job.sbatch
#   sbatch --export=ALL,F8_ARGS="--run figure" research/scripts/f8/lens_job.sbatch
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=nissenbaum
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=02:55:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
#SBATCH --job-name=f8-lens
#SBATCH --output=/share/ju/matt/mllmsci/logs/f8/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci

ARGS="${F8_ARGS:?set F8_ARGS, e.g. --run smoke --limit 20}"
SRC=/share/ju/matt/zoo/models/Gemma-4-31B-it
# lens_replay_f4.py --select-only writes this file; it holds sample_id and image_path.
SAMPLE=multirun/f8/lens/selection.parquet

export PATH=/share/ju/matt/bin:$HOME/.local/bin:$PATH
export TMPDIR=/scratch/$USER/tmp
export HF_HOME=/scratch/$USER/hf-home
export XDG_CACHE_HOME=/scratch/$USER/xdg-cache
export TRITON_CACHE_DIR=$TMPDIR/triton
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
mkdir -p "$TMPDIR" logs/f8

echo "host=$(hostname -s) job=${SLURM_JOB_ID} args='$ARGS' commit=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
nvidia-smi -L

export MLLMSCI_DRIVER_VENV=/share/ju/matt/mllmsci/.venv-mllmsci-vllm025cu129
source scripts/activate_stage_venv.sh
PY="${MLLMSCI_STAGE_PYTHON:-$MLLMSCI_DRIVER_VENV/bin/python}"
# The anaconda python loads an old libstdc++ first (see serve_and_query.sbatch).
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

[[ -f "$SAMPLE" ]] || "$PY" research/scripts/f8/lens_replay_f4.py --select-only
eval "$(bash research/scripts/whatami/stage_inputs.sh "$SRC" "$SAMPLE" "$PY")"
export F8_MODEL_DIR=$MODEL_LOCAL
echo "MODEL_LOCAL=$MODEL_LOCAL IMAGE_ROOT=$IMAGE_ROOT"

read -r -a ARGV <<< "$ARGS"
trap 'echo "SIGTERM: a requeue resumes"; exit 143' TERM
"$PY" research/scripts/f8/lens_replay_f4.py "${ARGV[@]}" --image-root "$IMAGE_ROOT" &
wait $!
echo "done $(date -u +%FT%TZ)"
