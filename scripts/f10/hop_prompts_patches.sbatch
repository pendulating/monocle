#!/usr/bin/env bash
# F10.26: the mean patch at each layer, from the `cool` store of each model. CPU only.
#
#   sbatch scripts/f10/hop_prompts_patches.sbatch
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --job-name=f1026-patches
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --array=0-2%1
#SBATCH --exclude=g3-compute-16,g3-compute-10,g3-compute-03,g3-compute-04,g3-compute-02
#SBATCH --open-mode=append
#SBATCH --output=/share/ju/matt/mllmsci/logs/f10/%x-%A_%a.out
set -euo pipefail
cd /share/ju/matt/mllmsci
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
MODELS=(gemma-4-31b qwen3.5-9b qwen3.6-27b)
model=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID: $model on $(hostname)"
exec .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_prompts.py patches --model "$model"
