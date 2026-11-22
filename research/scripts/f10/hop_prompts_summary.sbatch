#!/usr/bin/env bash
# F10.26: 1 summary for each of the 42 read stores. CPU only. Array task i maps to prompt i.
#
#   sbatch research/scripts/f10/hop_prompts_summary.sbatch
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --job-name=f1026-summary
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --array=0-41%3
#SBATCH --exclude=g3-compute-16,g3-compute-10,g3-compute-03,g3-compute-04,g3-compute-02
#SBATCH --open-mode=append
#SBATCH --output=/share/ju/matt/mllmsci/logs/f10/%x-%A_%a.out
set -euo pipefail
cd /share/ju/matt/mllmsci
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
MODELS=(gemma-4-31b qwen3.5-9b qwen3.6-27b)
ADJECTIVES=(cool nice ordinary pretty ugly fancy urban)
NOUNS=(place scene)
i=${SLURM_ARRAY_TASK_ID}
model=${MODELS[$((i / 14))]}
noun=${NOUNS[$(((i % 14) / 7))]}
adjective=${ADJECTIVES[$((i % 7))]}
echo "task $i: $model $noun $adjective on $(hostname)"
exec .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_prompts.py summary --model "$model" --adjective "$adjective" --noun "$noun"
