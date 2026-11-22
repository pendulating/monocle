#!/usr/bin/env bash
# F10.26: the words of the shared direction (task 7, step 4), then the figures. CPU only. A rerun replaces the files.
#
#   sbatch research/scripts/f10/hop_prompts_words.sbatch
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --job-name=f1026-words
#SBATCH --cpus-per-task=18
#SBATCH --mem=160G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --exclude=g3-compute-16,g3-compute-10,g3-compute-03,g3-compute-04,g3-compute-02
#SBATCH --open-mode=append
#SBATCH --output=/share/ju/matt/mllmsci/logs/f10/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
export OMP_NUM_THREADS=3 PYTHONUNBUFFERED=1
.venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_prompts.py words
exec .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_prompts.py figures
