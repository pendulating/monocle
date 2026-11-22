#!/usr/bin/env bash
# F10.26: task 0, the reproduction of F10.24 on the stores that exist. CPU only. A rerun replaces the files.
#
#   sbatch research/scripts/f10/hop_prompts_task0.sbatch
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --job-name=f1026-task0
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --exclude=g3-compute-16,g3-compute-10,g3-compute-03,g3-compute-04,g3-compute-02
#SBATCH --open-mode=append
#SBATCH --output=/share/ju/matt/mllmsci/logs/f10/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
exec .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_prompts.py task0
