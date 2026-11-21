#!/usr/bin/env bash
# F10.7: analyze the logit metrics. CPU only. A rerun replaces the files.
#
#   sbatch scripts/f10/analyze_logit.sbatch
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --job-name=f107-analyze
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/share/ju/matt/mllmsci/logs/f10/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
exec .venv-mllmsci-vllm025cu129/bin/python scripts/f10/analyze_logit.py
