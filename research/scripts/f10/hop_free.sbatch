#!/usr/bin/env bash
# F10.7: hop_free. CPU only. A rerun replaces the files.
#
#   sbatch research/scripts/f10/hop_free.sbatch
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --job-name=f10-hop_free
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/share/ju/matt/mllmsci/logs/f10/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
exec .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_free.py
