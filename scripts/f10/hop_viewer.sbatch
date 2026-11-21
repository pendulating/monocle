#!/usr/bin/env bash
# F10.6: write the reads of every token as runs for the Neuronpedia viewer. CPU only. A rerun replaces the files.
#
#   sbatch scripts/f10/hop_viewer.sbatch
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --job-name=f105-viewer
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/share/ju/matt/mllmsci/logs/f10/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
exec .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_viewer.py
