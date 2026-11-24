#!/usr/bin/env bash
# F10.35 step 2: the readout instruments on the F10.6 stores. CPU only. A rerun replaces the files.
# The 3 models run in sequence, thus summary.csv holds all 3.
#
#   sbatch research/scripts/f10/hop_instruments.sbatch
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --job-name=f1035-instruments
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/share/ju/matt/mllmsci/logs/f10/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
exec .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_instruments.py --stage all
