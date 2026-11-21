#!/usr/bin/env bash
# F10.19: compare the J-space of 3 models. CPU only. F10_SOLO runs the per-model scripts on 1 model.
#
#   sbatch scripts/f10/hop_compare.sbatch
#   sbatch --export=ALL,F10_SOLO=qwen3.6-27b scripts/f10/hop_compare.sbatch
# Optional env: F10_MODELS (a list with spaces), F10_OUT.
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --job-name=f1019
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=g3-compute-16,g3-compute-17
#SBATCH --output=/share/ju/matt/mllmsci/logs/f10/%x-%j.out
set -euo pipefail
cd /share/ju/matt/mllmsci
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
ARGS=()
[ -n "${F10_SOLO:-}" ] && ARGS+=(--solo "$F10_SOLO")
[ -n "${F10_MODELS:-}" ] && ARGS+=(--models $F10_MODELS)
[ -n "${F10_OUT:-}" ] && ARGS+=(--out "$F10_OUT")
exec .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_compare.py "${ARGS[@]}"
