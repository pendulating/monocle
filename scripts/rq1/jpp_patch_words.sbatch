#!/usr/bin/env bash
# RQ1, D146: read the 8 patch-word images with a released J++ lens on 1 B300.
# The preemptible QoS `g3` gives a GPU beside the QoS `nissenbaum` queue. The
# job is short, and a requeue only repeats it. JPP_ARGS holds the arguments of
# `monocle.patch_word_gif`, or of the module in JPP_MODULE.
#
# Warning: g3-compute-17 has B200 GPUs. Pass --exclude=g3-compute-17.
#   sbatch --exclude=g3-compute-17 --export=ALL,JPP_ARGS="..." scripts/rq1/jpp_patch_words.sbatch
#SBATCH --clusters=g3
#SBATCH --partition=g3
#SBATCH --account=nissenbaum
#SBATCH --qos=g3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=jpp
#SBATCH --output=/share/ju/matt/mllmsci/logs/f4/%x-%j.out
set -uo pipefail
ROOT=/share/ju/matt/mllmsci
cd $ROOT
set -a; source $ROOT/server.env; set +a
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$ROOT:$ROOT/sub/jacobian-lens${PYTHONPATH:+:$PYTHONPATH}
echo "host=$(hostname -s) job=${SLURM_JOB_ID} $(date -u +%FT%TZ)"
nvidia-smi -L
$ROOT/.venv-mllmsci-vllm025cu129/bin/python -m "${JPP_MODULE:-monocle.patch_word_gif}" ${JPP_ARGS:?set JPP_ARGS}
