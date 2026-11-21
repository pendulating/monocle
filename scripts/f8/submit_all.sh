#!/usr/bin/env bash
# F8.2 and F8.3: submit phase A for the 4 models, then phase B for the 4 models.
# The QoS gives 1 GPU, thus the jobs run one after the other. Phase A (the cue
# arms and their R3 readout) comes first, because it holds the primary tests.
# Use: bash scripts/f8/submit_all.sh [model ...]
set -euo pipefail
cd /share/ju/matt/mllmsci
MODELS=("${@:-gemma-4-12b qwen3.5-9b gemma-4-31b qwen3.8-27b}")
read -r -a MODELS <<< "${MODELS[*]}"
for PH in A B; do
  for M in "${MODELS[@]}"; do
    sbatch --parsable --export=ALL,F8_MODEL="$M",F8_PHASE="$PH" --job-name="f8$PH-$M" scripts/f8/cue_job.sbatch \
      | sed "s/^/f8$PH-$M /"
  done
done | tee -a logs/f8/submitted.txt
