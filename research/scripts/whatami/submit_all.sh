#!/usr/bin/env bash
# Submit the "what am I looking at?" probe for each model, one after the other.
#
# QoS nissenbaum gives the group 1 GPU, thus the jobs cannot run together. The
# chain uses afterany: a failed model does not block the next one. A requeued
# task passes through a state that afterok reads as a failure.
#
# Usage (from the project root):
#   bash research/scripts/whatami/submit_all.sh                         # all four models
#   bash research/scripts/whatami/submit_all.sh gemma-4-12b qwen3.5-9b  # some models
#   WHATAMI_TAG=smoke WHATAMI_LIMIT=5 WHATAMI_REPEATS=2 bash research/scripts/whatami/submit_all.sh gemma-4-12b
set -euo pipefail
cd /share/ju/matt/mllmsci

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(gemma-4-12b qwen3.5-9b llama-3.2-11b llama-3.2-90b)

EXPORTS="ALL"
for v in WHATAMI_SAMPLE WHATAMI_LIMIT WHATAMI_REPEATS WHATAMI_TAG; do
  [ -n "${!v:-}" ] && EXPORTS+=",$v=${!v}"
done

prev=""
for m in "${MODELS[@]}"; do
  dep=()
  [ -n "$prev" ] && dep=(--dependency="afterany:$prev")
  out=$(sbatch --parsable "${dep[@]}" --job-name="whatami-$m" --export="$EXPORTS,WHATAMI_MODEL=$m" \
        research/scripts/whatami/serve_and_query.sbatch)
  prev="${out%%;*}"
  echo "$m -> job $prev"
done
