#!/usr/bin/env bash
# Submit F4 for each model, one after the other (QoS nissenbaum gives 1 GPU).
#
# The chain uses afterany: a failed model does not block the next one.
#
# Usage (from the project root):
#   bash research/scripts/f4/submit_all.sh                          # all 6 models
#   bash research/scripts/f4/submit_all.sh gemma-4-12b qwen3.5-9b   # some models
#   F4_TAG=smoke F4_LIMIT=8 bash research/scripts/f4/submit_all.sh gemma-4-12b
set -euo pipefail
cd /share/ju/matt/mllmsci

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(gemma-4-12b qwen3.5-9b gemma-4-31b qwen3.8-27b llama-3.2-11b llama-3.2-90b)

EXPORTS="ALL"
# Pass each F4 setting explicitly: F4_MAX_TOKENS did not reach job 100679 with ALL only.
for v in F4_LIMIT F4_TAG F4_CASES F4_SAMPLE F4_MAX_TOKENS; do
  [ -n "${!v:-}" ] && EXPORTS+=",$v=${!v}"
done

prev=""
for m in "${MODELS[@]}"; do
  dep=()
  [ -n "$prev" ] && dep=(--dependency="afterany:$prev")
  # The B200 node cannot hold llama-3.2-90b (job 94674).
  [ "$m" = llama-3.2-90b ] && dep+=(--exclude=g3-compute-17)
  out=$(sbatch --parsable "${dep[@]}" --job-name="f4-$m" --export="$EXPORTS,F4_MODEL=$m" \
        research/scripts/f4/serve_and_query.sbatch)
  prev="${out%%;*}"
  echo "$m -> job $prev"
done
