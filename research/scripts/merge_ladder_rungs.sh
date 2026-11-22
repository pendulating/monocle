#!/bin/bash
# Merge the shards of each corpus-faithfulness rung into one lens.
#
# The merge is CPU work: JacobianLens.merge is an n_prompts-weighted mean, so
# it needs no GPU and no model. See
# research/docs/lens-corpus-faithfulness.md.
#
# Usage: bash research/scripts/merge_ladder_rungs.sh [R0 R1 ...]
set -euo pipefail
cd /share/ju/matt/mllmsci
PY=./.venv-3.12/bin/python
ROOT=outputs/_monocle/jlens/ladder
RUNGS=("${@:-R0 R1 R2 R3}")

for rung in ${RUNGS[@]}; do
  dir="$ROOT/$rung"
  n=$(ls "$dir"/shard*of*.pt 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "[merge] $rung: no shard, skipped"
    continue
  fi
  echo "[merge] $rung: $n shards"
  PYTHONPATH=sub/jacobian-lens $PY -m research.monocle.jlens_fit \
    --merge "$dir"/shard*of*.pt --out-dir "$dir"
done
