#!/bin/bash
# Score every ladder rung against the INPUT ablation truth.
#
# Why this replaces the old gate: Gate A scored a rung against a layer-20
# residual swap, and that swap agrees with the input ablation at only r=+0.08.
# A rung that wins Gate A thus does not have to read the picture better. Gate B
# scores against the input ablation itself, which IS the content truth.
#
# The run is cheap. --truth-from reads the 256-patch ablation that the
# 2026-08-23 run measured, so each image costs 1 backward pass, not 256
# forward passes.
set -euo pipefail
cd /share/ju/matt/mllmsci

CASE="${1:-subway_safety}"
TRUTH="${2:-outputs/_monocle/causal_maps/${CASE}_absent/maps.parquet}"
LAYERS="0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 36 42 46"
SUB=research/monocle/monocle.sub

if [ ! -f "$TRUTH" ]; then
  echo "no stored truth at $TRUTH; run causal_maps --reference absent first" >&2
  exit 2
fi

for rung in R0 R1 R2 R3; do
  lens="outputs/_monocle/jlens/ladder/$rung/gemma4_12b_lens.pt"
  if [ ! -f "$lens" ]; then
    echo "[gateB] skip $rung: the merged lens is absent"
    continue
  fi
  jid=$(sbatch --parsable --job-name="gateB-$rung" $SUB monocle.input_gate \
        --case "$CASE" --seed 20260821 --n-images 20 --layers $LAYERS \
        --batch 8 --group 1 --truth-from "$TRUTH" --lens "$lens" \
        --out-dir "outputs/_monocle/gateB_ladder_$rung")
  echo "[gateB] $rung -> $jid"
done
