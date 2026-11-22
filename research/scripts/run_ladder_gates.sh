#!/bin/bash
# Run the causal gate for each merged corpus-faithfulness rung.
#
# Every flag matches the stage-4 run that made the baseline, so the numbers
# compare directly to the record:
#   12 images -> 6 donor-target pairs, layers 12/18/20/24, batch 8.
# L20 carries no lens arm; it is the local instrument's anchor.
#
# See research/docs/lens-corpus-faithfulness.md.
#
# Usage: bash research/scripts/run_ladder_gates.sh [R0 R1 ...]
set -euo pipefail
cd /share/ju/matt/mllmsci
ROOT=outputs/_monocle/jlens/ladder
RUNGS=("${@:-R0 R1 R2 R3}")

for rung in ${RUNGS[@]}; do
 lens="$ROOT/$rung/gemma4_12b_lens.pt"
 if [ ! -f "$lens" ]; then
   echo "[gate] $rung: no merged lens at $lens, skipped"
   continue
 fi
 # 12 images keeps the link to the stage-4 record (6 pairs). 24 roughly
 # doubles the replicates. The analysis reserve of 32 covers both.
 for n in 12 24; do
  jid=$(sbatch --time=02:00:00 --job-name="gate-$rung-n$n" \
    research/monocle/monocle.sub monocle.patch_map_causal \
    --cases subway_safety --n-images "$n" --layers 12 18 20 24 --batch 8 \
    --lens "$lens" \
    --out-dir "outputs/_monocle/patch_map_causal_ladder_${rung}_n${n}" \
    2>&1 | awk '{print $NF}')
  echo "[gate] $rung n=$n -> job $jid"
 done
done
