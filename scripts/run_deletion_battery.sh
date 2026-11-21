#!/bin/bash
# The decisive test: does an instrument RANK the cells the model uses?
#
# A per-cell correlation is the wrong statistic. The judgment saturates, so the
# magnitudes do not decompose, and 2 honest per-cell methods agree at only
# r=0.26. The unit that carries the claim is a SET of cells, and the statistic
# is the separation of a directed deletion.
#
# The battery ranks the same 20 images by each instrument and runs the same
# deletion test. 3 numbers frame every arm:
#
#   | Reference | median separation z at k=32 |
#   |---|---|
#   | the measured input map (the ceiling) | 3.45 |
#   | the layer-20 donor map | 2.19 |
#   | random | 0 by construction |
#
# An instrument earns the name "causal lens" when it beats the model-free
# saliency arm and approaches the ceiling.
#
# Warning: the maps and the deletion test must read the SAME images. Give the
# seed that the maps were built with. deletion_curve defaults to 20260821.
#
# Usage: bash scripts/run_deletion_battery.sh <maps.parquet> <layer> <tag>
#          [case] [seed] [n_images]
set -euo pipefail
cd /share/ju/matt/mllmsci

MAPS="${1:?a gate maps.parquet}"
LAYER="${2:?the layer to read the instruments at}"
TAG="${3:?a name for this battery}"
CASE="${4:-subway_safety}"
SEED="${5:-20260821}"
N="${6:-20}"
GROUP=2
KS="1 2 4 8 16 32"
RANKDIR="outputs/_monocle/rankings/$TAG"
SUB=monocle/monocle.sub
# The system python3 is anaconda, and its numpy clashes with the
# project wheels. Name the project interpreter.
PY=/share/ju/matt/mllmsci/.venv-3.12/bin/python
mkdir -p "$RANKDIR"

# The model arms. Each one is a column of the gate maps parquet.
ARMS="lens_own cheap gxa meang_own meangp_own"
for arm in $ARMS; do
  out="$RANKDIR/${arm}_L${LAYER}.parquet"
  if ! "$PY" scripts/make_ranking.py --maps "$MAPS" --arm "$arm" \
        --layer "$LAYER" --group $GROUP --out "$out"; then
    echo "[battery] skip $arm: make_ranking failed at L$LAYER"
    continue
  fi
  jid=$(sbatch --parsable --job-name="del-$TAG-$arm" $SUB monocle.deletion_curve \
        --cases "$CASE" --n-images "$N" --group $GROUP --ks $KS --seed "$SEED" \
        --rank-from "$out" --tag "${TAG}_${arm}_L${LAYER}")
  echo "[battery] $arm -> $jid"
done

# The model-free null. It must FAIL for any lens claim to stand.
# Warning: these arms do not read a layer. Run the battery at 2
# layers and the same 3 jobs are submitted twice, and they write
# to one folder. Set SKIP_SALIENCY=1 on the second call.
if [ "${SKIP_SALIENCY:-0}" = "1" ]; then echo "[battery] saliency skipped"; exit 0; fi
for feat in edge luminance contrast; do
  out="$RANKDIR/saliency_${feat}.parquet"
  "$PY" scripts/saliency_ranking.py --maps "$MAPS" --case "$CASE" \
      --seed "$SEED" --rank "$feat" --group $GROUP --out "$out"
  jid=$(sbatch --parsable --job-name="del-$TAG-sal_$feat" $SUB monocle.deletion_curve \
        --cases "$CASE" --n-images "$N" --group $GROUP --ks $KS --seed "$SEED" \
        --rank-from "$out" --tag "${TAG}_saliency_${feat}")
  echo "[battery] saliency/$feat -> $jid"
done
echo "[battery] rankings in $RANKDIR"
