#!/bin/bash
# The verdict, for every case, at 1 sample size.
#
# It runs the gated single-statistic test (scripts/deletion_auc.py), writes the
# JSON, and draws the figure (scripts/plot_lens_verdict.py).
#
# Warning: an arm and its nulls must come from the SAME sample size, or the
# paired test drops every image the 2 do not share.
#
# Usage: bash scripts/run_final_verdict.sh <n20|n60> <out.png>
set -uo pipefail
cd /share/ju/matt/mllmsci
SIZE="${1:?n20 or n60}"
OUT="${2:-outputs/figures/lens_verdict_${SIZE}.png}"
PY=/share/ju/matt/mllmsci/.venv-3.12/bin/python
D=outputs/_monocle/deletion_curve
J="${OUT%.png}.json"
rm -f "$J"

for C in subway_safety libraries restaurants; do
  if [ "$SIZE" = n60 ]; then
    CEIL="$D/${C}_g2_ceil60_ablate"
    NPRE="${C}_g2_n60_saliency_"
    APRE="${C}_g2_n60_"
    EXTRA=""
  else
    case $C in
      subway_safety) T=gbJ; CEIL="$D/subway_safety_g2_self_ablate";;
      libraries)     T=gblibraries; CEIL="$D/libraries_g2_ceil_ablate";;
      restaurants)   T=gbrestaurants; CEIL="$D/restaurants_g2_ceil_ablate";;
    esac
    NPRE="${C}_g2_${T}_saliency_"
    APRE="${C}_g2_${T}_"
    EXTRA="$D/${C}_g2_maskm02_ablate $D/${C}_g2_maskm10_ablate"
  fi
  N=""; for f in edge luminance contrast row col saturation; do
    p="$D/${NPRE}${f}_ablate"; [ -d "$p" ] && N="$N $p"; done
  A="$CEIL"; for a in lens_own cheap gxa meang_own meangp_own; do
    p="$D/${APRE}${a}_L18_ablate"; [ -d "$p" ] && A="$A $p"; done
  for p in $EXTRA; do [ -d "$p" ] && A="$A $p"; done
  [ -d "$CEIL" ] || { echo "[verdict] $C: no ceiling at $SIZE, skip"; continue; }
  [ -n "$N" ] || { echo "[verdict] $C: no nulls at $SIZE, skip"; continue; }
  "$PY" scripts/deletion_auc.py --label "$C" --ceiling "$CEIL" \
    --arms $A --nulls $N --json-out "$J"
  echo
done
[ -f "$J" ] && "$PY" scripts/plot_lens_verdict.py --json "$J" --out "$OUT"
