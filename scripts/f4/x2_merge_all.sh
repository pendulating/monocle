#!/usr/bin/env bash
# F4 x2 round (D130): merge every arm of the full run, with its retry pass,
# into data/f4_traces/x2/<arm>/. The merge writes the flags copy_suspect,
# example_source and retried (merge_extractions.py 1.3.0). A missing text
# stops the merge of that arm; the other arms go on.
#
# Batch 2 (D132) sets the runs, the output, and the case suffix with
# F4X2_ROOT, F4X2_OUT and F4X2_SFX. The defaults give the canonical run.
#
# Use, from the project root:
#   bash scripts/f4/x2_merge_all.sh            # the full run
#   bash scripts/f4/x2_merge_all.sh --allow-partial
#   F4X2_ROOT=multirun/f4x2_b2 F4X2_OUT=data/f4_traces/b2/x2 F4X2_SFX=b2x2 bash scripts/f4/x2_merge_all.sh
#
# F9 (D142) gives F4X2_PLAN_ARMS=subj: the 2 F9 arms, each with its F9 case and
# the F4 cases of the same images (both batches), under the new schema:
#   F4X2_PLAN_ARMS=subj F4X2_ROOT=multirun/f4x2_subj F4X2_OUT=data/f4_traces/subj/x2 bash scripts/f4/x2_merge_all.sh
set -u
cd /share/ju/matt/mllmsci
PY=.venv-mllmsci-vllm025cu129/bin/python
CONF=dagspaces/pairwisevqa/conf/extract
R=${F4X2_ROOT:-multirun/f4x2}
OUT=${F4X2_OUT:-data/f4_traces/x2}
SFX=${F4X2_SFX:-x2}
if [ "${F4X2_PLAN_ARMS:-}" = subj ]; then
  PLAN="x2_cues_subj cues urban_cues_single_subj_v1 restaurant_subj_x2 restaurant_x2 restaurant_b2x2
x2_cues_street_subj cues_street urban_cues_street_subj_v1 street_subj_x2 street_x2 street_b2x2"
else
  PLAN="x2_cues cues urban_cues_single_v2 restaurant_${SFX} caption_${SFX}
x2_cues_street cues_street urban_cues_street_v3 street_${SFX}
x2_desc_restaurant desc_restaurant aesthetic_descriptors_restaurant_v6 restaurant_${SFX}
x2_desc_street desc_street aesthetic_descriptors_street_v6 street_${SFX}
x2_captions captions caption_options_v4 captions_${SFX}"
fi
# arm  out-dir  schema  cases...
while read -r ARM DIR SCHEMA CASES; do
  echo "[x2-merge] $ARM -> $OUT/$DIR"
  $PY scripts/f4/merge_extractions.py $R/$ARM/runs --retry-runs $R/x2r${ARM#x2}/runs \
    --schema $CONF/$SCHEMA.yaml --cases $CASES --out $OUT/$DIR "$@" || echo "[x2-merge] FAILED $ARM"
done <<< "$PLAN"
