#!/usr/bin/env bash
# F8.2 and F8.3: run all the tests after the GPU jobs end.
# 1. analyze_cue.py: the cue tests, the NTA Spearman tests, data/f8/r3_pool.parquet.
# 2. method_b.py: F8.1 method B with each R3 readout as the outcome (F8.3 (a)).
# Use: bash research/scripts/f8/finalize.sh
set -euo pipefail
cd /share/ju/matt/mllmsci
PY=.venv-mllmsci-vllm025cu129/bin/python
"$PY" research/scripts/f8/analyze_cue.py > research/figures/f8/cue/analyze_cue.log 2>&1
for rd in c_yes c_cool c_yes_i c_cool_i; do
  "$PY" research/scripts/f8/method_b.py --outcome-parquet data/f8/r3_pool.parquet --outcome-col "$rd" \
      --out-dir research/figures/f8/method_b_r3 2>&1 | grep -v -i warn | tail -3
done
