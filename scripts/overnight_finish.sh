#!/bin/bash
# Run each analysis as soon as its experiment lands.
#
# It does 3 jobs, each one time only:
#   1. the 7-question cross-question analysis, when the 7th map lands
#   2. the caption value repair, when no dense job is live
#   3. the presentation control, when the matched 5.3x arm lands
#
# It prints 1 line for each step, so it suits a Monitor.
set -u
cd /share/ju/matt/mllmsci
PY=.venv-3.12/bin/python
R=outputs/reports
mkdir -p $R
DEADLINE=$(( $(date +%s) + 25200 ))          # 7 hours

did_xq=0; did_repair=0; did_pres=0
while true; do
  # 1. cross-question, all 7 questions
  if [ "$did_xq" -eq 0 ]; then
    n=$(ls -d outputs/_monocle/xq30/*/*_absent/maps.parquet 2>/dev/null | wc -l)
    if [ "$n" -ge 7 ]; then
      echo "XQ: 7 questions present, running the coincidence analysis"
      $PY scripts/xq_pole_coincidence.py --root outputs/_monocle/xq30 \
          --out $R/xq_coincidence_7q.csv > $R/xq_coincidence_7q.log 2>&1 \
        && echo "XQ DONE -> $R/xq_coincidence_7q.csv" \
        || echo "XQ FAILED (see $R/xq_coincidence_7q.log)"
      did_xq=1
    fi
  fi

  # 2. the caption value repair; a dense job that started before the fix
  #    wrote a rank into `value`.
  if [ "$did_repair" -eq 0 ]; then
    live=$(squeue -u mwf62 -h -o "%j" 2>/dev/null | grep -c '^dense-')
    if [ "$live" -eq 0 ]; then
      echo "REPAIR: no dense job is live, checking the caption stores"
      $PY scripts/repair_caption_values.py --apply > $R/value_repair.log 2>&1 \
        && echo "REPAIR DONE: $(tail -2 $R/value_repair.log | head -1)" \
        || echo "REPAIR FAILED (see $R/value_repair.log)"
      did_repair=1
    fi
  fi

  # 3. the presentation control, once the matched arm exists for 4 cases
  if [ "$did_pres" -eq 0 ]; then
    n=$(ls -d outputs/_monocle/patch_caption_pres/*_x53 2>/dev/null | wc -l)
    if [ "$n" -ge 4 ]; then
      echo "PRES: the matched 5.3x arm is present for $n cases"
      $PY scripts/pres_control_report.py --out $R/pres_control \
          > $R/pres_control.log 2>&1 \
        && echo "PRES DONE -> $R/pres_control/pres_control.csv" \
        || echo "PRES FAILED (see $R/pres_control.log)"
      did_pres=1
    fi
  fi

  [ "$did_xq" -eq 1 ] && [ "$did_repair" -eq 1 ] && [ "$did_pres" -eq 1 ] \
    && { echo "FINISH: all 3 analyses are done"; break; }
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "FINISH: deadline reached (xq=$did_xq repair=$did_repair pres=$did_pres)"
    break
  fi
  sleep 240
done
