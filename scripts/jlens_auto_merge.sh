#!/bin/bash
# Merge the 16 shards as soon as they are complete.
#
# If the shards do not all land before the deadline, it merges what exists.
# A checkpoint holds the sum over the images that a shard did finish, so a
# partial merge is still a correct mean over fewer images.
set -u
cd /share/ju/matt/mllmsci
OUT=outputs/_monocle/jlens/mm_all47
PY=.venv-3.12/bin/python
DEADLINE=$(( $(date +%s) + 21600 ))          # 6 hours

# WARNING: an empty queue is NOT proof that the run stopped. SLURM can
# preempt every shard at the same moment, and the watchdog needs up to 2
# minutes to put them back. A merge on that moment would take the partial
# checkpoints and stop. So an empty queue must hold for 4 polls, 12 minutes.
empty=0
while true; do
  n=$(ls $OUT/shard*of16.pt 2>/dev/null | wc -l)
  live=$(squeue -u mwf62 -h -o "%j" 2>/dev/null | grep -c '^jlens-s')
  if [ "$n" -ge 16 ]; then
    echo "MERGE start: all 16 shards complete"; break
  fi
  if [ "$live" -eq 0 ]; then
    empty=$((empty+1))
    echo "MERGE wait: no shard job is live ($empty of 4), $n/16 complete"
    [ "$empty" -ge 4 ] && { echo "MERGE start: the queue stayed empty"; break; }
  else
    empty=0
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "MERGE start: deadline reached, $n/16 complete"; break
  fi
  sleep 180
done

$PY -m monocle.jlens_merge_mm --in-dir "$OUT" --out "$OUT/gemma4_12b_lens.pt" \
  2>&1 | sed 's/^/MERGE /'
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && { echo "MERGE FAILED rc=$rc"; exit 1; }

$PY scripts/jlens_verify_lens.py "$OUT/gemma4_12b_lens.pt" 47 2>&1 | sed 's/^/VERIFY /'
echo "MERGE DONE $OUT/gemma4_12b_lens.pt"
