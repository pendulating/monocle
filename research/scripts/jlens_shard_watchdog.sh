#!/bin/bash
# Keep the 16 multimodal J-lens shards alive until each one is complete.
#
# WHY THIS FILE IS IN THE PROJECT
#   The first watchdog lived in /tmp. A /tmp cleanup removed it and its state
#   files while the fit ran. This copy is on NFS, and it keeps its state next
#   to the shards, so a cleanup cannot break it.
#
# WHAT IT DOES
#   A shard is covered when a job named `jlens-sNN` is in the queue, or when
#   `shard<i>of16.pt` exists. Any other shard gets a new job. A node that
#   reports a broken GPU goes on the exclude list.
#
# It prints 1 line for each change only, so it suits a Monitor.
set -u
cd /share/ju/matt/mllmsci

OUT=outputs/_monocle/jlens/mm_all47
STATE=$OUT/.watchdog
mkdir -p "$STATE"
BAD=$STATE/badnodes.txt
[ -s "$BAD" ] || echo "elor-compute-02,ma-compute-02" > "$BAD"
MAX_TRIES=40
N=16
# Every 3rd shard asks for the Ada card, to spread the load over both types.
gres_for() { [ $(( $1 % 3 )) -eq 2 ] \
  && echo "nvidia_rtx_6000_ada_generation" || echo "nvidia_rtx_a6000"; }

prev_status=""
while true; do
  # A node with a broken GPU says so in the log. Add it, and never use it.
  for f in $(ls -t outputs/jlensmm_*.out 2>/dev/null | head -40); do
    if grep -q "Failed to get device handle\|CUDA unknown error" "$f" 2>/dev/null; then
      h=$(grep -oE "HOST=[^ .]+" "$f" | head -1 | cut -d= -f2)
      if [ -n "$h" ] && ! grep -q "$h" "$BAD"; then
        printf '%s,%s' "$(cat "$BAD")" "$h" > "$BAD.tmp" && mv "$BAD.tmp" "$BAD"
        echo "BADNODE add $h"
      fi
    fi
  done

  live=$(squeue -u mwf62 -h -o "%j" 2>/dev/null | grep -c '^jlens-s')
  done_n=$(ls $OUT/shard*of16.pt 2>/dev/null | wc -l)
  ckpt_n=$(ls $OUT/fit_ckpt_shard*of16.pt 2>/dev/null | wc -l)
  resub=0

  for i in $(seq 0 $((N-1))); do
    nn=$(printf '%02d' "$i")
    [ -f "$OUT/shard${i}of16.pt" ] && continue
    squeue -u mwf62 -h -o "%j" 2>/dev/null | grep -qx "jlens-s$nn" && continue
    tries_f=$STATE/tries_s$nn
    tries=$(cat "$tries_f" 2>/dev/null || echo 0)
    if [ "$tries" -ge "$MAX_TRIES" ]; then
      continue
    fi
    tries=$((tries+1)); echo "$tries" > "$tries_f"
    jid=$(sbatch --parsable -J "jlens-s$nn" \
          --gres=gpu:$(gres_for "$i"):1 \
          --exclude="$(cat "$BAD")" \
          research/monocle/jlens_fit_preempt.sub \
          --shard "$i/16" --all-layers --dim-batch 4 --n-images 128 \
          --out-dir "$OUT" 2>/dev/null)
    if [ -n "$jid" ]; then
      echo "RESUBMIT shard $i as $jid (try $tries)"
      resub=$((resub+1))
    else
      echo "RESUBMIT shard $i FAILED to submit (try $tries)"
    fi
  done

  status="live=$live done=$done_n/$N ckpt=$ckpt_n resubmitted=$resub"
  if [ "$status" != "$prev_status" ]; then
    echo "STATUS $status"
    prev_status="$status"
  fi
  [ "$done_n" -ge "$N" ] && { echo "ALL $N SHARDS COMPLETE"; break; }
  sleep 120
done
