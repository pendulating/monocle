#!/usr/bin/env bash
# Count what a trace-extraction launch landed and what it finished.
#
# Usage:
#   bash scripts/check_trace_extraction.sh <experiment>    # CUESV1Q, CUESV1GG, CUESV1QQ, CUESV2Q, CUESV2QQ, ...
#
# A monitor that sshd refused exits with success and never submits its stage
# job, thus a shard can be lost in silence. This script counts, for each
# case, the monitors that landed a stage job and the shards that wrote their
# final parquet.
#
# A finished shard holds `outputs/extractions/<case>_s<k>.parquet`. A shard
# in flight holds chunk files under `outputs/extractions/chunks/`; a full
# shard holds 6.

cd /share/ju/matt/mllmsci
EXPERIMENT="${1:-CUESV1Q}"
shopt -s nullglob
# 2 layouts. The monitor path writes `multirun/<date>_<EXP>/<time>/<k>/`. The
# monitor-free path of `submit_trace_shards.py` writes
# `multirun/<date>_<EXP>/runs/<case>__<shard>/` and `submitted_jobs.txt`.
sweeps=()
for base in multirun/*_${EXPERIMENT}/; do
  if [ -d "$base/runs" ]; then sweeps+=("$base"); fi
  for t in "$base"[0-9][0-9]-[0-9][0-9]-[0-9][0-9]/; do sweeps+=("$t"); done
done
if [ ${#sweeps[@]} -eq 0 ]; then
  echo "no $EXPERIMENT sweep directory yet"
  exit 0
fi
printf "%-20s %-12s %-8s %-8s %-8s\n" case sweep landed done chunks
for s in "${sweeps[@]}"; do
  # The monitor-free layout: "landed" means the array task was submitted.
  if [ -d "$s/runs" ]; then
    declare -A total_c=() done_c=() chunks_c=() landed_c=()
    for n in "$s"runs/*__[0-9]*/; do
      case=$(basename "$n" | sed 's/__[0-9]*$//')
      total_c[$case]=$(( ${total_c[$case]:-0} + 1 ))
      f=("$n"outputs/extractions/*_s*.parquet); done_c[$case]=$(( ${done_c[$case]:-0} + ${#f[@]} ))
      c=("$n"outputs/extractions/chunks/*.parquet); chunks_c[$case]=$(( ${chunks_c[$case]:-0} + ${#c[@]} ))
    done
    if [ -f "$s/submitted_jobs.txt" ]; then
      while IFS=$'\t' read -r jid case shard; do
        [ -n "$case" ] && landed_c[$case]=$(( ${landed_c[$case]:-0} + 1 ))
      done < "$s/submitted_jobs.txt"
    fi
    for case in "${!total_c[@]}"; do
      printf "%-20s %-12s %-8s %-8s %-8s\n" "$case" "$(basename "$s")" \
        "${landed_c[$case]:-0}/${total_c[$case]}" "${done_c[$case]}/${total_c[$case]}" "${chunks_c[$case]}"
    done | sort
    continue
  fi
  case=""
  landed=0; done_=0; chunks=0; total=0
  for n in "$s"[0-9]*/; do
    total=$((total + 1))
    [ -z "$case" ] && case=$(grep -o "results_path=.*" "$n.hydra/overrides.yaml" 2>/dev/null \
      | sed 's#.*/pairwise/##; s#_mvp_.*##' | head -1)
    ls "$n".slurm_jobs/extract/*_submitted.pkl >/dev/null 2>&1 && landed=$((landed + 1))
    f=("$n"outputs/extractions/*_s*.parquet); done_=$((done_ + ${#f[@]}))
    c=("$n"outputs/extractions/chunks/*.parquet); chunks=$((chunks + ${#c[@]}))
  done
  printf "%-20s %-12s %-8s %-8s %-8s\n" "${case:-?}" "$(basename "$s")" "$landed/$total" "$done_/$total" "$chunks"
done
echo
echo "jobs in the queue:"
squeue -u "$USER" -h -o "%j %T %P" | grep -E "${EXPERIMENT}-monitor|${EXPERIMENT}-shard|PAIRVQA-extract" | sort | uniq -c
