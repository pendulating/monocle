#!/usr/bin/env bash
# Mirror the canonical zoo models to node-local /scratch (the model registry).
#
# The marker convention is the same as sync_venv_to_scratch.sh and
# activate_stage_venv.sh: each mirror holds <mirror>/.sync_complete with a
# `src=<zoo path>` line, and dagspaces.common.model_registry sends model loads
# to the mirror only when that marker agrees. With no marker, or with a marker
# that does not agree, the stages fall back to the NFS zoo path. A partial or
# interrupted sync is thus safe.
#
# Usage:
#   scripts/sync_model_registry_to_scratch.sh              # the canonical set
#   scripts/sync_model_registry_to_scratch.sh NAME [...]   # specific zoo dirs
#
# Run this one time on each node:
#   klara:  sbatch -p pierson -w klara -c 4 --mem=8G \
#             --wrap 'bash /share/pierson/matt/mllmsci/scripts/sync_model_registry_to_scratch.sh'
#   ju:     sbatch -p ju -c 4 --mem=8G \
#             --wrap 'bash /share/pierson/matt/mllmsci/scripts/sync_model_registry_to_scratch.sh'
#
# The zoo models do not change after the download. If one IS replaced in place,
# run this script again on each node — the mirrors do not test freshness.
set -euo pipefail

ZOO="${MLLMSCI_MODEL_ZOO:-/share/pierson/matt/zoo/models}"
REG="${MLLMSCI_MODEL_REGISTRY:-/scratch/$USER/registry/models}"

# The canonical urbanpairvqa roster (the 5-model klara2x sweep, plus the
# 12B model that runs alone on klara1x). Phi-4 is deliberately absent: it
# gives degenerate output on each pairvqa task and is excluded from all
# sweeps. Pass other names on the command line when you need them, for
# example the vision models (Qwen3-VL-2B-Instruct, Qwen3-VL-30B-A3B-Instruct)
# or the speech models (granite-speech-4.1-2b).
CANONICAL=(
  Qwen3.5-9B
  gemma-4-12B-it
  Gemma-4-E2B-it
  Gemma-4-E4B-it
  Qwen3.5-2B
  Qwen3.5-4B
)

if [ "$#" -gt 0 ]; then MODELS=("$@"); else MODELS=("${CANONICAL[@]}"); fi

mkdir -p "$REG"
for name in "${MODELS[@]}"; do
  SRC="$ZOO/$name"
  DST="$REG/$name"
  if [ ! -d "$SRC" ]; then
    echo "sync_model_registry: skip $name — $SRC not found" >&2
    continue
  fi
  mkdir -p "$DST"
  if [ -f "$DST/.sync_complete" ] && ! grep -q "src=$SRC\$" "$DST/.sync_complete"; then
    echo "sync_model_registry: refused $name — $DST is a mirror of a different source:" >&2
    sed 's/^/  /' "$DST/.sync_complete" >&2
    continue
  fi
  (
    flock -n 9 || { echo "sync_model_registry: skip $name — another sync holds the lock"; exit 0; }
    rm -f "$DST/.sync_complete"
    echo "sync_model_registry: sync $name ..."
    rsync -a --delete --exclude=/.sync_complete --exclude=/.sync_lock "$SRC/" "$DST/"
    files=$(find "$DST" -type f ! -name '.sync_lock' | wc -l)
    bytes=$(du -sb "$DST" | cut -f1)
    {
      echo "src=$SRC"
      echo "host=$(hostname -s)"
      echo "date=$(date -Is)"
      echo "files=$files bytes=$bytes"
    } > "$DST/.sync_complete"
    echo "sync_model_registry: done $name ($files files, $bytes bytes)"
  ) 9>"$DST/.sync_lock"
done
echo "sync_model_registry: complete on $(hostname -s) -> $REG"
