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
#   scripts/sync_model_registry_to_scratch.sh              # the models of IMAGE_FORWARD_MIRROR_MODELS
#   scripts/sync_model_registry_to_scratch.sh NAME [...]   # these zoo folders
#
# The keys come from server.env:
#   IMAGE_FORWARD_MODEL_ZOO       the zoo folder (necessary)
#   IMAGE_FORWARD_MODEL_REGISTRY  the mirror folder (default /scratch/$USER/registry/models)
#   IMAGE_FORWARD_MIRROR_MODELS   the zoo folder names to mirror, with a space between them
#
# Run this one time on each node, from the repo root:
#   sbatch -p <partition> -w <node> -c 4 --mem=8G \
#     --wrap "bash $PWD/scripts/sync_model_registry_to_scratch.sh"
#
# The zoo models do not change after the download. If one IS replaced in place,
# run this script again on each node — the mirrors do not test freshness.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/load_server_env.sh"

ZOO="${IMAGE_FORWARD_MODEL_ZOO:?set IMAGE_FORWARD_MODEL_ZOO in server.env}"
REG="${IMAGE_FORWARD_MODEL_REGISTRY:-/scratch/$USER/registry/models}"

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  read -r -a MODELS <<< "${IMAGE_FORWARD_MIRROR_MODELS:-}"
fi
if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "sync_model_registry: no models. Give the names, or set IMAGE_FORWARD_MIRROR_MODELS in server.env." >&2
  exit 2
fi

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
