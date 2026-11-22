#!/usr/bin/env bash
# Copy the model weights and the probe images to node-local /scratch.
#
# The rule is the same as for the venv mirrors (vlm-narratives-docs/scratch-mirrors.md):
# a copy is valid only when its .sync_complete marker names the same source.
# The script writes the marker last, thus a killed copy is never used.
#
# Usage (sourced or run): bash research/scripts/whatami/stage_inputs.sh <model_dir> <sample.parquet> <python>
# Prints two lines: MODEL_LOCAL=<path> and IMAGE_ROOT=<path>.
set -euo pipefail

SRC_MODEL="$1"
SAMPLE="$2"
PY="$3"
ROOT=/scratch/$USER/whatami
DST_MODEL="$ROOT/models/$(basename "$SRC_MODEL")"
IMAGE_ROOT="$ROOT/images/$(basename "$SAMPLE" .parquet)"
mkdir -p "$ROOT/models" "$ROOT/images"

# One job at a time does the copy on a node. The lock is an atomic mkdir.
LOCK="$ROOT/.stage.lock"
for _ in $(seq 1 720); do mkdir "$LOCK" 2>/dev/null && break; sleep 5; done
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

# ── the model ─────────────────────────────────────────────────────────
# /scratch is shared with other users and can be full (F4 smoke jobs 91678 to
# 91680: "No space left on device"). When the free space is less than the
# model size plus 20 GB, the server reads the weights from NFS. The load is
# slower, but the job runs.
need_kb=$(( $(du -sk --exclude original --exclude .cache "$SRC_MODEL" | cut -f1) + 20 * 1024 * 1024 ))
free_kb=$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')
if [ -f "$DST_MODEL/.sync_complete" ] && grep -qx "src=$SRC_MODEL" "$DST_MODEL/.sync_complete"; then
  echo "[stage] model mirror OK: $DST_MODEL" >&2
elif [ "$free_kb" -lt "$need_kb" ]; then
  echo "[stage] /scratch has $((free_kb / 1048576)) GB free, the copy needs $((need_kb / 1048576)) GB: read the weights from NFS" >&2
  DST_MODEL="$SRC_MODEL"
else
  echo "[stage] copy $SRC_MODEL -> $DST_MODEL" >&2
  start=$SECONDS
  rm -f "$DST_MODEL/.sync_complete"
  # The weights are few large files, thus one rsync stream is enough.
  rsync -a --delete --exclude original/ --exclude .cache/ "$SRC_MODEL/" "$DST_MODEL/"
  echo "src=$SRC_MODEL" > "$DST_MODEL/.sync_complete"
  echo "[stage] model copied in $((SECONDS - start)) s ($(du -sh "$DST_MODEL" | cut -f1))" >&2
fi

# ── the images ────────────────────────────────────────────────────────
SAMPLE_ABS="$(readlink -f "$SAMPLE")"
SAMPLE_SUM="$(sha256sum "$SAMPLE_ABS" | cut -c1-16)"
if [ -f "$IMAGE_ROOT/.sync_complete" ] && grep -qx "src=$SAMPLE_ABS sha=$SAMPLE_SUM" "$IMAGE_ROOT/.sync_complete"; then
  echo "[stage] image mirror OK: $IMAGE_ROOT" >&2
else
  echo "[stage] copy the images of $SAMPLE_ABS -> $IMAGE_ROOT" >&2
  rm -rf "$IMAGE_ROOT"; mkdir -p "$IMAGE_ROOT"
  "$PY" - "$SAMPLE_ABS" "$IMAGE_ROOT" <<'PY'
import shutil, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
df = pd.read_parquet(sys.argv[1], columns=["sample_id", "image_path"])
dst = sys.argv[2]
with ThreadPoolExecutor(16) as ex:
    list(ex.map(lambda r: shutil.copyfile(r[1], f"{dst}/{r[0]}.jpg"), df.itertuples(index=False)))
print(f"[stage] {len(df)} images copied", file=sys.stderr)
PY
  echo "src=$SAMPLE_ABS sha=$SAMPLE_SUM" > "$IMAGE_ROOT/.sync_complete"
fi

echo "MODEL_LOCAL=$DST_MODEL"
echo "IMAGE_ROOT=$IMAGE_ROOT"
