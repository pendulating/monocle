#!/usr/bin/env bash
# Make room on the node-local /scratch before a GPU stage starts.
#
# Why
# ---
# A stage writes its torch and triton JIT caches to TMPDIR, which is
# /scratch/$USER when the node holds a /scratch. On a node whose /scratch is
# full, torch dies before the model loads:
#   OSError: [Errno 28] No space left on device: '/scratch/mwf62'
# Measured 2026-08-18: 5 of 113 tasks of the 1,000,000-pair battery died so.
#
# The junk is almost always a JIT cache that an earlier run left. This script
# removes the old ones, thus the node recovers space and the stage keeps the
# fast node-local path instead of falling back to /tmp.
#
# WHAT IT NEVER TOUCHES
# ---------------------
#   /scratch/$USER/venvs/      the venv mirrors (see sync_venv_to_scratch.sh)
#   /scratch/$USER/registry/   the model mirrors
# Both cost about 2 minutes each to rebuild and are the reason the node-local
# path is fast at all. The cleanup names the cache dirs it deletes and touches
# nothing else.
#
# CONCURRENCY
# -----------
# Many stage jobs start on one node at the same time. Only 1 does the cleanup:
# the lock is an atomic `mkdir`. A job that does not get the lock goes straight
# on. An age filter keeps the cleanup off a cache that a live job still uses.
#
# Usage:
#   bash scripts/prepare_scratch.sh              # clean, then report
#   bash scripts/prepare_scratch.sh --dry-run    # report only
#   SCRATCH_CLEAN_AGE_DAYS=1 bash scripts/prepare_scratch.sh
set -uo pipefail

AGE_DAYS="${SCRATCH_CLEAN_AGE_DAYS:-2}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

ROOT=/scratch
[ -d "${ROOT}" ] || { echo "[scratch] no /scratch on $(hostname -s); nothing to do"; exit 0; }

MINE="${ROOT}/${USER}"
mkdir -p "${MINE}" 2>/dev/null || {
  echo "[scratch] cannot create ${MINE}; the stage will fall back to /tmp"; exit 0; }

_free_gb() { df -Pk "${ROOT}" 2>/dev/null | tail -1 | tr -s ' ' | cut -d' ' -f4 \
  | tr -dc 0-9 | awk '{printf "%.1f", $1/1048576}'; }

BEFORE="$(_free_gb)"

# The cache dirs this project writes. Everything here is rebuildable.
# NB: `venvs` and `registry` are deliberately absent — see the header.
CACHES=(triton inductor "torchinductor_${USER}" vllm flashinfer .cache)

LOCK="${MINE}/.prepare_scratch.lock"
# Drop a lock that a killed job left behind.
if [ -d "${LOCK}" ]; then
  if [ -z "$(find "${LOCK}" -maxdepth 0 -mmin -30 2>/dev/null)" ]; then
    rmdir "${LOCK}" 2>/dev/null
  fi
fi

if ! mkdir "${LOCK}" 2>/dev/null; then
  echo "[scratch] another job on $(hostname -s) is cleaning; free=${BEFORE}GB"
  exit 0
fi
trap 'rmdir "${LOCK}" 2>/dev/null' EXIT

removed=0
for name in "${CACHES[@]}"; do
  d="${MINE}/${name}"
  [ -d "${d}" ] || continue
  # Count first, so the report is honest even under --dry-run.
  n="$(find "${d}" -mindepth 1 -maxdepth 2 -mtime "+${AGE_DAYS}" 2>/dev/null | wc -l)"
  [ "${n}" -eq 0 ] && continue
  removed=$((removed + n))
  if [ "${DRY}" -eq 0 ]; then
    find "${d}" -mindepth 1 -maxdepth 2 -mtime "+${AGE_DAYS}" \
      -exec rm -rf {} + 2>/dev/null
  fi
done

# Temp dirs that a killed vLLM or submitit run left directly under our root.
for pat in "tmp*" "*_dp_parts_*" "core.*"; do
  n="$(find "${MINE}" -mindepth 1 -maxdepth 1 -name "${pat}" -mtime "+${AGE_DAYS}" 2>/dev/null | wc -l)"
  [ "${n}" -eq 0 ] && continue
  removed=$((removed + n))
  [ "${DRY}" -eq 0 ] && find "${MINE}" -mindepth 1 -maxdepth 1 -name "${pat}" \
    -mtime "+${AGE_DAYS}" -exec rm -rf {} + 2>/dev/null
done

AFTER="$(_free_gb)"
if [ "${DRY}" -eq 1 ]; then
  echo "[scratch] $(hostname -s): would remove ${removed} entries older than ${AGE_DAYS}d; free=${BEFORE}GB"
else
  echo "[scratch] $(hostname -s): removed ${removed} entries older than ${AGE_DAYS}d; free ${BEFORE}GB -> ${AFTER}GB"
fi
exit 0
