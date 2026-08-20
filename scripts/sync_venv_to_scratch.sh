#!/usr/bin/env bash
# Mirror the shared-NFS venv onto node-local /scratch for fast imports.
#
# Why: the venv holds about 90k mostly-small files. NFS sequential bandwidth is
# adequate (about 170 MB/s), but per-file round trips dominate. Anything that
# walks the tree file-by-file (python imports, single-stream rsync) is
# latency-bound. A cold `import torch` + `import vllm` + `import flashinfer`
# from NFS costs about 13 minutes for each process spawn, and a vLLM stage
# spawns three processes. From local disk it is seconds.
#
# vLLM 0.25 makes this worse than 0.19: it imports FlashInfer at start, and
# flashinfer_cubin alone is 1.9 GB in about 20k files.
#
# Fast path: a venv tarball on NFS (built by --make-tarball from a complete
# local mirror) is ONE sequential stream — about 2 minutes to a new node.
# Fallback: a parallel rsync fan-out (NFS latency-bound work scales with the
# number of streams). Both paths end with a serial `rsync --delete` pass
# against the live NFS venv, which gives exact 1:1 parity. Only then does the
# script write the completion marker. The launchers refuse a mirror that has no
# marker (see scripts/activate_stage_venv.sh).
#
# Usage:
#   sync_venv_to_scratch.sh [SRC_VENV] [DST_VENV]   # deploy or refresh a mirror
#   sync_venv_to_scratch.sh --make-tarball [SRC_VENV] [DST_VENV]
#       # build the NFS bootstrap tarball from a complete local mirror
#
# Run this again each time the shared venv changes. It is incremental after the
# first run.
set -euo pipefail

MAKE_TARBALL=0
if [ "${1:-}" = "--make-tarball" ]; then MAKE_TARBALL=1; shift; fi

SRC="${1:-/share/pierson/matt/mllmsci/.venv-mllmsci-vllm025cu129}"
NAME="$(basename "$SRC" | sed 's/^\.//')"
DST="${2:-/scratch/$USER/venvs/$NAME}"
TARBALL="${SYNC_VENV_TARBALL:-$(dirname "$SRC")/.venv-mirrors/$NAME.tar.zst}"
JOBS="${SYNC_VENV_JOBS:-24}"
PYVER="${SYNC_VENV_PYVER:-python3.12}"

[ -d "$SRC" ] || { echo "sync_venv_to_scratch: SRC not found: $SRC" >&2; exit 1; }
case "$DST" in /scratch/*) ;; *) echo "sync_venv_to_scratch: DST must be under /scratch, got: $DST" >&2; exit 1;; esac

if [ "$MAKE_TARBALL" = 1 ]; then
  # Build from the LOCAL mirror (fast reads), never from NFS. The script needs
  # a complete mirror, so that the tarball cannot hold a half-synced tree.
  [ -f "$DST/.sync_complete" ] || { echo "sync_venv_to_scratch: no complete mirror at $DST — sync first" >&2; exit 1; }
  grep -q "src=$SRC\$" "$DST/.sync_complete" || { echo "sync_venv_to_scratch: the mirror at $DST is not a mirror of $SRC" >&2; exit 1; }
  mkdir -p "$(dirname "$TARBALL")"
  echo "[sync_venv] build $TARBALL from $DST"
  start=$SECONDS
  tar -C "$DST" --exclude=./.sync_complete -cf - . | zstd -T0 -3 -q -o "$TARBALL.tmp" -f
  mv -f "$TARBALL.tmp" "$TARBALL"
  echo "[sync_venv] tarball done in $((SECONDS - start))s: $(du -sh "$TARBALL" | cut -f1)"
  exit 0
fi

mkdir -p "$DST"
echo "[sync_venv] $SRC -> $DST (jobs=$JOBS)"
start=$SECONDS

# A mirror with no marker is partial or stale data from an interrupted sync. A
# marker for a different source venv means the names collided. Start again.
if [ -e "$DST/.sync_complete" ] && ! grep -q "src=$SRC\$" "$DST/.sync_complete"; then
  echo "sync_venv_to_scratch: $DST is a mirror of a different venv ($(cat "$DST/.sync_complete")) — refused" >&2
  exit 1
fi
rm -f "$DST/.sync_complete"

if [ ! -x "$DST/bin/python" ] && [ -f "$TARBALL" ]; then
  # New node and a tarball is available: one sequential NFS stream, then a
  # local extraction. About 2 minutes, instead of 30 minutes of parallel
  # per-file round trips.
  echo "[sync_venv] bootstrap from tarball $TARBALL ($(du -sh "$TARBALL" | cut -f1))"
  zstd -dc -T0 "$TARBALL" | tar -C "$DST" -xf -
  echo "[sync_venv] tarball extracted at ${SECONDS}s; reconcile against the live venv"
else
  # No tarball, or a refresh of an existing mirror: a parallel rsync fan-out at
  # the deepest broad level — one rsync for each site-packages child, plus each
  # venv entry that is not site-packages.
  SP_REL="lib/$PYVER/site-packages"
  [ -d "$SRC/$SP_REL" ] || { echo "sync_venv_to_scratch: no $SP_REL in $SRC (set SYNC_VENV_PYVER)" >&2; exit 1; }
  {
    # venv top-level entries, but not lib (bin, include, pyvenv.cfg, ...)
    find "$SRC" -mindepth 1 -maxdepth 1 ! -name lib
    # the lib subtree down to site-packages, exclusive
    find "$SRC/lib" -mindepth 1 -maxdepth 2 ! -path "$SRC/$SP_REL" ! -name "$PYVER"
    # each package inside site-packages
    find "$SRC/$SP_REL" -mindepth 1 -maxdepth 1
  } | sed "s|^$SRC/||" \
    | xargs -P "$JOBS" -I{} rsync -a --relative "$SRC/./{}" "$DST/"
fi

# Serial parity pass against the live NFS venv. This propagates deletions,
# drift since the tarball was built, and anything the fan-out missed.
rsync -a --delete --exclude=/.sync_complete "$SRC/" "$DST/"

# Write the completion marker. The launchers trust the mirror only when this
# marker exists, so a partial or killed sync can never become a working venv.
date -u +"%Y-%m-%dT%H:%M:%SZ src=$SRC" > "$DST/.sync_complete"

echo "[sync_venv] done in $((SECONDS - start))s, $(du -sh "$DST" | cut -f1) on $(hostname -s)"
