#!/usr/bin/env bash
# Build a pinned venv on node-local /scratch, one time on each node.
#
# The whatami SGLang venv (Llama 3.2 Vision) lives only on /scratch. An install
# onto NFS writes about 90k small files and did not end after 30 min
# (2026-10-20, g3-compute-15). On /scratch with the uv cache there, a build
# takes about 2 to 3 min.
#
# The marker holds the sha of the requirements file. A change to the pins
# makes the script build the venv again.
#
# Warning: compile a lock file with `uv pip compile --no-config`. Without it,
# uv reads the repo pyproject.toml, and its override (transformers>=5.5)
# replaces the transformers pin.
#
# Usage: bash scripts/whatami/ensure_scratch_venv.sh <name> <requirements.txt>
# Prints the venv path.
set -euo pipefail

NAME="$1"
REQ="$(readlink -f "$2")"
DST=/scratch/$USER/venvs/$NAME
PYTHON=/share/apps/software/anaconda3/bin/python3.12
SUM="$(sha256sum "$REQ" | cut -c1-16)"
export UV_CACHE_DIR=/scratch/$USER/uv-cache UV_LINK_MODE=copy
# The lock mixes the PyTorch cu130 index and PyPI; take each pin where it is.
export UV_INDEX_STRATEGY=unsafe-best-match
export PATH=/share/ju/matt/bin:$PATH

LOCK="$DST.lock"
mkdir -p "$(dirname "$DST")"
for _ in $(seq 1 720); do mkdir "$LOCK" 2>/dev/null && break; sleep 5; done
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

if [ -f "$DST/.sync_complete" ] && grep -qx "req_sha=$SUM" "$DST/.sync_complete"; then
  echo "[scratch_venv] OK: $DST" >&2
else
  echo "[scratch_venv] build $DST from $REQ" >&2
  start=$SECONDS
  rm -rf "$DST"
  # Run uv outside the repo, thus it does not read the repo pyproject.toml.
  (cd /scratch/$USER && uv venv --no-config --python "$PYTHON" "$DST" >&2 \
     && VIRTUAL_ENV=$DST uv pip install --no-config -r "$REQ" >&2)
  echo "req_sha=$SUM" > "$DST/.sync_complete"
  echo "[scratch_venv] built in $((SECONDS - start)) s" >&2
fi
echo "$DST"
