#!/usr/bin/env bash
# Build the project environment from uv.lock: vLLM 0.25, torch 2.11 + CUDA 12.9.
#
# pyproject.toml gives the dependencies and the extras. uv.lock gives the exact
# versions. [tool.uv.sources] in pyproject.toml gets torch from the PyTorch
# cu129 index and vLLM from the cu129 wheel of the vLLM release.
#
# Usage:
#   bash scripts/build_venv.sh                                  # .venv, the core only
#   bash scripts/build_venv.sh --extra wandb --extra embed      # add extras
#   bash scripts/build_venv.sh --all-extras                     # all extras
#   VENV=/path/to/venv bash scripts/build_venv.sh --extra research
#
# The extras: wandb, speech, embed, flash-attn, langextract, urban, research,
# notebooks, cambrian. flash-attn has no wheel for this torch: uv builds it from
# source, and that takes about 1 hour on a node with nvcc.
#
# Warning: `uv sync` makes the venv the same as the lock. It removes each
# package that the lock and the selected extras do not list. To keep the
# other packages, add --inexact.
#
# Warning: after you change a venv that has a /scratch mirror, sync the mirrors:
#   bash scripts/sync_venv_to_scratch.sh
#   bash scripts/sync_venv_to_scratch.sh --make-tarball
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"

command -v uv >/dev/null || { echo "build_venv: uv is not on PATH" >&2; exit 1; }

echo "[build] venv = $VENV"
cd "$ROOT"
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen "$@"

echo "[build] done. Installed versions:"
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
for name in ("vllm", "torch", "transformers", "flashinfer-python", "flash-attn",
             "triton", "numpy", "xgrammar", "image-forward"):
    try:
        print(f"  {name:20s} {md.version(name)}")
    except Exception:
        print(f"  {name:20s} not installed")
PY
