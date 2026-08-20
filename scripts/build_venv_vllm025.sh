#!/usr/bin/env bash
# Build the vLLM 0.25 / CUDA 12.9 virtual environment for mllmsci.
#
# The environment is `.venv-vllm025cu129` at the project root. It replaces the
# two-environment split that came before it:
#   .venv-3.12    vLLM 0.19.0, torch 2.10.0+cu128   (qwen3.5 path)
#   .venv-nightly vLLM 0.23.1,  torch 2.11.0+cu129   (gemma-4-12b path)
#
# The version pins for the CUDA stack are in requirements-vllm025cu129.txt.
# The remaining packages come from pyproject.toml.
#
# Usage:
#   bash scripts/build_venv_vllm025.sh              # build (or repair) the venv
#   VENV=/path/to/venv bash scripts/build_venv_vllm025.sh
#
# The script is safe to run again. It does not delete an existing environment.
#
# Warning: after you change this environment, sync the /scratch mirrors again:
#   bash scripts/sync_venv_to_scratch.sh
#   bash scripts/sync_venv_to_scratch.sh --make-tarball
set -euo pipefail

ROOT="${MLLMSCI_PROJECT_ROOT:-/share/pierson/matt/mllmsci}"
VENV="${VENV:-$ROOT/.venv-vllm025cu129}"
CONSTRAINTS="$ROOT/requirements-vllm025cu129.txt"
VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.25.0/vllm-0.25.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
TORCH_INDEX="https://download.pytorch.org/whl/cu129"

# flash-attn has no published wheel for torch 2.11+cu129, and a source build
# costs about an hour. UAIR built it once; copy that result, because both
# environments use CPython 3.12 and the same torch ABI.
FA_SRC="${FA_SRC:-/share/pierson/matt/UAIR/.venv-vllm025cu129/lib/python3.12/site-packages}"

command -v uv >/dev/null || { echo "build_venv_vllm025: uv is not on PATH" >&2; exit 1; }
[ -f "$CONSTRAINTS" ] || { echo "build_venv_vllm025: no constraints file at $CONSTRAINTS" >&2; exit 1; }

echo "[build] venv       = $VENV"
echo "[build] constraints= $CONSTRAINTS"

# ── 1. the environment ────────────────────────────────────────────────
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12 "$VENV"
fi
SP="$VENV/lib/python3.12/site-packages"

# `--index-strategy unsafe-best-match` is necessary when an extra index is
# active. The PyTorch index carries old copies of common packages (packaging,
# jinja2, ...), and by default uv accepts only the first index that holds a
# name. That rule stops the resolution with, for example,
# "packaging was found on download.pytorch.org, but not at the requested
# version". Both indexes here are official, so it is safe to look at all of
# them. The pins in the constraints file still control the CUDA stack.
pip_install() {
  uv pip install --python "$VENV/bin/python" \
    --constraint "$CONSTRAINTS" --index-strategy unsafe-best-match "$@"
}

# ── 2. torch first, from the cu129 index ──────────────────────────────
# vLLM 0.25 links against torch. Install torch before vLLM, so that no
# later step can pull the default-index cu128 build in as a dependency.
echo "[build] step 1/4: torch 2.11.0+cu129"
pip_install --index-url "$TORCH_INDEX" \
  torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129

# ── 3. flash-attn, copied from the UAIR build ─────────────────────────
echo "[build] step 2/4: flash-attn"
if [ -d "$SP/flash_attn" ]; then
  echo "[build]   flash_attn is already present — skip"
elif [ -d "$FA_SRC/flash_attn" ]; then
  cp -a "$FA_SRC/flash_attn" "$SP/"
  cp -a "$FA_SRC"/flash_attn-*.dist-info "$SP/"
  cp -a "$FA_SRC"/flash_attn_2_cuda.cpython-312-*.so "$SP/"
  echo "[build]   copied flash_attn from $FA_SRC"
else
  echo "[build]   WARNING: no flash_attn at $FA_SRC — build it with" >&2
  echo "[build]   MAX_JOBS=8 uv pip install --python $VENV/bin/python \\" >&2
  echo "[build]     --no-build-isolation flash-attn==2.8.3.post1" >&2
fi

# ── 4. vLLM 0.25 ──────────────────────────────────────────────────────
echo "[build] step 3/4: vLLM 0.25.0+cu129"
pip_install --extra-index-url "$TORCH_INDEX" "vllm @ $VLLM_WHEEL"

# ── 5. the project and its remaining dependencies ─────────────────────
# `--no-build-isolation-package flash-attn` stops uv from starting a source
# build when it resolves the pyproject `flash-attn` requirement.
echo "[build] step 4/4: mllmsci (editable) + remaining dependencies"
pip_install --extra-index-url "$TORCH_INDEX" \
  --no-build-isolation-package flash-attn \
  --override "$CONSTRAINTS" \
  -e "$ROOT"

# ── 6. report ─────────────────────────────────────────────────────────
echo "[build] done. Installed versions:"
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
for name in ("vllm", "torch", "transformers", "flashinfer-python", "flash-attn",
             "triton", "numpy", "xgrammar"):
    try:
        print(f"  {name:20s} {md.version(name)}")
    except Exception as exc:
        print(f"  {name:20s} MISSING ({exc})")
PY
echo "[build] next: bash scripts/sync_venv_to_scratch.sh"
