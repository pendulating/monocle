#!/usr/bin/env bash
# Assemble a minimal CUDA 12.9 toolkit for the FlashInfer JIT.
#
# Why: FlashInfer builds Blackwell (sm_120) kernels as `compute_120f`, and the
# `f` family target needs nvcc 12.9 or later. The system toolkit on these nodes
# is 12.8, which compiles sm_120 and sm_120a but NOT sm_120f, thus FlashInfer
# refuses the card and the job dies with the misleading message
# "FlashInfer requires GPUs with sm75 or higher".
#
# The PyPI wheel `nvidia-cuda-nvcc-cu12` 12.9.86 ships ptxas ONLY, with no
# nvcc. The NVIDIA redistributable tarball of the same version ships the real
# compiler, thus this script takes that.
#
# torch is built against CUDA 12.9, thus this toolkit MATCHES torch. The 12.8
# system toolkit never did.
#
# Usage: scripts/install_cuda_129_toolkit.sh [PREFIX]
# PREFIX is the install folder. The default is $IMAGE_FORWARD_CUDA_HOME, or else
# $HOME/cuda-12.9. Then set IMAGE_FORWARD_CUDA_HOME=PREFIX in server.env.
set -euo pipefail

PREFIX="${1:-${IMAGE_FORWARD_CUDA_HOME:-$HOME/cuda-12.9}}"
BASE="https://developer.download.nvidia.com/compute/cuda/redist"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# nvcc and the headers it needs. libcublas stays out: it is 934 MB and the
# FlashInfer sampling kernels do not include it. Add it if a build asks.
COMPONENTS=(
  "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz"
  "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.9.79-archive.tar.xz"
  "libcurand/linux-x86_64/libcurand-linux-x86_64-10.3.10.19-archive.tar.xz"
  "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.9.27-archive.tar.xz"
  "cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-12.9.86-archive.tar.xz"
  "cuda_profiler_api/linux-x86_64/cuda_profiler_api-linux-x86_64-12.9.79-archive.tar.xz"
)

mkdir -p "$PREFIX"
for rel in "${COMPONENTS[@]}"; do
  name="$(basename "$rel")"
  echo "[cuda129] fetch $name"
  curl -fsSL "$BASE/$rel" -o "$WORK/$name"
  echo "[cuda129] extract $name"
  tar -C "$WORK" -xf "$WORK/$name"
  # Every archive holds 1 top directory with bin/, include/, lib/ inside.
  top="$WORK/${name%-archive.tar.xz}-archive"
  cp -a "$top"/. "$PREFIX"/
  rm -rf "$top" "$WORK/$name"
done

# The redistributable tarballs put the libraries in `lib`. Every build system
# that follows the classic layout links `-L$CUDA_HOME/lib64`, thus FlashInfer
# fails at the LINK step with "cannot find -lcudart" even after nvcc compiled
# every object. The alias costs nothing and fixes that.
[ -d "$PREFIX/lib" ] && [ ! -e "$PREFIX/lib64" ] && ln -s lib "$PREFIX/lib64"

echo "[cuda129] done: $PREFIX"
"$PREFIX/bin/nvcc" --version | tail -2
echo "[cuda129] curand header: $([ -f "$PREFIX/include/curand.h" ] && echo present || echo MISSING)"
