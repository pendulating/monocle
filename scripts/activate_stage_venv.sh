# Sourced from the SLURM launcher setup blocks (not executed). This script
# does two jobs before python starts:
#
#   1. It selects the fastest venv that is available on the node.
#   2. It makes the FFmpeg shared libraries visible to torchcodec.
#
# Set IMAGE_FORWARD_SKIP_SCRATCH_VENV=1 before you source the script to skip job 1
# and keep job 2. The monitor launcher does this, because the monitor runs the
# driver: if it activated the mirror, sys.prefix would become a /scratch path,
# and no stage node could match the marker.

# ── 1. the node-local venv mirror ─────────────────────────────────────
# If the node holds a COMPLETE /scratch mirror of the venv that the driver runs
# from (see scripts/sync_venv_to_scratch.sh), the script activates the mirror
# and exports IMAGE_FORWARD_STAGE_PYTHON. The submitit command line then starts the
# stage from node-local disk (see
# dagspaces/common/orchestrator.py::_create_submitit_executor). A cold
# torch/vllm/flashinfer import over NFS costs about 13 minutes for each process
# spawn; from local disk it is seconds.
#
# The fallbacks are deliberately careful. On any doubt — no mirror, a partial
# sync, a mirror of a DIFFERENT venv, or IMAGE_FORWARD_DRIVER_VENV not set — the
# script leaves IMAGE_FORWARD_STAGE_PYTHON unset, and the stage runs exactly as
# before, from the driver interpreter over NFS.

if [ "${IMAGE_FORWARD_SKIP_SCRATCH_VENV:-0}" = "1" ]; then
  echo "[stage_venv] scratch mirror skipped on request (driver job)"
else
  _scratch_venv="${IMAGE_FORWARD_SCRATCH_VENV:-}"
  if [ -z "$_scratch_venv" ] && [ -n "${IMAGE_FORWARD_DRIVER_VENV:-}" ]; then
    # The name convention of sync_venv_to_scratch.sh: the leading dot is gone.
    _scratch_venv="/scratch/$USER/venvs/$(basename "$IMAGE_FORWARD_DRIVER_VENV" | sed 's/^\.//')"
  fi

  # A mirror is valid only when the marker names the SAME source venv.
  _mllmsci_mirror_ok() {
    [ -n "$1" ] \
      && [ -f "$1/.sync_complete" ] \
      && [ -x "$1/bin/python" ] \
      && [ -n "${IMAGE_FORWARD_DRIVER_VENV:-}" ] \
      && grep -q "src=$IMAGE_FORWARD_DRIVER_VENV\$" "$1/.sync_complete"
  }

  # ── the sync on demand ──────────────────────────────────────────────
  # A node that holds no mirror builds one HERE, before python starts. The
  # first job on a node pays about 2 minutes; every later job on that node
  # starts from local disk. Only klara was ever synced by hand, thus every
  # `gpu` partition node paid a cold NFS import of about 4 to 5 minutes, for
  # each job, for ever.
  #
  # Four guards keep this safe:
  #   1. A lock. Many tasks land on 1 node together. One syncs; the others
  #      wait for it and then use the mirror it made.
  #   2. The tarball. Without it the script would rsync 90k files from NFS,
  #      and 60 nodes would do that at once. No tarball, no sync — the job
  #      runs from NFS, as before.
  #   3. Free space. The venv needs about 15 GB.
  #   4. Every failure falls back to NFS. The stage always runs.
  #
  # Set IMAGE_FORWARD_SCRATCH_AUTOSYNC=0 to turn this off.
  _mllmsci_autosync() {
    local dst="$1" src="${IMAGE_FORWARD_DRIVER_VENV:-}" root
    [ "${IMAGE_FORWARD_SCRATCH_AUTOSYNC:-1}" = "1" ] || { echo "[stage_venv] autosync off"; return 1; }
    [ -n "$src" ] || return 1
    [ -d /scratch ] && [ -w /scratch ] || { echo "[stage_venv] no writable /scratch on $(hostname -s)"; return 1; }

    local tarball="${SYNC_VENV_TARBALL:-$(dirname "$src")/.venv-mirrors/$(basename "$src" | sed 's/^\.//').tar.zst}"
    if [ ! -f "$tarball" ]; then
      echo "[stage_venv] no bootstrap tarball ($tarball); NOT syncing from NFS by rsync"
      return 1
    fi

    local need_gb="${IMAGE_FORWARD_SCRATCH_MIN_FREE_GB:-25}" free_kb
    free_kb=$(df --output=avail -k /scratch 2>/dev/null | tail -1 | tr -dc 0-9)
    if [ -z "$free_kb" ] || [ "$free_kb" -lt $((need_gb * 1024 * 1024)) ]; then
      echo "[stage_venv] /scratch holds less than ${need_gb} GB free on $(hostname -s); the stage runs from NFS"
      return 1
    fi

    root="${IMAGE_FORWARD_PROJECT_ROOT:-/share/ju/matt/mllmsci}"
    mkdir -p "$(dirname "$dst")" 2>/dev/null
    echo "[stage_venv] no mirror on $(hostname -s) — sync starts (tarball $(du -h "$tarball" 2>/dev/null | cut -f1))"
    local t0=$SECONDS
    flock -w "${IMAGE_FORWARD_SCRATCH_AUTOSYNC_WAIT:-900}" "$(dirname "$dst")/.autosync.lock" \
      bash -c "if [ -f '$dst/.sync_complete' ] && grep -q 'src=$src\$' '$dst/.sync_complete'; then \
                 echo '[stage_venv] another task synced it while this one waited'; \
               else \
                 bash '$root/scripts/sync_venv_to_scratch.sh' '$src' '$dst'; \
               fi"
    local rc=$?
    echo "[stage_venv] sync ended rc=$rc after $((SECONDS - t0))s"
    return $rc
  }

  if ! _mllmsci_mirror_ok "$_scratch_venv"; then
    _mllmsci_autosync "$_scratch_venv" || true
  fi

  if _mllmsci_mirror_ok "$_scratch_venv"; then
    # shellcheck disable=SC1091
    source "$_scratch_venv/bin/activate"
    export IMAGE_FORWARD_STAGE_PYTHON="$_scratch_venv/bin/python"
    echo "[stage_venv] node-local venv on $(hostname -s): $_scratch_venv ($(cat "$_scratch_venv/.sync_complete"))"
  else
    echo "[stage_venv] no matching scratch mirror on $(hostname -s) (driver venv: ${IMAGE_FORWARD_DRIVER_VENV:-unset}); the stage runs from NFS"
  fi
  unset _scratch_venv
  unset -f _mllmsci_mirror_ok _mllmsci_autosync
fi

# ── 1b. the CUDA toolkit that FlashInfer compiles with ────────────────
# FlashInfer builds its kernels by JIT, and it takes the CUDA version from the
# nvcc it finds: CUDA_HOME, then `which nvcc`, then /usr/local/cuda. The system
# toolkit on these nodes is 12.8, and 12.8 CANNOT build `compute_120f` — the
# family target that FlashInfer picks for Blackwell (sm_120). The job then dies
# in start-up with a message that names the wrong cause:
#
#   RuntimeError: FlashInfer requires GPUs with sm75 or higher
#
# What really happens: `_normalize_cuda_arch` raises "SM 12.x requires
# CUDA >= 12.9", `__init__` swallows it, the target-arch set ends up EMPTY, and
# the later check reports the sm75 message. Measured 2026-08-20 on
# jjs533-compute-03 (RTX PRO 6000 Blackwell, sm_120).
#
# `scripts/install_cuda_129_toolkit.sh` assembles a 12.9 toolkit from the
# NVIDIA redistributables. It also MATCHES torch, which is built against 12.9;
# the 12.8 system toolkit never did. The PyPI wheel `nvidia-cuda-nvcc-cu12`
# does NOT help — it ships ptxas with no nvcc.
#
# Every node reads the same path, thus no node uses a different kernel path.
# Warning: the launchers `source ~/.bashrc` BEFORE this script, and that sets
# CUDA_HOME=/usr/local/cuda-12.8. Thus a "set it only when it is empty" test
# NEVER fires on a compute node, and the Blackwell bug stays. Our toolkit wins
# whenever it is present. Override with IMAGE_FORWARD_CUDA_HOME.
_mllmsci_cuda="${IMAGE_FORWARD_CUDA_HOME:-/share/ju/matt/cuda-12.9}"
if [ -x "$_mllmsci_cuda/bin/nvcc" ]; then
  if [ "${CUDA_HOME:-}" != "$_mllmsci_cuda" ]; then
    echo "[stage_venv] CUDA_HOME ${CUDA_HOME:-unset} -> $_mllmsci_cuda (nvcc $("$_mllmsci_cuda/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p' | tail -1))"
  fi
  export CUDA_HOME="$_mllmsci_cuda"
  export CUDA_PATH="$_mllmsci_cuda"
else
  echo "[stage_venv] WARNING no 12.9 toolkit at $_mllmsci_cuda; FlashInfer will fail on any sm_120 node"
fi
unset _mllmsci_cuda

# ── 2. the FFmpeg shared libraries for torchcodec ─────────────────────
# vLLM 0.25 imports `vllm.multimodal.video`, which imports torchcodec, which
# opens libtorchcodec_coreN.so with dlopen. That library NEEDS the FFmpeg
# libav*.so.N. vLLM 0.19 never did this, so the problem is new.
#
# On these nodes, FFmpeg 4 to 7 are absent. The libavfilter of the system
# FFmpeg 8 needs GLIBCXX_3.4.32, which the anaconda-base libstdc++ that the
# venv python resolves does NOT give. Without the line below, `import vllm`
# stops with "Could not load libtorchcodec".
#
# The fix is a self-contained FFmpeg 7.1 LGPL shared build, outside the venv.
# Its libav*.so have NO libstdc++ dependency (readelf shows no NEEDED
# libstdc++), so the GLIBCXX wall is gone.
#
# Warning: ld.so reads LD_LIBRARY_PATH one time, when the process starts. This
# line must thus run in the shell setup block BEFORE the stage python starts.
# Prepend ONLY the ffmpeg directory. A general library directory could hide the
# CUDA or torch libraries. The guard keeps the entry unique across repeat
# sources.
_ff_libdir="${IMAGE_FORWARD_FFMPEG_LIBDIR:-/share/ju/matt/zoo/ffmpeg-libs/n7.1/lib}"
if [ -d "$_ff_libdir" ] && [ -e "$_ff_libdir/libavutil.so.59" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$_ff_libdir:"*) : ;;                       # already there — do nothing
    *) export LD_LIBRARY_PATH="$_ff_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
else
  echo "[stage_venv] WARNING: no FFmpeg libs at $_ff_libdir — torchcodec, and thus 'import vllm' on 0.25, will fail" >&2
fi
unset _ff_libdir
