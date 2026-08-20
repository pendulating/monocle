# Sourced from the SLURM launcher setup blocks (not executed). This script
# does two jobs before python starts:
#
#   1. It selects the fastest venv that is available on the node.
#   2. It makes the FFmpeg shared libraries visible to torchcodec.
#
# Set MLLMSCI_SKIP_SCRATCH_VENV=1 before you source the script to skip job 1
# and keep job 2. The monitor launcher does this, because the monitor runs the
# driver: if it activated the mirror, sys.prefix would become a /scratch path,
# and no stage node could match the marker.

# ── 1. the node-local venv mirror ─────────────────────────────────────
# If the node holds a COMPLETE /scratch mirror of the venv that the driver runs
# from (see scripts/sync_venv_to_scratch.sh), the script activates the mirror
# and exports MLLMSCI_STAGE_PYTHON. The submitit command line then starts the
# stage from node-local disk (see
# dagspaces/common/orchestrator.py::_create_submitit_executor). A cold
# torch/vllm/flashinfer import over NFS costs about 13 minutes for each process
# spawn; from local disk it is seconds.
#
# The fallbacks are deliberately careful. On any doubt — no mirror, a partial
# sync, a mirror of a DIFFERENT venv, or MLLMSCI_DRIVER_VENV not set — the
# script leaves MLLMSCI_STAGE_PYTHON unset, and the stage runs exactly as
# before, from the driver interpreter over NFS.

if [ "${MLLMSCI_SKIP_SCRATCH_VENV:-0}" = "1" ]; then
  echo "[stage_venv] scratch mirror skipped on request (driver job)"
else
  _scratch_venv="${MLLMSCI_SCRATCH_VENV:-}"
  if [ -z "$_scratch_venv" ] && [ -n "${MLLMSCI_DRIVER_VENV:-}" ]; then
    # The name convention of sync_venv_to_scratch.sh: the leading dot is gone.
    _scratch_venv="/scratch/$USER/venvs/$(basename "$MLLMSCI_DRIVER_VENV" | sed 's/^\.//')"
  fi

  if [ -n "$_scratch_venv" ] \
     && [ -f "$_scratch_venv/.sync_complete" ] \
     && [ -x "$_scratch_venv/bin/python" ] \
     && [ -n "${MLLMSCI_DRIVER_VENV:-}" ] \
     && grep -q "src=$MLLMSCI_DRIVER_VENV\$" "$_scratch_venv/.sync_complete"; then
    # shellcheck disable=SC1091
    source "$_scratch_venv/bin/activate"
    export MLLMSCI_STAGE_PYTHON="$_scratch_venv/bin/python"
    echo "[stage_venv] node-local venv on $(hostname -s): $_scratch_venv ($(cat "$_scratch_venv/.sync_complete"))"
  else
    echo "[stage_venv] no matching scratch mirror on $(hostname -s) (driver venv: ${MLLMSCI_DRIVER_VENV:-unset}); the stage runs from NFS"
  fi
  unset _scratch_venv
fi

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
_ff_libdir="${MLLMSCI_FFMPEG_LIBDIR:-/share/pierson/matt/zoo/ffmpeg-libs/n7.1/lib}"
if [ -d "$_ff_libdir" ] && [ -e "$_ff_libdir/libavutil.so.59" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$_ff_libdir:"*) : ;;                       # already there — do nothing
    *) export LD_LIBRARY_PATH="$_ff_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
else
  echo "[stage_venv] WARNING: no FFmpeg libs at $_ff_libdir — torchcodec, and thus 'import vllm' on 0.25, will fail" >&2
fi
unset _ff_libdir
