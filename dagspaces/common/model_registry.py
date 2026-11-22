"""Node-local model registry: send zoo model loads to a /scratch mirror.

The model zoo is on NFS (``/share/ju/matt/zoo/models``). Weight loading
is a bandwidth-bound sequential read, and the pipeline pays it at each vLLM
engine start. The canonical models are thus mirrored to node-local /scratch by
``scripts/sync_model_registry_to_scratch.sh`` (one mirror for each node — the
same pattern as the venv mirror). This module resolves a model path to the
local mirror when the current node holds a complete, matching mirror. If it
does not, the module returns the original path.

Contract:

- Resolution happens at the LOAD BOUNDARY only (the vLLM engine kwargs or
  ``from_pretrained``). The Hydra configs, the W&B records, and the run
  metadata keep the canonical /share path.
- The mirror keeps the zoo basename, so each path-substring test (the
  reasoning-parser test, the AWQ test, the gemma4-unified test) gives the
  same result on the resolved path.
- A mirror is trusted only when ``<mirror>/.sync_complete`` exists and holds a
  line ``src=<original path>`` (the ``activate_stage_venv.sh`` marker
  convention). Freshness is the responsibility of the sync script — zoo models
  do not change after the download.
- The registry root comes from ``IMAGE_FORWARD_MODEL_REGISTRY`` (set in
  ``server.env``). If it is empty or not set, this module does nothing, so a
  machine without a mirror is not affected.
- A node that holds no mirror builds one HERE, before the weight load, with
  ``scripts/sync_model_registry_to_scratch.sh``. The first job on a node pays
  the copy; every later job on that node reads from local disk. Set
  ``IMAGE_FORWARD_MODEL_AUTOSYNC=0`` to turn this off.
- Each probe failure falls back to the original path.
"""

from __future__ import annotations

import os
import shutil
import subprocess

__all__ = ["resolve_model_source"]


def _mirror_ok(mirror: str, src_norm: str) -> bool:
    """Test the marker of *mirror* against *src_norm*."""
    marker = os.path.join(mirror, ".sync_complete")
    if not os.path.isfile(marker):
        return False
    with open(marker, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh.read().splitlines()]
    return f"src={src_norm}" in lines


def _dir_bytes(path: str) -> int:
    """Give the size of *path* in bytes. 0 means the walk failed."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _autosync(src_norm: str, mirror: str, stage_name: str) -> bool:
    """Build the node-local mirror of *src_norm*. True means it is ready.

    Warning: this reads the whole model over NFS. It runs one time for each
    node, and the sync script holds a lock, thus a second job on the same node
    does not start a second copy — it falls back to NFS instead.

    Four guards keep this safe:
      1. The lock in the sync script. One job syncs; the others use NFS.
      2. Free space. The copy needs the model size and 20 percent more.
      3. A time limit. A slow NFS read does not hold the GPU for ever.
      4. Every failure falls back to NFS. The stage always runs.
    """
    if (os.environ.get("IMAGE_FORWARD_MODEL_AUTOSYNC") or "1").strip() == "0":
        return False
    root = os.environ.get("IMAGE_FORWARD_PROJECT_ROOT") or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = os.path.join(root, "scripts", "sync_model_registry_to_scratch.sh")
    if not os.path.isfile(script) or shutil.which("rsync") is None:
        return False

    need = _dir_bytes(src_norm)
    parent = os.path.dirname(mirror.rstrip("/"))
    probe = parent if os.path.isdir(parent) else "/scratch"
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return False
    if need <= 0 or free < need * 1.2:
        print(f"[{stage_name}] model registry: no room for the mirror of "
              f"{os.path.basename(src_norm)} ({free // 2**30} GB free, "
              f"{need // 2**30} GB needed); the stage reads NFS")
        return False

    print(f"[{stage_name}] model registry: the node holds no mirror of "
          f"{os.path.basename(src_norm)}; the sync starts "
          f"({need // 2**30} GB)", flush=True)
    try:
        subprocess.run(
            ["bash", script, os.path.basename(src_norm)],
            check=False, timeout=1800,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        print(f"[{stage_name}] model registry: the sync failed ({exc}); "
              f"the stage reads NFS")
        return False
    return _mirror_ok(mirror, src_norm)


def resolve_model_source(path, *, stage_name: str = "model_registry") -> str:
    """Return the node-local mirror of *path* if one is synced, else *path*.

    You can call this on any model-shaped value. HF hub ids, empty values, and
    paths outside the zoo pass through without a change.
    """
    src = str(path or "")
    try:
        root = (os.environ.get("IMAGE_FORWARD_MODEL_REGISTRY") or "").strip()
        if not root or not src.startswith("/"):
            return src
        src_norm = src.rstrip("/")
        if not os.path.isdir(src_norm):
            # Never redirect a source that we cannot see. A basename collision
            # with a stale mirror would otherwise load the wrong weights.
            return src
        mirror = os.path.join(root, os.path.basename(src_norm))
        if not _mirror_ok(mirror, src_norm):
            # A mirror of a DIFFERENT source keeps its marker. Do not touch it:
            # the sync script refuses it too, and NFS gives the right weights.
            if os.path.isfile(os.path.join(mirror, ".sync_complete")):
                return src
            if not _autosync(src_norm, mirror, stage_name):
                return src
        print(f"[{stage_name}] model registry: {src_norm} -> {mirror} "
              f"(node-local mirror)")
        return mirror
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[{stage_name}] the model registry probe failed ({exc}); "
              f"the pipeline uses {src}")
        return src
