"""Node-local model registry: send zoo model loads to a /scratch mirror.

The model zoo is on NFS (``/share/pierson/matt/zoo/models``). Weight loading
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
- The registry root comes from ``MLLMSCI_MODEL_REGISTRY`` (set in
  ``server.env``). If it is empty or not set, this module does nothing, so a
  machine without a mirror is not affected.
- Each probe failure falls back to the original path.
"""

from __future__ import annotations

import os

__all__ = ["resolve_model_source"]


def resolve_model_source(path, *, stage_name: str = "model_registry") -> str:
    """Return the node-local mirror of *path* if one is synced, else *path*.

    You can call this on any model-shaped value. HF hub ids, empty values, and
    paths outside the zoo pass through without a change.
    """
    src = str(path or "")
    try:
        root = (os.environ.get("MLLMSCI_MODEL_REGISTRY") or "").strip()
        if not root or not src.startswith("/"):
            return src
        src_norm = src.rstrip("/")
        if not os.path.isdir(src_norm):
            # Never redirect a source that we cannot see. A basename collision
            # with a stale mirror would otherwise load the wrong weights.
            return src
        mirror = os.path.join(root, os.path.basename(src_norm))
        marker = os.path.join(mirror, ".sync_complete")
        if not os.path.isfile(marker):
            return src
        with open(marker, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh.read().splitlines()]
        if f"src={src_norm}" not in lines:
            return src
        print(f"[{stage_name}] model registry: {src_norm} -> {mirror} "
              f"(node-local mirror)")
        return mirror
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[{stage_name}] the model registry probe failed ({exc}); "
              f"the pipeline uses {src}")
        return src
