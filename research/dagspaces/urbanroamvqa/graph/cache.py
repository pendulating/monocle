"""Config-fingerprinted pickle cache for prebuilt street graphs.

A cached graph is only reused when it was built from the same graph config
and metadata parquet; otherwise it is rebuilt. Legacy caches (bare pickled
StreetGraph objects with no fingerprint) are treated as stale.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Any, Optional


def cfg_fingerprint(graph_cfg: Any, metadata_parquet: str) -> str:
    """Stable hash of the graph config (minus cache path) + input parquet path."""
    cfg_dict: Any
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(graph_cfg):
            cfg_dict = OmegaConf.to_container(graph_cfg, resolve=True)
        elif isinstance(graph_cfg, dict):
            cfg_dict = dict(graph_cfg)
        else:
            cfg_dict = dict(vars(graph_cfg))
    except Exception:
        cfg_dict = {}
    if isinstance(cfg_dict, dict):
        cfg_dict.pop("precomputed_path", None)
    payload = json.dumps(
        {"cfg": cfg_dict, "metadata_parquet": metadata_parquet},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cached_graph(path: Optional[str], fingerprint: str, tag: str) -> Optional[Any]:
    """Return the cached graph if present and fingerprint-valid, else None."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"{tag} Failed to read cache {path}: {exc} — rebuilding", flush=True)
        return None
    if not isinstance(payload, dict) or "graph" not in payload:
        print(f"{tag} Cache {path} predates fingerprint validation — rebuilding", flush=True)
        return None
    if payload.get("fingerprint") != fingerprint:
        print(f"{tag} Cache {path} was built with a different config — rebuilding", flush=True)
        return None
    print(f"{tag} Loaded cached graph from {path}", flush=True)
    return payload["graph"]


def save_cached_graph(path: Optional[str], graph: Any, fingerprint: str, tag: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"fingerprint": fingerprint, "graph": graph}, f)
    print(f"{tag} Saved graph cache to {path}", flush=True)
