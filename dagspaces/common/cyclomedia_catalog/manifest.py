"""Parse per-recording manifest.json files in parallel.

Each recording directory contains a manifest.json describing the Cyclomedia
imageId, the render coordinates (`label = "lon,lat"`), tile/zoom settings, and
per-face render + depthmap provenance.

This module reads them with a thread pool (NFS-bound work — threads help
because the stat/read syscalls release the GIL) and returns a Polars
DataFrame keyed by (dataset, group, recording_dir).
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Optional

import polars as pl

from .schema import ALL_FACES

__all__ = ["parse_manifests"]

log = logging.getLogger(__name__)


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_label(label: Any) -> tuple[Optional[float], Optional[float]]:
    """Label is `"lon,lat"`. Return (lat, lon)."""
    if not isinstance(label, str) or "," not in label:
        return None, None
    parts = label.split(",")
    if len(parts) < 2:
        return None, None
    lon = _safe_float(parts[0])
    lat = _safe_float(parts[1])
    return lat, lon


def _parse_one(task: tuple[str, str, str, str]) -> dict[str, Any]:
    """task = (dataset, group, recording_dir, recording_path). Returns one row dict."""
    dataset, group, recording_dir, rec_path = task
    row: dict[str, Any] = {
        "dataset": dataset,
        "group": group,
        "recording_dir": recording_dir,
        "manifest_ok": False,
        "manifest_image_id": None,
        "manifest_latitude": None,
        "manifest_longitude": None,
        "manifest_zoom": None,
        "manifest_tile_px": None,
        "manifest_tile_schema": None,
        "manifest_name_version": None,
        "manifest_mode": None,
        "manifest_checkpoint": None,
        "manifest_no_tiles": None,
        "depthmap_stitched": None,
    }
    # Per-face scalar columns: {face_elapsed_s_F, face_used_render_F, ...}
    # and per-face depthmap columns. We flatten here; the indexer later
    # explodes (recording × 6 faces) → one row per (recording, face).
    for face in ALL_FACES:
        row[f"face_elapsed_s_{face}"] = None
        row[f"face_used_render_{face}"] = None
        row[f"depthmap_present_{face}"] = None
        row[f"depthmap_used_render_{face}"] = None
        row[f"depthmap_render_size_{face}"] = None
        row[f"depthmap_rgb_render_size_{face}"] = None
        row[f"depthmap_downsample_factor_{face}"] = None

    mpath = os.path.join(rec_path, "manifest.json")
    try:
        with open(mpath, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return row

    row["manifest_ok"] = True
    row["manifest_image_id"] = data.get("imageId")
    lat, lon = _parse_label(data.get("label"))
    row["manifest_latitude"] = lat
    row["manifest_longitude"] = lon
    row["manifest_zoom"] = data.get("zoom")
    row["manifest_tile_px"] = data.get("tilePx")
    row["manifest_tile_schema"] = data.get("tileSchema")
    row["manifest_name_version"] = data.get("nameVersion")
    row["manifest_mode"] = data.get("mode")
    row["manifest_checkpoint"] = data.get("checkpoint")
    row["manifest_no_tiles"] = data.get("no_tiles")

    faces = data.get("faces") or {}
    for face in ALL_FACES:
        entry = faces.get(face) or {}
        row[f"face_elapsed_s_{face}"] = _safe_float(entry.get("elapsed_s"))
        row[f"face_used_render_{face}"] = entry.get("used_render")

    depthmaps = data.get("depthmaps") or {}
    row["depthmap_stitched"] = depthmaps.get("stitched_faces")
    dm_faces = depthmaps.get("faces") or {}
    for face in ALL_FACES:
        entry = dm_faces.get(face) or {}
        tiles_present = entry.get("tiles_present")
        used_render = entry.get("used_render")
        # "present" means either we rendered it or tiles exist
        present: Optional[bool]
        if used_render is None and tiles_present is None:
            present = None
        else:
            present = bool(used_render) or (bool(tiles_present) if tiles_present is not None else False)
        row[f"depthmap_present_{face}"] = present
        row[f"depthmap_used_render_{face}"] = used_render
        row[f"depthmap_render_size_{face}"] = entry.get("render_size")
        row[f"depthmap_rgb_render_size_{face}"] = entry.get("rgb_render_size")
        row[f"depthmap_downsample_factor_{face}"] = _safe_float(entry.get("downsample_factor"))

    return row


def parse_manifests(
    raw_root: str,
    recording_keys: Iterable[tuple[str, str, str]],
    workers: int = 32,
) -> pl.DataFrame:
    """Parse manifest.json for every (dataset, group, recording_dir) key.

    `recording_keys` yields tuples like `("plazas_sample", "5D52N", "5D52NXYZ")`.
    Returns a DataFrame with one row per recording.
    """
    tasks: list[tuple[str, str, str, str]] = []
    for dataset, group, recording_dir in recording_keys:
        rec_path = os.path.join(raw_root, dataset, group, recording_dir)
        tasks.append((dataset, group, recording_dir, rec_path))

    if not tasks:
        return pl.DataFrame()

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(_parse_one, tasks):
            rows.append(r)

    df = pl.DataFrame(rows)
    return df
