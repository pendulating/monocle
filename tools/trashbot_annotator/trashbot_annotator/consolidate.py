"""Merge per-video JSONL outputs into one long-format parquet table.

Each analysis window yields one row per interaction and one row per
norm dimension. Labels that fail the taxonomy check keep their raw
value in category_raw or dimension_raw and count as warnings.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .manifest import derive_fields, parse_control_mode_from_path
from .taxonomy import (
    ANNOTATOR_NAME,
    PROXEMIC_EVENTS,
    TALK_CATEGORIES,
    is_valid_address,
    is_valid_category,
    is_valid_dimension,
    is_valid_event,
    is_valid_zone,
    zone_for_distance,
)

# Column order of the consolidated parquet.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "dataset",
    "place",
    "capture_type",
    "control_mode",
    "sample_id",
    "video_path",
    "window_idx",
    "start_s",
    "end_s",
    "category",
    "category_raw",
    "address",
    "address_raw",
    "dimension",
    "dimension_raw",
    "relevance",
    "event",
    "event_raw",
    "distance_m",
    "zone",
    "zone_raw",
    "who",
    "evidence",
    "summary",
    "model",
    "annotator",
    "created_at",
)

MIN_RELEVANCE: int = 0
MAX_RELEVANCE: int = 3


def iter_windows(jsonl_path: Path) -> Iterator[dict[str, Any]]:
    """Read one JSONL file. Yield one record per valid line."""
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                yield {"__parse_error__": line}
                continue
            if not isinstance(record, dict):
                continue
            yield record


def _clean_relevance(value: Any, warnings: Counter[str]) -> int | None:
    """Return the relevance score when it is an int in range 0..3."""
    if isinstance(value, bool) or not isinstance(value, int):
        warnings["invalid_relevance"] += 1
        return None
    if value < MIN_RELEVANCE or value > MAX_RELEVANCE:
        warnings["invalid_relevance"] += 1
        return None
    return value


def _clean_distance(value: Any, warnings: Counter[str]) -> float | None:
    """Return the distance in metres when it is a non-negative number.

    Null is a valid answer from the model and counts as no warning.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        warnings["invalid_distance"] += 1
        return None
    distance = float(value)
    if distance != distance or distance < 0.0:
        warnings["invalid_distance"] += 1
        return None
    return distance


def flatten_window(
    record: dict[str, Any], meta: dict[str, Any], created_at: str, warnings: Counter[str]
) -> list[dict[str, Any]]:
    """Flatten one window record into long-format rows.

    Interactions fill the category and address columns. Norm dimensions
    fill the dimension columns. Proxemic events fill the event and
    distance columns. The other label columns stay null.
    """
    annotation = record.get("annotation")
    if not isinstance(annotation, dict):
        warnings["windows_without_annotation"] += 1
        return []

    base = {
        "dataset": meta.get("dataset"),
        "place": meta.get("place"),
        "capture_type": meta.get("capture_type"),
        "control_mode": meta.get("control_mode"),
        "sample_id": record.get("sample_id") or meta.get("sample_id"),
        "video_path": record.get("video_path") or meta.get("video_path"),
        "window_idx": record.get("window_idx"),
        "start_s": record.get("t_start_s"),
        "end_s": record.get("t_end_s"),
        "category": None,
        "category_raw": None,
        "address": None,
        "address_raw": None,
        "dimension": None,
        "dimension_raw": None,
        "relevance": None,
        "event": None,
        "event_raw": None,
        "distance_m": None,
        "zone": None,
        "zone_raw": None,
        "who": None,
        "evidence": None,
        "summary": annotation.get("summary"),
        "model": record.get("model"),
        "annotator": ANNOTATOR_NAME,
        "created_at": created_at,
    }

    rows: list[dict[str, Any]] = []
    interactions = annotation.get("interactions") or []
    if not isinstance(interactions, list):
        warnings["malformed_annotation_field"] += 1
        interactions = []
    for item in interactions:
        if not isinstance(item, dict):
            warnings["malformed_annotation_field"] += 1
            continue
        key = item.get("category")
        row = {**base, "who": item.get("who"), "evidence": item.get("evidence")}
        if isinstance(key, str) and is_valid_category(key):
            row["category"] = TALK_CATEGORIES[key]
        else:
            row["category_raw"] = key
            warnings["invalid_category"] += 1
        address = item.get("address")
        if address is None:
            # Records from the old prompt contract carry no address.
            pass
        elif is_valid_address(address):
            row["address"] = address
        else:
            row["address_raw"] = address
            warnings["invalid_address"] += 1
        rows.append(row)

    events = annotation.get("proxemic_events") or []
    if not isinstance(events, list):
        warnings["malformed_annotation_field"] += 1
        events = []
    for item in events:
        if not isinstance(item, dict):
            warnings["malformed_annotation_field"] += 1
            continue
        key = item.get("event")
        distance = _clean_distance(item.get("distance_m"), warnings)
        row = {
            **base,
            "who": item.get("who"),
            "evidence": item.get("evidence"),
            "distance_m": distance,
        }
        if is_valid_event(key):
            row["event"] = PROXEMIC_EVENTS[str(key)]
        else:
            row["event_raw"] = key
            warnings["invalid_event"] += 1
        zone = item.get("zone")
        if is_valid_zone(zone):
            row["zone"] = zone
        else:
            if zone is not None:
                row["zone_raw"] = zone
                warnings["invalid_zone"] += 1
            row["zone"] = zone_for_distance(distance)
        rows.append(row)

    dims = annotation.get("norm_dimensions") or []
    if not isinstance(dims, list):
        warnings["malformed_annotation_field"] += 1
        dims = []
    for item in dims:
        if not isinstance(item, dict):
            warnings["malformed_annotation_field"] += 1
            continue
        name = item.get("dimension")
        row = {
            **base,
            "evidence": item.get("evidence"),
            "relevance": _clean_relevance(item.get("relevance"), warnings),
        }
        if is_valid_dimension(name):
            row["dimension"] = name
        else:
            row["dimension_raw"] = name
            warnings["invalid_dimension"] += 1
        rows.append(row)

    return rows


def _meta_lookup(manifest: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    """Build a sample-id to metadata map from a manifest frame."""
    if manifest is None or manifest.empty:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for row in manifest.to_dict("records"):
        lookup[str(row["sample_id"])] = {
            "dataset": row.get("dataset"),
            "place": row.get("place"),
            "capture_type": row.get("capture_type"),
            "control_mode": row.get("control_mode"),
            "sample_id": row.get("sample_id"),
            "video_path": row.get("video_path"),
        }
    return lookup


def consolidate_dir(
    out_dir: str | Path,
    out_path: str | Path,
    manifest: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, Counter[str]]:
    """Read all JSONL files in out_dir. Write one parquet table.

    Metadata comes from the manifest when given. Otherwise the runner
    derives dataset, place, and capture type from the video path.
    Returns the frame and a counter of warning counts.
    """
    directory = Path(out_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"out-dir does not exist: {directory}")

    lookup = _meta_lookup(manifest)
    created_at = datetime.now(timezone.utc).isoformat()
    warnings: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    jsonl_files = sorted(directory.glob("*.jsonl"))
    for jsonl_file in jsonl_files:
        for record in iter_windows(jsonl_file):
            if "__parse_error__" in record:
                warnings["unparsable_lines"] += 1
                continue
            sample_id = str(record.get("sample_id") or jsonl_file.stem)
            meta = lookup.get(
                sample_id,
                {
                    "dataset": None,
                    "place": None,
                    "capture_type": None,
                    "control_mode": None,
                    "sample_id": sample_id,
                    "video_path": record.get("video_path"),
                },
            )
            video_path = record.get("video_path") or meta.get("video_path")
            if meta.get("dataset") is None and video_path:
                dataset, place, capture_type = derive_fields(str(video_path))
                meta = {
                    "dataset": dataset,
                    "place": place,
                    "capture_type": capture_type,
                    "control_mode": parse_control_mode_from_path(str(video_path)),
                    "sample_id": sample_id,
                    "video_path": video_path,
                }
            rows.extend(flatten_window(record, meta, created_at, warnings))

    df = pd.DataFrame(rows, columns=pd.Index(OUTPUT_COLUMNS))
    df["relevance"] = df["relevance"].astype("Int64")
    df["distance_m"] = df["distance_m"].astype("float64")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df, warnings
