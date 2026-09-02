"""Find repeat interactions with embeddings of event keyframes.

The workflow has three steps:

1. ``extract_event_frames`` cuts one keyframe per annotated window that
   holds a person event. It writes a frames manifest parquet.
2. ``scripts/embed_events.sub`` embeds the keyframes on a GPU node with
   Qwen3-VL-Embedding-8B. It writes an embeddings parquet.
3. ``find_repeats`` scores the cosine similarity between events from
   different clips. High pairs are repeat-visit candidates.

The keyframe embedding sees the person and the scene together. Two
events at the same spot share the background, so cross-day pairs at
one place are the trustworthy signal. Review candidates in the viewer
before you count them.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .runner import child_env

# The zoo ffmpeg build. The viewer uses the same order:
# TRASHBOT_FFMPEG override, zoo build, system build.
ZOO_FFMPEG = Path("/share/pierson/matt/zoo/ffmpeg-libs/n7.1/bin/ffmpeg")

# Column order of the frames manifest parquet.
FRAMES_COLUMNS: tuple[str, ...] = (
    "event_id",
    "sample_id",
    "window_idx",
    "t_frame_s",
    "image_path",
    "dataset",
    "place",
    "capture_type",
    "date",
    "video_path",
)

# Column order of the repeat pairs parquet.
PAIRS_COLUMNS: tuple[str, ...] = (
    "event_id_a",
    "event_id_b",
    "similarity",
    "sample_id_a",
    "sample_id_b",
    "place_a",
    "place_b",
    "date_a",
    "date_b",
    "same_place",
    "same_date",
)

# Dates inside sample ids: "2025-07-01-tappen-park_..." or "20260625_...".
_DATE_DASH_RE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_DATE_COMPACT_RE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")


def resolve_ffmpeg() -> Path | None:
    """Return the first usable ffmpeg binary."""
    candidates: list[Path] = []
    env_bin = os.environ.get("TRASHBOT_FFMPEG")
    if env_bin:
        candidates.append(Path(env_bin))
    candidates.append(ZOO_FFMPEG)
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def ffmpeg_env(binary: Path | None) -> dict[str, str]:
    """Subprocess environment with the shared libraries of the binary."""
    env = child_env()
    if binary is None:
        return env
    lib_dir = binary.parent.parent / "lib"
    current = env.get("LD_LIBRARY_PATH", "")
    if lib_dir.is_dir() and str(lib_dir) not in current.split(os.pathsep):
        env["LD_LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{current}" if current else str(lib_dir)
    return env


def parse_date_from_sample_id(sample_id: str) -> str | None:
    """Parse an ISO date from a sample id. None when no date appears."""
    m = _DATE_DASH_RE.search(sample_id)
    if m:
        year, month, day = m.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    m = _DATE_COMPACT_RE.search(sample_id)
    if m:
        year, month, day = m.groups()
        return f"{year}-{month}-{day}"
    return None


def select_event_windows(annotations: pd.DataFrame) -> pd.DataFrame:
    """Return one row per window that holds a person event.

    A window qualifies when it carries an interaction or a proxemic
    event. The row keeps the window midpoint as the keyframe time.
    """
    mask = annotations["category"].notna() | annotations["event"].notna()
    df = annotations[mask].copy()
    df = df.drop_duplicates(subset=["sample_id", "window_idx"])
    df["t_frame_s"] = (
        pd.to_numeric(df["start_s"], errors="coerce")
        + pd.to_numeric(df["end_s"], errors="coerce")
    ) / 2.0
    return df


def extract_event_frames(
    annotations: pd.DataFrame,
    out_dir: str | Path,
    limit: int | None = None,
) -> pd.DataFrame:
    """Cut one JPEG keyframe per event window. Return the frames manifest.

    Frames land under out_dir/frames. The manifest parquet is the input
    of the GPU embedding script.
    """
    binary = resolve_ffmpeg()
    if binary is None:
        raise FileNotFoundError("no usable ffmpeg binary; set TRASHBOT_FFMPEG")
    env = ffmpeg_env(binary)

    # Absolute paths only. The embedding job runs with another working
    # directory, so a relative image path breaks there.
    out_root = Path(out_dir).resolve()
    frames_dir = out_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    windows = select_event_windows(annotations)
    if limit is not None:
        windows = windows.head(max(limit, 0))

    rows: list[dict[str, object]] = []
    for row in windows.to_dict("records"):
        sample_id = str(row["sample_id"])
        window_idx = int(row["window_idx"])
        t_frame = float(row["t_frame_s"])
        video_path = str(row.get("video_path") or "")
        event_id = f"{sample_id}__w{window_idx}"
        image_path = frames_dir / f"{event_id}.jpg"
        if not video_path or not Path(video_path).is_file():
            continue
        if not image_path.is_file():
            cmd = [
                str(binary),
                "-y",
                "-ss", f"{t_frame:.2f}",
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                str(image_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, env=env, check=False)
            if proc.returncode != 0 or not image_path.is_file():
                continue
        rows.append(
            {
                "event_id": event_id,
                "sample_id": sample_id,
                "window_idx": window_idx,
                "t_frame_s": t_frame,
                "image_path": str(image_path),
                "dataset": row.get("dataset"),
                "place": row.get("place"),
                "capture_type": row.get("capture_type"),
                "date": parse_date_from_sample_id(sample_id),
                "video_path": video_path,
            }
        )

    manifest = pd.DataFrame(rows, columns=pd.Index(FRAMES_COLUMNS))
    out = out_root / "frames_manifest.parquet"
    manifest.to_parquet(out, index=False)
    return manifest


def _embedding_matrix(embeddings: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Stack the embedding lists into one L2-normalized matrix."""
    valid = embeddings[embeddings["embedding"].notna()]
    ids = valid["event_id"].astype(str).tolist()
    matrix = np.stack([np.asarray(e, dtype=np.float32) for e in valid["embedding"]])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms, ids


def find_repeats(
    embeddings: pd.DataFrame,
    frames_manifest: pd.DataFrame,
    min_sim: float = 0.60,
    cross_sample_only: bool = True,
) -> pd.DataFrame:
    """Score event pairs by cosine similarity. Return pairs over min_sim.

    With cross_sample_only, pairs inside one clip drop out. The result
    sorts by similarity, highest first.
    """
    matrix, ids = _embedding_matrix(embeddings)
    meta = frames_manifest.set_index("event_id")
    sims = matrix @ matrix.T

    rows: list[dict[str, object]] = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            similarity = float(sims[i, j])
            if similarity < min_sim:
                continue
            id_a, id_b = ids[i], ids[j]
            if id_a not in meta.index or id_b not in meta.index:
                continue
            a = meta.loc[id_a]
            b = meta.loc[id_b]
            if cross_sample_only and str(a["sample_id"]) == str(b["sample_id"]):
                continue
            rows.append(
                {
                    "event_id_a": id_a,
                    "event_id_b": id_b,
                    "similarity": similarity,
                    "sample_id_a": a["sample_id"],
                    "sample_id_b": b["sample_id"],
                    "place_a": a["place"],
                    "place_b": b["place"],
                    "date_a": a["date"],
                    "date_b": b["date"],
                    "same_place": bool(
                        pd.notna(a["place"]) and a["place"] == b["place"]
                    ),
                    "same_date": bool(
                        pd.notna(a["date"]) and a["date"] == b["date"]
                    ),
                }
            )

    pairs = pd.DataFrame(rows, columns=pd.Index(PAIRS_COLUMNS))
    return pairs.sort_values("similarity", ascending=False).reset_index(drop=True)


def cluster_repeats(pairs: pd.DataFrame) -> pd.DataFrame:
    """Group the pairs into clusters with union-find.

    Each cluster is one repeat-visit candidate: a set of events that
    chain together over the similarity threshold.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for row in pairs.to_dict("records"):
        union(str(row["event_id_a"]), str(row["event_id_b"]))

    clusters: dict[str, list[str]] = {}
    for node in parent:
        clusters.setdefault(find(node), []).append(node)

    rows = [
        {
            "cluster_id": i,
            "n_events": len(members),
            "event_ids": sorted(members),
        }
        for i, (_root, members) in enumerate(
            sorted(clusters.items(), key=lambda kv: -len(kv[1]))
        )
    ]
    return pd.DataFrame(rows, columns=pd.Index(("cluster_id", "n_events", "event_ids")))
