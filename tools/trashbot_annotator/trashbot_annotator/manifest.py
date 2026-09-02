"""Build a manifest of deployment videos under a data root.

One output row describes one video file. The row carries the sample id,
the absolute video path, and fields derived from the path segments.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePath

import pandas as pd

# Video extensions we collect. Case matters on some filesystems,
# so we list the common spellings.
#
# We drop the Insta360 .insv files. They hold a dual-fisheye stream which
# OpenCV reads as one flat frame, so the model sees two distorted circles.
# They also crashed the binary in earlier runs. Stitch them to .mp4 first
# if you need them.
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".MP4", ".mov", ".MOV")

# Extensions we know about and skip on purpose.
EXCLUDED_EXTENSIONS: tuple[str, ...] = (".insv", ".INSV")

# Column order of the manifest parquet.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "video_path",
    "dataset",
    "place",
    "capture_type",
    "control_mode",
)

# A date-park segment looks like "2025-07-01-tappen-park".
# Single-digit months appear in older folders ("2022-9-21-astor-place").
_DATE_PARK_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}-(.+)$")

# A pilot segment looks like "albee_square_pilot". The stem before the
# suffix names the place.
_PILOT_RE = re.compile(r"^(.+?)[-_]pilot$", re.IGNORECASE)

# Robot control mode tokens from the summer 2026 tree.
CONTROL_MODES: tuple[str, ...] = ("auton_t2t", "auton_waypoint", "joy_t2t")

# Directory names we skip during the walk. Consent folders hold consent
# statements, not deployment footage. The backup folder holds images.
DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset({"consent", "trashbot_backup_img"})

# Path segment -> capture type token. Match runs on the lowercased
# segment with "-" replaced by "_".
_CAPTURE_TOKENS: dict[str, str] = {
    "interview": "interview",
    "interviews": "interview",
    "onboard_camera": "onboard_camera",
    "overview_camera": "overview_camera",
    "landfill": "landfill",
    "recycle": "recycle",
    "tripod_cam": "tripod_cam",
    "trashbot_cam": "trashbot_cam",
}

# Fallback capture type.
CAPTURE_TYPE_OTHER: str = "other"


def normalize_place(place: str | None) -> str | None:
    """Normalize a place slug to hyphen form and lower case.

    Folder names spell the same park in two ways ("albee-square" and
    "albee_square"). The transcript corpus uses the hyphen form, so
    this function makes joins by place work.
    """
    if place is None:
        return None
    return place.strip().lower().replace("_", "-")


def parse_place(rel_parts: list[str]) -> str | None:
    """Parse the park slug from the relative path segments.

    A date-park segment like "2025-07-01-tappen-park" yields
    "tappen-park". A pilot segment like "albee_square_pilot" yields
    "albee-square". A segment like "oct_1" or "20250915" yields None.
    """
    for part in rel_parts:
        m = _DATE_PARK_RE.match(part)
        if m:
            return normalize_place(m.group(1))
    for part in rel_parts:
        m = _PILOT_RE.match(part)
        if m:
            return normalize_place(m.group(1))
    return None


def parse_control_mode(rel_parts: list[str]) -> str | None:
    """Parse the robot control mode from the relative path segments.

    A control mode token matches inside a segment on "_" boundaries.
    Examples: "auton_waypoint" and "20260623_auton_waypoint_data" both
    yield "auton_waypoint". No match yields None.
    """
    for part in rel_parts:
        norm = f"_{part.lower().replace('-', '_')}_"
        for mode in CONTROL_MODES:
            if f"_{mode}_" in norm:
                return mode
    return None


def parse_control_mode_from_path(video_path: str | os.PathLike[str]) -> str | None:
    """Parse the control mode from a full video path."""
    return parse_control_mode(list(PurePath(video_path).parts))


def parse_capture_type(rel_parts: list[str]) -> str:
    """Map the relative path segments to one capture type token.

    A segment matches when it equals a token, or when the token is a
    clear prefix or suffix of the segment. Examples:
    "Interviews" -> interview, "20260625_trashbot_cam" -> trashbot_cam,
    "interview_data" -> interview. No match yields "other".
    """
    for part in rel_parts:
        norm = part.lower().replace("-", "_")
        for token, mapped in _CAPTURE_TOKENS.items():
            if (
                norm == token
                or norm.startswith(f"{token}_")
                or norm.endswith(f"_{token}")
            ):
                return mapped
    return CAPTURE_TYPE_OTHER


def derive_fields(video_path: str | os.PathLike[str]) -> tuple[str | None, str | None, str]:
    """Derive dataset, place, and capture type from a video path.

    The dataset is the folder under an ancestor named "data". When no
    such ancestor exists, the dataset is None. Place and capture type
    come from the segments after that ancestor.
    """
    parts = PurePath(video_path).parts
    rel_parts: list[str]
    if "data" in parts:
        idx = parts.index("data")
        rel_parts = list(parts[idx + 1 :])
        dataset = rel_parts[0] if rel_parts else None
    else:
        dataset = None
        rel_parts = list(parts)
    place = parse_place(rel_parts)
    capture_type = parse_capture_type(rel_parts)
    return dataset, place, capture_type


def make_sample_id(data_root: str | os.PathLike[str], video_path: str | os.PathLike[str]) -> str:
    """Build the sample id from the path below the data root.

    The id joins the relative segments with "_" and drops the suffix.
    Example: "25_little-italy/oct_1/Recycle/clip.mp4" yields
    "25_little-italy_oct_1_Recycle_clip".
    """
    root = Path(data_root).resolve()
    full = Path(video_path).resolve()
    rel = full.relative_to(root)
    return str(rel.with_suffix("")).replace(os.sep, "_")


def build_manifest(
    data_root: str | os.PathLike[str],
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
) -> pd.DataFrame:
    """Walk the data root. Return one manifest row per video file.

    Directories whose lowercased name sits in exclude_dirs drop out of
    the walk, with all their content.
    """
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"data root does not exist: {root}")

    rows: list[dict[str, str | None]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in exclude_dirs]
        for name in filenames:
            if not name.endswith(VIDEO_EXTENSIONS):
                continue
            video_path = Path(dirpath) / name
            rel_parts = video_path.resolve().relative_to(root).parts
            rel_no_suffix = Path(*rel_parts).with_suffix("")
            rows.append(
                {
                    "sample_id": str(rel_no_suffix).replace(os.sep, "_"),
                    "video_path": str(video_path),
                    "dataset": rel_parts[0],
                    "place": parse_place(list(rel_parts)),
                    "capture_type": parse_capture_type(list(rel_parts)),
                    "control_mode": parse_control_mode(list(rel_parts)),
                }
            )

    df = pd.DataFrame(rows, columns=pd.Index(MANIFEST_COLUMNS))
    return df.sort_values("video_path", kind="stable").reset_index(drop=True)


def build_manifest_multi(
    data_roots: list[str | os.PathLike[str]],
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
) -> pd.DataFrame:
    """Build one manifest across several data roots.

    Sample ids stay relative to their own root. A duplicate sample id
    across roots raises ValueError, because downstream joins key on the
    sample id.
    """
    frames = [build_manifest(root, exclude_dirs=exclude_dirs) for root in data_roots]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=pd.Index(MANIFEST_COLUMNS)
    )
    duplicated = df["sample_id"].duplicated(keep=False)
    if bool(duplicated.any()):
        examples = sorted(df.loc[duplicated, "sample_id"].unique())[:5]
        raise ValueError(
            f"duplicate sample ids across data roots: {examples}"
        )
    return df.sort_values("video_path", kind="stable").reset_index(drop=True)


def write_manifest(
    data_roots: str | os.PathLike[str] | list[str | os.PathLike[str]],
    out_path: str | os.PathLike[str],
    exclude_dirs: frozenset[str] = DEFAULT_EXCLUDE_DIRS,
) -> pd.DataFrame:
    """Build the manifest and write it as a parquet file.

    One root or a list of roots works. A list produces one combined
    manifest.
    """
    if isinstance(data_roots, (str, os.PathLike)):
        roots: list[str | os.PathLike[str]] = [data_roots]
    else:
        roots = list(data_roots)
    df = build_manifest_multi(roots, exclude_dirs=exclude_dirs)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df
