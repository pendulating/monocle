"""Shared utilities for opf_vision_labels stage runners.

Detection rows are written as parquet with a consistent schema so that
the downstream ``unify`` stage can concatenate them without per-source
adapters. Each row is one detection; images with zero detections are
represented by a single sentinel row with ``class=None`` and NaN bbox.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from omegaconf import DictConfig


def in_camera_mask(
    bbox: tuple, img_w: int, img_h: int, camera_mask_cfg
) -> bool:
    """True if the bbox centroid sits inside the fixed camera-vehicle mask.

    The Cyclomedia capture car's own body is masked at a stable bottom-center
    region. Any detection whose centroid falls inside that region is almost
    always a false positive (camera dashboard, hood reflections, the safety-
    chevron pattern), and should be dropped before reaching pseudo-labels.
    """
    if not camera_mask_cfg or not bool(getattr(camera_mask_cfg, "enabled", False)):
        return False
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bottom_thresh = img_h * (1.0 - float(camera_mask_cfg.bottom_frac))
    if cy < bottom_thresh:
        return False
    x_center_px = img_w * float(camera_mask_cfg.x_center)
    half_w_px = img_w * float(camera_mask_cfg.x_half_width)
    return abs(cx - x_center_px) <= half_w_px


DETECTION_COLUMNS = [
    "sample_id",
    "image_path",
    "recording_id",
    "face",
    "dataset",
    "class",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "score",
    "teacher_model",
    "teacher_version",
]


@dataclass
class Detection:
    sample_id: str
    image_path: str
    cls: Optional[str]
    bbox_x1: Optional[float] = None
    bbox_y1: Optional[float] = None
    bbox_x2: Optional[float] = None
    bbox_y2: Optional[float] = None
    score: Optional[float] = None
    recording_id: Optional[str] = None
    face: Optional[str] = None
    dataset: Optional[str] = None
    teacher_model: Optional[str] = None
    teacher_version: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "recording_id": self.recording_id,
            "face": self.face,
            "dataset": self.dataset,
            "class": self.cls,
            "bbox_x1": self.bbox_x1,
            "bbox_y1": self.bbox_y1,
            "bbox_x2": self.bbox_x2,
            "bbox_y2": self.bbox_y2,
            "score": self.score,
            "teacher_model": self.teacher_model,
            "teacher_version": self.teacher_version,
        }


def detections_to_frame(detections: Iterable[Detection]) -> pd.DataFrame:
    rows = [d.to_row() for d in detections]
    if not rows:
        return pd.DataFrame(columns=DETECTION_COLUMNS)
    df = pd.DataFrame(rows)
    for col in DETECTION_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[DETECTION_COLUMNS]


def load_input_frame(cfg: DictConfig, inputs: Dict[str, str],
                     input_key: str = "dataset") -> pd.DataFrame:
    """Resolve the input image manifest for a stage.

    Priority: explicit ``inputs[input_key]`` → ``cfg.data.parquet_path``.
    The manifest must contain at minimum ``image_path``; ``sample_id``,
    ``recording_id``, ``face``, and ``dataset`` are preserved when present.
    """
    path = inputs.get(input_key)
    if not path:
        path = getattr(cfg.data, "parquet_path", None)
    if not path:
        raise ValueError(
            f"Stage needs an input manifest: provide '{input_key}' input "
            f"or set cfg.data.parquet_path"
        )
    df = pd.read_parquet(path)

    # Cyclomedia manifests carry both ``image_path`` (a /scratch staging
    # path used by SLURM workers) and ``image_path_original`` (the
    # canonical /share/ju path). The /scratch path is unreachable from the
    # login node and from any worker that hasn't pre-staged. Prefer the
    # original whenever it exists.
    if "image_path_original" in df.columns:
        df = df.copy()
        orig = df["image_path_original"].astype("string")
        df["image_path"] = orig.where(orig.notna(), df["image_path"])

    debug = bool(getattr(getattr(cfg, "runtime", {}), "debug", False))
    sample_n = getattr(getattr(cfg, "runtime", {}), "sample_n", None)
    if sample_n is not None and (debug or int(sample_n) > 0):
        df = df.head(int(sample_n)).reset_index(drop=True)
    return df


def ensure_output_dir(output_paths: Dict[str, str], key: str) -> str:
    path = output_paths.get(key)
    if not path:
        raise ValueError(f"Missing required output '{key}' in output_paths")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    return path


def write_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


def collect_outputs(output_paths: Dict[str, str],
                    produced_keys: Iterable[str]) -> Dict[str, str]:
    return {k: output_paths[k] for k in produced_keys if k in output_paths}
