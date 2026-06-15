"""Blur-region detection stage (Cyclomedia-specific).

Cyclomedia pre-blurs faces and license plates at pull time. Running a
face or plate detector directly on these images fails because the signal
is gone. Instead, this stage detects the *blur artifact itself*: a
closed region of low internal gradient energy bounded by a sharp
gradient discontinuity (the mask edge). Outputs per-image candidate
bboxes that ``classify_blur`` downstream assigns to ``face`` or
``license_plate`` via context heuristics.

Also responsible for excluding the fixed camera-vehicle mask at the
bottom-center of every frame — that region is a solid-color fill, not a
Gaussian blur, and should not become training signal.

Detection recipe (classical, configurable):

  1. Compute Sobel gradient magnitude on grayscale, smoothed in 8×8
     boxes to get an "internal gradient" map.
  2. Threshold: pixels with smoothed gradient < internal_gradient_max
     are candidate "low-texture" pixels.
  3. Morphologically close candidate pixels into solid regions and
     extract connected components.
  4. For each component, measure the boundary gradient strength on a
     thin annulus just outside the component. Reject components whose
     boundary gradient is below boundary_gradient_min (sky, smooth
     walls — natural smooth surfaces have low gradient on both sides).
  5. Shape-filter to axis-aligned bounding rectangles with aspect 1:1
     to 3:1 and side length 20-200 px.
  6. Drop boxes that fall inside the camera-vehicle mask region.
"""

from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from dagspaces.common.orchestrator import StageExecutionContext, StageResult
from dagspaces.common.runners.base import StageRunner

from ._common import (
    Detection,
    detections_to_frame,
    ensure_output_dir,
    load_input_frame,
    write_parquet,
)


def _sobel_gradient_magnitude(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    return np.sqrt(gx * gx + gy * gy)


def _is_inside_camera_mask(
    box: tuple, img_w: int, img_h: int, cm_cfg: DictConfig
) -> bool:
    if not cm_cfg or not bool(cm_cfg.get("enabled", False)):
        return False
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bottom_thresh = img_h * (1.0 - float(cm_cfg.bottom_frac))
    if cy < bottom_thresh:
        return False
    x_center_px = img_w * float(cm_cfg.x_center)
    half_w_px = img_w * float(cm_cfg.x_half_width)
    return abs(cx - x_center_px) <= half_w_px


def _detect_blur_regions(
    image_bgr: np.ndarray, model_cfg: DictConfig, cm_cfg: DictConfig
) -> List[tuple]:
    """Return list of (x1, y1, x2, y2, score) candidate blur boxes.

    ``score`` is a heuristic confidence in [0, 1] proportional to how
    much the boundary gradient exceeds the internal gradient — a more
    "mask-like" region scores higher.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gmag = _sobel_gradient_magnitude(gray, int(model_cfg.gradient_ksize))

    win = int(model_cfg.internal_window_px)
    if win < 1:
        win = 1
    smoothed = cv2.boxFilter(gmag, ddepth=-1, ksize=(win, win))

    low_mask = (smoothed < float(model_cfg.internal_gradient_max)).astype(np.uint8)
    # Close small gaps so a blur with stripey internal residue still forms one component.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(low_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    h, w = gray.shape

    min_side = int(model_cfg.min_side_px)
    max_side = int(model_cfg.max_side_px)
    aspect_min = float(model_cfg.aspect_min)
    aspect_max = float(model_cfg.aspect_max)
    boundary_min = float(model_cfg.boundary_gradient_min)
    annulus = max(int(model_cfg.boundary_annulus_px), 1)

    out = []
    for lbl in range(1, num_labels):
        x, y, ww, hh, _area = stats[lbl]
        if ww < min_side or hh < min_side:
            continue
        if ww > max_side or hh > max_side:
            continue
        aspect = ww / max(hh, 1)
        if aspect < aspect_min or aspect > aspect_max:
            continue

        x1 = max(x - annulus, 0)
        y1 = max(y - annulus, 0)
        x2 = min(x + ww + annulus, w)
        y2 = min(y + hh + annulus, h)

        component = (labels[y1:y2, x1:x2] == lbl)
        # Annulus = pixels just outside the component within the padded ROI.
        outer = (~component) & (
            cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
        )
        if not outer.any():
            continue
        boundary_gmag = float(gmag[y1:y2, x1:x2][outer].mean())
        if boundary_gmag < boundary_min:
            continue
        internal_gmag = float(smoothed[y1:y2, x1:x2][component].mean()) + 1e-6
        score = float(np.clip(boundary_gmag / (boundary_gmag + 4.0 * internal_gmag), 0.0, 1.0))

        box = (x, y, x + ww, y + hh)
        if _is_inside_camera_mask(box, w, h, cm_cfg):
            continue
        out.append((*box, score))
    return out


class BlurRegionRunner(StageRunner):
    stage_name = "blur_region"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg
        df = load_input_frame(cfg, context.inputs)
        if df.empty:
            print("[blur_region] empty input manifest")
            empty = detections_to_frame([])
            out_path = ensure_output_dir(context.output_paths, "detections")
            write_parquet(empty, out_path)
            return StageResult(outputs={"detections": out_path}, metadata={"rows": 0})

        model_cfg = cfg.model
        cm_cfg = getattr(cfg, "camera_mask", None)
        teacher_model = str(getattr(model_cfg, "teacher_model", "classical_gradient_v1"))
        teacher_version = str(getattr(model_cfg, "teacher_version", ""))

        detections: List[Detection] = []
        n_imgs = 0
        n_dets = 0
        t0 = time.time()
        for row in df.to_dict(orient="records"):
            img_path = row.get("image_path")
            if not img_path:
                continue
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"[blur_region] failed to read {img_path!r}")
                continue
            n_imgs += 1
            boxes = _detect_blur_regions(image, model_cfg, cm_cfg)
            sample_id = row.get("sample_id") or img_path
            common_kwargs = dict(
                sample_id=str(sample_id),
                image_path=str(img_path),
                recording_id=row.get("recording_id"),
                face=row.get("face"),
                dataset=row.get("dataset"),
                teacher_model=teacher_model,
                teacher_version=teacher_version,
            )
            if not boxes:
                detections.append(Detection(cls=None, **common_kwargs))
                continue
            for x1, y1, x2, y2, score in boxes:
                detections.append(
                    Detection(
                        cls="blur_candidate",
                        bbox_x1=float(x1),
                        bbox_y1=float(y1),
                        bbox_x2=float(x2),
                        bbox_y2=float(y2),
                        score=float(score),
                        **common_kwargs,
                    )
                )
                n_dets += 1

        out_path = ensure_output_dir(context.output_paths, "detections")
        write_parquet(detections_to_frame(detections), out_path)
        elapsed = time.time() - t0
        print(
            f"[blur_region] {n_imgs} images, {n_dets} candidates "
            f"({n_dets / max(n_imgs, 1):.2f} per image) in {elapsed:.1f}s"
        )
        return StageResult(
            outputs={"detections": out_path},
            metadata={"rows": len(detections), "images": n_imgs, "candidates": n_dets},
        )
