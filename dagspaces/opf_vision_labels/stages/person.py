"""Person detection stage (YOLOv8-pose).

Emits two classes per detected pedestrian:

  - ``person`` — whole-body bbox (used directly as a Stage 1 training class
    so the privacy filter can blur an entire person when needed).
  - ``face`` — face bbox derived from the head keypoints (nose, eyes,
    ears) returned by YOLOv8-pose. Visible head keypoints are taken as
    the seed; their bounding rect is expanded by ``face_expand`` and
    capped to the person bbox. If no head keypoints are visible (e.g.,
    person seen from behind), falls back to the top
    ``face_fallback_frac`` of the person bbox.

This replaces an earlier plan to detect Cyclomedia's pre-applied face
blur directly. Empirically the blur signature isn't separable from
ordinary high-detail surfaces by classical CV. Anchoring the face
region to a pose-derived head box is more robust and gives the OPF
vision head usable patch labels even when the actual blur is smaller
than the head box.
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from dagspaces.common.orchestrator import StageExecutionContext, StageResult
from dagspaces.common.runners.base import StageRunner

from ._common import (
    Detection,
    detections_to_frame,
    ensure_output_dir,
    in_camera_mask,
    load_input_frame,
    write_parquet,
)


# COCO-keypoints indices used by YOLOv8-pose (17 kpts).
_HEAD_KEYPOINTS = (0, 1, 2, 3, 4)  # nose, l_eye, r_eye, l_ear, r_ear


def _load_yolo_pose(model_cfg: DictConfig):
    from ultralytics import YOLO  # imported here so the dagspace stays importable
    model_source = str(model_cfg.model_source)
    return YOLO(model_source)


def _device_arg(model_cfg: DictConfig) -> str | int | None:
    dev = str(getattr(model_cfg, "device", "auto")).lower()
    if dev == "auto":
        return None  # let ultralytics pick
    return dev


def _bbox_from_keypoints(
    kpts_xy: np.ndarray, kpts_vis: np.ndarray, indices, expand: float,
    person_bbox: Tuple[float, float, float, float], img_w: int, img_h: int,
) -> Tuple[float, float, float, float] | None:
    """Bounding rect of visible kpts in `indices`, expanded and clipped."""
    visible = [(kpts_xy[i, 0], kpts_xy[i, 1]) for i in indices if kpts_vis[i] > 0.0]
    if not visible:
        return None
    xs = np.array([p[0] for p in visible])
    ys = np.array([p[1] for p in visible])
    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = max(0.5 * (x2 - x1), 1.0) * expand
    half_h = max(0.5 * (y2 - y1), 1.0) * expand
    fx1, fy1 = cx - half_w, cy - half_h
    fx2, fy2 = cx + half_w, cy + half_h

    px1, py1, px2, py2 = person_bbox
    fx1 = max(fx1, px1, 0.0)
    fy1 = max(fy1, py1, 0.0)
    fx2 = min(fx2, px2, float(img_w))
    fy2 = min(fy2, py2, float(img_h))
    if fx2 <= fx1 or fy2 <= fy1:
        return None
    return fx1, fy1, fx2, fy2


def _fallback_face_from_person(
    person_bbox: Tuple[float, float, float, float], frac: float,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = person_bbox
    h = y2 - y1
    return x1, y1, x2, y1 + h * frac


class PersonDetectorRunner(StageRunner):
    stage_name = "person"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg
        df = load_input_frame(cfg, context.inputs)
        out_path = ensure_output_dir(context.output_paths, "detections")

        if df.empty:
            print("[person] empty input manifest")
            write_parquet(detections_to_frame([]), out_path)
            return StageResult(outputs={"detections": out_path}, metadata={"rows": 0})

        model_cfg = cfg.model
        teacher_model = str(getattr(model_cfg, "teacher_model", "yolov8n-pose-coco"))
        teacher_version = str(getattr(model_cfg, "teacher_version", ""))
        person_score_min = float(getattr(cfg.thresholds, "person_score_min", 0.35))
        camera_mask_cfg = getattr(cfg, "camera_mask", None)
        face_expand = float(getattr(model_cfg, "face_expand", 1.6))
        face_fallback_frac = float(getattr(model_cfg, "face_fallback_frac", 0.25))
        imgsz = int(getattr(model_cfg, "imgsz", 1024))
        conf = float(getattr(model_cfg, "conf", person_score_min))
        iou = float(getattr(model_cfg, "iou", 0.5))
        max_det = int(getattr(model_cfg, "max_det", 64))
        device = _device_arg(model_cfg)

        model = _load_yolo_pose(model_cfg)

        detections: List[Detection] = []
        n_imgs = 0
        n_persons = 0
        n_faces = 0
        t0 = time.time()
        for row in df.to_dict(orient="records"):
            img_path = row.get("image_path")
            if not img_path or not os.path.exists(img_path):
                continue
            n_imgs += 1
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

            results = model.predict(
                source=img_path,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                verbose=False,
            )
            if not results:
                detections.append(Detection(cls=None, **common_kwargs))
                continue
            res = results[0]
            boxes = res.boxes
            kpts = res.keypoints
            if boxes is None or len(boxes) == 0:
                detections.append(Detection(cls=None, **common_kwargs))
                continue

            xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
            scores = boxes.conf.cpu().numpy()  # (N,)
            kpt_xy = kpts.xy.cpu().numpy() if kpts is not None else None  # (N, 17, 2)
            kpt_conf = (
                kpts.conf.cpu().numpy() if (kpts is not None and kpts.conf is not None)
                else None
            )  # (N, 17)
            img_h, img_w = res.orig_shape

            had_any = False
            for i in range(xyxy.shape[0]):
                score = float(scores[i])
                if score < person_score_min:
                    continue
                px1, py1, px2, py2 = (float(v) for v in xyxy[i])
                if in_camera_mask((px1, py1, px2, py2), img_w, img_h, camera_mask_cfg):
                    continue
                detections.append(
                    Detection(
                        cls="person",
                        bbox_x1=px1, bbox_y1=py1, bbox_x2=px2, bbox_y2=py2,
                        score=score,
                        **common_kwargs,
                    )
                )
                n_persons += 1
                had_any = True

                face_bbox = None
                if kpt_xy is not None and kpt_conf is not None:
                    face_bbox = _bbox_from_keypoints(
                        kpt_xy[i], kpt_conf[i], _HEAD_KEYPOINTS, face_expand,
                        (px1, py1, px2, py2), img_w, img_h,
                    )
                if face_bbox is None:
                    face_bbox = _fallback_face_from_person(
                        (px1, py1, px2, py2), face_fallback_frac,
                    )
                fx1, fy1, fx2, fy2 = face_bbox
                detections.append(
                    Detection(
                        cls="face",
                        bbox_x1=fx1, bbox_y1=fy1, bbox_x2=fx2, bbox_y2=fy2,
                        score=score,  # inherits person confidence
                        **common_kwargs,
                    )
                )
                n_faces += 1

            if not had_any:
                detections.append(Detection(cls=None, **common_kwargs))

        write_parquet(detections_to_frame(detections), out_path)
        elapsed = time.time() - t0
        print(
            f"[person] {n_imgs} images, {n_persons} persons, {n_faces} faces "
            f"in {elapsed:.1f}s"
        )
        return StageResult(
            outputs={"detections": out_path},
            metadata={"rows": len(detections), "images": n_imgs,
                      "persons": n_persons, "faces": n_faces},
        )
