"""Vehicle / license-plate region stage.

Runs YOLOv8 object detection filtered to COCO vehicle classes and emits
per-vehicle license-plate candidate bboxes via a position prior on the
vehicle bbox: roughly the lower ``plate_lower_frac`` of the vehicle, and
the central ``plate_center_frac`` horizontally. Front and rear plates
both fall in this region (whichever is visible). Side-on vehicles
produce a plate region that may not actually contain a plate; the
stage emits it anyway and lets downstream filtering or training handle
the noise — for Stage 1 with 16x16 patch labels this is acceptable.

The whole-vehicle bbox is **not** emitted as a class. Vehicles are not
an end-user privacy class; they exist only as the position prior for
license_plate.
"""

from __future__ import annotations

import os
import time
from typing import List

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


def _load_yolo(model_cfg: DictConfig):
    from ultralytics import YOLO
    return YOLO(str(model_cfg.model_source))


def _device_arg(model_cfg: DictConfig) -> str | int | None:
    dev = str(getattr(model_cfg, "device", "auto")).lower()
    if dev == "auto":
        return None
    return dev


def _plate_region_from_vehicle(
    vbox: tuple, lower_frac: float, center_frac: float,
) -> tuple:
    """Return the plate-candidate bbox derived from a vehicle bbox.

    Lower ``lower_frac`` of the vehicle vertically, central
    ``center_frac`` horizontally. Both fractions are in (0, 1].
    """
    x1, y1, x2, y2 = vbox
    w = x2 - x1
    h = y2 - y1
    plate_h = max(h * lower_frac, 1.0)
    plate_w = max(w * center_frac, 1.0)
    cx = 0.5 * (x1 + x2)
    py1 = y2 - plate_h
    py2 = y2
    px1 = cx - 0.5 * plate_w
    px2 = cx + 0.5 * plate_w
    return px1, py1, px2, py2


class VehicleDetectorRunner(StageRunner):
    stage_name = "vehicle"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg
        df = load_input_frame(cfg, context.inputs)
        out_path = ensure_output_dir(context.output_paths, "detections")

        if df.empty:
            print("[vehicle] empty input manifest")
            write_parquet(detections_to_frame([]), out_path)
            return StageResult(outputs={"detections": out_path}, metadata={"rows": 0})

        model_cfg = cfg.model
        teacher_model = str(getattr(model_cfg, "teacher_model", "yolov8n-coco"))
        teacher_version = str(getattr(model_cfg, "teacher_version", ""))
        vehicle_score_min = float(getattr(cfg.thresholds, "vehicle_score_min", 0.35))
        plate_lower_frac = float(getattr(model_cfg, "plate_lower_frac", 0.30))
        plate_center_frac = float(getattr(model_cfg, "plate_center_frac", 0.35))
        imgsz = int(getattr(model_cfg, "imgsz", 1024))
        conf = float(getattr(model_cfg, "conf", vehicle_score_min))
        iou = float(getattr(model_cfg, "iou", 0.5))
        max_det = int(getattr(model_cfg, "max_det", 64))
        classes = list(getattr(model_cfg, "classes", [2, 3, 5, 7]))
        device = _device_arg(model_cfg)
        camera_mask_cfg = getattr(cfg, "camera_mask", None)

        model = _load_yolo(model_cfg)

        detections: List[Detection] = []
        n_imgs = 0
        n_plates = 0
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
                classes=classes,
                device=device,
                verbose=False,
            )
            if not results:
                detections.append(Detection(cls=None, **common_kwargs))
                continue
            res = results[0]
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                detections.append(Detection(cls=None, **common_kwargs))
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            img_h, img_w = res.orig_shape
            had_any = False
            for i in range(xyxy.shape[0]):
                score = float(scores[i])
                if score < vehicle_score_min:
                    continue
                vx1, vy1, vx2, vy2 = (float(v) for v in xyxy[i])
                if in_camera_mask((vx1, vy1, vx2, vy2), img_w, img_h, camera_mask_cfg):
                    continue
                px1, py1, px2, py2 = _plate_region_from_vehicle(
                    (vx1, vy1, vx2, vy2), plate_lower_frac, plate_center_frac,
                )
                detections.append(
                    Detection(
                        cls="license_plate",
                        bbox_x1=px1, bbox_y1=py1, bbox_x2=px2, bbox_y2=py2,
                        score=score,  # inherits vehicle confidence
                        **common_kwargs,
                    )
                )
                n_plates += 1
                had_any = True

            if not had_any:
                detections.append(Detection(cls=None, **common_kwargs))

        write_parquet(detections_to_frame(detections), out_path)
        elapsed = time.time() - t0
        print(f"[vehicle] {n_imgs} images, {n_plates} plate candidates "
              f"in {elapsed:.1f}s")
        return StageResult(
            outputs={"detections": out_path},
            metadata={"rows": len(detections), "images": n_imgs,
                      "plate_candidates": n_plates},
        )
