"""House-number detection stage.

Uses a multimodal VLM (default Qwen3-VL-2B via vLLM) to perform text
spotting with bbox_2d output, then keeps only detections whose text
matches a short numeric house-number pattern (1-5 digits with an
optional trailing letter, e.g. ``12``, ``4A``, ``1234``) and whose bbox
is plausibly sized for a building-mounted number (5-200 px on each
side, aspect ratio <= 5:1).

Routes through ``dagspaces.common.vllm_inference.run_vllm_inference`` so
the stage inherits the shared chunked-DP-worker batching, multimodal
message construction, and guided-decoding plumbing used by urban-vqa.
"""

from __future__ import annotations

import copy
import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.orchestrator import StageExecutionContext, StageResult
from dagspaces.common.runners.base import StageRunner
from dagspaces.common.stage_utils import extract_last_json

from ._common import (
    Detection,
    detections_to_frame,
    ensure_output_dir,
    in_camera_mask,
    load_input_frame,
    write_parquet,
)


_SYSTEM_PROMPT = (
    "You are a precise text-spotting assistant for street-view imagery. "
    "Return only valid JSON that matches the requested schema."
)

_USER_PROMPT = (
    "Detect every visible house number, building number, or door number in this "
    "street-level image. House numbers are short numeric labels (typically 1-5 "
    "digits, sometimes with a trailing letter like '15B') mounted on building "
    "facades, doors, awnings, lintels, or small signs near entrances. "
    "Ignore street names, advertisements, license plates, traffic signs, "
    "graffiti, and any non-numeric text. Return a JSON object with a "
    "'detections' array. Each entry must have 'text' (the OCR'd characters) "
    "and 'bbox_2d' (an array of four integer pixel coordinates [x1, y1, x2, y2]). "
    "If no house numbers are visible, return an empty 'detections' array."
)


_HOUSE_NUMBER_RE = re.compile(r"^\s*(\d{1,5})([A-Za-z])?\s*$")


def _ensure_json_schema_dict(schema: Any) -> Optional[Dict[str, Any]]:
    if schema is None:
        return None
    if isinstance(schema, DictConfig):
        try:
            return OmegaConf.to_container(schema, resolve=True)
        except Exception:
            return None
    if isinstance(schema, dict):
        return copy.deepcopy(schema)
    return None


def _build_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Inject the fixed text-spotting prompt into every row."""
    out = df.copy()
    out["prompt"] = _USER_PROMPT
    return out


def _load_pil_image(path: Any):
    """Load a JPEG/PNG into a PIL ``RGB`` image, or return None on failure."""
    if not path or not isinstance(path, str):
        return None
    try:
        from PIL import Image as PILImage
    except ImportError:
        return None
    try:
        if not os.path.isfile(path):
            return None
        im = PILImage.open(path)
        im.load()
        return im.convert("RGB")
    except Exception:
        return None


def _make_preprocess(cfg: DictConfig):
    """Build a self-contained preprocess for the text-spotting VLM call.

    Avoids importing urbanvqa.prompts.unified.preprocess_simple — that
    helper assumes a urbanvqa-shaped config (cfg.prompt.system,
    cfg.sampling_params_vqa, cfg.prompt.user_template, ...) which our
    dagspace doesn't define. Rolling our own keeps the dependency
    surface narrow and avoids silent preprocess_failed errors that drop
    every row before vLLM ever sees it.
    """
    structured_schema = _ensure_json_schema_dict(
        getattr(getattr(cfg, "model", None), "structured_output", None)
    )
    sp_dict = OmegaConf.to_container(
        getattr(cfg.model, "sampling_params", {}), resolve=True
    ) or {}
    if structured_schema:
        sp_dict = dict(sp_dict)
        sp_dict["guided_decoding"] = {"json": structured_schema}

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        pil_img = _load_pil_image(row.get("image_path"))
        # vLLM's chat-template parser only knows ``image_url`` and
        # ``image_pil`` part types — a bare ``{"type": "image"}`` (or
        # even ``{"type": "image", "image": <PIL>}`` if the chunked-DP
        # rewrite doesn't run for some reason) raises NotImplementedError
        # in vllm/entrypoints/chat_utils.py::_parse_chat_message_content_part.
        # Emit ``image_pil`` directly so the message is valid regardless
        # of which vLLM entry point the runtime ends up using.
        user_content: List[Any]
        if pil_img is not None:
            user_content = [
                {"type": "text", "text": _USER_PROMPT},
                {"type": "image_pil", "image_pil": pil_img},
            ]
        else:
            user_content = _USER_PROMPT  # text-only fallback

        result: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "sampling_params": copy.deepcopy(sp_dict),
        }

        for key in ("sample_id", "image_path", "recording_id", "face", "dataset"):
            v = row.get(key)
            if isinstance(v, (str, int, float, type(None))):
                result[key] = v
        return result

    return preprocess


def _make_postprocess():
    def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        text = row.get("generated_text", "")
        parsed = extract_last_json(text) if text else None
        detections = []
        if isinstance(parsed, dict):
            raw = parsed.get("detections")
            if isinstance(raw, list):
                detections = raw
        return {
            "sample_id": row.get("sample_id"),
            "image_path": row.get("image_path"),
            "recording_id": row.get("recording_id"),
            "face": row.get("face"),
            "dataset": row.get("dataset"),
            "_parsed_detections": detections,
            "_raw_response": text,
        }

    return postprocess


def _filter_and_emit(
    raw_rows: pd.DataFrame,
    teacher_model: str,
    teacher_version: str,
    min_side: int,
    max_side: int,
    max_aspect: float,
    camera_mask_cfg,
) -> List[Detection]:
    out: List[Detection] = []
    for _, r in raw_rows.iterrows():
        sample_id = r.get("sample_id") or r.get("image_path")
        common = dict(
            sample_id=str(sample_id),
            image_path=str(r.get("image_path") or ""),
            recording_id=r.get("recording_id"),
            face=r.get("face"),
            dataset=r.get("dataset"),
            teacher_model=teacher_model,
            teacher_version=teacher_version,
        )
        detections = r.get("_parsed_detections") or []
        kept = 0
        for det in detections:
            if not isinstance(det, dict):
                continue
            text = det.get("text")
            bbox = det.get("bbox_2d")
            if not isinstance(text, str) or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            m = _HOUSE_NUMBER_RE.match(text)
            if not m:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox)
            except (TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            w = x2 - x1
            h = y2 - y1
            if w < min_side or h < min_side or w > max_side or h > max_side:
                continue
            if max(w, h) / max(min(w, h), 1.0) > max_aspect:
                continue
            if camera_mask_cfg is not None and in_camera_mask(
                (x1, y1, x2, y2), 1024, 1024, camera_mask_cfg
            ):
                # Conservative: assume 1024x1024 for Cyclomedia _1k. The OCR
                # bbox is in the model's input frame which matches the image.
                continue
            out.append(
                Detection(
                    cls="house_number",
                    bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
                    score=1.0,  # VLM doesn't return per-detection confidence
                    **common,
                )
            )
            kept += 1
        if kept == 0:
            out.append(Detection(cls=None, **common))
    return out


class HouseNumberRunner(StageRunner):
    stage_name = "house_number"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg
        df = load_input_frame(cfg, context.inputs)
        out_path = ensure_output_dir(context.output_paths, "detections")

        if df.empty:
            print("[house_number] empty input manifest")
            write_parquet(detections_to_frame([]), out_path)
            return StageResult(outputs={"detections": out_path}, metadata={"rows": 0})

        df = _build_input_frame(df)

        model_cfg = cfg.model
        teacher_model = str(getattr(model_cfg, "teacher_model", "qwen3-vl-2b"))
        teacher_version = str(getattr(model_cfg, "teacher_version", ""))
        min_side = int(getattr(cfg.thresholds, "house_number_min_side_px", 5))
        max_side = int(getattr(cfg.thresholds, "house_number_max_side_px", 200))
        max_aspect = float(getattr(cfg.thresholds, "house_number_max_aspect", 5.0))
        camera_mask_cfg = getattr(cfg, "camera_mask", None)

        preprocess = _make_preprocess(cfg)
        postprocess = _make_postprocess()

        from dagspaces.common.vllm_inference import run_vllm_inference

        t0 = time.time()
        raw = run_vllm_inference(
            df=df, cfg=cfg, preprocess=preprocess, postprocess=postprocess,
            stage_name="opfvl_house_number",
        )
        infer_s = time.time() - t0

        detections = _filter_and_emit(
            raw, teacher_model, teacher_version,
            min_side, max_side, max_aspect, camera_mask_cfg,
        )
        write_parquet(detections_to_frame(detections), out_path)
        n_real = sum(1 for d in detections if d.cls is not None)
        print(
            f"[house_number] {len(raw)} VLM responses in {infer_s:.1f}s; "
            f"{n_real} numeric detections kept after filter"
        )
        return StageResult(
            outputs={"detections": out_path},
            metadata={
                "rows": len(detections),
                "vlm_responses": int(len(raw)),
                "kept_detections": n_real,
            },
        )
