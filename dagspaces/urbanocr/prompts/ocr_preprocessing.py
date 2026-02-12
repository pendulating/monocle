"""OCR-specific preprocessing for text spotting with Qwen3-VL.

This module provides preprocessing and postprocessing functions for the OCR stage,
formatting prompts for text spotting and parsing structured JSON responses.
"""

import base64
import copy
import json
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# Default OCR system prompt (following Qwen3-VL cookbook pattern)
DEFAULT_OCR_SYSTEM_PROMPT = """You are a helpful assistant."""


# Confidence levels (categorical scale - LLMs perform better with discrete categories)
# 10 semantic levels for fine-grained confidence assessment
CONFIDENCE_LEVELS = [
    "guessing",           # ~0.00-0.10: can't really see it, making assumptions
    "very_uncertain",     # ~0.10-0.20: barely visible, highly doubtful
    "uncertain",          # ~0.20-0.35: partially visible, significant doubt
    "somewhat_uncertain", # ~0.35-0.50: can see something but unclear
    "plausible",          # ~0.50-0.65: readable but notable ambiguity
    "likely",             # ~0.65-0.75: fairly clear, some uncertainty
    "confident",          # ~0.75-0.85: clearly visible, minor uncertainty
    "very_confident",     # ~0.85-0.92: very clear, little doubt
    "certain",            # ~0.92-0.98: crystal clear, essentially no doubt
    "absolute",           # ~0.98-1.00: perfect clarity, zero doubt
]

# Mapping from categorical to numeric (for downstream analysis)
CONFIDENCE_TO_NUMERIC = {
    "guessing": 0.05,
    "very_uncertain": 0.15,
    "uncertain": 0.28,
    "somewhat_uncertain": 0.42,
    "plausible": 0.58,
    "likely": 0.70,
    "confident": 0.80,
    "very_confident": 0.88,
    "certain": 0.95,
    "absolute": 0.99,
}


# Default OCR user prompt (following Qwen3-VL cookbook pattern)
# Simple and direct - matches what works in the official examples
DEFAULT_OCR_USER_PROMPT = """Spot all the text in this image at word-level. Output in JSON format as:
[{"bbox_2d": [x1, y1, x2, y2], "text": "detected text", "confidence": "level", "text_type": "category"}, ...]

Confidence levels: guessing, very_uncertain, uncertain, somewhat_uncertain, plausible, likely, confident, very_confident, certain, absolute
Text types: sign, storefront, vehicle, billboard, graffiti, street_name, building_number, other

Only report text you can clearly read. Return [] if no text visible."""


# JSON schema for structured output enforcement
# Note: No maxItems on outer array - allows unlimited detections per image
OCR_OUTPUT_SCHEMA = {
    "type": "array",
    "description": "Array of text detections (no limit on count)",
    "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The detected text content"},
            "bbox_2d": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 999},
                "minItems": 4,
                "maxItems": 4,  # Exactly 4 coordinates: [x1, y1, x2, y2]
                "description": "Bounding box [x1, y1, x2, y2] normalized to 0-999"
            },
            "confidence": {
                "type": "string",
                "enum": ["guessing", "very_uncertain", "uncertain", "somewhat_uncertain", "plausible", "likely", "confident", "very_confident", "certain", "absolute"],
                "description": "Detection confidence level (10-point semantic scale)"
            },
            "text_type": {
                "type": "string",
                "enum": ["sign", "storefront", "vehicle", "billboard", "graffiti", "street_name", "building_number", "other"],
                "description": "Classification of text source"
            }
        },
        "required": ["text", "bbox_2d", "confidence", "text_type"]
    }
}


def _ensure_numpy_image(image: Any, sample_id: Optional[str] = None) -> "np.ndarray":
    """Coerce image-like input to numpy array."""
    if np is None:
        raise RuntimeError("NumPy is required for OCR image preprocessing")
    
    if image is None:
        raise ValueError(f"Image data is required (sample_id: {sample_id})")
    
    if isinstance(image, np.ndarray):
        return image
    
    # Handle PyArrow tensors
    try:
        import pyarrow as pa
        if isinstance(image, pa.Tensor):
            return image.to_numpy()
    except ImportError:
        pass
    
    if isinstance(image, str):
        raise ValueError(f"Expected image data, got string path (sample_id: {sample_id})")
    
    if _PIL_AVAILABLE and isinstance(image, PILImage.Image):
        return np.asarray(image.convert("RGB"))
    
    # Try to convert via numpy
    try:
        return np.asarray(image)
    except Exception as e:
        raise ValueError(f"Failed to convert image to numpy array (sample_id: {sample_id}): {e}")


def _convert_image_to_base64(image: Any, sample_id: Optional[str] = None) -> Optional[str]:
    """Convert numpy array to base64 string for vLLM inference."""
    if image is None:
        return None
    
    if np is None or not _PIL_AVAILABLE:
        raise RuntimeError("NumPy and PIL are required for image conversion")
    
    try:
        np_image = _ensure_numpy_image(image, sample_id)
        img_copy = np_image.copy()
        
        if img_copy.dtype != np.uint8:
            img_copy = img_copy.astype(np.uint8)
        
        pil_img = PILImage.fromarray(img_copy).convert("RGB")
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        
        return f"data:image/jpeg;base64,{base64_bytes.decode('utf-8')}"
    except Exception as e:
        raise ValueError(f"Failed to convert image to base64 (sample_id: {sample_id}): {e}")


def get_system_prompt(cfg: DictConfig) -> str:
    """Get OCR system prompt from config or default."""
    try:
        prompt_cfg = getattr(cfg, "prompt", None)
        if prompt_cfg:
            system = getattr(prompt_cfg, "system", None)
            if system and str(system).strip():
                return str(system).strip()
    except Exception:
        pass
    return DEFAULT_OCR_SYSTEM_PROMPT


def get_user_prompt(cfg: DictConfig) -> str:
    """Get OCR user prompt from config or default."""
    try:
        prompt_cfg = getattr(cfg, "prompt", None)
        if prompt_cfg:
            user = getattr(prompt_cfg, "user_template", None)
            if user and str(user).strip():
                return str(user).strip()
    except Exception:
        pass
    return DEFAULT_OCR_USER_PROMPT


def get_sampling_params(cfg: DictConfig) -> Dict[str, Any]:
    """Get sampling parameters for OCR inference."""
    try:
        params = getattr(cfg, "sampling_params_ocr", None)
        if params:
            return dict(params)
    except Exception:
        pass
    
    # Default sampling params for OCR
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 4096,  # OCR can produce long outputs
        "stop": [],
        "repetition_penalty": 1.1,  # Discourage repeating same detections
    }


def _normalize_sampling_params(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure stop is always a list."""
    sp = dict(sp)
    stop_val = sp.get("stop")
    if stop_val is None:
        sp["stop"] = []
    elif not isinstance(stop_val, list):
        sp["stop"] = [str(stop_val)]
    return sp


def ocr_preprocess(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess row for OCR inference.
    
    Converts image to base64 and builds messages structure for vLLM.
    
    Args:
        row: Input row with image and metadata
        cfg: Configuration object
        
    Returns:
        Preprocessed row with messages and sampling_params
    """
    sample_id = row.get("sample_id", "unknown")
    
    # Get prompts
    system_prompt = get_system_prompt(cfg)
    user_prompt = get_user_prompt(cfg)
    
    # Convert image to base64
    image = row.get("image")
    if image is None:
        raise ValueError(f"Image is required for OCR (sample_id: {sample_id})")
    
    image_base64 = _convert_image_to_base64(image, sample_id)
    if not image_base64:
        raise ValueError(f"Failed to convert image (sample_id: {sample_id})")
    
    # Build messages
    user_content = [
        {"type": "text", "text": user_prompt},
        {"type": "image_url", "image_url": {"url": image_base64}}
    ]
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    
    # Get sampling params
    sp = get_sampling_params(cfg)
    sp = _normalize_sampling_params(sp)
    
    # Check for structured output
    use_structured = getattr(getattr(cfg, "prompt", None), "structured_output", True)
    if use_structured:
        sp["guided_decoding"] = {"json": OCR_OUTPUT_SCHEMA}
    
    # Build result with only lightweight metadata
    result = {
        "messages": messages,
        "sampling_params": sp,
        "ts_start": datetime.utcnow().timestamp(),
    }
    
    # Preserve lightweight metadata columns
    for key in ["sample_id", "image_path", "location_group", "location_id", "face"]:
        if key in row and isinstance(row[key], (str, int, float, type(None))):
            result[key] = row[key]
    
    # Preserve tile info for coordinate transformation in postprocessing
    if "_tile_info" in row:
        result["_tile_info"] = row["_tile_info"]
    
    # Preserve tile identifiers
    for key in ["tile_idx", "tile_row", "tile_col"]:
        if key in row:
            result[key] = row[key]
    
    return result


def parse_ocr_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse OCR model response into list of detections.
    
    Args:
        response_text: Raw model response (expected to be JSON array)
        
    Returns:
        List of detection dictionaries
    """
    if not response_text or not response_text.strip():
        return []
    
    text = response_text.strip()
    
    # Try to parse as JSON directly
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and "detections" in result:
            return result["detections"]
        else:
            # Wrap single detection in list
            return [result] if result else []
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON array from response
    try:
        # Find array bounds
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON objects from response
    try:
        # Find all JSON objects
        objects = []
        pattern = r'\{[^{}]*\}'
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                obj = json.loads(match)
                if "text" in obj:
                    objects.append(obj)
            except json.JSONDecodeError:
                continue
        if objects:
            return objects
    except Exception:
        pass
    
    # Fallback: return empty list
    return []


def flatten_detections(
    detections: List[Dict[str, Any]],
    row_metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Flatten detections into individual rows with metadata.
    
    Args:
        detections: List of detection dictionaries from parse_ocr_response
        row_metadata: Metadata from the original row to include in each output row
        
    Returns:
        List of flat rows, one per detection
    """
    # Handle empty detections (works with lists and numpy arrays)
    is_empty = False
    if detections is None:
        is_empty = True
    elif hasattr(detections, "__len__"):
        is_empty = len(detections) == 0
    elif hasattr(detections, "size"):  # numpy array
        is_empty = detections.size == 0
    else:
        is_empty = not detections
    
    if is_empty:
        # Return single row indicating no detections
        return [{
            **row_metadata,
            "text": None,
            "bbox_x1": None,
            "bbox_y1": None,
            "bbox_x2": None,
            "bbox_y2": None,
            "confidence": None,
            "confidence_numeric": None,
            "text_type": None,
            "detection_count": 0,
        }]
    
    rows = []
    for det in detections:
        row = dict(row_metadata)
        
        # Extract text
        row["text"] = det.get("text")
        
        # Extract bounding box (handle numpy arrays and lists)
        bbox = det.get("bbox_2d")
        if bbox is None:
            bbox = det.get("bbox")
        
        # Convert to list if numpy array
        if bbox is not None:
            try:
                if hasattr(bbox, "tolist"):
                    bbox = bbox.tolist()
                elif not isinstance(bbox, list):
                    bbox = list(bbox)
            except Exception:
                bbox = None
        
        if bbox is not None and len(bbox) >= 4:
            try:
                row["bbox_x1"] = int(bbox[0])
                row["bbox_y1"] = int(bbox[1])
                row["bbox_x2"] = int(bbox[2])
                row["bbox_y2"] = int(bbox[3])
            except (ValueError, TypeError, IndexError):
                row["bbox_x1"] = None
                row["bbox_y1"] = None
                row["bbox_x2"] = None
                row["bbox_y2"] = None
        else:
            row["bbox_x1"] = None
            row["bbox_y1"] = None
            row["bbox_x2"] = None
            row["bbox_y2"] = None
        
        # Extract confidence (categorical or numeric)
        conf = det.get("confidence")
        if conf is not None:
            if isinstance(conf, str):
                # Categorical confidence
                row["confidence"] = conf.lower().strip()
                # Add numeric equivalent for analysis
                row["confidence_numeric"] = CONFIDENCE_TO_NUMERIC.get(row["confidence"], 0.5)
            else:
                # Legacy numeric confidence - convert to categorical
                try:
                    conf_float = float(conf)
                    row["confidence_numeric"] = conf_float
                    # Map to categorical
                    # Map numeric to 10-level categorical scale
                    if conf_float >= 0.98:
                        row["confidence"] = "absolute"
                    elif conf_float >= 0.92:
                        row["confidence"] = "certain"
                    elif conf_float >= 0.85:
                        row["confidence"] = "very_confident"
                    elif conf_float >= 0.75:
                        row["confidence"] = "confident"
                    elif conf_float >= 0.65:
                        row["confidence"] = "likely"
                    elif conf_float >= 0.50:
                        row["confidence"] = "plausible"
                    elif conf_float >= 0.35:
                        row["confidence"] = "somewhat_uncertain"
                    elif conf_float >= 0.20:
                        row["confidence"] = "uncertain"
                    elif conf_float >= 0.10:
                        row["confidence"] = "very_uncertain"
                    else:
                        row["confidence"] = "guessing"
                except (ValueError, TypeError):
                    row["confidence"] = None
                    row["confidence_numeric"] = None
        else:
            row["confidence"] = None
            row["confidence_numeric"] = None
        
        # Extract text type
        row["text_type"] = det.get("text_type")
        
        # Add detection count
        row["detection_count"] = len(detections)
        
        rows.append(row)
    
    return rows


def ocr_postprocess(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Postprocess OCR model response.
    
    Parses JSON response and prepares metadata for flattening.
    Note: Actual flattening to multiple rows happens in the stage.
    
    Args:
        row: Row with generated_text from model
        cfg: Configuration object
        
    Returns:
        Postprocessed row with parsed detections
    """
    ts_end = datetime.utcnow().timestamp()
    generated_text = str(row.get("generated_text", "")).strip()
    
    # Parse detections
    detections = parse_ocr_response(generated_text)
    
    # Build metadata
    metadata = {
        "ts_start": row.get("ts_start"),
        "ts_end": ts_end,
        "raw_response": generated_text,
        "detection_count": len(detections),
    }
    
    # Collect row metadata for flattening
    row_metadata = {}
    for key in ["sample_id", "image_path", "location_group", "location_id", "face"]:
        if key in row:
            row_metadata[key] = row[key]
    row_metadata["ts_processed"] = ts_end
    
    # Store detections and metadata for flattening
    result = {
        **row_metadata,
        "_detections": detections,
        "_metadata": metadata,
        "model_response": generated_text,
    }
    
    return result

