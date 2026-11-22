"""UrbanVQA stage — multimodal VQA inference via direct vLLM.

Thin preprocess/postprocess callback pair that feeds into
``dagspaces.common.vllm_inference.run_vllm_inference``.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

try:
    from jinja2 import Environment, StrictUndefined
    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

_DEFAULT_IMAGE_PROMPT = "What do you see in this image?"


# ---------------------------------------------------------------------------
# Prompt / template helpers
# ---------------------------------------------------------------------------

def _resolve_default_prompt(cfg: DictConfig) -> str:
    """Resolve the default prompt for image-only inputs."""
    try:
        data_cfg = getattr(cfg, "data", None)
        candidate = getattr(data_cfg, "default_prompt", None) if data_cfg is not None else None
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    except Exception:
        pass
    return _DEFAULT_IMAGE_PROMPT


@lru_cache(maxsize=512)
def _get_compiled_template(template_str: str):
    """Compile a Jinja2 template. The cache is keyed by the template string."""
    env = Environment(undefined=StrictUndefined)  # Raise error on undefined variables
    return env.from_string(template_str)


def render_prompt_template(template_str: str, context: Dict[str, Any]) -> str:
    """Render Jinja2 template with variable substitution."""
    if not _JINJA2_AVAILABLE:
        raise ValueError("Jinja2 is not available. Install with: pip install jinja2")
    template = _get_compiled_template(template_str)
    return template.render(**context)


def _prepare_image_content(image: Any, sample_id: Optional[str] = None) -> Dict[str, Any]:
    """Prepare image content in vLLM-compatible chat format.

    Detect the input format. Accept a PIL image, a local file path, an
    HTTP URL, or a base64 string.

    Args:
        image: Image in one of the supported formats.
        sample_id: Optional sample ID for error messages.

    Returns:
        Content block dict for a chat message. A PIL image maps to
        ``{"type": "image", "image": <PIL>}``. A URL or base64 string
        maps to ``{"type": "image_url", "image_url": {"url": ...}}``.

    Raises:
        ValueError: If the image is missing or the format is not supported.
    """
    if image is None:
        raise ValueError(f"No image provided for sample {sample_id}")

    if isinstance(image, str):
        # HTTP(S) URL: pass through in OpenAI image_url form.
        if image.startswith(("http://", "https://")):
            return {
                "type": "image_url",
                "image_url": {"url": image},
            }
        # Full data URI: base64 payload with an explicit type.
        if image.startswith("data:image/"):
            return {
                "type": "image_url",
                "image_url": {"url": image},
            }
        # An existing path wins over the base64 guess, so a bad path
        # raises a clear file error. A long string that names no file
        # is base64 data.
        if len(image) > 100 and not os.path.exists(image):
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
        # Local file path: load to PIL.
        try:
            from PIL import Image as PILImage

            image = PILImage.open(image).convert("RGB")
        except Exception as e:
            raise ValueError(
                f"Failed to load image from path '{image}' for sample {sample_id}: {e}"
            ) from e

    # PIL image (or PIL-like object): pass through without a copy.
    if hasattr(image, "convert") or hasattr(image, "size"):
        return {"type": "image", "image": image}

    # Torch tensor with pre-computed embeddings: pass through.
    try:
        import torch

        if isinstance(image, torch.Tensor):
            return {"type": "image_embeds", "image_embeds": image}
    except ImportError:
        pass

    raise ValueError(
        f"Unsupported image format {type(image).__name__} for sample {sample_id}"
    )


def _group_by_prompt_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """Group rows by prompt for efficient batch processing.

    Reorder rows so that rows with the same prompt sit together. Prompt
    tokenization can then be reused within each group. Apply this before
    you create a Ray Dataset.

    Args:
        df: DataFrame with an optional 'prompt' column.

    Returns:
        DataFrame sorted by prompt groups. Without a 'prompt' column,
        return the input unchanged.
    """
    if "prompt" not in df.columns:
        return df

    # Add a grouping key
    df = df.copy()
    df["_prompt_group"] = df.groupby("prompt").ngroup()
    # Sort by prompt group so same prompts are batched together
    df = df.sort_values("_prompt_group").reset_index(drop=True)
    # Drop the temporary grouping column
    df = df.drop(columns=["_prompt_group"], errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Guided decoding (structured output) helpers
# ---------------------------------------------------------------------------

def _normalize_sampling_params(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize sampling params to ensure compatibility with vLLM."""
    sp_normalized = dict(sp)
    stop_val = sp_normalized.get("stop")
    if stop_val is None:
        sp_normalized["stop"] = []
    elif not isinstance(stop_val, list):
        sp_normalized["stop"] = [str(stop_val)]
    return sp_normalized


def _ensure_json_schema_dict(schema: Any) -> Optional[Dict[str, Any]]:
    """Convert OmegaConf/DictConfig schemas into a plain Python dict."""
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


def _extract_enum_choices(schema: Any) -> Optional[List[str]]:
    """Recursively extract enum choices from a JSON schema."""
    if isinstance(schema, dict):
        enum_val = schema.get("enum")
        if isinstance(enum_val, (list, tuple)):
            choices = [
                str(item) for item in enum_val if isinstance(item, (str, int, float))
            ]
            if choices:
                return choices
        for value in schema.values():
            choices = _extract_enum_choices(value)
            if choices:
                return choices
    elif isinstance(schema, list):
        for item in schema:
            choices = _extract_enum_choices(item)
            if choices:
                return choices
    return None


def _build_guided_decoding_config(schema: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build the guided decoding config payload expected by vLLM."""
    if not schema:
        return None
    choices = _extract_enum_choices(schema)
    if choices:
        return {"choice": choices}
    return {"json": schema}


# ---------------------------------------------------------------------------
# Sample ID helpers
# ---------------------------------------------------------------------------

def _derive_sample_id_from_path(path_val: Optional[str]) -> Optional[str]:
    if not path_val:
        return None
    base_name = os.path.basename(path_val.rstrip("/"))
    if not base_name:
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", path_val.strip("/"))
        return sanitized or None
    stem, _ = os.path.splitext(base_name)
    candidate = stem or base_name
    return re.sub(r"[^a-zA-Z0-9_]", "_", candidate) or None


def _resolve_row_sample_id(row: Dict[str, Any]) -> Optional[str]:
    """Resolve sample_id from row data, deriving from image_path if needed."""
    sid = row.get("sample_id")
    if sid is not None and isinstance(sid, str) and sid.strip():
        return sid.strip()
    return _derive_sample_id_from_path(row.get("image_path"))


# ---------------------------------------------------------------------------
# Preprocess / Postprocess callbacks
# ---------------------------------------------------------------------------

def _make_preprocess(cfg: DictConfig):
    """Build the preprocess callback for run_vllm_inference.

    Returns a closure that converts each row dict into a dict with
    ``messages``, ``sampling_params``, and lightweight metadata.
    """
    from dagspaces.common.vllm_inference import _is_multimodal_model
    from ..prompts.unified import (
        preprocess_simple,
        unified_preprocess,
        _resolve_pil_image,
    )

    model_source = str(getattr(cfg.model, "model_source", ""))
    is_multimodal = _is_multimodal_model(model_source, cfg)

    # Detect prompting techniques
    hierarchical_enabled = bool(getattr(getattr(cfg.prompt, "hierarchical", None), "enabled", False))
    decision_tree_enabled = bool(getattr(getattr(cfg.prompt, "decision_tree", None), "enabled", False))
    any_technique = any([
        getattr(getattr(cfg.prompt, "adaptive", None), "enabled", False),
        getattr(getattr(cfg.prompt, "retrieval_augmented", None), "enabled", False),
        getattr(getattr(cfg.prompt, "chain_of_thought", None), "enabled", False),
        getattr(getattr(cfg.prompt, "react", None), "enabled", False),
        getattr(getattr(cfg.prompt, "contextual", None), "enabled", False),
        hierarchical_enabled,
        decision_tree_enabled,
    ])

    # Resolve guided decoding schema
    structured_schema = _ensure_json_schema_dict(
        getattr(getattr(cfg, "sampling_params_vqa", None), "structured_output", None)
        or getattr(cfg, "structured_output_schema", None)
    )
    guided_decoding_payload = _build_guided_decoding_config(structured_schema)

    # Get base sampling params for VQA
    try:
        sp_base = OmegaConf.to_container(
            getattr(cfg, "sampling_params_vqa", cfg.sampling_params), resolve=True
        )
    except Exception:
        sp_base = {}
    sp_base = _normalize_sampling_params(sp_base)

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)

        # Ensure sample_id
        if not row.get("sample_id"):
            row["sample_id"] = _resolve_row_sample_id(row)

        # Use unified preprocessing if any technique is enabled
        if any_technique:
            result = unified_preprocess(
                row, cfg,
                is_multimodal=is_multimodal,
                hierarchical_enabled=hierarchical_enabled,
                decision_tree_enabled=decision_tree_enabled,
            )
            if result is not None:
                # Add guided decoding if not already present. A per-row
                # "guided_decoding" payload dict (e.g. {"json": schema} or
                # {"choice": [...]}) overrides the cfg-level schema for that
                # row — used by callers whose constraint varies per sample
                # (e.g. roaming's legal-face menus).
                row_guided = row.get("guided_decoding")
                if "sampling_params" in result:
                    sp = dict(result.get("sampling_params", {}))
                    if isinstance(row_guided, dict) and row_guided:
                        sp["guided_decoding"] = copy.deepcopy(row_guided)
                    elif guided_decoding_payload and "guided_decoding" not in sp:
                        sp["guided_decoding"] = copy.deepcopy(guided_decoding_payload)
                    result["sampling_params"] = sp
                return result

        # Default: simple preprocessing
        result = preprocess_simple(row, cfg, is_multimodal=is_multimodal)

        # Override sampling params with VQA-specific values. A per-row
        # "guided_decoding" payload dict takes precedence over the cfg-level
        # schema (see note in the unified branch above).
        sp_local = dict(sp_base)
        row_guided = row.get("guided_decoding")
        if isinstance(row_guided, dict) and row_guided:
            sp_local["guided_decoding"] = copy.deepcopy(row_guided)
        elif guided_decoding_payload:
            sp_local["guided_decoding"] = copy.deepcopy(guided_decoding_payload)
        result["sampling_params"] = sp_local

        # For direct vLLM multimodal path: ensure image is in the messages
        # The common vllm_inference.py expects images in message content blocks
        if is_multimodal:
            pil_img = _resolve_pil_image(row)
            if pil_img is None and _names_an_image(row):
                from dagspaces.common.vllm_inference import ImageLoadError

                raise ImageLoadError(
                    f"image did not load: {str(row.get('image_path'))[:200]} "
                    f"(a missing file, an unreadable file, or a URL, which the stage "
                    f"does not download)"
                )
            if pil_img is not None:
                # Rewrite messages with PIL image in content blocks
                # (for compatibility with common vllm_inference multimodal path)
                messages = result.get("messages", [])
                new_messages = []
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content")
                        if isinstance(content, list):
                            # Replace bare {"type": "image"} with PIL-bearing block
                            new_content = []
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "image":
                                    new_content.append({
                                        "type": "image",
                                        "image": pil_img,
                                    })
                                else:
                                    new_content.append(block)
                            new_messages.append({**msg, "content": new_content})
                        elif isinstance(content, str):
                            # Text-only user message + image → multimodal
                            new_messages.append({
                                **msg,
                                "content": [
                                    {"type": "text", "text": content},
                                    {"type": "image", "image": pil_img},
                                ],
                            })
                        else:
                            new_messages.append(msg)
                    else:
                        new_messages.append(msg)
                result["messages"] = new_messages
                # Remove the separate "image" key to avoid serializing pixel data
                result.pop("image", None)

        # Carry forward lightweight metadata
        for key in ("sample_id", "prompt", "image_path", "image_url"):
            if key in row and isinstance(row[key], (str, int, float, type(None))):
                result.setdefault(key, row[key])

        return result

    return preprocess


def _names_an_image(row: Dict[str, Any]) -> bool:
    """True when the row gives an image (a path or an in-memory image)."""
    path = row.get("image_path")
    if isinstance(path, str) and path.strip():
        return True
    return row.get("image") is not None


def _make_postprocess(cfg: DictConfig):
    """Build the postprocess callback for run_vllm_inference."""
    from dagspaces.common.stage_utils import extract_last_json

    def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        generated = row.get("generated_text", "")
        reasoning = row.get("generated_reasoning", "")

        # Try to parse structured output
        answer = generated
        structured_fields: Dict[str, Any] = {}
        if generated and generated.strip().startswith("{"):
            parsed = extract_last_json(generated)
            if parsed:
                answer = parsed.get("answer", generated)
                structured_fields = parsed

        # Build clean result dict
        result: Dict[str, Any] = {
            "sample_id": row.get("sample_id"),
            "prompt": row.get("prompt"),
            "answer": str(answer) if answer is not None else "",
            "model_response": generated,
        }

        # Add reasoning if present
        if reasoning:
            result["model_reasoning"] = reasoning

        # Add structured fields
        for k, v in structured_fields.items():
            if k != "answer" and isinstance(v, (str, int, float, bool, type(None))):
                result[k] = v

        # Add metadata
        result["metadata"] = {
            "usage": row.get("usage"),
        }

        # Preserve lightweight row metadata
        for key in ("image_path", "image_url"):
            val = row.get(key)
            if val is not None and isinstance(val, str):
                result[key] = val

        return result

    return postprocess


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_vqa_stage(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run VQA inference on a dataset using direct vLLM.

    Args:
        df: DataFrame with columns: prompt, image_path/image_url, sample_id
        cfg: Configuration object with model, sampling_params, prompt settings

    Returns:
        DataFrame with columns: sample_id, prompt, answer, model_response, metadata
    """
    from dagspaces.common.vllm_inference import run_vllm_inference

    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    if df is None or len(df) == 0:
        print("[run_vqa_stage] Empty input, returning empty DataFrame")
        return pd.DataFrame()

    print(f"[run_vqa_stage] Processing {len(df)} rows with direct vLLM inference")

    preprocess = _make_preprocess(cfg)
    postprocess = _make_postprocess(cfg)

    return run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=preprocess,
        postprocess=postprocess,
        stage_name="urbanvqa_vqa",
    )
