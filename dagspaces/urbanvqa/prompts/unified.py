"""Unified dynamic prompting framework for VQA.

This module provides a unified preprocessing/postprocessing framework that
integrates all dynamic prompting techniques in priority order.
"""

from typing import Dict, Any, Optional
from omegaconf import DictConfig

# Import numpy for array checking
try:
    import numpy as np
except ImportError:
    np = None

# Import PIL for direct image passthrough to Ray Data LLM's PrepareImageStage
try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def _numpy_to_pil(image_source: Any) -> Optional["PILImage.Image"]:
    """Convert a numpy array (or PIL Image) to a PIL Image for direct passthrough.

    Ray Data LLM's PrepareImageStage natively accepts PIL Images inside
    messages (``{"type": "image", "image": pil_img}``).  Passing PIL directly
    eliminates the expensive base64 ↔ JPEG round-trip that was previously
    required for Arrow serialization.

    Args:
        image_source: numpy array (H×W×C uint8) or PIL Image.

    Returns:
        PIL Image in RGB mode, or None if conversion fails / input is None.
    """
    if image_source is None:
        return None
    if not _PIL_AVAILABLE:
        return None
    try:
        if isinstance(image_source, PILImage.Image):
            return image_source.convert("RGB")
        if np is not None and isinstance(image_source, np.ndarray):
            arr = image_source
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            return PILImage.fromarray(arr).convert("RGB")
    except Exception:
        return None
    return None


def _load_pil_from_path(image_path: Any) -> Optional["PILImage.Image"]:
    """Load a PIL Image directly from a file path on disk.

    This is the lazy-loading counterpart to the old eager
    ``_load_image_from_path`` that decoded every image into a numpy array
    inside the Ray Data pipeline.  By deferring the load to the per-row
    preprocess function, images are decoded just-in-time — immediately
    before ChatTemplateUDF / vLLMEngineStageUDF consume them — instead of
    sitting as ~786 KB numpy arrays in the Ray object store for the
    duration of the vLLM warmup phase.

    Args:
        image_path: Filesystem path (str) to a JPEG/PNG image.

    Returns:
        PIL Image in RGB mode, or None if the path is invalid or loading fails.
    """
    if not image_path or not isinstance(image_path, str):
        return None
    if not _PIL_AVAILABLE:
        return None
    import os
    try:
        if os.path.isfile(image_path):
            pil_img = PILImage.open(image_path)
            pil_img.load()  # force full decode so the file handle can close
            return pil_img.convert("RGB")
    except Exception:
        return None
    return None


def _resolve_pil_image(row: Dict[str, Any]) -> Optional["PILImage.Image"]:
    """Resolve a PIL Image from a row, trying in-memory data first, then disk.

    Priority order:
      1. ``row["image"]`` — numpy array or PIL Image already in memory
      2. ``row["image_path"]`` — lazy load from filesystem

    Args:
        row: Input row dict.

    Returns:
        PIL Image in RGB mode, or None.
    """
    # 1. Try in-memory image (numpy array or PIL Image from read_images path)
    pil = _numpy_to_pil(row.get("image"))
    if pil is not None:
        return pil
    # 2. Lazy-load from image_path (parquet-manifest path)
    return _load_pil_from_path(row.get("image_path"))


def preprocess_simple(row: Dict[str, Any], cfg: DictConfig, is_multimodal: bool = True) -> Dict[str, Any]:
    """Simple preprocessing without any dynamic techniques.

    Converts images to PIL and passes them directly in the OpenAI-style
    message structure so that Ray Data LLM's PrepareImageStage can forward
    them to vLLM without any encode/decode overhead.

    Args:
        row: Input row with prompt and image (numpy array or PIL Image)
        cfg: Configuration object
        is_multimodal: Whether multimodal processing is enabled

    Returns:
        Preprocessed row with messages and sampling_params
    """
    row = dict(row)
    system_prompt = getattr(cfg.prompt, "system", "You are a helpful assistant.")
    sampling_params = getattr(cfg, "sampling_params_vqa", {})

    from ..stages.vqa import _resolve_default_prompt  # local import to avoid cycles

    prompt_raw = row.get("prompt")
    if isinstance(prompt_raw, str):
        prompt = prompt_raw.strip()
    elif prompt_raw is None:
        prompt = ""
    else:
        prompt = str(prompt_raw).strip()
    if not prompt:
        prompt = _resolve_default_prompt(cfg)
    row["prompt"] = prompt

    # ── Image handling ──────────────────────────────────────────────────
    # Resolve the image lazily: try in-memory first (numpy / PIL from
    # read_images path), then fall back to loading from image_path on disk
    # (parquet-manifest path).  This avoids decoding images eagerly in the
    # Ray Data pipeline, which was the primary cause of object-store
    # spilling during the vLLM warmup phase.
    #
    # PrepareImageStage is DISABLED (has_image=False) to avoid fusing the
    # CPU-bound map operators with a concurrency-capped actor pool, which
    # was starving CPU parallelism and duplicating image data in memory.
    #
    # Messages contain a bare {"type": "image"} placeholder (no pixel
    # data) so that ChatTemplateStage inserts the correct vision tokens
    # (<|vision_start|>…<|vision_end|>) in the prompt string.
    pil_img = None
    if is_multimodal:
        pil_img = _resolve_pil_image(row)

    if is_multimodal and pil_img is not None:
        user_content = [
            {"type": "text", "text": prompt},
            # Bare placeholder — no pixel data in messages.  The actual
            # PIL Image travels in result["image"] below.
            {"type": "image"},
        ]
    else:
        # No image available - text-only message
        user_content = prompt

    # Return messages, sampling_params, and — for multimodal — the image
    # column that vLLMEngineStage pops and passes to vLLM as
    # multi_modal_data.
    result = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "sampling_params": sampling_params,
    }

    # Attach PIL Image list for vLLMEngineStage (expects List[PIL.Image])
    if is_multimodal and pil_img is not None:
        result["image"] = [pil_img]

    # Include lightweight, serializable metadata (strings/numbers only)
    for key in ["sample_id", "prompt"]:
        if key in row and isinstance(row[key], (str, int, float, type(None))):
            result[key] = row[key]

    # Ensure raw numpy/path columns are NOT carried forward
    result.pop("image_array", None)
    result.pop("image_data", None)
    result.pop("path", None)

    return result


def unified_preprocess(
    row: Dict[str, Any], 
    cfg: DictConfig,
    is_multimodal: bool = True,
    hierarchical_enabled: bool = False,
    decision_tree_enabled: bool = False
) -> Dict[str, Any]:
    """Unified preprocessing that applies enabled techniques in priority order.
    
    Execution Order:
    1. Pre-processing: Adaptive prompting → RAP
    2. Jinja2 template rendering
    3. Structural: Decision tree or hierarchical prompts
    4. Reasoning: Chain-of-Thought or ReAct
    5. Contextual adaptation
    
    Args:
        row: Input row with prompt and image
        cfg: Configuration object
        is_multimodal: Whether multimodal processing is enabled
        hierarchical_enabled: Whether hierarchical prompts are enabled
        decision_tree_enabled: Whether decision tree prompts are enabled
        
    Returns:
        Preprocessed row with messages and sampling_params
    """
    from ..stages.vqa import render_prompt_template
    from ..prompts.techniques import (
        preprocess_adaptive,
        preprocess_rap,
        preprocess_cot,
        preprocess_react,
        preprocess_contextual
    )
    
    # Check which techniques are enabled
    adaptive_enabled = getattr(getattr(cfg.prompt, "adaptive", None), "enabled", False)
    rap_enabled = getattr(getattr(cfg.prompt, "retrieval_augmented", None), "enabled", False)
    cot_enabled = getattr(getattr(cfg.prompt, "chain_of_thought", None), "enabled", False)
    react_enabled = getattr(getattr(cfg.prompt, "react", None), "enabled", False)
    contextual_enabled = getattr(getattr(cfg.prompt, "contextual", None), "enabled", False)
    
    # Get template config
    use_template = False
    template_str = None
    try:
        if hasattr(cfg.prompt, "user_template") and cfg.prompt.user_template:
            use_template = True
            template_str = cfg.prompt.user_template
    except Exception:
        pass
    
    # Start with original row — image column is passed through as-is.
    # The numpy→PIL conversion is handled lazily by preprocess_simple via
    # _numpy_to_pil(); no eager coercion is needed here.
    current_row = dict(row)
    from ..stages.vqa import _resolve_default_prompt  # local import to avoid cycles

    prompt_raw = row.get("prompt")
    if isinstance(prompt_raw, str):
        prompt = prompt_raw.strip()
    elif prompt_raw is None:
        prompt = ""
    else:
        prompt = str(prompt_raw).strip()
    if not prompt:
        prompt = _resolve_default_prompt(cfg)
    current_row["prompt"] = prompt
    
    # 1. Pre-processing techniques (adaptive, RAP)
    if adaptive_enabled:
        result = preprocess_adaptive(current_row, cfg)
        if result and "messages" in result:
            # Extract adapted prompt from messages
            user_msg = result["messages"][1]
            if isinstance(user_msg.get("content"), list):
                prompt = user_msg["content"][0]["text"]
            else:
                prompt = user_msg.get("content", prompt)
    
    if rap_enabled:
        result = preprocess_rap(current_row, cfg)
        if result and "messages" in result:
            # Extract augmented prompt from messages
            user_msg = result["messages"][1]
            if isinstance(user_msg.get("content"), list):
                prompt = user_msg["content"][0]["text"]
            else:
                prompt = user_msg.get("content", prompt)
    
    # 2. Render Jinja2 template if enabled (after adaptive/RAP)
    if use_template and template_str:
        try:
            # Build context from row data and config, but restrict to lightweight metadata
            context = {
                "prompt": prompt,
                "user_question": prompt,
            }
            excluded_ctx_cols = {"messages", "sampling_params", "image", "image_array", "image_data", "path"}
            for key, value in current_row.items():
                if key in excluded_ctx_cols:
                    continue
                if isinstance(value, (str, int, float, bool, type(None))):
                    context[key] = value
                elif isinstance(value, dict):
                    if all(isinstance(v, (str, int, float, bool, type(None))) for v in value.values()):
                        context[key] = value
                elif isinstance(value, list):
                    if all(isinstance(v, (str, int, float, bool, type(None))) for v in value):
                        context[key] = value
            # Add config variables
            if hasattr(cfg.prompt, "template_vars"):
                context.update(dict(cfg.prompt.template_vars))
            prompt = render_prompt_template(template_str, context)
        except Exception as e:
            import logging
            logging.warning(f"Failed to render template for sample {current_row.get('sample_id')}: {e}")
    
    # 3. Structural techniques (decision tree, hierarchical)
    # These are handled separately in run_vqa_stage, so we skip them here
    if hierarchical_enabled or decision_tree_enabled:
        # Return early - structural techniques handle their own preprocessing
        return None
    
    # 4. Reasoning techniques (CoT, ReAct, chaining)
    # These return complete messages, so we return early
    if cot_enabled:
        result = preprocess_cot(current_row, cfg)
        if result and "messages" in result:
            # CRITICAL: Only spread lightweight metadata, NOT image columns or complex objects
            # Exclude image columns and internal processing columns
            excluded_cols = {"messages", "sampling_params", "image", "image_array", "image_data", "path"}
            lightweight_metadata = {}
            for k, v in current_row.items():
                if k in excluded_cols:
                    continue
                # Only include simple, serializable types
                if isinstance(v, (str, int, float, bool, type(None))):
                    lightweight_metadata[k] = v
                elif isinstance(v, dict):
                    if all(isinstance(vv, (str, int, float, bool, type(None))) for vv in v.values()):
                        lightweight_metadata[k] = v
                elif isinstance(v, list):
                    if all(isinstance(vv, (str, int, float, bool, type(None))) for vv in v):
                        lightweight_metadata[k] = v
            return {
                **lightweight_metadata,
                **result,
            }
    elif react_enabled:
        result = preprocess_react(current_row, cfg)
        if result and "messages" in result:
            # CRITICAL: Only spread lightweight metadata, NOT image columns or complex objects
            # Exclude image columns and internal processing columns
            excluded_cols = {"messages", "sampling_params", "image", "image_array", "image_data", "path"}
            lightweight_metadata = {}
            for k, v in current_row.items():
                if k in excluded_cols:
                    continue
                # Only include simple, serializable types
                if isinstance(v, (str, int, float, bool, type(None))):
                    lightweight_metadata[k] = v
                elif isinstance(v, dict):
                    if all(isinstance(vv, (str, int, float, bool, type(None))) for vv in v.values()):
                        lightweight_metadata[k] = v
                elif isinstance(v, list):
                    if all(isinstance(vv, (str, int, float, bool, type(None))) for vv in v):
                        lightweight_metadata[k] = v
            return {
                **lightweight_metadata,
                **result,
            }
    
    # 5. Contextual adaptation
    if contextual_enabled:
        result = preprocess_contextual(current_row, cfg)
        if result and "messages" in result:
            # Extract adapted prompt from messages
            user_msg = result["messages"][1]
            if isinstance(user_msg.get("content"), list):
                prompt = user_msg["content"][0]["text"]
            else:
                prompt = user_msg.get("content", prompt)
    
    # Propagate accumulated prompt changes (from template rendering, adaptive,
    # RAP, contextual, etc.) back into current_row so that preprocess_simple
    # picks up the correct value.
    current_row["prompt"] = prompt

    # Fallback to simple prompt if no techniques enabled or no messages generated
    # Build standard messages with updated prompt
    # CRITICAL: Do NOT include image column in any data structures
    # Pass original row to preprocess_simple so it can read image, but result won't include it
    # We must NOT touch the dataset created by ray.data.read_images()
    return preprocess_simple(current_row, cfg, is_multimodal)


def unified_postprocess(
    row: Dict[str, Any],
    cfg: DictConfig,
    hierarchical_enabled: bool = False,
    decision_tree_enabled: bool = False
) -> Dict[str, Any]:
    """Unified postprocessing that handles all techniques.
    
    Args:
        row: Row with generated_text from model
        cfg: Configuration object
        hierarchical_enabled: Whether hierarchical prompts are enabled
        decision_tree_enabled: Whether decision tree prompts are enabled
        
    Returns:
        Postprocessed row with answer and metadata
    """
    from datetime import datetime
    
    result = {
        "sample_id": row.get("sample_id"),
        "answer": row.get("generated_text", ""),
    }
    
    # Handle self-consistency aggregation if enabled
    self_consistency_enabled = getattr(getattr(cfg.prompt, "self_consistency", None), "enabled", False)
    if self_consistency_enabled:
        # Placeholder: Would implement aggregation logic here
        # For now, just add consistency_sample_id if present
        if "_consistency_sample_id" in row:
            result["_consistency_sample_id"] = row["_consistency_sample_id"]
    
    # Handle hierarchical/decision tree outputs
    if hierarchical_enabled or decision_tree_enabled:
        # Extract all intermediate outputs from metadata
        metadata = row.get("metadata", {})
        if isinstance(metadata, dict):
            # Add all metadata fields except internal tracking fields
            for k, v in metadata.items():
                if k not in ["tree_id", "version", "nodes_visited", "depth"]:
                    result[k] = v
    
    # Add timing information if available
    if "ts_start" in row:
        ts_end = datetime.now().timestamp()
        result["metadata"] = {
            **result.get("metadata", {}),
            "processing_time": ts_end - row["ts_start"]
        }
    
    # Add complexity if available (from adaptive prompting)
    if "_complexity" in row:
        result["metadata"] = {
            **result.get("metadata", {}),
            "complexity": row["_complexity"]
        }
    
    return result
