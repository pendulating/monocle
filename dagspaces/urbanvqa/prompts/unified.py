"""Unified dynamic prompting framework for VQA.

This module provides a unified preprocessing/postprocessing framework that
integrates all dynamic prompting techniques in priority order.
"""

from typing import Dict, Any, Optional
from omegaconf import DictConfig
import base64
from io import BytesIO

# Import numpy for array checking
try:
    import numpy as np
except ImportError:
    np = None

# Import PIL for base64 conversion
try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def preprocess_simple(row: Dict[str, Any], cfg: DictConfig, is_multimodal: bool = True) -> Dict[str, Any]:
    """Simple preprocessing without any dynamic techniques.
    
    Simplified approach: Only handles numpy arrays from ray.data.read_images().
    Converts numpy arrays to base64 strings for vLLM inference.
    
    Args:
        row: Input row with prompt and image (numpy array)
        cfg: Configuration object
        is_multimodal: Whether multimodal processing is enabled
        
    Returns:
        Preprocessed row with messages and sampling_params
    """
    system_prompt = getattr(cfg.prompt, "system", "You are a helpful assistant.")
    sampling_params = getattr(cfg, "sampling_params_vqa", {})
    
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Prompt is required for VQA")
    
    # Convert numpy array to base64 for vLLM
    # ray.data.read_images() already provides numpy arrays in consistent format - no checks needed
    image_base64_str = None
    
    if is_multimodal:
        # CRITICAL: Read-only access - never modify row["image"]
        # ray.data.read_images() guarantees numpy array in consistent format
        from ..stages.vqa import _convert_image_to_base64
        image_base64_str = _convert_image_to_base64(row.get("image"), row)
    
    # Build messages with base64 string (PyArrow-serializable)
    if is_multimodal and image_base64_str is not None:
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_base64_str}}
        ]
    else:
        # No image available - text-only message
        user_content = prompt
    
    # Return ONLY messages and sampling_params (plus lightweight metadata)
    # Ray Data LLM processor expects ONLY these fields
    result = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "sampling_params": sampling_params,
    }
    
    # Include lightweight, serializable metadata (strings/numbers only)
    # CRITICAL: NEVER include image column - we must NOT touch images from ray.data.read_images()
    for key in ["sample_id", "prompt"]:
        if key in row and isinstance(row[key], (str, int, float, type(None))):
            result[key] = row[key]
    
    # Explicitly ensure image column is NOT in result
    result.pop("image", None)
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
        if hasattr(cfg.prompt, "template") and cfg.prompt.template:
            use_template = True
            template_str = cfg.prompt.template
    except Exception:
        pass
    
    # Start with original row and normalize image metadata for local use only
    current_row = dict(row)
    if "image" in current_row and current_row["image"] is not None:
        try:
            from ..stages.vqa import _ensure_numpy_image_value
            current_row["image"] = _ensure_numpy_image_value(current_row["image"], current_row.get("sample_id"))
        except Exception:
            # If coercion fails, drop local image reference to avoid leaking non-serializable objects.
            current_row.pop("image", None)
    prompt = str(row.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Prompt is required for VQA")
    
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
