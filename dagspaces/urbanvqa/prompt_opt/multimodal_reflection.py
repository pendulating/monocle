"""Multimodal reflection LM wrapper for GEPA.

This module provides a wrapper that intercepts GEPA's text-only reflection
prompts, extracts embedded image markers, and calls a vision-language model
with properly formatted multimodal messages.

Key Features:
- Parses [IMAGE:...] markers from GEPA's text prompts
- Supports local files, URLs, and base64-encoded images
- Converts to OpenAI-compatible multimodal message format
- Falls back to text-only if image loading fails
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)

# Pattern to match image markers: [IMAGE:...] 
# Matches: data URLs, http(s) URLs, and file:// URIs
IMAGE_MARKER_PATTERN = re.compile(
    r'\[IMAGE:((?:data:image/[^;]+;base64,[A-Za-z0-9+/=]+)|(?:https?://[^\]]+)|(?:file://[^\]]+))\]'
)


def _load_image_as_base64(image_ref: str) -> Optional[str]:
    """Load an image reference and return as base64 data URL or URL.
    
    Args:
        image_ref: Image reference - can be:
            - data:image/... (base64 data URL)
            - http:// or https:// (remote URL)
            - file:///path/to/image (local file)
            
    Returns:
        Base64 data URL for local files, or the URL for remote images.
        None if loading fails.
    """
    if image_ref.startswith("data:image"):
        # Already base64 data URL
        return image_ref
    
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        # Remote URL - return as-is for LiteLLM to handle
        return image_ref
    
    if image_ref.startswith("file://"):
        file_path = image_ref[7:]  # Remove file:// prefix
    else:
        file_path = image_ref
    
    # Load local file and encode as base64
    path = Path(file_path)
    if not path.exists():
        LOG.warning(f"Image file not found: {file_path}")
        return None
    
    try:
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            mime_type = "image/jpeg"  # Default assumption
        
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        
        return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        LOG.warning(f"Failed to load image {file_path}: {e}")
        return None


def _parse_multimodal_prompt(prompt: str) -> Tuple[str, List[str]]:
    """Parse a text prompt and extract image markers.
    
    Args:
        prompt: Text prompt potentially containing [IMAGE:...] markers
        
    Returns:
        Tuple of (text_with_placeholders, list_of_image_refs)
        The text has markers replaced with "[See image above]"
    """
    image_refs = []
    
    def replace_marker(match):
        image_ref = match.group(1)
        image_refs.append(image_ref)
        return "[See attached image]"
    
    text_cleaned = IMAGE_MARKER_PATTERN.sub(replace_marker, prompt)
    return text_cleaned, image_refs


def _build_multimodal_messages(
    prompt: str, 
    image_refs: List[str],
    max_images: int = 10,
) -> List[Dict[str, Any]]:
    """Build OpenAI-compatible multimodal messages.
    
    Args:
        prompt: Text prompt (with markers already replaced)
        image_refs: List of image references to include
        max_images: Maximum number of images to include
        
    Returns:
        List of message dicts in OpenAI multimodal format
    """
    content: List[Dict[str, Any]] = []
    
    # Add images first (limited to prevent context overflow)
    images_added = 0
    for ref in image_refs[:max_images]:
        image_data = _load_image_as_base64(ref)
        if image_data:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_data}
            })
            images_added += 1
    
    if images_added > 0:
        LOG.debug(f"Added {images_added} images to multimodal message")
    
    # Add text prompt after images
    content.append({"type": "text", "text": prompt})
    
    return [{"role": "user", "content": content}]


def create_multimodal_reflection_lm(
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    max_images_per_call: int = 5,
    timeout: Optional[float] = None,
) -> Callable[[str], str]:
    """Create a multimodal-aware reflection LM callable for GEPA.
    
    This wraps LiteLLM completion to handle image markers embedded in
    GEPA's text prompts, converting them to proper multimodal format.
    
    Args:
        model: Model name (e.g., "openai/qwen3-vl-30b" for vLLM)
        api_base: API base URL (e.g., "http://localhost:8000/v1")
        api_key: API key (or None/"EMPTY" for local servers)
        temperature: Sampling temperature (0.0 for deterministic)
        seed: Random seed for reproducibility
        max_images_per_call: Maximum images to include per reflection call
        timeout: Request timeout in seconds
        
    Returns:
        Callable that takes a prompt string and returns response string.
        Compatible with GEPA's reflection_lm interface.
    """
    import litellm
    
    def multimodal_reflection_lm(prompt: str) -> str:
        """Process a GEPA reflection prompt with embedded images.
        
        This function:
        1. Parses the text prompt for [IMAGE:...] markers
        2. Loads images and builds multimodal messages
        3. Calls the VLM via LiteLLM
        4. Returns the text response
        """
        # Parse out image markers
        text_prompt, image_refs = _parse_multimodal_prompt(prompt)
        
        if image_refs:
            # Build multimodal messages
            messages = _build_multimodal_messages(
                text_prompt, 
                image_refs,
                max_images=max_images_per_call,
            )
            LOG.info(f"Multimodal reflection with {len(image_refs)} image markers "
                     f"({min(len(image_refs), max_images_per_call)} included)")
        else:
            # No images - standard text message
            messages = [{"role": "user", "content": prompt}]
        
        # Build LiteLLM kwargs
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if api_base:
            kwargs["api_base"] = api_base
        if api_key and api_key != "EMPTY":
            kwargs["api_key"] = api_key
        if seed is not None:
            kwargs["seed"] = seed
        if timeout is not None:
            kwargs["timeout"] = timeout
        
        try:
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            LOG.error(f"Multimodal reflection failed: {e}")
            # Fallback: try without images if we had images
            if image_refs:
                LOG.warning("Retrying reflection without images...")
                kwargs["messages"] = [{"role": "user", "content": text_prompt}]
                try:
                    response = litellm.completion(**kwargs)
                    return response.choices[0].message.content or ""
                except Exception as e2:
                    LOG.error(f"Text-only fallback also failed: {e2}")
                    raise
            raise
    
    return multimodal_reflection_lm


def select_images_for_reflection(
    traces_with_scores: List[Tuple[Dict[str, Any], float]],
    max_images: int = 5,
    incorrect_ratio: float = 0.7,
    score_threshold: float = 1.0,
) -> List[Dict[str, Any]]:
    """Select traces for image inclusion with bias toward incorrect predictions.
    
    This implements smart image selection for reflection: we want the reflector
    to see more examples of what went wrong (incorrect predictions) while still
    having some correct examples for contrast.
    
    Args:
        traces_with_scores: List of (trace_dict, score) tuples
        max_images: Maximum number of images to include
        incorrect_ratio: Ratio of incorrect samples (default 0.7 = 70% incorrect)
        score_threshold: Score below which a sample is considered "incorrect"
        
    Returns:
        List of trace dicts selected for image inclusion, respecting the ratio
    """
    if not traces_with_scores or max_images <= 0:
        return []
    
    # Separate into correct and incorrect
    incorrect = [(t, s) for t, s in traces_with_scores if s < score_threshold and t.get("image_ref")]
    correct = [(t, s) for t, s in traces_with_scores if s >= score_threshold and t.get("image_ref")]
    
    # Calculate target counts
    num_incorrect = min(int(max_images * incorrect_ratio), len(incorrect))
    num_correct = min(max_images - num_incorrect, len(correct))
    
    # If we don't have enough incorrect, fill with correct (and vice versa)
    if num_incorrect < int(max_images * incorrect_ratio) and len(correct) > num_correct:
        num_correct = min(max_images - num_incorrect, len(correct))
    if num_correct < max_images - num_incorrect and len(incorrect) > num_incorrect:
        num_incorrect = min(max_images - num_correct, len(incorrect))
    
    # Select traces (prioritize lowest scores for incorrect, highest for correct)
    incorrect_sorted = sorted(incorrect, key=lambda x: x[1])  # Lowest scores first
    correct_sorted = sorted(correct, key=lambda x: -x[1])  # Highest scores first
    
    selected = []
    selected.extend([t for t, _ in incorrect_sorted[:num_incorrect]])
    selected.extend([t for t, _ in correct_sorted[:num_correct]])
    
    LOG.debug(f"Selected {len(selected)} images for reflection: "
              f"{num_incorrect} incorrect, {num_correct} correct")
    
    return selected








