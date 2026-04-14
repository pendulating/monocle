"""Persistent vLLM Processor for efficient GEPA optimization.

This module provides a persistent vLLM processor that eliminates model teardown
between GEPA iteration cycles. The model is loaded once and reused for all
evaluations, significantly reducing overhead.

Key Design Principles:
- Model loading happens once at initialization using vllm.LLM directly
- The LLM instance persists in memory across evaluate() calls
- Prompts/configs can be updated between evaluations
- Thread-safe context management for parallel evaluations

Uses vllm.LLM directly so the model stays in GPU memory across evaluations.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.urbanvqa.stages.vqa import (
    _build_guided_decoding_config,
    _ensure_json_schema_dict,
    MODEL_ZOO_BASE,
    _filter_vllm_engine_kwargs,
)

LOG = logging.getLogger(__name__)

# Thread-local storage for evaluation context
_thread_local = threading.local()

# Global processor cache
# Note: We use a simple dict for the cache. Thread safety is handled
# at the method level using instance locks that are recreated after deserialization.
_PROCESSOR_CACHE: Dict[str, "PersistentVLLMProcessor"] = {}

def _get_cache_lock() -> threading.Lock:
    """Get or create the global cache lock.
    
    We use a function to lazily create the lock, avoiding serialization issues
    when the module is imported before job submission.
    """
    global _CACHE_LOCK_INSTANCE
    if "_CACHE_LOCK_INSTANCE" not in globals() or _CACHE_LOCK_INSTANCE is None:
        globals()["_CACHE_LOCK_INSTANCE"] = threading.Lock()
    return _CACHE_LOCK_INSTANCE

_CACHE_LOCK_INSTANCE: Optional[threading.Lock] = None


@dataclass
class EvaluationContext:
    """Context for a single evaluation batch.
    
    This is stored in thread-local storage so that preprocess functions
    can access the current prompt configuration.
    """
    system_prompt: str = ""
    user_template: str = ""
    cfg: Optional[DictConfig] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def get_current_context() -> Optional[EvaluationContext]:
    """Get the current evaluation context for this thread."""
    return getattr(_thread_local, "context", None)


def set_current_context(ctx: EvaluationContext) -> None:
    """Set the current evaluation context for this thread."""
    _thread_local.context = ctx


@contextmanager
def evaluation_context(
    system_prompt: str = "",
    user_template: str = "",
    cfg: Optional[DictConfig] = None,
    **extra: Any,
):
    """Context manager for setting evaluation context.
    
    Usage:
        with evaluation_context(system_prompt="...", user_template="..."):
            results = processor.evaluate(df)
    """
    old_ctx = get_current_context()
    new_ctx = EvaluationContext(
        system_prompt=system_prompt,
        user_template=user_template,
        cfg=cfg,
        extra=extra,
    )
    set_current_context(new_ctx)
    try:
        yield new_ctx
    finally:
        set_current_context(old_ctx)


class PersistentVLLMProcessor:
    """Persistent vLLM processor that caches the model across evaluations.
    
    Uses vllm.LLM directly so the model stays loaded in GPU memory and
    subsequent evaluate() calls reuse the same engine.
    
    Usage:
        # Create once at the start of GEPA optimization
        processor = PersistentVLLMProcessor(cfg)
        processor.initialize()
        
        # For each iteration, just call evaluate with new prompts
        results = processor.evaluate(batch_df, user_template=candidate_prompt)
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
    ):
        """Initialize the persistent processor.
        
        Args:
            cfg: The base configuration (used for model settings)
            preprocess_fn: Optional custom preprocess function
            postprocess_fn: Optional custom postprocess function
        """
        self._base_cfg = deepcopy(cfg)
        self._custom_preprocess = preprocess_fn
        self._custom_postprocess = postprocess_fn
        self._llm = None  # vllm.LLM instance
        self._sampling_params = None
        self._initialized = False
        self._is_multimodal = False  # Set during initialize()
        # Lock is created lazily to avoid serialization issues
        self._lock: Optional[threading.Lock] = None
        
        # Cache key for identifying this processor configuration
        self._cache_key = self._compute_cache_key(cfg)
        # Structured output / guided decoding payload derived from config
        self._guided_decoding_payload = self._derive_guided_decoding_payload(cfg)
    
    def _get_lock(self) -> threading.Lock:
        """Get or create the instance lock (lazy initialization for serialization safety)."""
        if self._lock is None:
            self._lock = threading.Lock()
        return self._lock
    
    def __getstate__(self) -> Dict[str, Any]:
        """Prepare state for pickling - exclude non-serializable objects."""
        state = self.__dict__.copy()
        # Remove the lock - it will be recreated on __setstate__
        state["_lock"] = None
        # Remove the LLM instance - it cannot be pickled and must be recreated
        state["_llm"] = None
        state["_sampling_params"] = None
        state["_initialized"] = False
        # Keep _is_multimodal - it will be recalculated on initialize() anyway
        return state
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)
        # Lock will be recreated lazily via _get_lock()
        self._lock = None
        # LLM will be recreated via initialize()
        self._llm = None
        self._sampling_params = None
        self._initialized = False
        # _is_multimodal will be set correctly when initialize() is called
    
    @staticmethod
    def _compute_cache_key(cfg: DictConfig) -> str:
        """Compute a cache key based on model configuration."""
        model_cfg = getattr(cfg, "model", None)
        if model_cfg is None:
            return "default"
        
        # Key components that affect model loading
        key_parts = [
            str(getattr(model_cfg, "model_source", "")),
            str(getattr(model_cfg, "tensor_parallel_size", 1)),
            str(getattr(model_cfg, "concurrency", 1)),
        ]
        return ":".join(key_parts)
    
    @classmethod
    def get_or_create(
        cls,
        cfg: DictConfig,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
    ) -> "PersistentVLLMProcessor":
        """Get an existing processor or create a new one.
        
        This method provides a global cache of processors to enable
        sharing across multiple adapter instances.
        """
        key = cls._compute_cache_key(cfg)
        
        with _get_cache_lock():
            if key in _PROCESSOR_CACHE:
                cached = _PROCESSOR_CACHE[key]
                # Only reuse if actually initialized (LLM is loaded)
                if cached._initialized and cached._llm is not None:
                    LOG.info(f"Reusing cached PersistentVLLMProcessor (key={key})")
                    return cached
            
            LOG.info(f"Creating new PersistentVLLMProcessor (key={key})")
            processor = cls(cfg, preprocess_fn, postprocess_fn)
            _PROCESSOR_CACHE[key] = processor
            return processor
    
    def initialize(self) -> None:
        """Initialize the processor and load the model.
        
        This is a heavy operation that loads the vLLM model into GPU memory.
        Call this once at the start of optimization. The model will remain
        loaded for all subsequent evaluate() calls.
        """
        if self._initialized and self._llm is not None:
            LOG.debug("PersistentVLLMProcessor already initialized")
            return
        
        with self._get_lock():
            if self._initialized and self._llm is not None:
                return
            
            LOG.info("Initializing PersistentVLLMProcessor - loading model with vllm.LLM...")
            
            try:
                from vllm import LLM, SamplingParams
            except ImportError as e:
                raise RuntimeError("vLLM is required for PersistentVLLMProcessor") from e
            
            # Build engine configuration
            model_source, engine_kwargs = self._build_engine_config()
            
            # Create the persistent LLM instance - this loads the model once
            LOG.info(f"Loading model: {model_source}")
            self._llm = LLM(
                model=model_source,
                **engine_kwargs,
            )
            
            # Build default sampling params
            self._sampling_params = self._build_sampling_params()
            
            self._initialized = True
            LOG.info("PersistentVLLMProcessor initialized successfully - model is loaded and will persist")
    
    @staticmethod
    def _detect_multimodal_from_name(model_source: str) -> bool:
        """Detect if model is multimodal based on name patterns.
        
        Common patterns for vision-language models:
        - VL, VLM (Qwen-VL, InternVL)
        - vision (Phi-3-vision, LLaVA)
        - multimodal
        """
        if not model_source:
            return False
        
        name_lower = model_source.lower()
        patterns = [
            "-vl-", "-vl/", "vl-", "/vl-",  # Qwen-VL, InternVL
            "vision",  # Phi-3-vision
            "llava",   # LLaVA family
            "multimodal",
            "vlm",
            "paligemma",
            "idefics",
            "fuyu",
            "cogvlm",
        ]
        return any(pattern in name_lower for pattern in patterns)

    def _derive_guided_decoding_payload(self, cfg: DictConfig) -> Optional[Dict[str, Any]]:
        """Build guided_decoding payload from prompt.structured_output config."""
        try:
            prompt_cfg = getattr(cfg, "prompt", None)
            structured_cfg = getattr(prompt_cfg, "structured_output", None)
            if not structured_cfg or not getattr(structured_cfg, "enabled", False):
                return None

            schema = None
            schema_path = getattr(structured_cfg, "schema_path", None)
            if schema_path:
                import importlib.util

                spec = importlib.util.spec_from_file_location("schema_module", schema_path)
                if spec and spec.loader:
                    schema_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(schema_module)
                    if hasattr(schema_module, "VQAAnswer"):
                        schema = schema_module.VQAAnswer.model_json_schema()
            else:
                schema = getattr(structured_cfg, "json_schema", None)

            schema = _ensure_json_schema_dict(schema)
            if not schema:
                return None

            return _build_guided_decoding_config(schema)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning(f"Failed to derive guided decoding payload: {exc}")
            return None
    
    def _build_engine_config(self) -> Tuple[str, Dict[str, Any]]:
        """Build the vLLM engine configuration from the base config.
        
        Returns:
            Tuple of (model_source, engine_kwargs)
        """
        import os
        
        cfg = self._base_cfg
        model_cfg = getattr(cfg, "model", OmegaConf.create({}))
        
        # Resolve model source
        model_source = getattr(model_cfg, "model_source", None)
        if model_source is None:
            raise ValueError("model.model_source is required")
        
        resolved_model_source = model_source
        if not model_source.startswith("/") and "/" in model_source:
            # Try local zoo path
            local_path = os.path.join(MODEL_ZOO_BASE, model_source.split("/")[-1])
            if os.path.exists(local_path):
                resolved_model_source = local_path
        
        # Engine kwargs - filter to only include supported args for current vLLM version
        engine_kwargs = dict(OmegaConf.to_container(
            getattr(model_cfg, "engine_kwargs", OmegaConf.create({})),
            resolve=True
        ) or {})
        
        # Apply defaults
        engine_kwargs.setdefault("trust_remote_code", True)
        engine_kwargs.setdefault("enforce_eager", True)
        engine_kwargs.setdefault("enable_prefix_caching", True)
        engine_kwargs.setdefault("enable_chunked_prefill", True)
        
        # Tensor parallelism
        tp_size = getattr(model_cfg, "tensor_parallel_size", 1)
        if tp_size > 1:
            engine_kwargs["tensor_parallel_size"] = tp_size
        
        # Max model length if specified
        max_model_len = getattr(model_cfg, "max_model_len", None)
        if max_model_len:
            engine_kwargs["max_model_len"] = max_model_len
        
        # GPU memory utilization
        gpu_mem_util = getattr(model_cfg, "gpu_memory_utilization", None)
        if gpu_mem_util:
            engine_kwargs["gpu_memory_utilization"] = gpu_mem_util
        
        # Filter out unsupported kwargs for current vLLM version
        engine_kwargs = _filter_vllm_engine_kwargs(engine_kwargs)
        
        # For vision-language models, we may need limit_mm_per_prompt
        # Check multiple sources for multimodal detection
        is_multimodal = (
            getattr(model_cfg, "has_image", False) or
            getattr(getattr(cfg, "runtime", None), "multimodal_enabled", False) or
            self._detect_multimodal_from_name(resolved_model_source)
        )
        self._is_multimodal = is_multimodal  # Store for later use
        
        if is_multimodal:
            engine_kwargs.setdefault("limit_mm_per_prompt", {"image": 1})
            LOG.info(f"Multimodal mode enabled for model: {resolved_model_source}")

        # Enable guided decoding backend when structured outputs are configured
        if self._guided_decoding_payload and "guided_decoding_backend" not in engine_kwargs:
            engine_kwargs["guided_decoding_backend"] = "auto"
        
        return resolved_model_source, engine_kwargs
    
    def _build_sampling_params(self):
        """Build default sampling parameters from config."""
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        
        cfg = self._base_cfg
        sampling_cfg = getattr(cfg, "sampling_params_vqa", None) or {}
        if hasattr(sampling_cfg, "items"):
            sampling_cfg = dict(sampling_cfg)
        
        # Extract supported params
        params = {}
        for key in ["temperature", "top_p", "top_k", "max_tokens", "stop", "seed"]:
            if key in sampling_cfg:
                params[key] = sampling_cfg[key]
        
        # Defaults - use small max_tokens to force concise binary answers
        params.setdefault("max_tokens", 4)
        params.setdefault("temperature", 0.0)

        # Attach structured output payload for strict outputs (e.g., Yes/No enum).
        # NOTE: vLLM expects a StructuredOutputsParams object, not a raw dict.
        if self._guided_decoding_payload:
            params["structured_outputs"] = StructuredOutputsParams(
                **deepcopy(self._guided_decoding_payload)
            )
        
        return SamplingParams(**params)
    
    def evaluate(
        self,
        df: pd.DataFrame,
        system_prompt: str = "",
        user_template: str = "",
        cfg: Optional[DictConfig] = None,
    ) -> pd.DataFrame:
        """Evaluate a batch using the persistent vLLM engine.
        
        This method uses llm.chat() to process all samples in the batch.
        The model remains loaded in GPU memory between calls.
        
        Args:
            df: Input DataFrame with samples to evaluate
            system_prompt: System prompt to use (overrides config)
            user_template: User template to use (overrides config)
            cfg: Optional config override
            
        Returns:
            DataFrame with evaluation results
        """
        if not self._initialized or self._llm is None:
            self.initialize()
        
        eval_cfg = cfg or self._base_cfg
        
        # Prepare all conversations for batch inference
        conversations = []
        row_metadata = []
        
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            conv, metadata = self._prepare_conversation(row_dict, system_prompt, user_template, eval_cfg)
            conversations.append(conv)
            row_metadata.append(metadata)
        
        # Run batch inference with the persistent LLM
        LOG.info(f"Running batch inference on {len(conversations)} samples...")
        outputs = self._llm.chat(conversations, self._sampling_params, use_tqdm=False)
        
        # Clean up to prevent memory accumulation over many iterations
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        # Process outputs
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text if output.outputs else ""
            result = self._postprocess_output(generated_text, row_metadata[i])
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _prepare_conversation(
        self,
        row: Dict[str, Any],
        system_prompt: str,
        user_template: str,
        cfg: DictConfig,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Prepare a conversation for vLLM chat inference.
        
        Returns:
            Tuple of (conversation_messages, metadata_dict)
        """
        from dagspaces.urbanvqa.stages.vqa import (
            _sanitize_prompt_value,
            _resolve_row_sample_id,
        )
        
        row_values = dict(row)
        
        # Override prompt with user template if provided
        if user_template:
            prompt = user_template
        else:
            prompt = _sanitize_prompt_value(row_values.get("prompt"), cfg)
        
        resolved_sample_id = _resolve_row_sample_id(row_values)
        if resolved_sample_id is not None:
            row_values["sample_id"] = resolved_sample_id
        
        # Get system prompt
        sys_prompt = system_prompt if system_prompt else getattr(cfg.prompt, "system", "You are a helpful assistant.")
        
        # Use stored multimodal flag (set during initialization)
        is_multimodal = getattr(self, "_is_multimodal", False)
        
        # Build conversation
        messages = [{"role": "system", "content": sys_prompt}]
        
        if is_multimodal and "image" in row_values and row_values["image"] is not None:
            # Build multimodal user message with image
            image = row_values["image"]
            # Convert to PIL if needed
            pil_image = self._ensure_pil_image(image, row_values)
            
            if pil_image is not None:
                # Use vLLM's multimodal content format
                user_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_pil", "image_pil": pil_image},
                ]
            else:
                user_content = prompt
            
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        # Collect metadata for postprocessing
        metadata = {}
        excluded_cols = {
            "image", "image_array", "image_data", "path",
            "messages", "sampling_params",
            "llm_output", "generated_text",
        }
        for key, value in row_values.items():
            if key in excluded_cols:
                continue
            if isinstance(value, (str, int, float, bool, type(None))):
                metadata[key] = value
        
        return messages, metadata
    
    def _ensure_pil_image(self, image: Any, row: Dict[str, Any]) -> Optional[Any]:
        """Ensure image is a PIL Image object."""
        try:
            from PIL import Image
            import io
            import base64
            
            if isinstance(image, Image.Image):
                return image
            
            # Try to load from bytes
            if isinstance(image, bytes):
                return Image.open(io.BytesIO(image))
            
            # Try to load from base64 string
            if isinstance(image, str):
                if image.startswith("data:image"):
                    # Data URL format
                    header, data = image.split(",", 1)
                    image_bytes = base64.b64decode(data)
                    return Image.open(io.BytesIO(image_bytes))
                elif len(image) > 200 and not image.startswith("/"):
                    # Likely base64 encoded
                    try:
                        image_bytes = base64.b64decode(image)
                        return Image.open(io.BytesIO(image_bytes))
                    except Exception:
                        pass
                else:
                    # Might be a file path
                    import os
                    if os.path.exists(image):
                        return Image.open(image)
            
            # Try numpy array
            if hasattr(image, "shape") and hasattr(image, "dtype"):
                import numpy as np
                if isinstance(image, np.ndarray):
                    return Image.fromarray(image)
            
            LOG.warning(f"Could not convert image to PIL format: {type(image)}")
            return None
            
        except Exception as e:
            LOG.warning(f"Error converting image to PIL: {e}")
            return None
    
    def _postprocess_output(self, generated_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess a single output from the LLM."""
        import json
        import re
        from datetime import datetime
        
        ts_end = datetime.utcnow().isoformat()
        
        # Try to parse JSON if the output looks like JSON
        answer = generated_text
        if generated_text and generated_text.strip().startswith("{"):
            try:
                parsed = json.loads(generated_text)
                answer = parsed.get("answer", generated_text)
            except json.JSONDecodeError:
                pass
        elif generated_text:
            # Try to find JSON in the text
            json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"([^"]*)"[^{}]*\}', generated_text)
            if json_match:
                answer = json_match.group(1)
        
        # Build result from metadata
        result = dict(metadata)
        
        # Add final answer and model response
        result["answer"] = answer.strip() if isinstance(answer, str) else answer
        result["model_response"] = generated_text
        result["metadata"] = {"ts_end": ts_end}
        
        return result
    
    def shutdown(self) -> None:
        """Shutdown the processor and release GPU memory.
        
        Call this when optimization is complete to free GPU memory.
        This performs aggressive cleanup to ensure Slurm jobs can terminate.
        """
        with self._get_lock():
            if self._llm is not None:
                LOG.info("Shutting down PersistentVLLMProcessor - releasing GPU memory")
                
                # Try to explicitly shutdown the vLLM engine if it has a shutdown method
                try:
                    if hasattr(self._llm, "llm_engine") and self._llm.llm_engine is not None:
                        engine = self._llm.llm_engine
                        # Some vLLM versions have shutdown methods on the engine
                        if hasattr(engine, "shutdown"):
                            engine.shutdown()
                        elif hasattr(engine, "_shutdown"):
                            engine._shutdown()
                except Exception as e:
                    LOG.debug(f"Error during explicit engine shutdown: {e}")
                
                # Delete the LLM instance
                del self._llm
                self._llm = None
                self._sampling_params = None
                self._initialized = False
                
                # Force garbage collection to ensure cleanup
                import gc
                gc.collect()
                
                # Clear CUDA cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError:
                    pass
                except Exception as e:
                    LOG.debug(f"Error during CUDA cleanup: {e}")
                
                # Remove from global cache
                with _get_cache_lock():
                    if self._cache_key in _PROCESSOR_CACHE:
                        del _PROCESSOR_CACHE[self._cache_key]
                
                LOG.info("PersistentVLLMProcessor shutdown complete")


def clear_processor_cache() -> None:
    """Clear all cached processors and release GPU memory.

    Call this to release all GPU memory held by cached processors
    and allow Slurm jobs to terminate cleanly.
    """
    global _PROCESSOR_CACHE
    with _get_cache_lock():
        for processor in list(_PROCESSOR_CACHE.values()):
            try:
                processor.shutdown()
            except Exception as e:
                LOG.warning(f"Error shutting down processor: {e}")
        _PROCESSOR_CACHE.clear()

    # Final garbage collection
    import gc
    gc.collect()

    LOG.info("Cleared all cached PersistentVLLMProcessors")
