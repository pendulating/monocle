"""VQA (Visual Question Answering) stage for UrbanVQA pipeline.

This module provides simplified VQA functionality:
- Prompt + image → answer flow
- Jinja2 template support
- Structured JSON output support
- Batch inference with Ray Data
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
import pandas as pd
import json
import os
import base64
import logging
import re
from io import BytesIO
from omegaconf import DictConfig

# Import numpy for array checking
try:
    import numpy as np
except ImportError:
    np = None

# Import PIL for base64 conversion (temporary use only)
try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Import unified framework
from ..prompts.unified import preprocess_simple

try:
    import ray  # noqa: F401
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig  # type: ignore
    _RAY_OK = True
except Exception:
    _RAY_OK = False

# Jinja2 template support
try:
    from jinja2 import Environment, StrictUndefined
    _JINJA2_AVAILABLE = True
except Exception:
    _JINJA2_AVAILABLE = False

# Model zoo base path
MODEL_ZOO_BASE = "/share/pierson/matt/zoo/models"

# vLLM logging state
_VLLM_LOGS_SILENCED = False


def _ensure_numpy_image_value(image: Any, sample_id: Optional[str] = None) -> "np.ndarray":
    """Coerce image-like input to a NumPy array for Ray/vLLM compatibility."""
    if np is None:
        raise RuntimeError("NumPy is required for VQA image preprocessing but is not available.")
    if image is None:
        raise ValueError(
            "Image data is required for multimodal processing"
            + (f" (sample_id: {sample_id})" if sample_id else "")
        )
    if isinstance(image, np.ndarray):
        return image
    # Handle PyArrow tensors without importing at module import time
    try:
        import pyarrow as pa  # type: ignore
    except Exception:
        pa = None  # type: ignore
    if pa is not None and isinstance(image, pa.Tensor):
        return image.to_numpy()
    if isinstance(image, str):
        raise ValueError(
            "Expected in-memory image data for multimodal processing, but received a string path."
            + (f" (sample_id: {sample_id})" if sample_id else "")
        )
    if _PIL_AVAILABLE and isinstance(image, PILImage):
        return np.asarray(image.convert("RGB"))
    try:
        coerced = np.asarray(image)
    except Exception as exc:
        raise ValueError(
            f"Unsupported image type '{type(image).__name__}' for multimodal processing"
            + (f" (sample_id: {sample_id})" if sample_id else "")
        ) from exc
    if not isinstance(coerced, np.ndarray):
        raise ValueError(
            f"Failed to coerce image of type '{type(image).__name__}' to NumPy array"
            + (f" (sample_id: {sample_id})" if sample_id else "")
        )
    return coerced


# Helper functions for model path resolution and GPU detection
def _read_int_file(path: str) -> int:
    """Read integer from file, return -1 on error."""
    try:
        with open(path, "r") as f:
            s = f.read().strip()
        if s.lower() == "max":
            return -1
        return int(s)
    except Exception:
        return -1


def _parse_int(val: str) -> int:
    """Parse integer from string, return -1 on error."""
    try:
        return int(val)
    except Exception:
        return -1


def _parse_cpus_on_node(val: str) -> int:
    """Parse SLURM_CPUS_ON_NODE value which can be in various formats."""
    if not isinstance(val, str):
        return -1
    try:
        # Common forms: "32", "16(x2)", "2,2", "2,1"
        v = val.strip()
        if "(x" in v and v.endswith(")"):
            m = re.match(r"^(\d+)\(x(\d+)\)$", v)
            if m:
                a = int(m.group(1))
                b = int(m.group(2))
                return max(1, a * b)
        if "," in v:
            parts = [p for p in v.split(",") if p.strip()]
            acc = 0
            for p in parts:
                try:
                    acc += int(p)
                except Exception:
                    return -1
            return max(1, acc)
        return max(1, int(v))
    except Exception:
        return -1


def _detect_cgroup_mem_limit_bytes() -> int:
    """Return cgroup memory limit in bytes when available; otherwise -1.
    
    Supports cgroup v2 (memory.max) and v1 (memory.limit_in_bytes).
    """
    # cgroup v2
    v2 = "/sys/fs/cgroup/memory.max"
    lim = _read_int_file(v2)
    if lim > 0:
        # Filter out unrealistically large values (no limit)
        try:
            if lim > (1 << 56):
                return -1
        except Exception:
            pass
        return lim
    # cgroup v1
    v1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    lim = _read_int_file(v1)
    if lim > 0:
        try:
            if lim > (1 << 56):
                return -1
        except Exception:
            pass
        return lim
    return -1


def _detect_slurm_job_mem_bytes() -> int:
    """Infer SLURM job memory allocation in bytes from env vars.
    
    Prefers SLURM_MEM_PER_NODE; otherwise uses SLURM_MEM_PER_CPU * SLURM_CPUS_ON_NODE.
    Values are MB according to SLURM docs.
    """
    try:
        mem_per_node_mb = os.environ.get("SLURM_MEM_PER_NODE")
        if mem_per_node_mb:
            mb = _parse_int(mem_per_node_mb)
            if mb > 0:
                return mb * 1024 * 1024
    except Exception:
        pass
    try:
        mem_per_cpu_mb = os.environ.get("SLURM_MEM_PER_CPU")
        cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE")
        if mem_per_cpu_mb and cpus_on_node:
            mb = _parse_int(mem_per_cpu_mb)
            cpus = _parse_cpus_on_node(cpus_on_node)
            if mb > 0 and cpus > 0:
                return mb * cpus * 1024 * 1024
    except Exception:
        pass
    return -1


def _effective_total_memory_bytes() -> int:
    """Best-effort job-aware total memory for sizing Ray object store.
    
    Order of preference:
    1) cgroup memory limit (container/SLURM cgroup)
    2) SLURM env-based memory inference
    3) System total memory (psutil/sysconf)
    """
    # cgroup limit first
    cg = _detect_cgroup_mem_limit_bytes()
    if cg > 0:
        return cg
    # SLURM-derived
    sj = _detect_slurm_job_mem_bytes()
    if sj > 0:
        return sj
    # Fallback to system total
    try:
        import psutil  # type: ignore
        tot = int(getattr(psutil.virtual_memory(), "total", 0))
        if tot > 0:
            return tot
    except Exception:
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        pass
    return -1


def _resolve_model_path(model_source: str) -> str:
    """Resolve model path from zoo or HuggingFace Hub.
    
    Priority:
    1. If absolute path exists: use as-is
    2. If relative path in zoo: resolve to zoo/models/{model_source}
    3. If model name matches zoo directory: resolve to zoo/models/{name}
    4. Otherwise: use as HuggingFace Hub identifier
    
    Args:
        model_source: Model path/name from config (e.g., "Qwen2.5-VL-3B-Instruct" 
                     or "/share/pierson/matt/zoo/models/Qwen3-30B-A3B-Instruct-2507")
    
    Returns:
        Resolved path (absolute path or HuggingFace Hub identifier)
    """
    # Already an absolute path
    if os.path.isabs(model_source) and os.path.exists(model_source):
        return model_source
    
    # Check if it's in the zoo
    zoo_path = os.path.join(MODEL_ZOO_BASE, model_source)
    if os.path.exists(zoo_path):
        return zoo_path
    
    # Try fuzzy matching in zoo directory
    if os.path.exists(MODEL_ZOO_BASE):
        try:
            zoo_dirs = [d for d in os.listdir(MODEL_ZOO_BASE) 
                       if os.path.isdir(os.path.join(MODEL_ZOO_BASE, d))]
            # Check for exact match or partial match
            for dir_name in zoo_dirs:
                if model_source.lower() in dir_name.lower() or dir_name.lower() in model_source.lower():
                    resolved = os.path.join(MODEL_ZOO_BASE, dir_name)
                    if os.path.exists(os.path.join(resolved, "config.json")):
                        return resolved
        except Exception:
            pass  # Fallback to HuggingFace Hub
    
    # Fallback to HuggingFace Hub
    return model_source


def _is_multimodal_model(model_source: str, cfg: Optional[Any] = None) -> bool:
    """Detect if model supports multimodal inputs.
    
    Checks:
    1. Model name patterns (e.g., "Qwen2.5-VL", "InternVL", "Phi-3.5-vision")
    2. Explicit config flag: runtime.multimodal_enabled
    3. Model config metadata (if available)
    4. Config.json in model directory for local models
    
    Args:
        model_source: Model path/name
        cfg: Optional config object for checking runtime.multimodal_enabled
        
    Returns:
        True if multimodal capabilities detected
    """
    # Resolve model path first
    resolved_path = _resolve_model_path(model_source)
    
    # Check explicit flag first
    if cfg is not None and hasattr(cfg, "runtime"):
        if hasattr(cfg.runtime, "multimodal_enabled"):
            return bool(cfg.runtime.multimodal_enabled)
    
    # For local models, check config.json
    if os.path.isabs(resolved_path) and os.path.exists(resolved_path):
        config_path = os.path.join(resolved_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    # Check for multimodal indicators in config
                    model_type = config.get("model_type", "").lower()
                    arch = config.get("architectures", [])
                    if any("vision" in str(a).lower() or "multimodal" in str(a).lower() 
                           for a in arch):
                        return True
                    if "vision" in model_type or "multimodal" in model_type:
                        return True
            except Exception:
                pass  # Fallback to pattern matching
    
    # Pattern matching for common multimodal models
    multimodal_patterns = [
        r"Qwen.*VL",
        r"InternVL",
        r"Phi-3.*vision",
        r"SmolVLM",
        r"LLaVA",
        r"CLIP",
        r"vision",
        r"multimodal",
    ]
    
    model_lower = model_source.lower()
    return any(re.search(pattern, model_lower, re.IGNORECASE) for pattern in multimodal_patterns)


def _maybe_silence_vllm_logs() -> None:
    """Silence verbose vLLM logs by throttling INFO messages."""
    global _VLLM_LOGS_SILENCED
    if _VLLM_LOGS_SILENCED:
        return
    try:
        # Throttle INFO logs to every N messages; always allow WARNING+
        from dagspaces.urbanvqa.logging_filters import PatternModuloFilter  # local import for worker
        lg = logging.getLogger("vllm")
        try:
            n = int(os.environ.get("URBANVQA_VLLM_LOG_EVERY", "10") or "10")
        except Exception:
            n = 10
        lg.setLevel(logging.INFO)
        # Attach once
        try:
            existing_filters = getattr(lg, "filters", [])
            if not any(getattr(f, "__class__", object).__name__ == "PatternModuloFilter" for f in existing_filters):
                lg.addFilter(PatternModuloFilter(mod=n, pattern="Elapsed time for batch"))
        except Exception:
            pass
        # If explicit silence requested, escalate to ERROR
        if os.environ.get("RULE_TUPLES_SILENT"):
            lg.setLevel(logging.ERROR)
        _VLLM_LOGS_SILENCED = True
    except Exception:
        pass


def _ensure_ray_init(cfg) -> None:
    """Initialize Ray with SLURM-aware CPU and memory limits."""
    try:
        import ray  # type: ignore
        if not ray.is_initialized():
            # Detect SLURM CPU allocation
            cpus_alloc = None
            try:
                cpt = os.environ.get("SLURM_CPUS_PER_TASK")
                if cpt is not None and str(cpt).strip() != "":
                    cpus_alloc = int(cpt)
                else:
                    con = os.environ.get("SLURM_CPUS_ON_NODE")
                    if con is not None and str(con).strip() != "":
                        cpus_alloc = _parse_cpus_on_node(con)
            except Exception:
                cpus_alloc = None
            # Prefer proportion of system memory when available; fallback to job_memory_gb.
            obj_store_bytes = None
            try:
                # Allow explicit override via cfg.runtime.object_store_proportion (0.0-1.0)
                prop = getattr(cfg.runtime, "object_store_proportion", None)
                prop = float(prop) if prop is not None else None
            except Exception:
                prop = None
            # Honor env proportion if set and no explicit override provided
            try:
                if prop is None:
                    env_prop = os.environ.get("RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION")
                    if env_prop is not None and str(env_prop).strip() != "":
                        prop = float(env_prop)
            except Exception:
                pass
            if prop is not None and 0.0 < prop <= 0.95:
                try:
                    total_bytes = _effective_total_memory_bytes()
                    if total_bytes:
                        obj_store_bytes = int(total_bytes * float(prop))
                except Exception:
                    obj_store_bytes = None
            if obj_store_bytes is None:
                try:
                    # Prefer SLURM job mem if available to avoid using full node memory
                    slurm_bytes = _detect_slurm_job_mem_bytes()
                    if slurm_bytes > 0:
                        job_mem_gb = max(1, int(slurm_bytes / (1024 ** 3)))
                    else:
                        job_mem_gb = int(getattr(cfg.runtime, "job_memory_gb", 64) or 64)
                except Exception:
                    job_mem_gb = 64
                try:
                    job_mem_gb = int(getattr(cfg.runtime, "job_memory_gb", 64) or 64)
                except Exception:
                    job_mem_gb = 64
                try:
                    obj_store_bytes = int(max(1, job_mem_gb) * (1024 ** 3) * 0.90)
                except Exception:
                    obj_store_bytes = int(64 * (1024 ** 3) * 0.90)
            try:
                if cpus_alloc is not None and int(cpus_alloc) > 0:
                    ray.init(log_to_driver=True, object_store_memory=int(obj_store_bytes), num_cpus=int(cpus_alloc))
                else:
                    ray.init(log_to_driver=True, object_store_memory=int(obj_store_bytes))
            except Exception:
                # Best-effort fallback: let Ray auto-init
                try:
                    ray.init(log_to_driver=True)
                except Exception:
                    pass
            # Constrain Ray Data CPU limits to SLURM allocation when available
            try:
                if cpus_alloc is not None and int(cpus_alloc) > 0:
                    ctx = ray.data.DataContext.get_current()
                    ctx.execution_options.resource_limits = ctx.execution_options.resource_limits.copy(cpu=int(cpus_alloc))
            except Exception:
                pass
    except Exception:
        pass


def _detect_num_gpus() -> int:
    """Detect the number of GPUs allocated to this job.
    
    Priority order:
    1. CUDA_VISIBLE_DEVICES environment variable (set by launcher)
    2. SLURM_GPUS_PER_NODE or SLURM_GPUS_ON_NODE
    3. torch.cuda.device_count() if CUDA is available
    4. Return 1 as safe fallback
    """
    # Priority 1: CUDA_VISIBLE_DEVICES (most reliable for actual allocation)
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            # Parse comma-separated GPU indices (e.g., "0,1,2,3" -> 4 GPUs)
            gpu_indices = [x.strip() for x in cuda_visible.split(",") if x.strip()]
            if gpu_indices:
                return len(gpu_indices)
    except Exception:
        pass
    
    # Priority 2: SLURM environment variables
    try:
        slurm_gpus = os.environ.get("SLURM_GPUS_PER_NODE") or os.environ.get("SLURM_GPUS_ON_NODE")
        if slurm_gpus:
            # Can be a number like "4" or format like "gpu:4"
            try:
                if ":" in slurm_gpus:
                    return int(slurm_gpus.split(":")[-1])
                return int(slurm_gpus)
            except Exception:
                pass
    except Exception:
        pass
    
    # Priority 3: Torch CUDA device count
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                return count
    except Exception:
        pass
    
    # Fallback: 1 GPU
    return 1


def _detect_gpu_type() -> str:
    """Detect the GPU type/model name.
    
    Returns a normalized GPU type string (e.g., 'rtx_a6000', 'rtx_a5000', 'unknown').
    """
    try:
        import torch  # type: ignore
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Get the name of the first GPU (assuming homogeneous GPUs)
            gpu_name = torch.cuda.get_device_name(0).lower()
            
            # Normalize common GPU names
            if "a6000" in gpu_name:
                return "rtx_a6000"
            elif "a5000" in gpu_name:
                return "rtx_a5000"
            elif "a100" in gpu_name:
                return "a100"
            elif "v100" in gpu_name:
                return "v100"
            elif "a40" in gpu_name:
                return "a40"
            elif "rtx" in gpu_name:
                # Generic RTX
                return "rtx_generic"
            
            return "unknown"
    except Exception:
        pass
    
    return "unknown"


def _apply_gpu_aware_batch_settings(engine_kwargs: Dict[str, Any], cfg) -> Dict[str, Any]:
    """Apply GPU-type-aware batch size and max_num_seqs if not explicitly set.
    
    GPU-specific defaults (can be overridden in config):
    - RTX A6000: batch_size=4, max_num_seqs=4
    - RTX A5000: batch_size=2, max_num_seqs=2
    - Others: Use config defaults or fallback values
    
    Returns:
        Dictionary with GPU-specific batch settings
    """
    # GPU-aware defaults mapping
    GPU_BATCH_SETTINGS = {
        "rtx_a6000": {"batch_size": 4, "max_num_seqs": 4},
        "rtx_a5000": {"batch_size": 2, "max_num_seqs": 2},
        "a100": {"batch_size": 8, "max_num_seqs": 8},
        "v100": {"batch_size": 4, "max_num_seqs": 4},
        "a40": {"batch_size": 4, "max_num_seqs": 4},
    }
    
    gpu_type = _detect_gpu_type()
    gpu_settings = GPU_BATCH_SETTINGS.get(gpu_type, {})
    
    # Apply max_num_seqs from GPU settings if not in engine_kwargs and GPU type is recognized
    if "max_num_seqs" not in engine_kwargs and gpu_settings:
        try:
            engine_kwargs["max_num_seqs"] = gpu_settings["max_num_seqs"]
            if not os.environ.get("RULE_TUPLES_SILENT"):
                print(f"Auto-set max_num_seqs={gpu_settings['max_num_seqs']} for {gpu_type}")
        except Exception:
            pass
    
    # Note: batch_size is handled separately in the calling code since it's a model config param, not engine_kwargs
    # We'll return the gpu_settings for use there
    return gpu_settings


def _filter_vllm_engine_kwargs(ek: Dict[str, Any]) -> Dict[str, Any]:
    """Drop engine kwargs unsupported by the installed vLLM version.

    We try to introspect vllm.AsyncEngineArgs for accepted fields. If that
    fails, conservatively drop known newer flags.
    """
    try:
        import vllm as _v
        accepted = None
        # Prefer dataclass fields (older vLLM uses dataclasses)
        try:
            fields = getattr(getattr(_v, "AsyncEngineArgs", None), "__dataclass_fields__", None)
            if isinstance(fields, dict) and fields:
                accepted = set(fields.keys())
        except Exception:
            accepted = None
        # Fallback to signature introspection
        if accepted is None:
            try:
                import inspect as _inspect
                sig = _inspect.signature(_v.AsyncEngineArgs.__init__)
                accepted = set(k for k in sig.parameters.keys() if k != "self")
            except Exception:
                accepted = None
        if accepted:
            filtered = {k: v for k, v in ek.items() if k in accepted}
            if len(filtered) != len(ek):
                try:
                    if not os.environ.get("RULE_TUPLES_SILENT"):
                        dropped = [k for k in ek.keys() if k not in accepted]
                        print(f"Filtering unsupported vLLM engine kwargs: {dropped}")
                except Exception:
                    pass
            return filtered
    except Exception:
        pass
    # Conservative fallback for unknown versions: drop newer flags
    ek = dict(ek)
    for k in ("use_v2_block_manager",):
        ek.pop(k, None)
    return ek


def _convert_image_to_base64(image_source: Any, row: Dict[str, Any] = None) -> Optional[str]:
    """Convert numpy array image to base64 string (PyArrow-serializable).
    
    CRITICAL: This function does NOT modify the original image_source.
    It works on a copy to ensure ray.data.read_images() images are never tampered with.
    
    ray.data.read_images() already provides numpy arrays in consistent format - no checks needed.
    
    Args:
        image_source: Numpy array from ray.data.read_images() (read-only, never modified)
        row: Optional row dict (unused, kept for compatibility)
        
    Returns:
        Base64 string with data URI prefix, or None if conversion fails
    """
    if image_source is None:
        return None
    if np is None:
        raise RuntimeError("NumPy is required for converting images to base64 but is not available.")
    
    try:
        # Coerce to numpy array and copy to avoid mutating the original buffer
        sample_id = row.get("sample_id") if isinstance(row, dict) else None
        np_image = _ensure_numpy_image_value(image_source, sample_id)
        img_val = np_image.copy()
        if img_val.dtype != np.uint8:
            img_val = img_val.astype(np.uint8, copy=False)
        
        # Convert to PIL Image for encoding (PIL handles format conversion automatically)
        if not _PIL_AVAILABLE:
            return None
        
        pil_img = PILImage.fromarray(img_val).convert("RGB")
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        return f"data:image/jpeg;base64,{base64_bytes.decode('utf-8')}"
    except Exception:
        return None


def _normalize_sampling_params(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize sampling params to ensure compatibility with vLLM.
    
    Ensures stop is always a list (vLLM requirement).
    """
    sp_normalized = dict(sp)
    # Ensure stop is always a list
    stop_val = sp_normalized.get("stop")
    if stop_val is None:
        sp_normalized["stop"] = []
    elif not isinstance(stop_val, list):
        sp_normalized["stop"] = [str(stop_val)]
    return sp_normalized


def render_prompt_template(template_str: str, context: Dict[str, Any]) -> str:
    """Render Jinja2 template with variable substitution.
    
    Args:
        template_str: Jinja2 template string
        context: Dictionary of template variables
        
    Returns:
        Rendered prompt string
        
    Raises:
        ValueError: If Jinja2 is not available or template rendering fails
    """
    if not _JINJA2_AVAILABLE:
        raise ValueError("Jinja2 is not available. Install with: pip install jinja2")
    
    env = Environment(undefined=StrictUndefined)  # Raise error on undefined variables
    template = env.from_string(template_str)
    return template.render(**context)


def _prepare_image_content(image: Any, sample_id: str = None) -> Dict[str, Any]:
    """Prepare image content in vLLM-compatible OpenAI chat format.
    
    CRITICAL: This function does NOT modify the original image.
    It uses _convert_image_to_base64 which works on a copy.
    
    Simplified approach: Only handles numpy arrays from ray.data.read_images().
    Converts numpy arrays to base64 strings for PyArrow compatibility.
    
    Args:
        image: Numpy array from ray.data.read_images() (read-only, never modified)
        sample_id: Optional sample ID for error messages
        
    Returns:
        Dictionary with 'type': 'image_url' and base64 string (PyArrow-serializable!)
    """
    # Coerce to numpy array (handles PIL or other formats) without mutating original source
    np_image = _ensure_numpy_image_value(image, sample_id)
    
    # _convert_image_to_base64 works on a copy - does not modify original
    base64_str = _convert_image_to_base64(np_image, None)
    if base64_str:
        return {
            "type": "image_url",
            "image_url": {"url": base64_str}
        }
    
    raise ValueError(f"Failed to convert numpy array to base64 for sample {sample_id}")


def _group_by_prompt_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group rows by prompt for efficient batch processing.
    
    This optimization reorders rows so that rows with the same prompt
    are processed together, allowing prompt tokenization to be reused.
    
    Best Practice: Apply this before creating Ray Dataset to ensure
    same prompts are batched together within Ray Data's batch processing.
    
    Args:
        df: DataFrame with 'prompt' column
        
    Returns:
        DataFrame sorted by prompt groups
    """
    if "prompt" not in df.columns:
        return df
    
    # Add a grouping key
    df = df.copy()
    df['_prompt_group'] = df.groupby('prompt').ngroup()
    # Sort by prompt group to ensure same prompts are batched together
    df = df.sort_values('_prompt_group').reset_index(drop=True)
    # Drop the temporary grouping column
    df = df.drop(columns=['_prompt_group'], errors='ignore')
    return df


def run_vqa_stage(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run VQA inference on a dataset.
    
    Args:
        df: DataFrame with columns: prompt, image_path/image_url/image_base64, sample_id
        cfg: Configuration object
        
    Returns:
        DataFrame with columns: sample_id, prompt, answer, model_response, metadata
    """
    # Ensure Ray is initialized
    _ensure_ray_init(cfg)
    
    # Enable fallback to Arrow object extension types for PIL Images and other complex objects
    # This allows Ray Data to handle PIL Images in messages structure without Arrow conversion errors
    if _RAY_OK:
        try:
            from ray.data import DataContext
            DataContext.get_current().enable_fallback_to_arrow_object_ext_type = True
        except Exception:
            pass  # Continue if DataContext not available
    
    # Streaming path: if a Ray Dataset is passed, use it end-to-end
    is_ray_ds = hasattr(df, "map_batches") and hasattr(df, "count") and _RAY_OK
    
    if not is_ray_ds:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["sample_id", "prompt", "answer"])
        # Convert to Ray Dataset for processing
        if not _RAY_OK:
            raise RuntimeError("Ray is required for VQA stage but not available")
        
        # Apply prompt grouping optimization if enabled
        group_by_prompt = getattr(cfg.runtime, "group_by_prompt", False) if hasattr(cfg, "runtime") else False
        if group_by_prompt:
            df = _group_by_prompt_optimization(df)
        
        ds = ray.data.from_pandas(df)
    else:
        ds = df
    
    if np is None:
        raise RuntimeError("NumPy is required for VQA stage but is not available.")
    
    # Check for hierarchical prompts
    hierarchical_enabled = False
    hierarchical_steps = []
    try:
        hierarchical_config = getattr(cfg.prompt, "hierarchical", None)
        if hierarchical_config and getattr(hierarchical_config, "enabled", False):
            hierarchical_enabled = True
            hierarchical_steps = list(getattr(hierarchical_config, "steps", []))
    except Exception:
        pass
    
    # Check for decision tree prompts
    decision_tree_enabled = False
    decision_tree_config = None
    try:
        decision_tree_config = getattr(cfg.prompt, "decision_tree", None)
        if decision_tree_config and getattr(decision_tree_config, "enabled", False):
            decision_tree_enabled = True
    except Exception:
        pass
    
    # Check for other dynamic prompting techniques
    cot_enabled = False
    react_enabled = False
    rap_enabled = False
    adaptive_enabled = False
    contextual_enabled = False
    
    try:
        cot_enabled = getattr(getattr(cfg.prompt, "chain_of_thought", None), "enabled", False)
        react_enabled = getattr(getattr(cfg.prompt, "react", None), "enabled", False)
        rap_enabled = getattr(getattr(cfg.prompt, "retrieval_augmented", None), "enabled", False)
        adaptive_enabled = getattr(getattr(cfg.prompt, "adaptive", None), "enabled", False)
        contextual_enabled = getattr(getattr(cfg.prompt, "contextual", None), "enabled", False)
    except Exception:
        pass
    
    # Get system prompt from config
    system_prompt = getattr(cfg.prompt, "system", "You are a helpful assistant that answers questions about images accurately and concisely.")
    
    # Check for Jinja2 template support
    use_template = False
    template_str = None
    if _JINJA2_AVAILABLE:
        try:
            template_str = getattr(cfg.prompt, "template", None)
            if template_str:
                use_template = True
        except Exception:
            pass
    
    # Get sampling params
    sampling_params_vqa = getattr(cfg, "sampling_params_vqa", {})
    if not sampling_params_vqa:
        sampling_params_vqa = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "stop": [],
        }
    
    # Normalize the base sampling params to ensure stop is always a list (vLLM requirement)
    sampling_params_vqa = _normalize_sampling_params(sampling_params_vqa)
    
    # Check for structured JSON output
    structured_output_enabled = False
    json_schema = None
    try:
        structured_output_config = getattr(cfg.prompt, "structured_output", None)
        if structured_output_config and getattr(structured_output_config, "enabled", False):
            structured_output_enabled = True
            # Get schema from config or Pydantic model
            schema_path = getattr(structured_output_config, "schema_path", None)
            if schema_path:
                # Load Pydantic model and get schema
                import importlib.util
                spec = importlib.util.spec_from_file_location("schema_module", schema_path)
                if spec and spec.loader:
                    schema_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(schema_module)
                    # Assume model is named VQAAnswer or similar
                    if hasattr(schema_module, "VQAAnswer"):
                        json_schema = schema_module.VQAAnswer.model_json_schema()
            else:
                # Use inline JSON schema from config
                json_schema = getattr(structured_output_config, "json_schema", None)
    except Exception:
        pass
    
    # Resolve model path
    model_source_raw = getattr(cfg.model, "model_source", "")
    resolved_model_source = _resolve_model_path(model_source_raw)
    is_multimodal = _is_multimodal_model(resolved_model_source, cfg)
    
    # Build engine config
    engine_kwargs = getattr(cfg.model, "engine_kwargs", {})
    engine_kwargs = _filter_vllm_engine_kwargs(dict(engine_kwargs))
    
    # Ensure multimodal settings
    if is_multimodal:
        engine_kwargs.setdefault("limit_mm_per_prompt", {"image": 1})
        engine_kwargs.setdefault("trust_remote_code", True)
    
    # Get GPU settings
    num_gpus = _detect_num_gpus()
    gpu_type = _detect_gpu_type()
    gpu_settings = _apply_gpu_aware_batch_settings(
        engine_kwargs,
        cfg
    )
    
    batch_size = gpu_settings.get("batch_size", getattr(cfg.model, "batch_size", 16))
    concurrency = getattr(cfg.model, "concurrency", 1)
    
    # Get tensor parallelism size
    tp_val = engine_kwargs.get("tensor_parallel_size", 1)
    if tp_val > 1:
        # Adjust concurrency based on tensor parallelism
        # concurrency should match number of model replicas, not total GPU count
        concurrency = max(1, num_gpus // tp_val)
    
    # Runtime environment
    runtime_env_vars = {}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        runtime_env_vars["HF_TOKEN"] = hf_token
    
    accelerator_type = getattr(cfg.model, "accelerator_type", None)
    
    engine_config = vLLMEngineProcessorConfig(
        model_source=resolved_model_source,
        engine_kwargs=engine_kwargs,
        concurrency=concurrency,
        batch_size=batch_size,
        tokenize=False,  # Let vLLM handle tokenization to avoid Ray TokenizeUDF Arrow/PIL conversions
        apply_chat_template=False,  # Skip Ray ChatTemplateStage; we already supply final messages
        has_image=is_multimodal,
        accelerator_type=accelerator_type,
        runtime_env={"env_vars": runtime_env_vars} if runtime_env_vars else None,
    )
    
    # Preprocessing function - use unified framework
    # BEST PRACTICE: According to Ray Data LLM docs, preprocess should return ONLY messages and sampling_params
    # PIL Images are OK inside messages structure (Ray Data LLM handles them specially)
    # But we must NOT include image columns or PIL Images as separate row columns
    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess row for VQA with dynamic prompting techniques using unified framework.
        
        Returns ONLY messages and sampling_params (plus lightweight metadata).
        PIL Images are included ONLY inside messages structure, not as separate columns.
        """
        import logging
        
        _maybe_silence_vllm_logs()
        
        # Use unified preprocessing framework
        from dagspaces.urbanvqa.prompts.unified import unified_preprocess
        
        unified_result = unified_preprocess(
            row, cfg, is_multimodal,
            hierarchical_enabled, decision_tree_enabled
        )
        
        # If unified preprocessing returns None, it means structural techniques handle it
        # This shouldn't happen in standard flow, but handle gracefully
        if unified_result is None:
            # CRITICAL: Filter out image columns completely - we must NOT touch images from ray.data.read_images()
            # preprocess_simple will read from row["image"] but we don't include it in lightweight_row
            lightweight_row = {}
            excluded_cols = {"image", "image_array", "image_data", "path", "messages", "sampling_params"}
            for k, v in row.items():
                if k in excluded_cols:
                    continue
                # Only include simple, serializable types (NO image column)
                if isinstance(v, (str, int, float, type(None))):
                    lightweight_row[k] = v
                elif isinstance(v, dict):
                    if all(isinstance(vv, (str, int, float, type(None))) for vv in v.values()):
                        lightweight_row[k] = v
                elif isinstance(v, list):
                    if all(isinstance(vv, (str, int, float, type(None))) for vv in v):
                        lightweight_row[k] = v
                elif k in {"image_path", "image_url", "image_base64"} and isinstance(v, str):
                    lightweight_row[k] = v
            lightweight_row["prompt"] = str(row.get("prompt", "")).strip()
            # Pass original row to preprocess_simple so it can read image, but result won't include it
            result = preprocess_simple(row, cfg, is_multimodal)
            
            # DEBUG: Log result
            if result and "image" in result:
                logging.error(f"[_pre] RESULT HAS IMAGE COLUMN! This should NOT happen!")
            return result
        
        # If unified preprocessing returns messages, use them
        if "messages" in unified_result:
            # Add structured output if enabled
            sp_local = dict(unified_result.get("sampling_params", sampling_params_vqa))
            # Normalize sampling params to ensure stop is a list
            sp_local = _normalize_sampling_params(sp_local)
            if structured_output_enabled and json_schema:
                try:
                    schema_json_str = json.dumps(json_schema, ensure_ascii=False)
                    sp_local["guided_decoding"] = {"json": schema_json_str}
                except Exception:
                    pass
            
            # BEST PRACTICE: Return ONLY messages, sampling_params, and lightweight metadata
            # Ray Data LLM's preprocess function should completely replace the row
            # PIL Images inside messages are handled specially by Ray Data LLM
            result = {
                "messages": unified_result["messages"],
                "sampling_params": sp_local,
            }
            
            # Only include lightweight, serializable metadata (strings, numbers)
            # These will be preserved through the pipeline and available in postprocess
            for key in ["sample_id", "prompt"]:
                if key in row and isinstance(row[key], (str, int, float, type(None))):
                    result[key] = row[key]
            
            # Add timestamp as metadata
            result["ts_start"] = datetime.now().timestamp()
            
            # DEBUG: Log result structure
            result_keys = list(result.keys())
            result_types = {k: type(v).__name__ for k, v in result.items()}
            if "image" in result:
                logging.error(f"[_pre] RESULT HAS IMAGE COLUMN! This should NOT happen! keys={result_keys}, types={result_types}")
            else:
                logging.info(f"[_pre] Result has no image column: keys={result_keys}, types={result_types}")
            
            return result
        
        # Fallback: use simple preprocessing
        # CRITICAL: Do NOT filter or modify image column - pass original row but result won't include image
        # preprocess_simple will read from row["image"] but won't include it in return value
        result = preprocess_simple(row, cfg, is_multimodal)
        
        # DEBUG: Log result
        if result and "image" in result:
            logging.error(f"[_pre] RESULT HAS IMAGE COLUMN! This should NOT happen!")
        
        return result
    
    # Postprocessing function - use unified framework
    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess VQA response using unified framework."""
        from dagspaces.urbanvqa.prompts.unified import unified_postprocess
        
        # Use unified postprocessing
        unified_result = unified_postprocess(
            row, cfg, hierarchical_enabled, decision_tree_enabled
        )
        
        # Merge with structured output parsing
        ts_end = datetime.now().timestamp()
        generated_text = str(row.get("generated_text", "")).strip()
        
        # Parse structured JSON output if enabled
        if structured_output_enabled:
            try:
                if generated_text.strip().startswith("{"):
                    parsed = json.loads(generated_text)
                else:
                    # Try to extract JSON from text
                    import re
                    json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                    else:
                        parsed = {"answer": generated_text}
            except json.JSONDecodeError:
                parsed = {"answer": generated_text}
        else:
            parsed = {"answer": generated_text}
        
        # Filter out large/unnecessary columns that shouldn't be written to parquet
        # Keep only essential metadata and results
        excluded_cols = {
            "image", "image_array", "image_data",  # Large image arrays
            "messages", "sampling_params",  # Internal processing data
            "llm_output", "generated_text",  # Already extracted to model_response
            "json", "guided_decoding", "response_format", "structured_output",  # Internal config
            "_hierarchical_results", "_tree_results", "_tree_visited_nodes",  # Internal state
            "ts_start",  # Will be in metadata
        }
        
        # Build result with only necessary columns
        result = {}
        
        # Keep essential input columns (paths/URLs, not arrays) and all metadata columns
        # Preserve all metadata columns (including custom metadata from parquet)
        for col, val in row.items():
            if col in excluded_cols:
                continue
            # Exclude only large/complex objects
            if isinstance(val, (str, int, float, type(None))):
                result[col] = val
            elif isinstance(val, dict):
                # Only include if dict contains only simple types
                if all(isinstance(v, (str, int, float, type(None))) for v in val.values()):
                    result[col] = val
            elif isinstance(val, list):
                # Only include if list contains only simple types
                if all(isinstance(v, (str, int, float, type(None))) for v in val):
                    result[col] = val
        
        # Add unified result (may contain answer, metadata, etc.)
        result.update(unified_result)
        
        # Add final answer and model response
        result["answer"] = unified_result.get("answer", parsed.get("answer", generated_text))
        result["model_response"] = generated_text
        
        # Add metadata
        result["metadata"] = {
            **unified_result.get("metadata", {}),
            "ts_start": row.get("ts_start"),
            "ts_end": ts_end,
            "usage": row.get("usage") or row.get("token_counts"),
        }
        
        # Add structured fields if present
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if key != "answer":
                    result[key] = value
        
        return result
    
    # Process dataset
    if decision_tree_enabled and decision_tree_config:
        # Use decision tree processing
        ds_results = _process_decision_tree_prompts(
            ds, cfg, engine_config, decision_tree_config,
            system_prompt, sampling_params_vqa, is_multimodal,
            structured_output_enabled, json_schema
        )
    elif hierarchical_enabled and hierarchical_steps:
        # Use hierarchical processing
        ds_results = _process_hierarchical_prompts(
            ds, cfg, engine_config, hierarchical_steps, 
            system_prompt, sampling_params_vqa, is_multimodal,
            structured_output_enabled, json_schema
        )
    else:
        # Use standard processing
        # BEST PRACTICE: According to Ray Data LLM docs, preprocess should return ONLY messages and sampling_params
        # The preprocess function completely replaces the row - Ray Data LLM does NOT merge with original columns
        processor = build_llm_processor(engine_config, preprocess=_pre, postprocess=_post)
        ds_results = processor(ds)
    
    # Convert back to pandas if needed
    if is_ray_ds:
        return ds_results
    
    # Materialize and convert to pandas
    df_results = ds_results.to_pandas()
    return df_results


def _process_hierarchical_prompts(
    ds, cfg: DictConfig, engine_config: vLLMEngineProcessorConfig,
    steps: List[Dict[str, Any]], system_prompt: str,
    sampling_params_vqa: Dict[str, Any], is_multimodal: bool,
    structured_output_enabled: bool, json_schema: Optional[Dict[str, Any]]
):
    """Process hierarchical prompts by executing steps sequentially."""
    # Group steps by execution order
    parallel_groups = []
    current_group = []
    
    for step in steps:
        if step.get("parallel", False):
            current_group.append(step)
        else:
            if current_group:
                parallel_groups.append(current_group)
                current_group = []
            parallel_groups.append([step])
    
    if current_group:
        parallel_groups.append(current_group)
    
    # Execute steps sequentially
    current_ds = ds
    
    for group_idx, group in enumerate(parallel_groups):
        if len(group) == 1:
            # Sequential step
            step = group[0]
            
            # Preprocess for this step
            def _pre_hierarchical(row: Dict[str, Any]) -> Dict[str, Any]:
                """Preprocess for hierarchical step."""
                _maybe_silence_vllm_logs()
                
                # Get previous results from metadata
                step_results = row.get("_hierarchical_results", {})
                
                # Format step prompt
                step_prompt = step.get("prompt", "")
                
                # Replace placeholders with previous results
                for key, value in step_results.items():
                    step_prompt = step_prompt.replace(f"{{{{{key}}}}}", str(value))
                
                # Replace user question
                user_question = row.get("prompt", "")
                step_prompt = step_prompt.replace("{{final_question}}", user_question)
                step_prompt = step_prompt.replace("{{user_question}}", user_question)
                
                # Convert numpy array to base64 (PyArrow-serializable!)
                image_base64_str = None
                
                if is_multimodal:
                    if "image" not in row or row["image"] is None:
                        raise ValueError(f"Image is required for multimodal hierarchical step (sample_id: {row.get('sample_id')})")
                    
                    # Normalize to numpy array before conversion (does not mutate original buffer)
                    image_source = _ensure_numpy_image_value(row["image"], row.get("sample_id"))
                    
                    # _convert_image_to_base64 works on a copy - does not modify original
                    image_base64_str = _convert_image_to_base64(image_source, row)
                    if not image_base64_str:
                        raise ValueError(f"Failed to convert numpy array to base64 for hierarchical step (sample_id: {row.get('sample_id')})")
                
                # Build messages with base64 string (PyArrow-serializable!)
                if is_multimodal and image_base64_str is not None:
                    user_content = [
                        {"type": "text", "text": step_prompt},
                        {"type": "image_url", "image_url": {"url": image_base64_str}}
                    ]
                else:
                    user_content = step_prompt
                
                sp_local = dict(sampling_params_vqa)
                # Normalize sampling params to ensure stop is a list
                sp_local = _normalize_sampling_params(sp_local)
                if structured_output_enabled and json_schema:
                    try:
                        schema_json_str = json.dumps(json_schema, ensure_ascii=False)
                        sp_local["guided_decoding"] = {"json": schema_json_str}
                    except Exception:
                        pass
                
                # CRITICAL: Only return messages, sampling_params, and lightweight metadata
                # Do NOT include image columns or complex objects
                result = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "sampling_params": sp_local,
                }
                
                # Preserve all lightweight, serializable metadata columns (including custom metadata from parquet)
                # Exclude image arrays, PIL Images, and complex objects
                excluded_cols = {"image", "image_array", "image_data", "messages", "sampling_params", "path"}
                lightweight_metadata = {}
                for key, val in row.items():
                    if key in excluded_cols:
                        continue
                    # Only include if it's a simple type (not PIL Image or complex object)
                    if isinstance(val, (str, int, float, type(None))):
                        lightweight_metadata[key] = val
                    elif isinstance(val, dict):
                        # Check if dict contains only simple types
                        if all(isinstance(v, (str, int, float, type(None))) for v in val.values()):
                            lightweight_metadata[key] = val
                    elif isinstance(val, list):
                        # Check if list contains only simple types
                        if all(isinstance(v, (str, int, float, type(None))) for v in val):
                            lightweight_metadata[key] = val
                
                # Add step metadata
                lightweight_metadata["_hierarchical_step"] = step.get("name")
                lightweight_metadata["_hierarchical_output_key"] = step.get("output_key")
                
                result.update(lightweight_metadata)
                
                # CRITICAL: Explicitly ensure image column is NOT in result
                result.pop("image", None)
                result.pop("image_array", None)
                result.pop("image_data", None)
                result.pop("path", None)
                
                return result
            
            # Postprocess for this step
            def _post_hierarchical(row: Dict[str, Any]) -> Dict[str, Any]:
                """Postprocess hierarchical step response."""
                generated_text = str(row.get("generated_text", "")).strip()
                output_key = row.get("_hierarchical_output_key")
                
                # Get previous results
                step_results = row.get("_hierarchical_results", {})
                if output_key:
                    step_results[output_key] = generated_text
                
                # Filter out large/unnecessary columns (including image arrays and PIL Images)
                excluded_cols = {
                    "image", "image_array", "image_data", "path",  # Image data
                    "messages", "sampling_params",  # Internal processing
                    "llm_output", "generated_text",  # Already extracted
                    "json", "guided_decoding",  # Internal config
                }
                
                result = {}
                # Preserve all lightweight, serializable metadata columns (including custom metadata from parquet)
                # Exclude image arrays, PIL Images, and complex objects
                for k, v in row.items():
                    if k.startswith("_hierarchical") or k in excluded_cols:
                        continue
                    # Only include if it's a simple type (not PIL Image, numpy array, or complex object)
                    if isinstance(v, (str, int, float, type(None))):
                        result[k] = v
                    elif isinstance(v, dict):
                        # Check if dict contains only simple types
                        if all(isinstance(vv, (str, int, float, type(None))) for vv in v.values()):
                            result[k] = v
                    elif isinstance(v, list):
                        # Check if list contains only simple types
                        if all(isinstance(vv, (str, int, float, type(None))) for vv in v):
                            result[k] = v
                
                result["_hierarchical_results"] = step_results
                
                # Add output key to result
                if output_key:
                    result[output_key] = generated_text
                
                return result
            
            # Process this step
            step_processor = build_llm_processor(
                engine_config,
                preprocess=_pre_hierarchical,
                postprocess=_post_hierarchical
            )
            current_ds = step_processor(current_ds)
        
        else:
            # Parallel group - process all steps sequentially (they can be parallelized later)
            # For now, process sequentially but collect all outputs
            for parallel_step in group:
                def _pre_parallel(row: Dict[str, Any], step=parallel_step) -> Dict[str, Any]:
                    """Preprocess for parallel step."""
                    _maybe_silence_vllm_logs()
                    
                    # Get previous results (before this parallel group)
                    step_results = row.get("_hierarchical_results", {})
                    
                    # Format step prompt
                    step_prompt = step.get("prompt", "")
                    
                    # Replace placeholders
                    for key, value in step_results.items():
                        step_prompt = step_prompt.replace(f"{{{{{key}}}}}", str(value))
                    
                    user_question = row.get("prompt", "")
                    step_prompt = step_prompt.replace("{{final_question}}", user_question)
                    step_prompt = step_prompt.replace("{{user_question}}", user_question)
                    
                    # Convert numpy array to base64 (PyArrow-serializable!)
                    image_base64_str = None
                    
                    if is_multimodal:
                        if "image" not in row or row["image"] is None:
                            raise ValueError(f"Image is required for multimodal parallel step (sample_id: {row.get('sample_id')})")
                        
                        # Normalize to numpy array before conversion (does not mutate original buffer)
                        image_source = _ensure_numpy_image_value(row["image"], row.get("sample_id"))
                        
                        # _convert_image_to_base64 works on a copy - does not modify original
                        image_base64_str = _convert_image_to_base64(image_source, row)
                        if not image_base64_str:
                            raise ValueError(f"Failed to convert numpy array to base64 for parallel step (sample_id: {row.get('sample_id')})")
                    
                    # Build messages with base64 string (PyArrow-serializable!)
                    if is_multimodal and image_base64_str is not None:
                        user_content = [
                            {"type": "text", "text": step_prompt},
                            {"type": "image_url", "image_url": {"url": image_base64_str}}
                        ]
                    else:
                        user_content = step_prompt
                    
                    sp_local = dict(sampling_params_vqa)
                    # Normalize sampling params to ensure stop is a list
                    sp_local = _normalize_sampling_params(sp_local)
                    if structured_output_enabled and json_schema:
                        try:
                            schema_json_str = json.dumps(json_schema, ensure_ascii=False)
                            sp_local["guided_decoding"] = {"json": schema_json_str}
                        except Exception:
                            pass
                    
                    # CRITICAL: Only return messages, sampling_params, and lightweight metadata
                    # Do NOT include image columns or complex objects
                    result = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        "sampling_params": sp_local,
                    }
                    
                    # Preserve all lightweight, serializable metadata columns (including custom metadata from parquet)
                    # Exclude image arrays, PIL Images, and complex objects
                    excluded_cols = {"image", "image_array", "image_data", "messages", "sampling_params", "path"}
                    lightweight_metadata = {}
                    for key, val in row.items():
                        if key in excluded_cols:
                            continue
                        # Only include if it's a simple type (not PIL Image or complex object)
                        if isinstance(val, (str, int, float, type(None))):
                            lightweight_metadata[key] = val
                        elif isinstance(val, dict):
                            # Check if dict contains only simple types
                            if all(isinstance(v, (str, int, float, type(None))) for v in val.values()):
                                lightweight_metadata[key] = val
                        elif isinstance(val, list):
                            # Check if list contains only simple types
                            if all(isinstance(v, (str, int, float, type(None))) for v in val):
                                lightweight_metadata[key] = val
                    
                    # Add step metadata
                    lightweight_metadata["_hierarchical_step"] = step.get("name")
                    lightweight_metadata["_hierarchical_output_key"] = step.get("output_key")
                    # Only include step_results if it's a dict of simple types
                    if isinstance(step_results, dict) and all(isinstance(v, (str, int, float, type(None))) for v in step_results.values()):
                        lightweight_metadata["_hierarchical_results"] = step_results
                    
                    result.update(lightweight_metadata)
                    
                    # CRITICAL: Explicitly ensure image column is NOT in result
                    result.pop("image", None)
                    result.pop("image_array", None)
                    result.pop("image_data", None)
                    result.pop("path", None)
                    
                    return result
                
                def _post_parallel(row: Dict[str, Any], step=parallel_step) -> Dict[str, Any]:
                    """Postprocess parallel step."""
                    generated_text = str(row.get("generated_text", "")).strip()
                    output_key = row.get("_hierarchical_output_key")
                    
                    step_results = row.get("_hierarchical_results", {})
                    if output_key:
                        step_results[output_key] = generated_text
                    
                    # Filter out large/unnecessary columns
                    excluded_cols = {
                        "image", "image_array", "image_data", "messages", "sampling_params",
                        "llm_output", "generated_text", "json", "guided_decoding",
                    }
                    
                    result = {}
                    # Preserve all metadata columns (including custom metadata from parquet)
                    # Exclude only large/complex objects
                    for k, v in row.items():
                        if k.startswith("_hierarchical") or k in excluded_cols:
                            continue
                        # Only include if it's a simple type
                        if isinstance(v, (str, int, float, type(None))):
                            result[k] = v
                        elif isinstance(v, dict):
                            # Only include if dict contains only simple types
                            if all(isinstance(vv, (str, int, float, type(None))) for vv in v.values()):
                                result[k] = v
                        elif isinstance(v, list):
                            # Only include if list contains only simple types
                            if all(isinstance(vv, (str, int, float, type(None))) for vv in v):
                                result[k] = v
                    
                    result["_hierarchical_results"] = step_results
                    
                    if output_key:
                        result[output_key] = generated_text
                    
                    return result
                
                step_processor = build_llm_processor(
                    engine_config,
                    preprocess=_pre_parallel,
                    postprocess=_post_parallel
                )
                current_ds = step_processor(current_ds)
    
    # Final postprocessing to extract all outputs
    def _final_post(row: Dict[str, Any]) -> Dict[str, Any]:
        """Final postprocessing for hierarchical prompts."""
        step_results = row.get("_hierarchical_results", {})
        
        # Get final answer (last output key or specific key)
        answer = ""
        if step_results:
            # Use last output key as answer
            last_key = list(step_results.keys())[-1]
            answer = step_results[last_key]
        
        # Filter out large/unnecessary columns
        excluded_cols = {
            "image", "image_array", "image_data", "messages", "sampling_params",
            "llm_output", "generated_text", "json", "guided_decoding",
            "_hierarchical_results", "_hierarchical_step", "_hierarchical_output_key",
        }
        
        # Build result with only necessary columns
        result = {}
        
        # Keep essential input columns (paths/URLs, not arrays) and all metadata columns
        for col, val in row.items():
            if col in excluded_cols:
                continue
            # Preserve all metadata columns (including custom metadata from parquet)
            # Exclude only large/complex objects
            if isinstance(val, (str, int, float, type(None))):
                result[col] = val
            elif isinstance(val, dict):
                # Only include if dict contains only simple types
                if all(isinstance(v, (str, int, float, type(None))) for v in val.values()):
                    result[col] = val
            elif isinstance(val, list):
                # Only include if list contains only simple types
                if all(isinstance(v, (str, int, float, type(None))) for v in val):
                    result[col] = val
        
        # Include all intermediate outputs
        result.update(step_results)
        result["answer"] = answer
        result["model_response"] = answer
        
        return result
    
    # Apply final postprocessing
    current_ds = current_ds.map(_final_post)
    
    return current_ds


def _process_decision_tree_prompts(
    ds, cfg: DictConfig, engine_config: vLLMEngineProcessorConfig,
    tree_config: DictConfig, system_prompt: str,
    sampling_params_vqa: Dict[str, Any], is_multimodal: bool,
    structured_output_enabled: bool, json_schema: Optional[Dict[str, Any]]
):
    """Process decision tree prompts by traversing tree based on model responses."""
    from dagspaces.urbanvqa.prompts.decision_tree import DecisionTree
    
    # Load decision tree
    tree_path = getattr(tree_config, "tree_path", None)
    tree_format = getattr(tree_config, "tree_format", "yaml")
    max_depth = getattr(tree_config, "max_depth", 10)
    enable_cycle_detection = getattr(tree_config, "enable_cycle_detection", True)
    
    if not tree_path:
        raise ValueError("decision_tree.tree_path must be specified when decision_tree.enabled=true")
    
    # Resolve path (support relative and absolute paths)
    if not os.path.isabs(tree_path):
        # Try relative to config directory or project root
        possible_paths = [
            tree_path,
            os.path.join("dagspaces/urbanvqa/conf/prompts/decision_trees", os.path.basename(tree_path)),
            os.path.join(os.path.dirname(__file__), "..", "conf", "prompts", "decision_trees", os.path.basename(tree_path)),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                tree_path = path
                break
    
    if tree_format == "yaml":
        tree = DecisionTree.from_yaml(tree_path)
    else:
        tree = DecisionTree.from_json(tree_path)
    
    # Process each row through decision tree traversal
    current_ds = ds
    
    # Initialize tree state for all rows
    def _init_tree_state(row: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize tree traversal state for a row.
        
        CRITICAL: Do NOT include image column - we must not touch images from ray.data.read_images()
        """
        # Build result with only lightweight metadata, excluding image column
        result = {}
        excluded_cols = {"image", "image_array", "image_data", "path", "messages", "sampling_params"}
        for k, v in row.items():
            if k in excluded_cols:
                continue
            # Only include simple, serializable types
            if isinstance(v, (str, int, float, type(None))):
                result[k] = v
            elif isinstance(v, dict):
                if all(isinstance(vv, (str, int, float, type(None))) for vv in v.values()):
                    result[k] = v
            elif isinstance(v, list):
                if all(isinstance(vv, (str, int, float, type(None))) for vv in v):
                    result[k] = v
        
        # Add tree state
        result["_tree_current_node"] = tree.root_node_id
        result["_tree_visited_nodes"] = []
        result["_tree_results"] = {}
        result["_tree_depth"] = 0
        
        return result
    
    # Initialize before processing
    current_ds = current_ds.map(_init_tree_state)
    
    # Traverse tree node by node up to max_depth
    for depth in range(max_depth):
        # Preprocess for current tree node
        def _pre_tree_node(row: Dict[str, Any]) -> Dict[str, Any]:
            """Preprocess for current tree node."""
            _maybe_silence_vllm_logs()
            
            sample_id = row.get("sample_id", "unknown")
            current_node_id = row.get("_tree_current_node")
            
            # Helper function to create lightweight return dict (no image columns)
            def create_lightweight_result(include_messages=True):
                """Create result with only lightweight metadata, no image columns."""
                result = {}
                excluded_cols = {"image", "image_array", "image_data", "path", "messages", "sampling_params"}
                for k, v in row.items():
                    if k in excluded_cols:
                        continue
                    if isinstance(v, (str, int, float, type(None))):
                        result[k] = v
                    elif isinstance(v, dict):
                        if all(isinstance(vv, (str, int, float, type(None))) for vv in v.values()):
                            result[k] = v
                    elif isinstance(v, list):
                        if all(isinstance(vv, (str, int, float, type(None))) for vv in v):
                            result[k] = v
                if include_messages:
                    result["messages"] = []  # Empty messages to skip processing
                return result
            
            if not current_node_id:
                # Tree traversal complete for this row
                return create_lightweight_result(include_messages=True)
            
            node = tree.nodes.get(current_node_id)
            if not node:
                return create_lightweight_result(include_messages=True)
            
            # Check for cycle detection
            visited_nodes = row.get("_tree_visited_nodes", [])
            if enable_cycle_detection and current_node_id in visited_nodes:
                return create_lightweight_result(include_messages=True)
            
            # Handle convergence nodes (many-to-one)
            if node.node_type == "convergence":
                # Get inputs from all predecessor nodes
                convergence_inputs = tree.get_convergence_inputs(
                    current_node_id,
                    row.get("_tree_results", {})
                )
                
                # Aggregate inputs according to strategy
                aggregation_strategy = node.metadata.get("aggregation_strategy", "concatenate") if node.metadata else "concatenate"
                aggregated_text = tree.aggregate_inputs(convergence_inputs, aggregation_strategy)
                
                # Use aggregation prompt if provided
                if node.metadata and node.metadata.get("aggregation_prompt"):
                    node_prompt = node.metadata["aggregation_prompt"]
                    # Replace placeholders in aggregation prompt
                    for key, value in convergence_inputs.items():
                        node_prompt = node_prompt.replace(f"{{{{{key}}}}}", str(value))
                else:
                    # Use node prompt with {{aggregated_inputs}} placeholder
                    node_prompt = node.prompt.replace("{{aggregated_inputs}}", aggregated_text)
                    # Also replace individual input placeholders
                    for key, value in convergence_inputs.items():
                        node_prompt = node_prompt.replace(f"{{{{{key}}}}}", str(value))
            else:
                # Regular node
                node_prompt = node.prompt
                # Replace placeholders with previous results
                tree_results = row.get("_tree_results", {})
                for key, value in tree_results.items():
                    node_prompt = node_prompt.replace(f"{{{{{key}}}}}", str(value))
                
                # Replace user question placeholder
                user_question = row.get("prompt", "")
                node_prompt = node_prompt.replace("{{final_question}}", user_question)
                node_prompt = node_prompt.replace("{{user_question}}", user_question)
            
            # Convert numpy array to base64 (PyArrow-serializable!)
            image_base64_str = None
            
            if is_multimodal:
                if "image" not in row or row["image"] is None:
                    raise ValueError(f"Image is required for multimodal tree node (sample_id: {row.get('sample_id')})")
                
                # Normalize to numpy array before conversion (does not mutate original buffer)
                image_source = _ensure_numpy_image_value(row["image"], row.get("sample_id"))
                
                # _convert_image_to_base64 works on a copy - does not modify original
                image_base64_str = _convert_image_to_base64(image_source, row)
                if not image_base64_str:
                    raise ValueError(f"Failed to convert numpy array to base64 for tree node (sample_id: {row.get('sample_id')})")
            
            # Build messages with base64 string (PyArrow-serializable!)
            if is_multimodal and image_base64_str is not None:
                user_content = [
                    {"type": "text", "text": node_prompt},
                    {"type": "image_url", "image_url": {"url": image_base64_str}}
                ]
            else:
                user_content = node_prompt
            
            sp_local = dict(sampling_params_vqa)
            # Normalize sampling params to ensure stop is a list
            sp_local = _normalize_sampling_params(sp_local)
            if structured_output_enabled and json_schema:
                try:
                    schema_json_str = json.dumps(json_schema, ensure_ascii=False)
                    sp_local["guided_decoding"] = {"json": schema_json_str}
                except Exception:
                    pass
            
            # CRITICAL: Only return messages, sampling_params, and lightweight metadata
            # Do NOT include image columns or complex objects
            result = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "sampling_params": sp_local,
            }
            
            # Preserve all lightweight, serializable metadata columns (including custom metadata from parquet)
            # Exclude image arrays, PIL Images, and complex objects
            excluded_cols = {"image", "image_array", "image_data", "messages", "sampling_params", "path"}
            lightweight_metadata = {}
            for key, val in row.items():
                if key in excluded_cols:
                    continue
                # Only include if it's a simple type (not PIL Image or complex object)
                if isinstance(val, (str, int, float, type(None))):
                    lightweight_metadata[key] = val
                elif isinstance(val, dict):
                    # Check if dict contains only simple types
                    if all(isinstance(v, (str, int, float, type(None))) for v in val.values()):
                        lightweight_metadata[key] = val
                elif isinstance(val, list):
                    # Check if list contains only simple types
                    if all(isinstance(v, (str, int, float, type(None))) for v in val):
                        lightweight_metadata[key] = val
            
            # Add tree node metadata
            lightweight_metadata["_tree_node_id"] = current_node_id
            lightweight_metadata["_tree_output_key"] = node.output_key
            
            result.update(lightweight_metadata)
            
            # CRITICAL: Explicitly ensure image column is NOT in result
            result.pop("image", None)
            result.pop("image_array", None)
            result.pop("image_data", None)
            result.pop("path", None)
            
            return result
        
        # Postprocess tree node response and determine next node
        def _post_tree_node(row: Dict[str, Any]) -> Dict[str, Any]:
            """Postprocess tree node response and determine next node."""
            generated_text = str(row.get("generated_text", "")).strip()
            node_id = row.get("_tree_node_id")
            
            if not node_id:
                # Return lightweight result (no image columns)
                result = {}
                excluded_cols = {"image", "image_array", "image_data", "path", "messages", "sampling_params"}
                for k, v in row.items():
                    if k.startswith("_tree") or k in excluded_cols:
                        continue
                    if isinstance(v, (str, int, float, type(None))):
                        result[k] = v
                    elif isinstance(v, dict):
                        if all(isinstance(vv, (str, int, float, type(None))) for vv in v.values()):
                            result[k] = v
                    elif isinstance(v, list):
                        if all(isinstance(vv, (str, int, float, type(None))) for vv in v):
                            result[k] = v
                # Preserve tree state
                for k in ["_tree_current_node", "_tree_visited_nodes", "_tree_results", "_tree_depth"]:
                    if k in row:
                        result[k] = row[k]
                return result
            
            # Get current state
            tree_results = row.get("_tree_results", {})
            visited_nodes = row.get("_tree_visited_nodes", [])
            current_node_id = row.get("_tree_current_node")
            depth = row.get("_tree_depth", 0)
            
            # Store result
            output_key = row.get("_tree_output_key")
            if output_key:
                tree_results[output_key] = generated_text
            
            # Update visited nodes
            visited_nodes = visited_nodes + [current_node_id]
            
            # Get next node based on response
            node = tree.nodes.get(current_node_id)
            next_node_id = None
            
            if node and node.node_type != "leaf":
                # Determine next node based on response
                next_node_id = tree.get_next_node(
                    current_node_id,
                    generated_text,
                    context={}  # Could include confidence, etc.
                )
            
            # Build result
            # Filter out large/unnecessary columns (including image arrays and PIL Images)
            excluded_cols = {
                "image", "image_array", "image_data", "path",  # Image data
                "messages", "sampling_params",  # Internal processing
                "llm_output", "generated_text",  # Already extracted
                "json", "guided_decoding",  # Internal config
            }
            
            result = {}
            # Preserve all lightweight, serializable metadata columns (including custom metadata from parquet)
            # Exclude image arrays, PIL Images, and complex objects
            for k, v in row.items():
                if k.startswith("_tree") or k in excluded_cols:
                    continue
                # Only include if it's a simple type (not PIL Image, numpy array, or complex object)
                if isinstance(v, (str, int, float, type(None))):
                    result[k] = v
                elif isinstance(v, dict):
                    # Check if dict contains only simple types
                    if all(isinstance(vv, (str, int, float, type(None))) for vv in v.values()):
                        result[k] = v
                elif isinstance(v, list):
                    # Check if list contains only simple types
                    if all(isinstance(vv, (str, int, float, type(None))) for vv in v):
                        result[k] = v
            
            result["_tree_current_node"] = next_node_id
            result["_tree_visited_nodes"] = visited_nodes
            result["_tree_results"] = tree_results
            result["_tree_depth"] = depth + 1
            
            # Add output key to result
            if output_key:
                result[output_key] = generated_text
            
            return result
        
        # Process this depth level
        step_processor = build_llm_processor(
            engine_config,
            preprocess=_pre_tree_node,
            postprocess=_post_tree_node
        )
        
        current_ds = step_processor(current_ds)
        
        # Check if all rows have completed traversal
        # (In a full implementation, we'd check if all rows have reached leaf nodes)
        # For now, continue for max_depth iterations
    
    # Final postprocessing to extract all outputs
    def _final_post_tree(row: Dict[str, Any]) -> Dict[str, Any]:
        """Final postprocessing for decision tree prompts."""
        tree_results = row.get("_tree_results", {})
        
        # Get final answer (last output key or specific key)
        answer = ""
        if tree_results:
            # Use last output key as answer
            last_key = list(tree_results.keys())[-1]
            answer = tree_results[last_key]
        
        # Filter out large/unnecessary columns
        excluded_cols = {
            "image", "image_array", "image_data", "messages", "sampling_params",
            "llm_output", "generated_text", "json", "guided_decoding",
            "_tree_results", "_tree_node_id", "_tree_output_key", "_tree_visited_nodes",
        }
        
        # Build result with only necessary columns
        result = {}
        
        # Keep essential input columns (paths/URLs, not arrays) and all metadata columns
        for col, val in row.items():
            if col in excluded_cols:
                continue
            # Preserve all metadata columns (including custom metadata from parquet)
            # Exclude only large/complex objects
            if isinstance(val, (str, int, float, type(None))):
                result[col] = val
            elif isinstance(val, dict):
                # Only include if dict contains only simple types
                if all(isinstance(v, (str, int, float, type(None))) for v in val.values()):
                    result[col] = val
            elif isinstance(val, list):
                # Only include if list contains only simple types
                if all(isinstance(v, (str, int, float, type(None))) for v in val):
                    result[col] = val
        
        # Include all intermediate outputs
        result.update(tree_results)
        result["answer"] = answer
        result["model_response"] = answer
        result["metadata"] = {
            "tree_id": tree.tree_id,
            "version": tree.version,
            "nodes_visited": row.get("_tree_visited_nodes", []),
            "depth": row.get("_tree_depth", 0),
        }
        
        return result
    
    # Apply final postprocessing
    current_ds = current_ds.map(_final_post_tree)
    
    return current_ds

