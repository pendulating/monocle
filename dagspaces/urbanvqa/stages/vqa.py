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
import copy
from io import BytesIO
from pathlib import Path
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
    from ray.data.llm import build_processor, vLLMEngineProcessorConfig  # type: ignore
    # Alias for call sites that haven't been updated yet
    build_llm_processor = build_processor
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
_DEFAULT_IMAGE_PROMPT = "What do you see in this image?"
_DEBUG_PREVIEW_LIMIT = 5
_DEBUG_PREVIEW_COUNTER = 0

# Suppress multiprocessing resource tracker warnings at module import time
# and install the resilient tracker patch for all processes.
try:
    import warnings
    import os
    from ..resource_tracker_patch import apply_patch as _apply_resource_tracker_patch

    _apply_resource_tracker_patch()

    if "PYTHONWARNINGS" not in os.environ:
        os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"

    warnings.filterwarnings(
        "ignore",
        message="resource_tracker: process died unexpectedly",
        category=UserWarning,
        module="multiprocessing.resource_tracker",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*resource_tracker.*",
        category=UserWarning,
        module="multiprocessing",
    )
except Exception:
    pass


def _debug_log(event: str, payload: Dict[str, Any], cfg: Optional[DictConfig] = None, force: bool = False) -> None:
    """Emit structured debug logs when runtime.debug is enabled."""
    try:
        if not force:
            if cfg is not None:
                runtime_cfg = getattr(cfg, "runtime", None)
                if runtime_cfg is not None and not getattr(runtime_cfg, "debug", False):
                    return
        payload_out = {}
        for k, v in payload.items():
            if isinstance(v, np.ndarray):
                payload_out[k] = "<ndarray>"
            else:
                payload_out[k] = v
        print(
            json.dumps(
                {
                    "vqa_debug": {
                        "event": event,
                        "payload": payload_out,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                }
            ),
            flush=True,
        )
    except Exception:
        pass


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


def _sanitize_identifier(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
    else:
        candidate = str(value).strip()
    if not candidate or candidate.lower() in {"nan", "none"}:
        return None
    return candidate


def _normalize_path_value(path_val: Any) -> Optional[str]:
    if path_val is None:
        return None
    path_str = str(path_val).strip()
    return path_str or None


def _derive_sample_id_from_path(path_val: Optional[str]) -> Optional[str]:
    if not path_val:
        return None
    base_name = os.path.basename(path_val.rstrip("/"))
    if not base_name:
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", path_val.strip("/"))
        return sanitized or None
    stem, _ = os.path.splitext(base_name)
    candidate = stem or base_name
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", candidate)
    return sanitized or None


def _resolve_row_sample_id(row: Dict[str, Any]) -> Optional[str]:
    existing = _sanitize_identifier(row.get("sample_id"))
    if existing:
        return existing
    for key in ("image_path", "path"):
        sample = _derive_sample_id_from_path(_normalize_path_value(row.get(key)))
        if sample:
            return sample
    return None


def _sanitize_prompt_value(value: Any, cfg: DictConfig) -> str:
    if value is None:
        return _resolve_default_prompt(cfg)
    if isinstance(value, str):
        sanitized = value.strip()
    else:
        sanitized = str(value).strip()
    return sanitized or _resolve_default_prompt(cfg)


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


def _is_cambrian_model(model_source: str) -> bool:
    """Detect if model is a Cambrian model.
    
    Args:
        model_source: Model path/name
        
    Returns:
        True if Cambrian model detected
    """
    model_lower = str(model_source).lower()
    return "cambrian" in model_lower


def _is_multimodal_model(model_source: str, cfg: Optional[Any] = None) -> bool:
    """Detect if model supports multimodal inputs.
    
    Checks:
    1. Model name patterns (e.g., "Qwen2.5-VL", "InternVL", "Phi-3.5-vision", "Cambrian")
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
        r"cambrian",
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


# Import shared multiprocessing utilities
from ..multiprocessing_utils import get_suppress_child_warnings


def _suppress_multiprocessing_warnings(cfg: Optional[DictConfig] = None) -> None:
    """Suppress multiprocessing resource tracker warnings that occur during worker shutdown.
    
    Args:
        cfg: Optional configuration object. If provided, checks runtime.suppress_child_warnings.
            If not provided or config is True, suppresses warnings.
    
    These warnings are harmless and occur when worker processes terminate unexpectedly
    during Ray Data MapBatches operations. The resource tracker tries to clean up
    resources that were already cleaned up or never registered, causing UserWarnings
    and KeyError exceptions that don't impact functionality.
    """
    suppress = get_suppress_child_warnings(cfg)
    if suppress:
        # Set environment variable and call worker setup hook to ensure warnings are suppressed
        import os
        os.environ["URBANVQA_SUPPRESS_WARNINGS"] = "true"
        from ..multiprocessing_utils import _worker_process_setup_hook
        _worker_process_setup_hook()
    else:
        # Ensure environment variable is set to false if suppression is disabled
        import os
        os.environ["URBANVQA_SUPPRESS_WARNINGS"] = "false"


def _ensure_ray_init(cfg) -> None:
    """Initialize Ray with SLURM-aware CPU and memory limits.

    Delegates to the unified ``ensure_ray_init`` in ``multiprocessing_utils``
    which also applies ``DataContext`` resource limits after ``ray.init()``.
    """
    try:
        from ..multiprocessing_utils import ensure_ray_init
        ensure_ray_init(cfg, caller="run_vqa_stage")
    except Exception:
        # Minimal fallback
        try:
            import ray  # type: ignore
            if not ray.is_initialized():
                ray.init(log_to_driver=True)
        except Exception:
            pass


def _apply_ray_data_resource_limits(cfg) -> None:
    """Apply Ray Data execution resource limits to the current DataContext.

    This function is IDEMPOTENT and safe to call whether or not Ray was
    initialised by a different caller (e.g. the orchestrator).  It reads
    the object-store size from the same 3-priority hierarchy used by
    ``_ensure_ray_init`` and caps the streaming-executor budget at 80% of
    that value via **both** ``ctx.execution_options.resource_limits`` and
    ``ctx.override_object_store_memory_limit_fraction``.

    Both knobs are required because ``ResourceManager.get_global_limits()``
    computes::

        effective = MIN(resource_limits, physical_store * fraction)

    Setting only ``resource_limits`` leaves the fraction at 0.5, so the
    effective budget is capped at ``physical_store * 0.5`` regardless of
    what we request.  Setting the fraction ensures the denominator in the
    MIN is at least as large as our ``resource_limits`` value.

    Must be called **after** ``ray.init()`` and **before** any dataset
    materialisation (``processor(ds)``, ``ds.iter_batches()``, etc.).
    """
    try:
        import ray  # type: ignore
        if not ray.is_initialized():
            return  # nothing to configure yet

        # ── Detect SLURM CPU allocation ──────────────────────────────────
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

        # ── Object store size (3-priority hierarchy) ─────────────────────
        obj_store_bytes = None

        # Priority 1: RAY_OBJECT_STORE_MEMORY env var (absolute bytes)
        try:
            _env_obj_store = os.environ.get("RAY_OBJECT_STORE_MEMORY")
            if _env_obj_store is not None and str(_env_obj_store).strip():
                obj_store_bytes = int(str(_env_obj_store).strip())
        except Exception:
            pass

        # Priority 2: cfg.runtime.object_store_proportion (0.0-1.0)
        if obj_store_bytes is None:
            try:
                prop = getattr(cfg.runtime, "object_store_proportion", None)
                prop = float(prop) if prop is not None else None
            except Exception:
                prop = None
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
                    if total_bytes and total_bytes > 0:
                        obj_store_bytes = int(total_bytes * float(prop))
                except Exception:
                    obj_store_bytes = None

        # Priority 3: fallback to 30% of job_memory_gb
        if obj_store_bytes is None:
            try:
                job_mem_gb = int(getattr(cfg.runtime, "job_memory_gb", 64) or 64)
            except Exception:
                job_mem_gb = 64
            try:
                obj_store_bytes = int(max(1, job_mem_gb) * (1024 ** 3) * 0.30)
            except Exception:
                obj_store_bytes = int(64 * (1024 ** 3) * 0.30)

        # ── Apply limits to DataContext ───────────────────────────────────
        ctx = ray.data.DataContext.get_current()
        desired_budget = None
        limits_kwargs = {}
        if cpus_alloc is not None and int(cpus_alloc) > 0:
            limits_kwargs["cpu"] = int(cpus_alloc)
        if obj_store_bytes is not None and int(obj_store_bytes) > 0:
            # Use 80 % of the allocated object store for the streaming
            # pipeline (up from the default ~50 %).  The remaining 20 %
            # is headroom for non-Data Ray objects (e.g. actor refs).
            desired_budget = int(int(obj_store_bytes) * 0.80)
            limits_kwargs["object_store_memory"] = desired_budget
        if limits_kwargs:
            ctx.execution_options.resource_limits = (
                ctx.execution_options.resource_limits.copy(**limits_kwargs)
            )

        # ── Override the object-store fraction ────────────────────────────
        # ResourceManager.get_global_limits() computes:
        #     effective = MIN(resource_limits, physical_store * fraction)
        # The physical object store size is reported by Ray cluster_resources
        # as "object_store_memory".  We need `fraction` high enough so that
        # `physical_store * fraction >= desired_budget`.
        # We query the physical size and compute the needed fraction, capping
        # at 0.95 for safety.  If the query fails, 0.85 is a safe default.
        _fraction = 0.85  # sensible default
        try:
            cluster_res = ray.cluster_resources()
            physical_obj_store = cluster_res.get("object_store_memory", 0)
            if physical_obj_store > 0 and desired_budget is not None and desired_budget > 0:
                # fraction such that physical * fraction >= desired_budget,
                # with a 5 pp margin so the MIN doesn't clip us.
                _fraction = min(0.95, max(0.5, desired_budget / physical_obj_store + 0.05))
        except Exception:
            pass
        ctx.override_object_store_memory_limit_fraction = _fraction

        _obj_budget_gb = round(limits_kwargs.get("object_store_memory", 0) / (1024 ** 3), 2) if limits_kwargs.get("object_store_memory") else "unset"
        print(
            f"[_apply_ray_data_resource_limits] "
            f"object_store_memory budget={_obj_budget_gb} GB, "
            f"cpu={limits_kwargs.get('cpu', 'unset')}, "
            f"obj_store_fraction={_fraction:.2f}, "
            f"source={'RAY_OBJECT_STORE_MEMORY env' if os.environ.get('RAY_OBJECT_STORE_MEMORY') else 'heuristic'}",
            flush=True,
        )

        # ── Block-size tuning ─────────────────────────────────────────────
        # For multimodal / image-heavy workloads, each row can be 3–32 MB
        # (decoded RGB pixels).  The default 128 MB cap fragments blocks to
        # 4–40 rows, far below vLLM's batch_size of 64.  map_batches only
        # SPLITS blocks, never MERGES, so small blocks → small vLLM batches
        # → wasted GPU cycles.
        #
        # We set a large target (2 GB) so that every operator's
        # BlockOutputBuffer keeps at least ~64 image rows per block.
        # With 3 MB images → ~682 rows/block; with 32 MB images → ~64.
        # Memory safety is still governed by the resource_limits budget
        # above, not by block-size splitting.
        #
        # NOTE: Do NOT use None here — that disables splitting entirely,
        # causing _load_images_batch to accumulate millions of decoded
        # images into a single block and OOM / hang.
        ctx.target_max_block_size = 2 * 1024 * 1024 * 1024  # 2 GB
        ctx.target_min_block_size = 1 * 1024 * 1024  # 1 MB (merge tiny blocks)

        # ── Silence noisy high-memory issue-detector warnings ──────────
        # Ray Data's issue detectors warn about per-task memory usage for
        # Map(_preprocess), ChatTemplateUDF, and vLLMEngineStageUDF.
        # These are expected for image-heavy workloads and just clutter
        # the logs.  Disable the detectors entirely.
        try:
            ctx.issue_detectors_config.detectors = []
        except Exception:
            pass

    except Exception as exc:
        print(f"[_apply_ray_data_resource_limits] Warning: {exc}", flush=True)


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

    .. deprecated::
        No longer used in the primary preprocessing path.
        ``preprocess_simple`` now passes PIL Images directly via
        ``{"type": "image", "image": pil_img}`` in messages, which Ray Data
        LLM's PrepareImageStage forwards to vLLM with zero encode/decode
        overhead.  Kept for backward compatibility with non-standard callers.

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


def _ensure_json_schema_dict(schema: Any) -> Optional[Dict[str, Any]]:
    """Convert OmegaConf/DictConfig schemas into a plain Python dict."""
    if schema is None:
        return None
    if isinstance(schema, DictConfig):
        from omegaconf import OmegaConf

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

    .. deprecated::
        The primary preprocessing path (``preprocess_simple``) now passes PIL
        Images directly via ``{"type": "image", "image": pil_img}``.  This
        function is only still called by the advanced prompting techniques in
        ``techniques.py`` (CoT, ReAct, etc.) and the hierarchical / decision-tree
        code paths.  It can be migrated to the PIL passthrough in a future PR.

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
    # Suppress multiprocessing resource tracker warnings early
    _suppress_multiprocessing_warnings(cfg)
    
    # Ensure Ray is initialized
    _ensure_ray_init(cfg)
    
    # ── Apply Ray Data resource limits unconditionally ────────────────
    # _ensure_ray_init may be a no-op if the orchestrator already called
    # _ensure_ray_init_with_cpu_limits (which does NOT set resource_limits).
    # _apply_ray_data_resource_limits is idempotent and ensures the
    # streaming executor budget is always configured, regardless of who
    # started Ray.
    _apply_ray_data_resource_limits(cfg)
    
    # Enable fallback to Arrow object extension types for PIL Images and other complex objects
    # This allows Ray Data to handle PIL Images in messages structure without Arrow conversion errors
    if _RAY_OK:
        try:
            from ray.data import DataContext
            ctx = DataContext.get_current()
            ctx.enable_fallback_to_arrow_object_ext_type = True
        except Exception:
            pass  # Continue if DataContext not available
    
    # 1. Detect prompt override
    user_template = getattr(cfg.prompt, "user_template", "")
    has_override = user_template and user_template != "{{prompt}}"
    if has_override:
        logging.info(f"Using prompt override from config: {user_template[:100]}...")

    # Handle Ray Dataset input if provided
    is_ray_ds = hasattr(df, "map_batches") and hasattr(df, "count") and _RAY_OK
    
    if not is_ray_ds:
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["sample_id", "prompt", "answer"])
        
        # Apply prompt override to Pandas DataFrame if not already using streaming
        if has_override:
            df["prompt"] = user_template

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
    
    try:
        is_ray_dataset = hasattr(ds, "take")
        preview_payload: Dict[str, Any] = {
            "is_ray_dataset": is_ray_dataset,
            "object_type": type(ds).__name__,
            "has_prompt_override": has_override,
            "overridden_prompt_preview": user_template[:100] if has_override else None
        }
        if is_ray_dataset:
            try:
                sample_preview = ds.take(1)
                sanitized_sample = []
                for item in sample_preview:
                    sanitized_sample.append(
                        {
                            k: "<ndarray>" if isinstance(v, np.ndarray) else v
                            for k, v in item.items()
                        }
                    )
                preview_payload["sample"] = sanitized_sample
            except Exception as exc:
                preview_payload["sample_error"] = str(exc)
        else:
            preview_payload["rows"] = len(ds) if hasattr(ds, "__len__") else None
        _debug_log("vqa_stage_dataset_ready", preview_payload, cfg, force=True)
    except Exception:
        pass
    
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

    json_schema = _ensure_json_schema_dict(json_schema)
    if structured_output_enabled and not json_schema:
        structured_output_enabled = False
    guided_decoding_payload = _build_guided_decoding_config(json_schema)
    
    # Resolve model path
    model_source_raw = getattr(cfg.model, "model_source", "")
    resolved_model_source = _resolve_model_path(model_source_raw)
    is_multimodal = _is_multimodal_model(resolved_model_source, cfg)
    
    # Check if this is a Cambrian model requiring specialized handling
    if _is_cambrian_model(resolved_model_source):
        # Cambrian-13B is not Transformers-native and requires a persistent processor with a sidecar.
        # We delegate the entire inference task to PersistentCambrianProcessor.
        try:
            from .persistent_cambrian import PersistentCambrianProcessor
            
            # If input is already a Ray Dataset, materialize it to pandas as required by Cambrian processor
            if is_ray_ds:
                # SAFETY: If the dataset has an 'image' column (decoded pixels), we MUST drop it
                # before calling to_pandas() to avoid a fatal OOM on large datasets.
                # The Cambrian sidecar will load images lazily from paths.
                if "image" in ds.schema().names:
                    logging.info("Dropping 'image' column from Ray Dataset before pandas conversion for Cambrian")
                    ds = ds.drop_columns(["image"])
                df_input = ds.to_pandas()
                
                # Apply prompt override to the materialized DataFrame
                if has_override:
                    logging.info("Applying prompt override to materialized Cambrian input")
                    df_input["prompt"] = user_template
            else:
                df_input = df
                # Prompt override already applied to df above for non-ray-ds case
            
            # Pass prompts from config if provided (supports prompt_override_path)
            system_prompt = getattr(cfg.prompt, "system", "")
            user_template = getattr(cfg.prompt, "user_template", "")
            
            # If user_template is just "{{prompt}}", we let the sidecar use row["prompt"]
            if user_template == "{{prompt}}":
                user_template = ""
            
            # Initialize and run Cambrian processor
            processor = PersistentCambrianProcessor.get_or_create(cfg)
            results_df = processor.evaluate(
                df_input, 
                system_prompt=system_prompt,
                user_template=user_template,
                cfg=cfg
            )
            
            # Re-wrap in Ray Dataset if we started with one (for pipeline compatibility)
            if is_ray_ds:
                return ray.data.from_pandas(results_df)
            return results_df
            
        except Exception as e:
            import traceback
            logging.error(f"Failed to run Cambrian inference: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"Cambrian inference failed: {e}") from e
    
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
    
    # Prefer explicit config batch_size; fall back to GPU-aware default only when
    # the config does not specify one (OmegaConf raises on missing keys, so use
    # getattr with a sentinel to distinguish "set" from "absent").
    _cfg_batch = getattr(cfg.model, "batch_size", None)
    batch_size = _cfg_batch if _cfg_batch is not None else gpu_settings.get("batch_size", 16)
    concurrency = getattr(cfg.model, "concurrency", 1)
    
    # Get tensor parallelism size
    tp_val = engine_kwargs.get("tensor_parallel_size", 1)
    if tp_val > 1:
        # Adjust concurrency based on tensor parallelism
        # concurrency should match number of model replicas, not total GPU count
        concurrency = max(1, num_gpus // tp_val)
    
    # Runtime environment for vLLM engine
    # NOTE: worker_process_setup_hook is set at ray.init() time, not here
    # to avoid serialization issues when passing runtime_env to vLLMEngineProcessorConfig
    runtime_env_vars = {}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        runtime_env_vars["HF_TOKEN"] = hf_token
    
    # Conditionally add warning suppression env var based on config
    suppress_warnings = get_suppress_child_warnings(cfg)
    if suppress_warnings:
        runtime_env_vars["URBANVQA_SUPPRESS_WARNINGS"] = "true"
    else:
        runtime_env_vars["URBANVQA_SUPPRESS_WARNINGS"] = "false"
    
    # Build runtime_env with only env_vars (no worker_process_setup_hook)
    # The hook is set at ray.init() time in _ensure_ray_init()
    runtime_env = {}
    if runtime_env_vars:
        runtime_env["env_vars"] = runtime_env_vars
    
    accelerator_type = getattr(cfg.model, "accelerator_type", None)
    
    # ── CPU-stage concurrency ────────────────────────────────────────
    # ChatTemplateUDF loads the Qwen2VL image processor and is the
    # fastest stage in the pipeline (~45 img/s per actor).  vLLM is the
    # slowest (~6-7 img/s per GPU).  With 4 ChatTemplate actors vs 4
    # vLLM actors the 7× production surplus fills the object store to
    # 80+ GiB during the ~2 min CUDA-graph-capture phase (when vLLM
    # consumes zero rows).  2 actors still produce ~90 img/s — well
    # above the ~25-30 img/s vLLM steady-state rate — but halve the
    # burst.  DetokenizeUDF is lightweight text work; 2 actors is plenty.
    cpu_stage_pool = max(1, concurrency // 4)  # 1 for 4 GPUs, 2 for 8
    # Use an explicit (min, max) tuple for concurrency to pin the actor
    # pool size exactly.  Passing a bare int N causes Ray Data to create
    # an autoscaling pool (1, 2*N) for CPU stages, which triggers the
    # "configured utilization threshold couldn't be reached" warning and
    # leads to actor churn.
    engine_config = vLLMEngineProcessorConfig(
        model_source=resolved_model_source,
        engine_kwargs=engine_kwargs,
        concurrency=(concurrency, concurrency),
        batch_size=batch_size,
        # NOTE: Do NOT set experimental.max_tasks_in_flight_per_actor.
        # The default (4) via DEFAULT_MAX_TASKS_IN_FLIGHT is correct.
        # Setting it to 8 to "match max_concurrent_batches" was tried
        # and HURT throughput: it queued 8 batches per actor during CUDA
        # capture, producing 20-73 s first-batch latencies and increasing
        # pipeline buffering without improving GPU utilisation (vLLM's
        # continuous-batching scheduler saturates the GPU regardless of
        # how many Ray Data tasks are dispatched).
        #
        # ── CPU-stage configs ─────────────────────────────────────────
        # Fixed tuple (N, N) prevents the autoscaler from tearing down
        # and respawning ChatTemplateUDF / DetokenizeUDF actors.
        chat_template_stage={"concurrency": (cpu_stage_pool, cpu_stage_pool)},
        tokenize_stage=False,
        detokenize_stage={"concurrency": (cpu_stage_pool, cpu_stage_pool)},
        # PrepareImageStage disabled — images flow via messages dict.
        # Use the non-deprecated field (has_image triggers a warning).
        prepare_image_stage=False,
        accelerator_type=accelerator_type,
        runtime_env=runtime_env if runtime_env else None,
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
        
        _suppress_multiprocessing_warnings(cfg)
        _maybe_silence_vllm_logs()
        
        # Use unified preprocessing framework
        from dagspaces.urbanvqa.prompts.unified import unified_preprocess
        
        row_values = dict(row)
        # Ensure prompt and sample identifiers are available for downstream processing
        row_values["prompt"] = _sanitize_prompt_value(row_values.get("prompt"), cfg)
        resolved_sample_id = _resolve_row_sample_id(row_values)
        if resolved_sample_id is not None:
            row_values["sample_id"] = resolved_sample_id

        global _DEBUG_PREVIEW_COUNTER
        if getattr(getattr(cfg, "runtime", None), "debug", False) and _DEBUG_PREVIEW_COUNTER < _DEBUG_PREVIEW_LIMIT:
            _DEBUG_PREVIEW_COUNTER += 1
            _debug_log(
                "pre_row_input",
                {
                    key: row_values.get(key)
                    for key in ("sample_id", "prompt", "image_path", "path")
                },
                cfg,
            )
        
        unified_result = unified_preprocess(
            row_values, cfg, is_multimodal,
            hierarchical_enabled, decision_tree_enabled
        )
        
        # If unified preprocessing returns None, it means structural techniques handle it
        # This shouldn't happen in standard flow, but handle gracefully
        if unified_result is None:
            # CRITICAL: Filter out image columns completely - we must NOT touch images from ray.data.read_images()
            # preprocess_simple will read from row["image"] but we don't include it in lightweight_row
            lightweight_row = {}
            excluded_cols = {"image", "image_array", "image_data", "path", "messages", "sampling_params"}
            for k, v in row_values.items():
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
            lightweight_row["prompt"] = str(row_values.get("prompt", "")).strip()
            # preprocess_simple returns messages, sampling_params, and
            # the "image" column (list of PIL Images) for vLLMEngineStage.
            result = preprocess_simple(row_values, cfg, is_multimodal)
            return result
        
        # If unified preprocessing returns messages, use them
        if "messages" in unified_result:
            # Add structured output if enabled
            sp_local = dict(unified_result.get("sampling_params", sampling_params_vqa))
            # Normalize sampling params to ensure stop is a list
            sp_local = _normalize_sampling_params(sp_local)
            if guided_decoding_payload:
                try:
                    sp_local["guided_decoding"] = copy.deepcopy(guided_decoding_payload)
                except Exception:
                    sp_local["guided_decoding"] = guided_decoding_payload
            
            # Return messages, sampling_params, the image column (for
            # vLLMEngineStage), and lightweight metadata.
            result = {
                "messages": unified_result["messages"],
                "sampling_params": sp_local,
            }

            # Carry through the image column from preprocess_simple.
            # With PrepareImageStage disabled (prepare_image_stage=False), this is
            # the ONLY path for PIL Images to reach vLLMEngineStage.
            if "image" in unified_result:
                result["image"] = unified_result["image"]

            # Preserve ALL lightweight, serializable metadata (strings, numbers)
            # These will be preserved through the pipeline and available in postprocess.
            # This ensures columns like image_path, recording_id, face, latitude,
            # longitude, etc. survive preprocessing and appear in the output
            # DataFrame / wandb table.
            _excluded_metadata = {"image", "image_array", "image_data", "path",
                                  "messages", "sampling_params"}
            for key, val in row_values.items():
                if key in _excluded_metadata:
                    continue
                if isinstance(val, (str, int, float, type(None))):
                    result[key] = val
                elif isinstance(val, dict):
                    if all(isinstance(vv, (str, int, float, type(None))) for vv in val.values()):
                        result[key] = val
                elif isinstance(val, list):
                    if all(isinstance(vv, (str, int, float, type(None))) for vv in val):
                        result[key] = val
            
            # Add timestamp as metadata
            result["ts_start"] = datetime.now().timestamp()
            
            return result
        
        # Fallback: use simple preprocessing.
        # preprocess_simple returns messages, sampling_params, and the
        # "image" column (list of PIL Images) for vLLMEngineStage.
        result = preprocess_simple(row_values, cfg, is_multimodal)
        return result
    
    # Postprocessing function - use unified framework
    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess VQA response using unified framework."""
        from dagspaces.urbanvqa.prompts.unified import unified_postprocess
        
        # Use unified postprocessing
        unified_result = unified_postprocess(
            row, cfg, hierarchical_enabled, decision_tree_enabled
        )

        if getattr(getattr(cfg, "runtime", None), "debug", False):
            _debug_log(
                "post_row_output",
                {
                    "sample_id": row.get("sample_id"),
                    "generated_text": row.get("generated_text"),
                    "usage": row.get("usage"),
                },
                cfg,
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
            structured_output_enabled, json_schema, guided_decoding_payload
        )
    elif hierarchical_enabled and hierarchical_steps:
        # Use hierarchical processing
        ds_results = _process_hierarchical_prompts(
            ds, cfg, engine_config, hierarchical_steps,
            system_prompt, sampling_params_vqa, is_multimodal,
            structured_output_enabled, json_schema, guided_decoding_payload
        )
    else:
        # Use standard processing
        # BEST PRACTICE: According to Ray Data LLM docs, preprocess should return ONLY messages and sampling_params
        # The preprocess function completely replaces the row - Ray Data LLM does NOT merge with original columns
        #
        # _pre converts the row's numpy image → PIL and builds messages.
        # When images are already loaded via map_batches in the orchestrator,
        # _pre is lightweight (~1 MB working set per row).  The fallback
        # path loads from image_path on disk, which is heavier but still
        # bounded by the block size from the upstream operator.
        #
        # preprocess_map_kwargs tells Ray Data how much memory each
        # preprocess task uses, preventing the scheduler from launching
        # too many concurrent tasks and causing OOM / spilling.
        processor = build_processor(
            engine_config,
            preprocess=_pre,
            postprocess=_post,
            preprocess_map_kwargs={"num_cpus": 0.5},
        )
        _debug_log(
            "processor_built",
            {
                "engine_has_image": False,  # PrepareImageStage disabled; images flow via result["image"]
                "engine_concurrency": concurrency,
                "engine_batch_size": batch_size,
            },
            cfg,
            force=True,
        )
        ds_results = processor(ds)
        debug_enabled = getattr(getattr(cfg, "runtime", None), "debug", False)
        if debug_enabled and not is_ray_ds:
            try:
                preview = ds_results.take(1)
                sanitized = []
                for item in preview:
                    sanitized.append(
                        {
                            k: "<ndarray>" if isinstance(v, np.ndarray) else v
                            for k, v in item.items()
                        }
                    )
                _debug_log("post_processor_preview", {"preview": sanitized}, cfg)
            except Exception as exc:
                _debug_log("post_processor_preview_error", {"error": str(exc)}, cfg)
        elif debug_enabled and is_ray_ds:
            _debug_log("post_processor_preview_skipped_streaming", {}, cfg)
    
    # Convert back to pandas if needed
    if is_ray_ds:
        if getattr(getattr(cfg, "runtime", None), "debug", False):
            _debug_log("ray_results_count_skipped_streaming", {}, cfg, force=True)
        return ds_results
    
    # Materialize and convert to pandas
    df_results = ds_results.to_pandas()
    _debug_log("final_dataframe", {"rows": len(df_results), "columns": list(df_results.columns)}, cfg, force=True)
    return df_results


def _process_hierarchical_prompts(
    ds, cfg: DictConfig, engine_config: vLLMEngineProcessorConfig,
    steps: List[Dict[str, Any]], system_prompt: str,
    sampling_params_vqa: Dict[str, Any], is_multimodal: bool,
    structured_output_enabled: bool, json_schema: Optional[Dict[str, Any]],
    guided_decoding_payload: Optional[Dict[str, Any]]
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
                if guided_decoding_payload:
                    try:
                        sp_local["guided_decoding"] = copy.deepcopy(guided_decoding_payload)
                    except Exception:
                        sp_local["guided_decoding"] = guided_decoding_payload
                
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
            step_processor = build_processor(
                engine_config,
                preprocess=_pre_hierarchical,
                postprocess=_post_hierarchical,
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
                    if guided_decoding_payload:
                        try:
                            sp_local["guided_decoding"] = copy.deepcopy(guided_decoding_payload)
                        except Exception:
                            sp_local["guided_decoding"] = guided_decoding_payload
                    
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
                
                step_processor = build_processor(
                    engine_config,
                    preprocess=_pre_parallel,
                    postprocess=_post_parallel,
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
    structured_output_enabled: bool, json_schema: Optional[Dict[str, Any]],
    guided_decoding_payload: Optional[Dict[str, Any]]
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
            if guided_decoding_payload:
                try:
                    sp_local["guided_decoding"] = copy.deepcopy(guided_decoding_payload)
                except Exception:
                    sp_local["guided_decoding"] = guided_decoding_payload
            
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
        step_processor = build_processor(
            engine_config,
            preprocess=_pre_tree_node,
            postprocess=_post_tree_node,
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

