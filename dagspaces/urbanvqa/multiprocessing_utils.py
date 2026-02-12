"""Shared utilities for Ray initialisation and multiprocessing warnings."""

import os
import re
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

from .resource_tracker_patch import apply_patch as _apply_resource_tracker_patch


def _worker_process_setup_hook() -> None:
    """Setup function for Ray worker processes to suppress multiprocessing warnings.
    
    This function is called by Ray in each worker process to configure logging
    and suppress harmless resource_tracker warnings. It checks the environment
    variable URBANVQA_SUPPRESS_WARNINGS to determine if warnings should be suppressed.
    """
    try:
        import warnings
        import os
        
        # Check environment variable to determine if we should suppress warnings
        # Default to True (suppress) if not set
        suppress_warnings = os.environ.get("URBANVQA_SUPPRESS_WARNINGS", "true").lower() in ("true", "1", "yes")
        
        if not suppress_warnings:
            return

        _apply_resource_tracker_patch()
        
        # Set PYTHONWARNINGS for this process
        os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:multiprocessing.resource_tracker"
        
        # Suppress UserWarning from multiprocessing.resource_tracker
        warnings.filterwarnings(
            "ignore",
            message="resource_tracker: process died unexpectedly",
            category=UserWarning,
            module="multiprocessing.resource_tracker"
        )
        warnings.filterwarnings(
            "ignore",
            message=".*resource_tracker.*",
            category=UserWarning,
            module="multiprocessing"
        )
    except Exception:
        pass





def get_suppress_child_warnings(cfg: Optional[DictConfig]) -> bool:
    """Get the suppress_child_warnings setting from config.
    
    Args:
        cfg: Configuration object.
    
    Returns:
        True if warnings should be suppressed (default), False otherwise.
    """
    try:
        return getattr(getattr(cfg, "runtime", None), "suppress_child_warnings", True)
    except Exception:
        return True


# ---------------------------------------------------------------------------
#  SLURM helpers
# ---------------------------------------------------------------------------

def _parse_cpus_on_node(val: str) -> int:
    """Parse SLURM_CPUS_ON_NODE value which can be in various formats."""
    try:
        v = val.strip()
        if "(x" in v and v.endswith(")"):
            m = re.match(r"^(\d+)\(x(\d+)\)$", v)
            if m:
                return max(1, int(m.group(1)) * int(m.group(2)))
        if "," in v:
            acc = 0
            for p in v.split(","):
                acc += int(p)
            return max(1, acc)
        return max(1, int(v))
    except Exception:
        return -1


def _detect_slurm_cpus() -> Optional[int]:
    """Detect SLURM CPU allocation from environment variables."""
    try:
        cpt = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpt is not None and str(cpt).strip() != "":
            return int(cpt)
        con = os.environ.get("SLURM_CPUS_ON_NODE")
        if con is not None and str(con).strip() != "":
            v = _parse_cpus_on_node(con)
            return v if v > 0 else None
    except Exception:
        pass
    return None


def _compute_object_store_bytes(cfg: DictConfig) -> int:
    """Compute the target object-store size in bytes (3-priority hierarchy).

    1. ``RAY_OBJECT_STORE_MEMORY`` env var (absolute bytes)
    2. ``cfg.runtime.object_store_proportion`` (fraction of effective memory)
    3. 50% of ``cfg.runtime.job_memory_gb``
    """
    # Priority 1: explicit env var
    try:
        _env = os.environ.get("RAY_OBJECT_STORE_MEMORY")
        if _env is not None and str(_env).strip():
            return int(str(_env).strip())
    except Exception:
        pass

    # Priority 2: config proportion
    prop = None
    try:
        prop = getattr(cfg.runtime, "object_store_proportion", None)
        prop = float(prop) if prop is not None else None
    except Exception:
        prop = None
    if prop is None:
        try:
            env_prop = os.environ.get("RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION")
            if env_prop is not None and str(env_prop).strip() != "":
                prop = float(env_prop)
        except Exception:
            pass
    if prop is not None and 0.0 < prop <= 0.95:
        try:
            # Best-effort system memory detection
            from .stages.vqa import _effective_total_memory_bytes  # noqa: delayed import
            total_bytes = _effective_total_memory_bytes()
            if total_bytes and total_bytes > 0:
                return int(total_bytes * float(prop))
        except Exception:
            pass

    # Priority 3: 50% of job_memory_gb  (was 30%; raised to reduce
    # object-store back-pressure for image-heavy streaming pipelines).
    # With job_memory_gb=256 this gives ~128 GiB, well above the previous
    # 76.8 GiB default and closer to Ray's recommended ≥50% of available memory.
    try:
        job_mem_gb = int(getattr(cfg.runtime, "job_memory_gb", 64) or 64)
    except Exception:
        job_mem_gb = 64
    return int(max(1, job_mem_gb) * (1024 ** 3) * 0.50)


# ---------------------------------------------------------------------------
#  Unified Ray initialisation
# ---------------------------------------------------------------------------

def ensure_ray_init(cfg: DictConfig, *, caller: str = "shared") -> None:
    """Initialise Ray once, with SLURM-aware resources and runtime env.

    Safe to call from both the orchestrator and ``run_vqa_stage`` — it is a
    no-op if Ray is already running.  After ``ray.init()`` it unconditionally
    applies resource-limits to the ``DataContext`` via the idempotent
    ``_apply_ray_data_resource_limits`` helper in ``stages.vqa``.

    Args:
        cfg: Hydra DictConfig with at least ``runtime`` and optionally
             ``runtime.job_memory_gb``, ``runtime.object_store_proportion``.
        caller: Label for log messages (e.g. ``"orchestrator"``,
                ``"run_vqa_stage"``).
    """
    try:
        import ray  # type: ignore
    except ImportError:
        return

    if ray.is_initialized():
        # Ray already up — just (re-)apply resource limits.
        try:
            from .stages.vqa import _apply_ray_data_resource_limits
            _apply_ray_data_resource_limits(cfg)
        except Exception:
            pass
        return

    # ── Warning suppression runtime env ──────────────────────────────
    suppress_warnings = get_suppress_child_warnings(cfg)
    runtime_env: dict = {}
    if suppress_warnings:
        runtime_env["env_vars"] = {
            "URBANVQA_SUPPRESS_WARNINGS": "true",
            "PYTHONWARNINGS": "ignore::UserWarning:multiprocessing.resource_tracker",
            # Disable Ray's log deduplication so every worker log line is
            # printed individually — critical for diagnosing batch-size
            # and throughput issues across the cluster.
            "RAY_DEDUP_LOGS": "0",
        }
    else:
        runtime_env["env_vars"] = {
            "URBANVQA_SUPPRESS_WARNINGS": "false",
            "RAY_DEDUP_LOGS": "0",
        }

    # ── PYTHONPATH (shared filesystem) ───────────────────────────────
    project_root = Path(__file__).resolve().parents[1]  # dagspaces/urbanvqa -> mllmsci
    dagspaces_path = project_root / "dagspaces"
    env_vars = dict(runtime_env.get("env_vars", {}))

    current_pythonpath = env_vars.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
    pythonpath_parts: list[str] = [str(project_root)]
    if current_pythonpath:
        pythonpath_parts.extend([p for p in current_pythonpath.split(os.pathsep) if p])
    seen: set[str] = set()
    merged: list[str] = []
    for part in pythonpath_parts:
        if part and part not in seen:
            seen.add(part)
            merged.append(part)
    env_vars["PYTHONPATH"] = os.pathsep.join(merged)
    runtime_env["env_vars"] = env_vars

    is_shared_fs = str(project_root).startswith("/share/")
    if not is_shared_fs:
        runtime_env["working_dir"] = str(project_root)
    try:
        skip_upload = bool(getattr(cfg.runtime, "ray_skip_module_upload", is_shared_fs))
    except Exception:
        skip_upload = is_shared_fs
    if not skip_upload and dagspaces_path.exists():
        py_modules = list(runtime_env.get("py_modules", []))
        root_str = str(project_root)
        if root_str not in py_modules:
            py_modules.append(root_str)
        runtime_env["py_modules"] = py_modules

    # ── Detect resources ─────────────────────────────────────────────
    cpus_alloc = _detect_slurm_cpus()
    obj_store_bytes = _compute_object_store_bytes(cfg)

    _obj_gb = round(obj_store_bytes / (1024 ** 3), 2) if obj_store_bytes else "?"
    _src = "RAY_OBJECT_STORE_MEMORY env" if os.environ.get("RAY_OBJECT_STORE_MEMORY") else "heuristic"
    print(
        f"[ensure_ray_init:{caller}] object_store_memory={obj_store_bytes} bytes "
        f"({_obj_gb} GB), cpus={cpus_alloc}, source={_src}",
        flush=True,
    )

    namespace = os.environ.get("RAY_NAMESPACE") or os.environ.get("WANDB_GROUP") or "urbanvqa"

    # ── ray.init ─────────────────────────────────────────────────────
    init_kwargs: dict = {
        "log_to_driver": True,
        "object_store_memory": int(obj_store_bytes),
        "namespace": str(namespace),
    }
    if runtime_env:
        init_kwargs["runtime_env"] = runtime_env
    if cpus_alloc is not None and cpus_alloc > 0:
        init_kwargs["num_cpus"] = int(cpus_alloc)

    try:
        ray.init(**init_kwargs)
    except Exception:
        # Fallback: basic init (no object_store_memory)
        fallback_kwargs: dict = {"log_to_driver": True}
        if runtime_env:
            fallback_kwargs["runtime_env"] = runtime_env
        if cpus_alloc is not None and cpus_alloc > 0:
            fallback_kwargs["num_cpus"] = int(cpus_alloc)
        try:
            ray.init(**fallback_kwargs)
        except Exception:
            pass

    # ── Apply DataContext resource limits ─────────────────────────────
    try:
        from .stages.vqa import _apply_ray_data_resource_limits
        _apply_ray_data_resource_limits(cfg)
    except Exception:
        pass

