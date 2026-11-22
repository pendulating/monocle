"""Compute metadata collection for W&B run config.

This module builds the metadata dictionary that pipelines log to
wandb.config. It covers system info, CPU/GPU/memory allocation, SLURM job
parameters, model settings, and runtime config keys.

Host detection primitives live in the sibling host_info module.

The wandb_logger module previously held this code. Public names did not
change.
"""

import getpass
import os
import platform
import socket
from typing import Any, Dict, List, Optional

from .host_info import (
    _collect_gpu_details,
    _detect_gpu_type,
    _detect_memory_gb,
    _detect_num_cpus,
    _detect_num_gpus,
)

# ---------------------------------------------------------------------------
# Compute metadata collection
# ---------------------------------------------------------------------------

#: Runtime config keys that all dagspaces always extract.
_BASE_RUNTIME_KEYS: List[str] = [
    "debug",
    "sample_n",
    "job_memory_gb",
    "rows_per_block",
]

#: GPU sanitizer env-var suffixes (appended to the dagspace prefix).
_SANITIZER_SUFFIXES = (
    "SANITIZED_DROPPED_GPUS",
    "GPU_SANITIZE_ORIGINAL",
    "GPU_SANITIZE_REASON",
    "GPU_SANITIZE_TS",
    "TENSOR_PARALLEL_SIZE",
)


def _derive_checkpoint_name(lora_path: str, model_source: str) -> str:
    """Derive a human-readable checkpoint name from lora_path and model_source.

    Combines the base model name with a meaningful path segment from the LoRA
    path, skipping generic directory names like "checkpoint", "outputs", "sft".
    """
    base_name = os.path.basename(model_source) if model_source else "unknown"
    _generic = {"checkpoint", "outputs", "sft", "checkpoints", "output", "model",
                "models", "grpo", "lora"}
    parts = lora_path.rstrip("/").split("/")
    suffix = None
    for part in reversed(parts):
        if part and part.lower() not in _generic and not part.startswith("."):
            suffix = part
            break
    if suffix:
        return f"{base_name}+{suffix}"
    return base_name


def collect_compute_metadata(
    cfg=None,
    *,
    env_var_prefix: str = "",
    extra_runtime_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Collect comprehensive compute metadata for wandb logging.

    Captures system configuration including CPU count and architecture, GPU
    count/type/memory, RAM allocation, SLURM job parameters, Python
    environment, and model configuration.

    Args:
        cfg: Optional Hydra config to extract model/runtime parameters.
        env_var_prefix: Dagspace prefix for GPU sanitizer env vars (e.g.
            "IMAGE_FORWARD" reads IMAGE_FORWARD_GPU_SANITIZE_ORIGINAL etc.).  Empty string
            skips sanitizer var collection.
        extra_runtime_keys: Additional runtime config keys to log beyond the
            base set.

    Returns:
        Dictionary with compute metadata suitable for wandb.config.
    """
    metadata: Dict[str, Any] = {}

    # System info
    try:
        metadata["system"] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
        }
    except Exception:
        pass

    # CPU info
    num_cpus = _detect_num_cpus()
    if num_cpus is not None:
        metadata["compute.cpu_count"] = num_cpus

    try:
        metadata["compute.cpu_architecture"] = platform.machine()
    except Exception:
        pass

    # GPU info
    num_gpus = _detect_num_gpus()
    metadata["compute.gpu_count"] = num_gpus

    if num_gpus > 0:
        gpu_type = _detect_gpu_type()
        if gpu_type:
            metadata["compute.gpu_type"] = gpu_type

        gpu_details = _collect_gpu_details()
        if gpu_details:
            metadata["compute.gpus"] = gpu_details

    # CUDA device mapping
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        metadata["compute.cuda_visible_devices"] = [
            d.strip() for d in cuda_visible.split(",") if d.strip()
        ]

    # GPU sanitizer metadata (dagspace-specific env var prefix)
    if env_var_prefix:
        pfx = env_var_prefix.rstrip("_") + "_"
        dropped = os.environ.get(f"{pfx}SANITIZED_DROPPED_GPUS")
        original = os.environ.get(f"{pfx}GPU_SANITIZE_ORIGINAL")
        reason = os.environ.get(f"{pfx}GPU_SANITIZE_REASON")
        ts = os.environ.get(f"{pfx}GPU_SANITIZE_TS")
        tp_raw = os.environ.get(f"{pfx}TENSOR_PARALLEL_SIZE")

        sanitize_meta: Dict[str, Any] = {}
        if original:
            sanitize_meta["original"] = [
                d.strip() for d in original.split(",") if d.strip()
            ]
        if dropped:
            sanitize_meta["dropped"] = [
                d.strip() for d in dropped.split(",") if d.strip()
            ]
        if reason:
            sanitize_meta["reason"] = reason
        if ts:
            try:
                sanitize_meta["timestamp"] = int(ts)
            except Exception:
                sanitize_meta["timestamp"] = ts
        if tp_raw:
            try:
                sanitize_meta["tensor_parallel_size"] = int(tp_raw)
            except Exception:
                sanitize_meta["tensor_parallel_size"] = tp_raw
        if sanitize_meta:
            metadata["compute.gpu_sanitize"] = sanitize_meta

    # Memory info
    mem_gb = _detect_memory_gb()
    if mem_gb is not None:
        metadata["compute.memory_gb"] = round(mem_gb, 2)

    # SLURM job info
    slurm_info: Dict[str, Any] = {}
    for key in [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_JOB_NUM_NODES",
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_GPUS_PER_NODE",
        "SLURM_TASKS_PER_NODE",
        "SLURM_PARTITION",
        "SLURM_SUBMIT_DIR",
        "SUBMITIT_JOB_ID",
    ]:
        val = os.environ.get(key)
        if val:
            slurm_info[key.lower()] = val

    if slurm_info:
        metadata["slurm"] = slurm_info

    # Model and runtime config
    if cfg is not None:
        try:
            model_cfg = getattr(cfg, "model", None)
            if model_cfg:
                model_info: Dict[str, Any] = {}

                model_source = getattr(model_cfg, "model_source", None)
                if model_source:
                    model_info["model_source"] = str(model_source)

                engine_kwargs = getattr(model_cfg, "engine_kwargs", None)
                if engine_kwargs:
                    ek: Dict[str, Any] = {}
                    for key in [
                        "max_model_len",
                        "max_num_seqs",
                        "max_num_batched_tokens",
                        "gpu_memory_utilization",
                        "tensor_parallel_size",
                        "enable_chunked_prefill",
                        "enable_prefix_caching",
                        "dtype",
                        "kv_cache_dtype",
                    ]:
                        try:
                            val = getattr(engine_kwargs, key, None)
                            if val is not None:
                                ek[key] = val
                        except Exception:
                            pass
                    if ek:
                        model_info["engine_kwargs"] = ek

                batch_size = getattr(model_cfg, "batch_size", None)
                if batch_size is not None:
                    model_info["batch_size"] = int(batch_size)

                concurrency = getattr(model_cfg, "concurrency", None)
                if concurrency is not None:
                    model_info["concurrency"] = int(concurrency)

                # --- Fields for cross-model comparison ---
                model_family = getattr(model_cfg, "model_family", None)
                if model_family:
                    model_info["model_family"] = str(model_family)

                lora_path = getattr(model_cfg, "lora_path", None)
                is_finetuned_flag = getattr(model_cfg, "is_finetuned", False)
                if lora_path:
                    model_info["lora_path"] = str(lora_path)
                    model_info["is_finetuned"] = True
                elif is_finetuned_flag:
                    model_info["is_finetuned"] = True
                else:
                    model_info["is_finetuned"] = False

                chat_template_kwargs = getattr(model_cfg, "chat_template_kwargs", None)
                if chat_template_kwargs:
                    try:
                        ctk = {}
                        for key in ("enable_thinking",):
                            val = getattr(chat_template_kwargs, key, None)
                            if val is not None:
                                ctk[key] = val
                        if ctk:
                            model_info["chat_template_kwargs"] = ctk
                    except Exception:
                        pass

                # Single-book training metadata
                single_book = getattr(model_cfg, "single_book", False)
                model_info["single_book"] = bool(single_book)
                book_id = getattr(model_cfg, "book_id", None)
                if book_id:
                    model_info["book_id"] = str(book_id)

                # Derive checkpoint_name for human-readable identification
                checkpoint_name = getattr(model_cfg, "checkpoint_name", None)
                if checkpoint_name:
                    model_info["checkpoint_name"] = str(checkpoint_name)
                elif lora_path:
                    model_info["checkpoint_name"] = _derive_checkpoint_name(
                        str(lora_path), str(model_source) if model_source else ""
                    )
                elif model_source:
                    model_info["checkpoint_name"] = os.path.basename(str(model_source))

                if model_info:
                    metadata["model"] = model_info
        except Exception:
            pass

        try:
            runtime_cfg = getattr(cfg, "runtime", None)
            if runtime_cfg:
                runtime_info: Dict[str, Any] = {}
                all_runtime_keys = _BASE_RUNTIME_KEYS + (
                    extra_runtime_keys or []
                )
                for key in all_runtime_keys:
                    try:
                        val = getattr(runtime_cfg, key, None)
                        if val is not None:
                            runtime_info[key] = val
                    except Exception:
                        pass
                if runtime_info:
                    metadata["runtime"] = runtime_info
        except Exception:
            pass

    return metadata
