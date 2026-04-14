from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import gepa
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import yaml

from dagspaces.common.wandb_logger import WandbLogger

from .dataset import materialize_supervised_frame
from .gepa_adapter import GEPAVQAAdapter

LOG = logging.getLogger(__name__)


def _teardown_persistent_processor() -> None:
    """Tear down the persistent vLLM processor and clean up resources.
    
    This ensures the Slurm job can terminate by:
    1. Shutting down the persistent vLLM processor
    2. Clearing any global processor caches
    """
    try:
        from dagspaces.urbanvqa.stages.persistent_vllm import clear_processor_cache
        LOG.info("Tearing down persistent vLLM processor...")
        clear_processor_cache()
    except ImportError:
        LOG.debug("persistent_vllm module not available, skipping processor cleanup")
    except Exception as e:
        LOG.warning(f"Error during processor cache cleanup: {e}")
    
    # Clear the global processor reference in the adapter module
    try:
        from dagspaces.urbanvqa.prompt_opt import gepa_adapter
        if hasattr(gepa_adapter, "_PERSISTENT_PROCESSOR") and gepa_adapter._PERSISTENT_PROCESSOR is not None:
            LOG.info("Clearing global persistent processor reference...")
            gepa_adapter._PERSISTENT_PROCESSOR = None
    except Exception as e:
        LOG.warning(f"Error clearing global processor reference: {e}")
    
    # Force garbage collection and CUDA cache clear
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    except Exception as e:
        LOG.debug(f"Error during CUDA cleanup: {e}")
    
    LOG.info("Persistent processor teardown complete")


def _ensure_artifact_dir(cfg: DictConfig) -> Path:
    hydra_run_dir = Path(HydraConfig.get().runtime.output_dir)
    base_dir = getattr(cfg.gepa.artifacts, "base_dir", "outputs/gepa")
    artifact_dir = Path(base_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = hydra_run_dir / artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _frame_to_records(frame: pd.DataFrame) -> List[Dict]:
    if frame.empty:
        return []
    return frame.to_dict(orient="records")


def _derive_seed_candidate(cfg: DictConfig) -> Dict[str, str]:
    """Derive the seed candidate from config, using gepa.components if defined.
    
    This reads the gepa.components config to determine which prompts to include
    and their source paths. If gepa.components is not defined, falls back to
    the legacy behavior of using prompt.system and prompt.user_template.
    """
    candidate: Dict[str, str] = {}
    
    # Read components from gepa.components config
    components_cfg = getattr(cfg.gepa, "components", None)
    
    if components_cfg:
        # Use the components config to determine seed candidate keys and sources
        for comp in components_cfg:
            name = comp.get("name")
            target = comp.get("target")
            if name and target:
                # Resolve the target path from the config
                value = OmegaConf.select(cfg, target, default="")
                candidate[name] = value if value else ""
    else:
        # Legacy fallback: use hardcoded prompt paths
        candidate = {
            "system_prompt": getattr(cfg.prompt, "system", ""),
            "user_prompt": getattr(cfg.prompt, "user_template", ""),
        }
    
    return candidate


def run_gepa_optimization(cfg: DictConfig) -> None:
    mode = getattr(cfg.gepa, "mode", "optimize")
    wandb_suffix = getattr(cfg.gepa, "wandb_suffix", f"gepa_{mode}")
    optimizer_cfg = getattr(cfg.gepa, "optimizer", None)
    if optimizer_cfg is None:
        optimizer_cfg_resolved = {}
    else:
        optimizer_cfg_resolved = (
            OmegaConf.to_container(optimizer_cfg, resolve=True)
            if OmegaConf.is_config(optimizer_cfg)
            else dict(optimizer_cfg)
        )

    run_config = {
        "mode": mode,
        "dataset": {
            "train_limit": OmegaConf.select(cfg, "gepa.dataset.train.limit"),
            "val_limit": OmegaConf.select(cfg, "gepa.dataset.val.limit"),
        },
        "optimizer": optimizer_cfg_resolved,
    }

    # Setup environment for GEPA reflection if configured
    # This allows GEPA's internal LLM client (e.g. LiteLLM) to find the correct API base/key
    reflection_cfg = getattr(cfg.llm, "reflection", None)
    if reflection_cfg:
        api_base = getattr(reflection_cfg, "api_base", None)
        api_key = getattr(reflection_cfg, "api_key", None)
        api_key_env = getattr(reflection_cfg, "api_key_env", None)
        
        if api_base:
            os.environ["OPENAI_API_BASE"] = api_base
            LOG.info("Set OPENAI_API_BASE=%s for GEPA reflection", api_base)
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif api_key_env:
            env_key = os.getenv(api_key_env)
            if env_key:
                os.environ["OPENAI_API_KEY"] = env_key
        else:
            # If no key provided and using vLLM, set a dummy key to satisfy clients
            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = "EMPTY"

    with WandbLogger(cfg, stage="gepa", run_id=wandb_suffix, run_config=run_config) as logger:
        if mode == "validate":
            _run_validation(cfg, logger)
            return
        if mode != "optimize":
            raise ValueError(f"Unsupported GEPA mode: {mode}")

        _run_optimize(cfg, logger)


def _load_candidate_from_file(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return {
        "system_prompt": data.get("system_prompt", ""),
        "user_prompt": data.get("user_prompt", ""),
    }


def _resolve_validation_candidate(cfg: DictConfig) -> Dict[str, str]:
    validation_cfg = getattr(cfg.gepa, "validation", OmegaConf.create({}))
    prompt_path = validation_cfg.get("prompt_path") or OmegaConf.select(cfg, "runtime.prompt_override_path")
    if not prompt_path:
        raise ValueError("Validation requires gepa.validation.prompt_path or runtime.prompt_override_path to be set.")
    resolved = Path(prompt_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Validation prompt file not found: {resolved}")
    return _load_candidate_from_file(str(resolved))


def _run_optimize(cfg: DictConfig, logger: Optional[WandbLogger]) -> None:
    adapter = GEPAVQAAdapter(cfg)
    train_frame = materialize_supervised_frame(cfg, "train")
    val_frame = materialize_supervised_frame(cfg, "val")

    seed_candidate = _derive_seed_candidate(cfg)
    optimization_cfg = getattr(cfg.gepa, "optimization", OmegaConf.create({}))
    max_metric_calls = getattr(optimization_cfg, "max_metric_calls", 120)
    reflection_minibatch_size = getattr(optimization_cfg, "reflection_minibatch_size", 3)
    
    # Configure reflection LM (text-only or multimodal)
    use_multimodal_reflection = getattr(optimization_cfg, "multimodal_reflection", False)
    reflection_cfg = cfg.llm.reflection
    
    if use_multimodal_reflection:
        # Use multimodal reflection wrapper - reflector will see images
        from .multimodal_reflection import create_multimodal_reflection_lm
        
        reflection_model_name = getattr(reflection_cfg, "model", "")
        if reflection_model_name and not reflection_model_name.startswith("openai/"):
            provider = getattr(reflection_cfg, "provider", "")
            if provider == "vllm":
                reflection_model_name = f"openai/{reflection_model_name}"
        
        # Resolve API key
        api_key = getattr(reflection_cfg, "api_key", None)
        if not api_key:
            api_key_env = getattr(reflection_cfg, "api_key_env", None)
            if api_key_env:
                api_key = os.getenv(api_key_env)
        
        reflection_model = create_multimodal_reflection_lm(
            model=reflection_model_name,
            api_base=getattr(reflection_cfg, "api_base", None),
            api_key=api_key,
            temperature=getattr(reflection_cfg, "temperature", 0.0),
            seed=getattr(reflection_cfg, "seed", getattr(cfg, "seed", 0)),
            max_images_per_call=getattr(optimization_cfg, "reflection_max_images", 5),
            timeout=getattr(reflection_cfg, "request_timeout_s", None),
        )
        LOG.info(
            "Multimodal reflection enabled: reflector will see images "
            "(max %d images, %.0f%% incorrect ratio)",
            getattr(optimization_cfg, "reflection_max_images", 5),
            getattr(optimization_cfg, "reflection_incorrect_ratio", 0.7) * 100,
        )
    else:
        # Standard text-only reflection
        reflection_model = getattr(reflection_cfg, "model", None)
        if reflection_model and not reflection_model.startswith("openai/"):
            if getattr(reflection_cfg, "provider", "") == "vllm":
                reflection_model = f"openai/{reflection_model}"
    
    # NOTE: llm.task is deprecated - VQA inference uses cfg.model directly via run_vqa_stage()
    optimizer_cfg = getattr(cfg.gepa, "optimizer", None)
    if optimizer_cfg is None:
        optimizer_payload = {}
    elif OmegaConf.is_config(optimizer_cfg):
        optimizer_payload = OmegaConf.to_container(optimizer_cfg, resolve=True) or {}
    else:
        optimizer_payload = dict(optimizer_cfg) or {}

    LOG.info("Starting GEPA optimization with %s training rows and %s validation rows", len(train_frame), len(val_frame))
    with open("/share/pierson/matt/mllmsci/gepa_debug.log", "a") as f:
        f.write(f"DEBUG: Train frame columns: {train_frame.columns.tolist()}\n")
        if not train_frame.empty:
            f.write(f"DEBUG: First train record keys: {list(train_frame.iloc[0].to_dict().keys())}\n")

    # Enable wandb logging if logger is enabled
    use_wandb = logger is not None and logger.enabled
    
    optimize_kwargs = dict(
        seed_candidate=seed_candidate,
        trainset=_frame_to_records(train_frame),
        valset=_frame_to_records(val_frame),
        adapter=adapter,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=reflection_minibatch_size,
        reflection_lm=reflection_model,
        use_wandb=use_wandb,
    )
    for key, value in optimizer_payload.items():
        if value is not None:
            optimize_kwargs[key] = value

    try:
        result = gepa.optimize(**optimize_kwargs)
    finally:
        # Always tear down the persistent processor to allow Slurm job to terminate
        optimization_cfg = getattr(cfg.gepa, "optimization", OmegaConf.create({}))
        if getattr(optimization_cfg, "persistent_processor", False):
            _teardown_persistent_processor()

    artifact_dir = _ensure_artifact_dir(cfg)
    best_candidate_path = artifact_dir / "best_prompts.yaml"
    with best_candidate_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result.best_candidate, handle, sort_keys=False)

    metrics_path = artifact_dir / "metrics.json"
    summary = {
        "best_score": getattr(result, "best_score", None),
        "max_metric_calls": max_metric_calls,
        "train_rows": len(train_frame),
        "val_rows": len(val_frame),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if logger and logger.enabled:
        logger.log_metrics(
            {
                "gepa/train_rows": len(train_frame),
                "gepa/val_rows": len(val_frame),
                "gepa/best_score": summary["best_score"],
            }
        )

    traces = getattr(result, "best_candidate_traces", None)
    if traces:
        traces_path = artifact_dir / "traces.jsonl"
        with traces_path.open("w", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(json.dumps(trace))
                handle.write("\n")

    LOG.info("GEPA optimization completed. Best prompts written to %s", best_candidate_path)


def _run_validation(cfg: DictConfig, logger: Optional[WandbLogger]) -> None:
    adapter = GEPAVQAAdapter(cfg)
    validation_cfg = getattr(cfg.gepa, "validation", OmegaConf.create({}))
    split = validation_cfg.get("split", "val")
    frame = materialize_supervised_frame(cfg, split)
    limit: Optional[int] = validation_cfg.get("limit")
    if isinstance(limit, int) and limit > 0 and limit < len(frame):
        seed = getattr(cfg, "seed", 0)
        frame = frame.sample(n=limit, random_state=seed).reset_index(drop=True)

    candidate = _resolve_validation_candidate(cfg)
    score, traces = adapter.evaluate(candidate, [frame])
    LOG.info("Validation score on split '%s' (%d rows): %.4f", split, len(frame), score)

    expected = validation_cfg.get("expected_score")
    tolerance = validation_cfg.get("tolerance", 1e-3)
    if expected is not None and tolerance is not None:
        if abs(score - expected) > tolerance:
            raise AssertionError(
                f"Validation score {score:.4f} deviates from expected {expected:.4f} (tolerance={tolerance})"
            )

    artifact_dir = _ensure_artifact_dir(cfg)
    metrics_path = artifact_dir / "validation_metrics.json"
    summary = {
        "score": score,
        "rows": len(frame),
        "split": split,
        "expected_score": expected,
        "tolerance": tolerance,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if logger and logger.enabled:
        logger.log_metrics(
            {
                "gepa/validation_score": score,
                "gepa/validation_rows": len(frame),
                "gepa/validation_rows": len(frame),
            }
        )

    if traces:
        trace_path = artifact_dir / "validation_traces.jsonl"
        with trace_path.open("w", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(json.dumps(trace))
                handle.write("\n")

    LOG.info("Validation artifacts written to %s", artifact_dir)
