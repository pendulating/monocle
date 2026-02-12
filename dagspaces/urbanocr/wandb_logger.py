"""Centralized W&B logging for Urban OCR pipeline.

This module provides a unified interface for W&B logging across all pipeline stages,
handling distributed execution (Ray, SLURM) and run lifecycle management.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import platform
import socket
import traceback
import getpass
import shutil

# Configure wandb for distributed environments (SLURM, Ray)
if "TMPDIR" not in os.environ:
    shared_tmp = "/share/pierson/matt/tmp/wandb"
    os.makedirs(shared_tmp, exist_ok=True)
    os.environ["TMPDIR"] = shared_tmp

# Apply defaults from repo-level wandb/settings before importing wandb
def _apply_wandb_settings_defaults() -> None:
    try:
        settings_path = os.environ.get("WANDB_SETTINGS_PATH")
        if not settings_path:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            settings_path = os.path.join(base_dir, "wandb", "settings")
        if not (isinstance(settings_path, str) and os.path.exists(settings_path)):
            return
        import configparser
        cp = configparser.ConfigParser()
        try:
            cp.read(settings_path)
        except Exception:
            return
        sect = "default" if cp.has_section("default") else (cp.sections()[0] if cp.sections() else None)
        if not sect:
            return
        sec = cp[sect]
        try:
            entity = sec.get("entity", fallback=None)
        except Exception:
            entity = None
        try:
            project = sec.get("project", fallback=None)
        except Exception:
            project = None
        try:
            base_url = sec.get("base_url", fallback=None)
        except Exception:
            base_url = None
        if entity and not os.environ.get("WANDB_ENTITY"):
            os.environ["WANDB_ENTITY"] = str(entity)
        if project and not os.environ.get("WANDB_PROJECT"):
            os.environ["WANDB_PROJECT"] = str(project)
        if base_url and not os.environ.get("WANDB_BASE_URL"):
            os.environ["WANDB_BASE_URL"] = str(base_url)
    except Exception:
        pass

_apply_wandb_settings_defaults()

import wandb as wandb_module
from omegaconf import DictConfig, OmegaConf


@dataclass
class WandbConfig:
    """W&B configuration extracted from Hydra config."""
    
    enabled: bool = False
    project: str = "URBANOCR"
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    table_sample_rows: int = 1000
    table_sample_seed: int = 777
    
    @classmethod
    def from_hydra_config(cls, cfg) -> "WandbConfig":
        """Extract W&B config from Hydra config."""
        try:
            wandb_cfg = getattr(cfg, "wandb", None)
            env_entity = os.environ.get("WANDB_ENTITY")
            env_project = os.environ.get("WANDB_PROJECT")
            if wandb_cfg is None:
                return cls(
                    enabled=False,
                    project=(env_project or "URBANOCR"),
                    entity=(env_entity if env_entity and env_entity.strip() else None),
                    group=_get_group_from_config(cfg),
                    tags=[],
                    table_sample_rows=1000,
                    table_sample_seed=777,
                )
            
            proj_attr = getattr(wandb_cfg, "project", None)
            if proj_attr is None or str(proj_attr).strip() == "":
                project = env_project or "URBANOCR"
            else:
                project = str(proj_attr or "URBANOCR")
            entity_cfg = _get_optional_str(wandb_cfg, "entity")
            entity = entity_cfg or (env_entity if env_entity and env_entity.strip() else None)
            return cls(
                enabled=bool(getattr(wandb_cfg, "enabled", False)),
                project=project,
                entity=entity,
                group=_get_group_from_config(cfg),
                tags=_get_list(wandb_cfg, "tags"),
                table_sample_rows=int(getattr(wandb_cfg, "table_sample_rows", 1000)),
                table_sample_seed=int(getattr(wandb_cfg, "table_sample_seed", 777)),
            )
        except Exception as e:
            print(f"[wandb] Warning: Failed to parse config: {e}", file=sys.stderr)
            return cls()


def _get_optional_str(obj, attr: str) -> Optional[str]:
    try:
        val = getattr(obj, attr, None)
        if val is not None and str(val).strip():
            return str(val)
    except Exception:
        pass
    return None


def _get_list(obj, attr: str) -> List[str]:
    try:
        val = getattr(obj, attr, None)
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            return [str(x) for x in val]
        return [str(val)]
    except Exception:
        return []


def _get_group_from_config(cfg) -> Optional[str]:
    try:
        grp = getattr(cfg.wandb, "group", None)
        if grp and str(grp).strip():
            return str(grp)
    except Exception:
        pass
    
    env_group = os.environ.get("WANDB_GROUP")
    if env_group and env_group.strip():
        return env_group
    
    submitit_job_id = os.environ.get("SUBMITIT_JOB_ID")
    if submitit_job_id and submitit_job_id.strip():
        return f"slurm-{submitit_job_id}"
    
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id and slurm_job_id.strip():
        return f"slurm-{slurm_job_id}"
    
    return None


def _detect_num_gpus() -> int:
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            gpu_indices = [x.strip() for x in cuda_visible.split(",") if x.strip()]
            if gpu_indices:
                return len(gpu_indices)
    except Exception:
        pass
    
    try:
        slurm_gpus = os.environ.get("SLURM_GPUS_PER_NODE") or os.environ.get("SLURM_GPUS_ON_NODE")
        if slurm_gpus:
            try:
                if ":" in slurm_gpus:
                    return int(slurm_gpus.split(":")[-1])
                return int(slurm_gpus)
            except Exception:
                pass
    except Exception:
        pass
    
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                return count
    except Exception:
        pass
    
    return 0


def _detect_gpu_type() -> Optional[str]:
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_name = torch.cuda.get_device_name(0)
            return gpu_name
    except Exception:
        pass
    return None


def collect_compute_metadata(cfg=None) -> Dict[str, Any]:
    """Collect comprehensive compute metadata for wandb logging."""
    metadata: Dict[str, Any] = {}
    
    try:
        metadata["system"] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
        }
    except Exception:
        pass
    
    num_gpus = _detect_num_gpus()
    metadata["compute.gpu_count"] = num_gpus
    
    if num_gpus > 0:
        gpu_type = _detect_gpu_type()
        if gpu_type:
            metadata["compute.gpu_type"] = gpu_type
    
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        metadata["compute.cuda_visible_devices"] = [d.strip() for d in cuda_visible.split(",") if d.strip()]
    
    slurm_info = {}
    for key in ["SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NODELIST", "SLURM_PARTITION"]:
        val = os.environ.get(key)
        if val:
            slurm_info[key.lower()] = val
    
    if slurm_info:
        metadata["slurm"] = slurm_info
    
    return metadata


class WandbLogger:
    """Thread-safe centralized W&B logger for Urban OCR."""
    
    _lock = threading.Lock()
    _wandb = None
    _wandb_available = None
    
    def __init__(
        self,
        cfg,
        stage: str,
        run_id: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
        self.stage = stage
        self.run_id = run_id
        self.run_config = run_config or {}
        self.wb_config = WandbConfig.from_hydra_config(cfg)
        self._run = None
        
        if WandbLogger._wandb is None:
            with WandbLogger._lock:
                if WandbLogger._wandb is None:
                    WandbLogger._wandb = wandb_module
                    WandbLogger._wandb_available = True
    
    @property
    def enabled(self) -> bool:
        return self.wb_config.enabled and WandbLogger._wandb_available
    
    @property
    def wandb(self):
        return WandbLogger._wandb
    
    def _get_run_name(self) -> str:
        try:
            exp_name = str(getattr(self.cfg.experiment, "name", "URBANOCR") or "URBANOCR")
        except Exception:
            exp_name = "URBANOCR"
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        parts = [exp_name, self.stage, timestamp]
        return "-".join(parts)
    
    def _get_mode(self) -> str:
        mode_env = os.environ.get("WANDB_MODE")
        if mode_env:
            mode_lower = mode_env.lower().strip()
            if mode_lower in ("online", "offline", "disabled"):
                return mode_lower
        return "online"
    
    def _is_ray_worker(self) -> bool:
        try:
            import ray
            if ray.is_initialized():
                try:
                    worker = ray._private.worker.global_worker
                    return worker.mode == ray.WORKER_MODE
                except Exception:
                    pass
        except ImportError:
            pass
        return False
    
    def start(self) -> None:
        if not self.enabled:
            return
        
        if self._is_ray_worker():
            print(f"[wandb] Skipping initialization in Ray worker for {self.stage}", flush=True)
            return
        
        if self._run is not None:
            return
        
        try:
            for k in ("WANDB_SERVICE", "WANDB__SERVICE", "WANDB_SERVICE_SOCKET"):
                if k in os.environ:
                    os.environ.pop(k, None)
            os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
        except Exception:
            pass

        mode = self._get_mode()
        run_name = self._get_run_name()
        wandb_dir = os.environ.get("WANDB_DIR") or os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())
        
        print(f"[wandb] Starting run: {run_name} (mode={mode})", flush=True)
        
        try:
            self._run = self.wandb.init(
                project=self.wb_config.project,
                entity=self.wb_config.entity,
                group=self.wb_config.group,
                job_type=self.stage,
                name=run_name,
                config=self.run_config,
                mode=mode,
                dir=wandb_dir,
                tags=self.wb_config.tags,
            )
            
            try:
                compute_metadata = collect_compute_metadata(self.cfg)
                if compute_metadata:
                    self.set_config(compute_metadata, allow_val_change=True)
            except Exception:
                pass
            
            print(f"[wandb] ✓ Run started: {run_name}", flush=True)
            
        except Exception as e:
            print(f"[wandb] ✗ Failed to start run: {e}", file=sys.stderr, flush=True)
            self._run = None
    
    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        
        try:
            run_name = getattr(self._run, "name", "unknown")
            print(f"[wandb] Finishing run: {run_name}", flush=True)
            self.wandb.finish()
            print(f"[wandb] ✓ Run completed: {run_name}", flush=True)
            self._run = None
        except Exception as e:
            print(f"[wandb] ✗ Failed to finish run: {e}", file=sys.stderr, flush=True)
            self._run = None
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
        if not self.enabled or self._run is None:
            return
        
        try:
            if metrics:
                self.wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            print(f"[wandb] Warning: Failed to log metrics: {e}", file=sys.stderr)
    
    def log_table(
        self,
        df,
        key: str,
        prefer_cols: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
        panel_group: Optional[str] = None,
    ) -> None:
        if not self.enabled or self._run is None or df is None:
            return
        
        try:
            import pandas as pd
            
            table_key = f"{panel_group}/{key}" if panel_group else key
            
            try:
                df_local = df.loc[:, ~df.columns.duplicated()]
            except Exception:
                df_local = df
            
            cols = [c for c in (prefer_cols or []) if c in df_local.columns]
            if not cols:
                cols = list(df_local.columns)[:12]
            
            max_rows = max_rows or self.wb_config.table_sample_rows
            total_rows = len(df_local)
            
            if total_rows > max_rows:
                df_sample = df_local.sample(
                    n=max_rows,
                    random_state=self.wb_config.table_sample_seed
                ).reset_index(drop=True)
            else:
                df_sample = df_local.reset_index(drop=True)
            
            table = self.wandb.Table(dataframe=df_sample[cols])
            self.wandb.log({table_key: table})
            
            print(f"[wandb] ✓ Logged table '{table_key}': {len(df_sample):,} rows", flush=True)
            
        except Exception as e:
            print(f"[wandb] Warning: Failed to log table '{key}': {e}", file=sys.stderr)
    
    def set_summary(self, key: str, value: Any) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            self._run.summary[key] = value
        except Exception as e:
            print(f"[wandb] Warning: Failed to set summary '{key}': {e}", file=sys.stderr)
    
    def set_config(self, data: Dict[str, Any], allow_val_change: bool = True) -> None:
        if not self.enabled or self._run is None or not data:
            return
        try:
            self._run.config.update(dict(data), allow_val_change=allow_val_change)
        except Exception as e:
            print(f"[wandb] Warning: Failed to update config: {e}", file=sys.stderr)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False

