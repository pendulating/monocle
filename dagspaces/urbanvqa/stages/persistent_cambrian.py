"""Persistent Cambrian Processor for efficient GEPA optimization.

This module provides a persistent Cambrian-13B processor that handles the 
custom vision architecture of Cambrian models using transformers and torch.
It eliminates model teardown between GEPA iteration cycles.

The design follows the same interface as PersistentVLLMProcessor to enable
seamless integration into the GEPA optimization pipeline.
"""

from __future__ import annotations

import logging
import threading
import sys
import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import io
import base64

import subprocess
import json
import time

LOG = logging.getLogger(__name__)

try:
    import ray
except ImportError:
    ray = None

# Global processor cache
_PROCESSOR_CACHE: Dict[str, "PersistentCambrianProcessor"] = {}
_CACHE_LOCK = threading.Lock()

def _local_sanitize_prompt(value: Any, cfg: Dict[str, Any]) -> str:
    """Helper for prompt sanitization."""
    default_prompt = "What do you see in this image?"
    try:
        data_cfg = cfg.get("data", {})
        candidate = data_cfg.get("default_prompt")
        if isinstance(candidate, str) and candidate.strip():
            default_prompt = candidate.strip()
    except Exception:
        pass
        
    if value is None:
        return default_prompt
    if isinstance(value, str):
        sanitized = value.strip()
    else:
        sanitized = str(value).strip()
    return sanitized or default_prompt

if ray is not None:
    @ray.remote(num_gpus=1)
    class CambrianInferenceActor:
        """Isolated Ray Actor that manages a Python 3.11 sidecar process."""
        
        def __init__(self, cfg: Any):
            if isinstance(cfg, DictConfig):
                self._cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            else:
                self._cfg_dict = cfg
            self._proc = None
            self._initialized = False
            
        def initialize(self):
            if self._initialized:
                return
            
            project_root = "/share/pierson/matt/mllmsci"
            cambrian_venv = os.path.join(project_root, ".venv-cambrian")
            python_exe = os.path.join(cambrian_venv, "bin/python")
            cambrian_src = os.path.join(project_root, "sub/cambrian")
            
            print(f"Actor: Starting Python 3.11 sidecar bridge using {python_exe}...", flush=True)
            LOG.info(f"Actor: Starting Python 3.11 sidecar bridge using {python_exe}...")
            
            env = os.environ.copy()
            # Clean up env to ensure subprocess is isolated
            env["PYTHONPATH"] = f"{cambrian_src}:{project_root}"
            env["VIRTUAL_ENV"] = cambrian_venv
            env.pop("PYTHONHOME", None)
            env["PYTHONNOUSERSITE"] = "1"
            
            # CRITICAL: Disable expandable_segments to avoid PyTorch internal assertion failure
            # "!block->expandable_segment_ INTERNAL ASSERT FAILED"
            # We also clear max_split_size_mb to use stable defaults in the older environment.
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
            
            # Start the sidecar subprocess
            abs_file_path = os.path.abspath(__file__)
            print(f"Actor: Sidecar script path: {abs_file_path}", flush=True)
            
            try:
                self._proc = subprocess.Popen(
                    [python_exe, "-u", abs_file_path, "--sidecar"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, # Merge stderr into stdout for easy reading
                    env=env,
                    text=True,
                    bufsize=1
                )
            except Exception as e:
                print(f"Actor: FATAL: Failed to spawn sidecar subprocess: {e}", flush=True)
                raise
            
            # Send initialization command
            print("Actor: Sending initialize command to sidecar...", flush=True)
            self._send({"command": "initialize", "cfg": self._cfg_dict})
            
            # Wait for "ok" response
            print("Actor: Waiting for sidecar 'ok' response...", flush=True)
            resp = self._recv()
            if resp.get("status") == "ok":
                self._initialized = True
                print("Actor: Sidecar initialized successfully", flush=True)
                LOG.info("Actor: Sidecar initialized successfully")
            else:
                error = resp.get("error", "Unknown error during sidecar init")
                print(f"Actor: Sidecar init failed: {error}", flush=True)
                LOG.error(f"Actor: Sidecar init failed: {error}")
                raise RuntimeError(f"Cambrian sidecar failed to initialize: {error}")

        def _send(self, msg: Dict[str, Any]):
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()

        def _recv(self) -> Dict[str, Any]:
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    return {"status": "error", "error": "Sidecar process exited unexpectedly"}
                
                line = line.strip()
                if not line:
                    continue

                # Protocol messages MUST be JSON dictionaries.
                # If it doesn't start with '{', it's definitely a log message.
                if not line.startswith("{"):
                    # Use print() instead of LOG.info() so it appears in Ray's stdout
                    print(f"[Sidecar] {line}", flush=True)
                    continue

                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        return data
                    else:
                        # It's valid JSON (like a float or int) but not our dictionary protocol
                        print(f"[Sidecar JSON] {line}", flush=True)
                except json.JSONDecodeError:
                    print(f"[Sidecar] {line}", flush=True)

        def evaluate(
            self,
            df: pd.DataFrame,
            system_prompt: str = "",
            user_template: str = "",
            cfg_override: Optional[Any] = None,
        ) -> List[Dict[str, Any]]:
            """Delegate evaluation to the Python 3.11 sidecar."""
            if not self._initialized:
                self.initialize()
                
            # Serialize DataFrame to records for transport.
            # CRITICAL: We only send essential columns to avoid memory/IPC bottlenecks with large datasets.
            essential_cols = ["sample_id", "prompt", "image_path", "image_url", "image_base64", "path"]
            available_cols = [c for c in essential_cols if c in df.columns]
            
            # If no essential columns found, we might need some others for context, 
            # but usually these are enough.
            serializable_df = df[available_cols].copy()
            df_records = serializable_df.to_dict(orient="records")
            
            # Use pre-resolved config if provided, otherwise convert here
            resolved_cfg = cfg_override
            if isinstance(cfg_override, DictConfig):
                resolved_cfg = OmegaConf.to_container(cfg_override, resolve=True)
            
            # Split into chunks for IPC to avoid pipe buffer issues and memory spikes
            # 10,000 records is roughly 5-10MB of JSON, safe for most pipes.
            ipc_chunk_size = 1000
            all_results = []
            
            total_records = len(df_records)
            print(f"Actor: Delegating {total_records} records to sidecar in chunks of {ipc_chunk_size}...", flush=True)
            
            for start_idx in range(0, total_records, ipc_chunk_size):
                end_idx = min(start_idx + ipc_chunk_size, total_records)
                chunk = df_records[start_idx:end_idx]
                
                print(f"Actor: Sending chunk {start_idx//ipc_chunk_size + 1} ({len(chunk)} records)...", flush=True)
                self._send({
                    "command": "evaluate",
                    "df_records": chunk,
                    "system_prompt": system_prompt,
                    "user_template": user_template,
                    "cfg_override": resolved_cfg
                })
                
                print(f"Actor: Waiting for chunk {start_idx//ipc_chunk_size + 1} results...", flush=True)
                resp = self._recv()
                if resp.get("status") == "ok":
                    all_results.extend(resp.get("results", []))
                else:
                    error = resp.get("error", "Unknown error during chunk evaluation")
                    print(f"Actor: Chunk evaluation failed: {error}", flush=True)
                    LOG.error(f"Actor: Chunk evaluation failed: {error}")
                    raise RuntimeError(f"Cambrian sidecar evaluation failed: {error}")
            
            print(f"Actor: Evaluation complete. Collected {len(all_results)} results.", flush=True)
            return all_results

        def shutdown(self):
            if self._proc:
                try:
                    self._send({"command": "shutdown"})
                    self._proc.wait(timeout=5)
                except:
                    self._proc.kill()
                self._proc = None
                self._initialized = False

class PersistentCambrianProcessor:
    """Persistent Cambrian processor that manages multiple isolated Ray Actors."""
    
    def __init__(
        self,
        cfg: DictConfig,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
    ):
        self._base_cfg = deepcopy(cfg)
        self._custom_preprocess = preprocess_fn
        self._custom_postprocess = postprocess_fn
        
        self._actors: List[CambrianInferenceActor] = []
        self._initialized = False
        # Lock is created lazily to avoid serialization issues
        self._lock: Optional[threading.Lock] = None
        
        # Concurrency determined by model config, defaulting to available GPUs
        # Priority:
        # 1. model.concurrency (if > 0)
        # 2. detected number of GPUs / tensor_parallel_size
        # 3. fallback to 1
        conf_concurrency = getattr(cfg.model, "concurrency", 0)
        tp_size = max(1, int(getattr(getattr(cfg.model, "engine_kwargs", {}), "tensor_parallel_size", 1) or 1))
        
        if conf_concurrency and conf_concurrency > 0:
            self._concurrency = conf_concurrency
        else:
            from .vqa import _detect_num_gpus
            num_gpus = _detect_num_gpus()
            self._concurrency = max(1, num_gpus // tp_size)
        
        self._tp_size = tp_size

    def _get_lock(self) -> threading.Lock:
        """Get or create the instance lock (lazy initialization for serialization safety)."""
        if getattr(self, "_lock", None) is None:
            self._lock = threading.Lock()
        assert self._lock is not None
        return self._lock

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare state for pickling - exclude non-serializable objects."""
        state = self.__dict__.copy()
        state["_lock"] = None
        state["_actors"] = []
        state["_initialized"] = False
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)
        self._lock = None
        self._actors = []
        self._initialized = False

    @classmethod
    def get_or_create(
        cls,
        cfg: DictConfig,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
    ) -> "PersistentCambrianProcessor":
        model_source = getattr(cfg.model, "model_source", "cambrian-13b")
        # Include concurrency in cache key to ensure we don't return a processor
        # with the wrong number of actors if the config changes.
        concurrency = getattr(cfg.model, "concurrency", 0)
        key = f"cambrian:{model_source}:c{concurrency}"
        
        with _CACHE_LOCK:
            if key in _PROCESSOR_CACHE:
                cached = _PROCESSOR_CACHE[key]
                return cached
            
            processor = cls(cfg, preprocess_fn, postprocess_fn)
            _PROCESSOR_CACHE[key] = processor
            return processor

    def initialize(self) -> None:
        """Spawn the stable Ray Actors."""
        if self._initialized and self._actors:
            return
            
        with self._get_lock():
            if self._initialized and self._actors:
                return
                
            LOG.info(f"Spawning {self._concurrency} stable Ray Actor(s) for Cambrian bridge...")
            
            try:
                if not ray.is_initialized():
                    print("Processor: Initializing Ray...", flush=True)
                    ray.init(ignore_reinit_error=True)
                
                # Resolve config to a plain dict in the main process (where Hydra resolvers are registered)
                # to avoid "Unsupported interpolation type hydra" errors in the Ray Actor.
                print("Processor: Resolving config...", flush=True)
                resolved_cfg = OmegaConf.to_container(self._base_cfg, resolve=True)
                
                # Note: We do NOT set a virtualenv here. We want the Actor to run in the
                # stable project environment (3.12) so it can manage the 3.11 sidecar.
                # We also avoid naming the actor to ensure a fresh instance per run.
                print(f"Processor: Spawning {self._concurrency} Ray Actor(s)...", flush=True)
                self._actors = [
                    CambrianInferenceActor.options(
                        num_gpus=self._tp_size,
                    ).remote(resolved_cfg)
                    for _ in range(self._concurrency)
                ]
                
                print(f"Processor: Waiting for {len(self._actors)} Actor(s) initialization...", flush=True)
                ray.get([actor.initialize.remote() for actor in self._actors])
                
                self._initialized = True
                print("Processor: Initialization complete.", flush=True)
                LOG.info(f"{len(self._actors)} Cambrian bridge actor(s) spawned and sidecar(s) initialized")
            except Exception as e:
                print(f"Processor: FATAL: Failed to spawn or initialize Cambrian bridge actor: {e}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                LOG.error(f"Failed to spawn Cambrian bridge actor: {e}")
                raise

    def evaluate(
        self,
        df: pd.DataFrame,
        system_prompt: str = "",
        user_template: str = "",
        cfg: Optional[DictConfig] = None,
    ) -> pd.DataFrame:
        """Delegate evaluation to the bridge actors in parallel."""
        if not self._initialized:
            self.initialize()
            
        # Resolve config override in the main process (where Hydra resolvers are registered)
        resolved_cfg = None
        if cfg is not None:
            resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
            
        if not self._actors:
            raise RuntimeError("No actors available for evaluation")

        try:
            # Split the dataframe into chunks for each actor
            n_actors = len(self._actors)
            import numpy as np
            chunks = np.array_split(df, n_actors)
            
            print(f"Processor: Parallel evaluation across {n_actors} actors...", flush=True)
            
            futures = []
            for i, actor in enumerate(self._actors):
                if len(chunks[i]) > 0:
                    futures.append(actor.evaluate.remote(
                        chunks[i], system_prompt, user_template, resolved_cfg
                    ))
            
            # Wait for all results
            results_nested = ray.get(futures)
            
            # Combine results
            combined_results = []
            for res_list in results_nested:
                combined_results.extend(res_list)
                
            return pd.DataFrame(combined_results)
        except Exception as e:
            LOG.error(f"Cambrian bridge evaluation failed: {e}")
            raise

    def shutdown(self) -> None:
        """Kill the bridge actors."""
        with self._get_lock():
            if self._actors:
                LOG.info(f"Shutting down {len(self._actors)} Cambrian bridge actor(s)")
                try:
                    ray.get([actor.shutdown.remote() for actor in self._actors])
                except Exception as e:
                    LOG.warning(f"Error during actor shutdown: {e}")
                
                for actor in self._actors:
                    ray.kill(actor)
                self._actors = []
                self._initialized = False

# =============================================================================
# SIDECAR WORKER CODE (Runs in Python 3.11 environment)
# =============================================================================

def run_sidecar_loop():
    """Main loop for the Python 3.11 Cambrian inference process."""
    import sys
    import os
    import torch
    from PIL import Image
    
    # We must ensure Cambrian source is in path inside the subprocess
    project_root = "/share/pierson/matt/mllmsci"
    cambrian_path = os.path.join(project_root, "sub/cambrian")
    if cambrian_path not in sys.path:
        sys.path.insert(0, cambrian_path)

    model = None
    tokenizer = None
    image_processor = None
    context_len = None
    base_cfg = None

    def _reply(msg: Dict[str, Any]):
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    def _sidecar_sanitize_prompt(value: Any, cfg: Dict[str, Any]) -> str:
        """Local helper for prompt sanitization in sidecar."""
        default_prompt = "What do you see in this image?"
        try:
            data_cfg = cfg.get("data", {})
            candidate = data_cfg.get("default_prompt")
            if isinstance(candidate, str) and candidate.strip():
                default_prompt = candidate.strip()
        except Exception:
            pass
            
        if value is None:
            return default_prompt
        if isinstance(value, str):
            sanitized = value.strip()
        else:
            sanitized = str(value).strip()
        return sanitized or default_prompt

    def _ensure_pil_image(image_source: Any) -> Optional[Image.Image]:
        import io
        import base64
        import numpy as np
        try:
            if isinstance(image_source, Image.Image):
                return image_source.convert("RGB")
            if isinstance(image_source, str):
                if image_source.startswith("data:image"):
                    header, data = image_source.split(",", 1)
                    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
                elif os.path.exists(image_source):
                    return Image.open(image_source).convert("RGB")
            # If it's a list (possibly a numpy array serialized to list by JSON)
            if isinstance(image_source, (list, tuple)):
                return Image.fromarray(np.array(image_source).astype('uint8')).convert("RGB")
            if isinstance(image_source, np.ndarray):
                return Image.fromarray(image_source).convert("RGB")
            return None
        except Exception as e:
            print(f"Sidecar: Error loading image: {e}", flush=True)
            return None

    for line in sys.stdin:
        try:
            req = json.loads(line)
            cmd = req.get("command")

            if cmd == "initialize":
                from cambrian.model.builder import load_pretrained_model
                from cambrian.mm_utils import get_model_name_from_path
                
                base_cfg = req.get("cfg", {})
                model_cfg = base_cfg.get("model", {})
                model_path = model_cfg.get("model_source")
                model_name = get_model_name_from_path(model_path)
                
                # Extract Cambrian-specific settings from config
                cambrian_cfg = model_cfg.get("cambrian", {})
                use_flash_attn = cambrian_cfg.get("use_flash_attn", True)  # Default to True for performance
                
                # Check if flash_attn is actually available before enabling
                if use_flash_attn:
                    try:
                        import flash_attn
                        print(f"Sidecar: flash_attn v{flash_attn.__version__} found, enabling Flash Attention 2", flush=True)
                    except ImportError:
                        print("Sidecar: WARNING: flash_attn not installed, falling back to standard attention", flush=True)
                        print("Sidecar: To enable Flash Attention 2, run: pip install flash-attn --no-build-isolation", flush=True)
                        use_flash_attn = False
                
                print(f"Sidecar: Loading weights for {model_name} from {model_path}...", flush=True)
                print(f"Sidecar: Flash Attention 2 enabled: {use_flash_attn}", flush=True)
                
                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    model_path=model_path,
                    model_base=None,
                    model_name=model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    use_flash_attn=use_flash_attn,
                )
                
                print("Sidecar: Weights loaded. Setting up tokenizer...", flush=True)
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
                
                print("Sidecar: Initialization complete.", flush=True)
                _reply({"status": "ok"})

            elif cmd == "evaluate":
                from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
                from cambrian.conversation import conv_templates
                from cambrian.mm_utils import tokenizer_image_token, process_images
                
                df_records = req.get("df_records", [])
                system_prompt = req.get("system_prompt")
                user_template = req.get("user_template")
                eval_cfg = req.get("cfg_override") or base_cfg
                
                print(f"Sidecar: Evaluating batch of {len(df_records)} samples...", flush=True)
                
                # Determine conversation mode
                model_name_str = eval_cfg.get("model", {}).get("model_name", "cambrian-13b").lower()
                if "llama-3" in model_name_str or "8b" in model_name_str:
                    conv_mode = "llama_3"
                elif "phi3" in model_name_str:
                    conv_mode = "phi3"
                elif "34b" in model_name_str:
                    conv_mode = "chatml_direct"
                else:
                    conv_mode = "vicuna_v1"
                
                # Get Cambrian-specific settings
                cambrian_cfg = eval_cfg.get("model", {}).get("cambrian", {})
                inference_batch_size = cambrian_cfg.get("inference_batch_size", 4)
                gc_frequency = cambrian_cfg.get("gc_frequency", 10)
                max_new_tokens = eval_cfg.get("sampling_params_vqa", {}).get("max_tokens", 128)
                
                print(f"Sidecar: Using TRUE BATCHED inference with batch_size={inference_batch_size}", flush=True)
                print(f"Sidecar: GC frequency: every {gc_frequency} batches", flush=True)
                
                results = []
                total_batches = (len(df_records) + inference_batch_size - 1) // inference_batch_size
                batch_times = []
                
                for batch_idx in range(0, len(df_records), inference_batch_size):
                    batch_start_time = time.time()
                    chunk = df_records[batch_idx : batch_idx + inference_batch_size]
                    current_batch_num = batch_idx // inference_batch_size + 1
                    
                    # ========== PHASE 1: Load all images in batch ==========
                    batch_images = []
                    batch_rows = []
                    batch_prompts = []
                    
                    for row in chunk:
                        image_src = row.get("image") or row.get("image_path") or row.get("image_url") or row.get("path")
                        pil_image = _ensure_pil_image(image_src)
                        if not pil_image:
                            print(f"Sidecar: WARNING: Failed to load image for sample {row.get('sample_id')}. Skipping.", flush=True)
                            continue
                        
                        # Build prompt for this sample
                        qs = user_template or _sidecar_sanitize_prompt(row.get("prompt"), eval_cfg)
                        if model.config.mm_use_im_start_end:
                            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                        else:
                            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                        
                        conv = conv_templates[conv_mode].copy()
                        sys_p = system_prompt or eval_cfg.get("prompt", {}).get("system", "")
                        if sys_p:
                            conv.system = sys_p
                        conv.append_message(conv.roles[0], qs)
                        conv.append_message(conv.roles[1], None)
                        prompt_text = conv.get_prompt()
                        
                        batch_images.append(pil_image)
                        batch_rows.append(row)
                        batch_prompts.append(prompt_text)
                    
                    if not batch_images:
                        print(f"Sidecar: WARNING: Batch {current_batch_num} had no loadable images! Skipping.", flush=True)
                        continue
                    
                    # Log image loading confirmation and first prompt for sanity check
                    if batch_idx == 0:
                        first_img = batch_images[0]
                        first_path = batch_rows[0].get("image_path", "unknown")
                        print(f"Sidecar: ✓ IMAGE LOADED: {first_path} -> {first_img.size[0]}x{first_img.size[1]} pixels", flush=True)
                        print(f"Sidecar: SAMPLE PROMPT: {batch_prompts[0][:200]}...", flush=True)
                    
                    actual_batch_size = len(batch_images)
                    
                    # ========== PHASE 2: Batch process through vision encoders ==========
                    # This is the KEY optimization - process_images handles all 4 encoders at once
                    image_sizes = [img.size for img in batch_images]
                    image_tensors = process_images(batch_images, image_processor, model.config)
                    image_tensors = [t.to(device=model.device, dtype=torch.float16) for t in image_tensors]
                    
                    # ========== PHASE 3: Batch tokenization with padding ==========
                    # Tokenize all prompts
                    input_ids_list = [
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                        for prompt in batch_prompts
                    ]
                    
                    # Pad to same length for batching
                    max_len = max(ids.shape[0] for ids in input_ids_list)
                    padded_input_ids = []
                    attention_masks = []
                    
                    for ids in input_ids_list:
                        pad_len = max_len - ids.shape[0]
                        if pad_len > 0:
                            # Left-pad with pad_token_id
                            padding = torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype)
                            padded_ids = torch.cat([padding, ids])
                            mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(ids.shape[0], dtype=torch.long)])
                        else:
                            padded_ids = ids
                            mask = torch.ones(ids.shape[0], dtype=torch.long)
                        padded_input_ids.append(padded_ids)
                        attention_masks.append(mask)
                    
                    batch_input_ids = torch.stack(padded_input_ids).to(model.device)
                    batch_attention_mask = torch.stack(attention_masks).to(model.device)
                    
                    # ========== PHASE 4: Single batched generate call ==========
                    with torch.inference_mode():
                        output_ids = model.generate(
                            batch_input_ids,
                            attention_mask=batch_attention_mask,
                            images=image_tensors,
                            image_sizes=image_sizes,
                            do_sample=False,
                            temperature=0,
                            num_beams=1,
                            max_new_tokens=max_new_tokens,
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    
                    # ========== PHASE 5: Decode outputs ==========
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    
                    for idx, (row, output_text) in enumerate(zip(batch_rows, outputs)):
                        output_text = output_text.strip()
                        
                        # Prepare result record
                        res = row.copy()
                        for col in ["image", "image_array", "image_data"]:
                            if col in res:
                                del res[col]
                        
                        res["answer"] = output_text
                        res["model_response"] = output_text
                        
                        if not output_text:
                            print(f"Sidecar: WARNING: Empty output for sample {res.get('sample_id')}", flush=True)
                        
                        results.append(res)
                    
                    # Cleanup tensors
                    del batch_input_ids, batch_attention_mask, image_tensors, output_ids
                    
                    # Track timing
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    throughput = actual_batch_size / batch_time
                    
                    # Progress logging (every batch for visibility)
                    if current_batch_num % 1 == 0:
                        avg_time = sum(batch_times[-10:]) / len(batch_times[-10:])
                        print(f"Sidecar: Batch {current_batch_num}/{total_batches} | "
                              f"{actual_batch_size} imgs in {batch_time:.2f}s | "
                              f"{throughput:.2f} img/s | Avg: {avg_time:.2f}s/batch", flush=True)
                    
                    # Periodic GC (reduced frequency for performance)
                    if current_batch_num % gc_frequency == 0:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                
                # Final stats
                total_time = sum(batch_times)
                total_images = len(results)
                avg_throughput = total_images / total_time if total_time > 0 else 0
                print(f"Sidecar: Evaluation complete. {total_images} images in {total_time:.1f}s "
                      f"({avg_throughput:.2f} img/s average)", flush=True)
                
                _reply({"status": "ok", "results": results})

            elif cmd == "shutdown":
                _reply({"status": "ok"})
                sys.exit(0)

        except Exception as e:
            import traceback
            _reply({"status": "error", "error": str(e), "traceback": traceback.format_exc()})

if __name__ == "__main__":
    if "--sidecar" in sys.argv:
        print("Sidecar: Process starting...", flush=True)
        try:
            run_sidecar_loop()
        except Exception as e:
            import traceback
            print(f"Sidecar: Fatal error in loop: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            sys.exit(1)
        print("Sidecar: Process exiting normally.", flush=True)

