"""Shared vLLM direct inference utility with multimodal support.

Direct vLLM LLM.generate() calls. Designed for single-machine multi-GPU setups.

Multimodal extension: when messages contain image content blocks
(``{"type": "image", "image": pil_img}``), images are extracted and passed
to vLLM via ``multi_modal_data={"image": images}`` on each prompt.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import pandas as pd
from omegaconf import OmegaConf


def _remap_lora_keys_for_vlm(lora_path: str, model_source: str, stage_name: str) -> str:
    """Remap LoRA adapter keys from CausalLM to VLM prefix if needed.

    Adapters trained on AutoModelForCausalLM have keys like
    ``base_model.model.model.layers.X...`` which vLLM parses to
    ``model.layers.X...``.  But VLM architectures (e.g.
    Qwen3_5ForConditionalGeneration) expect ``model.language_model.layers.X...``.

    If the base model uses a ``language_model`` prefix and the adapter does not,
    creates a remapped copy of the adapter in a ``_vlm_remapped/`` subdirectory.
    Returns the (possibly new) lora_path.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    import glob

    # Quick check: does the base model use language_model prefix?
    sf_files = sorted(glob.glob(os.path.join(model_source, "*.safetensors")))
    if not sf_files:
        return lora_path
    with safe_open(sf_files[0], framework="pt") as f:
        base_keys = f.keys()
        has_lm_prefix = any("language_model.layers." in k for k in base_keys)
    if not has_lm_prefix:
        return lora_path

    # Check adapter keys
    adapter_sf = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_sf):
        return lora_path
    with safe_open(adapter_sf, framework="pt") as f:
        adapter_keys = list(f.keys())
    # After vLLM strips "base_model.model.", keys become "model.layers.X...".
    # We need them to be "model.language_model.layers.X..." instead.
    needs_remap = any(
        k.startswith("base_model.model.model.layers.") for k in adapter_keys
    ) and not any(
        "language_model.layers." in k for k in adapter_keys
    )
    if not needs_remap:
        return lora_path

    # Create remapped adapter
    remapped_dir = os.path.join(lora_path, "_vlm_remapped")
    remapped_sf = os.path.join(remapped_dir, "adapter_model.safetensors")
    if os.path.exists(remapped_sf):
        print(f"[{stage_name}] Using cached VLM-remapped LoRA: {remapped_dir}")
        return remapped_dir

    print(f"[{stage_name}] Remapping LoRA keys: model.layers → model.language_model.layers")
    os.makedirs(remapped_dir, exist_ok=True)

    # Remap and save weights
    import torch
    tensors = {}
    with safe_open(adapter_sf, framework="pt") as f:
        for key in f.keys():
            new_key = key.replace(
                "base_model.model.model.layers.",
                "base_model.model.model.language_model.layers.",
            )
            tensors[new_key] = f.get_tensor(key)
    save_file(tensors, remapped_sf)

    # Copy adapter_config.json and other metadata
    import shutil
    for fname in ("adapter_config.json", "tokenizer_config.json", "tokenizer.json",
                  "chat_template.jinja", "README.md"):
        src = os.path.join(lora_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, remapped_dir)

    print(f"[{stage_name}] VLM-remapped LoRA saved to {remapped_dir} "
          f"({len(tensors)} tensors)")
    return remapped_dir


def _fallback_strip_reasoning(text: str) -> str:
    """Fallback regex-based stripping of reasoning/thinking blocks.

    Used only when no family-specific vLLM reasoning parser is available or
    the parser fails. See ``_split_reasoning`` for the primary path.

    Handles multiple formats used by different model families:
    - ``<think>...</think>`` (Qwen3+, DeepSeek-R1, open-source reasoning models)
    - ``<|begin_of_thought|>...<|end_of_thought|>`` (context-reasoner-ppo, some PPO models)

    Also handles unterminated blocks (model ran out of tokens mid-reasoning).
    Returns the remaining text, stripped.
    """
    # <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"<think>[\s\S]*$", "", text)
    # <|begin_of_thought|...end_of_thought|> (with optional trailing ] or >)
    text = re.sub(r"<\|begin_of_thought\|[\s\S]*?<\|end_of_thought\|[>\]\s]*", "", text)
    text = re.sub(r"<\|begin_of_thought\|[\s\S]*$", "", text)
    return text.strip()


# Backwards-compat alias — imported by dagspaces/grpo_training/stages/rewards.py.
# Prefer `_split_reasoning` for new code.
_strip_think_blocks = _fallback_strip_reasoning


def _detect_reasoning_parser(model_source: str) -> Optional[str]:
    """Map a model path to the vLLM reasoning-parser name for that family.

    Returns a parser name registered in ``vllm.reasoning.ReasoningParserManager``,
    or ``None`` for non-thinking families (Phi-4, Llama, Gemma-3, etc.) where
    no reasoning extraction is needed.
    """
    s = (model_source or "").lower()
    # Order matters — check more specific names first.
    if "gemma-4" in s or "gemma4" in s:
        return "gemma4"
    if "gpt-oss" in s:
        return "gptoss"
    if "deepseek-r1" in s or "deepseek_r1" in s or "deepseek-v3" in s:
        return "deepseek_r1"
    if "qwen3" in s:  # covers qwen3, qwen3.5, qwen3-vl, etc.
        return "qwen3"
    # Non-thinking families: Phi-4, Llama-3.x, Gemma-3, Qwen2.5, OpenThinker (custom tags → regex).
    return None


def _split_reasoning(
    text: str,
    model_source: str,
    thinking_enabled: bool,
    tokenizer,
) -> Tuple[str, str]:
    """Split model output into ``(reasoning, content)``.

    Primary path: vLLM's family-specific ``ReasoningParser``. These parsers
    understand the exact reasoning format for each architecture (Qwen3
    ``<think>...</think>``, Gemma-4 ``thought\\n...\\n``, etc.) and are
    maintained upstream alongside each model's chat template.

    Fallback path: regex (``_fallback_strip_reasoning``) when no parser
    matches the model family, the parser fails, or the parser returns
    content that still contains raw reasoning tags.

    Args:
        text: raw decoded model output.
        model_source: path or identifier used to pick a parser.
        thinking_enabled: whether the chat template was configured with
            thinking on — passed to the parser so it classifies truncated
            output correctly (unterminated ``<think>`` is reasoning when
            enabled, content when disabled).
        tokenizer: the tokenizer used for generation (parsers need it).

    Returns:
        ``(reasoning, content)`` — either may be the empty string.
    """
    if not text:
        return "", ""

    parser_name = _detect_reasoning_parser(model_source)
    if parser_name is not None:
        try:
            from vllm.reasoning import ReasoningParserManager
            parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
            parser = parser_cls(
                tokenizer,
                chat_template_kwargs={"enable_thinking": thinking_enabled},
            )
            reasoning, content = parser.extract_reasoning(text, None)
            reasoning = (reasoning or "").strip()
            content = (content or "").strip()
            # Safety: if parser handed back content that still contains raw
            # reasoning tags, something went wrong — fall through to regex.
            if "<think>" not in content and "</think>" not in content:
                return reasoning, content
        except Exception:
            pass  # fall through

    # Fallback path.
    content = _fallback_strip_reasoning(text)
    m = re.search(r"<think>([\s\S]*?)</think>", text)
    if m:
        reasoning = m.group(1).strip()
    else:
        m2 = re.search(r"<\|begin_of_thought\|([\s\S]*?)<\|end_of_thought\|", text)
        reasoning = m2.group(1).strip() if m2 else ""
    return reasoning, content


# ---------------------------------------------------------------------------
# GPU / environment helpers
# ---------------------------------------------------------------------------

def _is_multimodal_model(model_source: str, cfg=None) -> bool:
    """Detect if model supports multimodal (vision) inputs.

    Checks:
    1. Explicit config flag: runtime.multimodal_enabled or model.multimodal
    2. Known VLM model name patterns (qwen-vl, internvl, phi-3-vision, etc.)
    3. Model config file presence of vision-related architectures
    """
    if cfg is not None:
        try:
            if hasattr(cfg, "runtime") and hasattr(cfg.runtime, "multimodal_enabled"):
                return bool(cfg.runtime.multimodal_enabled)
        except Exception:
            pass
        try:
            if hasattr(cfg, "model") and hasattr(cfg.model, "multimodal"):
                return bool(cfg.model.multimodal)
        except Exception:
            pass

    model_lower = (model_source or "").lower()
    multimodal_patterns = [
        r"qwen.*vl", r"qwen2.*vl", r"qwen3.*vl",
        r"internvl", r"intern.*vl",
        r"phi.*vision", r"phi-3.*v",
        r"llava", r"cambrian",
        r"smolvlm", r"smol.*vlm",
        r"cogvlm", r"cogagent",
        r"minicpm.*v", r"glm.*v",
        r"gemma.*it.*vision", r"gemma-4",
        r"multimodal",
    ]
    return any(re.search(pattern, model_lower, re.IGNORECASE) for pattern in multimodal_patterns)


def _extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    """Extract PIL images from chat message content blocks.

    Messages may contain multimodal content blocks like::

        {"role": "user", "content": [
            {"type": "image", "image": <PIL.Image>},
            {"type": "text", "text": "What do you see?"}
        ]}

    Returns a list of PIL Image objects found across all messages.
    """
    images = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    img = block.get("image")
                    if img is not None:
                        images.append(img)
    return images


def _flatten_messages_for_template(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten multimodal message content blocks to text-only for chat template.

    vLLM's tokenizer.apply_chat_template expects text-only content or handles
    multimodal content via special tokens. For models that support it, we pass
    messages as-is. For others, we flatten to text-only and handle images
    separately via multi_modal_data.
    """
    flat = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            # Extract text parts only for the chat template
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image":
                        # Insert image placeholder token for models that use them
                        text_parts.append("<image>")
                elif isinstance(block, str):
                    text_parts.append(block)
            flat.append({**msg, "content": "\n".join(text_parts)})
        else:
            flat.append(msg)
    return flat


def get_pcie_nccl_env_vars() -> Dict[str, str]:
    """Return NCCL environment variables required for PCIe-only GPUs (no NVLink)."""
    return {
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
        "NCCL_CUMEM_HOST_ENABLE": "0",
        "NCCL_DEBUG": "WARN",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }


def get_vllm_runtime_env_vars() -> Dict[str, str]:
    """Return the shared runtime environment for in-process vLLM launches.

    Note: VLLM_USE_V1 and VLLM_ENABLE_V1_MULTIPROCESSING were removed in
    vLLM 0.10+ (V1 is now the only engine). We no longer set them.
    """
    return {
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True",
        "CUDA_LAUNCH_BLOCKING": "0",
    }


def _run_nvidia_smi(args: List[str]) -> List[str]:
    """Run nvidia-smi without importing torch in the parent process."""
    try:
        result = subprocess.run(
            ["nvidia-smi", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def detect_num_gpus() -> int:
    """Detect the number of GPUs available.

    Resolution order:
    1. MLLMSCI_TENSOR_PARALLEL_SIZE env override
    2. CUDA_VISIBLE_DEVICES
    3. SLURM GPU env vars
    4. nvidia-smi -L
    5. Fallback to 1
    """
    # Env override
    tp_env = os.environ.get("MLLMSCI_TENSOR_PARALLEL_SIZE")
    if tp_env:
        try:
            val = int(tp_env)
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass

    # CUDA_VISIBLE_DEVICES
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            ids = [x.strip() for x in cuda_visible.split(",") if x.strip()]
            if ids:
                return len(ids)
    except Exception:
        pass

    # SLURM
    try:
        slurm_gpus = (
            os.environ.get("SLURM_GPUS_PER_NODE")
            or os.environ.get("SLURM_GPUS_ON_NODE")
        )
        if slurm_gpus:
            if ":" in slurm_gpus:
                return int(slurm_gpus.split(":")[-1])
            return int(slurm_gpus)
    except Exception:
        pass

    # nvidia-smi
    gpu_lines = _run_nvidia_smi(["-L"])
    if gpu_lines:
        return len(gpu_lines)

    return 1


def detect_gpu_type() -> str:
    """Detect GPU type, returning a normalised string like 'rtx_a6000'."""
    names = _run_nvidia_smi(["--query-gpu=name", "--format=csv,noheader,nounits"])
    if names:
        name = names[0].lower()
        if "a6000" in name:
            return "rtx_a6000"
        if "a5000" in name:
            return "rtx_a5000"
        if "a100" in name:
            return "a100"
        if "h100" in name:
            return "h100"
        if "v100" in name:
            return "v100"
        if "a40" in name:
            return "a40"
        if "rtx" in name:
            return "rtx_generic"
    return "unknown"


def apply_gpu_aware_settings(engine_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Set max_num_seqs based on GPU type if not already specified.

    Returns the GPU-type defaults dict (may be empty).
    """
    GPU_DEFAULTS = {
        "rtx_a6000": {"batch_size": 4, "max_num_seqs": 4},
        "rtx_a5000": {"batch_size": 2, "max_num_seqs": 2},
        "a100": {"batch_size": 8, "max_num_seqs": 8},
        "h100": {"batch_size": 16, "max_num_seqs": 16},
        "v100": {"batch_size": 4, "max_num_seqs": 4},
        "a40": {"batch_size": 4, "max_num_seqs": 4},
    }
    gpu_type = detect_gpu_type()
    defaults = GPU_DEFAULTS.get(gpu_type, {})
    if defaults and "max_num_seqs" not in engine_kwargs:
        engine_kwargs["max_num_seqs"] = defaults["max_num_seqs"]
        print(f"[vllm_inference] Auto-set max_num_seqs={defaults['max_num_seqs']} for {gpu_type}")
    return defaults


def filter_vllm_engine_kwargs(ek: Dict[str, Any]) -> Dict[str, Any]:
    """Drop engine kwargs not accepted by the installed vLLM LLM class.

    vLLM's LLM.__init__ accepts **kwargs and forwards them to EngineArgs,
    so we check both signatures to build the accepted set.
    """
    try:
        import inspect
        from vllm import LLM as _LLM

        sig = inspect.signature(_LLM.__init__)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

        if has_var_keyword:
            # LLM forwards **kwargs to EngineArgs — check EngineArgs too
            accepted = {k for k in sig.parameters if k != "self"}
            try:
                from vllm.config import EngineArgs
                ea_sig = inspect.signature(EngineArgs.__init__)
                accepted |= {k for k in ea_sig.parameters if k != "self"}
            except ImportError:
                try:
                    from vllm.engine.arg_utils import EngineArgs
                    ea_sig = inspect.signature(EngineArgs.__init__)
                    accepted |= {k for k in ea_sig.parameters if k != "self"}
                except ImportError:
                    # Can't resolve EngineArgs — pass everything through
                    ek = dict(ek)
                    for k in ("concurrency", "batch_size"):
                        ek.pop(k, None)
                    return ek
        else:
            accepted = {k for k in sig.parameters if k != "self"}

        filtered = {k: v for k, v in ek.items() if k in accepted}
        dropped = [k for k in ek if k not in accepted]
        if dropped:
            print(f"[vllm_inference] Dropped unsupported vLLM kwargs: {dropped}")
        return filtered
    except Exception:
        pass
    # Conservative fallback — only drop known non-vLLM keys
    ek = dict(ek)
    for k in ("use_v2_block_manager", "concurrency", "batch_size"):
        ek.pop(k, None)
    return ek


# ---------------------------------------------------------------------------
# Engine kwargs builder
# ---------------------------------------------------------------------------

def _build_engine_kwargs(cfg) -> Dict[str, Any]:
    """Build vLLM LLM constructor kwargs from Hydra config."""
    model_source = str(getattr(cfg.model, "model_source"))
    from omegaconf import OmegaConf as _OC
    _raw_ek = getattr(cfg.model, "engine_kwargs", {})
    ek = _OC.to_container(_raw_ek, resolve=True) if _OC.is_config(_raw_ek) else dict(_raw_ek)

    # Model
    ek["model"] = model_source

    # Tensor parallelism
    if "tensor_parallel_size" not in ek:
        ek["tensor_parallel_size"] = detect_num_gpus()
        print(f"[vllm_inference] Auto-detected {ek['tensor_parallel_size']} GPU(s) for tensor parallelism")

    # GPU-aware tuning
    apply_gpu_aware_settings(ek)

    # Safe defaults
    ek.setdefault("trust_remote_code", True)
    ek.setdefault("distributed_executor_backend", "mp")
    if int(ek.get("tensor_parallel_size", 1) or 1) > 1:
        ek.setdefault("disable_custom_all_reduce", True)

    # AWQ auto-detection
    if "awq" in model_source.lower() and "quantization" not in ek:
        ek["quantization"] = "awq"

    # Multimodal cache + vision encoder DP defaults.  These are only applied
    # if the user has not set them explicitly. Reduces the chance of the
    # mm_processor LRU cache leaking RAM during very large multimodal jobs
    # (see vllm GH issues #15294, #35191) and enables batch-level vision
    # data parallelism when supported by the installed vLLM.
    if _is_multimodal_model(model_source, cfg):
        ek.setdefault("mm_processor_cache_gb", 2)
        ek.setdefault("mm_encoder_tp_mode", "data")

    # Convert nested hf_overrides dicts to a callable that does deep updates.
    # vLLM's config.update() with a dict like {"text_config": {"vocab_size": X}}
    # replaces text_config entirely instead of merging.  A callable gets the
    # PretrainedConfig object and can update nested attributes properly.
    _hf_ov = ek.get("hf_overrides")
    if isinstance(_hf_ov, dict) and any(isinstance(v, dict) for v in _hf_ov.values()):
        def _make_hf_override_fn(overrides):
            def _fn(config):
                for key, val in overrides.items():
                    if isinstance(val, dict) and hasattr(config, key):
                        sub = getattr(config, key)
                        for sk, sv in val.items():
                            setattr(sub, sk, sv)
                    else:
                        setattr(config, key, val)
                return config
            return _fn
        ek["hf_overrides"] = _make_hf_override_fn(_hf_ov)

    # Preserve data_parallel_size — it's consumed by run_vllm_inference() to
    # decide whether to spawn DP workers.  Not passed to LLM() directly
    # (vLLM 0.19 workers get DP config from VLLM_DP_* env vars instead).
    dp_size = ek.pop("data_parallel_size", None)

    # Filter to accepted kwargs
    ek = filter_vllm_engine_kwargs(ek)

    # Re-attach for run_vllm_inference / run_vllm_embed to consume
    if dp_size is not None:
        ek["data_parallel_size"] = dp_size

    # Auto-detect: when not explicitly configured and there are more GPUs
    # than tensor_parallel_size needs, use the surplus for data parallelism.
    if "data_parallel_size" not in ek or int(ek.get("data_parallel_size", 1) or 1) <= 1:
        tp_size = int(ek.get("tensor_parallel_size", 1) or 1)
        total_gpus = detect_num_gpus()
        if total_gpus > tp_size and total_gpus % tp_size == 0:
            auto_dp = total_gpus // tp_size
            ek["data_parallel_size"] = auto_dp
            print(f"[vllm_inference] Auto-detected data_parallel_size={auto_dp} "
                  f"({total_gpus} GPUs / TP={tp_size})")

    return ek


# ---------------------------------------------------------------------------
# SamplingParams builder
# ---------------------------------------------------------------------------

def _resolve_server_url(cfg) -> Optional[str]:
    """Return the vLLM OpenAI-compatible server URL to use, or ``None``.

    Priority (highest wins):
    1. ``cfg.model.vllm_server_url`` (explicit per-run override)
    2. ``VLLM_SERVER_URL`` environment variable (set by ``eval_all``)

    A returned URL should be the base URL (e.g. ``http://host:8000/v1``);
    the trailing ``/v1`` is normalised by the OpenAI client.
    """
    url: Optional[str] = None
    try:
        url = str(getattr(cfg.model, "vllm_server_url", "") or "")
    except Exception:
        url = None
    if not url:
        url = os.environ.get("VLLM_SERVER_URL", "") or None
    return url or None


def _sp_to_openai_kwargs(sp_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Translate our sampling_params dict to OpenAI API kwargs + extra_body.

    Returns ``(kwargs, extra_body)`` where ``kwargs`` are fields accepted
    by ``openai.chat.completions.create`` directly and ``extra_body`` are
    vLLM-specific extensions (``guided_json``, ``top_k``, etc.).
    """
    sp = dict(sp_dict or {})
    kwargs: Dict[str, Any] = {}
    extra_body: Dict[str, Any] = {}

    # Direct-mapping OpenAI params
    for k in ("max_tokens", "temperature", "top_p", "n", "stop",
              "presence_penalty", "frequency_penalty", "seed"):
        if k in sp and sp[k] is not None:
            kwargs[k] = sp[k]

    # vLLM extensions via extra_body
    for k in ("top_k", "min_p", "repetition_penalty", "length_penalty",
              "ignore_eos", "skip_special_tokens"):
        if k in sp and sp[k] is not None:
            extra_body[k] = sp[k]

    # Guided/structured decoding
    guided = sp.get("guided_decoding") or sp.get("structured_output")
    if guided and isinstance(guided, dict):
        if "json" in guided:
            extra_body["guided_json"] = guided["json"]
        if "regex" in guided:
            extra_body["guided_regex"] = guided["regex"]
        if "choice" in guided:
            extra_body["guided_choice"] = guided["choice"]
        if "grammar" in guided:
            extra_body["guided_grammar"] = guided["grammar"]

    return kwargs, extra_body


def _run_server_inference(
    df: "pd.DataFrame",
    cfg,
    preprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    postprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    stage_name: str,
    server_url: str,
) -> "pd.DataFrame":
    """Route inference through an OpenAI-compatible vLLM server.

    Mirrors the contract of the in-process path (see ``run_vllm_inference``):
    the same ``preprocess`` / ``postprocess`` callbacks are invoked, and
    ``generated_text`` / ``generated_reasoning`` / ``usage`` are populated
    on each row before postprocess.

    Concurrency is provided by a thread pool (~32 workers); the server
    handles continuous batching on its end.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI

    # Resolve model name — either the full local path (if the server was
    # launched with --served-model-name pointing at it) or a short name
    # the user configured via model.vllm_served_model_name.
    served_name = ""
    try:
        served_name = str(getattr(cfg.model, "vllm_served_model_name", "") or "")
    except Exception:
        pass
    if not served_name:
        served_name = str(getattr(cfg.model, "model_source", "") or "")

    client = OpenAI(base_url=server_url, api_key="EMPTY", timeout=600.0)
    print(f"[{stage_name}] Server-mode inference → {server_url} (model={served_name})")

    # Chat template kwargs (e.g. enable_thinking)
    ctk_extra: Dict[str, Any] = {}
    try:
        from dagspaces.common.stage_utils import resolve_thinking_mode
        _thinking_enabled = resolve_thinking_mode(cfg.model, default=True)
    except Exception:
        _thinking_enabled = True
    ctk_extra["enable_thinking"] = _thinking_enabled
    _model_source = str(getattr(cfg.model, "model_source", "") or "")

    # Preprocess all rows
    print(f"[{stage_name}] Preprocessing {len(df)} rows...")
    preprocessed_rows: List[Dict[str, Any]] = []
    for row in df.to_dict("records"):
        try:
            preprocessed_rows.append(preprocess(row))
        except Exception as e:
            row["__preprocess_error__"] = str(e)
            preprocessed_rows.append(row)

    # Build request jobs
    def _make_request(idx: int):
        row = preprocessed_rows[idx]
        if "__preprocess_error__" in row:
            return idx, "", "", None, row["__preprocess_error__"]

        messages = row.get("messages") or []
        sp_dict = row.get("sampling_params") or {}
        sp_kwargs, extra_body = _sp_to_openai_kwargs(sp_dict)
        extra_body["chat_template_kwargs"] = ctk_extra

        try:
            resp = client.chat.completions.create(
                model=served_name,
                messages=messages,
                extra_body=extra_body,
                **sp_kwargs,
            )
        except Exception as e:
            return idx, "", "", None, f"request_error: {e}"

        msg = resp.choices[0].message if resp.choices else None
        content = (getattr(msg, "content", None) or "") if msg else ""
        # vLLM reasoning parsers populate reasoning_content when configured
        reasoning = (getattr(msg, "reasoning_content", None) or "") if msg else ""
        if not reasoning:
            # Server didn't parse reasoning — do it client-side.
            reasoning, content = _split_reasoning(
                content, _model_source, _thinking_enabled, tokenizer=None,
            )

        usage = None
        try:
            u = resp.usage
            if u is not None:
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0),
                    "completion_tokens": getattr(u, "completion_tokens", 0),
                    "total_tokens": getattr(u, "total_tokens", 0),
                }
        except Exception:
            pass
        return idx, content.strip(), reasoning.strip(), usage, None

    max_workers = int(os.environ.get("VLLM_SERVER_CLIENT_CONCURRENCY", "32"))
    results_raw: List[Optional[Tuple[int, str, str, Any, Optional[str]]]] = [None] * len(preprocessed_rows)
    print(f"[{stage_name}] Dispatching {len(preprocessed_rows)} requests "
          f"(concurrency={max_workers})...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_make_request, i) for i in range(len(preprocessed_rows))]
        done = 0
        for fut in as_completed(futures):
            idx, content, reasoning, usage, err = fut.result()
            results_raw[idx] = (idx, content, reasoning, usage, err)
            done += 1
            if done % 50 == 0 or done == len(preprocessed_rows):
                print(f"[{stage_name}]   {done}/{len(preprocessed_rows)} complete")

    # Postprocess
    n_errors = sum(1 for r in results_raw if r and r[4])
    if n_errors:
        print(f"[{stage_name}] WARNING: {n_errors} request errors — rows marked with __postprocess_error__")

    out_rows: List[Dict[str, Any]] = []
    for entry in results_raw:
        idx, content, reasoning, usage, err = entry  # type: ignore
        row = preprocessed_rows[idx]
        row["generated_text"] = content
        row["generated_reasoning"] = reasoning
        if usage is not None:
            row["usage"] = usage
        if err:
            row["__request_error__"] = err
        try:
            out_rows.append(postprocess(row))
        except Exception as e:
            row["__postprocess_error__"] = str(e)
            out_rows.append(row)

    print(f"[{stage_name}] Completed server inference, {len(out_rows)} results")
    return pd.DataFrame(out_rows)


def _shutdown_llm(llm: Any, stage_name: str = "vllm") -> None:
    """Explicitly shut down a vLLM ``LLM``'s engine workers.

    Why this is necessary: ``vllm.LLM`` does **not** define ``__del__``, and
    ``v1.LLMEngine.__del__`` only cleans up the DP process group — it does
    not stop the ``EngineCore`` / ``WorkerProc_TP*`` child processes spawned
    by multiprocessing. When the calling function returns, those workers
    keep running; Python's ``multiprocessing._exit_function`` atexit hook
    then blocks at process exit trying to join them, which holds the
    SLURM job open indefinitely until walltime or external cancellation.

    Call this before returning from any inference function that instantiates
    an ``LLM``. Safe to call multiple times; errors are swallowed (cleanup
    must not mask the real return value).
    """
    if llm is None:
        return
    try:
        # Primary path: engine_core_client (V1, multiproc mode).
        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None) if engine is not None else None
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
    except Exception as e:
        print(f"[{stage_name}] engine shutdown warning: {e}", flush=True)
    # Force refcount to 0 and run GC so LLMEngine.__del__ fires for DP group cleanup.
    try:
        import gc
        del llm  # type: ignore[misc]
        gc.collect()
    except Exception:
        pass
    # Best-effort: terminate any multiprocessing children that survived.
    try:
        import multiprocessing as _mp
        survivors = [p for p in _mp.active_children() if p.is_alive()]
        for p in survivors:
            try:
                p.terminate()
            except Exception:
                pass
        if survivors:
            print(f"[{stage_name}] terminated {len(survivors)} surviving mp children", flush=True)
    except Exception:
        pass


def _build_sampling_params(sp_dict: Dict[str, Any]):
    """Convert a plain dict to vLLM SamplingParams, handling structured output.

    Supports both vLLM <=0.11 (GuidedDecodingParams) and >=0.12
    (StructuredOutputsParams) APIs transparently.
    """
    from vllm import SamplingParams

    sp = dict(sp_dict or {})

    # Extract guided_decoding / structured_output if present
    guided = sp.pop("guided_decoding", None) or sp.pop("structured_output", None)
    # Remove keys that aren't valid SamplingParams fields
    for k in ("early_stopping", "length_penalty", "response_format",
              "detokenize"):
        sp.pop(k, None)

    if guided and isinstance(guided, dict):
        # vLLM >=0.12: StructuredOutputsParams replaces GuidedDecodingParams
        try:
            from vllm.sampling_params import StructuredOutputsParams
            sp["structured_outputs"] = StructuredOutputsParams(**guided)
        except ImportError:
            # vLLM <=0.11: fall back to GuidedDecodingParams
            try:
                from vllm.sampling_params import GuidedDecodingParams
                sp["guided_decoding"] = GuidedDecodingParams(**guided)
            except ImportError:
                pass

    return SamplingParams(**sp)


# ---------------------------------------------------------------------------
# Data-parallel inference via multiprocessing (vLLM 0.19+ pattern)
# ---------------------------------------------------------------------------

_DP_WORKER_SCRIPT = r'''
"""Standalone DP worker — spawned as a fresh subprocess per rank.

Each worker is a completely independent LLM instance with its own
CUDA_VISIBLE_DEVICES slice.  No vLLM DP coordination (VLLM_DP_* env vars)
is used — vLLM 0.19 blocks DP for dense (non-MoE) models via ParallelConfig.

Supports multimodal: image file paths are passed alongside prompts and
loaded lazily in the worker process.
"""
import os, pickle, sys, time, traceback

def main():
    task_path = sys.argv[1]
    result_path = sys.argv[2]

    with open(task_path, "rb") as f:
        task = pickle.load(f)

    rank        = task["rank"]
    dp_size     = task["dp_size"]
    engine_kwargs = task["engine_kwargs"]
    prompts     = task["prompts"]
    image_refs  = task["image_refs"]
    sp_dict     = task["sp_dict"]
    stage_name  = task["stage_name"]
    pcie_env    = task["pcie_env"]
    runtime_env = task["runtime_env"]
    is_multimodal  = task["is_multimodal"]

    # Apply env vars BEFORE any CUDA/torch import
    for k, v in {**pcie_env, **runtime_env}.items():
        os.environ.setdefault(k, v)

    # Clear any inherited vLLM DP coordination vars — each worker is
    # fully independent with its own CUDA_VISIBLE_DEVICES slice.
    for var in ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE",
                "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT"):
        os.environ.pop(var, None)

    print(f"[{stage_name}] DP rank {rank}/{dp_size}: starting "
          f"(pid={os.getpid()}, CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
          f"prompts={len(prompts)}, multimodal={is_multimodal})", flush=True)

    try:
        t0 = time.time()
        from vllm import LLM, SamplingParams

        print(f"[{stage_name}] DP rank {rank}/{dp_size}: vLLM imported in "
              f"{time.time() - t0:.1f}s, creating LLM engine...", flush=True)

        t1 = time.time()
        llm = LLM(**engine_kwargs)
        tokenizer = llm.get_tokenizer()
        print(f"[{stage_name}] DP rank {rank}/{dp_size}: LLM created in "
              f"{time.time() - t1:.1f}s, starting generation...", flush=True)

        # Build SamplingParams
        sp = dict(sp_dict or {})
        guided = sp.pop("guided_decoding", None) or sp.pop("structured_output", None)
        for k in ("early_stopping", "length_penalty", "response_format", "detokenize"):
            sp.pop(k, None)
        if guided and isinstance(guided, dict):
            try:
                from vllm.sampling_params import StructuredOutputsParams
                sp["structured_outputs"] = StructuredOutputsParams(**guided)
            except ImportError:
                try:
                    from vllm.sampling_params import GuidedDecodingParams
                    sp["guided_decoding"] = GuidedDecodingParams(**guided)
                except ImportError:
                    pass
        sampling_params = SamplingParams(**sp)

        t2 = time.time()
        if is_multimodal:
            from vllm import TokensPrompt
            from PIL import Image
            mm_prompts = []
            for i, prompt in enumerate(prompts):
                ref = image_refs[i] if image_refs else None
                if ref is not None:
                    token_ids = tokenizer.encode(prompt)
                    img = Image.open(ref).convert("RGB")
                    mm_prompts.append(TokensPrompt(
                        prompt_token_ids=token_ids,
                        multi_modal_data={"image": img},
                    ))
                else:
                    mm_prompts.append(prompt)
            outputs = llm.generate(mm_prompts, sampling_params)
        else:
            outputs = llm.generate(prompts, sampling_params)

        print(f"[{stage_name}] DP rank {rank}/{dp_size}: generation done in "
              f"{time.time() - t2:.1f}s ({len(outputs)} outputs)", flush=True)

        serialised = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            prompt_tokens = len(out.prompt_token_ids) if out.prompt_token_ids else 0
            completion_tokens = (
                len(out.outputs[0].token_ids)
                if out.outputs and out.outputs[0].token_ids else 0
            )
            serialised.append({
                "generated_text": text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })

        with open(result_path, "wb") as f:
            pickle.dump({"rank": rank, "outputs": serialised, "error": None}, f)

        print(f"[{stage_name}] DP rank {rank}/{dp_size}: wrote {len(serialised)} "
              f"results, total elapsed {time.time() - t0:.1f}s", flush=True)

    except Exception:
        tb = traceback.format_exc()
        print(f"[{stage_name}] DP rank {rank}/{dp_size}: FAILED\n{tb}",
              flush=True, file=sys.stderr)
        with open(result_path, "wb") as f:
            pickle.dump({"rank": rank, "outputs": None, "error": tb}, f)
        sys.exit(1)
    finally:
        try:
            if llm is not None:
                engine = getattr(llm, "llm_engine", None)
                core = getattr(engine, "engine_core", None) if engine else None
                if core is not None and hasattr(core, "shutdown"):
                    core.shutdown()
                del llm
        except Exception:
            pass

if __name__ == "__main__":
    main()
'''


_DP_FULL_WORKER_SCRIPT = r'''
"""Full-pipeline DP worker — preprocess + infer + postprocess in-worker.

Processes rows in fixed-size chunks. For each chunk:
  1. preprocess_fn is called per row (cheap — builds the message dict)
  2. images are decoded + resized in parallel CPU threads
  3. llm.chat() is called once on the chunk
  4. results are postprocessed and accumulated
  5. chunk-local objects are dropped before the next iteration

This bounds engine-core memory (no big pre-rendered prompt buffer) and gives
us streaming progress + ETA every ~``log_every`` rows. Critical for large
multimodal jobs (e.g. ~120k pairs per worker) where the previous unbounded
``llm.chat(big_list)`` path spent hours in single-threaded HF processor work
and OOM-killed the engine core.

Each worker receives raw row dicts and cloudpickled preprocess/postprocess
callables. Preprocessing happens locally in the worker, co-located with
the GPU.
"""
import os, pickle, sys, time, traceback

def main():
    task_path = sys.argv[1]
    result_path = sys.argv[2]

    with open(task_path, "rb") as f:
        task = pickle.load(f)

    rank        = task["rank"]
    dp_size     = task["dp_size"]
    engine_kwargs = task["engine_kwargs"]
    rows        = task["rows"]
    stage_name  = task["stage_name"]
    pcie_env    = task["pcie_env"]
    runtime_env = task["runtime_env"]
    model_source = task["model_source"]
    thinking_enabled = task["thinking_enabled"]
    chunk_size  = int(task.get("chunk_size", 64) or 64)
    log_every   = int(task.get("log_every", 1000) or 1000)
    image_max_pixels = task.get("image_max_pixels")  # Optional[int]
    image_load_workers = int(task.get("image_load_workers", 16) or 16)
    flush_every = int(task.get("flush_every", 1000) or 1000)
    streaming_output_dir = task.get("streaming_output_dir")  # Optional[str]
    row_offset  = int(task.get("row_offset", 0) or 0)

    # Deserialize preprocess/postprocess via cloudpickle
    import cloudpickle
    preprocess_fn = cloudpickle.loads(task["preprocess_bytes"])
    postprocess_fn = cloudpickle.loads(task["postprocess_bytes"])

    # Apply env vars BEFORE any CUDA/torch import
    for k, v in {**pcie_env, **runtime_env}.items():
        os.environ.setdefault(k, v)

    for var in ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE",
                "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT"):
        os.environ.pop(var, None)

    print(f"[{stage_name}] DP rank {rank}/{dp_size}: starting "
          f"(pid={os.getpid()}, CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
          f"rows={len(rows)}, chunk_size={chunk_size}, "
          f"flush_every={flush_every}, "
          f"image_max_pixels={image_max_pixels}, "
          f"image_load_workers={image_load_workers}, "
          f"streaming_output_dir={streaming_output_dir})", flush=True)

    if streaming_output_dir:
        os.makedirs(streaming_output_dir, exist_ok=True)

    llm = None
    try:
        t0 = time.time()
        from vllm import LLM, SamplingParams
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor
        import re
        import math

        print(f"[{stage_name}] DP rank {rank}/{dp_size}: vLLM imported in "
              f"{time.time() - t0:.1f}s, creating LLM engine...", flush=True)

        t1 = time.time()
        # Allow loading local images via file:// URLs in llm.chat() (still
        # needed when callers pass image_url blocks; we usually rewrite
        # them to image_pil blocks below).
        engine_kwargs.setdefault("allowed_local_media_path", "/")
        llm = LLM(**engine_kwargs)
        print(f"[{stage_name}] DP rank {rank}/{dp_size}: LLM created in "
              f"{time.time() - t1:.1f}s, processing {len(rows)} rows in "
              f"chunks of {chunk_size}...", flush=True)

        # ─── Helpers ────────────────────────────────────────────────────
        def _strip_thinking(text):
            text = re.sub(r"<think>[\s\S]*?</think>", "", text)
            text = re.sub(r"<think>[\s\S]*$", "", text)
            return text.strip()

        def _extract_reasoning(raw_text):
            reasoning, content = "", raw_text
            if not thinking_enabled:
                return reasoning, content
            if "<think>" in raw_text:
                m = re.search(r"<think>([\s\S]*?)</think>([\s\S]*)", raw_text)
                if m:
                    reasoning = m.group(1).strip()
                    content = m.group(2).strip()
                else:
                    content = _strip_thinking(raw_text)
            elif "</think>" in raw_text:
                parts = raw_text.split("</think>", 1)
                reasoning = parts[0].strip()
                content = parts[1].strip() if len(parts) > 1 else ""
            return reasoning, content

        def _resize_to_max_pixels(img, max_pixels):
            """Downscale a PIL image so width*height <= max_pixels."""
            if max_pixels is None or max_pixels <= 0:
                return img
            w, h = img.size
            cur = w * h
            if cur <= max_pixels:
                return img
            scale = math.sqrt(max_pixels / cur)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            return img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        def _load_one_image(path):
            try:
                with Image.open(path) as im:
                    im.load()
                    img = im.convert("RGB")
                if image_max_pixels:
                    img = _resize_to_max_pixels(img, image_max_pixels)
                return img
            except Exception as e:
                return e  # caller treats Exception as a load failure

        # Collect (msg_block, path) tuples that need lazy loading. The
        # block is the dict we'll mutate in-place to attach the loaded
        # PIL Image. Supports image_url (file://), image (with str path
        # or PIL), and image_pil blocks.
        def _collect_image_blocks(messages):
            tasks = []  # list of (block_dict, str_path)
            for msg in messages:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "image_url":
                        url = (block.get("image_url") or {}).get("url", "")
                        path = url.removeprefix("file://") if isinstance(url, str) else ""
                        if path:
                            tasks.append((block, path))
                    elif btype == "image":
                        val = block.get("image")
                        if isinstance(val, str):
                            path = val.removeprefix("file://")
                            tasks.append((block, path))
                        # if it's already a PIL.Image, leave it; the
                        # _convert_image_blocks pass below will rename
                        # type → image_pil for vLLM.
                    elif btype == "image_pil":
                        pass  # already loaded
            return tasks

        def _convert_image_blocks(messages):
            """Normalize image blocks to vLLM's image_pil format."""
            for msg in messages:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "image" and "image" in block:
                        val = block["image"]
                        if hasattr(val, "size"):  # PIL.Image
                            block["type"] = "image_pil"
                            block["image_pil"] = block.pop("image")
            return messages

        thread_pool = ThreadPoolExecutor(max_workers=image_load_workers)

        def _hydrate_chunk_images(chunk_messages):
            """Decode + resize all image paths for a chunk in parallel."""
            all_tasks = []  # list of (block, path)
            for msgs in chunk_messages:
                all_tasks.extend(_collect_image_blocks(msgs))
            if not all_tasks:
                return 0
            paths = [p for _, p in all_tasks]
            blocks = [b for b, _ in all_tasks]
            results = list(thread_pool.map(_load_one_image, paths))
            n_failed = 0
            for block, img in zip(blocks, results):
                if isinstance(img, Exception):
                    block["__image_load_error__"] = str(img)
                    n_failed += 1
                    continue
                # Replace whatever was here with image_pil
                block.pop("image_url", None)
                block.pop("image", None)
                block["type"] = "image_pil"
                block["image_pil"] = img
            return n_failed

        # ─── First pass: build SamplingParams from the first valid row ──
        # We need a sampling params object before generation. Peek at
        # the first row to extract sampling_params dict — assumes uniform
        # sampling per stage, which is the existing convention.
        sampling_params = None
        first_pp_row = None
        peek_idx = 0
        while peek_idx < len(rows) and first_pp_row is None:
            try:
                first_pp_row = preprocess_fn(rows[peek_idx])
            except Exception:
                first_pp_row = None
            peek_idx += 1
        if first_pp_row is not None:
            _first_sp = first_pp_row.get("sampling_params") or {}
            sp = dict(_first_sp)
            guided = sp.pop("guided_decoding", None) or sp.pop("structured_output", None)
            for k in ("early_stopping", "length_penalty", "response_format", "detokenize"):
                sp.pop(k, None)
            if guided and isinstance(guided, dict):
                try:
                    from vllm.sampling_params import StructuredOutputsParams
                    sp["structured_outputs"] = StructuredOutputsParams(**guided)
                except ImportError:
                    try:
                        from vllm.sampling_params import GuidedDecodingParams
                        sp["guided_decoding"] = GuidedDecodingParams(**guided)
                    except ImportError:
                        pass
            sampling_params = SamplingParams(**sp)
        else:
            sampling_params = SamplingParams()

        chat_kwargs = {}
        if thinking_enabled:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": True}

        # ─── Chunked main loop ──────────────────────────────────────────
        total = len(rows)
        results = [None] * total  # filled in original order
        processed = 0
        last_log_at = 0
        n_failed_preprocess = 0
        n_failed_image_load = 0

        # Streaming parquet flushing.  We track the next row index that
        # has not yet been written to a shard.  Each chunk's results are
        # always contiguous from chunk_start..chunk_end, so a flush boundary
        # is reached whenever (processed - flushed_upto) >= flush_every.
        flushed_upto = 0
        flush_part_idx = 0
        try:
            import pandas as _pd
        except Exception:
            _pd = None  # parquet shards become a no-op if pandas missing

        def _flush_shard(end_idx):
            """Write results[flushed_upto:end_idx] to a parquet shard."""
            nonlocal flushed_upto, flush_part_idx
            if not streaming_output_dir or _pd is None:
                flushed_upto = end_idx
                return
            slab = results[flushed_upto:end_idx]
            slab = [r for r in slab if r is not None]
            if not slab:
                flushed_upto = end_idx
                return
            shard_name = (
                f"rank{rank:02d}_part{flush_part_idx:04d}_"
                f"rows{flushed_upto + row_offset:08d}-"
                f"{end_idx + row_offset:08d}.parquet"
            )
            shard_path = os.path.join(streaming_output_dir, shard_name)
            try:
                df_shard = _pd.DataFrame(slab)
                df_shard.to_parquet(shard_path, index=False, compression="snappy")
                print(
                    f"[{stage_name}] DP rank {rank}/{dp_size}: "
                    f"flushed {len(slab)} rows → {shard_path}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[{stage_name}] DP rank {rank}/{dp_size}: "
                    f"streaming flush FAILED ({type(e).__name__}: {e})",
                    flush=True,
                )
            flush_part_idx += 1
            flushed_upto = end_idx

        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk_rows = rows[chunk_start:chunk_end]

            t_chunk = time.time()
            # Pre-chunk heartbeat: appears immediately so the user can see
            # the worker is alive even on the cold-start chunk (where vLLM's
            # first llm.chat() does extra one-time work).
            print(
                f"[{stage_name}] DP rank {rank}/{dp_size}: "
                f"starting chunk {chunk_start // chunk_size + 1}"
                f"/{(total + chunk_size - 1) // chunk_size} "
                f"(rows {chunk_start}–{chunk_end - 1})",
                flush=True,
            )

            # Preprocess (build messages for each row)
            chunk_pp_rows = []         # parallel to chunk_rows
            chunk_messages = []        # parallel to a subset of valid rows
            valid_indices_local = []   # indices into chunk_rows

            for ci, row in enumerate(chunk_rows):
                try:
                    pp = preprocess_fn(row)
                except Exception as e:
                    row["__preprocess_error__"] = str(e)
                    row["generated_text"] = ""
                    row["generated_reasoning"] = ""
                    chunk_pp_rows.append(row)
                    n_failed_preprocess += 1
                    continue
                chunk_pp_rows.append(pp)
                msgs = pp.get("messages")
                if not msgs:
                    pp["generated_text"] = ""
                    pp["generated_reasoning"] = ""
                    n_failed_preprocess += 1
                    continue
                _convert_image_blocks(msgs)
                chunk_messages.append(msgs)
                valid_indices_local.append(ci)

            # Parallel image decode + resize for the chunk
            t_img = time.time()
            n_img_fail = _hydrate_chunk_images(chunk_messages)
            n_failed_image_load += n_img_fail
            img_secs = time.time() - t_img

            # Drop conversations whose image loads failed (any block in
            # the conversation got tagged with __image_load_error__)
            ok_messages = []
            ok_local_indices = []
            for ci, msgs in zip(valid_indices_local, chunk_messages):
                bad = False
                for m in msgs:
                    cnt = m.get("content")
                    if not isinstance(cnt, list):
                        continue
                    for blk in cnt:
                        if isinstance(blk, dict) and "__image_load_error__" in blk:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    chunk_pp_rows[ci]["__image_load_error__"] = "image decode failed"
                    chunk_pp_rows[ci]["generated_text"] = ""
                    chunk_pp_rows[ci]["generated_reasoning"] = ""
                else:
                    ok_messages.append(msgs)
                    ok_local_indices.append(ci)

            # Generate. tqdm is disabled because vLLM's per-conversation
            # progress bar floods stderr at multi-line-per-row cadence on
            # SLURM (no carriage-return collapsing).  Our per-chunk and
            # per-flush log lines + the pre-chunk "starting chunk N" line
            # provide enough visibility.
            t_gen = time.time()
            if ok_messages:
                outputs = llm.chat(
                    ok_messages,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                    **chat_kwargs,
                )
            else:
                outputs = []
            gen_secs = time.time() - t_gen

            # Postprocess this chunk and place into the results array
            ok_local_set = set(ok_local_indices)
            out_iter = iter(outputs)
            for ci, pp_row in enumerate(chunk_pp_rows):
                if ci in ok_local_set:
                    out = next(out_iter)
                    raw_text = out.outputs[0].text if out.outputs else ""
                    reasoning, content = _extract_reasoning(raw_text)
                    pp_row["generated_text"] = content
                    pp_row["generated_reasoning"] = reasoning
                    pt = len(out.prompt_token_ids) if out.prompt_token_ids else 0
                    ct = (len(out.outputs[0].token_ids)
                          if out.outputs and out.outputs[0].token_ids else 0)
                    pp_row["usage"] = {
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "total_tokens": pt + ct,
                    }
                try:
                    results[chunk_start + ci] = postprocess_fn(pp_row)
                except Exception as e:
                    pp_row["__postprocess_error__"] = str(e)
                    results[chunk_start + ci] = pp_row

            # Free chunk-local objects so RSS does not grow
            del chunk_rows, chunk_pp_rows, chunk_messages, ok_messages, outputs
            processed = chunk_end

            # Streaming progress: log every chunk so users always see
            # forward motion. log_every is only used as a hint for the
            # default chunk_size — actual reporting cadence is per-chunk.
            chunk_secs = time.time() - t_chunk
            elapsed = time.time() - t1
            rate = processed / max(elapsed, 1e-6)
            eta = (total - processed) / max(rate, 1e-6)
            print(
                f"[{stage_name}] DP rank {rank}/{dp_size}: "
                f"{processed}/{total} ({100.0 * processed / total:.1f}%) "
                f"| chunk {chunk_secs:.1f}s "
                f"(img {img_secs:.1f}s, gen {gen_secs:.1f}s) "
                f"| {rate:.2f} rows/s | elapsed {elapsed:.0f}s "
                f"| ETA {eta:.0f}s",
                flush=True,
            )
            last_log_at = processed

            # Streaming parquet flush whenever we have at least
            # flush_every newly-completed rows since the last shard.
            if processed - flushed_upto >= flush_every:
                _flush_shard(processed)

        thread_pool.shutdown(wait=False)

        # Sanity: every slot should be filled
        for i, r in enumerate(results):
            if r is None:
                results[i] = rows[i]

        # Final flush of any rows that did not cross the flush_every boundary
        if flushed_upto < total:
            _flush_shard(total)

        print(f"[{stage_name}] DP rank {rank}/{dp_size}: done, "
              f"{len(results)} results, "
              f"preprocess_failed={n_failed_preprocess}, "
              f"image_load_failed={n_failed_image_load}, "
              f"total elapsed {time.time() - t0:.1f}s", flush=True)

        with open(result_path, "wb") as f:
            pickle.dump({"rank": rank, "results": results, "error": None}, f)
        return

    except Exception:
        tb = traceback.format_exc()
        print(f"[{stage_name}] DP rank {rank}/{dp_size}: FAILED\n{tb}",
              flush=True, file=sys.stderr)
        with open(result_path, "wb") as f:
            pickle.dump({"rank": rank, "results": None, "error": tb}, f)
        sys.exit(1)
    finally:
        try:
            if llm is not None:
                engine = getattr(llm, "llm_engine", None)
                core = getattr(engine, "engine_core", None) if engine else None
                if core is not None and hasattr(core, "shutdown"):
                    core.shutdown()
                del llm
        except Exception:
            pass

if __name__ == "__main__":
    main()
'''


def _run_data_parallel_full(
    engine_kwargs: Dict[str, Any],
    dp_size: int,
    rows: List[Dict[str, Any]],
    preprocess: Callable,
    postprocess: Callable,
    stage_name: str,
    model_source: str,
    thinking_enabled: bool,
    timeout: int = 86400,
    chunk_size: int = 64,
    log_every: int = 1000,
    image_max_pixels: Optional[int] = None,
    image_load_workers: int = 16,
    flush_every: int = 1000,
    streaming_output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Full-pipeline DP: preprocess + infer + postprocess inside each worker.

    Each worker processes its row shard in fixed-size chunks. Per chunk it
    decodes/resizes images in parallel CPU threads, runs ``llm.chat()``, and
    drops the chunk's intermediate state before moving on. This bounds
    engine-core memory and gives streaming progress every ``log_every`` rows.

    Returns a flat list of postprocessed result dicts in input order.
    """
    import pickle
    import tempfile
    import cloudpickle

    if len(rows) < dp_size:
        raise RuntimeError(
            f"[{stage_name}] Too few rows ({len(rows)}) for data_parallel_size={dp_size}."
        )

    tp_size = engine_kwargs.get("tensor_parallel_size", 1)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    all_devices = [d.strip() for d in visible.split(",") if d.strip()] if visible else []

    # Shard rows across DP ranks
    floor_n = len(rows) // dp_size
    remainder = len(rows) % dp_size
    shards = []
    shard_offsets = []  # absolute row offset of each shard within the full df
    for r in range(dp_size):
        start = r * floor_n + min(r, remainder)
        end = (r + 1) * floor_n + min(r + 1, remainder)
        shards.append(rows[start:end])
        shard_offsets.append(start)

    # Streaming parquet shard directory.  Callers (run_vllm_inference)
    # resolve this from HydraConfig.sweep.dir and pass it in fully-qualified.
    # If None, fall back to a subdirectory under cwd (Hydra's chdir=null
    # means this lands at the project root, which is wrong for multirun —
    # callers should always provide one).
    if streaming_output_dir is None:
        streaming_output_dir = os.path.join(os.getcwd(), "streaming", stage_name)
    try:
        os.makedirs(streaming_output_dir, exist_ok=True)
        print(f"[{stage_name}] Streaming parquet shards → {streaming_output_dir}",
              flush=True)
    except Exception as e:
        print(f"[{stage_name}] Could not create streaming dir "
              f"{streaming_output_dir}: {e}", flush=True)
        streaming_output_dir = None

    pcie_env = get_pcie_nccl_env_vars()
    runtime_env = get_vllm_runtime_env_vars()

    # Serialize preprocess/postprocess via cloudpickle
    preprocess_bytes = cloudpickle.dumps(preprocess)
    postprocess_bytes = cloudpickle.dumps(postprocess)

    tmpdir = os.environ.get("TMPDIR", "/tmp")
    worker_engine_kwargs = {k: v for k, v in engine_kwargs.items() if k != "data_parallel_size"}

    # Write worker script
    worker_script = tempfile.NamedTemporaryFile(
        mode="w", suffix="_dp_full_worker.py", dir=tmpdir, delete=False,
    )
    worker_script.write(_DP_FULL_WORKER_SCRIPT)
    worker_script.close()

    print(f"[{stage_name}] Launching {dp_size} full-pipeline DP workers "
          f"(TP={tp_size}, {len(rows)} total rows)...", flush=True)

    task_files = []
    result_files = []
    procs: List[subprocess.Popen] = []

    for rank in range(dp_size):
        if all_devices:
            rank_devices = all_devices[rank * tp_size:(rank + 1) * tp_size]
        else:
            rank_devices = []

        task = {
            "rank": rank,
            "dp_size": dp_size,
            "engine_kwargs": worker_engine_kwargs,
            "rows": shards[rank],
            "stage_name": stage_name,
            "pcie_env": pcie_env,
            "runtime_env": runtime_env,
            "preprocess_bytes": preprocess_bytes,
            "postprocess_bytes": postprocess_bytes,
            "model_source": model_source,
            "thinking_enabled": thinking_enabled,
            "chunk_size": chunk_size,
            "log_every": log_every,
            "image_max_pixels": image_max_pixels,
            "image_load_workers": image_load_workers,
            "flush_every": flush_every,
            "streaming_output_dir": streaming_output_dir,
            "row_offset": shard_offsets[rank],
        }

        task_path = os.path.join(tmpdir, f"{stage_name}_dpfull{rank}_task.pkl")
        result_path = os.path.join(tmpdir, f"{stage_name}_dpfull{rank}_result.pkl")
        with open(task_path, "wb") as pickle_f:
            pickle.dump(task, pickle_f)
        task_files.append(task_path)
        result_files.append(result_path)

        child_env = dict(os.environ)
        if rank_devices:
            child_env["CUDA_VISIBLE_DEVICES"] = ",".join(rank_devices)
        for var in ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE",
                     "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT"):
            child_env.pop(var, None)

        devices_str = child_env.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(f"[{stage_name}] DP rank {rank}: {len(shards[rank])} rows, "
              f"CUDA_VISIBLE_DEVICES={devices_str}", flush=True)

        proc = subprocess.Popen(
            [sys.executable, worker_script.name, task_path, result_path],
            env=child_env, stdout=sys.stdout, stderr=sys.stderr,
        )
        procs.append(proc)

    # Wait for all workers
    print(f"[{stage_name}] Waiting for {dp_size} full-pipeline DP workers "
          f"(timeout={timeout}s)...", flush=True)
    errors = []
    for rank, proc in enumerate(procs):
        try:
            retcode = proc.wait(timeout=timeout)
            if retcode != 0:
                errors.append(f"DP rank {rank} (pid={proc.pid}) exited with code {retcode}")
        except subprocess.TimeoutExpired:
            proc.kill()
            errors.append(f"DP rank {rank} (pid={proc.pid}) timed out after {timeout}s, killed")

    # Collect results
    all_results = []
    for rank in range(dp_size):
        rpath = result_files[rank]
        if not os.path.exists(rpath):
            errors.append(f"DP rank {rank}: no result file at {rpath}")
            continue
        with open(rpath, "rb") as pickle_f:
            result = pickle.load(pickle_f)
        if result.get("error"):
            errors.append(f"DP rank {rank} failed:\n{result['error']}")
        else:
            all_results.extend(result["results"])

    # Cleanup temp files
    for p in [worker_script.name, *task_files, *result_files]:
        try:
            os.unlink(p)
        except OSError:
            pass

    if errors:
        raise RuntimeError(
            f"[{stage_name}] Full-pipeline DP inference failed:\n" + "\n".join(errors)
        )

    return all_results


def _run_data_parallel(
    engine_kwargs: Dict[str, Any],
    dp_size: int,
    prompts: List[str],
    sp_dict: Dict[str, Any],
    stage_name: str,
    image_refs: Optional[List[Optional[str]]] = None,
    is_multimodal: bool = False,
    timeout: int = 86400,
) -> List[Dict[str, Any]]:
    """Spawn dp_size subprocess workers for vLLM data-parallel inference.

    Follows vLLM 0.19's DP pattern: each worker is a fresh Python process
    with ``VLLM_DP_*`` env vars set.  vLLM handles GPU assignment and NCCL
    coordination internally.

    Multimodal support: ``image_refs`` carries file paths (strings) alongside
    ``prompts``.  Workers load images lazily via PIL, avoiding serialization
    of large PIL objects across process boundaries.

    Returns a list of output dicts in the same order as ``prompts``.
    """
    import pickle
    import tempfile

    if len(prompts) < dp_size:
        raise RuntimeError(
            f"[{stage_name}] Too few prompts ({len(prompts)}) for "
            f"data_parallel_size={dp_size}."
        )

    tp_size = engine_kwargs.get("tensor_parallel_size", 1)
    total_gpus_needed = dp_size * tp_size
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    all_devices = [d.strip() for d in visible.split(",") if d.strip()] if visible else []
    if all_devices and len(all_devices) < total_gpus_needed:
        raise RuntimeError(
            f"[{stage_name}] data_parallel_size={dp_size} x "
            f"tensor_parallel_size={tp_size} = {total_gpus_needed} GPUs required, "
            f"but only {len(all_devices)} visible: {all_devices}"
        )

    # Split prompts (and image refs) across DP ranks
    floor_n = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def shard_range(rank: int) -> tuple:
        start = rank * floor_n + min(rank, remainder)
        end = (rank + 1) * floor_n + min(rank + 1, remainder)
        return start, end

    prompt_shards = []
    image_ref_shards = []
    for r in range(dp_size):
        s, e = shard_range(r)
        prompt_shards.append(prompts[s:e])
        image_ref_shards.append(image_refs[s:e] if image_refs else [])

    pcie_env = get_pcie_nccl_env_vars()
    runtime_env = get_vllm_runtime_env_vars()

    tmpdir = os.environ.get("TMPDIR", "/tmp")
    task_files = []
    result_files = []
    procs: List[subprocess.Popen] = []

    # Write worker script to a temp file
    worker_script = tempfile.NamedTemporaryFile(
        mode="w", suffix="_dp_worker.py", dir=tmpdir, delete=False,
    )
    worker_script.write(_DP_WORKER_SCRIPT)
    worker_script.close()

    # Workers are fully independent — no data_parallel_size in kwargs
    worker_engine_kwargs = {k: v for k, v in engine_kwargs.items()
                           if k != "data_parallel_size"}

    print(f"[{stage_name}] Launching {dp_size} DP workers "
          f"(TP={tp_size}, {len(prompts)} total prompts, "
          f"multimodal={is_multimodal})...", flush=True)

    for rank in range(dp_size):
        # GPU slice for this rank
        if all_devices:
            rank_devices = all_devices[rank * tp_size:(rank + 1) * tp_size]
        else:
            rank_devices = []

        task = {
            "rank": rank,
            "dp_size": dp_size,
            "engine_kwargs": worker_engine_kwargs,
            "prompts": prompt_shards[rank],
            "image_refs": image_ref_shards[rank],
            "sp_dict": sp_dict,
            "stage_name": stage_name,
            "pcie_env": pcie_env,
            "runtime_env": runtime_env,
            "is_multimodal": is_multimodal,
        }

        task_path = os.path.join(tmpdir, f"{stage_name}_dp{rank}_task.pkl")
        result_path = os.path.join(tmpdir, f"{stage_name}_dp{rank}_result.pkl")
        with open(task_path, "wb") as pickle_f:
            pickle.dump(task, pickle_f)
        task_files.append(task_path)
        result_files.append(result_path)

        # Build clean env: assign GPU slice, clear stale DP vars
        child_env = dict(os.environ)
        if rank_devices:
            child_env["CUDA_VISIBLE_DEVICES"] = ",".join(rank_devices)
        for var in ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE",
                     "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT"):
            child_env.pop(var, None)

        devices_str = child_env.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(f"[{stage_name}] DP rank {rank}: {len(prompt_shards[rank])} prompts, "
              f"CUDA_VISIBLE_DEVICES={devices_str}", flush=True)

        proc = subprocess.Popen(
            [sys.executable, worker_script.name, task_path, result_path],
            env=child_env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        procs.append(proc)

    # Wait for all workers
    print(f"[{stage_name}] Waiting for {dp_size} DP workers "
          f"(timeout={timeout}s)...", flush=True)
    errors = []
    for rank, proc in enumerate(procs):
        try:
            retcode = proc.wait(timeout=timeout)
            if retcode != 0:
                errors.append(
                    f"DP rank {rank} (pid={proc.pid}) exited with code {retcode}"
                )
        except subprocess.TimeoutExpired:
            proc.kill()
            errors.append(
                f"DP rank {rank} (pid={proc.pid}) timed out after {timeout}s, killed"
            )

    # Collect results
    rank_results: Dict[int, List[Dict[str, Any]]] = {}
    for rank in range(dp_size):
        rpath = result_files[rank]
        if not os.path.exists(rpath):
            errors.append(f"DP rank {rank}: no result file at {rpath}")
            continue
        with open(rpath, "rb") as pickle_f:
            result = pickle.load(pickle_f)
        if result.get("error"):
            errors.append(f"DP rank {rank} failed:\n{result['error']}")
        else:
            rank_results[rank] = result["outputs"]

    # Cleanup temp files
    for p in [worker_script.name, *task_files, *result_files]:
        try:
            os.unlink(p)
        except OSError:
            pass

    if errors:
        raise RuntimeError(
            f"[{stage_name}] Data-parallel inference failed:\n"
            + "\n".join(errors)
        )

    # Reassemble in original order
    all_outputs = []
    for rank in range(dp_size):
        results = rank_results[rank]
        if len(results) != len(prompt_shards[rank]):
            raise RuntimeError(
                f"[{stage_name}] DP rank {rank} output count mismatch: "
                f"expected {len(prompt_shards[rank])}, got {len(results)}"
            )
        all_outputs.extend(results)

    return all_outputs


# ---------------------------------------------------------------------------
# Data-parallel embedding inference
# ---------------------------------------------------------------------------

_DP_EMBED_WORKER_SCRIPT = r'''
"""Standalone DP embedding worker — fresh subprocess per rank.

Each worker is a completely independent LLM instance with its own
CUDA_VISIBLE_DEVICES slice.  Uses LLM(runner="pooling") and llm.embed().

Streams incremental chunk pickles into ``{result_path}.chunks/`` every
CHUNK_BATCHES batches, using atomic temp+fsync+rename, so partial data
survives a timeout or SIGKILL. Writes ``{result_path}`` as a completion
marker (or error report) once all inputs are processed.
"""
import os, pickle, sys, time, traceback
import numpy as np

# Flush a chunk every CHUNK_BATCHES batches.  At batch_size=16 this is
# ~800 embeddings per chunk; losing one chunk on timeout costs a few
# minutes, not hours.
CHUNK_BATCHES = 50


def _atomic_write_pickle(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _flush_chunk(chunk_dir, rank, chunk_idx, start_offset, embeddings):
    path = os.path.join(chunk_dir, f"chunk{chunk_idx:05d}.pkl")
    _atomic_write_pickle(path, {
        "rank": rank,
        "chunk_idx": chunk_idx,
        "start": start_offset,
        "count": len(embeddings),
        "embeddings": embeddings,
    })


def main():
    task_path = sys.argv[1]
    result_path = sys.argv[2]
    chunk_dir = result_path + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    with open(task_path, "rb") as f:
        task = pickle.load(f)

    rank           = task["rank"]
    dp_size        = task["dp_size"]
    engine_kwargs  = task["engine_kwargs"]
    prompt_texts   = task["prompt_texts"]
    image_refs     = task["image_refs"]
    stage_name     = task["stage_name"]
    pcie_env       = task["pcie_env"]
    runtime_env    = task["runtime_env"]
    batch_size     = task["batch_size"]

    for k, v in {**pcie_env, **runtime_env}.items():
        os.environ.setdefault(k, v)

    # Clear any inherited vLLM DP coordination vars
    for var in ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE",
                "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT"):
        os.environ.pop(var, None)

    print(f"[{stage_name}] DP rank {rank}/{dp_size}: starting embed worker "
          f"(pid={os.getpid()}, CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
          f"prompts={len(prompt_texts)}, chunk_dir={chunk_dir})", flush=True)

    llm = None
    chunk_idx = 0
    flushed_count = 0
    try:
        t0 = time.time()
        from vllm import LLM
        from PIL import Image

        t1 = time.time()
        llm = LLM(**engine_kwargs)
        print(f"[{stage_name}] DP rank {rank}/{dp_size}: LLM created in "
              f"{time.time() - t1:.1f}s", flush=True)

        total = len(prompt_texts)
        pending = []
        batches_since_flush = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_inputs = []
            for j in range(start, end):
                inp = {"prompt": prompt_texts[j]}
                ref = image_refs[j] if image_refs else None
                if ref is not None:
                    inp["multi_modal_data"] = {
                        "image": Image.open(ref).convert("RGB")
                    }
                batch_inputs.append(inp)

            outputs = llm.embed(batch_inputs)
            for out in outputs:
                emb = out.outputs.embedding
                if not isinstance(emb, np.ndarray):
                    emb = np.array(emb, dtype=np.float32)
                pending.append(emb)
            del batch_inputs, outputs
            batches_since_flush += 1

            if batches_since_flush >= CHUNK_BATCHES:
                _flush_chunk(chunk_dir, rank, chunk_idx, flushed_count, pending)
                flushed_count += len(pending)
                chunk_idx += 1
                pending = []
                batches_since_flush = 0
                print(f"[{stage_name}] DP rank {rank}/{dp_size}: "
                      f"{flushed_count}/{total} embedded "
                      f"(chunk {chunk_idx} flushed)", flush=True)
            elif (start // batch_size) % 20 == 0 or end == total:
                print(f"[{stage_name}] DP rank {rank}/{dp_size}: "
                      f"{flushed_count + len(pending)}/{total} embedded",
                      flush=True)

        # Final chunk flush for remaining pending
        if pending:
            _flush_chunk(chunk_dir, rank, chunk_idx, flushed_count, pending)
            flushed_count += len(pending)
            chunk_idx += 1
            pending = []

        # Completion marker (chunks are authoritative for data)
        _atomic_write_pickle(result_path, {
            "rank": rank,
            "num_chunks": chunk_idx,
            "total": flushed_count,
            "error": None,
        })

        print(f"[{stage_name}] DP rank {rank}/{dp_size}: done, "
              f"{flushed_count} embeddings in {chunk_idx} chunks, "
              f"elapsed {time.time() - t0:.1f}s", flush=True)

    except Exception:
        tb = traceback.format_exc()
        print(f"[{stage_name}] DP rank {rank}/{dp_size}: FAILED\n{tb}",
              flush=True, file=sys.stderr)
        try:
            _atomic_write_pickle(result_path, {
                "rank": rank,
                "num_chunks": chunk_idx,
                "total": flushed_count,
                "error": tb,
            })
        except Exception:
            pass
        sys.exit(1)
    finally:
        try:
            if llm is not None:
                engine = getattr(llm, "llm_engine", None)
                core = getattr(engine, "engine_core", None) if engine else None
                if core is not None and hasattr(core, "shutdown"):
                    core.shutdown()
                del llm
        except Exception:
            pass

if __name__ == "__main__":
    main()
'''


def _run_data_parallel_embed(
    engine_kwargs: Dict[str, Any],
    dp_size: int,
    prompt_texts: List[str],
    image_refs: List[Optional[str]],
    stage_name: str,
    batch_size: int = 16,
    timeout: int = 255600,
) -> Tuple[List[Any], List[str]]:
    """Spawn dp_size subprocess workers for vLLM embedding inference.

    Same pattern as ``_run_data_parallel`` but uses ``LLM(runner="pooling")``
    and ``llm.embed()``.

    Workers stream incremental chunk pickles to
    ``{tmpdir}/{stage_name}_dp{rank}_result.pkl.chunks/``, so even if a rank
    is killed by the watchdog (or SLURM) the already-embedded rows survive.

    The default 255600s (~71h) watchdog leaves ~1h of SLURM headroom for
    cleanup and parquet flushing before the job's hard limit (slurm_gpu_4x
    is 72h).

    Returns:
        A tuple ``(all_embeddings, errors)`` where ``all_embeddings`` has
        length ``len(prompt_texts)`` with ``None`` placeholders for any
        positions a worker never produced. ``errors`` lists human-readable
        failure reasons — empty on full success.  When non-empty, chunk
        directories for the failing ranks are left on disk for manual
        recovery (their paths are printed in stdout).
    """
    import glob
    import pickle
    import tempfile

    if len(prompt_texts) < dp_size:
        raise RuntimeError(
            f"[{stage_name}] Too few inputs ({len(prompt_texts)}) for "
            f"data_parallel_size={dp_size}."
        )

    tp_size = engine_kwargs.get("tensor_parallel_size", 1)
    total_gpus_needed = dp_size * tp_size
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    all_devices = [d.strip() for d in visible.split(",") if d.strip()] if visible else []
    if all_devices and len(all_devices) < total_gpus_needed:
        raise RuntimeError(
            f"[{stage_name}] data_parallel_size={dp_size} x "
            f"tensor_parallel_size={tp_size} = {total_gpus_needed} GPUs required, "
            f"but only {len(all_devices)} visible: {all_devices}"
        )

    # Shard inputs across DP ranks
    floor_n = len(prompt_texts) // dp_size
    remainder = len(prompt_texts) % dp_size

    def shard_range(rank: int) -> tuple:
        start = rank * floor_n + min(rank, remainder)
        end = (rank + 1) * floor_n + min(rank + 1, remainder)
        return start, end

    prompt_shards = []
    image_ref_shards = []
    for r in range(dp_size):
        s, e = shard_range(r)
        prompt_shards.append(prompt_texts[s:e])
        image_ref_shards.append(image_refs[s:e] if image_refs else [])

    pcie_env = get_pcie_nccl_env_vars()
    runtime_env = get_vllm_runtime_env_vars()

    tmpdir = os.environ.get("TMPDIR", "/tmp")
    task_files = []
    result_files = []
    procs: List[subprocess.Popen] = []

    worker_script = tempfile.NamedTemporaryFile(
        mode="w", suffix="_dp_embed_worker.py", dir=tmpdir, delete=False,
    )
    worker_script.write(_DP_EMBED_WORKER_SCRIPT)
    worker_script.close()

    # Workers are fully independent — no data_parallel_size in kwargs
    worker_engine_kwargs = {k: v for k, v in engine_kwargs.items()
                           if k != "data_parallel_size"}

    print(f"[{stage_name}] Launching {dp_size} DP embed workers "
          f"(TP={tp_size}, {len(prompt_texts)} total inputs)...", flush=True)

    for rank in range(dp_size):
        # GPU slice for this rank
        if all_devices:
            rank_devices = all_devices[rank * tp_size:(rank + 1) * tp_size]
        else:
            rank_devices = []

        task = {
            "rank": rank,
            "dp_size": dp_size,
            "engine_kwargs": worker_engine_kwargs,
            "prompt_texts": prompt_shards[rank],
            "image_refs": image_ref_shards[rank],
            "stage_name": stage_name,
            "pcie_env": pcie_env,
            "runtime_env": runtime_env,
            "batch_size": batch_size,
        }

        task_path = os.path.join(tmpdir, f"{stage_name}_dp{rank}_task.pkl")
        result_path = os.path.join(tmpdir, f"{stage_name}_dp{rank}_result.pkl")
        with open(task_path, "wb") as pickle_f:
            pickle.dump(task, pickle_f)
        task_files.append(task_path)
        result_files.append(result_path)

        # Build clean env: assign GPU slice, clear stale DP vars
        child_env = dict(os.environ)
        if rank_devices:
            child_env["CUDA_VISIBLE_DEVICES"] = ",".join(rank_devices)
        for var in ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE",
                     "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT"):
            child_env.pop(var, None)

        devices_str = child_env.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(f"[{stage_name}] DP rank {rank}: {len(prompt_shards[rank])} inputs, "
              f"CUDA_VISIBLE_DEVICES={devices_str}", flush=True)

        proc = subprocess.Popen(
            [sys.executable, worker_script.name, task_path, result_path],
            env=child_env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        procs.append(proc)

    # Wait for all workers — per-rank errors collected so we can still
    # recover chunks from ranks that succeeded or partially completed.
    print(f"[{stage_name}] Waiting for {dp_size} DP embed workers "
          f"(timeout={timeout}s)...", flush=True)
    rank_errors: Dict[int, List[str]] = {rank: [] for rank in range(dp_size)}
    for rank, proc in enumerate(procs):
        try:
            retcode = proc.wait(timeout=timeout)
            if retcode != 0:
                rank_errors[rank].append(
                    f"process (pid={proc.pid}) exited with code {retcode}"
                )
        except subprocess.TimeoutExpired:
            proc.kill()
            rank_errors[rank].append(
                f"process (pid={proc.pid}) timed out after {timeout}s, killed"
            )

    # Collect results per rank from chunk dirs + marker files
    rank_results: Dict[int, List[Any]] = {}
    chunk_dirs: Dict[int, str] = {}
    for rank in range(dp_size):
        rpath = result_files[rank]
        chunk_dir = rpath + ".chunks"
        chunk_dirs[rank] = chunk_dir

        # Load whatever chunks exist (ordered by filename)
        rank_embeddings: List[Any] = []
        if os.path.isdir(chunk_dir):
            chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "chunk*.pkl")))
            for cpath in chunk_paths:
                try:
                    with open(cpath, "rb") as cf:
                        cdata = pickle.load(cf)
                    rank_embeddings.extend(cdata.get("embeddings") or [])
                except Exception as e:
                    rank_errors[rank].append(
                        f"failed to read chunk {os.path.basename(cpath)}: {e}"
                    )

        # Inspect completion marker (may or may not exist)
        if os.path.exists(rpath):
            try:
                with open(rpath, "rb") as mf:
                    marker = pickle.load(mf)
            except Exception as e:
                rank_errors[rank].append(f"corrupt marker {rpath}: {e}")
                marker = None
            if marker and marker.get("error"):
                rank_errors[rank].append(
                    f"worker raised:\n{marker['error']}"
                )

        expected = len(prompt_shards[rank])
        if len(rank_embeddings) < expected:
            rank_errors[rank].append(
                f"incomplete: {len(rank_embeddings)}/{expected} embeddings "
                f"(partial chunks preserved at {chunk_dir})"
            )

        rank_results[rank] = rank_embeddings

    # Flatten per-rank errors for caller
    errors: List[str] = []
    for rank in range(dp_size):
        for msg in rank_errors[rank]:
            errors.append(f"DP rank {rank}: {msg}")

    # Cleanup: always remove worker script + task pickles. For each rank,
    # only remove chunk files + marker if the rank had no errors; otherwise
    # leave the chunk directory intact for manual recovery.
    cleanup_paths: List[str] = [worker_script.name, *task_files]
    recoverable_chunk_dirs: List[str] = []
    for rank in range(dp_size):
        rpath = result_files[rank]
        chunk_dir = chunk_dirs[rank]
        if rank_errors[rank]:
            if os.path.isdir(chunk_dir):
                recoverable_chunk_dirs.append(chunk_dir)
            continue
        if os.path.isdir(chunk_dir):
            # Sweep every file (chunks AND any leftover atomic-write .tmp)
            # so rmdir at the end succeeds.
            for name in os.listdir(chunk_dir):
                cleanup_paths.append(os.path.join(chunk_dir, name))
            cleanup_paths.append(chunk_dir)  # rmdir last
        if os.path.exists(rpath):
            cleanup_paths.append(rpath)

    for p in cleanup_paths:
        try:
            if os.path.isdir(p):
                os.rmdir(p)
            else:
                os.unlink(p)
        except OSError:
            pass

    if errors:
        print(f"[{stage_name}] DP embed completed with {len(errors)} error(s):",
              flush=True)
        for err in errors:
            print(f"  - {err}", flush=True)
        if recoverable_chunk_dirs:
            print(f"[{stage_name}] Recoverable chunk dirs (NOT deleted):",
                  flush=True)
            for p in recoverable_chunk_dirs:
                print(f"  - {p}", flush=True)

    # Reassemble in original order; insert None placeholders for any gaps
    # so the caller can still stream partial results to disk.
    all_embeddings: List[Any] = []
    for rank in range(dp_size):
        got = rank_results[rank]
        expected = len(prompt_shards[rank])
        all_embeddings.extend(got)
        if len(got) < expected:
            all_embeddings.extend([None] * (expected - len(got)))

    return all_embeddings, errors


# ---------------------------------------------------------------------------
# Transformers fallback for models vLLM no longer supports (e.g. Mllama)
# ---------------------------------------------------------------------------

# Model families that bypass vLLM entirely and use transformers generate().
_TRANSFORMERS_FALLBACK_FAMILIES = {"llama-vision"}


def _run_transformers_text_inference(
    df: pd.DataFrame,
    cfg,
    preprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    postprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    stage_name: str = "transformers_inference",
) -> pd.DataFrame:
    """Text-only inference via native transformers for unsupported vLLM models.

    Mirrors the run_vllm_inference interface (preprocess/postprocess callables)
    but uses AutoModelForCausalLM.generate() instead of vLLM.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_source = str(cfg.model.model_source)
    print(f"[{stage_name}] Using native transformers fallback for {model_source}")

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation

    # Use flash_attention_2 if available for ~2x speedup on long sequences
    try:
        import flash_attn  # noqa: F401
        _attn_impl = "flash_attention_2"
    except ImportError:
        _attn_impl = "sdpa"
    print(f"[{stage_name}] Attention: {_attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,  # Mllama has intentional embed/lm_head size diffs
        attn_implementation=_attn_impl,
    )

    # Load and merge LoRA adapter if specified
    _lora_path = str(getattr(cfg.model, "lora_path", "") or "")
    if _lora_path:
        from peft import PeftModel
        print(f"[{stage_name}] Loading LoRA adapter: {_lora_path}")
        model = PeftModel.from_pretrained(model, _lora_path)
        model = model.merge_and_unload()
        print(f"[{stage_name}] LoRA merged into base model")

    model.eval()

    # Determine thinking mode (single source of truth).
    from dagspaces.common.stage_utils import resolve_thinking_mode
    _thinking_enabled = resolve_thinking_mode(cfg.model, default=True)
    _strip_thinking = not _thinking_enabled

    # Preprocess rows
    print(f"[{stage_name}] Preprocessing {len(df)} rows...")
    preprocessed_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(df.to_dict("records")):
        try:
            preprocessed_rows.append(preprocess(row))
        except Exception as e:
            row["__preprocess_error__"] = str(e)
            preprocessed_rows.append(row)

    # Separate valid rows from failed ones
    valid_indices = []
    failed_rows = []
    prompts = []
    sp_first = {}
    for i, row in enumerate(preprocessed_rows):
        if "__preprocess_error__" in row:
            row["generated_text"] = ""
            row["generated_reasoning"] = ""
            failed_rows.append((i, postprocess(row)))
        else:
            messages = row.get("messages", [])
            sp = row.get("sampling_params", {})
            if not sp_first:
                sp_first = sp  # sampling params are same for all rows
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            valid_indices.append(i)

    max_tokens = int(sp_first.get("max_tokens", 1024))
    temperature = float(sp_first.get("temperature", 0.0))
    do_sample = temperature > 0

    # Batch inference — Mllama 11B in bf16 ≈ 22GB, leaving ~26GB on A6000 for
    # KV cache.  batch_size=16 with max_tokens=1024 fits comfortably.
    batch_size = 16
    generated_texts: List[str] = []
    generated_reasonings: List[str] = []
    print(f"[{stage_name}] Generating {len(prompts)} prompts in batches of {batch_size}...")

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=8192,
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        for j in range(len(batch_prompts)):
            gen_ids = output_ids[j, prompt_len:]
            raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            reasoning, content = _split_reasoning(
                raw_text, model_source, _thinking_enabled, tokenizer,
            )
            generated_texts.append(content)
            generated_reasonings.append(reasoning)

        print(f"[{stage_name}] Batch {start // batch_size + 1}: "
              f"{min(start + batch_size, len(prompts))}/{len(prompts)} done")

    # Reassemble results in original order
    results = [None] * len(preprocessed_rows)
    for i, row_data in failed_rows:
        results[i] = row_data
    for vi, gen_text, gen_reasoning in zip(valid_indices, generated_texts, generated_reasonings):
        row = preprocessed_rows[vi]
        row["generated_text"] = gen_text
        row["generated_reasoning"] = gen_reasoning
        results[vi] = postprocess(row)

    print(f"[{stage_name}] Completed transformers inference, {len(results)} results")
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_vllm_inference(
    df: pd.DataFrame,
    cfg,
    preprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    postprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    stage_name: str = "vllm_inference",
) -> pd.DataFrame:
    """Run vLLM batch inference on a DataFrame.

    1. Calls preprocess(row_dict) for each row. The preprocessor must set
       ``row["messages"]`` (list of chat dicts) and ``row["sampling_params"]``
       (plain dict).
    2. Builds prompts via ``tokenizer.apply_chat_template``.
    3. Calls ``llm.generate(prompts, sampling_params)`` in configurable batches.
    4. Sets ``row["generated_text"]`` and usage info, then calls postprocess.
    5. Returns a DataFrame of all postprocessed rows.

    Args:
        df: Input DataFrame (or anything with .to_pandas()).
        cfg: Hydra config with model.model_source, model.engine_kwargs, etc.
        preprocess: Row dict -> row dict with "messages" and "sampling_params".
        postprocess: Row dict (with "generated_text") -> final row dict.
        stage_name: Label for log messages.

    Returns:
        pd.DataFrame of postprocessed results.
    """
    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    if df is None or len(df) == 0:
        print(f"[{stage_name}] Empty input, returning empty DataFrame")
        return pd.DataFrame()

    # Server mode: route to a long-lived vLLM OpenAI-compatible server.
    # This lets eval_all share one loaded model across all benchmark jobs.
    _server_url = _resolve_server_url(cfg)
    if _server_url:
        return _run_server_inference(
            df=df, cfg=cfg, preprocess=preprocess, postprocess=postprocess,
            stage_name=stage_name, server_url=_server_url,
        )

    # Models whose architectures vLLM no longer supports get routed to a
    # native transformers generate() fallback.
    _model_family = str(getattr(cfg.model, "model_family", ""))
    if _model_family in _TRANSFORMERS_FALLBACK_FAMILIES:
        return _run_transformers_text_inference(
            df=df, cfg=cfg, preprocess=preprocess, postprocess=postprocess,
            stage_name=stage_name,
        )

    # Set runtime env vars before importing vLLM.
    for k, v in {**get_pcie_nccl_env_vars(), **get_vllm_runtime_env_vars()}.items():
        os.environ.setdefault(k, v)

    env_snapshot = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS", "<unset>"),
        "SLURM_GPUS_ON_NODE": os.environ.get("SLURM_GPUS_ON_NODE", "<unset>"),
        "VLLM_WORKER_MULTIPROC_METHOD": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "<unset>"),
    }
    print(f"[{stage_name}] Runtime env: {env_snapshot}")

    # Import vLLM after env vars are set
    from vllm import LLM, SamplingParams

    # Build engine kwargs
    engine_kwargs = _build_engine_kwargs(cfg)

    # Check for LoRA adapter path in model config
    lora_path = None
    lora_request = None
    try:
        lora_path = str(OmegaConf.select(cfg, "model.lora_path") or "")
    except Exception as e:
        print(f"[{stage_name}] LoRA path lookup via OmegaConf.select failed: {e}")
    # Fallback: direct attribute access (handles plain dicts and DictConfig)
    if not lora_path:
        try:
            lora_path = str(getattr(cfg.model, "lora_path", "") or "")
        except Exception as e:
            print(f"[{stage_name}] LoRA path fallback failed: {e}")
    print(f"[{stage_name}] LoRA path resolved: {repr(lora_path)}")
    if lora_path:
        # Remap adapter keys if needed (CausalLM → VLM prefix mismatch)
        model_source = engine_kwargs.get("model", "")
        lora_path = _remap_lora_keys_for_vlm(lora_path, model_source, stage_name)
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("sft", 1, lora_path)
        print(f"[{stage_name}] LoRA adapter: {lora_path}")

    # Determine thinking mode for reasoning extraction (single source of truth:
    # model.thinking_mode, falling back to chat_template_kwargs.enable_thinking).
    # The chat template flag controls *what prompt* the model sees; reasoning
    # is always extracted from output via `_split_reasoning` regardless.
    from dagspaces.common.stage_utils import resolve_thinking_mode
    _thinking_enabled = resolve_thinking_mode(cfg.model, default=True)
    _strip_thinking = not _thinking_enabled  # retained for legacy log lines
    _model_source = str(engine_kwargs.get("model", "") or "")
    _parser_name = _detect_reasoning_parser(_model_source)
    print(f"[{stage_name}] Reasoning extraction: parser={_parser_name or 'regex-fallback'}, "
          f"thinking_enabled={_thinking_enabled}")

    # Check for data parallelism (pop before passing to LLM — vLLM 0.19
    # workers get DP config from VLLM_DP_* env vars, not LLM kwargs).
    dp_size = int(engine_kwargs.pop("data_parallel_size", 1) or 1)

    print(
        f"[{stage_name}] Initializing vLLM with: "
        f"{ {k: v for k, v in engine_kwargs.items() if k != 'model'} }"
    )
    print(f"[{stage_name}] Model: {engine_kwargs.get('model')}")
    if dp_size > 1:
        print(f"[{stage_name}] Data parallelism enabled: {dp_size} replicas "
              f"x TP={engine_kwargs.get('tensor_parallel_size', 1)}")

    # For DP mode, use full-pipeline workers: preprocess + infer + postprocess
    # all happen inside each worker subprocess.  This avoids the bottleneck
    # of sequentially preprocessing all rows in the main process (especially
    # expensive for pairwise image stitching or OCR tiling).
    if dp_size > 1:
        # Chunked execution + parallel image hydration knobs.  These default
        # to values that work for multimodal jobs at ~1M-pixel images on
        # A6000-class GPUs; tune via cfg.model.* if needed.  Smaller
        # chunk_size = more frequent progress, slightly more Python overhead;
        # 64 is the sweet spot for cold-start visibility.
        chunk_size = int(getattr(cfg.model, "chunk_size", 0) or 64)
        log_every = int(getattr(cfg.model, "log_every", 0) or 1000)
        flush_every = int(getattr(cfg.model, "flush_every", 0) or 1000)
        image_load_workers = int(getattr(cfg.model, "image_load_workers", 0) or 16)

        # Where to drop streaming parquet shards.  Preference order:
        #   1. cfg.model.streaming_output_dir (explicit override)
        #   2. HydraConfig.sweep.dir (the multirun sweep root, e.g.
        #      multirun/2026-04-10_URBANPAIRVQA/11-28-27)
        #   3. HydraConfig.runtime.output_dir (per-job dir)
        #   4. cfg.runtime.output_dir (set by the orchestrator)
        #   5. os.getcwd() fallback (pre-Hydra behaviour)
        # Hydra's chdir=null means os.getcwd() is the project root, which is
        # the wrong place — we have to ask HydraConfig explicitly.
        _streaming_root = None
        _override = getattr(cfg.model, "streaming_output_dir", None)
        if _override:
            _streaming_root = str(_override)
        if _streaming_root is None:
            try:
                from hydra.core.hydra_config import HydraConfig as _HC
                _hc = _HC.get()
                _sweep = getattr(getattr(_hc, "sweep", None), "dir", None)
                if _sweep:
                    _streaming_root = os.path.abspath(str(_sweep))
                if _streaming_root is None:
                    _rt = getattr(getattr(_hc, "runtime", None), "output_dir", None)
                    if _rt:
                        _streaming_root = os.path.abspath(str(_rt))
            except Exception:
                pass
        if _streaming_root is None:
            _rt_out = None
            try:
                _rt_out = getattr(
                    getattr(cfg, "runtime", object()), "output_dir", None
                )
            except Exception:
                pass
            if _rt_out:
                _streaming_root = os.path.abspath(str(_rt_out))
        if _streaming_root is None:
            _streaming_root = os.getcwd()
        streaming_output_dir = os.path.join(
            _streaming_root, "streaming", stage_name
        )
        # Use the same max_pixels vLLM's HF processor would have applied.
        try:
            _mm_pk = OmegaConf.to_container(
                cfg.model.engine_kwargs.mm_processor_kwargs, resolve=True
            ) or {}
        except Exception:
            _mm_pk = {}
        image_max_pixels = _mm_pk.get("max_pixels") if isinstance(_mm_pk, dict) else None
        try:
            image_max_pixels = int(image_max_pixels) if image_max_pixels else None
        except Exception:
            image_max_pixels = None

        print(f"[{stage_name}] Using full-pipeline DP workers "
              f"(chunk_size={chunk_size}, log_every={log_every}, "
              f"flush_every={flush_every}, "
              f"image_max_pixels={image_max_pixels}, "
              f"image_load_workers={image_load_workers}, "
              f"streaming_output_dir={streaming_output_dir})")
        raw_rows = df.to_dict("records")
        dp_results = _run_data_parallel_full(
            engine_kwargs=engine_kwargs,
            dp_size=dp_size,
            rows=raw_rows,
            preprocess=preprocess,
            postprocess=postprocess,
            stage_name=stage_name,
            model_source=_model_source,
            thinking_enabled=_thinking_enabled,
            chunk_size=chunk_size,
            log_every=log_every,
            image_max_pixels=image_max_pixels,
            image_load_workers=image_load_workers,
            flush_every=flush_every,
            streaming_output_dir=streaming_output_dir,
        )
        print(f"[{stage_name}] Completed inference, {len(dp_results)} results")
        return pd.DataFrame(dp_results)

    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()

    try:
        # Preprocess all rows
        print(f"[{stage_name}] Preprocessing {len(df)} rows...")
        preprocessed_rows: List[Dict[str, Any]] = []
        failed_indices: List[int] = []  # indices of preprocess-failed rows
        for idx, row in enumerate(df.to_dict("records")):
            try:
                preprocessed_rows.append(preprocess(row))
            except Exception as e:
                row["__preprocess_error__"] = str(e)
                print(f"[{stage_name}] Preprocess error on row {idx}: {e}")
                preprocessed_rows.append(row)
                failed_indices.append(idx)

        if failed_indices:
            print(f"[{stage_name}] WARNING: {len(failed_indices)} rows failed "
                  f"preprocessing and will be skipped for inference")

        # Separate valid rows from failed ones for inference
        failed_set = set(failed_indices)
        preliminary_valid = [i for i in range(len(preprocessed_rows)) if i not in failed_set]

        # Determine model's max context length for prompt validation.
        _max_model_len = None
        if llm is not None:
            try:
                _max_model_len = llm.llm_engine.model_config.max_model_len
            except Exception:
                pass
        if _max_model_len is None:
            try:
                _max_model_len = int(getattr(tokenizer, "model_max_length", 0) or 0)
                if _max_model_len <= 0 or _max_model_len > 1_000_000:
                    _max_model_len = None
            except Exception:
                pass

        # Detect if this is a multimodal model
        _is_mm = _is_multimodal_model(str(engine_kwargs.get("model", "")), cfg)
        if _is_mm:
            print(f"[{stage_name}] Multimodal model detected — images will be "
                  f"passed via multi_modal_data")

        # Build prompts and sampling params dicts for valid rows only.
        prompts: List[str] = []
        row_images: List[Optional[List[Any]]] = []  # per-row PIL image lists for single-process
        row_image_refs: List[Optional[str]] = []  # per-row file path refs for DP serialization
        sp_dicts: List[Dict[str, Any]] = []
        valid_indices: List[int] = []
        _oversized_count = 0

        for i in preliminary_valid:
            row = preprocessed_rows[i]
            messages = row.get("messages")
            images = None
            if messages:
                # Extract images from multimodal content blocks
                if _is_mm:
                    images = _extract_images_from_messages(messages)
                    if images:
                        # For multimodal models, pass messages with image
                        # placeholders — the tokenizer handles the rest
                        template_messages = messages
                    else:
                        template_messages = messages
                else:
                    template_messages = messages

                try:
                    chat_template_kwargs = dict(
                        getattr(cfg.model, "chat_template_kwargs", {}) or {}
                    )
                    prompt = tokenizer.apply_chat_template(
                        template_messages, tokenize=False, add_generation_prompt=True,
                        **chat_template_kwargs,
                    )
                except Exception:
                    # Fallback for non-multimodal: flatten and retry
                    try:
                        flat_messages = _flatten_messages_for_template(messages)
                        prompt = tokenizer.apply_chat_template(
                            flat_messages, tokenize=False, add_generation_prompt=True,
                        )
                    except Exception:
                        parts = []
                        for msg in messages:
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                content = " ".join(
                                    b.get("text", "") for b in content
                                    if isinstance(b, dict) and b.get("type") == "text"
                                )
                            parts.append(f"{role}: {content}")
                        prompt = "\n\n".join(parts) + "\n\nAssistant:"
            else:
                prompt = str(row.get("article_text", ""))

            # Validate prompt length against model context window
            if _max_model_len is not None:
                sp = row.get("sampling_params", {})
                max_new = int(sp.get("max_tokens", 0) or 0)
                prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                if prompt_tokens + max(max_new, 1) > _max_model_len:
                    row["__preprocess_error__"] = (
                        f"Prompt too long: {prompt_tokens} tokens + "
                        f"{max_new} max_tokens > {_max_model_len} context limit"
                    )
                    failed_set.add(i)
                    _oversized_count += 1
                    continue

            prompts.append(prompt)
            row_images.append(images if images else None)
            # Collect serializable image path for DP mode (PIL objects can't cross process boundaries)
            _img_ref = row.get("image_path") or preprocessed_rows[i].get("image_path")
            row_image_refs.append(str(_img_ref) if _img_ref else None)
            sp_dicts.append(row.get("sampling_params", {}))
            valid_indices.append(i)

        if _oversized_count:
            print(
                f"[{stage_name}] WARNING: {_oversized_count} prompts exceed model "
                f"context length ({_max_model_len}) and will be skipped"
            )

        # -----------------------------------------------------------------------
        # Inference: single-process path (DP handled via early return above)
        # -----------------------------------------------------------------------
        if True:  # noqa: SIM108 — preserves indentation after DP branch removal
            # Build SamplingParams objects with dedup optimization
            sp_objects: List[Any] = []
            _sp_cache: Dict[int, Any] = {}  # id(dict) -> SamplingParams
            for sp_dict in sp_dicts:
                sp_id = id(sp_dict)
                if sp_id not in _sp_cache:
                    _sp_cache[sp_id] = _build_sampling_params(sp_dict)
                sp_objects.append(_sp_cache[sp_id])

            if len(_sp_cache) == 1 and sp_objects:
                sampling_params_list = sp_objects[0]  # single object, vLLM broadcasts
                print(f"[{stage_name}] Using shared SamplingParams for all {len(prompts)} prompts")
            else:
                sampling_params_list = sp_objects

            try:
                batch_size = int(getattr(cfg.model, "batch_size", 0) or 0)
            except Exception:
                batch_size = 0
            if batch_size <= 0:
                batch_size = max(len(prompts), 1)

            # Check if any rows have multimodal data
            _has_any_images = any(imgs is not None for imgs in row_images)

            # Run inference in batches
            print(
                f"[{stage_name}] Running inference on {len(prompts)} prompts "
                f"(batch_size={batch_size}, multimodal={_has_any_images})..."
            )
            outputs = []
            shared_sp = sampling_params_list if not isinstance(sampling_params_list, list) else None
            for start in range(0, len(prompts), batch_size):
                end = min(start + batch_size, len(prompts))
                prompt_batch = prompts[start:end]
                sampling_batch = shared_sp if shared_sp else sampling_params_list[start:end]
                print(
                    f"[{stage_name}] Generating batch {start // batch_size + 1}: "
                    f"rows {start}-{end - 1}",
                )

                if _has_any_images:
                    # Multimodal path: build per-prompt dicts with multi_modal_data
                    from vllm import TokensPrompt
                    mm_prompts = []
                    for j in range(start, end):
                        imgs = row_images[j]
                        if imgs:
                            # Tokenize with image placeholders to get token IDs
                            token_ids = tokenizer.encode(prompt_batch[j - start])
                            mm_prompts.append(TokensPrompt(
                                prompt_token_ids=token_ids,
                                multi_modal_data={"image": imgs if len(imgs) > 1 else imgs[0]},
                            ))
                        else:
                            mm_prompts.append(prompt_batch[j - start])
                    outputs.extend(llm.generate(mm_prompts, sampling_batch, lora_request=lora_request))
                else:
                    outputs.extend(llm.generate(prompt_batch, sampling_batch, lora_request=lora_request))

            # Verify output count matches input count
            if len(outputs) != len(prompts):
                raise RuntimeError(
                    f"[{stage_name}] vLLM output count mismatch: "
                    f"expected {len(prompts)} outputs for {len(prompts)} prompts, "
                    f"got {len(outputs)}. This indicates silent data loss."
                )

            # Postprocess — merge inference outputs back with failed rows
            print(f"[{stage_name}] Postprocessing {len(outputs)} outputs...")
            results: List[Dict[str, Any]] = []
            output_iter = iter(outputs)
            for idx, row in enumerate(preprocessed_rows):
                if idx in failed_set:
                    row["generated_text"] = ""
                    row["generated_reasoning"] = ""
                    try:
                        result = postprocess(row)
                    except Exception as e:
                        row["__postprocess_error__"] = str(e)
                        result = row
                    results.append(result)
                    continue

                output = next(output_iter)
                if output.outputs:
                    raw_text = output.outputs[0].text
                    reasoning, content = _split_reasoning(
                        raw_text, _model_source, _thinking_enabled, tokenizer,
                    )
                    row["generated_text"] = content
                    row["generated_reasoning"] = reasoning
                else:
                    row["generated_text"] = ""
                    row["generated_reasoning"] = ""

                try:
                    prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0
                    completion_tokens = (
                        len(output.outputs[0].token_ids) if output.outputs and output.outputs[0].token_ids else 0
                    )
                    row["usage"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                except Exception:
                    row["usage"] = None

                try:
                    result = postprocess(row)
                except Exception as e:
                    row["__postprocess_error__"] = str(e)
                    result = row
                results.append(result)

        print(f"[{stage_name}] Completed inference, {len(results)} results")
        return pd.DataFrame(results)
    finally:
        _shutdown_llm(llm, stage_name=stage_name)


# ---------------------------------------------------------------------------
# Embedding inference via LLM.encode()
# ---------------------------------------------------------------------------

def run_vllm_embed(
    df: pd.DataFrame,
    cfg,
    preprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    postprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    stage_name: str = "vllm_embed",
) -> pd.DataFrame:
    """Run vLLM embedding inference on a DataFrame.

    Uses ``LLM(runner="pooling")`` and ``llm.embed()`` per the vLLM pooling
    model API.  See: https://docs.vllm.ai/en/latest/models/pooling_models/embed/

    The preprocess function must set ``row["messages"]`` (list of chat dicts).
    The postprocess function receives ``row["embedding"]`` (numpy ndarray).

    Args:
        df: Input DataFrame.
        cfg: Hydra config with model.model_source, model.engine_kwargs, etc.
        preprocess: Row dict -> row dict with "messages".
        postprocess: Row dict (with "embedding") -> final row dict.
        stage_name: Label for log messages.

    Returns:
        pd.DataFrame of postprocessed results.
    """
    import numpy as np
    from PIL import Image

    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    if df is None or len(df) == 0:
        print(f"[{stage_name}] Empty input, returning empty DataFrame")
        return pd.DataFrame()

    # Set runtime env vars before importing vLLM
    for k, v in {**get_pcie_nccl_env_vars(), **get_vllm_runtime_env_vars()}.items():
        os.environ.setdefault(k, v)

    env_snapshot = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "SLURM_GPUS_ON_NODE": os.environ.get("SLURM_GPUS_ON_NODE", "<unset>"),
    }
    print(f"[{stage_name}] Runtime env: {env_snapshot}")

    from vllm import LLM

    engine_kwargs = _build_engine_kwargs(cfg)

    # runner="pooling" tells vLLM to use the pooling/embedding pipeline.
    engine_kwargs["runner"] = "pooling"

    # Pop data_parallel_size — vLLM 0.19 workers get DP config from env vars
    dp_size = int(engine_kwargs.pop("data_parallel_size", 1) or 1)

    print(f"[{stage_name}] Initializing vLLM embedding engine: "
          f"{ {k: v for k, v in engine_kwargs.items() if k != 'model'} }")
    print(f"[{stage_name}] Model: {engine_kwargs.get('model')}")
    if dp_size > 1:
        print(f"[{stage_name}] Data parallelism enabled: {dp_size} replicas "
              f"x TP={engine_kwargs.get('tensor_parallel_size', 1)}")

    # For DP mode, load tokenizer standalone; for single-process, create LLM.
    if dp_size > 1:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            engine_kwargs["model"], trust_remote_code=True
        )
        llm = None
    else:
        llm = LLM(**engine_kwargs)
        tokenizer = llm.llm_engine.tokenizer

    try:
        # ── Phase 1: Preprocess rows into messages (no image loading) ─────
        print(f"[{stage_name}] Preprocessing {len(df)} rows...")
        preprocessed_rows: List[Dict[str, Any]] = []
        failed_indices: List[int] = []
        for idx, row in enumerate(df.to_dict("records")):
            try:
                preprocessed_rows.append(preprocess(row))
            except Exception as e:
                row["__preprocess_error__"] = str(e)
                print(f"[{stage_name}] Preprocess error on row {idx}: {e}")
                preprocessed_rows.append(row)
                failed_indices.append(idx)

        failed_set = set(failed_indices)

        # ── Phase 2: Build prompt texts + image paths (lightweight) ───────
        chat_template_kwargs = dict(
            getattr(cfg.model, "chat_template_kwargs", {}) or {}
        )

        prompt_texts: List[str] = []
        image_refs: List[Optional[str]] = []
        valid_indices: List[int] = []

        for i in range(len(preprocessed_rows)):
            if i in failed_set:
                continue
            row = preprocessed_rows[i]
            messages = row.get("messages")
            if not messages:
                continue

            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    **chat_template_kwargs,
                )
            except Exception:
                try:
                    flat = _flatten_messages_for_template(messages)
                    prompt_text = tokenizer.apply_chat_template(
                        flat, tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    parts = []
                    for msg in messages:
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                b.get("text", "") for b in content
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        parts.append(content)
                    prompt_text = "\n".join(parts)

            img_ref = None
            for msg in messages:
                for item in (msg.get("content", []) if isinstance(msg.get("content"), list) else []):
                    if isinstance(item, dict) and item.get("type") == "image":
                        ref = item.get("image")
                        if isinstance(ref, str):
                            img_ref = ref.removeprefix("file://")
                        elif isinstance(ref, Image.Image):
                            img_ref = ref

            prompt_texts.append(prompt_text)
            image_refs.append(img_ref)
            valid_indices.append(i)

        batch_size = int(getattr(cfg.model, "batch_size", 0) or 0)
        if batch_size <= 0:
            batch_size = max(len(prompt_texts), 1)

        # ── Phase 3: Embed — data-parallel or single-process ─────────────
        if dp_size > 1:
            print(f"[{stage_name}] Running data-parallel embedding: "
                  f"{len(prompt_texts)} inputs across {dp_size} replicas...")
            all_embeddings = _run_data_parallel_embed(
                engine_kwargs=engine_kwargs,
                dp_size=dp_size,
                prompt_texts=prompt_texts,
                image_refs=image_refs,
                stage_name=stage_name,
                batch_size=batch_size,
            )

            if len(all_embeddings) != len(prompt_texts):
                raise RuntimeError(
                    f"[{stage_name}] DP embed output count mismatch: "
                    f"expected {len(prompt_texts)}, got {len(all_embeddings)}"
                )

            # Postprocess
            print(f"[{stage_name}] Postprocessing {len(all_embeddings)} embeddings...")
            results: List[Dict[str, Any]] = []
            emb_iter = iter(all_embeddings)
            for idx, row in enumerate(preprocessed_rows):
                if idx in failed_set:
                    row["embedding"] = None
                    try:
                        result = postprocess(row)
                    except Exception as e:
                        row["__postprocess_error__"] = str(e)
                        result = row
                    results.append(result)
                    continue

                row["embedding"] = next(emb_iter)
                try:
                    result = postprocess(row)
                except Exception as e:
                    row["__postprocess_error__"] = str(e)
                    result = row
                results.append(result)

        else:
            # Single-process path
            total_batches = (len(prompt_texts) + batch_size - 1) // batch_size
            print(f"[{stage_name}] Embedding {len(prompt_texts)} inputs "
                  f"(batch_size={batch_size}, {total_batches} batches)...")

            all_outputs = []
            for start in range(0, len(prompt_texts), batch_size):
                end = min(start + batch_size, len(prompt_texts))
                print(f"[{stage_name}] Embedding batch {start // batch_size + 1}/{total_batches}: "
                      f"rows {start}-{end - 1}")

                batch_inputs = []
                for j in range(start, end):
                    embed_input: Dict[str, Any] = {"prompt": prompt_texts[j]}
                    ref = image_refs[j]
                    if ref is not None:
                        if isinstance(ref, Image.Image):
                            embed_input["multi_modal_data"] = {"image": ref}
                        else:
                            embed_input["multi_modal_data"] = {
                                "image": Image.open(ref).convert("RGB")
                            }
                    batch_inputs.append(embed_input)

                batch_outputs = llm.embed(batch_inputs)
                all_outputs.extend(batch_outputs)
                del batch_inputs

            if len(all_outputs) != len(prompt_texts):
                raise RuntimeError(
                    f"[{stage_name}] vLLM embed output count mismatch: "
                    f"expected {len(prompt_texts)}, got {len(all_outputs)}"
                )

            # Postprocess
            print(f"[{stage_name}] Postprocessing {len(all_outputs)} embeddings...")
            results: List[Dict[str, Any]] = []
            output_iter = iter(all_outputs)
            for idx, row in enumerate(preprocessed_rows):
                if idx in failed_set:
                    row["embedding"] = None
                    try:
                        result = postprocess(row)
                    except Exception as e:
                        row["__postprocess_error__"] = str(e)
                        result = row
                    results.append(result)
                    continue

                output = next(output_iter)
                emb_data = output.outputs.embedding
                if not isinstance(emb_data, np.ndarray):
                    emb_data = np.array(emb_data, dtype=np.float32)
                row["embedding"] = emb_data
                try:
                    result = postprocess(row)
                except Exception as e:
                    row["__postprocess_error__"] = str(e)
                    result = row
                results.append(result)

        print(f"[{stage_name}] Completed embedding, {len(results)} results")
        return pd.DataFrame(results)
    finally:
        _shutdown_llm(llm, stage_name=stage_name)
