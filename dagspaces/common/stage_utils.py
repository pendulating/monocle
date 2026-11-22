"""Shared stage utilities for all dagspace stage modules.

This module consolidates utility functions that were previously duplicated across
multiple stage files. Import from here rather than defining local copies.

Public API
----------
maybe_silence_vllm_logs()
    Configure vLLM logger levels based on environment variables.

to_json_str(value)
    Serialize an arbitrary Python value to a JSON string.

serialize_arrow_unfriendly_in_row(row, columns)
    In-place convert nested/non-serializable columns to JSON strings in a row dict.

extract_last_json(text)
    Extract the last JSON object or array from a text string.

sanitize_for_json(value)
    Recursively convert a value to JSON-serializable builtins.

run_with_retry(fn, ...)
    Call a function again with exponential backoff after a failure.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar

T = TypeVar("T")

__all__ = [
    "ensure_dotenv",
    "maybe_silence_vllm_logs",
    "to_json_str",
    "serialize_arrow_unfriendly_in_row",
    "extract_last_json",
    "sanitize_for_json",
    "resolve_thinking_mode",
    "run_with_retry",
]

# Module-level guards so one-time setup work only runs once per process.
_DOTENV_LOADED = False
_VLLM_LOGS_SILENCED = False


def _find_project_root() -> Optional[str]:
    """Return the first folder above this file that holds ``server.env``, ``.env`` or ``pyproject.toml``."""
    search_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        for marker in ("server.env", ".env", "pyproject.toml"):
            if os.path.isfile(os.path.join(search_dir, marker)):
                return search_dir
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent
    return None


def ensure_dotenv() -> None:
    """Load the site settings (``server.env``) and the secrets (``.env``) into ``os.environ``.

    ``server.env`` holds the site settings: the SLURM partition, the venv, the
    caches and the NCCL keys. ``IMAGE_FORWARD_SERVER_ENV`` can name a
    different file. ``.env`` holds the API keys. Each file is optional, and
    each loads without the other. A variable that the environment already
    sets is not overwritten. A second call does nothing.

    See ``server.env.example`` for the keys.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        return
    root = _find_project_root()
    site_env = os.environ.get("IMAGE_FORWARD_SERVER_ENV") or (
        os.path.join(root, "server.env") if root else None
    )
    for env_file in (site_env, os.path.join(root, ".env") if root else None):
        if env_file and os.path.isfile(env_file):
            _load(env_file, override=False)


def maybe_silence_vllm_logs() -> None:
    """Configure vLLM logger levels according to environment variables.

    Reads ``VLLM_LOGGING_LEVEL`` (default ``"WARNING"``).  Also installs a
    ``PatternModuloFilter`` on the root ``vllm`` logger to throttle the
    high-frequency "Elapsed time for batch" message; the throttle period is
    controlled by ``IMAGE_FORWARD_VLLM_LOG_EVERY`` (default ``10``).

    The function is idempotent: subsequent calls after the first are no-ops.
    All exceptions are silently swallowed so that a missing ``vllm`` package
    does not break callers.
    """
    global _VLLM_LOGS_SILENCED
    if _VLLM_LOGS_SILENCED:
        return
    try:
        vllm_log_level = os.environ.get("VLLM_LOGGING_LEVEL", "WARNING")

        _LEVEL_MAP = {
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
        }
        level = _LEVEL_MAP.get(vllm_log_level.upper(), logging.WARNING)

        for name in ("vllm", "vllm.logger", "vllm.engine", "vllm.core", "vllm.worker"):
            lg = logging.getLogger(name)
            lg.setLevel(level)
            lg.propagate = True

        # Install a modulo filter on the root vllm logger to throttle the
        # high-frequency "Elapsed time for batch" message.
        try:
            from dagspaces.common.logging_filters import PatternModuloFilter

            root_vllm = logging.getLogger("vllm")
            try:
                n = int(os.environ.get("IMAGE_FORWARD_VLLM_LOG_EVERY", "10") or "10")
            except Exception:
                n = 10
            existing = getattr(root_vllm, "filters", [])
            if not any(
                getattr(f, "__class__", object).__name__ == "PatternModuloFilter"
                for f in existing
            ):
                root_vllm.addFilter(PatternModuloFilter(mod=n, pattern="Elapsed time for batch"))
        except Exception:
            pass

        _VLLM_LOGS_SILENCED = True
    except Exception:
        pass


def to_json_str(value: Any) -> Optional[str]:
    """Serialize *value* to a JSON string.

    Returns ``None`` when *value* is ``None``.  Falls back to ``str(value)``
    if ``json.dumps`` raises.
    """
    try:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        try:
            return str(value)
        except Exception:
            return None


def serialize_arrow_unfriendly_in_row(
    row: Dict[str, Any],
    columns: Iterable[str],
) -> None:
    """In-place convert non-Arrow-serializable column values to JSON strings.

    Handles ``dict``, ``list``, ``tuple``, and the vLLM structured-output and
    ``SamplingParams`` objects.  The vLLM types are imported lazily, so the
    function stays usable where vLLM is not installed.

    The structured-output type has a different name in each vLLM generation:
    ``GuidedDecodingParams`` up to 0.11, ``StructuredOutputsParams`` from 0.12.
    vLLM 0.25 removed ``GuidedDecodingParams``.  Import each type on its own —
    one import statement for both would lose ``SamplingParams`` as well when
    only the structured-output name is absent.
    """
    _vllm_types: tuple = ()
    for _mod, _name in (
        ("vllm.sampling_params", "SamplingParams"),
        ("vllm.sampling_params", "StructuredOutputsParams"),
        ("vllm.sampling_params", "GuidedDecodingParams"),
    ):
        try:
            import importlib

            _t = getattr(importlib.import_module(_mod), _name, None)
            if isinstance(_t, type):
                _vllm_types = _vllm_types + (_t,)
        except Exception:
            pass

    for col in columns:
        if col not in row:
            continue
        val = row[col]
        if isinstance(val, (dict, list, tuple)):
            row[col] = to_json_str(val)
        elif _vllm_types and isinstance(val, _vllm_types):
            row[col] = str(val)


def extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the last JSON object from *text*.

    First attempts to parse the entire string.  On failure, uses a regex to
    locate all ``{...}`` blocks and tries each from the last one backwards.
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            import re as _re

            snippets = _re.findall(r"\{[\s\S]*\}", text)
            for snip in reversed(snippets or []):
                try:
                    obj = json.loads(snip)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue
        except Exception:
            pass
        return None
    except Exception:
        return None


def sanitize_for_json(value: Any) -> Any:
    """Recursively convert *value* to JSON-serializable Python builtins."""
    _GuidedDecodingParams = None
    _SamplingParams = None
    try:
        from vllm.sampling_params import GuidedDecodingParams, SamplingParams  # type: ignore

        _GuidedDecodingParams = GuidedDecodingParams
        _SamplingParams = SamplingParams
    except Exception:
        pass

    try:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if _GuidedDecodingParams is not None and isinstance(value, _GuidedDecodingParams):
            try:
                return {
                    "json": getattr(value, "json", None),
                    "disable_fallback": getattr(value, "disable_fallback", True),
                    "disable_additional_properties": getattr(
                        value, "disable_additional_properties", True
                    ),
                }
            except Exception:
                return str(value)
        if _SamplingParams is not None and isinstance(value, _SamplingParams):
            return str(value)
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for k, v in value.items():
                try:
                    key = str(k)
                except Exception:
                    key = repr(k)
                out[key] = sanitize_for_json(v)
            return out
        if isinstance(value, (list, tuple, set)):
            return [sanitize_for_json(v) for v in value]
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                return sanitize_for_json(tolist())
            except Exception:
                pass
        return str(value)
    except Exception:
        return None


def resolve_thinking_mode(cfg_model: Any, *, default: bool = True) -> bool:
    """Resolve the model's thinking mode to a single boolean.

    This is the **single source of truth** for the thinking mode contract.
    All phases should derive their thinking flag from this helper.

    Priority (highest wins):

    1. ``cfg_model.thinking_mode`` -- preferred, string ``"on"``/``"off"``
       (also accepts bool or ``"true"``/``"false"``/``"yes"``/``"no"``/
       ``"auto"``). ``"auto"`` defers to priority 2.
    2. ``cfg_model.chat_template_kwargs.enable_thinking`` -- legacy bool,
       still honored for backwards compatibility.
    3. The ``default`` kwarg (``True``).
    """
    def _coerce(val: Any) -> Optional[bool]:
        if val is None:
            return None
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("on", "true", "yes", "1", "enabled", "enable"):
                return True
            if s in ("off", "false", "no", "0", "disabled", "disable"):
                return False
            if s == "auto":
                return None  # defer
        return None

    # Priority 1: explicit thinking_mode field
    try:
        tm = getattr(cfg_model, "thinking_mode", None)
    except Exception:
        tm = None
    if tm is None and isinstance(cfg_model, dict):
        tm = cfg_model.get("thinking_mode")
    resolved = _coerce(tm)
    if resolved is not None:
        return resolved

    # Priority 2: chat_template_kwargs.enable_thinking (legacy)
    try:
        ctk = getattr(cfg_model, "chat_template_kwargs", None)
    except Exception:
        ctk = None
    if ctk is None and isinstance(cfg_model, dict):
        ctk = cfg_model.get("chat_template_kwargs")
    if ctk is not None:
        try:
            et = getattr(ctk, "enable_thinking", None)
        except Exception:
            et = None
        if et is None and isinstance(ctk, dict):
            et = ctk.get("enable_thinking")
        resolved = _coerce(et)
        if resolved is not None:
            return resolved

    return default


def run_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int,
    backoff_s: float,
    retryable: Tuple[Type[BaseException], ...] = (Exception,),
    on_retry: Optional[Callable[[int, BaseException], None]] = None,
    label: str = "operation",
) -> T:
    """Call ``fn`` up to ``attempts`` times with exponential backoff.

    The wait before the next try is ``backoff_s * 2**attempt``, with
    ``attempt`` counted from zero. The helper re-raises the last exception
    after the final attempt. An exception whose type is not in ``retryable``
    propagates at once, with no further try. ``on_retry(attempt, exc)`` runs
    before each wait; the helper logs a warning when ``on_retry`` is None.
    """
    tries = max(1, attempts)
    last_exc: Optional[BaseException] = None
    for attempt in range(tries):
        try:
            return fn()
        except retryable as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt >= tries - 1:
                break
            wait_s = backoff_s * (2 ** attempt)
            if on_retry is not None:
                on_retry(attempt, exc)
            else:
                logging.getLogger(__name__).warning(
                    "%s failed on attempt %d/%d (%s: %s); next try in %.1fs",
                    label, attempt + 1, tries, type(exc).__name__, exc, wait_s,
                )
            time.sleep(wait_s)
    assert last_exc is not None  # a failed retry loop always sets this
    raise last_exc
