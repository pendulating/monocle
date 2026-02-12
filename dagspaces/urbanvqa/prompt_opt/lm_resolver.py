from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import DictConfig


@dataclass(frozen=True)
class LMClientConfig:
    provider: str
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    request_timeout_s: Optional[float] = None
    max_tokens: Optional[int] = None

    def as_litellm_kwargs(self) -> Dict[str, Any]:
        """Translate into kwargs suitable for LiteLLM/OpenAI compatible clients."""
        payload: Dict[str, Any] = {"model": self.model}
        if self.api_base:
            payload["api_base"] = self.api_base
        if self.api_key:
            payload["api_key"] = self.api_key
        if self.headers:
            payload["custom_headers"] = dict(self.headers)
        if self.request_timeout_s is not None:
            payload["timeout"] = self.request_timeout_s
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload


def _resolve_api_key(entry: DictConfig) -> Optional[str]:
    if entry.get("api_key"):
        return entry.api_key
    if entry.get("api_key_env"):
        return os.getenv(entry.api_key_env)
    return None


def resolve_lm_client(entry: DictConfig) -> LMClientConfig:
    headers = entry.get("headers") or {}
    if not isinstance(headers, dict):
        headers = {}
    return LMClientConfig(
        provider=entry.provider,
        model=entry.model,
        api_base=entry.get("api_base"),
        api_key=_resolve_api_key(entry),
        headers=headers,
        request_timeout_s=entry.get("request_timeout_s"),
        max_tokens=entry.get("max_tokens"),
    )


def resolve_lm_clients(cfg: DictConfig) -> Dict[str, LMClientConfig]:
    """Resolve LLM client configurations from config.
    
    Only resolves the 'reflection' client. The 'task' client is deprecated
    as VQA inference uses cfg.model directly via run_vqa_stage().
    """
    llm_cfg = getattr(cfg, "llm", None)
    if llm_cfg is None:
        raise ValueError("cfg.llm section is required to resolve language models")
    clients: Dict[str, LMClientConfig] = {}
    # Only resolve reflection - task is deprecated (VQA uses cfg.model directly)
    entry = getattr(llm_cfg, "reflection", None)
    if entry is None:
        raise ValueError("cfg.llm.reflection is missing")
    clients["reflection"] = resolve_lm_client(entry)
    return clients

