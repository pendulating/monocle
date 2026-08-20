"""A loaded vLLM engine that answers a batch of text prompts under a JSON schema.

Two stages share this engine.

- `trace_extract` drives it through LangExtract, which builds the prompt from
  few-shot examples and derives the schema from the same examples.
- `ic_extract` drives it directly, with a schema written by hand.

The engine knows about neither. It renders a chat prompt, applies a guided-JSON
constraint, and gives back the content without the thought block.

Warning: call `shutdown()`. `vllm.LLM` does not stop its worker processes on its
own, and a live worker holds the SLURM job open until the walltime ends.

See `vlm-narratives-docs/ic-ingredient-extraction.md`.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

__version__ = "1.0.0"

# The default token budget of 1 answer.
DEFAULT_MAX_TOKENS = 2048


def repair_truncated_json(text: str) -> Optional[str]:
    """Close a LangExtract answer that the token cap cut in the middle.

    Guided decoding writes valid JSON up to the cap. The cap then cuts it, and
    the parser drops the WHOLE answer — every extraction, not only the last
    one. This function cuts back to the last extraction that closed, and it
    closes the array and the object.

    Warning: this is the shape of the LangExtract answer, `{"extractions":
    [...]}`, where depth 2 is the array. A schema with more than 1 array needs
    `ic_schema.close_truncated_json`, which reads the real bracket stack.

    Returns:
        The repaired text, or None when no extraction closed.
    """
    depth = 0
    in_string = False
    escape = False
    last_element_end: Optional[int] = None
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            # Depth 3 is 1 extraction inside the array inside the root object.
            # A close that returns to 2 ends an extraction.
            if depth == 2:
                last_element_end = i + 1
    if last_element_end is None:
        return None
    return text[:last_element_end] + "]}"


def _to_plain(value: Any) -> Any:
    """Turn an OmegaConf node into a plain dict, list, or scalar."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


class VLLMEngine:
    """A loaded vLLM engine, with the chat template of its own tokenizer.

    The engine stays alive for the whole stage. `run_vllm_inference` cannot
    serve this shape: it takes a whole DataFrame and returns when it is done,
    but an extraction stage calls the engine one time for each batch.
    """

    def __init__(self, cfg: Any, stage_name: str = "structured_extract") -> None:
        from dagspaces.common import vllm_inference as vi
        from dagspaces.common.stage_utils import resolve_thinking_mode

        self._vi = vi
        self.stage_name = stage_name
        self.model_source = str(getattr(cfg.model, "model_source"))

        # The extractor tags text. It never thinks out loud, and a thought block
        # only wastes decode steps. Default it off, and let a config say
        # otherwise.
        self.thinking_enabled = resolve_thinking_mode(cfg.model, default=False)
        self.chat_template_kwargs = dict(
            _to_plain(getattr(cfg.model, "chat_template_kwargs", {})) or {}
        )
        self.system_prompt = str(getattr(cfg.model, "system_prompt", "") or "")

        for key, value in {
            **vi.get_pcie_nccl_env_vars(),
            **vi.get_vllm_runtime_env_vars(),
        }.items():
            os.environ.setdefault(key, value)

        from vllm import LLM

        engine_kwargs = vi._build_engine_kwargs(cfg)
        # Data parallelism belongs to `run_vllm_inference`, which shards a
        # DataFrame across worker processes. This path holds 1 engine, thus the
        # key would reach `LLM()` and raise.
        engine_kwargs.pop("data_parallel_size", None)
        print(f"[{stage_name}] engine kwargs: {engine_kwargs}", flush=True)

        # The number of answers that reached the token cap, and the number that
        # `repair_fn` saved. See `generate`.
        self.truncated = 0
        self.repaired = 0

        # How to close a cut answer. `trace_extract` keeps the default.
        # `ic_extract` sets this to None, because its own parser repairs the
        # answer AND reports the repair for each row.
        self.repair_fn: Optional[Callable[[str], Optional[str]]] = repair_truncated_json

        # The finish reason of each answer of the LAST `generate` call. A caller
        # that must mark a cut answer per row reads this; `truncated` above only
        # counts.
        self.last_finish_reasons: List[str] = []

        # Debug: write the first N raw answers to a JSONL file. A silent trace
        # gives no row, thus the parquet cannot say WHY it was silent.
        self.debug_answers_path: Optional[str] = None
        self.debug_answers_left = 0

        t0 = time.time()
        self.llm = LLM(**engine_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        print(f"[{stage_name}] engine ready in {time.time() - t0:.1f}s", flush=True)

    def render(self, prompt: str) -> str:
        """Put 1 prompt into the model's chat format."""
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
        except Exception as exc:
            print(f"[{self.stage_name}] chat template failed: {exc}", flush=True)
            return prompt

    def generate(
        self,
        prompts: Sequence[str],
        *,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        guided_json: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Answer every prompt. Give back the content, without the thought."""
        sp_dict: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if guided_json:
            sp_dict["structured_output"] = {"json": guided_json}
        sampling_params = self._vi._build_sampling_params(sp_dict)

        rendered = [self.render(p) for p in prompts]
        # `use_tqdm=False`: the bar writes 1 line for each step into the SLURM
        # log, and it hides every message the stage prints.
        outputs = self.llm.generate(rendered, sampling_params, use_tqdm=False)

        answers: List[str] = []
        self.last_finish_reasons = []
        for out in outputs:
            raw = out.outputs[0].text if out.outputs else ""
            reason = (
                str(getattr(out.outputs[0], "finish_reason", "")) if out.outputs else ""
            )
            self.last_finish_reasons.append(reason)
            # A truncated answer is an INVISIBLE loss. Guided decoding writes
            # valid JSON up to the cap, the cap cuts it, and the parser then
            # returns nothing at all. The trace looks silent when it is only
            # cut off. Count it, so the rate reaches the metadata.
            if reason == "length":
                self.truncated += 1
                if self.repair_fn is not None:
                    fixed = self.repair_fn(raw)
                    if fixed is not None:
                        self.repaired += 1
                        raw = fixed
            # A model that still emits a thought block hides the JSON behind it.
            # The family parser removes the block; the regex path catches the
            # rest.
            _, content = self._vi._split_reasoning(
                raw, self.model_source, self.thinking_enabled, self.tokenizer
            )
            answer = content or raw
            self._dump(out, raw, answer)
            answers.append(answer)
        return answers

    def _dump(self, out: Any, raw: str, answer: str) -> None:
        """Write 1 raw answer to the debug file, while the budget lasts."""
        if not self.debug_answers_path or self.debug_answers_left <= 0:
            return
        self.debug_answers_left -= 1
        record = {
            "finish_reason": (
                getattr(out.outputs[0], "finish_reason", "") if out.outputs else ""
            ),
            "output_tokens": (
                len(getattr(out.outputs[0], "token_ids", []) or []) if out.outputs else 0
            ),
            "raw_chars": len(raw),
            "raw": raw[:8000],
            "answer_differs": answer != raw,
        }
        try:
            with open(self.debug_answers_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            print(f"[{self.stage_name}] debug dump failed: {exc}", flush=True)
            self.debug_answers_path = None

    def as_language_model(self, spec: Any) -> Any:
        """Wrap this engine for `lx.extract`.

        The import is late on purpose. This module must not need LangExtract,
        because `ic_extract` uses the engine and never touches that library.
        """
        from dagspaces.common.langextract_backend import VLLMLanguageModel

        guided_json = spec.guided_json()

        def _generate(prompts: Sequence[str]) -> List[str]:
            return self.generate(
                prompts,
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
                guided_json=guided_json,
            )

        return VLLMLanguageModel(_generate, stage_name=self.stage_name)

    def shutdown(self) -> None:
        self._vi._shutdown_llm(getattr(self, "llm", None), self.stage_name)
        self.llm = None
