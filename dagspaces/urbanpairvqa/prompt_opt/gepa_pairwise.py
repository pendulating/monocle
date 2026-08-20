"""GEPA prompt optimization for pairwise VQA — self-distillation setup.

Takes a completed pairwise production run as the supervised dataset (the task
model's OWN labels, in presented orientation) and searches for the prompt that
best reproduces them — a "retranslation" of the original prompt by the same
model that produced the classifications. The same in-process engine serves as
both the task model and the GEPA reflection model, so one GPU suffices.

Differences from the urbanvqa GEPA stack (dagspaces/urbanvqa/prompt_opt/):
  * two images per example (presented left/right), limit_mm_per_prompt=2
  * gemma4_unified support: the custom chat_template.jinja + {"type":"image"}
    blocks + TokensPrompt/multi_modal_data path proven by the pairwise
    production runs (llm.chat is NOT gated for the unified arch)
  * ordinal 5-class metric (1 - |ord(pred) - ord(expected)| / 4) instead of
    binary yes/no F1
  * in-process reflection (str -> str callable on the shared engine) instead
    of a separate OpenAI-compatible vLLM server

Run on a GPU node under .venv-nightly (vLLM >= 0.23 for gemma-12b):

  python -m dagspaces.urbanpairvqa.prompt_opt.gepa_pairwise \\
      --result-parquet multirun/2026-06-29_URBANPAIRVQA/14-37-17/0/outputs/pairwise/subway_safety_mvp_20260629_143729.parquet \\
      --outdir outputs/gepa_pairwise/subway_gemma12b_smoke

See scripts/gepa_pairwise_subway.sub for the sbatch wrapper.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import yaml

try:
    from gepa.core.adapter import GEPAAdapter, EvaluationBatch
except ImportError:  # pragma: no cover - mirrors urbanvqa fallback shim
    import dataclasses

    class GEPAAdapter:  # type: ignore[too-many-ancestors]
        def evaluate(self, batch, candidate, capture_traces=False):
            raise NotImplementedError

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            raise NotImplementedError

    @dataclasses.dataclass
    class EvaluationBatch:  # type: ignore[no-redef] - lets adapter tests run without gepa
        outputs: list
        scores: list
        trajectories: Any = None

from dagspaces.urbanpairvqa.stages.pairwise_vqa import _canonicalize_label

LOG = logging.getLogger(__name__)
REPO = Path("/share/pierson/matt/mllmsci")

ORDINAL = ["MuchLess", "Less", "Same", "More", "MuchMore"]
ORD_IDX = {lab: i for i, lab in enumerate(ORDINAL)}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ORDINAL}},
    "required": ["answer"],
    "additionalProperties": False,
}
# Matches the fixed line the pairwise stage hard-appends after the evolvable
# template (_render_pair_prompt). Kept OUT of the candidate so GEPA cannot
# destroy the A/B anchor, and so best_prompts.yaml maps 1:1 onto
# prompt.user_template for later validation through the real stage.
FIXED_SUFFIX = "Interpret labels as Image A relative to Image B."

# Caption prompts for --reflection-descriptions. Deliberately TASK-NEUTRAL —
# the whole point of the caption-mediated variant is that any task semantics
# reaching the reflector must come from the model's own visual grounding, not
# from us. Do not mention the task, the attribute, or the label scale here.
#
# Two styles, forming successive rungs of the semantics-channel ladder:
#   independent  — per-image factual captions; comparison AND preference
#                  forbidden (the reflector must do the differencing itself)
#   contrastive  — difference-focused; comparatives and soft impressions
#                  allowed, but an overall verdict/preference stays forbidden
#                  (the task's relational structure passes through; the
#                  evaluative verdict still can only come from the model)
DESCRIBE_SYSTEM = "You are a careful observer describing photographs."
DESCRIBE_PROMPT = (
    "Describe what each of the two photographs shows, factually and "
    "concretely, in two or three sentences each. Do not compare the two "
    "photographs and do not state any preference. Format your answer as:\n"
    "Image A: <description>\nImage B: <description>"
)
DESCRIBE_CONTRASTIVE_PROMPT = (
    "Describe the most notable differences between the two photographs, "
    "factually and concretely, in three or four sentences. Comparative "
    "statements are encouraged: say which photograph shows more or less of "
    "something, and how the two scenes differ in character or overall "
    "impression. Do not state which photograph you prefer or judge one as "
    "better overall. Refer to the photographs as Image A and Image B."
)
DESCRIBE_STYLES = {
    "independent": DESCRIBE_PROMPT,
    "contrastive": DESCRIBE_CONTRASTIVE_PROMPT,
}

# Prefixed to reflective-record Inputs when captions are attached, to prevent
# the modality confusion observed in the first caption-mediated sweep (evolved
# prompts described the task as comparing "text descriptions").
REFLECTION_INPUT_NOTE = (
    "The assistant being optimized sees the two photographs themselves, not "
    "any text. The description below is a separate observer's account of the "
    "same photographs, provided only to help you understand this example."
)

# --- axis-slot candidate mode ------------------------------------------------
# Everything except the attribute phrase is frozen: the scaffold carries the
# response-style calibration that free-form GEPA repeatedly rediscovered from
# scratch (A-relative-to-B polarity discipline, Same-suppression, conservative
# Much*), written here task-free. GEPA then evolves ONLY the axis phrase, so
# all search pressure goes to semantics and the score landscape over axes is
# directly interpretable.
AXIS_SYSTEM = (
    "You are a careful visual judge. You will be shown two street-level "
    "photographs: Image A (first) and Image B (second). Answer with exactly "
    "one label from MuchLess, Less, Same, More, MuchMore, describing Image A "
    "relative to Image B on the attribute in the question. Polarity is the "
    "highest priority: first decide which image shows the attribute more "
    "strongly, then double-check you have not swapped Image A and Image B. "
    "If you can see a real difference between the two photographs, choose a "
    "directional label rather than Same. Reserve MuchLess and MuchMore for "
    "stark, unmistakable differences."
)
AXIS_USER_TEMPLATE = (
    "Which photograph looks more {axis}: Image A or Image B?\n\n"
    "Return exactly one label from: MuchLess, Less, Same, More, MuchMore. "
    'Interpret this as "Image A looks <label> {axis} than Image B".'
)
AXIS_COMPONENT_NOTE = (
    "The text under optimization is a short attribute phrase X completing "
    "the question 'Which photograph looks more X: Image A or Image B?'. "
    "Propose only a short descriptive phrase (at most a dozen words), not a "
    "full prompt."
)
# Replaces gepa's default reflection template in axis mode. The default asks
# the reflector to "write a new instruction", which reliably yields meta-
# prompts ("You are an expert visual analyst... generate a phrase X...")
# instead of phrases — observed for 11/11 proposals in job 840227. Must keep
# the <curr_param>/<side_info> placeholders; task-neutral like everything
# else the reflector sees.
AXIS_REFLECTION_TEMPLATE = """An assistant compares pairs of street-level photographs (Image A and Image B) and answers with one label from MuchLess, Less, Same, More, MuchMore, describing Image A relative to Image B on a single attribute. The attribute is given by a short phrase X, used in the question "Which photograph looks more X: Image A or Image B?".

The current attribute phrase X is:
```
<curr_param>
```

The following are examples of photograph pairs (each with an observer's description), the assistant's answer under the current attribute phrase, and feedback on the expected answer:
```
<side_info>
```

Propose a NEW attribute phrase X so that the assistant's answers better match the expected answers. Study the examples and infer what property of the scenes the expected answers track, then name that property as a short phrase. The phrase must describe a concrete attribute of a photographed scene, in at most a dozen words. It must NOT be an instruction, NOT a full prompt, and NOT an explanation.

Provide ONLY the new attribute phrase within ``` blocks."""

# Multimodal reflection (rung 5): the reflector is shown the ACTUAL failing
# pairs, not captions of them. gepa replaces each embedded Image with an
# "[IMAGE-N]" text placeholder and appends the pixels as inline content, so the
# template refers the reflector to the photographs themselves. Task-neutral,
# like every reflector-facing string. Keeps the <curr_param>/<side_info>
# placeholders and the "``` blocks" extraction contract.
MULTIMODAL_AXIS_REFLECTION_TEMPLATE = """An assistant compares pairs of street-level photographs (Image A and Image B) and answers with one label from MuchLess, Less, Same, More, MuchMore, describing Image A relative to Image B on a single attribute. The attribute is given by a short phrase X, used in the question "Which photograph looks more X: Image A or Image B?".

The current attribute phrase X is:
```
<curr_param>
```

Below are examples. Each includes the two photographs the assistant actually saw (shown as Image A and Image B), its answer under the current attribute phrase, and feedback on the expected answer:
```
<side_info>
```

Look at the photographs themselves. Infer what visible property of the scenes the expected answers track — the property on which Image A and Image B differ in the direction the feedback indicates — then name that property as a short phrase X. The phrase must describe a concrete, visible attribute of a photographed scene, in at most a dozen words. It must NOT be an instruction, NOT a full prompt, and NOT an explanation.

Provide ONLY the new attribute phrase within ``` blocks."""

# Shown per reflective record in image mode (task-neutral): tells the reflector
# the two photographs ARE what the assistant saw for that example.
MULTIMODAL_INPUT_NOTE = (
    "The two photographs below (Image A and Image B) are exactly what the "
    "assistant was shown for this example."
)
DEFAULT_SEED_AXIS = "notable"


def _parse_multimodal_reflection(messages: Any) -> tuple:
    """Decode gepa's multimodal reflection payload into (text, [PIL images]).

    When a reflective record carries ``gepa.Image`` objects, gepa calls the
    reflection_lm with a messages list
    ``[{"role":"user","content":[{"type":"text",...},
        {"type":"image_url","image_url":{"url":"data:...;base64,..."}}, ...]}]``
    (OpenAI vision format) instead of a plain string. This reverses that back
    into a text blob (with the inline ``[IMAGE-N]`` placeholders intact) plus
    the decoded images, in order, for the vLLM two-image prompt path.
    """
    import base64
    import io

    from PIL import Image as PILImage

    parts: List[Any] = []
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        content = messages[0].get("content", [])
        parts = content if isinstance(content, list) else [content]
    texts: List[str] = []
    images: List[Any] = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            texts.append(str(p.get("text", "")))
        elif p.get("type") == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            if "," in url:
                url = url.split(",", 1)[1]
            images.append(
                PILImage.open(io.BytesIO(base64.b64decode(url))).convert("RGB"))
    return "\n".join(texts), images


def _sanitize_axis(text: str) -> str:
    """Reduce a reflector proposal to a usable short attribute phrase.

    Reflection LMs asked for a 'new instruction' tend to return fenced,
    quoted, or multi-line text; the axis slot needs a bare phrase.
    """
    t = str(text).strip()
    for line in t.splitlines():
        line = line.strip().strip("`\"'*").strip()
        if not line or line.startswith("#"):
            continue
        # drop a leading "X:" / "Axis:" style tag if present
        if ":" in line[:20]:
            head, _, tail = line.partition(":")
            if len(head.split()) <= 2 and tail.strip():
                line = tail.strip().strip("`\"'").strip()
        t = line
        break
    t = t.rstrip(".").strip()
    words = t.split()
    if len(words) > 16:
        t = " ".join(words[:16])
    return t or DEFAULT_SEED_AXIS

DEFAULT_RESULT_PARQUET = (
    REPO / "multirun/2026-06-29_URBANPAIRVQA/14-37-17/0/outputs/pairwise/"
    "subway_safety_mvp_20260629_143729.parquet"
)
DEFAULT_MODEL_DIR = "/share/pierson/matt/zoo/models/gemma-4-12B-it"
DEFAULT_SEED_PROMPT_YAML = (
    REPO / "dagspaces/urbanpairvqa/conf/prompt/pairwise_subway_safety_ordinal.yaml"
)

# Maximally generic seed for the prompt-recovery experiment: says nothing about
# what attribute is being compared or what the images depict. Tests whether
# reflection alone can climb from "compare the two images" to a task-accurate
# prompt using only the supervised labels.
GENERIC_SEED = {
    "system_prompt": "You are a helpful assistant.",
    "user_prompt": (
        "Compare the two images. Answer with exactly one label from: "
        "MuchLess, Less, Same, More, MuchMore."
    ),
}


# ---------------------------------------------------------------------------
# Supervised dataset from a production pairwise run
# ---------------------------------------------------------------------------

def build_frames(result_parquet: Path, train_n: int, val_n: int,
                 seed: int) -> tuple:
    """Stratified, disjoint train/val record lists from a production run.

    Targets are ``presented_label`` (the answer in PRESENTED orientation —
    the model sees images in presented order, so the supervised target must
    be too). NotSure rows are dropped: abstentions are NaN on the ordinal
    scale and would need a 6th enum entry the seed prompt doesn't define.
    """
    from dagspaces.urbanvqa.prompt_opt.dataset import stratified_sample

    df = pd.read_parquet(result_parquet, columns=[
        "pair_id", "repeat_idx", "presented_label",
        "presented_left_path", "presented_right_path"])
    df = df[(df["repeat_idx"] == 0) & df["presented_label"].isin(ORDINAL)]
    df = df.rename(columns={"presented_label": "expected_answer",
                            "pair_id": "sample_id"})

    train = stratified_sample(df, label_column="expected_answer",
                              num_rows=train_n, seed=seed)
    rest = df[~df["sample_id"].isin(set(train["sample_id"]))]
    val = stratified_sample(rest, label_column="expected_answer",
                            num_rows=val_n, seed=seed + 1)

    cols = ["sample_id", "expected_answer",
            "presented_left_path", "presented_right_path"]
    return (train[cols].to_dict(orient="records"),
            val[cols].to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Shared task + reflection engine
# ---------------------------------------------------------------------------

class PairwiseTaskEngine:
    """One in-process vLLM engine used for both task evals and reflection.

    Task calls follow the exact prompt-construction path the pairwise
    production runs use for gemma4_unified on klara_1x:
    apply_chat_template(custom chat_template.jinja, {"type":"image"} blocks)
    -> TokensPrompt + multi_modal_data -> llm.generate with structured output.
    """

    def __init__(self, model_dir: str, *, max_model_len: int = 16384,
                 gpu_memory_utilization: float = 0.90,
                 reflection_temperature: float = 0.6,
                 reflection_max_tokens: int = 4096,
                 limit_mm_images: int = 2) -> None:
        from vllm import LLM
        from dagspaces.common.vllm_inference import _gemma4_unified_chat_template

        self.model_dir = model_dir
        self.reflection_temperature = reflection_temperature
        self.reflection_max_tokens = reflection_max_tokens
        self.n_task_calls = 0
        self.n_reflection_calls = 0

        # Task prompts always carry exactly 2 images; multimodal reflection
        # (rung 5) shows the reflector a whole minibatch of failing pairs at
        # once, i.e. 2 x reflection_minibatch images in a single prompt, so the
        # per-prompt cap must be raised for that mode.
        LOG.info("Loading engine: %s (max_model_len=%d, limit_mm_images=%d)",
                 model_dir, max_model_len, limit_mm_images)
        self.llm = LLM(
            model=model_dir,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            limit_mm_per_prompt={"image": max(2, limit_mm_images)},
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.is_g4u, self.g4u_template = _gemma4_unified_chat_template(model_dir)
        LOG.info("gemma4_unified=%s (custom chat template %s)",
                 self.is_g4u, "loaded" if self.g4u_template else "absent")

    # -- prompt construction -------------------------------------------------

    def _template_kwargs(self) -> Dict[str, Any]:
        return {"chat_template": self.g4u_template} if self.is_g4u else {}

    def _render_pair_prompt_ids(self, system: str, user_text: str) -> List[int]:
        image_block = {"type": "image"} if self.is_g4u else {"type": "image_pil"}
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                dict(image_block), dict(image_block),
                {"type": "text", "text": user_text},
            ]},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **self._template_kwargs())
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _render_text_prompt_ids(self, user_text: str) -> List[int]:
        messages = [{"role": "user", "content": user_text}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **self._template_kwargs())
        return self.tokenizer.encode(text, add_special_tokens=False)

    # -- task path ------------------------------------------------------------

    def classify(self, rows: Sequence[Mapping[str, Any]], system: str,
                 user_text: str) -> List[str]:
        """Return one raw answer string per row (two images each)."""
        from PIL import Image
        from vllm import TokensPrompt
        from dagspaces.common.vllm_inference import _build_sampling_params

        prompt_ids = self._render_pair_prompt_ids(system, user_text)
        sp = _build_sampling_params({
            "temperature": 0.0,
            "max_tokens": 128,
            "guided_decoding": {"json": ANSWER_SCHEMA},
        })

        prompts = []
        for row in rows:
            images = [
                Image.open(str(row["presented_left_path"])).convert("RGB"),
                Image.open(str(row["presented_right_path"])).convert("RGB"),
            ]
            prompts.append(TokensPrompt(
                prompt_token_ids=list(prompt_ids),
                multi_modal_data={"image": images},
            ))

        outputs = self.llm.generate(prompts, sp, use_tqdm=False)
        self.n_task_calls += len(rows)
        return [o.outputs[0].text if o.outputs else "" for o in outputs]

    def describe(self, rows: Sequence[Mapping[str, Any]],
                 style: str = "independent") -> List[str]:
        """Task-neutral two-image captions (one text per row), no guided
        decoding. Used by the caption-mediated reflection variants."""
        from PIL import Image
        from vllm import TokensPrompt
        from dagspaces.common.vllm_inference import _build_sampling_params

        prompt_ids = self._render_pair_prompt_ids(
            DESCRIBE_SYSTEM, DESCRIBE_STYLES[style])
        sp = _build_sampling_params({"temperature": 0.0, "max_tokens": 200})
        prompts = []
        for row in rows:
            images = [
                Image.open(str(row["presented_left_path"])).convert("RGB"),
                Image.open(str(row["presented_right_path"])).convert("RGB"),
            ]
            prompts.append(TokensPrompt(
                prompt_token_ids=list(prompt_ids),
                multi_modal_data={"image": images},
            ))
        outputs = self.llm.generate(prompts, sp, use_tqdm=False)
        self.n_task_calls += len(rows)
        return [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]

    def _render_multimodal_reflection_ids(self, text: str,
                                          n_images: int) -> List[int]:
        """Reflection prompt token ids with ``n_images`` image placeholders.

        Text first, then the image blocks — matching the order gepa hands the
        pixels (an OpenAI ``content`` list of ``[text, image, image, ...]``).
        The text already contains the ``[IMAGE-N]`` markers that tell the
        reflector which photograph is which."""
        image_block = {"type": "image"} if self.is_g4u else {"type": "image_pil"}
        content: List[Any] = [{"type": "text", "text": text}]
        content += [dict(image_block) for _ in range(n_images)]
        messages = [{"role": "user", "content": content}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **self._template_kwargs())
        return self.tokenizer.encode(rendered, add_special_tokens=False)

    # -- reflection path (GEPA reflection_lm: str | list[dict] -> str) --------

    def reflect(self, prompt) -> str:
        """Text reflection (``str``) or multimodal reflection (gepa passes a
        messages ``list`` when reflective records embed ``gepa.Image``s)."""
        from dagspaces.common.vllm_inference import _build_sampling_params
        from vllm import TokensPrompt

        images = None
        if isinstance(prompt, str):
            ids = self._render_text_prompt_ids(prompt)
        else:
            text, images = _parse_multimodal_reflection(prompt)
            ids = self._render_multimodal_reflection_ids(text, len(images))
        sp = _build_sampling_params({
            "temperature": self.reflection_temperature,
            "max_tokens": self.reflection_max_tokens,
        })
        tp_kwargs: Dict[str, Any] = {"prompt_token_ids": ids}
        if images:
            tp_kwargs["multi_modal_data"] = {"image": images}
            LOG.info("[reflect] multimodal call #%d: %d images, %d prompt tokens",
                     self.n_reflection_calls + 1, len(images), len(ids))
        out = self.llm.generate([TokensPrompt(**tp_kwargs)], sp, use_tqdm=False)
        self.n_reflection_calls += 1
        return out[0].outputs[0].text if out and out[0].outputs else ""


# ---------------------------------------------------------------------------
# GEPA adapter
# ---------------------------------------------------------------------------

def _ordinal_score(pred: str, expected: str) -> float:
    """1 at exact match, decreasing linearly with ordinal distance."""
    p, e = ORD_IDX.get(pred), ORD_IDX.get(expected)
    if p is None or e is None:
        return 0.0
    return 1.0 - abs(p - e) / (len(ORDINAL) - 1)


class PairwiseGEPAAdapter(GEPAAdapter):
    """Mirrors GEPAVQAAdapter's structure for the two-image ordinal task.

    ``feedback_hint`` (optional) is extra task context appended to each
    reflective record. Leave it EMPTY for the generic-seed prompt-recovery
    experiment — any task description here leaks the answer to the
    reflection model.

    ``candidate_mode``: 'prompt' evolves {system_prompt, user_prompt}
    free-form; 'axis' freezes the AXIS_SYSTEM/AXIS_USER_TEMPLATE scaffold and
    evolves only the attribute phrase {axis}.
    """

    def render_candidate(self, candidate: Mapping[str, str]) -> tuple:
        """(system, user_text) actually sent to the task model."""
        if self._candidate_mode == "axis":
            axis = _sanitize_axis(candidate.get("axis", DEFAULT_SEED_AXIS))
            return AXIS_SYSTEM, (
                f"{AXIS_USER_TEMPLATE.format(axis=axis)}\n\n{FIXED_SUFFIX}")
        system = str(candidate.get("system_prompt", ""))
        return system, f"{candidate.get('user_prompt', '')}\n\n{FIXED_SUFFIX}"

    def __init__(self, engine: PairwiseTaskEngine, *,
                 feedback_hint: str = "",
                 reflection_descriptions: bool = False,
                 caption_style: str = "independent",
                 candidate_mode: str = "prompt",
                 reflection_images: bool = False,
                 description_cache_path: Optional[Path] = None) -> None:
        if caption_style not in DESCRIBE_STYLES:
            raise ValueError(f"unknown caption_style: {caption_style}")
        if candidate_mode not in ("prompt", "axis"):
            raise ValueError(f"unknown candidate_mode: {candidate_mode}")
        if reflection_images and reflection_descriptions:
            raise ValueError("reflection_images and reflection_descriptions are "
                             "mutually exclusive channels")
        self._engine = engine
        self._feedback_hint = feedback_hint.strip()
        self._reflection_descriptions = reflection_descriptions
        self._reflection_images = reflection_images
        self._caption_style = caption_style
        self._candidate_mode = candidate_mode
        self._descriptions: Dict[str, str] = {}
        self._description_cache_path = description_cache_path

    def _descriptions_for(self, traces: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
        """Lazily caption pairs as they first appear in a reflective minibatch
        (cheaper than precomputing the whole train set; cached by sample_id)."""
        missing = [t for t in traces
                   if t.get("sample_id") not in self._descriptions
                   and t.get("left") and t.get("right")]
        if missing:
            rows = [{"presented_left_path": t["left"],
                     "presented_right_path": t["right"]} for t in missing]
            texts = self._engine.describe(rows, style=self._caption_style)
            for t, text in zip(missing, texts):
                self._descriptions[str(t["sample_id"])] = text
            if self._description_cache_path is not None:
                with self._description_cache_path.open("a", encoding="utf-8") as fh:
                    for t, text in zip(missing, texts):
                        fh.write(json.dumps({"sample_id": t["sample_id"],
                                             "description": text}) + "\n")
        return self._descriptions

    def evaluate(self, batch: Sequence[Mapping[str, Any]],
                 candidate: Mapping[str, str], *,
                 capture_traces: bool = True):
        rows = list(batch)
        if not rows:
            return EvaluationBatch(outputs=[], scores=[], trajectories=[])

        system, user_text = self.render_candidate(candidate)
        raw = self._engine.classify(rows, system, user_text)

        scores: List[float] = []
        traces: List[Dict[str, Any]] = []
        for row, answer in zip(rows, raw):
            pred = _canonicalize_label(answer)
            expected = str(row["expected_answer"])
            score = _ordinal_score(pred, expected)
            scores.append(score)
            traces.append({
                "sample_id": row.get("sample_id"),
                "answer": pred,
                "raw_answer": answer,
                "expected_answer": expected,
                "score": score,
                "exact": float(pred == expected),
                "left": row.get("presented_left_path"),
                "right": row.get("presented_right_path"),
            })
        if len(rows) >= 100:
            exact = sum(t["exact"] for t in traces) / len(traces)
            LOG.info("[val-eval] n=%d ordinal=%.4f exact=%.4f (task_calls=%d)",
                     len(rows), sum(scores) / len(scores), exact,
                     self._engine.n_task_calls)
        return EvaluationBatch(
            outputs=traces, scores=scores,
            trajectories=traces if capture_traces else None)

    def make_reflective_dataset(self, candidate: Mapping[str, str],
                                eval_batch, components_to_update: List[str]):
        dataset: Dict[str, List[Mapping[str, Any]]] = {}
        if not eval_batch.trajectories:
            return {}
        descriptions: Dict[str, str] = {}
        if self._reflection_descriptions:
            descriptions = self._descriptions_for(eval_batch.trajectories)
        gepa_image = None
        if self._reflection_images:
            from gepa import Image as gepa_image  # noqa: N813 (lazy: gepa only on GPU node)
        for component in components_to_update:
            records = []
            for trace in eval_batch.trajectories:
                pred, expected = trace["answer"], trace["expected_answer"]
                if pred == expected:
                    feedback = f"Correct: the model answered {pred}."
                else:
                    dist = abs(ORD_IDX.get(pred, 2) - ORD_IDX.get(expected, 2))
                    feedback = (
                        f"Expected {expected} but the model answered {pred} "
                        f"(ordinal distance {dist} on the MuchLess..MuchMore "
                        f"scale).")
                if self._feedback_hint:
                    feedback = f"{feedback} {self._feedback_hint}"
                inputs: Dict[str, Any] = {}
                if self._candidate_mode == "axis":
                    inputs["component_role"] = AXIS_COMPONENT_NOTE
                if self._reflection_images:
                    # Attach the ACTUAL pair the assistant saw. gepa replaces
                    # each Image with an [IMAGE-N] placeholder in the rendered
                    # text and appends the pixels inline for the reflection VLM.
                    left, right = trace.get("left"), trace.get("right")
                    if gepa_image is not None and left and right:
                        inputs["note"] = MULTIMODAL_INPUT_NOTE
                        inputs["Image A"] = gepa_image(path=str(left))
                        inputs["Image B"] = gepa_image(path=str(right))
                elif self._reflection_descriptions:
                    # GEPA's reflection template shows the current instruction
                    # once (<curr_param>), so the candidate is NOT repeated per
                    # record; Inputs carries the model's own captions instead —
                    # the only task-semantics channel in these variants. The
                    # note wards off the modality confusion seen in the first
                    # caption sweep (prompts about comparing "descriptions").
                    inputs["note"] = REFLECTION_INPUT_NOTE
                    inputs["image_descriptions"] = descriptions.get(
                        str(trace.get("sample_id")), "(unavailable)")
                elif self._candidate_mode == "prompt":
                    inputs["user_prompt"] = str(candidate.get("user_prompt", ""))
                records.append({
                    "Inputs": inputs,
                    "Generated Outputs": pred,
                    "Feedback": f"{feedback} Score: {trace['score']:.2f}",
                    "score": trace["score"],
                })
            dataset[component] = records
        return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_seed_candidate(prompt_yaml: Path) -> Dict[str, str]:
    data = yaml.safe_load(prompt_yaml.read_text())
    return {
        "system_prompt": str(data.get("system", "")).strip(),
        "user_prompt": str(data.get("user_template", "")).strip(),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-parquet", type=Path, default=DEFAULT_RESULT_PARQUET,
                    help="production pairwise result parquet (supervision source)")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--seed-prompt-yaml", type=Path, default=DEFAULT_SEED_PROMPT_YAML)
    ap.add_argument("--seed-mode", choices=["yaml", "generic"], default="yaml",
                    help="'generic' seeds from GENERIC_SEED ('Compare the two "
                         "images...') instead of the task prompt YAML")
    ap.add_argument("--feedback-hint", default="",
                    help="extra task context appended to reflective feedback; "
                         "leave empty for the generic-seed recovery experiment "
                         "(a hint leaks the task to the reflection model)")
    ap.add_argument("--reflection-descriptions", action="store_true",
                    help="caption-mediated reflection: include the model's own "
                         "task-neutral descriptions of each pair in the "
                         "reflective records (captions cached to "
                         "descriptions.jsonl in the outdir)")
    ap.add_argument("--reflection-images", action="store_true",
                    help="MULTIMODAL reflection: show the reflector the actual "
                         "failing image pairs (not captions of them). Mutually "
                         "exclusive with --reflection-descriptions. Use a small "
                         "--reflection-minibatch (each record adds 2 images to "
                         "one prompt); the mm-per-prompt cap is raised to "
                         "2*minibatch automatically.")
    ap.add_argument("--caption-style", choices=sorted(DESCRIBE_STYLES),
                    default="independent",
                    help="'independent' = per-image factual captions (no "
                         "comparison); 'contrastive' = difference-focused "
                         "captions (comparatives allowed, verdict forbidden)")
    ap.add_argument("--candidate-mode", choices=["prompt", "axis"],
                    default="prompt",
                    help="'axis' freezes the calibration scaffold and evolves "
                         "only the attribute phrase in 'Which photograph "
                         "looks more X?'")
    ap.add_argument("--seed-axis", default=DEFAULT_SEED_AXIS,
                    help="seed attribute phrase for --candidate-mode axis")
    ap.add_argument("--train-n", type=int, default=600)
    ap.add_argument("--val-n", type=int, default=300)
    ap.add_argument("--max-metric-calls", type=int, default=6000)
    ap.add_argument("--reflection-minibatch", type=int, default=16)
    ap.add_argument("--reflection-temperature", type=float, default=0.6)
    ap.add_argument("--reflection-max-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--outdir", type=Path,
                    default=REPO / "outputs/gepa_pairwise/subway_gemma12b")
    args = ap.parse_args()

    import gepa

    args.outdir.mkdir(parents=True, exist_ok=True)
    trainset, valset = build_frames(args.result_parquet, args.train_n,
                                    args.val_n, args.seed)
    LOG.info("dataset: %d train / %d val (from %s)",
             len(trainset), len(valset), args.result_parquet)

    if args.candidate_mode == "axis":
        seed_candidate = {"axis": args.seed_axis}
    elif args.seed_mode == "generic":
        seed_candidate = dict(GENERIC_SEED)
    else:
        seed_candidate = _load_seed_candidate(args.seed_prompt_yaml)
    LOG.info("seed candidate (%s/%s): %s", args.candidate_mode, args.seed_mode,
             seed_candidate)
    # Multimodal reflection puts 2 images per record into one reflection prompt;
    # raise the per-prompt image cap to cover the whole minibatch.
    limit_mm_images = (2 * args.reflection_minibatch
                       if args.reflection_images else 2)
    if args.reflection_images:
        LOG.info("multimodal reflection: minibatch=%d -> up to %d images/prompt",
                 args.reflection_minibatch, limit_mm_images)
    engine = PairwiseTaskEngine(
        args.model_dir,
        max_model_len=args.max_model_len,
        reflection_temperature=args.reflection_temperature,
        reflection_max_tokens=args.reflection_max_tokens,
        limit_mm_images=limit_mm_images,
    )
    adapter = PairwiseGEPAAdapter(
        engine,
        feedback_hint=args.feedback_hint,
        reflection_descriptions=args.reflection_descriptions,
        caption_style=args.caption_style,
        candidate_mode=args.candidate_mode,
        reflection_images=args.reflection_images,
        description_cache_path=(args.outdir / "descriptions.jsonl"
                                if args.reflection_descriptions else None),
    )

    # Axis mode needs a custom reflection template (the default "write a new
    # instruction" yields meta-prompts, not phrases); its multimodal variant
    # refers the reflector to the photographs instead of an observer's caption.
    if args.candidate_mode == "axis":
        reflection_template = (MULTIMODAL_AXIS_REFLECTION_TEMPLATE
                               if args.reflection_images
                               else AXIS_REFLECTION_TEMPLATE)
    else:
        reflection_template = None

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=engine.reflect,
        reflection_prompt_template=reflection_template,
        reflection_minibatch_size=args.reflection_minibatch,
        max_metric_calls=args.max_metric_calls,
        run_dir=str(args.outdir / "gepa_state"),
        seed=args.seed,
        track_best_outputs=True,
        display_progress_bar=False,
        use_wandb=False,
    )

    best_path = args.outdir / "best_prompts.yaml"
    with best_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(result.best_candidate, fh, sort_keys=False,
                       allow_unicode=True, width=88)

    # NB: the per-candidate "[val-eval]" log lines are emitted for EVERY full-val
    # evaluation, so the LAST one is the last candidate tried, NOT the returned
    # best. The score of the returned candidate is max(val_aggregate_scores)
    # (task evals are temperature 0, hence deterministic). Record it explicitly
    # so downstream analysis cannot misread the log.
    _val_scores = getattr(result, "val_aggregate_scores", None) or []
    summary = {
        "best_score": getattr(result, "best_score", None),
        "best_val_ordinal": (max(_val_scores) if _val_scores else None),
        "val_aggregate_scores": getattr(result, "val_aggregate_scores", None),
        "train_n": len(trainset),
        "val_n": len(valset),
        "max_metric_calls": args.max_metric_calls,
        "task_calls": engine.n_task_calls,
        "reflection_calls": engine.n_reflection_calls,
        "model_dir": args.model_dir,
        "result_parquet": str(args.result_parquet),
        "seed_mode": args.seed_mode,
        "seed_candidate": seed_candidate,
        "seed_prompt_yaml": str(args.seed_prompt_yaml),
        "feedback_hint": args.feedback_hint,
        "reflection_descriptions": args.reflection_descriptions,
        "reflection_images": args.reflection_images,
        "reflection_minibatch": args.reflection_minibatch,
        "caption_style": args.caption_style,
        "candidate_mode": args.candidate_mode,
        "n_descriptions": len(adapter._descriptions),
    }
    if args.candidate_mode == "axis":
        best_sys, best_user = adapter.render_candidate(result.best_candidate)
        summary["best_axis"] = _sanitize_axis(
            result.best_candidate.get("axis", ""))
        summary["rendered_best_system"] = best_sys
        summary["rendered_best_user"] = best_user
    (args.outdir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    LOG.info("GEPA done. best_score=%s; prompts -> %s", summary["best_score"], best_path)

    # vLLM 0.23 leaves non-daemon worker threads alive after the engine is
    # idle, which keeps the SLURM job running until its time limit (job
    # 832687 idled 7h post-completion). All artifacts are flushed above, so a
    # hard exit is safe — same rationale as the urbanvqa runner's persistent
    # processor teardown.
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
