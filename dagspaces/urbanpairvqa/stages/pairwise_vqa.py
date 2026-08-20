from __future__ import annotations

from typing import Any, Dict, Optional
import hashlib
import os

import pandas as pd
from omegaconf import DictConfig, OmegaConf

_ORDINAL_LABELS = ("MuchLess", "Less", "Same", "More", "MuchMore")
_ORDINAL_SCORE = {
    "MuchLess": -2,
    "Less": -1,
    "Same": 0,
    "More": 1,
    "MuchMore": 2,
}
_INVERT_LABEL = {
    "MuchLess": "MuchMore",
    "Less": "More",
    "Same": "Same",
    "More": "Less",
    "MuchMore": "MuchLess",
    # Abstention is symmetric — swapping A/B doesn't resolve the uncertainty.
    "NotSure": "NotSure",
}

# Canonical token for the optional abstention label (gated by
# prompt.structured_output.allow_not_sure). Deliberately ABSENT from
# _ORDINAL_SCORE / _ORDINAL_LABELS so it never gets folded into the ordinal
# scale — see _score_labels.
_NOT_SURE_LABEL = "NotSure"

# Supported user-message content layouts (prompt.image_layout, optional).
# "images_then_text" is the production default: image A, image B, then the
# full text prompt. The alternates exist for presentation-robustness probes:
#   interleaved_labels — "Image A:" text, image A, "Image B:" text, image B,
#                        prompt (textual anchors adjacent to each image)
#   text_first         — prompt, image A, image B
_IMAGE_LAYOUTS = ("images_then_text", "interleaved_labels", "text_first")


def _canonicalize_label(value: Any, not_sure_label: str = _NOT_SURE_LABEL) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Same"
    norm = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
    # The configured abstention label (exact or substring) maps to the canonical
    # "NotSure" token regardless of how it was spelled in the schema/prompt.
    ns_norm = str(not_sure_label or "").lower().replace("_", "").replace("-", "").replace(" ", "")
    if ns_norm and (norm == ns_norm or ns_norm in norm):
        return _NOT_SURE_LABEL
    aliases = {
        "muchless": "MuchLess",
        "less": "Less",
        "same": "Same",
        "equal": "Same",
        "more": "More",
        "muchmore": "MuchMore",
        "yes": "More",
        "no": "Less",
        "true": "More",
        "false": "Less",
        "notsure": _NOT_SURE_LABEL,
        "unsure": _NOT_SURE_LABEL,
        "cannottell": _NOT_SURE_LABEL,
        "canttell": _NOT_SURE_LABEL,
        "unclear": _NOT_SURE_LABEL,
        "unknown": _NOT_SURE_LABEL,
        "idk": _NOT_SURE_LABEL,
    }
    if norm in aliases:
        return aliases[norm]
    # Abstention can also arrive JSON-wrapped or as a phrase ("I am not sure").
    # Detect it before the ordinal substring fallback, which would otherwise
    # mislabel "not sure" as "Same" (it contains no ordinal substring).
    if "notsure" in norm or "unsure" in norm or "cannottell" in norm:
        return _NOT_SURE_LABEL
    # Substring fallback (e.g. a JSON-wrapped answer like `{"answer": "MuchMore"}`
    # that the alias dict missed). Check the compound labels before their
    # substrings so "MuchMore"/"MuchLess" aren't shadowed by "More"/"Less".
    for label in ("MuchLess", "MuchMore", "Less", "More", "Same"):
        if label.lower() in raw.lower():
            return label
    return "Same"


def _deterministic_debug_label(pair_id: str, seed: int, labels=_ORDINAL_LABELS) -> str:
    digest = hashlib.sha256(f"{pair_id}|{seed}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(labels)
    return labels[idx]


def _not_sure_enabled(cfg: DictConfig) -> bool:
    """Whether the abstention label is added to the structured-output options.

    Off by default — toggle via ``prompt.structured_output.allow_not_sure=true``
    on any run or sweep.
    """
    sc = getattr(getattr(cfg, "prompt", {}), "structured_output", None)
    return bool(getattr(sc, "allow_not_sure", False)) if sc is not None else False


def _not_sure_label(cfg: DictConfig) -> str:
    """The label string the model emits when abstaining (default ``NotSure``)."""
    sc = getattr(getattr(cfg, "prompt", {}), "structured_output", None)
    label = getattr(sc, "not_sure_label", None) if sc is not None else None
    return str(label) if label else _NOT_SURE_LABEL


def _not_sure_guidance(cfg: DictConfig, label: str) -> str:
    """Prompt text appended when the abstention label is enabled.

    Deliberately domain-neutral — this string is shared by every case in the
    consolidated ranking battery, so it must not smuggle in a judgment word
    (the pre-2026-08-11 wording said "equally appealing", which reads as a hint
    on the restaurant / street-photography cases and as a non-sequitur on the
    road-quality / library ones). Override per prompt with
    ``prompt.structured_output.not_sure_text`` if a case truly needs it.
    """
    sc = getattr(getattr(cfg, "prompt", {}), "structured_output", None)
    custom = getattr(sc, "not_sure_text", None) if sc is not None else None
    if custom:
        return str(custom)
    return (
        f"If the two images do not give you enough information to make this "
        f"comparison, you may answer \"{label}\". Use \"{label}\" only for true "
        f"uncertainty — not when the two look about the same (use \"Same\" for "
        f"that)."
    )


def _augment_schema_with_not_sure(schema: Dict[str, Any], label: str) -> Dict[str, Any]:
    """Append ``label`` to the guided-decoding answer enum (in place + return).

    No-op with a warning if the schema lacks ``properties.answer.enum`` — the
    abstention toggle only makes sense for the enum-constrained ordinal schema.
    """
    try:
        answer = schema["properties"]["answer"]
        enum = answer.get("enum")
        if isinstance(enum, list) and label not in enum:
            answer["enum"] = [*enum, label]
    except (KeyError, TypeError, AttributeError):
        print(
            "[pairwise_vqa] WARN: allow_not_sure set but schema has no "
            "properties.answer.enum to extend; leaving schema unchanged.",
            flush=True,
        )
    return schema


def _resolve_system_prompt(cfg: DictConfig) -> Optional[str]:
    """Return the system-turn text, or ``None`` to omit the system turn entirely.

    ``prompt.system: null`` (or an empty / whitespace-only string, or a missing
    key) means **no system message is sent at all** — the consolidated ranking
    battery's default, so no role framing constrains the output distribution.

    Three conditions that are easy to confuse, verified against the Qwen3.5 and
    Gemma-4 chat templates (2026-08-11):

    | ``prompt.system``   | rendered                                    |
    |---------------------|---------------------------------------------|
    | ``"You are ..."``   | normal system turn                          |
    | ``""``              | a vestigial EMPTY system turn — not the same |
    |                     | as none, and a token pattern the model has   |
    |                     | barely seen in training. Never do this.      |
    | ``null`` / absent   | no system turn at all                        |

    Empty-string is normalized to ``None`` here precisely so nobody trips that
    middle row. Neither model family's template injects a default persona when
    the system turn is absent, so this really is a no-persona condition — it is
    not silently replaced by a template default.

    Before 2026-08-11 a missing key fell back to a hardcoded
    "You are a helpful assistant.", which meant persona removal by key deletion
    silently substituted a *different* persona. That fallback is gone.
    """
    raw = getattr(getattr(cfg, "prompt", {}), "system", None)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _resolve_image_layout(cfg: DictConfig) -> str:
    """Validate and return ``prompt.image_layout`` (default production layout).

    Called at stage entry too so skip_inference dry runs fail fast on typos
    instead of only erroring once a GPU job reaches preprocessing.
    """
    layout = str(
        getattr(getattr(cfg, "prompt", {}), "image_layout", None) or "images_then_text"
    )
    if layout not in _IMAGE_LAYOUTS:
        raise ValueError(
            f"Unknown prompt.image_layout: {layout!r} "
            f"(expected one of {list(_IMAGE_LAYOUTS)})"
        )
    return layout


def _render_pair_prompt(row: Dict[str, Any], cfg: DictConfig) -> str:
    template = str(
        getattr(
            getattr(cfg, "prompt", {}),
            "user_template",
            "Compare the first image (Image A) and the second image (Image B). "
            "Return one label: MuchLess, Less, Same, More, MuchMore.",
        )
    )
    pair_id = str(row.get("pair_id", "unknown_pair"))
    extra = ""
    if _not_sure_enabled(cfg):
        extra = "\n\n" + _not_sure_guidance(cfg, _not_sure_label(cfg))
    return (
        f"{template}{extra}\n\n"
        f"Pair ID: {pair_id}\n"
        "Interpret labels as Image A relative to Image B."
    )


def _invert_label(label: str) -> str:
    return _INVERT_LABEL.get(label, "Same")


def _score_labels(labels: pd.Series) -> pd.Series:
    """Map ordinal labels to integer scores.

    The abstention label ("NotSure") is intentionally absent from
    ``_ORDINAL_SCORE`` and therefore becomes ``NaN`` — an abstention is not a
    0/"Same" judgment and must not be folded into the ordinal scale.  When no
    abstentions are present the column stays ``int64`` (identical to legacy,
    abstention-free runs); when they are, it becomes ``float64`` with ``NaN``
    marking the abstained rows so downstream ``to_numeric(...).dropna()`` paths
    naturally exclude them.
    """
    scores = labels.map(_ORDINAL_SCORE)
    if scores.isna().any():
        return scores.astype("float64")
    return scores.astype(int)


def _derive_labels(df: pd.DataFrame, not_sure_label: str = _NOT_SURE_LABEL) -> pd.DataFrame:
    df = df.copy()
    if "answer" not in df.columns:
        df["answer"] = ""
    df["presented_answer"] = df["answer"]
    df["presented_label"] = df["presented_answer"].apply(
        lambda v: _canonicalize_label(v, not_sure_label)
    )
    df["presented_score"] = _score_labels(df["presented_label"])

    swapped = df.get("is_swapped", False)
    if isinstance(swapped, pd.Series):
        is_swapped = swapped.fillna(False).astype(bool)
    else:
        is_swapped = pd.Series([bool(swapped)] * len(df), index=df.index)

    df["relative_label"] = df["presented_label"]
    df.loc[is_swapped, "relative_label"] = df.loc[is_swapped, "presented_label"].apply(_invert_label)
    df["relative_score"] = _score_labels(df["relative_label"])
    return df


def _make_pairwise_preprocess(cfg: DictConfig):
    """Build a preprocess callback that passes two separate images.

    Each row carries paths to two images. The preprocess loads both as PIL
    and places them as separate image content blocks in the message — Image A
    (first) and Image B (second). The VLM sees both at full resolution.

    Counterbalancing is already handled by the pair sampler:
    ``presented_left_path`` and ``presented_right_path`` reflect the
    (possibly swapped) presentation order.

    ``prompt.image_layout`` (optional) selects the content-block order — see
    ``_IMAGE_LAYOUTS``. Absent means ``images_then_text`` (production layout).
    """
    system_prompt = _resolve_system_prompt(cfg)
    image_layout = _resolve_image_layout(cfg)
    sp_cfg = dict(getattr(cfg, "sampling_params_vqa", {}) or {})
    stop_val = sp_cfg.get("stop")
    if stop_val is None:
        sp_cfg["stop"] = []
    elif not isinstance(stop_val, list):
        sp_cfg["stop"] = [str(stop_val)]

    structured_cfg = getattr(getattr(cfg, "prompt", {}), "structured_output", None)
    if structured_cfg and getattr(structured_cfg, "enabled", False):
        schema = OmegaConf.to_container(structured_cfg.json_schema, resolve=True)
        if _not_sure_enabled(cfg):
            schema = _augment_schema_with_not_sure(schema, _not_sure_label(cfg))
        sp_cfg["guided_decoding"] = {"json": schema}

    def _preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        left_path = str(row.get("presented_left_path", "")).strip()
        right_path = str(row.get("presented_right_path", "")).strip()
        prompt = str(row.get("prompt", ""))

        # Pass images as file:// URLs — vLLM loads them lazily during its
        # internal rendering pipeline, so we don't hold all PIL images in
        # memory at once.
        left = {"type": "image_url", "image_url": {"url": f"file://{left_path}"}}
        right = {"type": "image_url", "image_url": {"url": f"file://{right_path}"}}
        text = {"type": "text", "text": prompt}
        if image_layout == "interleaved_labels":
            content = [
                {"type": "text", "text": "Image A:"},
                left,
                {"type": "text", "text": "Image B:"},
                right,
                text,
            ]
        elif image_layout == "text_first":
            content = [text, left, right]
        else:
            content = [left, right, text]
        messages = [{"role": "user", "content": content}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})

        result = dict(row)
        result["messages"] = messages
        result["sampling_params"] = dict(sp_cfg)
        result["sample_id"] = str(row.get("pair_id", ""))
        return result

    return _preprocess


def _make_pairwise_postprocess():
    """Build a postprocess callback that extracts the answer."""
    def _postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key in ["pair_id", "sample_id", "canonical_pair_id", "repeat_idx",
                     "sample_id_a", "sample_id_b", "image_path_a", "image_path_b",
                     "presented_left_path", "presented_right_path",
                     "presented_order", "is_swapped"]:
            if key in row:
                result[key] = row[key]

        result["answer"] = str(row.get("generated_text", "")).strip()
        result["model_response"] = row.get("generated_text", "")
        result["model_reasoning"] = row.get("generated_reasoning", "")
        return result

    return _postprocess


def run_pairwise_vqa_stage(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Execute pairwise relative comparison over two-image tuples.

    Each pair is presented as two separate images to the VLM (Image A first,
    Image B second) at full resolution.  Counterbalancing (which image is A
    vs B) is handled upstream by the pair sampler.  The full-pipeline DP
    workers handle image loading and inference in parallel.
    """
    from dagspaces.common.vllm_inference import run_vllm_inference

    if df is None or df.empty:
        return pd.DataFrame(columns=["pair_id", "relative_label", "relative_score", "answer"])

    required = {"pair_id"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Pairwise stage missing required columns: {missing}")

    runtime_cfg = getattr(cfg, "runtime", {})
    skip_inference = bool(getattr(runtime_cfg, "skip_inference", False))
    pair_seed = int(getattr(getattr(cfg, "pair_sampler", {}), "pair_seed", 777))

    not_sure_label = _not_sure_label(cfg)
    _resolve_image_layout(cfg)  # fail fast on typos, incl. skip_inference runs
    if skip_inference:
        out = df.copy()
        debug_labels = _ORDINAL_LABELS
        if _not_sure_enabled(cfg):
            # Exercise the abstention plumbing on dry runs too.
            debug_labels = (*_ORDINAL_LABELS, not_sure_label)
        out["answer"] = out["pair_id"].apply(
            lambda x: _deterministic_debug_label(str(x), pair_seed, debug_labels)
        )
        return _derive_labels(out, not_sure_label)

    local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.update(local_cfg, "model.engine_kwargs.limit_mm_per_prompt.image", 2, merge=True)

    # Prepare lightweight DataFrame: render prompts, set sample_id (no image loading)
    print(f"[pairwise_vqa] Preparing {len(df)} pairs...", flush=True)
    df = df.copy()
    df["sample_id"] = df["pair_id"].astype(str)
    df["prompt"] = df.apply(lambda row: _render_pair_prompt(row.to_dict(), cfg), axis=1)

    # Single inference call — DP workers load images and run inference in parallel
    preprocess = _make_pairwise_preprocess(cfg)
    postprocess = _make_pairwise_postprocess()

    inferred = run_vllm_inference(
        df=df,
        cfg=local_cfg,
        preprocess=preprocess,
        postprocess=postprocess,
        stage_name="urbanpairvqa_pairwise",
    )

    return _derive_labels(inferred, not_sure_label)
