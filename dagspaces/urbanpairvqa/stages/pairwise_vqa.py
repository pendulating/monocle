from __future__ import annotations

from typing import Any, Dict
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
}


def _canonicalize_label(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Same"
    norm = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
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
    }
    if norm in aliases:
        return aliases[norm]
    for label in _ORDINAL_LABELS:
        if label.lower() in raw.lower():
            return label
    return "Same"


def _deterministic_debug_label(pair_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{pair_id}|{seed}".encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(_ORDINAL_LABELS)
    return _ORDINAL_LABELS[idx]


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
    return (
        f"{template}\n\n"
        f"Pair ID: {pair_id}\n"
        "Interpret labels as Image A relative to Image B."
    )


def _invert_label(label: str) -> str:
    return _INVERT_LABEL.get(label, "Same")


def _derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "answer" not in df.columns:
        df["answer"] = ""
    df["presented_answer"] = df["answer"]
    df["presented_label"] = df["presented_answer"].apply(_canonicalize_label)
    df["presented_score"] = df["presented_label"].map(_ORDINAL_SCORE).fillna(0).astype(int)

    swapped = df.get("is_swapped", False)
    if isinstance(swapped, pd.Series):
        is_swapped = swapped.fillna(False).astype(bool)
    else:
        is_swapped = pd.Series([bool(swapped)] * len(df), index=df.index)

    df["relative_label"] = df["presented_label"]
    df.loc[is_swapped, "relative_label"] = df.loc[is_swapped, "presented_label"].apply(_invert_label)
    df["relative_score"] = df["relative_label"].map(_ORDINAL_SCORE).fillna(0).astype(int)
    return df


def _make_pairwise_preprocess(cfg: DictConfig):
    """Build a preprocess callback that passes two separate images.

    Each row carries paths to two images. The preprocess loads both as PIL
    and places them as separate image content blocks in the message — Image A
    (first) and Image B (second). The VLM sees both at full resolution.

    Counterbalancing is already handled by the pair sampler:
    ``presented_left_path`` and ``presented_right_path`` reflect the
    (possibly swapped) presentation order.
    """
    system_prompt = str(getattr(getattr(cfg, "prompt", {}), "system", "You are a helpful assistant."))
    sp_cfg = dict(getattr(cfg, "sampling_params_vqa", {}) or {})
    stop_val = sp_cfg.get("stop")
    if stop_val is None:
        sp_cfg["stop"] = []
    elif not isinstance(stop_val, list):
        sp_cfg["stop"] = [str(stop_val)]

    structured_cfg = getattr(getattr(cfg, "prompt", {}), "structured_output", None)
    if structured_cfg and getattr(structured_cfg, "enabled", False):
        schema = OmegaConf.to_container(structured_cfg.json_schema, resolve=True)
        sp_cfg["guided_decoding"] = {"json": schema}

    def _preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        left_path = str(row.get("presented_left_path", "")).strip()
        right_path = str(row.get("presented_right_path", "")).strip()
        prompt = str(row.get("prompt", ""))

        # Pass images as file:// URLs — vLLM loads them lazily during its
        # internal rendering pipeline, so we don't hold all PIL images in
        # memory at once.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"file://{left_path}"}},
                {"type": "image_url", "image_url": {"url": f"file://{right_path}"}},
                {"type": "text", "text": prompt},
            ]},
        ]

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

    if skip_inference:
        out = df.copy()
        out["answer"] = out["pair_id"].apply(lambda x: _deterministic_debug_label(str(x), pair_seed))
        return _derive_labels(out)

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

    return _derive_labels(inferred)
