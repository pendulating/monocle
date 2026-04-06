from __future__ import annotations

from typing import Any, Dict, Optional
import hashlib
import os

import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import DictConfig, OmegaConf

from dagspaces.urbanvqa.stages.vqa import run_vqa_stage

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
        # Support binary prompts like "is A more wealthy than B?"
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


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _stitch_pair(path_a: str, path_b: str, max_height: int) -> np.ndarray:
    img_a = _load_rgb(path_a)
    img_b = _load_rgb(path_b)
    scale_a = max_height / float(max(1, img_a.height))
    scale_b = max_height / float(max(1, img_b.height))
    new_a = (max(1, int(img_a.width * scale_a)), max_height)
    new_b = (max(1, int(img_b.width * scale_b)), max_height)
    img_a = img_a.resize(new_a)
    img_b = img_b.resize(new_b)

    combined = Image.new("RGB", (img_a.width + img_b.width, max_height), color=(255, 255, 255))
    combined.paste(img_a, (0, 0))
    combined.paste(img_b, (img_a.width, 0))
    return np.asarray(combined)


def _render_pair_prompt(row: Dict[str, Any], cfg: DictConfig) -> str:
    template = str(
        getattr(
            getattr(cfg, "prompt", {}),
            "user_template",
            "Compare image A (left) vs image B (right). Return one label: MuchLess, Less, Same, More, MuchMore.",
        )
    )
    pair_id = str(row.get("pair_id", "unknown_pair"))
    return (
        f"{template}\n\n"
        f"Pair ID: {pair_id}\n"
        "Interpret labels as LEFT image relative to RIGHT image."
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


def _prepare_pairwise_batch(batch: pd.DataFrame, cfg: DictConfig, max_height: int) -> pd.DataFrame:
    rows = []
    for record in batch.to_dict(orient="records"):
        left_path = str(record.get("presented_left_path", "")).strip() or str(record.get("image_path_a", "")).strip()
        right_path = str(record.get("presented_right_path", "")).strip() or str(record.get("image_path_b", "")).strip()
        if not left_path or not right_path:
            continue
        if not os.path.exists(left_path) or not os.path.exists(right_path):
            continue
        stitched = _stitch_pair(left_path, right_path, max_height=max_height)
        row = dict(record)
        row["presented_left_path"] = left_path
        row["presented_right_path"] = right_path
        row["image"] = stitched
        row["sample_id"] = str(record.get("pair_id"))
        row["prompt"] = _render_pair_prompt(record, cfg)
        rows.append(row)
    return pd.DataFrame(rows)


def run_pairwise_vqa_stage(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Execute pairwise relative comparison over two-image tuples."""
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

    max_height = int(getattr(getattr(cfg, "pairwise", {}), "stitch_max_height", 512))
    model_cfg = OmegaConf.to_container(cfg, resolve=False)
    local_cfg = OmegaConf.create(model_cfg)
    # Stitched pair runs as one multimodal image through the existing VQA engine.
    OmegaConf.update(local_cfg, "model.engine_kwargs.limit_mm_per_prompt.image", 1, merge=True)

    inferred: Any
    try:
        import ray  # type: ignore

        # Stream stitching in Ray batches to avoid materializing all image arrays in memory.
        ds = ray.data.from_pandas(df)
        prep_batch_size = int(getattr(getattr(cfg, "pairwise", {}), "prepare_batch_size", 16) or 16)
        vqa_input = ds.map_batches(
            _prepare_pairwise_batch,
            batch_format="pandas",
            fn_kwargs={"cfg": cfg, "max_height": max_height},
            batch_size=prep_batch_size,
        )
        inferred = run_vqa_stage(vqa_input, local_cfg)
        if hasattr(inferred, "map_batches"):
            return inferred.map_batches(_derive_labels, batch_format="pandas")
    except Exception:
        rows = _prepare_pairwise_batch(df, cfg, max_height=max_height)
        if rows.empty:
            raise ValueError("No pair rows had readable presented/canonical image paths.")
        inferred = run_vqa_stage(rows, local_cfg)

    return _derive_labels(inferred)

