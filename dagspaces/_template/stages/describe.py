"""The describe stage: ask a vision-language model one question about each image.

Input rows: `sample_id`, `image_path`, and the columns that the prompt uses.
Output rows: the input columns, `answer` and `generated_text`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, cast

import pandas as pd
from jinja2 import Template
from omegaconf import DictConfig, OmegaConf

Row = Dict[str, Any]


def make_preprocess(cfg: DictConfig) -> Callable[[Row], Row]:
    """Return a function that gives each row its chat messages and sampling params."""
    user_template = Template(str(cfg.prompt.user_template))
    system = str(OmegaConf.select(cfg, "prompt.system", default="") or "")
    sampling = cast(Dict[str, Any], OmegaConf.to_container(cfg.sampling_params, resolve=True))

    def preprocess(row: Row) -> Row:
        row = dict(row)
        content: list = []
        if row.get("image_path"):
            # A path, not pixels: run_vllm_inference opens the images batch by batch.
            content.append({"type": "image", "image": str(row["image_path"])})
        content.append({"type": "text", "text": user_template.render(**row)})
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": content})
        row["messages"] = messages
        row["sampling_params"] = dict(sampling)
        return row

    return preprocess


def postprocess(row: Row) -> Row:
    """Keep the plain input columns and add the answer."""
    out = {
        k: v
        for k, v in row.items()
        if k not in ("messages", "sampling_params") and isinstance(v, (str, int, float, bool, type(None)))
    }
    out["answer"] = str(row.get("generated_text") or "").strip()
    return out


def run_describe_stage(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run the model on each row of `df`."""
    from dagspaces.common.vllm_inference import run_vllm_inference

    return run_vllm_inference(
        df, cfg, preprocess=make_preprocess(cfg), postprocess=postprocess, stage_name="describe"
    )
