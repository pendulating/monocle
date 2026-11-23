"""The runner of the `readout` stage: chunks, resume, the 3 outputs.

The stage reads the input table (``dataset`` input or ``data.parquet_path``),
and reads the J-lens for each row with ``research.monocle.readout.Reader``.

Each chunk of ``readout.chunk_rows`` rows writes its files under
``<results dir>/chunks/`` (and ``store/``). The answers file of a chunk is the
last write, thus it marks the chunk done. A rerun skips a done chunk, so a job
that SLURM stops keeps everything it finished. At the end, the runner joins the
chunk files into ``results.parquet`` and ``answers.parquet``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _safe_log_table,
)
from dagspaces.common.runners.base import StageRunner
from dagspaces.common.runners.dataframe import load_stage_input

#: make_reader(cfg) -> (reader, extras). extras: {"manifest": dict, "vocab": ids or None, "tokenizer": tok}
ReaderFactory = Callable[[DictConfig], Tuple[Any, Dict[str, Any]]]


def _default_reader(cfg: DictConfig) -> Tuple[Any, Dict[str, Any]]:
    from research.monocle.readout import load_reader

    model = OmegaConf.to_container(cfg.model, resolve=True)
    readout = OmegaConf.to_container(cfg.readout, resolve=True)
    return load_reader(model, readout)  # type: ignore[arg-type]


def image_lists(df: pd.DataFrame, cfg: DictConfig) -> List[List[str]]:
    """The image paths of each row: ``image_paths`` (a list) or ``image_path``."""
    root = OmegaConf.select(cfg, "data.image_root", default=None)

    def full(p: Any) -> str:
        p = str(p)
        return p if (not root or os.path.isabs(p)) else os.path.join(str(root), p)

    if "image_paths" in df.columns:
        return [[full(p) for p in list(v)] for v in df["image_paths"]]
    if "image_path" in df.columns:
        # load_stage_input already joined data.image_root to image_path.
        return [[str(p)] for p in df["image_path"]]
    raise ValueError("the input table needs an image_path or an image_paths column")


def _atomic_parquet(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_npy(arr: np.ndarray, path: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _load_images(paths: List[str]) -> List[Any]:
    from PIL import Image

    out = []
    for p in paths:
        try:
            with Image.open(p) as im:
                out.append(im.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"the image {p} does not load: {exc}") from exc
    return out


class ReadoutRunner(StageRunner):
    """Read the J-lens at the selected positions of each (images, prompt) row."""

    stage_name = "readout"

    def __init__(self, make_reader: Optional[ReaderFactory] = None,
                 load_images: Callable[[List[str]], List[Any]] = _load_images) -> None:
        self._make_reader = make_reader or _default_reader
        self._load_images = load_images

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg
        parquet_path = context.inputs.get("dataset") or OmegaConf.select(cfg, "data.parquet_path", default=None)
        if not parquet_path:
            raise ValueError(f"node '{context.node.key}' needs the input 'dataset' or 'data.parquet_path'")
        df = load_stage_input(cfg, str(parquet_path)).reset_index(drop=True)
        # readout.prompt, when set, is the prompt of every row.
        fixed = OmegaConf.select(cfg, "readout.prompt", default=None)
        if fixed is None and "prompt" not in df.columns:
            raise ValueError("the input table needs a prompt column, or set readout.prompt")
        prompts = [str(fixed)] * len(df) if fixed is not None else [str(v) for v in df["prompt"]]
        if "sample_id" not in df.columns:
            df["sample_id"] = [str(i) for i in range(len(df))]
        images = image_lists(df, cfg)
        print(f"[readout] {len(df)} rows from {parquet_path}", flush=True)

        results_path = context.output_paths["results"]
        out_dir = os.path.dirname(os.path.abspath(results_path))
        answers_path = context.output_paths.get("answers") or os.path.join(out_dir, "answers.parquet")
        store = bool(OmegaConf.select(cfg, "readout.store", default=False))
        store_dir = context.output_paths.get("store") or os.path.join(out_dir, "store")
        chunk_dir = os.path.join(out_dir, "chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        if store:
            os.makedirs(store_dir, exist_ok=True)

        size = int(OmegaConf.select(cfg, "readout.chunk_rows", default=50))
        n_chunks = (len(df) + size - 1) // size

        def ids_of(c: int) -> List[str]:
            return [str(v) for v in df["sample_id"].iloc[c * size:(c + 1) * size]]

        def done(c: int) -> bool:
            path = os.path.join(chunk_dir, f"answers-{c:05d}.parquet")
            if not os.path.exists(path):
                return False
            have = [str(v) for v in pd.read_parquet(path, columns=["sample_id"])["sample_id"]]
            if have != ids_of(c):
                raise RuntimeError(f"{path} holds other rows than chunk {c} of this input. "
                                   "Use a new output folder, or delete the chunks.")
            return True

        todo = [c for c in range(n_chunks) if not done(c)]
        print(f"[readout] {n_chunks} chunks of {size} rows; {n_chunks - len(todo)} are done", flush=True)
        reader = None
        for c in todo:
            if reader is None:
                reader, extras = self._make_reader(cfg)
                self._write_manifest(extras, cfg, out_dir, store_dir if store else None)
            part = df.iloc[c * size:(c + 1) * size]
            res, ans, idx, reads, resid = [], [], [], [], []
            for i, row in part.iterrows():
                r = reader.read_row(str(row["sample_id"]), self._load_images(images[i]), prompts[i])
                res += r.results
                ans.append(r.answer)
                if store:
                    idx += r.index
                    reads.append(r.reads)
                    resid.append(r.residuals)
            _atomic_parquet(pd.DataFrame(res), os.path.join(chunk_dir, f"results-{c:05d}.parquet"))
            if store:
                _atomic_npy(np.concatenate(reads), os.path.join(store_dir, f"reads-{c:05d}.npy"))
                _atomic_npy(np.concatenate(resid), os.path.join(store_dir, f"residuals-{c:05d}.npy"))
                _atomic_parquet(pd.DataFrame(idx), os.path.join(store_dir, f"index-{c:05d}.parquet"))
            # The last write: it marks the chunk done.
            _atomic_parquet(pd.DataFrame(ans), os.path.join(chunk_dir, f"answers-{c:05d}.parquet"))
            print(f"[readout] chunk {c + 1}/{n_chunks} done", flush=True)

        results = pd.concat([pd.read_parquet(os.path.join(chunk_dir, f"results-{c:05d}.parquet"))
                             for c in range(n_chunks)], ignore_index=True) if n_chunks else pd.DataFrame()
        answers = pd.concat([pd.read_parquet(os.path.join(chunk_dir, f"answers-{c:05d}.parquet"))
                             for c in range(n_chunks)], ignore_index=True) if n_chunks else pd.DataFrame()
        _atomic_parquet(results, results_path)
        _atomic_parquet(answers, answers_path)
        print(f"[readout] wrote {len(results)} reads and {len(answers)} answers to {out_dir}", flush=True)

        if context.logger:
            _safe_log_table(context.logger, answers, "readout/answers",
                            prefer_cols=[c for c in ("sample_id", "top", "margin", "yes_no") if c in answers.columns])
            first = results[results["rank"] == 0] if "rank" in results.columns else results
            _safe_log_table(context.logger, first, "readout/top1",
                            prefer_cols=["sample_id", "position", "token_at", "selectors", "layer", "token"])
        outputs = _collect_outputs(context, {k: s.optional for k, s in context.node.outputs.items()})
        return StageResult(outputs=outputs, metadata={"rows": len(answers), "reads": len(results)})

    @staticmethod
    def _write_manifest(extras: Dict[str, Any], cfg: DictConfig, out_dir: str, store_dir: Optional[str]) -> None:
        manifest = dict(extras.get("manifest") or {})
        manifest["input"] = OmegaConf.select(cfg, "data.parquet_path", default=None)
        text = json.dumps(manifest, indent=1, default=str) + "\n"
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            fh.write(text)
        if store_dir:
            with open(os.path.join(store_dir, "manifest.json"), "w") as fh:
                fh.write(text)
            tok, keep = extras.get("tokenizer"), extras.get("vocab")
            if tok is not None:
                ids = list(keep) if keep is not None else list(range(int(manifest.get("vocab_size", 0))))
                vocab = pd.DataFrame({"index": range(len(ids)), "token_id": ids,
                                      "token": [tok.decode([i]) for i in ids]})
                _atomic_parquet(vocab, os.path.join(store_dir, "vocab.parquet"))
