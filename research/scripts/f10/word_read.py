#!/usr/bin/env python
"""F10.2: read evaluative and status words on the image patches, with a J++ lens.

The image comes first, then the F9 prompt, in 1 forward pass of 1 image. For each
image and layer, the lens gives the logit of each word at each patch. The score of
a word is pooled in 2 forms, before any control is subtracted:

| Column | Value |
|--------|-------|
| `mean_{word}` | the mean over the patches |
| `block_{word}` | the highest block of the patch grid (4 x 4 blocks) |

The analysis subtracts the control words (`analyze_reads.py`). The output is 1 part
file for each chunk in `--out-dir`, 1 row for each image and layer. A restart skips
the done images. Rules: vlm-narratives-docs/facct-runbook.md, F10.2.

Use:
    python research/scripts/f10/word_read.py --model-key gemma-4-31b --model-dir DIR --lens FILE \
        --image-root DIR --out-dir data/f10/reads/gemma-4-31b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "sub/jacobian-lens"), str(ROOT / "research/scripts/f8")]

__version__ = "1.0.0"

SEED = 20261030
N_IMAGES = 1000
LAYERS = {"gemma-4-31b": [8, 16, 24, 32, 40, 45, 48, 56], "qwen3.5-9b": [4, 8, 12, 16, 20, 24, 28]}
POOL_SIZE = {"gemma-4-31b": 4, "qwen3.5-9b": 8}  # 4 x 4 blocks of the grid, as `### The J++ replication`
TARGETS = ("good", "nice", "yummy", "stylish", "cool", "bad", "ugly", "rich", "wealthy", "poor", "homeless")
SENSES = ("cold", "warm", "flavor", "delicious", "quality")
PROMPTS = {"restaurant": "Does this look like a nice restaurant to eat at?",
           "street": "Would this street make a good spot for taking a photo?"}
POOLS = {"restaurant": "data/f4_traces/subj/f4_restaurants_10k.parquet", "street": "data/f4_traces/subj/f4_street_10k.parquet"}


def block_max(x: torch.Tensor, rows: int, cols: int, pool: int) -> torch.Tensor:
    """Give the highest block mean of x [rows * cols, words], for pool x pool blocks."""
    g = x.view(rows, cols, -1)
    means = [g[r:r + pool, c:c + pool].mean(dim=(0, 1)) for r in range(0, rows, pool) for c in range(0, cols, pool)]
    return torch.stack(means).max(dim=0).values


def sample_images() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p, columns=["image_idx", "sample_id"]).sample(N_IMAGES, random_state=SEED).assign(case=c)
                      for c, p in POOLS.items()], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", required=True, choices=list(LAYERS))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from jlens.hooks import ActivationRecorder
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from research.monocle import extract, jlens_read
    from research.monocle.probe_words import POOL
    from r3_readout import load_image

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    proc = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.float32, device_map=device,
                                                        attn_implementation="sdpa").eval()
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    lens = jlens_read.load_lens(args.lens)
    tok = proc.tokenizer
    layers = LAYERS[args.model_key]

    ids: dict[str, int] = {}
    for w in (*TARGETS, *SENSES, *POOL):
        t = tok.encode(" " + w, add_special_tokens=False)
        if len(t) == 1:
            ids[w] = t[0]
        elif w in TARGETS:
            raise ValueError(f"{w!r} is {len(t)} tokens")
    words = list(ids)
    index = torch.tensor([ids[w] for w in words], device=device)
    image_token = int(model.config.image_token_id)
    print(f"[f10.2] {len(words)} words ({len(POOL)} in the pool), layers {layers}", flush=True)
    (out / "meta.json").write_text(json.dumps({"layers": layers, "ids": ids, "version": __version__}) + "\n")

    images = sample_images()
    if args.limit:
        images = images.groupby("case", group_keys=False).head(args.limit)
    parts = sorted(out.glob("part-*.parquet"))
    done = set()
    for p in parts:
        done |= set(map(tuple, pd.read_parquet(p, columns=["case", "image_idx"]).drop_duplicates().to_numpy()))
    todo = [r for r in images.itertuples(index=False) if (r.case, r.image_idx) not in done]
    print(f"[f10.2] {len(images)} images, {len(done)} done, {len(todo)} to do", flush=True)

    rows, n_part, t0 = [], len(parts), time.perf_counter()
    for n, r in enumerate(todo):
        img = load_image(args.image_root, r.sample_id)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPTS[r.case]}]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = dict(proc(text=[text], images=[[img]], return_tensors="pt").to(device))
        pos = (inputs["input_ids"][0] == image_token).nonzero().flatten()
        grid = extract.infer_grid(inputs, pos.numel(), img.size)
        with ActivationRecorder(lens_model.layers, at=layers) as rec, torch.inference_mode():
            model(**inputs, use_cache=False)
            for l in layers:
                h = rec.activations[l][0, pos].float()
                x = lens_model.unembed(lens.transport(h, l)).float()[:, index]
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"non-finite lens logits at layer {l}")
                row = {"case": r.case, "image_idx": r.image_idx, "sample_id": r.sample_id, "layer": l}
                row |= dict(zip((f"mean_{w}" for w in words), x.mean(0).tolist()))
                row |= dict(zip((f"block_{w}" for w in words), block_max(x, grid.n_rows, grid.n_cols, POOL_SIZE[args.model_key]).tolist()))
                rows.append(row)
        if n == 0:
            print(f"[f10.2] check tokens {inputs['input_ids'].shape[1]}, patches {pos.numel()}, grid {grid.n_rows} x {grid.n_cols}", flush=True)
        if (n + 1) % args.chunk == 0 or n + 1 == len(todo):
            pd.DataFrame(rows).to_parquet(out / f"part-{n_part:05d}.parquet", index=False)
            rows, n_part = [], n_part + 1
            print(f"[f10.2] {n + 1}/{len(todo)} images, {(n + 1) / (time.perf_counter() - t0):.2f} images/s", flush=True)
    print("[f10.2] finished", flush=True)


if __name__ == "__main__":
    main()
