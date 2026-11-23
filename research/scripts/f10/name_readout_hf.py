#!/usr/bin/env python
"""F10.1 in float32: the readouts of `name_swap_readout.py`, with the HF model.

vLLM cannot run these models in float32 (its Triton attention kernel needs too much
shared memory), and bfloat16 rounding moves each Gemma value (D154). The columns and
the contexts are those of `name_swap_readout.py`, thus `analyze_names.py` reads the output.
`--shard i/n` splits the items. A restart skips the done items. `retest.parquet`
repeats the first 50 items, in reverse order.

Use:
    python research/scripts/f10/name_readout_hf.py --model-key gemma-4-31b --model-dir DIR \
        --image-root DIR --out-dir data/f10/runs-fp32/gemma-4-31b --shard 0/3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "research/scripts/f8"), str(ROOT / "research/scripts/f10")]

from research.scripts.f10.name_swap_readout import NAMES, READOUTS, WORDS, render  # noqa: E402
from research.scripts.f8.r3_readout import load_image  # noqa: E402

__version__ = "1.0.0"

ITEMS = ("data/f10/name_items.parquet", "data/f10/name_items_bare.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor

    shard, n_shards = map(int, args.shard.split("/"))
    device = "cuda:0"
    proc = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.float32, device_map=device,
                                                        attn_implementation="sdpa").eval()
    tok = proc.tokenizer
    ids = {}
    for w in WORDS:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"{w!r} is {len(t)} tokens: {t}")
        ids[w] = t[0]

    def read(ch: pd.DataFrame, show: bool = False) -> pd.DataFrame:
        rows = []
        for r in ch.itertuples(index=False):
            img = load_image(args.image_root, r.sample_id)
            row = {"item_id": r.item_id, "model": args.model_key}
            for rd, suffix in READOUTS.items():
                ctx = render(tok, r.question, suffix, False)
                inputs = dict(proc(text=[ctx], images=[[img]], return_tensors="pt").to(device))
                with torch.inference_mode():
                    lp = torch.log_softmax(model(**inputs, use_cache=False).logits[0, -1].float(), dim=-1)
                if not torch.isfinite(lp).all():
                    raise RuntimeError(f"non-finite logits: {r.item_id}")
                for w, i in ids.items():
                    row[f"lp_{w}_{rd}"] = float(lp[i])
                cap, low = NAMES[rd]
                row[cap] = row[f"lp_Yes_{rd}"] - row[f"lp_No_{rd}"]
                row[low] = row[f"lp_yes_{rd}"] - row[f"lp_no_{rd}"]
                row[f"top_{rd}"] = tok.decode([int(lp.argmax())])
                row[f"lp_top_{rd}"] = float(lp.max())
                if show:
                    print(f"[f10.1hf] check {ctx!r}, tokens {inputs['input_ids'].shape[1]}, {cap} {row[cap]:.3f}", flush=True)
            show = False
            rows.append(row)
        return pd.DataFrame(rows)

    items = pd.concat([pd.read_parquet(p) for p in ITEMS], ignore_index=True)
    if args.limit:
        items = items.head(args.limit)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mine = sorted(out.glob(f"part-s{shard}-*.parquet"))
    done = set()
    for p in mine:
        done |= set(pd.read_parquet(p, columns=["item_id"])["item_id"])
    todo = items.iloc[shard::n_shards]
    todo = todo[~todo["item_id"].isin(done)].reset_index(drop=True)
    print(f"[f10.1hf] shard {args.shard}: {len(todo)} to do, {len(done)} done", flush=True)
    n_part, t0 = len(mine), time.perf_counter()
    for start in range(0, len(todo), args.chunk):
        ch = todo.iloc[start:start + args.chunk]
        read(ch, show=start == 0).to_parquet(out / f"part-s{shard}-{n_part:05d}.parquet", index=False)
        n_part += 1
        n = start + len(ch)
        print(f"[f10.1hf] {n}/{len(todo)} items, {n / (time.perf_counter() - t0):.2f} items/s", flush=True)
    if shard == 0 and not (out / "retest.parquet").exists():
        read(items.head(50).iloc[::-1]).to_parquet(out / "retest.parquet", index=False)
    print("[f10.1hf] finished", flush=True)


if __name__ == "__main__":
    main()
