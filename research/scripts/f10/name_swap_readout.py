#!/usr/bin/env python
"""F10.1: read Yes against No at the first answer token, with the offline vLLM engine.

The chat is 1 user turn: the text, then the image (D148). Thinking is off. Each
context gets 1 request with 1 token, and `logprob_token_ids` returns the raw
logprob of each answer word from that 1 forward pass. F8 sends 1 request for
each word, and on Gemma 2 requests give different values for the same token.

| Readout | Context | Value |
|---------|---------|-------|
| `c_yes_i` | the question and the answer form | logp(`Yes`) - logp(`No`) |
| `c_yes` | the question alone | the same |
| `c_lc_i`, `c_lc` | as above | logp(`yes`) - logp(`no`) |

Nothing masks the output, thus `top_*` is the first token of the free answer.
The output is 1 part file for each chunk in `--out-dir`, and `retest.parquet`:
the first `--retest` items again, in reverse order, for the noise of the readout.
A restart skips the done items. Rules: vlm-narratives-docs/facct-runbook.md, F10.1.

Use:
    python research/scripts/f10/name_swap_readout.py --model-key gemma-4-12b --model-path DIR \
        --image-root DIR --items data/f10/name_items.parquet --out-dir data/f10/runs/gemma-4-12b
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "f8"))
from r3_readout import INSTRUCT, QWEN_MM, load_image  # noqa: E402

__version__ = "1.2.0"

WORDS = ("Yes", "No", "yes", "no")
READOUTS = {"i": INSTRUCT["yes"], "q": ""}
NAMES = {"i": ("c_yes_i", "c_lc_i"), "q": ("c_yes", "c_lc")}


def render(tok, question: str, suffix: str, strip_bos: bool) -> str:
    msgs = [{"role": "user", "content": [{"type": "text", "text": question + suffix}, {"type": "image"}]}]
    ctx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if strip_bos and tok.bos_token and ctx.startswith(tok.bos_token):
        ctx = ctx[len(tok.bos_token):]  # the engine adds the BOS token
    return ctx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retest", type=int, default=200)
    ap.add_argument("--dtype", default="auto", help="float32 removes the rounding noise of bfloat16 on Gemma (F10.3)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ids = {}
    for w in WORDS:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"{w!r} is {len(t)} tokens: {t}")
        ids[w] = t[0]
    probe = tok("x").input_ids
    strip_bos = tok.bos_token_id is not None and probe[0] == tok.bos_token_id
    print(f"[f10] {args.model_key} word ids {ids} strip_bos={strip_bos}", flush=True)

    kw = {"mm_processor_kwargs": QWEN_MM} if args.model_key.startswith("qwen") else {}
    # The arms differ before the image, thus a prefix cache saves little.
    llm = LLM(model=args.model_path, trust_remote_code=True, max_model_len=4096, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.90, seed=0, enable_prefix_caching=False, logprobs_mode="raw_logprobs", dtype=args.dtype, **kw)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(WORDS), logprob_token_ids=list(ids.values()))

    def read(ch: pd.DataFrame, show: bool = False) -> pd.DataFrame:
        with ThreadPoolExecutor(16) as ex:
            imgs = list(ex.map(lambda s: load_image(args.image_root, s), ch["sample_id"]))
        jobs = [(i, rd) for i in range(len(ch)) for rd in READOUTS]
        ctxs = {(i, rd): render(tok, ch["question"].iat[i], READOUTS[rd], strip_bos) for i, rd in jobs}
        prompts = [{"prompt": ctxs[j], "multi_modal_data": {"image": imgs[j[0]]}} for j in jobs]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        rows = [{"item_id": x, "model": args.model_key} for x in ch["item_id"]]
        for (i, rd), o in zip(jobs, outs):
            # An output in the wrong order would give wrong values with no error.
            if o.prompt is not None and o.prompt != ctxs[(i, rd)]:
                raise RuntimeError(f"output order: item {i} {rd}")
            g = o.outputs[0]
            lp = g.logprobs[0]
            row = rows[i]
            for w in WORDS:
                row[f"lp_{w}_{rd}"] = lp[ids[w]].logprob
            cap, low = NAMES[rd]
            row[cap] = row[f"lp_Yes_{rd}"] - row[f"lp_No_{rd}"]
            row[low] = row[f"lp_yes_{rd}"] - row[f"lp_no_{rd}"]
            row[f"top_{rd}"] = tok.decode(g.token_ids[:1])
            row[f"lp_top_{rd}"] = lp[g.token_ids[0]].logprob
            if show and i == 0:
                bos = o.prompt_token_ids.count(tok.bos_token_id) if tok.bos_token_id is not None else 0
                print(f"[f10] check {ctxs[(i, rd)]!r}", flush=True)
                print(f"[f10] check prompt tokens {len(o.prompt_token_ids)}, bos {bos}, top {row[f'top_{rd}']!r}, "
                      f"{ {w: round(row[f'lp_{w}_{rd}'], 3) for w in WORDS} }", flush=True)
        return pd.DataFrame(rows)

    items = pd.read_parquet(args.items)
    if args.limit:
        items = items.head(args.limit)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = sorted(out.glob("part-*.parquet"))
    done = set().union(*[set(pd.read_parquet(p, columns=["item_id"])["item_id"]) for p in parts]) if parts else set()
    todo = items[~items["item_id"].isin(done)].reset_index(drop=True)
    print(f"[f10] {len(items)} items, {len(done)} done, {len(todo)} to do", flush=True)
    n_parts, t0 = len(parts), time.perf_counter()
    for start in range(0, len(todo), args.chunk):
        ch = todo.iloc[start:start + args.chunk]
        read(ch, show=start == 0).to_parquet(out / f"part-{n_parts:05d}.parquet", index=False)
        n_parts += 1
        n = start + len(ch)
        print(f"[f10] {n}/{len(todo)} items, {n / (time.perf_counter() - t0):.1f} items/s", flush=True)
    if args.retest and not (out / "retest.parquet").exists():
        read(items.head(args.retest).iloc[::-1]).to_parquet(out / "retest.parquet", index=False)
    print("[f10] finished", flush=True)


if __name__ == "__main__":
    main()
