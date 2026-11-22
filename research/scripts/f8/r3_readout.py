#!/usr/bin/env python
"""F8.3: the R3 readout after a stored answer, with the offline vLLM engine.

For each item (1 image, 1 prompt, 1 stored final answer) the script builds the
2-turn chat of D18 and reads 2 logit differences at the first token of the
readout turn (vlm-narratives-docs/facct-runbook.md, F8.3):

| Readout | Context | Value |
|---------|---------|-------|
| `c_yes` | the chat with the generation prompt, thinking off | logp(`Yes`) - logp(`No`) |
| `c_cool` | the same context plus the prefix `This place is` | logp(`_cool`) - logp(`_lame`) |
| `c_yes_i`, `c_cool_i` | as above, with the answer form after the question (`INSTRUCT`) | the same |

The script renders the chat with the chat template of the model. It does not
use the server, because the gemma template removes the empty thought channel
from a final assistant message. Thus the server cannot give the prefix of
`c_cool` with the empty thought channel that the generation prompt has.

Each answer word gets 1 request with 1 token. `allowed_token_ids` forces the
word, and the engine returns its raw logprob (`logprobs_mode` raw_logprobs:
before the mask). The first pass sends 1 request for each base context of
each item. The second pass sends the other requests, which then read the
cached prefix.

The items file has 1 row for each item: `item_id` (unique), `case`
(restaurant or street), `sample_id`, `prompt`, `response`. The output is 1
part file for each chunk in `--out-dir`. A restart skips the done items.

Use:
    python research/scripts/f8/r3_readout.py --model-key gemma-4-12b --model-path DIR \
        --image-root DIR --items items.parquet --out-dir data/f8/r3/gemma-4-12b/f4
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

QUESTION = {  # D18, verbatim
    "restaurant": "Is this a cool place to eat at?",
    "street": "Is this a cool place to take a photo in?",
}
PREFIX = "This place is"  # Q36 (Claude): the next word has a leading space
WORDS = {"yes": "Yes", "no": "No", "cool": " cool", "lame": " lame"}  # D26
# The instructed form adds the answer form to the D18 question (Claude, after
# the smoke job 103155): with the verbatim question, gemma-4-12b starts with
# "Whether" (logp -0.02), `Yes` is at rank 12 to 47, and `_lame` at rank
# 20,000 to 80,000. The verbatim readouts stay; the instructed ones are more.
INSTRUCT = {
    "yes": " Answer with one word: yes or no.",
    "cool": ' Answer with "This place is cool." or "This place is lame."',
}
# readout -> (base context, prefix, word a, word b); c = logp(a) - logp(b)
READOUTS = {
    "c_yes": ("plain", "", "yes", "no"),
    "c_cool": ("plain", PREFIX, "cool", "lame"),
    "c_yes_i": ("i_yes", "", "yes", "no"),
    "c_cool_i": ("i_cool", PREFIX, "cool", "lame"),
}
QWEN_MM = {"min_pixels": 784, "max_pixels": 1048576}  # as the F4 server


def word_ids(tok) -> dict[str, int]:
    ids = {}
    for k, w in WORDS.items():
        t = tok.encode(w, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"{w!r} is {len(t)} tokens: {t}")
        ids[k] = t[0]
    return ids


def render(tok, case: str, prompt: str, response: str, strip_bos: bool) -> dict[str, str]:
    """Give the base contexts of the readouts as text with 1 image placeholder."""
    out = {}
    for base, suffix in (("plain", ""), ("i_yes", INSTRUCT["yes"]), ("i_cool", INSTRUCT["cool"])):
        msgs = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": response},
            {"role": "user", "content": [{"type": "text", "text": QUESTION[case] + suffix}]},
        ]
        ctx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if strip_bos and tok.bos_token and ctx.startswith(tok.bos_token):
            ctx = ctx[len(tok.bos_token):]  # the engine adds the BOS token
        out[base] = ctx
    return out


def load_image(root: str, sample_id: str) -> Image.Image:
    with Image.open(Path(root) / f"{sample_id}.jpg") as im:
        return im.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--items", required=True, nargs="+", help="1 or more items parquet files, run in order")
    ap.add_argument("--out-dir", required=True, nargs="+", help="1 out dir for each items file")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=12288)
    args = ap.parse_args()
    assert len(args.items) == len(args.out_dir)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ids = word_ids(tok)
    probe = tok("x").input_ids
    strip_bos = tok.bos_token_id is not None and probe[0] == tok.bos_token_id
    print(f"[r3] {args.model_key} word ids {ids} strip_bos={strip_bos}", flush=True)

    kw = {}
    if args.model_key.startswith("qwen"):
        kw["mm_processor_kwargs"] = QWEN_MM
    llm = LLM(model=args.model_path, trust_remote_code=True, max_model_len=args.max_model_len,
              limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.90, seed=0,
              enable_prefix_caching=True, logprobs_mode="raw_logprobs", **kw)

    def sp(word: str) -> SamplingParams:
        return SamplingParams(max_tokens=1, temperature=0.0, logprobs=20, allowed_token_ids=[ids[word]])

    for items_path, out_dir in zip(args.items, args.out_dir):
        items = pd.read_parquet(items_path)
        items = items[items["response"].fillna("").str.strip() != ""]
        if args.limit:
            items = items.head(args.limit)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        done = set()
        for p in out.glob("part-*.parquet"):
            done |= set(pd.read_parquet(p, columns=["item_id"])["item_id"])
        todo = items[~items["item_id"].isin(done)].reset_index(drop=True)
        print(f"[r3] {items_path}: {len(items)} items, {len(done)} done, {len(todo)} to do", flush=True)
        t0 = time.perf_counter()
        stats = {"mask_failed": 0, "still_failed": 0}
        n_parts = len(list(out.glob("part-*.parquet")))
        for start in range(0, len(todo), args.chunk):
            ch = todo.iloc[start:start + args.chunk]
            with ThreadPoolExecutor(16) as ex:
                imgs = list(ex.map(lambda s: load_image(args.image_root, s), ch["sample_id"]))
            ctxs = [render(tok, r.case, r.prompt, r.response, strip_bos) for r in ch.itertuples(index=False)]
            # Pass 1 fills the prefix cache with 1 request for each base context;
            # pass 2 sends the other requests, which read the cached prefix.
            reqs = [(i, rd, w, base, pre) for i in range(len(ch)) for rd, (base, pre, a, b) in READOUTS.items()
                    for w in (a, b)]
            first, seen = [], set()
            for q in reqs:
                if (q[0], q[3]) not in seen:
                    seen.add((q[0], q[3]))
                    first.append(q)
            rest = [q for q in reqs if q not in first]
            res: dict[tuple[int, str, str], tuple[float, int, str, str]] = {}

            def run(jobs: list) -> list:
                """Send the jobs; store each forced value; give back the jobs where the mask failed."""
                prompts = [{"prompt": ctxs[i][base] + pre, "multi_modal_data": {"image": imgs[i]}}
                           for i, _, _, base, pre in jobs]
                outs = llm.generate(prompts, [sp(w) for _, _, w, _, _ in jobs], use_tqdm=False)
                failed = []
                for (i, rd, w, base, pre), o in zip(jobs, outs):
                    # An output in the wrong order would give wrong values with no error.
                    if o.prompt is not None and o.prompt != ctxs[i][base] + pre:
                        raise RuntimeError(f"output order: item {i} {rd} {w}")
                    g = o.outputs[0]
                    tid = g.token_ids[0]
                    lp = g.logprobs[0]
                    top = json.dumps([[tok.decode([t]), round(v.logprob, 3)]
                                      for t, v in sorted(lp.items(), key=lambda kv: -kv[1].logprob)[:5]])
                    if tid == ids[w]:
                        res[(i, rd, w)] = (lp[tid].logprob, lp[tid].rank, top, "forced")
                    else:
                        # qwen3.5-9b, job 103179: 1 request in a large batch came back with
                        # another token, thus the mask did not apply. The raw top 20 can
                        # still hold the word.
                        failed.append((i, rd, w, base, pre))
                        if ids[w] in lp:
                            res[(i, rd, w)] = (lp[ids[w]].logprob, lp[ids[w]].rank, top, "top20")
                        else:
                            res.setdefault((i, rd, w), (float("nan"), -1, top, "missing"))
                    if start == 0 and i == 0:
                        print(f"[r3] check {rd} {w}: prompt tokens {len(o.prompt_token_ids)}, "
                              f"head {o.prompt_token_ids[:3]}, sampled {tid}, value {res[(i, rd, w)][:2]}, "
                              f"top {top}", flush=True)
                return failed

            for jobs in (first, rest):
                failed = run(jobs)
                if failed:
                    print(f"[r3] mask failed on {len(failed)} of {len(jobs)} requests; send them again 1 at a time",
                          flush=True)
                    still = []
                    for q in failed:
                        still += run([q])
                    stats["mask_failed"] += len(failed)
                    stats["still_failed"] += len(still)
            rows = []
            for i, r in enumerate(ch.itertuples(index=False)):
                row = {"item_id": r.item_id, "model": args.model_key}
                for rd, (_, _, a, b) in READOUTS.items():
                    sfx = rd[len("c_yes"):] if rd.startswith("c_yes") else rd[len("c_cool"):]
                    for w in (a, b):
                        (row[f"lp_{w}{sfx}"], row[f"rank_{w}{sfx}"], row[f"top_{w}{sfx}"],
                         row[f"how_{w}{sfx}"]) = res[(i, rd, w)]
                    row[rd] = res[(i, rd, a)][0] - res[(i, rd, b)][0]
                rows.append(row)
            pd.DataFrame(rows).to_parquet(out / f"part-{n_parts:05d}.parquet", index=False)
            n_parts += 1
            n = start + len(ch)
            print(f"[r3] {n}/{len(todo)} items, {n / (time.perf_counter() - t0):.1f} items/s, "
                  f"mask failed {stats['mask_failed']}, still failed {stats['still_failed']}", flush=True)
        print(f"[r3] {items_path} finished", flush=True)


if __name__ == "__main__":
    main()
