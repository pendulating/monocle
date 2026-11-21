#!/usr/bin/env python
"""F10.30: the yes-or-no answer of the 14 prompts on many images, from a vLLM server.

The prompt is the F10 read prompt: `Is this {noun} {adjective}? Answer with one word: yes or no.`
Thinking is off. A request takes 1 token and the top log probabilities, and the record keeps the
8 answer forms (`Yes`, `yes`, ` Yes`, ` yes`, `No`, `no`, ` No`, ` no`) and the floor of the list.
A form that is not in the list is empty; the reader fills it with the floor (D164).

Use (inside the server job, F4_PROBE=1 F4_ANSWERS=1 in `scripts/f4/serve_and_query.sbatch`):
    python scripts/f10/answer_query.py --base-url URL --model NAME --sample PARQUET --image-root DIR --out FILE
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

__version__ = "1.0.0"
ADJECTIVES = ("cool", "nice", "ordinary", "pretty", "ugly", "fancy", "urban")
NOUNS = ("place", "scene")
FORMS = ("Yes", "yes", " Yes", " yes", "No", "no", " No", " no")


async def one(client, args, row, noun, adj, sem, out, stats):
    path = Path(args.image_root) / f"{row.sample_id}.jpg"
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{path}"}},
                                             {"type": "text", "text": f"Is this {noun} {adj}? Answer with one word: yes or no."}]}]
    async with sem:
        for attempt in range(4):
            try:
                got = await client.chat.completions.create(model=args.model, messages=messages, max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=args.top,
                                                           extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    stats["errors"] += 1
                    print(f"error {row.image_idx} {noun} {adj}: {e}", flush=True)
                    return
                await asyncio.sleep(2 ** attempt)
    top = got.choices[0].logprobs.content[0].top_logprobs
    lp = {t.token: t.logprob for t in reversed(top)}
    floor = min(t.logprob for t in top)
    out.write(json.dumps({"image_idx": int(row.image_idx), "noun": noun, "adjective": adj, "floor": floor, "first": top[0].token, "n_top": len(top),
                          **{f"lp_{k}": lp.get(k) for k in FORMS}}) + "\n")
    out.flush()
    stats["done"] += 1
    if stats["done"] % 5000 == 0:
        print(f"[f10.30] {stats['done']}", flush=True)


async def main_async(args) -> None:
    images = pd.read_parquet(args.sample)
    if args.limit:
        images = images.head(args.limit)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["image_idx"], r["noun"], r["adjective"]))
            except json.JSONDecodeError:
                continue
    client = AsyncOpenAI(base_url=args.base_url, api_key="none", timeout=600)
    sem = asyncio.Semaphore(args.concurrency)
    stats = {"done": 0, "errors": 0, "skipped": len(done)}
    print(f"[f10.30] {len(images)} images, {len(ADJECTIVES) * len(NOUNS)} prompts, {len(done)} done before", flush=True)
    with path.open("a") as out:
        await asyncio.gather(*(one(client, args, row, n, a, sem, out, stats) for row in images.itertuples() for n in NOUNS for a in ADJECTIVES
                               if (int(row.image_idx), n, a) not in done))
    print(f"[f10.30] finished: {stats}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("base-url", "model", "sample", "image-root", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top", type=int, default=2000, help="the length of the log probability list; above 20 needs --max-logprobs on the server")
    ap.add_argument("--concurrency", type=int, default=96)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
