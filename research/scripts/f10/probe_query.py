#!/usr/bin/env python
"""F10.14: read the yes-or-no margin of many adjectives on the same images.

The prompt keeps the form of the read prompt: `Is this place {x}? Answer with one word: yes
or no.` Thinking is off. A request takes 1 token and the top log probabilities. `cool` is
in the list, thus its margin must agree with `answers.parquet` of F10.7.

The list is fixed before any result. It has 4 groups: general valence, aesthetic properties,
aesthetic values, and neutral checks.

Use (inside the server job, F4_PROBE=1 in `research/scripts/f4/serve_and_query.sbatch`):
    python research/scripts/f10/probe_query.py --base-url URL --model NAME --sample PARQUET --image-root DIR --out FILE
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

__version__ = "1.0.0"

GROUPS = {
    "target": ["cool"],
    "valence": ["nice", "good", "bad", "great"],
    "property": ["beautiful", "pretty", "ugly", "clean", "dirty", "neat", "messy", "modern", "new", "old", "historic", "colorful", "bright", "dark", "busy", "crowded",
                 "quiet", "empty", "green", "gritty", "industrial", "urban", "run-down"],
    "value": ["authentic", "unique", "charming", "interesting", "boring", "trendy", "stylish", "fashionable", "expensive", "cheap", "upscale", "poor", "rich", "safe",
              "dangerous", "friendly", "welcoming", "lively", "functional", "ordinary", "famous"],
    "neutral": ["real", "outdoors", "in a city", "in New York"],
}
FORMS = ("Yes", "yes", "No", "no")


async def one(client, args, row, word, sem, out, stats):
    path = Path(args.image_root) / f"{row.sample_id}.jpg"
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"file://{path}"}},
                                             {"type": "text", "text": f"Is this place {word}? Answer with one word: yes or no."}]}]
    async with sem:
        for attempt in range(4):
            try:
                got = await client.chat.completions.create(model=args.model, messages=messages, max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=args.top,
                                                           extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    stats["errors"] += 1
                    print(f"error {row.image_idx} {word}: {e}", flush=True)
                    return
                await asyncio.sleep(2 ** attempt)
    top = got.choices[0].logprobs.content[0].top_logprobs
    lp = {t.token.strip(): t.logprob for t in reversed(top)}
    floor = min(t.logprob for t in top)
    out.write(json.dumps({"image_idx": int(row.image_idx), "word": word, "floor": floor, "first": top[0].token, **{f"lp_{k}": lp.get(k) for k in FORMS}}) + "\n")
    out.flush()
    stats["done"] += 1
    if stats["done"] % 2000 == 0:
        print(f"[f10.14] {stats['done']}", flush=True)


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
                done.add((r["image_idx"], r["word"]))
            except json.JSONDecodeError:
                continue
    client = AsyncOpenAI(base_url=args.base_url, api_key="none", timeout=600)
    sem = asyncio.Semaphore(args.concurrency)
    stats = {"done": 0, "errors": 0}
    with path.open("a") as out:
        await asyncio.gather(*(one(client, args, row, w, sem, out, stats) for row in images.itertuples() for g in GROUPS.values() for w in g if (int(row.image_idx), w) not in done))
    print(f"[f10.14] finished: {stats}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("base-url", "model", "sample", "image-root", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top", type=int, default=20, help="the length of the log probability list; above 20 needs --max-logprobs on the server")
    ap.add_argument("--concurrency", type=int, default=96)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
