#!/usr/bin/env python3
"""Send the "what am I looking at?" probe to a running vLLM server.

Each image gets R independent requests (one request for each repeat, each with
its own seed). The script sends the repeats of one image next to each other, so
the server can reuse the cached image encoding. It appends each result to a
JSONL file, and a restart skips every (image_idx, repeat) that is already done
without an error. At the end, it writes the JSONL to a parquet file.

The server must allow the image directory (--allowed-local-media-path), because
the images go as file:// URLs.

F4 (`scripts/f4/serve_and_query.sbatch`) uses the same script with other
prompts and the model-card sampling (--top-p, --top-k, --min-p,
--presence-penalty, --repetition-penalty). The defaults keep the whatami
values. With a reasoning parser on the server, the thinking goes to the
`reasoning` column.

Usage:
    python scripts/whatami/query.py --base-url http://127.0.0.1:8000/v1 \
        --model gemma-4-12b --sample data/whatami/sample_1k.parquet \
        --image-root /scratch/$USER/whatami/images --out-dir multirun/whatami/gemma-4-12b
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

PROMPT = "What am I looking at?"


def request_seed(sample_id: str, repeat: int) -> int:
    """Give a fixed seed for each (image, repeat), the same for every model."""
    h = hashlib.sha256(f"{sample_id}:{repeat}".encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def done_keys(path: Path) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    if not path.exists():
        return keys
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partial last line from a killed job
            if row.get("error") is None:
                keys.add((row["image_idx"], row["repeat"]))
    return keys


async def one_request(client, args, row, repeat, sampling, sem, out_fh, lock, stats):
    seed = request_seed(row.sample_id, repeat)
    local_path = Path(args.image_root) / f"{row.sample_id}.jpg"  # see stage_inputs.sh
    # F8.2 gives each row its own prompt (the cue arms); else all rows share --prompt.
    prompt = getattr(row, args.prompt_column) if args.prompt_column else args.prompt
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"file://{local_path}"}},
            {"type": "text", "text": prompt},
        ],
    }]
    extra_body = {
        "top_k": sampling["top_k"],
        "min_p": sampling["min_p"],
        "repetition_penalty": sampling["repetition_penalty"],
    }
    if args.chat_template_kwargs:
        extra_body["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
    rec = {
        "model_key": args.model,
        "served_model": args.served_model or args.model,
        "image_idx": int(row.image_idx),
        "sample_id": row.sample_id,
        "repeat": repeat,
        "seed": seed,
        "prompt": prompt,
        "sampling": json.dumps(sampling, sort_keys=True),
        "chat_template_kwargs": args.chat_template_kwargs,
    }
    async with sem:
        t0 = time.perf_counter()
        rec["started_at"] = datetime.now(timezone.utc).isoformat()
        try:
            resp = await client.chat.completions.create(
                model=args.served_model or args.model,
                messages=messages,
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                max_tokens=sampling["max_tokens"],
                presence_penalty=sampling["presence_penalty"],
                frequency_penalty=sampling["frequency_penalty"],
                seed=seed,
                n=1,
                extra_body=extra_body,
            )
            choice = resp.choices[0]
            # vLLM 0.25 names the field `reasoning`; older servers name it `reasoning_content`.
            extra = choice.message.model_extra or {}
            rec.update(
                response=choice.message.content,
                reasoning=extra.get("reasoning") or extra.get("reasoning_content")
                or getattr(choice.message, "reasoning_content", None),
                finish_reason=choice.finish_reason,
                prompt_tokens=resp.usage.prompt_tokens if resp.usage else None,
                completion_tokens=resp.usage.completion_tokens if resp.usage else None,
                error=None,
            )
        except Exception as e:  # noqa: BLE001 - log every failure, retry on restart
            rec.update(response=None, reasoning=None, finish_reason=None,
                       prompt_tokens=None, completion_tokens=None,
                       error=f"{type(e).__name__}: {e}"[:2000])
        rec["latency_s"] = round(time.perf_counter() - t0, 3)
    async with lock:
        out_fh.write(json.dumps(rec) + "\n")
        out_fh.flush()
        stats["n"] += 1
        stats["err"] += rec["error"] is not None
        if stats["n"] % 500 == 0:
            rate = stats["n"] / (time.perf_counter() - stats["t0"])
            print(f"[query] {stats['n']}/{stats['todo']} done, {stats['err']} errors, {rate:.1f} req/s", flush=True)


async def run(args) -> int:
    sample = pd.read_parquet(args.sample)
    if args.limit:
        sample = sample.head(args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "responses.jsonl"
    done = done_keys(jsonl)
    todo = [(row, r) for row in sample.itertuples(index=False)
            for r in range(args.repeats) if (int(row.image_idx), r) not in done]
    print(f"[query] {len(sample)} images x {args.repeats} repeats; {len(done)} done, {len(todo)} to do", flush=True)

    sampling = {
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty, "presence_penalty": args.presence_penalty,
        "frequency_penalty": 0.0,
        "max_tokens": args.max_tokens,
    }
    (out_dir / "run_config.json").write_text(json.dumps({
        "model": args.model, "served_model": args.served_model, "prompt": args.prompt,
        "prompt_column": args.prompt_column,
        "repeats": args.repeats, "sampling": sampling, "sample": str(args.sample),
        "n_images": len(sample), "chat_template_kwargs": args.chat_template_kwargs,
    }, indent=2))

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout, max_retries=2)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"n": 0, "err": 0, "todo": len(todo), "t0": time.perf_counter()}
    # A job that was killed in a write (a preemption) can leave a partial last
    # line with no newline. Close that line, thus the next row starts a new line.
    if jsonl.exists() and jsonl.stat().st_size:
        with jsonl.open("rb") as fh:
            fh.seek(-1, 2)
            partial = fh.read(1) != b"\n"
        if partial:
            with jsonl.open("a") as fh:
                fh.write("\n")
    with jsonl.open("a") as fh:
        await asyncio.gather(*(one_request(client, args, row, r, sampling, sem, fh, lock, stats)
                               for row, r in todo))
    print(f"[query] finished: {stats['n']} sent, {stats['err']} errors", flush=True)

    rows = []
    for line in jsonl.open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a partial line from a killed job; its key was sent again
    df = pd.DataFrame(rows)
    # Keep the last good result for each key; a key with no good result keeps its last error.
    df["_ok"] = df["error"].isna()
    df = df.sort_values(["_ok", "started_at"]).drop_duplicates(["image_idx", "repeat"], keep="last")
    df = df.drop(columns="_ok").sort_values(["image_idx", "repeat"])
    df.to_parquet(out_dir / "responses.parquet", index=False)
    n_ok = int(df["error"].isna().sum())
    print(f"[query] responses.parquet: {len(df)} keys, {n_ok} good", flush=True)
    return 0 if n_ok == len(sample) * args.repeats else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="model key, also the served model name")
    ap.add_argument("--served-model", default=None)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--image-root", required=True, help="dir that holds the staged images as <sample_id>.jpg")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--prompt-column", default=None, help="read the prompt of each row from this sample column (F8.2)")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--chat-template-kwargs", default=None, help='JSON, e.g. {"enable_thinking": false}')
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    raise SystemExit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
