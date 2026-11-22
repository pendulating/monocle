"""Generate for many (photograph, arm) rows at once.

The cost this removes
---------------------
Measured on job 475934, gemma-4-12b on one A6000: a caption-loop image costs
about 154 s, and only about 35 s of that is the map and the contrast reads.
The rest is decoding, one token at a time, at batch 1. Per photograph the run
takes up to 2,160 sequential decode steps and about 24 batched forwards.

So the run is a decode-bound job wearing the clothes of a forward-bound one.

Why the rows stack
------------------
`keep_safe`, `keep_danger` and every `keep_random` draw keep exactly `k`
cells. They thus drop the same NUMBER of patches, and an ablated row's length
depends only on that number. Rows of one length stack with no pad, and a pad
is what we must never put inside a bidirectional image block.

`whole_image` drops nothing, so it is its own length and its own group.

The rows also stack ACROSS photographs, because the prompt text is the same
for one case and every photograph gives 256 patches. That is where the win
is: 100 sequential generations become about 7 batched ones.

What this does NOT preserve
---------------------------
Bit-exact traces. In bf16 the batch size moves a logit by up to 0.67, and a
greedy step flips whenever the top 2 tokens sit closer than that. Most steps
have a clear winner, so most traces are unchanged, but over hundreds of steps
some will differ, and one flipped token changes everything after it.

A trace from a batch is still a real greedy trace of this model. It is not
the SAME trace as batch 1. Numbers read from it are valid; a run is not
bit-comparable with a batch-1 run. `equivalence` measures how far apart the
2 land, so the claim is checked and not assumed.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import torch

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import patch_ablate as pab  # noqa: E402

#: Rows per generation batch. 16 rows of about 700 tokens is roughly 2 GB of
#: KV cache for this model, against 24 GB of weights on a 48 GB card.
DEFAULT_BATCH = 16

#: What the gemma-4 template writes to close a thought channel.
CHANNEL_CLOSE = "<channel|>"


def log(m: str) -> None:
    print(f"[batch-gen] {m}", flush=True)


def group_by_length(jobs: list, key: str = "n_drop") -> dict:
    """Job indices, grouped by the count that fixes their row length.

    Rows only stack when they agree in length, and for an ablated row the
    length follows the number of patches dropped. Grouping by that count is
    thus the same as grouping by what can share a batch.
    """
    out: dict = defaultdict(list)
    for i, j in enumerate(jobs):
        out[int(j[key])].append(i)
    return dict(out)


def generate_many(
    model: Any, proc: Any, inputs: dict, max_new: int,
) -> list:
    """One decoded string for each row. Greedy, so no seed enters.

    Every row shares a prompt length, because `stack_rows` refuses anything
    else. Thus one slice point serves the whole batch.
    """
    n = int(inputs["input_ids"].shape[1])
    tok = getattr(proc, "tokenizer", proc)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=False, use_cache=True,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return [tok.decode(row[n:], skip_special_tokens=False) for row in out]


def run_batched(
    jobs: list, build_row: Callable, model: Any, proc: Any, *,
    max_new: int, batch: int = DEFAULT_BATCH, key: str = "n_drop",
    label: str = "",
) -> list:
    """Generate for every job, in batches of rows that share a length.

    `build_row` takes one job and returns a SINGLE-row input dict. The order
    of the returned list matches the order of `jobs`.
    """
    out: list = [None] * len(jobs)
    groups = group_by_length(jobs, key)
    n_batches = sum(-(-len(v) // batch) for v in groups.values())
    log(f"{label}{len(jobs)} rows in {len(groups)} length groups "
        f"-> {n_batches} batched generations of up to {batch}")

    done = 0
    for n_key in sorted(groups):
        idxs = groups[n_key]
        for start in range(0, len(idxs), batch):
            chunk = idxs[start:start + batch]
            stacked = pab.stack_rows([build_row(jobs[i]) for i in chunk])
            for text, i in zip(generate_many(model, proc, stacked, max_new),
                               chunk):
                out[i] = text
            done += 1
            log(f"{label}{done}/{n_batches} ({len(chunk)} rows, "
                f"{n_key} patches dropped)")
    return out


PAD_MARKS = ("<eos>", "<pad>")


def trim_pad(text: str) -> str:
    """Drop the pad run a batch leaves on a row that stopped early.

    `generate` runs a batch until its LONGEST row stops, and every row that
    finished first is filled with `<eos>`. That fill is an artifact of the
    batch, not a different trace, so a comparison that counts it reports a
    divergence where the model said the same words.
    """
    out = text
    changed = True
    while changed:
        changed = False
        for m in PAD_MARKS:
            if out.endswith(m):
                out, changed = out[:-len(m)], True
    return out


def split_at_close(text: str) -> str:
    """The part of a trace a read actually uses.

    `thinking_read` keeps the thought up to and including the close marker,
    and throws the rest away. Text after the marker thus cannot change any
    number, and counting it would fail a run for words nobody reads.
    """
    at = text.find(CHANNEL_CLOSE)
    return text if at < 0 else text[:at + len(CHANNEL_CLOSE)]


def equivalence(
    model: Any, proc: Any, rows: list, max_new: int, *,
    normalize: Optional[Callable] = None, score: Optional[Callable] = None,
) -> dict:
    """Does a row generate the same text alone as it does in a batch?

    This is the gate on the whole optimization. It reports the exact-match
    rate and the first place a pair diverges, so a caller sees HOW far apart
    they are instead of trusting that they agree.

    It also times both paths, so the same run answers "do they agree?" and
    "was it worth it?". The 2nd number is a floor: a batch runs until the
    LONGEST row in it stops, so a batch of similar rows does better than a
    batch of 1 short row and 1 long one.

    **The text is not what a battery reports.** A trace feeds a read, and the
    read gives the number. `score(i, text)` runs that read, so `dc` says how
    far the REPORTED value moves. A run must gate on `dc`, not on `rate`: 2
    traces can pick different words for the same thought and still rate the
    photograph the same.
    """
    t0 = time.perf_counter()
    alone = [generate_many(model, proc, pab.stack_rows([r]), max_new)[0]
             for r in rows]
    t1 = time.perf_counter()
    together = generate_many(model, proc, pab.stack_rows(rows), max_new)
    t2 = time.perf_counter()
    norm = normalize or trim_pad
    alone = [norm(t) for t in alone]
    together = [norm(t) for t in together]
    same = [a == b for a, b in zip(alone, together)]
    first_diff = []
    for a, b in zip(alone, together):
        if a == b:
            first_diff.append(-1)
            continue
        at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                  min(len(a), len(b)))
        first_diff.append(at)
    out = {"n": len(rows), "exact": int(sum(same)),
           "rate": sum(same) / max(len(rows), 1),
           "first_diff": first_diff, "alone": alone, "together": together,
           "t_alone": t1 - t0, "t_together": t2 - t1,
           "speedup": (t1 - t0) / max(t2 - t1, 1e-9)}
    if score is not None:
        ca = [float(score(i, t)) for i, t in enumerate(alone)]
        cb = [float(score(i, t)) for i, t in enumerate(together)]
        out.update({"c_alone": ca, "c_together": cb,
                    "dc": [b - a for a, b in zip(ca, cb)]})
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def smoke(case: str, n_rows: int, model_dir: str, device: str,
          max_new: int) -> int:
    """Does a batched trace REPORT what a lone trace reports?

    The words are allowed to drift. The number is not: a battery compares
    `c_think` between arms, and the split it looks for is under 1 logit. So
    the gate is `dc`, and the text is only there to show what moved.
    """
    from PIL import Image

    from research.monocle import attn_route as ar
    from research.monocle import jlens_read
    from research.monocle import single as sg
    from research.monocle import thinking_read as tr

    proc, model, tmpl = ar.load_eager(model_dir, device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    pos_id, neg_id = sg.contrast_ids(proc.tokenizer, case)
    image_token_id = int(model.config.image_token_id)
    open_text = tr.open_prompt(proc, tmpl, case)

    #: Every row drops the same COUNT, so the rows share a length.
    drop = list(range(64))
    images, rows = [], []
    for r in sg.load_single_rows(n_rows, case, seed=20260821):
        img = Image.open(r["path"]).convert("RGB")
        images.append(img)
        rows.append(tr.masked_inputs(
            proc, tmpl, img, case, open_text, device=device,
            drop_patches=drop, image_token_id=image_token_id))

    def score(i: int, text: str) -> float:
        """`c_think` for row `i`, read after ITS OWN trace."""
        block, _ = tr.split_thought(text)
        return tr.read_after_thought(
            model, proc, lens_model, tmpl, images[i], case, open_text, block,
            pos_id, neg_id, device=device, drop_patches=drop,
            image_token_id=image_token_id)

    got = equivalence(model, proc, rows, max_new,
                      normalize=lambda t: trim_pad(split_at_close(t)),
                      score=score)
    print(f"\n[batch-gen] exact text {got['exact']}/{got['n']} "
          f"({got['rate']:.0%})")
    print(f"[batch-gen] {got['t_alone']:.1f} s one at a time, "
          f"{got['t_together']:.1f} s in 1 batch of {got['n']} "
          f"-> {got['speedup']:.2f}x")
    dc = got["dc"]
    ad = [abs(x) for x in dc]
    print(f"[batch-gen] dc: median |dc| {sorted(ad)[len(ad) // 2]:.3f}  "
          f"max |dc| {max(ad):.3f}  mean dc {sum(dc) / len(dc):+.3f}")
    for i, (a, b, d) in enumerate(zip(got["c_alone"], got["c_together"], dc)):
        print(f"  row {i}  c_alone={a:+8.3f}  c_batch={b:+8.3f}  "
              f"dc={d:+7.3f}  text_same={got['first_diff'][i] < 0}")
    print("[batch-gen] smoke complete.")
    return 0


def main(argv: Optional[list] = None) -> int:
    import argparse
    from research.monocle import patch_attribution as pa
    from research.monocle import single as sg
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="subway_safety",
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-rows", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    return smoke(a.case, a.n_rows, a.model_dir, a.device, a.max_new)


if __name__ == "__main__":
    raise SystemExit(main())
