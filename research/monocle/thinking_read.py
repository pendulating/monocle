"""Read the verdict AFTER the model reasons, not instead of it.

The 2 forms of the same prompt
------------------------------
The gemma-4 template builds 2 different generation prompts from one set of
messages. `enable_thinking` picks between them:

| Mode | The prompt ends with | The model then writes |
|---|---|---|
| off | `<|turn>model\\n<|channel>thought\\n<channel|>` | the answer |
| on | `<|turn>model\\n` | a thought, then the answer |

Off is NOT "no thought channel". It opens a channel and closes it in the same
breath, so the model has nowhere to reason and goes straight to the label.

Why a thinking read needs 2 passes
----------------------------------
With the channel open, the forced JSON prefix cannot go on the end of the
prompt: it would land INSIDE the thought, and the position we read would not
be the one that emits the label. So:

1. generate with the channel open, and let the model close it;
2. append the closed thought to the prompt, then the forced prefix;
3. read `c` at the last position, in ONE forward pass.

Step 3 makes the thinking read the same measurement as the plain one: an
fp32 contrast at the token that emits the label. The only difference is that
the model has reasoned first, and the reasoning is in the context.

What can go wrong, and is recorded
----------------------------------
The model may never close the channel inside `max_new`. Then there IS no
answer position, and a read would be at an arbitrary place in a half-written
thought. `closed` is False for such a row, and the caller must drop it rather
than treat the number as a verdict.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import single as sg  # noqa: E402

#: What the template writes to close a thought channel.
CHANNEL_CLOSE = "<channel|>"
#: What the model writes to open one, when the prompt leaves it to the model.
CHANNEL_OPEN = "<|channel>thought"
#: A trace longer than this is a runaway, not a thought about one photograph.
MAX_NEW = 384


def split_thought(generated: str) -> tuple[str, bool]:
    """(the thought block, whether the model closed it).

    The block keeps its closing marker, because the prompt for the read must
    contain a CLOSED channel. A trace with no marker is not usable, and the
    caller is told so rather than being handed a number.
    """
    at = generated.find(CHANNEL_CLOSE)
    if at < 0:
        return generated, False
    return generated[:at + len(CHANNEL_CLOSE)], True


def reasoning_text(block: str) -> str:
    """The words the model reasoned with, without the channel markers."""
    body = block
    at = body.find(CHANNEL_OPEN)
    if at >= 0:
        body = body[at + len(CHANNEL_OPEN):]
    if body.endswith(CHANNEL_CLOSE):
        body = body[:-len(CHANNEL_CLOSE)]
    return body.strip()


def generate_thought(
    model: Any, proc: Any, inputs: dict, max_new: int = MAX_NEW,
) -> str:
    """Greedy, so a trace is a property of the model and not of a seed."""
    n = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=False, use_cache=True)
    tok = getattr(proc, "tokenizer", proc)
    return tok.decode(out[0, n:], skip_special_tokens=False)


def open_prompt(proc: Any, tmpl: str, case: str) -> str:
    """The prompt with the thought channel LEFT OPEN.

    It does not depend on the photograph, so a batched caller renders it once
    and reuses it for every row.
    """
    return sg.render_single_text(
        proc, tmpl, case, force_prefix="", enable_thinking=True)


def masked_inputs(
    proc: Any, tmpl: str, image: Any, case: str, text: str, *,
    device: str, drop_patches: Optional[list], image_token_id: Optional[int],
) -> dict:
    """One row for `text`, with `drop_patches` taken out of the photograph."""
    row = sg.build_single_inputs(
        proc, tmpl, image, case, device=device, text=text)
    if drop_patches:
        row = pab.ablate_row(
            row, pab.slots_of(row, image_token_id), drop_patches)
    return row


def read_after_thought(
    model: Any, proc: Any, lens_model: Any, tmpl: str, image: Any, case: str,
    open_text: str, block: str, pos_id: int, neg_id: int, *,
    device: str = "cuda:0", drop_patches: Optional[list] = None,
    image_token_id: Optional[int] = None,
) -> float:
    """`c` at the label position, with a CLOSED thought in the context.

    The mask must match the one the thought was written under, or the model
    reasons about one picture and rates another.
    """
    read_inputs = masked_inputs(
        proc, tmpl, image, case, open_text + block + sg.FORCE_PREFIX,
        device=device, drop_patches=drop_patches,
        image_token_id=image_token_id)
    return float(pab.contrast(model, lens_model, read_inputs,
                              pos_id, neg_id)[0])


def describe_thought(proc: Any, block: str, closed: bool) -> dict:
    """The record kept for one trace."""
    return {"closed": closed, "reasoning": reasoning_text(block),
            "n_thought_tokens": len(
                getattr(proc, "tokenizer", proc).encode(
                    block, add_special_tokens=False))}


def thinking_read(
    model: Any, proc: Any, lens_model: Any, tmpl: str, image: Any, case: str,
    pos_id: int, neg_id: int, *, device: str = "cuda:0",
    max_new: int = MAX_NEW, drop_patches: Optional[list] = None,
    image_token_id: Optional[int] = None,
) -> dict:
    """`c` at the label position, after the model has reasoned.

    `drop_patches` masks the photograph BEFORE the model reasons, so the
    thought is about the cells that are left. The mask is applied to both
    passes, or the model would reason about one picture and rate another.

    This is the 1-row path. A battery uses `open_prompt`, a batched
    generation, then `read_after_thought`, which is the same 3 steps with the
    slow one done many rows at a time.
    """
    if drop_patches and image_token_id is None:
        raise ValueError(
            "a masked read needs `image_token_id`, so that BOTH passes mask "
            "the same cells")

    open_text = open_prompt(proc, tmpl, case)
    gen_inputs = masked_inputs(
        proc, tmpl, image, case, open_text, device=device,
        drop_patches=drop_patches, image_token_id=image_token_id)
    block, closed = split_thought(
        generate_thought(model, proc, gen_inputs, max_new))
    c = read_after_thought(
        model, proc, lens_model, tmpl, image, case, open_text, block,
        pos_id, neg_id, device=device, drop_patches=drop_patches,
        image_token_id=image_token_id)
    return {"c_think": c, **describe_thought(proc, block, closed)}


# ---------------------------------------------------------------------------
# Smoke test: does the model close the channel, and does the read move?
# ---------------------------------------------------------------------------
def smoke(case: str, n_images: int, model_dir: str, device: str,
          max_new: int) -> int:
    """Prove the 2-pass read before a battery is launched.

    The unknown is whether the model closes its thought inside `max_new`. If
    it does not, there is no answer position and the whole readout is void.
    """
    from PIL import Image

    from research.monocle import attn_route as ar
    from research.monocle import jlens_read

    proc, model, tmpl = ar.load_eager(model_dir, device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    pos_id, neg_id = sg.contrast_ids(proc.tokenizer, case)
    print(f"[think] case={case} pos={pos_id} neg={neg_id} max_new={max_new}")

    for r in sg.load_single_rows(n_images, case, seed=20260821):
        img = Image.open(r["path"]).convert("RGB")
        plain = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        c_plain = float(pab.contrast(model, lens_model, plain,
                                     pos_id, neg_id)[0])
        got = thinking_read(model, proc, lens_model, tmpl, img, case,
                            pos_id, neg_id, device=device, max_new=max_new)
        print(f"\n  {r['image_id']}  c_plain={c_plain:+.3f}  "
              f"c_think={got['c_think']:+.3f}  closed={got['closed']}  "
              f"thought_tokens={got['n_thought_tokens']}")
        print(f"    reasoning: {got['reasoning'][:400]!r}")
    print("\n[think] smoke complete.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    from research.monocle import patch_attribution as pa
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="subway_safety",
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    return smoke(a.case, a.n_images, a.model_dir, a.device, a.max_new)


if __name__ == "__main__":
    raise SystemExit(main())
