"""The single-image lens path: one image, one forced 2-point rating.

Why this path exists
--------------------
The pairwise path sends 2 images, and the lens then reads a DIFF of 2 maps.
Measured on the pairwise prompt, that is not a symmetric operation:

| Block | attention on A's patches | on B's patches |
|---|---|---|
| 42 | 0.12 | 0.26 |
| 46 | 0.12 | 0.24 |

The model reads image B broadly and reads image A mostly through A's final
patch. Which image comes first is OUR choice, so that asymmetry sits inside
every pairwise map. One image removes it.

It also removes the DIFF, the second bidirectional island, and about 450
prompt tokens, and the shorter prompt raises the image share of attention at
the blocks the lens reads.

This module does NOT replace `canonical`. The pairwise registry still needs
it, and `canonical.build_pair_inputs` is untouched.

The label contract
------------------
The lens reads ONE position: the one that emits the label. Thus a label must
be 1 token and 2 labels must not share a token. `monocle.labels` holds and
checks that rule. `Safe`, `Neutral`, `Dangerous`, `Good` and `Poor` all pass.
The digits 1 to 7 pass the rule and still fail, because 7-11% of image PATCHES
read a bare digit at L40-L43 — see `monocle/labels.py`.

The prompt
----------
The text comes from the vqa config, not from a copy kept here, so the
lens reads the prompt the run sends. That is the rule `canonical` follows.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import canonical  # noqa: E402
from monocle import labels as lb  # noqa: E402

#: The vqa prompt contract. One source of truth for the run and the lens.
PROMPT_YAML = REPO / "dagspaces/vqa/conf/prompt/rating_2point.yaml"

#: The forced JSON prefix, shared with the pairwise path so the answer
#: position sits in the same place.
FORCE_PREFIX = canonical.FORCE_PREFIX

#: Per case: what the image shows, the question, and the 3 labels.
#:
#: The subject and the question track the pairwise battery, which opens with
#: "Both show New York City ..." and then asks for the same attribute. The
#: labels are the natural poles of that attribute, and every one of them is a
#: single token — `python -m monocle.labels <labels>` checks a new set.
#:
#: subway_safety keeps Safe/Dangerous. The other 6 cases all ask about
#: quality, so they share Good/Poor and the contrast direction is the same
#: vector across them.
#:
#: There is no middle label. A 3-point form with `Neutral` gave
#: p(Neutral) = 1.000 on every image tested, so the output never varied and
#: no patch map could be checked against it. See the prompt config.
CASE_RATING: dict[str, dict[str, str]] = {
    "subway_safety": {
        "subject": "New York City subway station entrances",
        "question": "How safe does this station entrance look?",
        "positive": "Safe", "negative": "Dangerous"},
    "libraries": {
        "subject": "New York City public library buildings",
        "question": "How well maintained does this library building look?",
        "positive": "Good", "negative": "Poor"},
    "schools": {
        "subject": "New York City K-12 schools",
        "question": "How good a school does this look to be?",
        "positive": "Good", "negative": "Poor"},
    "road_quality": {
        "subject": "New York City roadways",
        "question": "What condition is this roadway in?",
        "positive": "Good", "negative": "Poor"},
    "parks_plazas": {
        "subject": "New York City parks or public plazas",
        "question": "How well maintained does this park or plaza look?",
        "positive": "Good", "negative": "Poor"},
    "restaurants": {
        "subject": "New York City restaurants",
        "question": "How appealing is this restaurant to eat at?",
        "positive": "Good", "negative": "Poor"},
    "street_photography": {
        "subject": "New York City blocks",
        "question": ("How appealing is this block for a street photography "
                     "photoshoot?"),
        "positive": "Good", "negative": "Poor"},
}


def rating(case: str) -> dict[str, str]:
    if case not in CASE_RATING:
        raise ValueError(
            f"unknown case {case!r}; expected one of {list(CASE_RATING)}")
    return dict(CASE_RATING[case])


def prompt_text(case: str, overrides: Optional[dict] = None) -> str:
    """The user text for one case, rendered from the vqa config."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(PROMPT_YAML)
    cfg.merge_with(OmegaConf.create(rating(case)))
    if overrides:
        cfg.merge_with(OmegaConf.create(dict(overrides)))
    whole = OmegaConf.create({"prompt": cfg})
    return OmegaConf.to_container(whole, resolve=True)["prompt"]["user_template"]


#: The production order, and the 3 probe orders `modality_order` adds.
#:
#: `neutral_first` and `question_first` are a PAIR. Both hold a filler block
#: and a question block, and both put the image between them. They differ
#: only in which block comes first, so the patch positions, the distance from
#: the patches to the answer, and the total length are all identical. Their
#: difference is thus the question-conditioning effect alone.
LAYOUTS = ("image_first", "text_first", "neutral_first", "question_first")

#: Layouts that need a length-matched filler block.
FILLER_LAYOUTS = ("neutral_first", "question_first")


def build_messages(
    case: str, overrides: Optional[dict] = None,
    layout: str = "image_first", filler: Optional[str] = None,
) -> list[dict]:
    """One user turn. The block order is the variable.

    | Layout | Order | The question reaches the patches |
    |---|---|---|
    | `image_first` | image, question | no |
    | `text_first` | question, image | yes |
    | `neutral_first` | filler, image, question | no |
    | `question_first` | question, image, filler | yes |

    `image_first` is production. The image comes first because the gemma-4
    card asks for that order, and because a single image needs no label text
    to separate it from a second one. This is the single-image form of
    `images_then_labels`.

    A patch attends to every token BEFORE it and to no token after it. So the
    2 layouts that put the question first let the question shape the patch
    residual, and the other 2 do not.

    **Warning:** only `image_first` is canonical. 3 gates demand
    `images_then_labels`, so no other layout may enter the registry.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
    if layout in FILLER_LAYOUTS and not filler:
        raise ValueError(f"layout {layout!r} needs a filler block")
    image = {"type": "image"}
    text = {"type": "text", "text": prompt_text(case, overrides)}
    pad = {"type": "text", "text": filler} if filler else None
    if layout == "image_first":
        content = [image, text]
    elif layout == "text_first":
        content = [text, image]
    elif layout == "neutral_first":
        content = [pad, image, text]
    else:
        content = [text, image, pad]
    return [{"role": "user", "content": content}]


def render_single_text(
    proc: Any, tmpl: str, case: str, *,
    overrides: Optional[dict] = None,
    force_prefix: str = FORCE_PREFIX,
    enable_thinking: bool = False,
    layout: str = "image_first",
    filler: Optional[str] = None,
) -> str:
    """The prompt STRING, so a caller can append a reasoning trace to it.

    `enable_thinking` picks between the 2 forms the gemma-4 template builds:

    | Mode | The generation prompt ends with | The model then writes |
    |---|---|---|
    | off | `<|turn>model\n<|channel>thought\n<channel|>` | the answer |
    | on | `<|turn>model\n` | a thought, then the answer |

    Off does NOT skip the thought channel. It opens one and closes it at
    once, so the model has no room to reason. On leaves it open and also puts
    `<|think|>` in a system turn at the top.

    Thus a thinking read cannot append the forced prefix here: the prefix
    would land INSIDE the open thought. Generate first, then append the
    closed thought and the prefix, and read at that position.
    """
    kw = {"enable_thinking": True} if enable_thinking else {}
    tok = getattr(proc, "tokenizer", proc)
    return tok.apply_chat_template(
        build_messages(case, overrides, layout=layout, filler=filler),
        tokenize=False,
        add_generation_prompt=True, chat_template=tmpl, **kw) + force_prefix


def build_single_inputs(
    proc: Any, tmpl: str, image: Any, case: str, *,
    overrides: Optional[dict] = None,
    force_prefix: str = FORCE_PREFIX,
    device: str = "cuda:0",
    enable_thinking: bool = False,
    text: Optional[str] = None,
    layout: str = "image_first",
    filler: Optional[str] = None,
) -> dict:
    """Model inputs for ONE image on the single-image prompt path.

    The forced JSON prefix is appended, exactly as `canonical.render_chat_text`
    does it. Without it the last position is the end of the generation prompt,
    where the model's natural next token is `{`, and a Safe-against-Dangerous
    contrast read there is a DISPOSITION, not the answer. With it the last
    position emits the label itself.

    Pass `force_prefix=""` to read the earlier position on purpose.

    Pass `text` to send a prompt string the caller built, which is how a
    thinking read re-feeds its own reasoning trace.
    """
    if text is None:
        text = render_single_text(
            proc, tmpl, case, overrides=overrides, force_prefix=force_prefix,
            enable_thinking=enable_thinking, layout=layout, filler=filler)
    return proc(text=[text], images=[[image]],
                return_tensors="pt").to(device)


def contrast_ids(
    tokenizer: Any, case: str, overrides: Optional[dict] = None,
) -> tuple[int, int]:
    """(positive id, negative id). Raises unless the label contract holds."""
    r = rating(case)
    if overrides:
        r.update({k: v for k, v in overrides.items() if k in r})
    ids = lb.check_label_tokens(tokenizer, [r["positive"], r["negative"]])
    return ids[r["positive"]], ids[r["negative"]]


def photograph_id(path) -> str:
    """A name that is unique to one photograph.

    A Cyclomedia face sits at `.../<unit>/faces/<face>.jpg`, so the file stem
    is only the face letter. 4 photographs of 4 different units all read `B`,
    and a panel written by that name writes over the panel before it. Put the
    unit in front of the face. Any other layout keeps the stem.
    """
    p = Path(str(path))
    if p.parent.name == "faces":
        return f"{p.parent.parent.name}_{p.stem}"
    return p.stem


def load_single_rows(
    n_images: int, case: str, kind: str = "proxy", *, seed: int = 777,
) -> list[dict]:
    """Single images drawn from the SAME pool the pairwise battery uses.

    Each image is then read ALONE. The pair rows are only a catalogue of
    photographs here: the sampler builds a pair from 2 different units drawn
    by a random permutation, so taking both photographs of a pair is the same
    as drawing 2 photographs from 2 random units.

    **The `image_id` names the PHOTOGRAPH, not a slot.** An id like
    `<pair>_A` would say these images were read as a pair, and they are not.
    `photograph_id` builds the name, because a bare file stem collides.
    `pair_id` and `slot` stay as provenance columns for tracing back to the
    registry.

    Duplicates are dropped. A unit can recur across pairs, and the same
    photograph twice would let a donor be its own target, which reads as an
    empty map.
    """
    from monocle import safety_workspace as sw

    rows: list[dict] = []
    seen: set[str] = set()
    # Each pair yields 2 photographs, and some are dropped as duplicates, so
    # ask for extra and stop when the quota is met.
    for pair in sw.load_pairs(n_images, case, kind, seed=seed):
        left, right = canonical.presented_images(pair)
        pid = str(pair["pair_id"])
        for slot, path in (("A", left), ("B", right)):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"image_id": photograph_id(key), "pair_id": pid,
                         "slot": slot, "path": path, "case": case})
            if len(rows) >= n_images:
                return rows
    return rows


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def smoke(case: str, n_images: int, model_dir: str, device: str) -> int:
    """Prove the path end to end: geometry, labels, and a real answer."""
    import torch
    from PIL import Image

    from monocle import attn_route as ar
    from monocle import safety_workspace as sw

    proc, model, tmpl = ar.load_eager(model_dir, device)
    tok = proc.tokenizer
    pos_id, neg_id = contrast_ids(tok, case)
    n_suffix = len(tok.encode(FORCE_PREFIX, add_special_tokens=False))
    image_token_id = int(model.config.image_token_id)
    print(f"[single] case={case} labels={rating(case)}")
    print(f"[single] token ids: positive={pos_id} negative={neg_id}")

    for row in load_single_rows(n_images, case):
        img = Image.open(row["path"]).convert("RGB")
        inputs = build_single_inputs(proc, tmpl, img, case, device=device)
        seq_len = int(inputs["input_ids"].shape[1])
        blocks = sw.image_token_blocks(inputs, image_token_id, expect=1)
        answer_pos = sw.read_positions(seq_len, n_suffix)["label"]
        with torch.no_grad():
            out = model(**inputs, use_cache=False, logits_to_keep=1)
        lg = out.logits[0, -1, :].float()
        p = torch.softmax(lg, dim=-1)
        c = float(lg[pos_id] - lg[neg_id])
        top = int(lg.argmax())
        print(f"  {row['image_id']}  seq={seq_len}  blocks={len(blocks)}  "
              f"patches={[len(b) for b in blocks]}  answer_pos={answer_pos}")
        print(f"    p(pos)={float(p[pos_id]):.3f} "
              f"p(neg)={float(p[neg_id]):.3f}  c={c:+.3f}  "
              f"argmax={tok.convert_ids_to_tokens([top])[0]!r}")
        if len(blocks) != 1:
            raise RuntimeError(
                f"expected 1 image block, got {len(blocks)} — the single path "
                "must build exactly 1 bidirectional island")
    print("[single] smoke complete.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="subway_safety",
                    choices=list(CASE_RATING))
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--model-dir",
                    default="/share/ju/matt/zoo/models/gemma-4-12B-it")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    return smoke(a.case, a.n_images, a.model_dir, a.device)


if __name__ == "__main__":
    raise SystemExit(main())
