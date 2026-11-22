#!/usr/bin/env python
"""F10.4: read the whole vocabulary at the text positions after the image, with a J++ lens.

The image comes first, thus each text position can see it. The decoder is causal: with
the question first, the positions of the question cannot see the image. At each position
of `POSITIONS`, the lens gives 1 logit for each token of the vocabulary. A read with no
image gives the baseline of the same text. A word that the image adds is a candidate hop.

| File in `--out-dir` | Content |
|---------------------|---------|
| `part-*.parquet` | 1 row for each image, prompt, position and layer: the top tokens, the top tokens that the image adds, and `yes_no` of the prompt |
| `means.pt` | the mean logit of each token, for each (prompt, position, case, borough), primary layer only |
| `means_depth.pt` | the same for each (prompt, position, case, borough, layer), all layers |
| `text.pt` | the same read with no image: 1 vector for each (prompt, position, layer) |

Rules: vlm-narratives-docs/facct-runbook.md, F10.4.

Use:
    python research/scripts/f10/hop_read.py --model-key gemma-4-31b --model-dir DIR --lens FILE \
        --image-root DIR --out-dir data/f10/hops/gemma-4-31b --n-per-cell 50
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
sys.path[:0] = [str(ROOT), str(ROOT / "sub/jacobian-lens"), str(ROOT / "research/scripts/f8"), str(ROOT / "research/scripts/f10")]

from word_read import LAYERS, PROMPTS as F9  # noqa: E402

__version__ = "1.1.0"

SEED = 20261102
PRIMARY = {"gemma-4-31b": 45, "qwen3.5-9b": 24}
TOP = 30
COOL = "Is this place cool?"
ANSWER = " Answer with one word: yes or no."
SPIDER = "Fact: The number of legs on the animal that spins webs is "
OTHER = "Fact: The number of wheels on the vehicle that carries many passengers is "
# prompt -> (user text, text of the model turn so far, position names)
# `place` is the token ` place`, `next` is the token after it, `is` is the last token, `final` is the last token.
POSITIONS = {"cool": (COOL + ANSWER, "", ("place", "next", "final")),
             "open": (COOL, "This place is", ("place", "is")),
             "f9": (None, "", ("final",))}


def locate(seq: list[int], place: int, names: tuple[str, ...]) -> dict[str, int]:
    """Give the index of each named position in seq, from the last ` place` token."""
    found = [i for i, t in enumerate(seq) if t == place]
    last = len(seq) - 1
    out = {}
    for name in names:
        if name in ("final", "is"):
            out[name] = last
        elif not found:
            raise RuntimeError("no ` place` token")
        else:
            out[name] = found[-1] + (name == "next")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", required=True, choices=list(LAYERS))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-cell", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--heldout", default=None, help="a pilot output dir: use other images, and keep a random fold in `means.pt`")
    args = ap.parse_args()

    from jlens.hooks import ActivationRecorder
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from research.monocle import jlens_read
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
    layers, primary = LAYERS[args.model_key], PRIMARY[args.model_key]
    place = tok.encode(" place", add_special_tokens=False)
    if len(place) != 1:
        raise ValueError(f"` place` is {len(place)} tokens")
    place = place[0]
    yes, no = (tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))

    def render(prompt: str, case: str, image: bool, first: bool = True) -> str:
        user, prefill, _ = POSITIONS[prompt]
        text = {"type": "text", "text": F9[case] if user is None else user}
        content = ([{"type": "image"}, text] if first else [text, {"type": "image"}]) if image else [text]
        msgs = [{"role": "user", "content": content}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefill

    def read(inputs: dict, at: dict[str, int]) -> tuple[dict, torch.Tensor]:
        """Give {(position, layer): logits [vocab]} and the logits of the model at the last position."""
        pos = torch.tensor(list(at.values()), device=device)
        with ActivationRecorder(lens_model.layers, at=layers) as rec, torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0, -1].float()
            got = {}
            for l in layers:
                x = lens_model.unembed(lens.transport(rec.activations[l][0, pos].float(), l)).float()
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"non-finite lens logits at layer {l}")
                got |= {(name, l): x[i] for i, name in enumerate(at)}
        return got, logits

    def top(x: torch.Tensor, k: int = 12) -> list[str]:
        return [tok.decode([int(i)]) for i in x.topk(k).indices]

    # The positive control of the source: the lens must name the hop that the text hides.
    # Raw lens logits can carry a fixed bias on some tokens, thus the second line removes the read of a similar prompt.
    spider, other = (dict(tok(t, return_tensors="pt").to(device)) for t in (SPIDER, OTHER))
    got, logits = read(spider, {"final": spider["input_ids"].shape[1] - 1})
    base, _ = read(other, {"final": other["input_ids"].shape[1] - 1})
    print(f"[f10.4] spider, first token {tok.decode(spider['input_ids'][0, :1])!r}, model top {top(logits, 5)}", flush=True)
    for l in layers:
        print(f"[f10.4] spider L{l}: {top(got['final', l])}", flush=True)
        print(f"[f10.4] spider - other L{l}: {top(got['final', l] - base['final', l])}", flush=True)

    # The same reads with no image: the baseline of each prompt.
    text_read = {}
    for prompt, (_, _, names) in POSITIONS.items():
        for case in F9:
            inputs = dict(proc(text=[render(prompt, case, False)], return_tensors="pt").to(device))
            seq = inputs["input_ids"][0].tolist()
            got, _ = read(inputs, locate(seq, place, names))
            text_read |= {(prompt, case, n, l): v for (n, l), v in got.items()}
            if prompt != "f9" or case == "street":
                print(f"[f10.4] text only, {prompt}, {case}: {tok.decode(seq[-14:])!r}", flush=True)
    torch.save({k: v.cpu() for k, v in text_read.items()}, out / "text.pt")

    items = pd.read_parquet("data/f10/name_items.parquet", columns=["case", "image_idx", "sample_id", "borough"]).drop_duplicates(["case", "image_idx"])
    if args.heldout:
        used = pd.concat(pd.read_parquet(p, columns=["case", "image_idx"]) for p in Path(args.heldout).glob("part-*.parquet"))
        used = set(map(tuple, used.drop_duplicates().to_numpy()))
        items = items[[(c, i) not in used for c, i in zip(items["case"], items["image_idx"])]]
    images = items.groupby(["case", "borough"], group_keys=False).sample(args.n_per_cell, random_state=SEED + bool(args.heldout))
    # A random half of each case and borough: the 2 halves differ by no property of the images.
    fold = images.groupby(["case", "borough"])["image_idx"].transform(lambda x: x.sample(frac=1, random_state=SEED).rank(method="first") % 2)
    images = images.assign(fold=fold.astype(int))
    parts = sorted(out.glob("part-*.parquet"))
    if parts:
        raise RuntimeError("this pilot does not resume: the means cover 1 pass. Use a new dir")
    (out / "meta.json").write_text(json.dumps({"layers": layers, "primary": primary, "version": __version__, "top": TOP}) + "\n")
    print(f"[f10.4] {len(images)} images, layers {layers}, primary {primary}", flush=True)

    # With the question first, the question cannot see the image: 2 images must give the same read.
    pair = [load_image(args.image_root, s) for s in images["sample_id"].iloc[:2]]
    case = images["case"].iloc[0]
    seen = []
    for img in pair:
        inputs = dict(proc(text=[render("cool", case, True, first=False)], images=[[img]], return_tensors="pt").to(device))
        seq = inputs["input_ids"][0].tolist()
        seen.append(read(inputs, locate(seq, place, ("place",)))[0]["place", primary])
    print(f"[f10.4] question first, ` place`, 2 images: max logit gap {float((seen[0] - seen[1]).abs().max()):.4f}", flush=True)

    rows, n_part, t0 = [], 0, time.perf_counter()
    sums: dict = {}
    depth: dict = {}
    for n, r in enumerate(images.itertuples(index=False)):
        img = load_image(args.image_root, r.sample_id)
        for prompt, (_, _, names) in POSITIONS.items():
            inputs = dict(proc(text=[render(prompt, r.case, True)], images=[[img]], return_tensors="pt").to(device))
            seq = inputs["input_ids"][0].tolist()
            at = locate(seq, place, names)
            if n == 0:
                print(f"[f10.4] check {prompt}: tokens {len(seq)}, "
                      f"{ {k: tok.decode([seq[i]]) for k, i in at.items()} }, image before: {seq.index(int(model.config.image_token_id)) < min(at.values())}", flush=True)
            got, logits = read(inputs, at)
            lp = torch.log_softmax(logits, dim=-1)
            for (name, l), x in got.items():
                v, i = x.topk(TOP)
                av, ai = (x - text_read[prompt, r.case, name, l]).topk(TOP)
                rows.append({"case": r.case, "image_idx": r.image_idx, "sample_id": r.sample_id, "borough": r.borough, "prompt": prompt,
                             "position": name, "layer": l, "yes_no": float(lp[yes] - lp[no]), "top_id": i.tolist(), "top_logit": v.tolist(),
                             "added_id": ai.tolist(), "added_logit": av.tolist()})
                s, c = depth.get((prompt, name, r.case, r.borough, l), (0, 0))
                depth[prompt, name, r.case, r.borough, l] = (s + x.double().cpu(), c + 1)
                if l == primary:
                    key = (prompt, name, r.case, r.borough, r.fold) if args.heldout else (prompt, name, r.case, r.borough)
                    s, c = sums.get(key, (0, 0))
                    sums[key] = (s + x.double().cpu(), c + 1)
        if (n + 1) % args.chunk == 0 or n + 1 == len(images):
            pd.DataFrame(rows).to_parquet(out / f"part-{n_part:05d}.parquet", index=False)
            rows, n_part = [], n_part + 1
            print(f"[f10.4] {n + 1}/{len(images)} images, {(n + 1) / (time.perf_counter() - t0):.2f} images/s", flush=True)
    torch.save({k: (s / c).float() for k, (s, c) in sums.items()}, out / "means.pt")
    torch.save({k: (s / c).float() for k, (s, c) in depth.items()}, out / "means_depth.pt")
    print("[f10.4] finished", flush=True)


if __name__ == "__main__":
    main()
