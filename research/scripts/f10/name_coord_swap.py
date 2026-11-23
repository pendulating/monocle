#!/usr/bin/env python
"""F10.3: swap the lens coordinates of 2 place names, and read the same answer as F10.1.

The swap copies the clamp swap of the J++ source (`lens_evals/causal_evals/runner.py`).
For 2 tokens, V holds the unit vectors W_U[token] @ J_l, in 2 columns. A clean pass
gives the coordinates c = pinv(V) h of each position. At each band layer the edit
pins the running residual to the clean coordinates with the 2 entries exchanged:

    h <- h + V (scale * sigma(c_clean) - pinv(V) h)

The edit goes to the output of each band layer, at each position but position 0.
The readout is `c_yes_i` of F10.1, thus the model gets the same text before the image.

Warning: run this in float32. The edit on Gemma-4-31B is about 0.04% of the
residual norm, and the step of bfloat16 is about 0.4%. Rounding removes such an edit.

| Condition | Prompts | Edit |
|-----------|---------|------|
| `clean` | all 5 | none |
| `swap` | `manhattan`, `bronx`, `bare`; `brooklyn`, `queens` | the 2 names of the prompt pair |
| `swap2` | `manhattan`, `bronx` | the same, scale 2 |
| `swap_name` | `manhattan`, `bronx` | the same, only at the token of the name |
| `swap_logit` | `manhattan`, `bronx` | the rows of W_U, with no lens (the null basis of the J++ source) |
| `null` | `manhattan`, `bronx` | the name against an unrelated noun |
| `random` | `manhattan`, `bronx` | a fixed pair of random unit vectors |
| `placebo` | `manhattan`, `bronx` | `Brooklyn` and `Queens` |

`--heldout` runs the test of the removal hypothesis on 400 other images, on the prompts
`manhattan` and `bronx`, with the conditions `clean`, `swap`, `swap2`, `delete`, `add`,
`null`, `null2`, `rand_match`, `rand_match2` (runbook, the held-out test of removal).

| File in `--out-dir` | Content |
|---------------------|---------|
| `part-*.parquet` | 1 row for each image, prompt and condition |
| `coords-*.parquet` | the clean coordinates of the 2 names, for each image, prompt and band layer |
| `meta.json` | the band, the token ids, the cosine of each pair |

Rules: research/docs/facct-runbook.md, F10.3.

Use:
    python research/scripts/f10/name_coord_swap.py --model-key gemma-4-31b --model-dir DIR --lens FILE \
        --image-root DIR --out-dir data/f10/swap/gemma-4-31b
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch

__version__ = "1.3.0"

SEED = 20261030
BANDS = {"gemma-4-31b": [8, 16, 24, 32, 40, 48, 56], "qwen3.5-9b": [4, 8, 12, 16, 20, 24, 28]}
PROMPTS = {"manhattan": "in Manhattan", "bronx": "in the Bronx", "bare": "in Bronx",
           "brooklyn": "in Brooklyn", "queens": "in Queens"}
NAME_OF = {"manhattan": " Manhattan", "bronx": " Bronx", "bare": " Bronx", "brooklyn": " Brooklyn", "queens": " Queens"}
NULL_POOL = (" table", " window", " pencil", " carpet", " engine", " ladder", " button", " pillow", " basket", " mirror")
WORDS = ("Yes", "No", "yes", "no")
NEAR_PARALLEL = 0.999  # as the J++ source: a pinv of 2 such vectors is ill-conditioned


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def swapped_targets(clean: torch.Tensor, pinv: torch.Tensor, scale: float) -> torch.Tensor:
    """Give the clean coordinates [P, 2] with the 2 entries exchanged, times the scale."""
    return scale * (clean @ pinv.T).flip(-1)


def deleted_targets(clean: torch.Tensor, pinv: torch.Tensor) -> torch.Tensor:
    """Give the clean coordinates [P, 2] with the first entry at 0: the name goes."""
    c = clean @ pinv.T
    return torch.stack([torch.zeros_like(c[:, 0]), c[:, 1]], dim=1)


def added_targets(clean: torch.Tensor, pinv: torch.Tensor) -> torch.Tensor:
    """Give the clean coordinates [P, 2] with the first entry in both: the other name comes in."""
    c = clean @ pinv.T
    return torch.stack([c[:, 0], c[:, 0]], dim=1)


def matched_basis(basis: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Give 2 random vectors with the angle and the norms of basis [d, 2]: an edit keeps its size."""
    q, _ = torch.linalg.qr(torch.randn(basis.shape, generator=gen, dtype=basis.dtype))
    _, r = torch.linalg.qr(basis.cpu())
    return (q @ r).to(basis.device)


def clamp_hook(basis: torch.Tensor, pinv: torch.Tensor, targets: torch.Tensor, pos: torch.Tensor, ratios: list,
               along: torch.Tensor | None = None, log: list | None = None, replay: torch.Tensor | None = None):
    """Give the hook of 1 layer: pin the coordinates of h [..., d] on basis [d, 2] to targets, and change nothing else.

    `log` keeps the coefficients of the edit. `along` and `replay` apply such coefficients along
    other vectors of the same shape. A clamp moves the name coordinates, and the move carries to
    the next layer, thus the clamp gets smaller there. Random vectors do not move them. Only a
    replay gives the same size at each layer.
    """
    def hook(module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        new = tensor.clone()
        sel = new[0, pos].float()
        coef = replay if replay is not None else targets - sel @ pinv.T
        if log is not None:
            log.append(coef.detach())
        edited = sel + coef @ (along if along is not None else basis).T
        ratios.append(float((edited - sel).norm() / sel.norm()))
        new[0, pos] = edited.to(new.dtype)
        return new if torch.is_tensor(output) else (new, *output[1:])

    return hook


def question(arm: str) -> str:
    from research.scripts.f8.r3_readout import INSTRUCT

    return f"Is this place {PROMPTS[arm]} cool?{INSTRUCT['yes']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-key", required=True, choices=list(BANDS))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--items", default="data/f10/name_items.parquet")
    ap.add_argument("--n-per-cell", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--heldout", default=None, help="the F10.3 output dir: run the held-out test on images outside it")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    from research.monocle import jlens_read
    from research.scripts.f8.r3_readout import load_image

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    proc = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.float32, device_map=device,
                                                        attn_implementation="sdpa").eval()
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    tok = proc.tokenizer
    band = BANDS[args.model_key]
    lens = jlens_read.load_lens(args.lens)

    def token_id(s: str) -> int:
        t = tok.encode(s, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"{s!r} is {len(t)} tokens: {t}")
        return t[0]

    W_U = lens_model._lm_head.weight.detach().float()
    names = sorted(set(NAME_OF.values()))
    pool = [w for w in NULL_POOL if len(tok.encode(w, add_special_tokens=False)) == 1]
    ids = {w: token_id(w) for w in [*names, *pool]}
    answer = {w: token_id(w) for w in WORDS}
    Js = {l: lens.jacobians[l].float().to(device) for l in band}
    jpp = {w: {l: unit(W_U[i] @ Js[l]) for l in band} for w, i in ids.items()}
    raw = {w: {l: unit(W_U[ids[w]]) for l in band} for w in names}
    del Js
    gen = torch.Generator().manual_seed(SEED)
    rand = [{l: unit(torch.randn(W_U.shape[1], generator=gen)).to(device) for l in band} for _ in range(2)]

    def pair(a: dict, b: dict) -> dict:
        """Give the 2-column basis and its pinv for each band layer."""
        res = {}
        for l in band:
            if abs(float(a[l] @ b[l])) > NEAR_PARALLEL:
                raise RuntimeError(f"near-parallel vectors at layer {l}")
            V = torch.stack([a[l], b[l]], dim=1)
            res[l] = (V, torch.linalg.pinv(V))
        return res

    mb = pair(jpp[" Manhattan"], jpp[" Bronx"])
    bq = pair(jpp[" Brooklyn"], jpp[" Queens"])
    logit_mb = pair(raw[" Manhattan"], raw[" Bronx"])
    rand_pair = pair(rand[0], rand[1])
    gen_match = torch.Generator().manual_seed(SEED + 2)
    match = {l: matched_basis(mb[l][0], gen_match) for l in band}  # the 2 name vectors have about the same angle in both orders
    meta = {"band": band, "ids": ids, "answer": answer, "version": __version__,
            "cos_manhattan_bronx": {l: float(jpp[" Manhattan"][l] @ jpp[" Bronx"][l]) for l in band},
            "cos_brooklyn_queens": {l: float(jpp[" Brooklyn"][l] @ jpp[" Queens"][l]) for l in band}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("[f10.3]", json.dumps(meta), flush=True)

    def render(arm: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "text", "text": question(arm)}, {"type": "image"}]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    texts = {arm: render(arm) for arm in PROMPTS}

    def forward(inputs: dict, hooks: dict | None = None) -> torch.Tensor:
        handles = [lens_model.layers[l].register_forward_hook(h) for l, h in (hooks or {}).items()]
        try:
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False).logits[0, -1].float()
        finally:
            for h in handles:
                h.remove()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite logits")
        return logits

    def score(logits: torch.Tensor) -> dict:
        lp = torch.log_softmax(logits, dim=-1)
        r = {f"lp_{w}": float(lp[i]) for w, i in answer.items()}
        r["c_yes_i"] = r["lp_Yes"] - r["lp_No"]
        r["c_lc_i"] = r["lp_yes"] - r["lp_no"]
        r["top"] = tok.decode([int(logits.argmax())])
        return r

    def edit(inputs: dict, residuals: dict, basis: dict, pos: torch.Tensor, scale: float = 1.0, kind: str = "swap",
             logs: dict | None = None) -> dict:
        ratios: list = []
        hooks = {}
        for l, (V, P) in basis.items():
            clean = residuals[l][pos]
            t = {"swap": lambda: swapped_targets(clean, P, scale), "match": lambda: swapped_targets(clean, P, scale),
                 "delete": lambda: deleted_targets(clean, P), "add": lambda: added_targets(clean, P)}[kind]()
            if kind == "match":
                hooks[l] = clamp_hook(V, P, t, pos, ratios, along=match[l], replay=logs[scale][l][0])
            else:
                log = logs.setdefault(scale, {}).setdefault(l, []) if logs is not None else None
                hooks[l] = clamp_hook(V, P, t, pos, ratios, log=log)
        r = score(forward(inputs, hooks))
        r["edit_ratio"] = sorted(ratios)[len(ratios) // 2]
        return r

    items = pd.read_parquet(args.items)
    images = items.drop_duplicates(["case", "image_idx"])
    if args.heldout:
        used = pd.concat(pd.read_parquet(p, columns=["case", "image_idx"]) for p in Path(args.heldout).glob("part-*.parquet"))
        used = set(map(tuple, used.drop_duplicates().to_numpy()))
        images = images[[(c, i) not in used for c, i in zip(images["case"], images["image_idx"])]]
    images = images.groupby(["case", "borough"], group_keys=False).sample(args.n_per_cell, random_state=SEED + bool(args.heldout))
    parts = sorted(out.glob("part-*.parquet"))
    done = set()
    for p in parts:
        done |= set(map(tuple, pd.read_parquet(p, columns=["case", "image_idx"]).drop_duplicates().to_numpy()))
    todo = [r for r in images.itertuples(index=False) if (r.case, r.image_idx) not in done]
    print(f"[f10.3] {len(images)} images, {len(done)} done, {len(todo)} to do; {len(pool)} null words", flush=True)

    from jlens.hooks import ActivationRecorder

    rows, coords, n_part, t0, n_fwd = [], [], len(parts), time.perf_counter(), 0
    for n, r in enumerate(todo):
        img = load_image(args.image_root, r.sample_id)
        rng = random.Random(f"{r.case}-{r.image_idx}")
        for arm, text in texts.items():
            if args.heldout and arm not in ("manhattan", "bronx"):
                continue
            inputs = dict(proc(text=[text], images=[[img]], return_tensors="pt").to(device))
            seq = inputs["input_ids"][0]
            name_pos = (seq == ids[NAME_OF[arm]]).nonzero().flatten()
            if len(name_pos) != 1:
                raise RuntimeError(f"{arm}: {len(name_pos)} name tokens")
            all_pos = torch.arange(1, len(seq), device=device)
            with ActivationRecorder(lens_model.layers, at=band) as rec:
                base = score(forward(inputs))
                residuals = {l: rec.activations[l][0].detach().float() for l in band}
            key = {"model": args.model_key, "case": r.case, "image_idx": r.image_idx, "sample_id": r.sample_id,
                   "borough": r.borough, "arm": arm}
            rows.append({**key, "cond": "clean", **base, "edit_ratio": 0.0})
            n_fwd += 1
            basis = bq if arm in ("brooklyn", "queens") else mb
            conds = {"swap": (basis, all_pos, 1.0, "swap")}
            if args.heldout:
                other = " Bronx" if arm == "manhattan" else " Manhattan"
                named = pair(jpp[NAME_OF[arm]], jpp[other])
                word = rng.choice(pool)
                noun = pair(jpp[NAME_OF[arm]], jpp[word])
                conds = {"swap": (named, all_pos, 1.0, "swap"), "swap2": (named, all_pos, 2.0, "swap"),
                         "delete": (named, all_pos, 1.0, "delete"), "add": (named, all_pos, 1.0, "add"),
                         "null": (noun, all_pos, 1.0, "swap"), "null2": (noun, all_pos, 2.0, "swap"),
                         "rand_match": (named, all_pos, 1.0, "match"), "rand_match2": (named, all_pos, 2.0, "match")}
            elif arm in ("manhattan", "bronx"):
                word = rng.choice(pool)
                conds |= {"swap2": (mb, all_pos, 2.0, "swap"), "swap_name": (mb, name_pos, 1.0, "swap"),
                          "swap_logit": (logit_mb, all_pos, 1.0, "swap"),
                          "null": (pair(jpp[NAME_OF[arm]], jpp[word]), all_pos, 1.0, "swap"),
                          "random": (rand_pair, all_pos, 1.0, "swap"), "placebo": (bq, all_pos, 1.0, "swap")}
            if n == 0 and arm == "manhattan" and not parts:
                # A pin to the clean coordinates must give the clean answer.
                same = {l: (V, P) for l, (V, P) in mb.items()}
                ratios: list = []
                hooks = {l: clamp_hook(V, P, residuals[l][all_pos] @ P.T, all_pos, ratios) for l, (V, P) in same.items()}
                check = score(forward(inputs, hooks))
                print(f"[f10.3] check tokens {len(seq)}, name at {name_pos.tolist()}, bos "
                      f"{int((seq == tok.bos_token_id).sum()) if tok.bos_token_id is not None else 0}, clean {base['c_yes_i']:.4f}, "
                      f"identity clamp {check['c_yes_i']:.4f}, edit {max(ratios):.2e}", flush=True)
            logs: dict = {}
            for cond, (b, pos, scale, kind) in conds.items():
                keep = logs if args.heldout and cond in ("swap", "swap2") or kind == "match" else None
                rows.append({**key, "cond": cond, **edit(inputs, residuals, b, pos, scale, kind, keep)})
                n_fwd += 1
            for l in band:
                V, P = basis[l]
                c = residuals[l] @ P.T
                coords.append({**key, "layer": l, "first": float(c[name_pos, 0]), "second": float(c[name_pos, 1]),
                               "mean_abs_first": float(c[:, 0].abs().mean()), "mean_abs_second": float(c[:, 1].abs().mean())})
        if (n + 1) % args.chunk == 0 or n + 1 == len(todo):
            pd.DataFrame(rows).to_parquet(out / f"part-{n_part:05d}.parquet", index=False)
            pd.DataFrame(coords).to_parquet(out / f"coords-{n_part:05d}.parquet", index=False)
            rows, coords, n_part = [], [], n_part + 1
            dt = time.perf_counter() - t0
            print(f"[f10.3] {n + 1}/{len(todo)} images, {n_fwd / dt:.2f} forwards/s", flush=True)
    print("[f10.3] finished", flush=True)


if __name__ == "__main__":
    main()
