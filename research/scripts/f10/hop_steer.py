#!/usr/bin/env python
"""F10.29, items 5 and 7: does the shared aesthetic direction move the answer, and what does it say at other tokens?

Stages:
  dirs   (CPU) Build the residual-space directions of F10.26 task 7 again at 1 layer for each position, orient them
         so that + is the `nice` side, and measure the dose unit sigma = std over the 591 images of the projection.
         Also a random control direction. Output: `data/f10/steer/{model}/dirs.npz`, `dirs.json`.
  run    (GPU) For 3 prompts (cool, nice, ugly with `place`) and 1 image for each NTA (197), add alpha * sigma * u
         to the residual at 1 layer with a forward hook, and read the 8-form margin and the first answer token.
         Variants: aesthetic direction at the noun token only; the same at the noun and every later token;
         the judgment direction (last token, late layer) at the last token; a random control at the noun.
         Then greedy generation of 16 tokens for 10 images at alpha -6, 0, +6 (aesthetic, noun onward).
         Output: `data/f10/steer/{model}/margins-{prompt}.parquet`, `generations.parquet`.
  words  (GPU) Text-only prompts with the word `person`, `building`, `place`, `scene` as the noun, or `beautiful`,
         `trendy`, `subjective`, `cool` as the adjective. Read the residual of that token at the direction's layer
         through the J++ lens, with and without alpha * sigma * u added. Output: `data/f10/steer/{model}/words.parquet`.
  words_model (GPU) The same prompts, but the model's own forward: add alpha * sigma * u at the word token (aesthetic)
         or at the last token (judgment) and read the model's own next-token words at the word token and at the last
         token, and the 8-form margin. The lens is linear, thus `words` gives the same word shift at every token;
         this stage lets the later layers act. Output: `words_model.parquet`.
  patch_dirs (CPU) F10.35: directions on the mean image patch (shared, cool, pretty, ugly) at L16 and L32.
  patches (GPU) F10.35: add a patch direction to all 256 image patches, read the patch words and the answers.
  dose   (GPU) Larger doses on 40 images: alpha in -48 to 48 for the aesthetic direction (noun onward) at 4 layers
         from early to the best layer, the judgment direction (last token), and the control. Output: `dose-{prompt}.parquet`.

The hook adds the vector to the output of block `layer`, as `research/monocle/jlens_steer.py` does. One hook is live at a
time, and a `finally` removes it. The model runs in float32, as the F10 stores did.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_steer.py dirs --model gemma-4-31b
    sbatch --export=ALL,F10_MODEL=gemma-4-31b --job-name=f1029-gemma-4-31b research/scripts/f10/hop_steer.sbatch
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "sub/jacobian-lens"), str(ROOT / "research/scripts/f8"), str(ROOT / "research/scripts/f10")]

__version__ = "1.0.0"
OUT = ROOT / "data/f10/steer"
MODELS = ("gemma-4-31b", "qwen3.5-9b", "qwen3.6-27b")
# The layer of each direction: the best leave-1-prompt-out layer of F10.26 task 7 (`t7_loo.csv`, k = 1).
LAYERS = {"gemma-4-31b": {"place": 32, "final": 44}, "qwen3.5-9b": {"place": 28, "final": 28}, "qwen3.6-27b": {"place": 48, "final": 60}}
# More layers of the aesthetic direction for the dose stage: mid depth and earlier, where later blocks can still act.
PLACE_LAYERS = {"gemma-4-31b": (8, 16, 24, 32), "qwen3.5-9b": (8, 12, 20, 28), "qwen3.6-27b": (16, 24, 36, 48)}
ALPHAS = (-6.0, -3.0, 3.0, 6.0)
PROMPTS = (("cool", "place"), ("nice", "place"), ("ugly", "place"))
VARIANTS = (("aesthetic", "place", "noun"), ("aesthetic", "place", "tail"), ("judgment", "final", "final"), ("control", "place", "noun"))
GEN_IMAGES = 10
GEN_TOKENS = 16
GEN_ALPHAS = (-6.0, 0.0, 6.0)
NOUN_WORDS = ("person", "building", "place", "scene")
ADJ_WORDS = ("beautiful", "trendy", "subjective", "cool")
TOP = 20
SEED = 7


def dirs(model: str) -> None:
    """Build the 2 directions and the control on the CPU, from the F10.26 summaries."""
    from research.scripts.f10.hop_prompts import PROMPTS as ALL14, Summary, dual_weights, prompt_name, shared_direction

    out = OUT / model
    out.mkdir(parents=True, exist_ok=True)
    A = {(a, n): Summary(model, a, n) for a, n in ALL14}
    ref = A[("cool", "place")]
    save, info = {}, {"version": __version__, "model": model, "layers": LAYERS[model], "d": None}
    cells = [(pos, layer) for pos, layer in LAYERS[model].items()] + [("place", l) for l in PLACE_LAYERS[model] if l != LAYERS[model]["place"]]
    info["place_layers"] = {}
    for pos, layer in cells:
        b = ref.layers.index(layer)
        cols, names = [], []
        for a, n in ALL14:
            S = A[(a, n)]
            X = S.residual(pos)[:, b].astype(np.float64)
            X = X - X.mean(0, keepdims=True)
            y = S.margin()
            y = (y - np.nanmean(y)) / np.nanstd(y)
            _, w, _ = dual_weights(X, y, np.flatnonzero(np.isfinite(y)))
            cols.append(w)
            names.append(prompt_name(a, n))
            S.drop(f"residual_{pos}")
        W = np.column_stack(cols)
        u, share = shared_direction(W, 1)
        u = u[:, 0]
        if u @ W[:, names.index(prompt_name("nice", "place"))] < 0:
            u = -u
        X = ref.residual(pos)[:, b].astype(np.float64)
        proj = (X - X.mean(0, keepdims=True)) @ u
        sigma = float(proj.std())
        ref.drop(f"residual_{pos}")
        cell = {"layer": layer, "sigma": sigma, "share_sv1": float(share[0]), "residual_norm_median": float(np.median(np.linalg.norm(X, axis=1))),
                "loadings": {names[j]: float(W[:, j] @ u) for j in range(len(names))}}
        if layer == LAYERS[model][pos]:
            save[f"{pos}_u"] = u.astype(np.float32)
            info[pos] = cell
        if pos == "place":
            save[f"place_L{layer}_u"] = u.astype(np.float32)
            info["place_layers"][str(layer)] = cell
        info["d"] = int(len(u))
        print(f"[f10.29] {model} {pos} L{layer}: sigma {sigma:.3f}, residual norm {cell['residual_norm_median']:.1f}, loadings nice {cell['loadings']['place nice']:.2f} ugly {cell['loadings']['place ugly']:.2f}", flush=True)
    rng = np.random.default_rng(SEED)
    c = rng.standard_normal(info["d"])
    save["control_u"] = (c / np.linalg.norm(c)).astype(np.float32)
    np.savez(out / "dirs.npz", **save)
    (out / "dirs.json").write_text(json.dumps(info, indent=1))


PATCH_LAYERS = {"gemma-4-31b": (16, 32)}  # the layers at which `patches` adds a patch direction
PATCH_ADJ = ("cool", "pretty", "ugly")  # the adjectives with their own patch direction (noun `place`)
PATCH_FOLDS = 5


def answer_margin(model: str, adjective: str, noun: str, image_idx: np.ndarray) -> np.ndarray:
    """Give the 8-form logit margin of each image. It is finite also where the model refuses."""
    from scipy.special import logsumexp

    from research.scripts.f10.hop_hierarchy import answer_store

    a = pd.read_parquet(ROOT / "data/f10/hop_logit" / answer_store(model, adjective, noun) / "answers.parquet").set_index("image_idx")
    yes, no = ["lp_Yes", "lp_yes", "lp_ Yes", "lp_ yes"], ["lp_No", "lp_no", "lp_ No", "lp_ no"]
    m = pd.Series(logsumexp(a[yes], axis=1) - logsumexp(a[no], axis=1), index=a.index)
    return m.reindex(image_idx).to_numpy(float)


def patch_dirs(model: str) -> None:
    """Fit directions on the mean image patch (CPU). Output: `patch_dirs.npz`, `patch_dirs.json`.

    For each layer in PATCH_LAYERS and each of the 14 prompts, a ridge of the z-scored 8-form margin on the mean
    patch of the 591 images gives a unit weight. `shared` is the first singular vector of the 14 weights, with + on
    the `nice` side, as `dirs` does at the noun. `cool`, `pretty` and `ugly` are the weights of those prompts, with +
    toward yes. sigma is the std over the images of the projection of the mean patch. The test: a fit on 4 of 5 NTA
    folds, and the r of the held-out projection with the margin. The cosines compare each direction with the noun
    direction at the same layer (`dirs.npz`). The patch norm is the median norm of single patches (1 shard).
    """
    from research.scripts.f10.hop_prompts import PROMPTS as ALL14, dual_weights, mean_patch, prompt_name, shared_direction
    from research.scripts.f10.hop_store import ALL_ROOT, open_store

    out = OUT / model
    store = open_store(model, ALL_ROOT)
    idx = store.images["image_idx"].to_numpy()
    codes = pd.factorize(store.images["nta2020"])[0]
    fold = np.random.default_rng(SEED).permutation(codes.max() + 1)[codes] % PATCH_FOLDS
    P = mean_patch(model)  # [images, layers, d], in the store order
    noun = np.load(out / "dirs.npz")
    Y = {(a, n): answer_margin(model, a, n, idx) for a, n in ALL14}
    shard = np.load(sorted((store.path / "patches").glob("residuals-*.npy"))[0], mmap_mode="r")
    save, info = {}, {"version": __version__, "model": model, "margin": "8-form logit margin (answer store)", "layers": {}}
    for layer in PATCH_LAYERS[model]:
        b = store.manifest["layers"].index(layer)
        X = P[:, b].astype(np.float64)

        def fit(rows: np.ndarray) -> dict:
            cols, names = [], []
            for (a, n), y in Y.items():
                z = (y - y[rows].mean()) / y[rows].std()
                cols.append(dual_weights(X, z, rows)[1])
                names.append(prompt_name(a, n))
            W = np.column_stack(cols)
            u, share = shared_direction(W, 1)
            u = u[:, 0] * np.sign(u[:, 0] @ W[:, names.index(prompt_name("nice", "place"))])
            got = {"shared": u, **{a: W[:, names.index(prompt_name(a, "place"))] for a in PATCH_ADJ}}
            return got | {"_share": float(share[0])}

        full = fit(np.arange(len(X)))
        held = {k: np.full(len(X), np.nan) for k in ("shared", *PATCH_ADJ)}
        for f in range(PATCH_FOLDS):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            part = fit(tr)
            for k in held:
                held[k][te] = (X[te] - X[tr].mean(0)) @ part[k]
        cell = {"share_sv1": full["_share"],
                "patch_norm_median": float(np.median(np.linalg.norm(np.asarray(shard[:, :, b], dtype=np.float32), axis=-1))),
                "mean_patch_norm_median": float(np.median(np.linalg.norm(X, axis=1))), "directions": {}}
        noun_u = noun[f"place_L{layer}_u"].astype(np.float64)
        for k in ("shared", *PATCH_ADJ):
            u = full[k] / np.linalg.norm(full[k])
            proj = (X - X.mean(0)) @ u
            test = {a: float(np.corrcoef(held[k], Y[(a, "place")])[0, 1]) for a in ("cool", "nice", "pretty", "ugly")}
            cell["directions"][k] = {"sigma": float(proj.std()), "cos_noun_direction": float(u @ noun_u), "heldout_r": test}
            save[f"{k}_L{layer}_u"] = u.astype(np.float32)
            print(f"[f10.35] {model} L{layer} {k}: sigma {proj.std():.3f}, cos with noun {u @ noun_u:+.3f}, held-out r "
                  + ", ".join(f"{a} {r:+.2f}" for a, r in test.items()), flush=True)
        info["layers"][str(layer)] = cell
    c = np.random.default_rng(SEED + 1).standard_normal(P.shape[-1])
    save["control_u"] = (c / np.linalg.norm(c)).astype(np.float32)
    np.savez(out / "patch_dirs.npz", **save)
    (out / "patch_dirs.json").write_text(json.dumps(info, indent=1))


def load_dirs(model: str) -> tuple[dict, dict]:
    d = np.load(OUT / model / "dirs.npz")
    # The file names the directions by position; the stages name them by role.
    vec = {"aesthetic_u": d["place_u"], "judgment_u": d["final_u"], "control_u": d["control_u"]}
    vec |= {k: d[k] for k in d.files if k.startswith("place_L")}
    return vec, json.loads((OUT / model / "dirs.json").read_text())


def gpu_setup(args: argparse.Namespace):
    """Load the model, the lens and the tokens, as `hop_all.py` does."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from research.monocle import jlens_read

    device = "cuda:0"
    proc = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.float32, device_map=device, attn_implementation="sdpa").eval()
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    lens = jlens_read.load_lens(args.lens) if args.lens else None
    return torch, proc, model, lens_model, lens, device


def make_hook(torch, add_vec, positions: list[int] | None, start: int | None):
    """Add `add_vec` to the block output at `positions`, or at every position from `start` on."""
    def hook(module, inputs, output):
        is_tuple = not torch.is_tensor(output)
        t = output[0] if is_tuple else output
        new = t.clone()
        add = add_vec.to(new.dtype).to(new.device)
        if start is not None:
            new[0, start:, :] = new[0, start:, :] + add
        else:
            for p in positions:
                new[0, p, :] = new[0, p, :] + add
        return (new, *tuple(output[1:])) if is_tuple else new
    return hook


def forward(torch, model, lens_model, inputs: dict, layer: int | None, hook):
    """One forward with at most 1 hook. Give the logits of the last token."""
    handle = None
    try:
        if hook is not None:
            handle = lens_model.layers[layer].register_forward_hook(hook)
        with torch.inference_mode():
            out = model(**{**inputs, "use_cache": False})
    finally:
        if handle is not None:
            handle.remove()
    return out.logits[0, -1].float()


def grow(torch, cur: dict, nxt: int) -> dict:
    """Give the inputs with 1 more token. Every 2-D tensor of the sequence length grows by 1."""
    n = cur["input_ids"].shape[1]
    new = {}
    for k, v in cur.items():
        if torch.is_tensor(v) and v.ndim == 2 and v.shape[1] == n and k != "pixel_values":
            fill = torch.full((1, 1), nxt if k == "input_ids" else (1 if k == "attention_mask" else 0), dtype=v.dtype, device=v.device)
            new[k] = torch.cat([v, fill], dim=1)
        else:
            new[k] = v
    return new


def run(args: argparse.Namespace) -> None:
    from research.scripts.f10.hop_all import ANSWERS, pooled_margin
    from research.scripts.f10.hop_depth import ask, locate
    from research.scripts.f8.r3_readout import load_image

    torch, proc, model, lens_model, _, device = gpu_setup(args)
    tok = proc.tokenizer
    vec, info = load_dirs(args.model_key)
    image_token = int(model.config.image_token_id)
    one = lambda t: tok.encode(t, add_special_tokens=False)[0]
    forms = torch.tensor([one(t) for t in ANSWERS], device=device)
    yes_ids, no_ids = forms[: len(ANSWERS) // 2], forms[len(ANSWERS) // 2:]
    eos = {int(i) for i in ([tok.eos_token_id] if isinstance(tok.eos_token_id, int) else tok.eos_token_id or [])}
    images = pd.read_parquet(ROOT / "data/f10/hop_depth/images.parquet")
    sample = images.sort_values("image_idx").groupby("nta2020").head(1).sort_values("image_idx")
    if args.limit:
        sample = sample.head(args.limit)
    out = OUT / args.model_key
    out.mkdir(parents=True, exist_ok=True)
    print(f"[f10.29] {args.model_key}: {len(sample)} images, {len(PROMPTS)} prompts, {len(VARIANTS)} variants, alphas {ALPHAS}", flush=True)

    def render(question: str, image: bool) -> str:
        text = {"type": "text", "text": question}
        msgs = [{"role": "user", "content": [{"type": "image"}, text] if image else [text]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def read(logits) -> dict:
        lp = torch.log_softmax(logits, -1)
        p = lp.exp()
        return {"margin": float(pooled_margin(lp[forms])), "p_yes": float(p[yes_ids].sum()), "p_no": float(p[no_ids].sum()), "top": tok.decode([int(lp.argmax())])}

    def vector(name: str, pos: str, alpha: float):
        return torch.tensor(vec[f"{name}_u"] * (alpha * info[pos]["sigma"]), device=device)

    gens = []
    for adj, noun in PROMPTS:
        path = out / f"margins-{noun}-{adj}.parquet"
        if path.exists():
            print(f"[f10.29] {path.name} exists", flush=True)
            continue
        question = ask(adj, noun)
        first, place, cool, mark = one("Is"), one(f" {noun}"), one(f" {adj}"), one("?")
        rows, t0 = [], time.perf_counter()
        for k, r in enumerate(sample.itertuples(index=False)):
            inputs = dict(proc(text=[render(question, True)], images=[[load_image(args.image_root, r.sample_id)]], return_tensors="pt").to(device))
            seq = inputs["input_ids"][0].tolist()
            at = locate(seq, image_token, first, place, cool, mark)
            base = {"prompt": f"{noun} {adj}", "image_idx": int(r.image_idx), "nta2020": r.nta2020, "borough": r.borough}
            rows.append(base | {"direction": "none", "where": "none", "alpha": 0.0} | read(forward(torch, model, lens_model, inputs, None, None)))
            for name, pos, where in VARIANTS:
                layer = info[pos]["layer"]
                for alpha in ALPHAS:
                    v = vector(name, pos, alpha)
                    hook = make_hook(torch, v, None, at["place"]) if where == "tail" else make_hook(torch, v, [at["place"] if where == "noun" else at["final"]], None)
                    rows.append(base | {"direction": name, "where": where, "alpha": alpha, "layer": layer} | read(forward(torch, model, lens_model, inputs, layer, hook)))
            if k < GEN_IMAGES:
                for alpha in GEN_ALPHAS:
                    v = vector("aesthetic", "place", alpha)
                    hook = make_hook(torch, v, None, at["place"]) if alpha else None
                    layer = info["place"]["layer"]
                    cur, toks = inputs, []
                    handle = None
                    try:
                        if hook is not None:
                            handle = lens_model.layers[layer].register_forward_hook(hook)
                        for _ in range(GEN_TOKENS):
                            with torch.inference_mode():
                                nxt = int(model(**{**cur, "use_cache": False}).logits[0, -1].argmax())
                            toks.append(nxt)
                            if nxt in eos:
                                break
                            cur = grow(torch, cur, nxt)
                    finally:
                        if handle is not None:
                            handle.remove()
                    gens.append(base | {"alpha": alpha, "text": tok.decode(toks)})
            if k % 20 == 0:
                print(f"[f10.29] {noun} {adj}: image {k + 1}/{len(sample)}, {time.perf_counter() - t0:.0f} s", flush=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        pd.DataFrame(gens).to_parquet(out / "generations.parquet", index=False)
        print(f"[f10.29] wrote {path.name} ({len(rows)} rows) in {time.perf_counter() - t0:.0f} s", flush=True)


def words(args: argparse.Namespace) -> None:
    """Item 7: the lens words of a token with the direction added, with no image."""
    from research.scripts.f10.hop_depth import ask

    torch, proc, model, lens_model, lens, device = gpu_setup(args)
    tok = proc.tokenizer
    vec, info = load_dirs(args.model_key)
    one = lambda t: tok.encode(t, add_special_tokens=False)
    from jlens.hooks import ActivationRecorder

    def render(question: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    cases = [(w, "noun", ask("cool", w)) for w in NOUN_WORDS] + [(w, "adjective", ask(w, "place")) for w in ADJ_WORDS]
    rows = []
    for word, role, question in cases:
        ids = one(f" {word}")
        if len(ids) != 1:
            rows.append({"word": word, "role": role, "note": f"` {word}` is {len(ids)} tokens; skipped"})
            print(f"[f10.29] ` {word}` is {len(ids)} tokens; skipped", flush=True)
            continue
        inputs = dict(proc(text=[render(question)], return_tensors="pt").to(device))
        seq = inputs["input_ids"][0].tolist()
        at = {"word": seq.index(ids[0]), "final": len(seq) - 1}
        layers = sorted({info[p]["layer"] for p in info["layers"]})
        with ActivationRecorder(lens_model.layers, at=layers) as rec, torch.inference_mode():
            model(**inputs, use_cache=False)
            h = {l: rec.activations[l][0].float() for l in layers}
        for name, pos in (("aesthetic", "place"), ("judgment", "final")):
            layer = info[pos]["layer"]
            u = torch.tensor(vec[f"{name}_u"], device=device)
            for where in ("word", "final"):
                h0 = h[layer][at[where]]
                with torch.inference_mode():
                    base = lens_model.unembed(lens.transport(h0[None], layer))[0].float()
                for alpha in (-6.0, 0.0, 6.0):
                    with torch.inference_mode():
                        x = lens_model.unembed(lens.transport((h0 + alpha * info[pos]["sigma"] * u)[None], layer))[0].float()
                    top = [tok.decode([int(i)]) for i in x.topk(TOP).indices]
                    rise = [tok.decode([int(i)]) for i in (x - base).topk(TOP).indices] if alpha else []
                    rows.append({"word": word, "role": role, "direction": name, "layer": layer, "token": where, "alpha": alpha,
                                 "top": " | ".join(top), "rise": " | ".join(rise), "note": ""})
        print(f"[f10.29] words: {word} done", flush=True)
    out = OUT / args.model_key
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "words.parquet", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 200)
    print(pd.DataFrame(rows)[["word", "direction", "token", "alpha", "top"]].to_string(index=False))


def words_model(args: argparse.Namespace) -> None:
    """Item 7 with the model's own readout: inject at the word token, read the next-token words and the margin."""
    from research.scripts.f10.hop_all import ANSWERS, pooled_margin
    from research.scripts.f10.hop_depth import ask

    torch, proc, model, lens_model, _, device = gpu_setup(args)
    tok = proc.tokenizer
    vec, info = load_dirs(args.model_key)
    one = lambda t: tok.encode(t, add_special_tokens=False)
    forms = torch.tensor([one(t)[0] for t in ANSWERS], device=device)

    def render(question: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def top(logits, k: int = 12) -> str:
        return " | ".join(tok.decode([int(i)]) for i in logits.topk(k).indices)

    cases = [(w, "noun", ask("cool", w)) for w in NOUN_WORDS] + [(w, "adjective", ask(w, "place")) for w in ADJ_WORDS]
    rows = []
    for word, role, question in cases:
        ids = one(f" {word}")
        if len(ids) != 1:
            continue
        inputs = dict(proc(text=[render(question)], return_tensors="pt").to(device))
        seq = inputs["input_ids"][0].tolist()
        at = {"word": seq.index(ids[0]), "final": len(seq) - 1}
        for name, pos, where in (("none", "place", "word"), ("aesthetic", "place", "word"), ("aesthetic", "place", "tail"), ("judgment", "final", "final"), ("control", "place", "word")):
            layer = info[pos]["layer"]
            for alpha in ((0.0,) if name == "none" else (-6.0, 6.0, -24.0, 24.0)):
                hook = None
                if name != "none":
                    v = torch.tensor(vec[f"{name}_u"] * (alpha * info[pos]["sigma"]), device=device)
                    hook = make_hook(torch, v, None, at["word"]) if where == "tail" else make_hook(torch, v, [at[where]], None)
                handle = None
                try:
                    if hook is not None:
                        handle = lens_model.layers[layer].register_forward_hook(hook)
                    with torch.inference_mode():
                        out = model(**{**inputs, "use_cache": False}).logits[0].float()
                finally:
                    if handle is not None:
                        handle.remove()
                lp = torch.log_softmax(out[-1], -1)
                rows.append({"word": word, "role": role, "direction": name, "where": where, "alpha": alpha, "layer": layer if name != "none" else -1,
                             "next_after_word": top(out[at["word"]]), "next_at_final": top(out[-1]), "margin": float(pooled_margin(lp[forms]))})
        print(f"[f10.29] words_model: {word} done", flush=True)
    out_dir = OUT / args.model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / "words_model.parquet", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(pd.DataFrame(rows)[["word", "direction", "where", "alpha", "margin", "next_at_final"]].to_string(index=False))


DOSE_ALPHAS = (-48.0, -24.0, -12.0, -6.0, 6.0, 12.0, 24.0, 48.0)
DOSE_IMAGES = 40


def dose(args: argparse.Namespace) -> None:
    """Item 5 at larger doses, on 40 images."""
    from research.scripts.f10.hop_all import ANSWERS, pooled_margin
    from research.scripts.f10.hop_depth import ask, locate
    from research.scripts.f8.r3_readout import load_image

    torch, proc, model, lens_model, _, device = gpu_setup(args)
    tok = proc.tokenizer
    vec, info = load_dirs(args.model_key)
    image_token = int(model.config.image_token_id)
    one = lambda t: tok.encode(t, add_special_tokens=False)[0]
    forms = torch.tensor([one(t) for t in ANSWERS], device=device)
    yes_ids, no_ids = forms[: len(ANSWERS) // 2], forms[len(ANSWERS) // 2:]
    images = pd.read_parquet(ROOT / "data/f10/hop_depth/images.parquet")
    sample = images.sort_values("image_idx").groupby("nta2020").head(1).sort_values("image_idx")
    sample = sample.iloc[:: max(1, len(sample) // DOSE_IMAGES)].head(DOSE_IMAGES)
    out = OUT / args.model_key
    out.mkdir(parents=True, exist_ok=True)

    def render(question: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def read(logits) -> dict:
        lp = torch.log_softmax(logits, -1)
        p = lp.exp()
        return {"margin": float(pooled_margin(lp[forms])), "p_yes": float(p[yes_ids].sum()), "p_no": float(p[no_ids].sum()), "top": tok.decode([int(lp.argmax())])}

    for adj, noun in PROMPTS:
        path = out / f"dose-{noun}-{adj}.parquet"
        if path.exists():
            continue
        question = ask(adj, noun)
        first, place, cool, mark = one("Is"), one(f" {noun}"), one(f" {adj}"), one("?")
        rows, t0 = [], time.perf_counter()
        for r in sample.itertuples(index=False):
            inputs = dict(proc(text=[render(question)], images=[[load_image(args.image_root, r.sample_id)]], return_tensors="pt").to(device))
            seq = inputs["input_ids"][0].tolist()
            at = locate(seq, image_token, first, place, cool, mark)
            base = {"prompt": f"{noun} {adj}", "image_idx": int(r.image_idx), "nta2020": r.nta2020, "borough": r.borough}
            rows.append(base | {"direction": "none", "where": "none", "alpha": 0.0} | read(forward(torch, model, lens_model, inputs, None, None)))
            # The aesthetic direction at each place layer (noun onward), the judgment direction at the last token, the control at the best place layer.
            variants = [("aesthetic", f"place_L{l}_u", int(l), c["sigma"], "tail") for l, c in info["place_layers"].items()]
            variants += [("judgment", "judgment_u", info["final"]["layer"], info["final"]["sigma"], "final"), ("control", "control_u", info["place"]["layer"], info["place"]["sigma"], "tail")]
            for name, key, layer, sigma, where in variants:
                for alpha in DOSE_ALPHAS:
                    v = torch.tensor(vec[key] * (alpha * sigma), device=device)
                    hook = make_hook(torch, v, None, at["place"]) if where == "tail" else make_hook(torch, v, [at["final"]], None)
                    rows.append(base | {"direction": name, "where": where, "alpha": alpha, "layer": layer} | read(forward(torch, model, lens_model, inputs, layer, hook)))
        pd.DataFrame(rows).to_parquet(path, index=False)
        print(f"[f10.29] wrote {path.name} ({len(rows)} rows) in {time.perf_counter() - t0:.0f} s", flush=True)


PATCH_ALPHAS = (-96.0, -48.0, -24.0, -12.0, 12.0, 24.0, 48.0, 96.0)
PATCH_DIRS = ("shared", "cool", "pretty", "ugly", "control")
PATCH_PROMPTS = (("cool", "place"), ("pretty", "place"), ("ugly", "place"))
PATCH_READ = (24, 32, 40, 48, 56)  # the layers at which we read the patches; a layer before the hook is a check
PATCH_TOP = 30
PATCH_IMAGES = 40


def patch_sample(kind: str = "default") -> pd.DataFrame:
    """Give the 5 teaser images, then the 40 images of `dose` (no repeats). `teaser`: the 5 teaser images alone."""
    images = pd.read_parquet(ROOT / "data/f10/hop_depth/images.parquet")
    teaser = pd.read_csv(ROOT / "research/figures/f10/mechanism/patch_teaser_images.csv")["sample_id"]
    first = images.set_index("sample_id").loc[teaser].reset_index()
    if kind == "teaser":
        return first.assign(teaser=True)
    nta = images.sort_values("image_idx").groupby("nta2020").head(1).sort_values("image_idx")
    rest = nta.iloc[:: max(1, len(nta) // PATCH_IMAGES)].head(PATCH_IMAGES)
    both = pd.concat([first.assign(teaser=True), rest.assign(teaser=False)], ignore_index=True)
    return both.drop_duplicates("sample_id", keep="first").reset_index(drop=True)


def patches_stage(args: argparse.Namespace) -> None:
    """(GPU) Add a patch direction to all image patches at 1 layer, and read the patches and the answer.

    For each image, condition (direction, layer, alpha) and prompt, 1 forward. In the `cool` forward we also record
    the patches at PATCH_READ and keep the top PATCH_TOP lens ids of each patch, the projection of the mean patch on
    the direction, and the projection of the noun token on the noun direction at L32 (`dirs.npz`). The patches come
    before the question, thus their states are the same for each prompt. Output, 1 file pair for each image:
    `patches/{sample_id}.parquet` (the rows) and `patches/{sample_id}.npz` (ids [conditions, patches, layers, top]).
    A requeue skips the images that are done.
    """
    from research.scripts.f10.hop_all import ANSWERS, parts, pooled_margin
    from research.scripts.f10.hop_depth import ask, locate
    from jlens.hooks import ActivationRecorder
    from research.scripts.f8.r3_readout import load_image

    torch, proc, model, lens_model, lens, device = gpu_setup(args)
    tok = proc.tokenizer
    pdirs = np.load(OUT / args.model_key / "patch_dirs.npz")
    pinfo = json.loads((OUT / args.model_key / "patch_dirs.json").read_text())
    noun_u = torch.tensor(np.load(OUT / args.model_key / "dirs.npz")["place_L32_u"], device=device)
    image_token = int(model.config.image_token_id)
    one = lambda t: tok.encode(t, add_special_tokens=False)[0]
    forms = torch.tensor([one(t) for t in ANSWERS], device=device)
    yes_ids, no_ids = forms[: len(ANSWERS) // 2], forms[len(ANSWERS) // 2:]
    for adj, noun in PATCH_PROMPTS:
        if len(tok.encode(f" {adj}", add_special_tokens=False)) != 1:
            raise RuntimeError(f"` {adj}` is not 1 token")
    out = OUT / args.model_key / "patches"
    out.mkdir(parents=True, exist_ok=True)
    sample = patch_sample(args.images)
    if args.limit:
        sample = sample.head(args.limit)
    hook_layers = [int(x) for x in args.hook_layers.split(",")] if args.hook_layers else PATCH_LAYERS[args.model_key]
    conds = [("none", -1, 0.0)] + [(d, l, a) for l in hook_layers for d in PATCH_DIRS for a in PATCH_ALPHAS]
    print(f"[f10.35] {args.model_key}: {len(sample)} images, {len(conds)} conditions, {len(PATCH_PROMPTS)} prompts", flush=True)

    def render(question: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def vector(name: str, layer: int, alpha: float):
        key = "control_u" if name == "control" else f"{name}_L{layer}_u"
        # The control takes the sigma of the shared direction, so its dose has the same size.
        sigma = pinfo["layers"][str(layer)]["directions"]["shared" if name == "control" else name]["sigma"]
        return torch.tensor(pdirs[key] * (alpha * sigma), device=device), torch.tensor(pdirs[key], device=device)

    def patch_hook(add, idx):
        def hook(module, inputs, output):
            is_tuple = not torch.is_tensor(output)
            t = output[0] if is_tuple else output
            new = t.clone()
            new[0, idx, :] = new[0, idx, :] + add.to(new.dtype)
            return (new, *tuple(output[1:])) if is_tuple else new
        return hook

    t0 = time.perf_counter()
    for k, r in enumerate(sample.itertuples(index=False)):
        if (out / f"{r.sample_id}.npz").exists():
            continue
        image = load_image(args.image_root, r.sample_id)
        rows, ids_all = [], []
        for adj, noun in PATCH_PROMPTS:
            inputs = dict(proc(text=[render(ask(adj, noun))], images=[[image]], return_tensors="pt").to(device))
            seq = inputs["input_ids"][0].tolist()
            at = locate(seq, image_token, one("Is"), one(f" {noun}"), one(f" {adj}"), one("?"))
            idx = torch.tensor(parts(seq, image_token)[1], device=device)
            if len(idx) != 256:
                raise RuntimeError(f"{r.sample_id}: {len(idx)} patches")
            record = adj == "cool"
            for name, layer, alpha in conds:
                handles = []
                try:
                    if name != "none":
                        add, unit = vector(name, layer, alpha)
                        handles.append(lens_model.layers[layer].register_forward_hook(patch_hook(add, idx)))
                    with torch.inference_mode():
                        if record:
                            with ActivationRecorder(lens_model.layers, at=sorted({*PATCH_READ, 32})) as rec:
                                logits = model(**{**inputs, "use_cache": False}).logits[0, -1].float()
                            h = {l: rec.activations[l][0].float() for l in sorted({*PATCH_READ, 32})}
                        else:
                            logits = model(**{**inputs, "use_cache": False}).logits[0, -1].float()
                finally:
                    for hd in handles:
                        hd.remove()
                lp = torch.log_softmax(logits, -1)
                p = lp.exp()
                row = {"sample_id": r.sample_id, "image_idx": int(r.image_idx), "borough": r.borough, "teaser": bool(r.teaser),
                       "prompt": f"{noun} {adj}", "direction": name, "layer": layer, "alpha": alpha,
                       "margin": float(pooled_margin(lp[forms])), "p_yes": float(p[yes_ids].sum()), "p_no": float(p[no_ids].sum()),
                       "top": tok.decode([int(lp.argmax())])}
                if record:
                    u = unit if name != "none" else None
                    ids = []
                    with torch.inference_mode():
                        for l in PATCH_READ:
                            hp = h[l][idx]
                            ids.append(lens_model.unembed(lens.transport(hp, l)).float().topk(PATCH_TOP, dim=-1).indices)
                            if u is not None:
                                row[f"proj_L{l}"] = float(hp.mean(0) @ u)
                        row["noun_proj_L32"] = float(h[32][at["place"]] @ noun_u)
                    ids_all.append(torch.stack(ids, dim=1).to(torch.int32).cpu().numpy())
                rows.append(row)
        pd.DataFrame(rows).to_parquet(out / f"{r.sample_id}.parquet", index=False)
        tmp = out / f"{r.sample_id}.tmp.npz"
        np.savez(tmp, ids=np.stack(ids_all), conditions=np.array([f"{n}|{l}|{a}" for n, l, a in conds]), read_layers=np.array(PATCH_READ))
        tmp.rename(out / f"{r.sample_id}.npz")
        print(f"[f10.35] image {k + 1}/{len(sample)} {r.sample_id}: {time.perf_counter() - t0:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    d = sub.add_parser("dirs")
    d.add_argument("--model", required=True, choices=MODELS)
    pd_ = sub.add_parser("patch_dirs")
    pd_.add_argument("--model", required=True, choices=MODELS)
    for name in ("run", "words", "words_model", "dose", "patches"):
        s = sub.add_parser(name)
        s.add_argument("--model-key", required=True, choices=MODELS)
        s.add_argument("--model-dir", required=True)
        s.add_argument("--image-root", default="")
        s.add_argument("--lens", default="")
        s.add_argument("--limit", type=int, default=0)
        s.add_argument("--images", default="default", choices=("default", "teaser"), help="patches: the image set")
        s.add_argument("--hook-layers", default="", help="patches: comma list of hook layers (default PATCH_LAYERS)")
    args = ap.parse_args()
    if args.stage == "dirs":
        dirs(args.model)
    elif args.stage == "patch_dirs":
        patch_dirs(args.model)
    else:
        {"run": run, "words": words, "words_model": words_model, "dose": dose, "patches": patches_stage}[args.stage](args)


if __name__ == "__main__":
    main()
