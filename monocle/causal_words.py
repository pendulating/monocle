"""What the causal patch says, not only where it sits.

A saliency map names a place. A lens names a word. This module joins the 2:
it takes the MEASURED 1-patch causal map, picks the patches at the 2 poles,
and reads the lens words at those patches.

The control is inside the photograph. Random patches of the SAME photograph
carry the same scene, the same light and the same camera. So a word that
appears at a pole and not at a random patch is a word about the causal role,
not about the picture.

Usage:
    python -m monocle.causal_words --cases restaurants --n-images 60 \\
        --truth-from outputs/_monocle/causal_maps60/restaurants_absent/maps.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from monocle import attn_route as ar
from monocle import extract  # noqa: F401  (kept for the hook convention)
from monocle import jlens_read
from monocle import patch_attribution as pa
from monocle import single as sg

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LENS = str(REPO / "outputs/_monocle/jlens/mm/gemma4_12b_lens.pt")
DEFAULT_OUT = str(REPO / "outputs/_monocle/causal_words")


def log(msg: str) -> None:
    print(f"[causal-words] {msg}", flush=True)


def pick_patches(truth: np.ndarray, n_pole: int, n_rand: int,
                 rng: np.random.Generator) -> dict[str, list[int]]:
    """The 2 poles of the measured map, and a matched random draw.

    The random draw leaves the poles out, so no patch carries 2 roles.
    """
    order = np.argsort(truth)
    neg = [int(i) for i in order[:n_pole]]
    pos = [int(i) for i in order[::-1][:n_pole]]
    taken = set(pos) | set(neg)
    pool = [i for i in range(truth.size) if i not in taken]
    rand = [int(i) for i in rng.choice(pool, size=min(n_rand, len(pool)),
                                       replace=False)]
    return {"pos": pos, "neg": neg, "random": rand}


def words_at(logits: torch.Tensor, patches: list[int], tok: Any,
             top_k: int, mask: Optional[torch.Tensor] = None
             ) -> list[tuple[int, int, str, float]]:
    """The `top_k` words of each patch. Returns (patch, rank, word, value).

    Without `mask` the words come from the raw lens logits. That readout is
    correct, but a lens gives `0` and `<pad>` for most patches below block 30.

    With `mask` the words come from `scoring.score_patches`, which keeps only
    real word tokens and takes away the global token distribution. Use it when
    you want to SEE the words. Use the raw readout when you want to measure
    what the lens says without help.
    """
    out = []
    if not patches:
        return out
    if mask is not None:
        from monocle import scoring
        df = scoring.score_patches(logits[patches, :], tok, k=top_k,
                                   token_mask=mask)
        for r in df.itertuples():
            out.append((int(patches[int(r.patch_idx)]), int(r.rank),
                        str(r.token), float(r.score)))
        return out
    sel = logits[patches, :]
    vals, idx = torch.topk(sel, k=top_k, dim=-1)
    for row, p in enumerate(patches):
        for r in range(top_k):
            word = tok.decode([int(idx[row, r])])
            out.append((int(p), r, word, float(vals[row, r])))
    return out


def run_case(case: str, *, proc: Any, model: Any, lens_model: Any, tmpl: str,
             lens: Any, n_images: int, layers: list[int], top_k: int,
             n_pole: int, n_rand: int, seed: int, device: str,
             out_dir: Path, truth_from: str,
             mask: Optional[torch.Tensor] = None,
             prompt_case: Optional[str] = None) -> pd.DataFrame:
    from PIL import Image
    from monocle.input_gate import load_truth

    tok = proc.tokenizer
    ask = prompt_case or case
    if prompt_case:
        log(f"the question comes from {ask}, the images from {case}")
    stored = load_truth(truth_from, case, 1)
    log(f"the truth comes from {truth_from} ({len(stored)} images)")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    t0 = time.time()
    for i, r in enumerate(sg.load_single_rows(n_images, case, seed=seed)):
        key = str(r["image_id"])
        if key not in stored:
            log(f"  skip {key}: the stored truth omits it")
            continue
        truth = np.asarray(stored[key], dtype=float).ravel()
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, ask, device=device)
        acts, pos, final = jlens_read.record_patch_activations(
            model, lens_model, inputs, layers)
        if int(pos.numel()) != truth.size:
            raise ValueError(f"{key}: {int(pos.numel())} patches against "
                             f"{truth.size} truth cells")
        per_layer = jlens_read.lens_patch_logits(
            lens, lens_model, acts, pos, final, layers=layers)
        groups = pick_patches(truth, n_pole, n_rand, rng)
        for layer, lg in per_layer.items():
            for role, patches in groups.items():
                for p, rank, word, val in words_at(lg, patches, tok, top_k,
                                                   mask=mask):
                    rows.append({"case": case, "ask": ask, "image_id": key,
                                 "layer": int(layer), "role": role,
                                 "patch": p, "rank": rank, "word": word,
                                 "logit": val,
                                 "truth": float(truth[p])})
        del acts, per_layer, final
        log(f"  {i + 1}/{n_images} {key} "
            f"|truth|max={np.max(np.abs(truth)):.3f} "
            f"{time.time() - t0:.0f}s")
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(case_dir / "words.parquet", index=False)
    log(f"wrote {case_dir} ({len(df)} word rows)")
    return df


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["restaurants"])
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--layers", nargs="+", type=int, default=[18, 24])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--n-pole", type=int, default=3)
    ap.add_argument("--n-rand", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--truth-from", default=None,
                    help="1 file, or 1 for each case in --cases order")
    ap.add_argument("--truth-root",
                    default=str(REPO / "outputs/_monocle/causal_maps60"),
                    help="build the truth path as <root>/<case>_absent")
    ap.add_argument("--prompt-case", default=None)
    ap.add_argument("--lens", default=DEFAULT_LENS)
    ap.add_argument("--tag", default="words")
    ap.add_argument("--scored", action="store_true",
                    help="keep real words only, and take away the token prior")
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    lens = jlens_read.load_lens(args.lens)
    log(f"lens fitted at {lens.source_layers}")
    mask = None
    if args.scored:
        from monocle import scoring
        vocab = int(model.config.text_config.vocab_size
                    if hasattr(model.config, "text_config")
                    else model.config.vocab_size)
        mask = scoring.build_token_mask(proc.tokenizer, vocab)
        log("the words are scored: real tokens only, and the prior removed")
    out_dir = Path(DEFAULT_OUT) / args.tag
    for case in args.cases:
        truth = args.truth_from or str(
            Path(args.truth_root) / f"{case}_absent" / "maps.parquet")
        run_case(case, proc=proc, model=model, lens_model=lens_model,
                 tmpl=tmpl, lens=lens, n_images=args.n_images,
                 layers=sorted(set(args.layers)), top_k=args.top_k,
                 n_pole=args.n_pole, n_rand=args.n_rand, seed=args.seed,
                 device=args.device, out_dir=out_dir, truth_from=truth,
                 mask=mask, prompt_case=args.prompt_case)
    log("causal words complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
