"""What is in a red square, in the model's own words.

The gap
-------
The deletion, substitution and transfer tests all show that the squares the
map calls Dangerous carry the judgment. No test so far says WHAT they hold. A
reader still cannot check a coral square against the photograph, and every
image statistic we can compute comes back null.

There are no human labels here. So the anchor is the model's own vocabulary,
and the loop closes inside the model with no word list of ours in it:

    the cells -> a caption the model writes -> a verdict the model gives

The steps
---------
1. Keep only the k cells one arm names, and take the other 64-k away.
2. Ask an OPEN question and let the model write a sentence.
3. Feed that sentence back as TEXT ONLY, with the safety question, and read
   `c` again.

If the cells the map calls Dangerous produce sentences that the model then
reads as more dangerous, their content is verbalizable as danger. That is the
first result here a person can read and disagree with.

| Arm | Prediction for the text-only `c` |
|---|---|
| keep the Safe cells | HIGHEST |
| keep k random cells | in between |
| keep the Dangerous cells | LOWEST |

What a null would mean
----------------------
Two ways to fail, and they are different:

- the captions are all the same (the model cannot describe a quarter of a
  photograph, so the probe never got started);
- the captions differ and the verdicts do not (the content moves the judgment
  without being sayable).

The module records the captions, so the 2 can be told apart by reading them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import deletion_curve as dcv  # noqa: E402
from research.monocle import jlens_read  # noqa: E402
from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import patch_map_causal as pmc  # noqa: E402
from research.monocle import batch_gen as bg  # noqa: E402
from research.monocle import single as sg  # noqa: E402
from research.monocle import thinking_read as tr  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/caption_loop"
CAPTION_QUESTION = "Describe what you can see in this photograph, in one sentence."
#: A quarter of the picture. Fewer cells and the model has nothing to say.
DEFAULT_K = 16
DEFAULT_GROUP = 2
MAX_NEW = 48
#: 280 soft tokens give a 16 x 16 grid, so a group of `g` gives this many.
N_PATCHES = 256


def cells_for(group: int) -> int:
    """How many cells a `group` x `group` grid holds."""
    if group < 1 or N_PATCHES % (group * group):
        raise ValueError(
            f"group {group} does not divide {N_PATCHES} patches evenly")
    return N_PATCHES // (group * group)


def k_for(group: int, fraction: float, fixed=None) -> int:
    """How many cells an arm keeps.

    **`k` must follow the grid.** An arm of 16 cells is a quarter of a 64-cell
    grid and the WHOLE of a 16-cell one. A fixed `k` across group sizes thus
    compares different amounts of the photograph, and the cell size and the
    kept area move together.

    A fraction keeps the AREA equal, so only the cell size varies.
    """
    n = cells_for(group)
    if fixed is not None:
        if fixed > n:
            raise ValueError(
                f"k={fixed} is more than the {n} cells of a group-{group} "
                "grid")
        return fixed
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"keep-fraction must be inside (0, 1), got {fraction}")
    return max(1, round(fraction * n))


def log(m: str) -> None:
    print(f"[caption] {m}", flush=True)


def caption_inputs(
    proc: Any, tmpl: str, image: Image.Image, device: str,
) -> dict:
    """An OPEN question, and no forced prefix, so the model writes prose."""
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": CAPTION_QUESTION}]}]
    text = proc.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template=tmpl)
    return proc(text=[text], images=[[image]],
                return_tensors="pt").to(device)


def described_text(case: str, caption: str) -> str:
    """The rating question, asked about a DESCRIPTION instead of a picture."""
    body = sg.prompt_text(case)
    return (f"A photograph is described as follows.\n\n\"{caption.strip()}\""
            f"\n\n{body}")


def text_only_inputs(
    proc: Any, tmpl: str, case: str, caption: str, device: str,
) -> dict:
    """No image at all. The caption is the only evidence the model gets."""
    messages = [{"role": "user", "content": [
        {"type": "text", "text": described_text(case, caption)}]}]
    text = proc.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template=tmpl) + sg.FORCE_PREFIX
    return proc(text=[text], return_tensors="pt").to(device)


def generate(model: Any, proc: Any, inputs: dict, max_new: int) -> str:
    """Greedy, so the caption is a property of the model and not of a seed."""
    n = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=False, use_cache=True)
    return proc.tokenizer.decode(out[0, n:], skip_special_tokens=True).strip()


def keep_only_patches(
    cells, blocks: list[list[int]], n_cells: int,
) -> list[int]:
    """The patches to REMOVE so that only `cells` are left."""
    keep = {int(c) for c in cells}
    return dcv.cells_to_patches(
        [c for c in range(n_cells) if c not in keep], blocks)


def load_ranking(rank_dir: Path, case: str, group: int, seed: int) -> dict:
    """{image_id: values} from a `causal_maps` absence run.

    Recomputing the map is 6 batched forwards for each photograph at group 2
    and 22 at group 1, and `causal_maps` has already done exactly that work.

    **The guard is the point.** A ranking from another case, grid, seed or
    reference would draw a figure that looks right and means nothing. That is
    the same trap that retired the donor map, so the tuple is CHECKED and
    never assumed.
    """
    summ = rank_dir / "summary.json"
    maps = rank_dir / "maps.parquet"
    for f in (summ, maps):
        if not f.exists():
            raise SystemExit(f"{f} is missing; --rank-from needs a finished "
                             "causal_maps absence run")
    meta = json.loads(summ.read_text())
    want = {"case": case, "group": group, "seed": seed, "reference": "absent"}
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise SystemExit(
            "the ranking does not match this run: "
            + "; ".join(f"{k} is {got!r}, needs {need!r}"
                        for k, (got, need) in bad.items()))
    df = pd.read_parquet(maps)
    return {str(r.image_id): np.asarray(r.values_, dtype=float)
            if hasattr(r, "values_") else np.asarray(r["values"], dtype=float)
            for _, r in df.iterrows()}


def plan_arms(
    values: np.ndarray, blocks: list, k: int, n_random: int, rng: Any,
) -> list:
    """Every (arm, draw) of one photograph, with the patches each removes."""
    safe, danger = dcv.rank_cells(values)
    arms = {"keep_safe": [safe[:k]], "keep_danger": [danger[:k]],
            "keep_random": [rng.choice(values.size, size=k, replace=False)
                            for _ in range(n_random)],
            "whole_image": [np.arange(values.size)]}
    out = []
    for arm, draws in arms.items():
        for d, cells in enumerate(draws):
            drop = keep_only_patches(cells, blocks, values.size)
            out.append({"arm": arm, "draw": d, "cells": cells, "drop": drop,
                        "n_drop": len(drop)})
    return out


def run_case(
    case: str, proc: Any, model: Any, lens_model: Any, tmpl: str, *,
    n_images: int, group: int, k: int, n_random: int, seed: int, batch: int,
    out_dir: Path, device: str, thinking: bool = True,
    think_tokens: int = tr.MAX_NEW, gen_batch: int = bg.DEFAULT_BATCH,
    ranking: Optional[dict] = None,
) -> pd.DataFrame:
    """Measure every arm of every photograph, in 3 phases.

    | Phase | What runs | Batched over |
    |---|---|---|
    | 1 | the map and the controls | the cells of one photograph |
    | 2 | the reasoning and the caption | rows, ACROSS photographs |
    | 3 | the `c` reads | nothing; 1 row at a time |

    Phase 2 is the whole cost of the job, and it is where the rows stack.
    Phase 3 stays at 1 row on purpose: those numbers are ABSOLUTE and are
    compared between arms, so every one of them must sit on the same batch
    reference. Batching them would move them against each other by up to
    0.67 logits.
    """
    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)
    rng = np.random.default_rng(seed)
    open_text = tr.open_prompt(proc, tmpl, case) if thinking else ""

    # ---------------------------------------------------------------- phase 1
    store, jobs, maps = [], [], []
    for i, r in enumerate(sg.load_single_rows(n_images, case, seed=seed)):
        img = Image.open(r["path"]).convert("RGB")
        rate = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        slots = pab.slots_of(rate, image_token_id)
        c0 = float(pab.contrast(model, lens_model, rate, pos_id, neg_id)[0])
        ctl = pab.controls(model, lens_model, rate, slots, c0, pos_id, neg_id,
                           image_token_id=image_token_id, batch=batch)
        pab.gate_rebuild(ctl)
        if ranking is not None:
            if r["image_id"] not in ranking:
                raise SystemExit(
                    f"{r['image_id']} is not in the ranking; a partial "
                    "ranking would silently drop photographs")
            values = ranking[r["image_id"]]
        else:
            values = -pab.ablation_map(
                model, lens_model, rate, slots, c0, pos_id, neg_id,
                batch=batch, group=group, image_token_id=image_token_id)
        blocks = (pmc.patch_groups(len(slots), group) if group > 1
                  else [[j] for j in range(len(slots))])
        cap_base = caption_inputs(proc, tmpl, img, device)
        safe, danger = dcv.rank_cells(values)
        store.append({"row": r, "img": img, "rate": rate, "slots": slots,
                      "cap_base": cap_base,
                      "cap_slots": pab.slots_of(cap_base, image_token_id),
                      "c0": c0, "ctl": ctl})
        maps.append({
            "case": case, "image_id": r["image_id"], "path": str(r["path"]),
            "group": group, "k": k, "c_image": c0,
            "image_worth": ctl["image_worth"],
            "worth_ok": bool(ctl["worth_ok"]),
            "values": np.asarray(values, dtype=float).tolist(),
            "safe_cells": [int(x) for x in safe[:k]],
            "danger_cells": [int(x) for x in danger[:k]]})
        for job in plan_arms(values, blocks, k, n_random, rng):
            jobs.append({**job, "image": i, "image_id": r["image_id"]})
        log(f"  planned {i + 1}/{n_images} {r['image_id']} c={c0:+.2f}")

    # ---------------------------------------------------------------- phase 2
    def think_row(job):
        return tr.masked_inputs(
            proc, tmpl, store[job["image"]]["img"], case, open_text,
            device=device, drop_patches=job["drop"],
            image_token_id=image_token_id)

    def cap_row(job):
        st = store[job["image"]]
        return (pab.ablate_row(st["cap_base"], st["cap_slots"], job["drop"])
                if job["drop"] else st["cap_base"])

    traces = (bg.run_batched(jobs, think_row, model, proc,
                             max_new=think_tokens, batch=gen_batch,
                             label="think ")
              if thinking else [""] * len(jobs))
    captions = bg.run_batched(jobs, cap_row, model, proc, max_new=MAX_NEW,
                              batch=gen_batch, label="caption ")

    # ---------------------------------------------------------------- phase 3
    rows = []
    for job, raw, caption in zip(jobs, traces, captions):
        st = store[job["image"]]
        drop = job["drop"]
        rate_row = (pab.ablate_row(st["rate"], st["slots"], drop) if drop
                    else st["rate"])
        c_rate = float(pab.contrast(model, lens_model, rate_row,
                                    pos_id, neg_id)[0])
        if thinking:
            block, closed = tr.split_thought(raw)
            th = {"c_think": tr.read_after_thought(
                      model, proc, lens_model, tmpl, st["img"], case,
                      open_text, block, pos_id, neg_id, device=device,
                      drop_patches=drop, image_token_id=image_token_id),
                  **tr.describe_thought(proc, block, closed)}
        else:
            th = {"c_think": float("nan"), "closed": False, "reasoning": "",
                  "n_thought_tokens": 0}
        caption = caption.strip()
        t_in = text_only_inputs(proc, tmpl, case, caption, device)
        c_text = float(pab.contrast(model, lens_model, t_in,
                                    pos_id, neg_id)[0])
        rows.append({
            "case": case, "image_id": job["image_id"], "arm": job["arm"],
            "draw": job["draw"], "k": k, "group": group, "c_image": st["c0"],
            "image_worth": st["ctl"]["image_worth"],
            "worth_ok": bool(st["ctl"]["worth_ok"]),
            "caption": caption, "n_words": len(caption.split()),
            "c_text": c_text, "c_rate": c_rate,
            "p_rate": float(1.0 / (1.0 + np.exp(-c_rate))),
            "c_think": th["c_think"], "think_closed": bool(th["closed"]),
            "reasoning": th["reasoning"],
            "n_thought_tokens": int(th["n_thought_tokens"])})

    df = pd.DataFrame(rows)
    case_dir = out_dir / f"{case}_g{group}_k{k}"
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(case_dir / "captions.parquet", index=False)
    pd.DataFrame(maps).to_parquet(case_dir / "maps.parquet", index=False)

    strong = df[df.worth_ok]
    log(f"{case}: {strong.image_id.nunique()} strong images")
    log(f"{'arm':<14} {'median c_rate':>14} {'median c_think':>15} "
        f"{'median c_text':>14} {'thoughts closed':>16}")
    for arm in ("keep_safe", "keep_random", "keep_danger", "whole_image"):
        sub = strong[strong.arm == arm]
        if not len(sub):
            continue
        log(f"{arm:<14} {sub.c_rate.median():>14.3f} "
            f"{sub.c_think.median():>15.3f} "
            f"{sub.c_text.median():>14.3f} "
            f"{f'{int(sub.think_closed.sum())}/{len(sub)}':>16}")
    # A trace that never closed has NO answer position, so its `c_think` is
    # not a verdict. Say how many, and let the reader drop them.
    open_traces = int((~strong.think_closed).sum())
    if thinking and open_traces:
        log(f"{open_traces} of {len(strong)} thoughts never closed; their "
            "c_think is not a verdict and must be dropped")
    (case_dir / "summary.json").write_text(json.dumps({
        "case": case, "k": k, "group": group, "seed": seed,
        "gen_batch": gen_batch, "map_batch": batch,
        "ranking_loaded": ranking is not None,
        "n_images": int(strong.image_id.nunique()),
        "median_c_text": {a: float(strong[strong.arm == a].c_text.median())
                          for a in strong.arm.unique()},
        "median_c_rate": {a: float(strong[strong.arm == a].c_rate.median())
                          for a in strong.arm.unique()},
        "median_c_think": {a: float(strong[strong.arm == a].c_think.median())
                           for a in strong.arm.unique()},
        "thoughts_closed": int(strong.think_closed.sum()),
        "thoughts_total": int(len(strong)),
        "distinct_captions": {a: int(strong[strong.arm == a].caption.nunique())
                              for a in strong.arm.unique()},
    }, indent=2))
    return df


def draw_panels(case_dir: Path, readout: str = "rating") -> int:
    """The 3-column figure. Imported late so a CPU-only host can still run."""
    from research.monocle import caption_panel as cp
    return cp.draw_run(case_dir, readout=readout)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["subway_safety"],
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--group", type=int, nargs="+", default=[DEFAULT_GROUP],
                    help="k x k blocks of patches. Give several to compare "
                         "cell sizes, e.g. --group 1 2 4. One model load "
                         "serves them all")
    ap.add_argument("--keep-fraction", type=float, default=0.25,
                    help="how much of the photograph each arm keeps. `k` is "
                         "derived from it per group, so the arms hold the "
                         "same AREA at every cell size")
    ap.add_argument("--k", type=int, default=None,
                    help="cells to keep, fixed. Only for a single --group; "
                         "it would mean different areas across groups")
    ap.add_argument("--n-random", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--batch", type=int, default=12,
                    help="rows per batched forward in the MAP. Raising it is "
                         "safe, because the zero is measured at the same "
                         "size, but it re-baselines: the maps of 2026-08-23 "
                         "were measured at 12")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false",
                    help="skip the reasoning-on readout; it is the slow part")
    ap.add_argument("--think-tokens", type=int, default=tr.MAX_NEW,
                    help="cap on one reasoning trace. A trace that does not "
                         "close inside it has no answer position")
    ap.add_argument("--gen-batch", type=int, default=bg.DEFAULT_BATCH,
                    help="rows per batched GENERATION. This is the whole "
                         "cost of a run. It does not touch a `c`: those are "
                         "read 1 row at a time")
    ap.add_argument("--rank-from", default=None,
                    help="a causal_maps absence run to take the cell "
                         "ranking from, instead of measuring it again. The "
                         "case, group, seed and reference are checked")
    ap.add_argument("--figures", action="store_true",
                    help="draw the 3-column panels after the run. They can "
                         "also be drawn later with monocle.caption_panel, "
                         "which needs no GPU")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    from research.monocle import attn_route as ar

    args = build_parser().parse_args(argv)
    if args.k is not None and len(args.group) > 1:
        raise SystemExit(
            "--k is fixed, so it cannot serve several groups: the same k is "
            "a different SHARE of each grid. Use --keep-fraction.")
    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    for case in args.cases:
        for group in args.group:
            k = k_for(group, args.keep_fraction, args.k)
            log(f"{case}: group {group}, {cells_for(group)} cells, k={k} "
                f"({k / cells_for(group):.0%} of the photograph)")
            ranking = (load_ranking(Path(args.rank_from), case, group,
                                    args.seed)
                       if args.rank_from else None)
            run_case(case, proc, model, lens_model, tmpl,
                     n_images=args.n_images, group=group, k=k,
                     n_random=args.n_random, seed=args.seed,
                     batch=args.batch, out_dir=Path(args.out_dir),
                     device=args.device, thinking=args.thinking,
                     think_tokens=args.think_tokens,
                     gen_batch=args.gen_batch, ranking=ranking)
            if args.figures:
                case_dir = Path(args.out_dir) / f"{case}_g{group}_k{k}"
                for readout in (("rating", "thinking") if args.thinking
                                else ("rating",)):
                    draw_panels(case_dir, readout=readout)
    log("caption loop complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
