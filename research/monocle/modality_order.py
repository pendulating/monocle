"""Does the modality order make the image encoding blind to the question?

The claim under test
--------------------
The gemma-4 card asks for the image BEFORE the text. One reading of that rule
says Google keeps the perception task-neutral: a question that follows the
image cannot steer which image features the model encodes.

The mechanism is not in doubt. `blockwise_overlay` in transformers unmasks a
vision token to its own island alone, and the overlay is OR'd with the causal
mask. So a patch reads every token BEFORE it, plus its own island, and never
text that follows. Put the question after the image and it cannot reach the
patch residual.

This module measures what happens when you put the question FIRST.

The design
----------
Each photograph runs 4 times: 2 layouts by 2 questions. The 2 questions are
the case's own and another case's, which is the cross-question control that
`research/monocle/input_gate.py` already uses.

| Layout | Order | The question reaches the patches |
|---|---|---|
| `image_first` | image, question | no. It comes after them |
| `text_first` | question, image | yes. It comes before them |

For each (photograph, layout, layer) we report how far the patch residual
moves when only the question changes:

    gap  = || h_own - h_other || / || h_own ||      over the patch block
    cos  = the mean per-patch cosine, CENTERED

**Warning: read the gap, not the raw cosine.** Gemma carries massive
activations, so a raw cosine sits near 1.0 for any 2 residuals and says
nothing. Both are reported, and `cos_raw` is there to show that trap, not to
carry a conclusion.

The prediction each hypothesis makes
------------------------------------
| Hypothesis | `image_first` gap | `text_first` gap |
|---|---|---|
| the order makes the encoding blind | about 0.03 | much larger |
| the question never touches the encoding | about 0.03 | about 0.03 |

`image_first` is the control and its value is on record: the gap is 0.031 at
block 18 and 0.078 at block 42, and 2 DIFFERENT photographs sit at a centered
cosine of 0.22 and 0.00. See `causal-lens-retrospective.md`.

The gradient rides along
------------------------
`g` is the backward half, and it carries the question under BOTH layouts. It
is the scale bar: a gap in `h` means little until you see what a real
question-driven change looks like at the same layer.

Scope
-----
`text_first` leaves the canonical path. 3 gates demand `images_then_labels`,
so no run of this module may enter the registry. It is a probe.

Usage:
    sbatch --time=01:00:00 research/monocle/monocle.sub monocle.modality_order \\
        --case subway_safety --other-case restaurants --n-images 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import single as sg  # noqa: E402

DEFAULT_LAYERS = (6, 12, 18, 20, 24, 30, 36, 42, 46)
DEFAULT_OUT = str(REPO / "outputs/_monocle/modality_order")


def log(m: str) -> None:
    print(f"[modality-order] {m}", flush=True)


def relative_gap(a: np.ndarray, b: np.ndarray) -> float:
    """`|| a - b || / || a ||` over the whole patch block.

    This is the statistic to read. It is a ratio of norms, so a channel with a
    huge activation cannot push it toward a fixed value the way it pushes a
    cosine.
    """
    denom = float(np.linalg.norm(a))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(a - b) / denom)


def centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """The mean per-patch cosine, after the mean over patches goes.

    Gemma's massive activations sit in the same few channels at every patch.
    They are thus almost entirely in the patch MEAN, and removing it is what
    makes a cosine mean anything on this model.
    """
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    num = (ac * bc).sum(axis=1)
    den = np.linalg.norm(ac, axis=1) * np.linalg.norm(bc, axis=1)
    ok = den > 0
    if not ok.any():
        return float("nan")
    return float(np.mean(num[ok] / den[ok]))


def raw_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """The mean per-patch cosine with NO centering. The trap, for the record."""
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 0
    if not ok.any():
        return float("nan")
    return float(np.mean(num[ok] / den[ok]))


#: The filler clause. It repeats until it matches the question in TOKENS.
#:
#: It must not name a scene and must not name a judgment, because the pair
#: `neutral_first` / `question_first` reads the difference between this block
#: and the question. Any content here enters that difference.
FILLER_UNIT = "This is a photograph. "

#: The decoy block. It answers the limit that the repeated filler leaves.
#:
#: The filler is 1 sentence, repeated, so it is low in information. A map that
#: changes under `question_first` could thus change because the leading block
#: carries information, and not because it carries the QUESTION. The decoy
#: separates the 2. It is varied prose, it is informative, and it says nothing
#: about a street, a building, a person, or a judgment.
#:
#: Keep it free of every word the rating scale uses. A word such as `safe`,
#: `clean`, or `busy` would make the decoy a weak question, not a control.
DECOY_TEXT = (
    "Jupiter holds four large moons. Galileo Galilei first saw them in 1610. "
    "Io carries hundreds of active volcanoes, and sulfur compounds cover the "
    "ground. Europa keeps a shell of water ice above a salty ocean. Ganymede "
    "is the largest moon in the solar system, and it makes its own magnetic "
    "field. Callisto shows a cratered crust that records billions of years. "
    "The four bodies orbit in a chain, and the inner three keep a resonance "
    "of one to two to four. Tidal forces from Jupiter heat their interiors, "
    "and that heat drives the volcanoes of Io. The Juno spacecraft measured "
    "the gravity field of each body during its passes."
)
#: The names a filler block may take. `neutral` is the repeated sentence.
FILLER_KINDS = ("neutral", "decoy")


def matched_filler(tok: Any, target_text: str, unit: str = FILLER_UNIT) -> str:
    """A filler string whose token count equals that of `target_text`.

    A decode of a truncated id list does not always re-tokenize to the same
    length, so the loop checks and corrects instead of trusting it.
    """
    def n_tokens(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    target = n_tokens(target_text)
    ids = tok(unit * (target + 4), add_special_tokens=False)["input_ids"]
    if len(ids) < target:
        raise ValueError("the filler unit is too short to match the question")
    keep = target
    for _ in range(64):
        text = tok.decode(ids[:keep])
        got = n_tokens(text)
        if got == target:
            return text
        keep += target - got
        keep = max(1, min(keep, len(ids)))
    raise ValueError(
        f"no filler of {target} tokens after 64 tries; got {got}")


def matched_decoy(
    tok: Any, target_text: str, source: str = DECOY_TEXT,
) -> str:
    """A decoy string whose token count equals that of `target_text`.

    The source passage is longer than a question, so the function cuts it. A
    decode of a truncated id list does not always re-tokenize to the same
    length, so the loop checks and corrects instead of trusting it.
    """
    def n_tokens(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    target = n_tokens(target_text)
    ids = tok(source, add_special_tokens=False)["input_ids"]
    while len(ids) < target + 4:
        ids = ids + ids
    keep = target
    for _ in range(64):
        text = tok.decode(ids[:keep])
        got = n_tokens(text)
        if got == target:
            return text
        keep += target - got
        keep = max(1, min(keep, len(ids)))
    raise ValueError(f"no decoy of {target} tokens after 64 tries; got {got}")


def build_filler(tok: Any, target_text: str, kind: str = "neutral") -> str:
    """The leading block, of the kind the caller names.

    Both kinds match `target_text` in TOKENS, so the patch positions and the
    sequence length stay the same. Only the content changes.
    """
    if kind not in FILLER_KINDS:
        raise ValueError(f"filler kind must be one of {FILLER_KINDS}")
    if kind == "neutral":
        return matched_filler(tok, target_text)
    return matched_decoy(tok, target_text)


def read_one(
    model: Any, lens_model: Any, proc: Any, tmpl: str, image: Any,
    ask: str, layers: tuple[int, ...], *, layout: str, device: str,
    image_token_id: int, filler: Optional[str] = None,
) -> dict[int, dict[str, np.ndarray]]:
    """One forward and one backward. Returns `h` and `g` at the patch slots.

    The contrast `c` comes from `ask`, so the gradient carries that question.
    """
    inputs = sg.build_single_inputs(
        proc, tmpl, image, ask, device=device, layout=layout, filler=filler)
    slots = pab.slots_of(inputs, image_token_id)
    seq_len = int(inputs["input_ids"].shape[1])
    pos_id, neg_id = sg.contrast_ids(proc.tokenizer, ask)
    grads, acts, c = pa.attribute_pair(
        model, lens_model, inputs, list(layers), pos_id, neg_id, seq_len - 1)
    out = {}
    for ly in layers:
        out[ly] = {
            "h": acts[ly][0, slots, :].float().cpu().numpy(),
            "g": grads[ly][0, slots, :].float().cpu().numpy(),
        }
    out["meta"] = {"c": float(c), "seq_len": seq_len,
                   "n_patches": len(slots),
                   "first_patch": int(min(slots)),
                   "last_patch": int(max(slots))}
    return out


def per_patch_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The relative gap of EACH patch, so a map can be drawn or scored."""
    num = np.linalg.norm(a - b, axis=1)
    den = np.linalg.norm(a, axis=1)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def compare(
    own: dict, other: dict, layers: tuple[int, ...],
) -> list[dict]:
    """The 3 statistics, for `h` and for `g`, at every layer."""
    rows = []
    for ly in layers:
        row = {"layer": ly}
        for part in ("h", "g"):
            a, b = own[ly][part], other[ly][part]
            row[f"{part}_gap"] = relative_gap(a, b)
            row[f"{part}_cos"] = centered_cosine(a, b)
            row[f"{part}_cos_raw"] = raw_cosine(a, b)
        rows.append(row)
    return rows


def run(
    model: Any, lens_model: Any, proc: Any, tmpl: str, *,
    case: str, other_case: str, n_images: int, seed: int,
    layers: tuple[int, ...], layouts: tuple[str, ...], device: str,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from PIL import Image

    tok = proc.tokenizer
    image_token_id = int(model.config.image_token_id)
    # The filler matches the OWN question in tokens. The cross-question arm
    # keeps that same filler, so a case pair of unequal length would break
    # the position match. The log states both lengths for that reason.
    own_q = sg.prompt_text(case)
    other_q = sg.prompt_text(other_case)
    n_own = len(tok(own_q, add_special_tokens=False)["input_ids"])
    n_other = len(tok(other_q, add_special_tokens=False)["input_ids"])
    filler = None
    if any(lay in sg.FILLER_LAYOUTS for lay in layouts):
        filler = matched_filler(tok, own_q)
        log(f"question tokens: {case} {n_own}, {other_case} {n_other}")
        if n_own != n_other:
            log("WARNING: the 2 questions differ in length. The filler "
                "matches the OWN question only, so the cross-question arm "
                "does not hold the positions fixed.")
        log(f"filler: {len(tok(filler, add_special_tokens=False)['input_ids'])}"
            f" tokens, {filler[:60]!r}...")

    rows: list[dict] = []
    patch_rows: list[dict] = []
    t0 = time.time()
    for i, r in enumerate(sg.load_single_rows(n_images, case, seed=seed)):
        img = Image.open(r["path"]).convert("RGB")
        for layout in layouts:
            pad = filler if layout in sg.FILLER_LAYOUTS else None
            own = read_one(model, lens_model, proc, tmpl, img, case, layers,
                           layout=layout, device=device,
                           image_token_id=image_token_id, filler=pad)
            other = read_one(model, lens_model, proc, tmpl, img, other_case,
                             layers, layout=layout, device=device,
                             image_token_id=image_token_id, filler=pad)
            for row in compare(own, other, layers):
                row.update({
                    "case": case, "other_case": other_case,
                    "layout": layout, "target": r["image_id"],
                    "c_own": own["meta"]["c"], "c_other": other["meta"]["c"],
                    "seq_len_own": own["meta"]["seq_len"],
                    "seq_len_other": other["meta"]["seq_len"],
                    "first_patch_own": own["meta"]["first_patch"],
                    "first_patch_other": other["meta"]["first_patch"],
                    "patch_shift": (other["meta"]["first_patch"]
                                    - own["meta"]["first_patch"]),
                    "answer_distance": (own["meta"]["seq_len"] - 1
                                        - own["meta"]["last_patch"]),
                    "n_patches": own["meta"]["n_patches"],
                })
                rows.append(row)
                # The per-patch map, so a later run can ask WHERE the
                # question lands and whether those cells are the causal ones.
                patch_rows.append({
                    "case": case, "layout": layout, "layer": row["layer"],
                    "target": r["image_id"],
                    "gap": per_patch_gap(own[row["layer"]]["h"],
                                         other[row["layer"]]["h"]).tolist(),
                })
        log(f"  {i + 1}/{n_images} {r['image_id']} {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    patches = pd.DataFrame(patch_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "gaps.parquet", index=False)
    patches.to_parquet(out_dir / "patch_gaps.parquet", index=False)
    return df, patches


#: The 4 contrasts the arm set decomposes the verdict shift into. Each names
#: (the arm to subtract, the arm to read) and what the difference isolates.
CONTRASTS = (
    ("image_first", "neutral_first",
     "the patches move later; no question conditions them"),
    ("neutral_first", "question_first",
     "the SAME positions; only the pre-image block changes"),
    ("question_first", "text_first",
     "the answer moves nearer the patches"),
    ("image_first", "text_first", "the whole reversal"),
)


def report(df: pd.DataFrame) -> dict:
    """The table, and the contrasts the probe exists to make."""
    from scipy import stats

    have = [lay for lay in sg.LAYOUTS if lay in set(df.layout.unique())]
    log("=" * 78)
    log("How far the patch residual moves when ONLY the question changes")
    log(f"{'layer':>6} " + " ".join(f"{lay:>16}" for lay in have))
    summary: dict = {}
    for layer, sub in df.groupby("layer"):
        cells = []
        for lay in have:
            s = sub[sub.layout == lay]
            cells.append(f"{s.h_gap.mean():>16.4f}")
            summary.setdefault("h_gap", {}).setdefault(
                lay, {})[int(layer)] = float(s.h_gap.mean())
        log(f"{layer:>6} " + " ".join(cells))

    log("")
    log("the same statistic on the GRADIENT, which carries the question")
    for layer, sub in df.groupby("layer"):
        cells = " ".join(f"{lay.split('_')[0]} "
                         f"{sub[sub.layout == lay].g_gap.mean():.2f}"
                         for lay in have)
        log(f"  L{layer:<3d} {cells}")

    # The geometry of each arm, so a reader can check the position match
    # instead of trusting it.
    log("")
    log("the geometry of each arm")
    log(f"  {'layout':<16} {'first patch':>12} {'answer distance':>16} "
        f"{'seq len':>9} {'patch shift':>12}")
    for lay in have:
        s = df[df.layout == lay]
        log(f"  {lay:<16} {s.first_patch_own.mean():>12.1f} "
            f"{s.answer_distance.mean():>16.1f} "
            f"{s.seq_len_own.mean():>9.1f} "
            f"{sorted(s.patch_shift.unique())!s:>12}")
        summary.setdefault("geometry", {})[lay] = {
            "first_patch": float(s.first_patch_own.mean()),
            "answer_distance": float(s.answer_distance.mean()),
            "seq_len": float(s.seq_len_own.mean()),
        }

    # The verdict. `c` does not depend on the layer, so 1 row per arm.
    verdict = (df[df.layer == df.layer.min()]
               [["layout", "target", "c_own"]].drop_duplicates())
    log("")
    log("the verdict, and what each contrast isolates")
    log(f"  {'contrast':<34} {'shift':>8} {'p':>8} {'n':>4}  isolates")
    for base, arm, meaning in CONTRASTS:
        if base not in have or arm not in have:
            continue
        a = verdict[verdict.layout == base].set_index("target")["c_own"]
        b = verdict[verdict.layout == arm].set_index("target")["c_own"]
        common = sorted(set(a.index) & set(b.index))
        d = b.loc[common].to_numpy() - a.loc[common].to_numpy()
        t = stats.ttest_1samp(d, 0.0)
        name = f"{arm} - {base}"
        log(f"  {name:<34} {d.mean():>+8.3f} {t.pvalue:>8.4f} "
            f"{len(common):>4}  {meaning}")
        summary.setdefault("verdict", {})[name] = {
            "shift": float(d.mean()), "p": float(t.pvalue),
            "n": len(common), "isolates": meaning}
    log("")
    log("  mean c by arm: " + "  ".join(
        f"{lay} {verdict[verdict.layout == lay].c_own.mean():+.2f}"
        for lay in have))
    return summary


def list_lengths(model_dir: str) -> int:
    """The token length of each case prompt, and where the patches start.

    Under `text_first` the question precedes the image. Two questions of
    different length thus move the patch positions, and the rotary phase
    moves with them. A pair of cases with the SAME length removes that
    confound by construction, and this listing finds one.
    """
    from transformers import AutoProcessor

    from dagspaces.common.model_registry import resolve_model_source
    from dagspaces.common.vllm_inference import _gemma4_unified_chat_template

    _, tmpl = _gemma4_unified_chat_template(model_dir)
    proc = AutoProcessor.from_pretrained(
        resolve_model_source(model_dir, stage_name="monocle"))
    tok = proc.tokenizer
    log(f"{'case':<20} {'image_first':>12} {'text_first':>12}")
    for case in sorted(sg.CASE_RATING):
        cells = []
        for layout in sg.LAYOUTS:
            text = sg.render_single_text(proc, tmpl, case, layout=layout)
            cells.append(len(tok(text, add_special_tokens=False)["input_ids"]))
        log(f"{case:<20} {cells[0]:>12} {cells[1]:>12}")
    log("")
    log("Pick 2 cases whose text_first length AGREES. The patches then sit at")
    log("the same positions, and no rotary shift can enter the gap.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="subway_safety")
    ap.add_argument("--other-case", default="restaurants",
                    help="The cross-question control.")
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260821,
                    help="Match the run that measured a stored map.")
    ap.add_argument("--layers", nargs="+", type=int,
                    default=list(DEFAULT_LAYERS))
    ap.add_argument("--layouts", nargs="+", default=list(sg.LAYOUTS))
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--list-lengths", action="store_true",
                    help="Print the prompt length of each case, then stop.")
    args = ap.parse_args()

    if args.list_lengths:
        return list_lengths(args.model_dir)

    for lay in args.layouts:
        if lay not in sg.LAYOUTS:
            raise SystemExit(f"unknown layout {lay!r}")
    from research.monocle import attn_route as ar
    from research.monocle import jlens_read

    log(f"case {args.case}, the other question from {args.other_case}")
    log(f"layouts {args.layouts}, layers {args.layers}")
    # `load_eager` is the loader `input_gate` uses, and the gap on record for
    # `image_first` came from that gate. The same loader keeps the control
    # comparable.
    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    out_dir = Path(args.out_dir) / args.case
    df, _ = run(model, lens_model, proc, tmpl,
             case=args.case, other_case=args.other_case,
             n_images=args.n_images, seed=args.seed,
             layers=tuple(args.layers), layouts=tuple(args.layouts),
             device=args.device, out_dir=out_dir)
    summary = report(df)
    (out_dir / "summary.json").write_text(json.dumps(
        {"case": args.case, "other_case": args.other_case,
         "n_images": args.n_images, "seed": args.seed,
         "summary": summary}, indent=1))
    log(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
