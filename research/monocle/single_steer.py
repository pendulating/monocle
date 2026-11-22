"""Single-image steering through the Jacobian lens (the reframed causal rung).

The pairwise steering study (`jlens_steer.py`) pulled the More-vs-Less LABEL
direction back through the fitted transport and found a structural null: the
averaged Jacobian carries about 5 percent of that direction. This module
reframes the test on the user's 2026-09-01 direction:

1. One image per forward, not a pair.
2. One SINGLE-TOKEN polar word pair per case (see ``CASE_WORDS``). The lens
   reads one next-token distribution, so a multi-token label only shows its
   first piece. The polar words remove that limit.
3. The steering direction is ``d = grad(logit_pos - logit_neg)`` at the
   answer position, under the case's own single-image question.
4. The dose-response is the model's OWN judgment of the SAME image:
   ``logit_pos - logit_neg`` and the restricted argmax, as a function of
   the injection dose alpha. A matched random control direction shows
   specificity.

Modes
-----
``--coverage``   The cheap structural gate, before any sweep. For each case:
                 compute ``d`` from calibration images, then report the
                 relative reconstruction error of both pull-back forms at
                 each layer. When the pinv error stays near 1.0, the
                 transport cannot carry the direction and a sweep can only
                 confirm a null.
``--smoke``      2 calib + 2 eval images, 1 layer, 3 alphas, with asserts.
``(default)``    The full sweep for one case: a depth read at alpha=0 (the
                 lens readout of p_pos/p_neg per layer), then the steering
                 grid ``layers x alphas x {signal, control}``.

The prompt is NOT a registered-run prompt: no single-image run is registered.
The text mirrors the canonical contract shape (subject line, question, label
list, JSON instruction) and the forward is teacher-forced with the same
``{"answer": "`` prefix, so the last position emits the answer word.

Env: klara, .venv-nightly + LD_PRELOAD (see research/monocle/monocle.sub). The pure
helpers are CPU-testable (tests/test_single_steer.py).

Usage:
    sbatch --time=1:00:00 research/monocle/monocle.sub monocle.single_steer --smoke
    sbatch --time=2:00:00 research/monocle/monocle.sub monocle.single_steer --coverage
    sbatch --time=4:00:00 research/monocle/monocle.sub monocle.single_steer \\
        --case subway_safety --n-images 60 --pullback pinv
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
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402
from research.monocle import canonical  # noqa: E402  (CPU-safe)
from research.monocle.jlens_steer import (  # noqa: E402  (CPU-safe pure helpers)
    all_image_token_positions,
    build_layer_vectors,
    load_lens,
    load_model_and_lens,
    make_injection_hook,
    parse_float_list,
    parse_int_list,
    scale_from_samples,
    split_calibration_eval,
    steered_answer_logits,
)

CASE_DEFAULT = "subway_safety"
KIND_DEFAULT = "proxy"
LENS_DEFAULT = str(REPO / "outputs/_monocle/jlens/mm_all47/gemma4_12b_lens.pt")
OUT_DIR_DEFAULT = str(REPO / "outputs/_monocle/single_steer")

FORCE_PREFIX = canonical.FORCE_PREFIX
SEED = 777

DEFAULT_LAYERS = "24,36,41,42,45"
#: The injection arms. ``signal`` is the pinv pull-back of the polar
#: direction and ``control`` its matched random pull-back. ``axis`` is the
#: judgment axis fitted on the calibration images (see `judgment_axis`), and
#: ``control_raw`` a random unit direction with no pull-back, its control.
#: In ``--positions patches`` the axis is fitted on the patch residuals (every
#: patch row carries its image's judgment) and injected at every patch.
#: ``grad`` is the mean unit gradient of the contrast at the answer position
#: over the calibration images, read from ``--grad-vectors`` (the output of
#: `monocle.grad_align`).
ARMS = ("signal", "control", "axis", "control_raw", "grad")
DEFAULT_ALPHAS = "-16,-8,-4,-2,0,2,4,8,16"

#: One single-token polar word pair per case. The positive word names the
#: attribute pole of the case's recorded question; the negative word names the
#: opposite pole. Every word MUST be one tokenizer token — `polar_token_ids`
#: enforces this at run time. Verified 2026-09-01 on the gemma-4-12b
#: tokenizer; "appealing", "dislike", "neglected", "inviting", "rundown" and
#: "shabby" all split into 2+ pieces and are not usable.
CASE_WORDS: dict[str, tuple[str, str]] = {
    "subway_safety": ("safe", "dangerous"),
    "libraries": ("maintained", "worn"),
    "schools": ("good", "bad"),
    "road_quality": ("smooth", "damaged"),
    "parks_plazas": ("maintained", "worn"),
    "restaurants": ("delicious", "gross"),
    "street_photography": ("beautiful", "ugly"),
}

#: The subject line, from the case's recorded pairwise question.
CASE_SUBJECTS: dict[str, str] = {
    "subway_safety": "a New York City subway station entrance",
    "libraries": "a New York City public library building",
    "schools": "a New York City K-12 school",
    "road_quality": "a New York City roadway",
    "parks_plazas": "a New York City park or public plaza",
    "restaurants": "a New York City restaurant",
    "street_photography": "a New York City block",
}

#: The single-image question. Each question names both poles, so the label
#: list repeats the words the question uses.
CASE_QUESTIONS: dict[str, str] = {
    "subway_safety": "Does the station entrance look safe or dangerous?",
    "libraries": "Does the library building look maintained or worn?",
    "schools": "Does this look like a good school or a bad school?",
    "road_quality": "Does the roadway look smooth or damaged?",
    "parks_plazas": "Does it look maintained or worn?",
    "restaurants": "Does eating at this restaurant look delicious or gross?",
    "street_photography": ("As a location for a street photography photoshoot, "
                           "does this block look beautiful or ugly?"),
}

USER_TEMPLATE = """This image shows {subject}.
{question}

Return exactly one label from:
- {pos}
- {neg}

Respond with a JSON object: {{"answer": "<label>"}}.
"""


def log(msg: str) -> None:
    print(f"[single-steer] {msg}", flush=True)


# ---------------------------------------------------------------------------
# A) Pure, CPU-testable core
# ---------------------------------------------------------------------------
def user_text(case: str) -> str:
    """The single-image user text for one case."""
    if case not in CASE_WORDS:
        raise ValueError(f"unknown case {case!r}")
    pos, neg = CASE_WORDS[case]
    return USER_TEMPLATE.format(
        subject=CASE_SUBJECTS[case], question=CASE_QUESTIONS[case],
        pos=pos, neg=neg)


def polar_token_ids(tokenizer: Any, case: str) -> tuple[int, int]:
    """(pos id, neg id) for the case's word pair.

    Raises when a word is not exactly one token, or when the two ids collide.
    The single-token demand is the point of the reframe: the lens reads one
    next-token distribution, so a multi-token word only shows its first piece.
    """
    pos, neg = CASE_WORDS[case]
    ids = []
    for w in (pos, neg):
        toks = tokenizer.encode(w, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(
                f"{case}: answer word {w!r} is {len(toks)} tokens ({toks}); "
                f"pick a single-token word")
        ids.append(int(toks[0]))
    if ids[0] == ids[1]:
        raise ValueError(f"{case}: pos and neg share token id {ids[0]}")
    return ids[0], ids[1]


def polar_variant_ids(
    tokenizer: Any, case: str,
) -> tuple[set[int], set[int]]:
    """All single-token surface forms of each pole: bare and space-prefixed.

    The steered model sometimes emits the polar word WITH a leading space
    (a different token id) inside the JSON string. That is the correct
    answer with a formatting artifact, so the contract metric must count
    it. A variant that is not a single token is skipped.
    """
    out: list[set[int]] = []
    for w in CASE_WORDS[case]:
        ids: set[int] = set()
        for form in (w, " " + w):
            toks = tokenizer.encode(form, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(int(toks[0]))
        if not ids:
            raise ValueError(f"{case}: no single-token form of {w!r}")
        out.append(ids)
    if out[0] & out[1]:
        raise ValueError(f"{case}: pole variant sets overlap: {out}")
    return out[0], out[1]


def judgment_axis(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The unit regression direction of the judgment on the residual.

    ``h`` is ``[n, d_model]`` (the answer residual of n calibration images)
    and ``y`` is ``[n]`` (their own ``logit_diff``). The direction is the
    centered cross-moment ``sum_i (y_i - mean y) (h_i - mean h)``, unit
    normalised. Its sign points toward the positive pole. This is the axis
    the images themselves move along; `steer-coord.md` shows the pinv
    vector is orthogonal to it.
    """
    h = h.float()
    y = y.float()
    if h.ndim != 2 or y.ndim != 1 or h.shape[0] != y.shape[0]:
        raise ValueError("h must be [n, d] and y must be [n]")
    if h.shape[0] < 3:
        raise ValueError("the axis needs at least 3 images")
    yc = y - y.mean()
    if float(yc.abs().max()) == 0.0:
        raise ValueError("the judgment is constant over the calibration set")
    m = (yc[:, None] * (h - h.mean(0, keepdim=True))).sum(0)
    return m / (m.norm() + 1e-12)


def score_polar_logits(
    logits: torch.Tensor, tok_pos: int, tok_neg: int,
) -> dict:
    """Metrics of one answer-position logit vector on the polar pair.

    ``logit_diff`` is the judgment scale (positive = the positive pole).
    ``argmax_pos`` is the restricted argmax over the two words.
    ``top_token_id`` records the UNRESTRICTED argmax, so a collapse of the
    answer distribution (the failure mode of the raw-scale grid) shows up.
    """
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    return {
        "p_pos": float(probs[tok_pos]),
        "p_neg": float(probs[tok_neg]),
        "logit_diff": float(logits[tok_pos] - logits[tok_neg]),
        "argmax_pos": bool(float(logits[tok_pos]) >= float(logits[tok_neg])),
        "top_token_id": int(logits.argmax()),
    }


def unique_image_paths(rows: list[dict]) -> list[str]:
    """Unique single-image paths from pairwise rows, in first-seen order.

    Each pairwise row contributes its left then its right presented path. The
    order is deterministic, so the calibration/eval split stays disjoint
    across resumes.
    """
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        for key in ("presented_left_path", "presented_right_path"):
            p = str(row.get(key, "") or "").strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def summarize(
    df: pd.DataFrame,
    pos_ids: Optional[set[int]] = None,
    neg_ids: Optional[set[int]] = None,
) -> dict:
    """Aggregate long rows per (direction, layer, alpha).

    With the polar variant-id sets (see `polar_variant_ids`), each cell also
    carries:

    - ``frac_top_pos`` / ``frac_top_neg`` — the fraction of forwards whose
      UNRESTRICTED top token is a surface form of that pole. These are the
      semantic flip fractions: what the model actually says.
    - ``frac_on_contract`` — their sum. A cell below ~1.0 is (partly)
      off-contract: its logit_diff moves inside a broken answer
      distribution and is not, by itself, a steering result.
    """
    summary: dict = {}
    for (direction, layer, alpha), g in df.groupby(
            ["direction", "layer", "alpha"], sort=True):
        cell = {
            "direction": direction, "layer": int(layer), "alpha": float(alpha),
            "n": int(len(g)),
            "mean_logit_diff": round(float(g["logit_diff"].mean()), 4),
            "frac_argmax_pos": round(float(g["argmax_pos"].mean()), 4),
            "mean_p_pos": round(float(g["p_pos"].mean()), 4),
            "mean_p_neg": round(float(g["p_neg"].mean()), 4),
            "n_top_tokens": int(g["top_token_id"].nunique()),
        }
        if pos_ids is not None and neg_ids is not None:
            top_pos = g["top_token_id"].isin(list(pos_ids))
            top_neg = g["top_token_id"].isin(list(neg_ids))
            cell["frac_top_pos"] = round(float(top_pos.mean()), 4)
            cell["frac_top_neg"] = round(float(top_neg.mean()), 4)
            cell["frac_on_contract"] = round(
                float((top_pos | top_neg).mean()), 4)
        summary[f"{direction}/L{layer}/a{alpha:+g}"] = cell
    return summary


def print_dose_response(
    df: pd.DataFrame, layers: list[int], alphas: list[float],
    arms: tuple[str, ...] | list[str] = ("signal", "control"),
) -> None:
    """Plain-text dose-response table per layer, one column per arm."""
    for layer in layers:
        log("=" * (10 + 25 * len(arms)))
        log(f"DOSE-RESPONSE  layer {layer}   (logit_diff | frac pos)")
        log(f"  {'alpha':>7} | " + " | ".join(f"{a:^22}" for a in arms))
        for alpha in alphas:
            cells = []
            for direction in arms:
                g = df[(df["layer"] == layer) & (df["alpha"] == alpha)
                       & (df["direction"] == direction)]
                if len(g):
                    cells.append(
                        f"{g['logit_diff'].mean():+7.3f}  "
                        f"{g['argmax_pos'].mean():.3f}")
                else:
                    cells.append(" " * 22)
            tag = "  <- baseline" if alpha == 0 else ""
            log(f"  {alpha:>+7g} | " + " | ".join(cells) + tag)


# ---------------------------------------------------------------------------
# B) Prompt build (GPU path; pure text parts testable)
# ---------------------------------------------------------------------------
def render_single_chat(
    tokenizer: Any, tmpl: str, text: str,
    *, force_prefix: str = FORCE_PREFIX,
) -> str:
    """Chat text for one single-image forward, with the teacher-forced prefix.

    One user turn, image block first (the gemma-4 card asks the message to
    open with the image), then the text. No system turn — this matches the
    canonical contract.
    """
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": text}],
    }]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template=tmpl)
    return rendered + force_prefix


def strip_image_mask(inputs: dict) -> dict:
    """Drop the multimodal token-type ids, so the model builds a CAUSAL mask
    over the image tokens. gemma-4-12b reads ``use_bidirectional_attention
    == "vision"`` and lets every image token attend to every other from
    block 0; without the type ids the block ids are all -1 and the image
    block is causal. An out-of-distribution forward: the model was trained
    bidirectional."""
    inputs.pop("mm_token_type_ids", None)
    inputs.pop("token_type_ids", None)
    return inputs


IMAGE_MASK_MODES = ("bidirectional", "causal", "self")


def image_mask_mode(args) -> str:
    """The mask mode from ``--image-mask``, or ``causal`` from the old flag."""
    mode = getattr(args, "image_mask", None) or "bidirectional"
    if getattr(args, "causal_image", False):
        mode = "causal"
    if mode not in IMAGE_MASK_MODES:
        raise ValueError(f"image mask {mode!r}; choose from {IMAGE_MASK_MODES}")
    return mode


def install_image_mask(model, image_token_id: int, mode: str) -> None:
    """Set how the image tokens attend to each other in the decoder.

    | mode | image token reads |
    |---|---|
    | bidirectional | every image token (the trained mask) |
    | causal | the earlier image tokens only |
    | self | itself and the text before it, no other image token |

    A forward pre-hook on the model records the image positions and pops the
    multimodal type ids, so the block overlay never applies. For ``self`` an
    and-mask goes into ``create_masks_for_generate`` of modeling_gemma4, the
    module-level name the forward calls (modeling_gemma4_unified for the
    12B, modeling_gemma4 for the E-series). The and-mask forbids attention from
    an image query to a different image key. The sliding and the full
    layers get the same and-mask. The 3 modes are out of distribution for the
    image block except the first.
    """
    if mode not in IMAGE_MASK_MODES:
        raise ValueError(f"image mask {mode!r}; choose from {IMAGE_MASK_MODES}")
    if mode == "bidirectional":
        return
    # The 12B class lives in modeling_gemma4_unified and the E-series in
    # modeling_gemma4; each module imports its own create_masks_for_generate.
    # Patch the module the model's class lives in, not a fixed one.
    mg = sys.modules[type(model).__module__]
    if not hasattr(mg, "create_masks_for_generate"):
        raise RuntimeError(
            f"{mg.__name__} has no create_masks_for_generate to wrap")

    state: dict = {"image": None}

    def pre(module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is not None:
            state["image"] = ids == image_token_id
        kwargs.pop("mm_token_type_ids", None)
        kwargs.pop("token_type_ids", None)
        return args, kwargs

    model.register_forward_pre_hook(pre, with_kwargs=True)
    if mode == "causal":
        return

    impl = model.config.get_text_config()._attn_implementation
    if impl not in ("sdpa", "eager", "flex_attention"):
        raise RuntimeError(
            f"attention {impl!r} does not apply an and-mask; use sdpa or eager")
    orig = mg.create_masks_for_generate
    if getattr(orig, "_monocle_self_mask", False):
        return

    def self_only(batch_idx, head_idx, q_idx, kv_idx):
        img = state["image"]
        q_img = img[batch_idx, q_idx]
        k_img = img[batch_idx, kv_idx]
        return ~(q_img & k_img & (q_idx != kv_idx))

    def wrapped(**kw):
        if kw.get("and_mask_function") is not None:
            raise RuntimeError("the model already passes an and-mask")
        kw["and_mask_function"] = self_only
        return orig(**kw)

    wrapped._monocle_self_mask = True  # type: ignore[attr-defined]
    mg.create_masks_for_generate = wrapped


def make_build_inputs(proc, tmpl, case: str, device: str = "cuda:0",
                      causal_image: bool = False):
    """Single-image input builder, closing over processor + template."""
    from PIL import Image

    text = render_single_chat(proc.tokenizer, tmpl, user_text(case))

    def build(image_path: str) -> dict:
        img = Image.open(image_path).convert("RGB")
        out = dict(proc(
            text=[text], images=[[img]], return_tensors="pt").to(device))
        return strip_image_mask(out) if causal_image else out

    return build


def load_images(case: str, kind: str = KIND_DEFAULT) -> list[str]:
    """Unique single-image paths of the case's registered run, seed order."""
    df = canonical.load_results(case, kind, columns=[
        "pair_id", "repeat_idx",
        "presented_left_path", "presented_right_path"])
    df = df[df["repeat_idx"] == 0]
    return unique_image_paths(df.to_dict("records"))


# ---------------------------------------------------------------------------
# C) Direction + scale calibration (GPU)
# ---------------------------------------------------------------------------
def compute_direction_and_scales(
    model, lens_model, build, calib_paths: list[str],
    layers: list[int], tok_pos: int, tok_neg: int,
    positions_mode: str = "answer",
    image_token_id: Optional[int] = None,
    scale_mode: str = "centered",
    extras: Optional[dict] = None,
) -> tuple[torch.Tensor, dict[int, float], dict[int, dict[str, float]]]:
    """Unit steering direction ``d`` + per-layer injection scales.

    When ``extras`` is a dict, it receives ``resid[l]`` (the residual
    samples per layer, ``[n, d_model]`` in answer mode) and ``logit_diff``
    (the calibration images' own judgment), so a caller can fit the
    judgment axis on the same images.

    One forward per calibration image under the case's own question.
    ``d_i = d(logit_pos - logit_neg)/dh_final`` at the answer position, by
    autograd through ``lens_model.unembed`` (final norm + lm_head + softcap).
    Unit-normalise, average, renormalise. Scales follow `jlens_steer`:
    ``centered`` is the median centered residual norm (the informative
    signal); ``raw`` reproduces a collapse and stays only for reference.
    """
    if positions_mode == "patches" and image_token_id is None:
        raise ValueError("positions_mode='patches' requires image_token_id")
    try:
        from jlens.hooks import ActivationRecorder
    except ModuleNotFoundError:
        sys.path.insert(0, str(REPO / "sub/jacobian-lens"))
        from jlens.hooks import ActivationRecorder

    final = lens_model.n_layers - 1
    record_at = sorted(set(layers) | {final})
    d_sum: Optional[torch.Tensor] = None
    resid_samples: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    image_means: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    calib_ld: list[float] = []
    row_counts: list[int] = []

    for path in calib_paths:
        inputs = build(path)
        inputs["use_cache"] = False
        with ActivationRecorder(lens_model.layers, at=record_at) as rec:
            with torch.no_grad():
                model(**inputs)
            acts = {i: rec.activations[i] for i in record_at}

        h_final = acts[final][0, -1].detach().float().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = lens_model.unembed(h_final).float()
            contrast = logits[tok_pos] - logits[tok_neg]
            contrast.backward()
        grad = h_final.grad.detach()
        d_i = grad / (grad.norm() + 1e-12)
        d_sum = d_i.clone() if d_sum is None else d_sum + d_i
        calib_ld.append(float(logits[tok_pos] - logits[tok_neg]))

        if positions_mode == "patches":
            pos = all_image_token_positions(inputs, image_token_id)
        else:
            pos = [int(inputs["input_ids"].shape[1]) - 1]
        row_counts.append(len(pos))
        for l in layers:
            resid_samples[l].append(acts[l][0, pos, :].float().cpu())
            image_means[l].append(acts[l][0, pos, :].float().mean(0).cpu())

    assert d_sum is not None
    d = (d_sum / (d_sum.norm() + 1e-12)).float().cpu()
    scales: dict[int, float] = {}
    scale_report: dict[int, dict[str, float]] = {}
    for l in layers:
        h = torch.cat(resid_samples[l], dim=0)
        scale_report[l] = {
            "raw": scale_from_samples(h, "raw"),
            "centered": scale_from_samples(h, "centered"),
            # The spread of the per-image MEAN residual over the injected
            # positions. In answer mode it equals ``centered``. In patches
            # mode a uniform injection moves the mean patch vector by the
            # full dose, so the dose must be measured against how far that
            # mean moves between images, not against one token's spread.
            "centered_mean": scale_from_samples(
                torch.stack(image_means[l]), "centered"),
        }
        scales[l] = scale_report[l][scale_mode]
    if extras is not None:
        extras["resid"] = {l: torch.cat(resid_samples[l], dim=0)
                           for l in layers}
        extras["logit_diff"] = torch.tensor(calib_ld)
        # One judgment per residual row. In answer mode that is one per
        # image; in patches mode every patch row of an image carries the
        # image's judgment, so the axis regresses the patch residual on it.
        extras["y_rows"] = torch.repeat_interleave(
            torch.tensor(calib_ld), torch.tensor(row_counts))
    return d, scales, scale_report


# ---------------------------------------------------------------------------
# D) Depth read at alpha=0 (GPU)
# ---------------------------------------------------------------------------
def depth_read_rows(
    lens, lens_model, acts: dict[int, torch.Tensor],
    read_layers: list[int], tok_pos: int, tok_neg: int,
    image_path: str,
) -> list[dict]:
    """Lens readout of the polar pair at the answer position, per layer.

    ``acts`` holds the clean-forward residuals. Transport one position per
    layer (cheap), unembed, softmax, record p_pos / p_neg / the restricted
    argmax.
    """
    rows: list[dict] = []
    for l in read_layers:
        h = acts[l][0, -1].float()
        t = lens.transport(h, l).to(h.device)
        with torch.no_grad():
            logits = lens_model.unembed(t).float()
        m = score_polar_logits(logits, tok_pos, tok_neg)
        rows.append({"image_path": image_path, "layer": int(l), **m})
    return rows


# ---------------------------------------------------------------------------
# E) Coverage mode (the structural gate)
# ---------------------------------------------------------------------------
def run_coverage(args: argparse.Namespace) -> int:
    """Reconstruction errors of both pull-backs, per case per layer.

    The question this answers: does the transport carry the case's polar
    direction any better than it carried the pairwise label direction
    (err_pinv about 0.998)? When it does not, a steering sweep is
    structurally closed before it runs.
    """
    torch.manual_seed(SEED)
    layers = parse_int_list(args.layers)
    cases = (list(CASE_WORDS) if args.cases == "all"
             else [c.strip() for c in args.cases.split(",") if c.strip()])
    for c in cases:
        if c not in CASE_WORDS:
            raise ValueError(f"unknown case {c!r}")

    lens = load_lens(args.lens)
    missing = [l for l in layers if l not in lens.jacobians]
    if missing:
        raise ValueError(f"layers {missing} not fitted ({lens.source_layers})")
    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(
        args.model_dir)

    out: dict = {"lens": args.lens, "layers": layers, "n_calib": args.n_calib,
                 "cases": {}}
    for case in cases:
        t0 = time.time()
        tok_pos, tok_neg = polar_token_ids(proc.tokenizer, case)
        paths = load_images(case, args.kind)
        calib, _ = split_calibration_eval(
            [{"p": p} for p in paths], args.n_calib, 0)
        build = make_build_inputs(proc, tmpl, case)
        d, _, _ = compute_direction_and_scales(
            model, lens_model, build, [r["p"] for r in calib],
            layers, tok_pos, tok_neg,
            scale_mode=args.scale)
        report: dict = {}
        build_layer_vectors(
            lens, d, d, layers, pullback="pinv", report=report)
        per_layer = {}
        for l in layers:
            errs = report["signal"][l]
            frac = float(np.sqrt(max(0.0, 1.0 - errs["err_pinv"] ** 2)))
            per_layer[str(l)] = {**errs, "rowspace_frac": round(frac, 4)}
        out["cases"][case] = {
            "words": CASE_WORDS[case],
            "token_ids": [tok_pos, tok_neg],
            "per_layer": per_layer,
        }
        log(f"{case}: " + " ".join(
            f"L{l}: eT={per_layer[str(l)]['err_transpose']:.4f} "
            f"eP={per_layer[str(l)]['err_pinv']:.4f} "
            f"row={per_layer[str(l)]['rowspace_frac']:.3f}"
            for l in layers) + f"  ({time.time() - t0:.0f}s)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "coverage.json"
    path.write_text(json.dumps(out, indent=2))
    log(f"wrote {path}")
    return 0


# ---------------------------------------------------------------------------
# F) Full experiment
# ---------------------------------------------------------------------------
def run_experiment(args: argparse.Namespace) -> int:
    torch.manual_seed(SEED)
    layers = parse_int_list(args.layers)
    alphas = parse_float_list(args.alphas)
    out_dir = Path(args.out_dir) / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "steer_results.parquet"
    depth_path = out_dir / "depth.parquet"
    summary_path = out_dir / "summary.json"

    log(f"case={args.case} layers={layers} alphas={alphas} "
        f"positions={args.positions} scale={args.scale} "
        f"pullback={args.pullback} n_calib={args.n_calib} "
        f"n_images={args.n_images}")

    if args.no_lens:
        bad = [a for a in args.arms.split(",") if a.strip() in ("signal", "control")]
        if bad:
            raise ValueError(f"--no-lens cannot run the lens arms {bad}")
        lens = None
        read_layers = list(layers)
        log("no lens: the axis and the raw random arms only, no depth read")
    else:
        lens = load_lens(args.lens)
        log(f"lens: {lens!r}")
        read_layers = sorted(lens.jacobians)
        missing = [l for l in layers if l not in lens.jacobians]
        if missing:
            raise ValueError(f"layers {missing} not fitted ({lens.source_layers})")

    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(
        args.model_dir)
    tok_pos, tok_neg = polar_token_ids(proc.tokenizer, args.case)
    log(f"polar words {CASE_WORDS[args.case]} -> ids ({tok_pos}, {tok_neg})")

    paths = load_images(args.case, args.kind)
    calib, eval_rows = split_calibration_eval(
        [{"p": p} for p in paths], args.n_calib, args.n_images)
    calib_paths = [r["p"] for r in calib]
    eval_paths = [r["p"] for r in eval_rows]
    log(f"images: {len(calib_paths)} calibration + {len(eval_paths)} eval "
        f"(disjoint; {len(paths)} unique in the run)")

    mask_mode = image_mask_mode(args)
    build = make_build_inputs(proc, tmpl, args.case,
                              causal_image=mask_mode != "bidirectional")
    install_image_mask(model, image_token_id, mask_mode)
    log(f"image mask: {mask_mode}")

    # ---- calibration -------------------------------------------------------
    t0 = time.time()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise ValueError(f"unknown arms {bad}; choose from {ARMS}")
    extras: dict = {}
    d_signal, scales, scale_report = compute_direction_and_scales(
        model, lens_model, build, calib_paths,
        layers, tok_pos, tok_neg,
        positions_mode=args.positions, image_token_id=image_token_id,
        scale_mode=args.scale, extras=extras)
    if not torch.isfinite(d_signal).all():
        raise RuntimeError("non-finite steering direction")
    d_model = int(d_signal.numel())
    gen = torch.Generator().manual_seed(SEED)
    d_control = torch.randn(d_model, generator=gen)
    d_control = d_control / d_control.norm()
    log(f"direction ready | scale={args.scale} | " + " ".join(
        f"L{l}: raw={scale_report[l]['raw']:.1f} "
        f"centered={scale_report[l]['centered']:.2f} "
        f"centered_mean={scale_report[l]['centered_mean']:.2f}"
        for l in layers)
        + f" | calib {time.time() - t0:.0f}s")

    pb_report: dict = {}
    if lens is not None:
        vecs = build_layer_vectors(
            lens, d_signal, d_control, layers,
            pullback=args.pullback, report=pb_report)
        log(f"pull-back={args.pullback}; reconstruction errors: "
            + " ".join(f"L{l}={pb_report['signal'][l]}" for l in layers))
    else:
        vecs = {}
    # The judgment axis, fitted on the calibration images alone, and a raw
    # random control at the same scale. Neither goes through the lens.
    if "axis" in arms:
        vecs["axis"] = {l: judgment_axis(extras["resid"][l],
                                         extras["y_rows"])
                        for l in layers}
        log("axis: calibration logit_diff range "
            f"{float(extras['logit_diff'].min()):+.2f} to "
            f"{float(extras['logit_diff'].max()):+.2f}"
            + ("; cos(axis, v) = " + " ".join(
                f"L{l}={float(vecs['axis'][l] @ vecs['signal'][l]):+.3f}"
                for l in layers) if "signal" in vecs else ""))
    if "grad" in arms:
        if not args.grad_vectors:
            raise ValueError("the grad arm needs --grad-vectors")
        gv = np.load(args.grad_vectors)
        vecs["grad"] = {l: torch.tensor(gv[f"v_grad_L{l}"]).float()
                        for l in layers}
        log("grad arm from " + args.grad_vectors + "; cos(grad, v) = "
            + " ".join(f"L{l}={float(vecs['grad'][l] @ vecs['signal'][l]):+.3f}"
                       for l in layers))
    if "control_raw" in arms:
        gen_raw = torch.Generator().manual_seed(SEED + 1)
        u = torch.randn(d_model, generator=gen_raw)
        u = u / u.norm()
        vecs["control_raw"] = {l: u.clone() for l in layers}
    np.savez(out_dir / "vectors.npz",
             d=d_signal.numpy(), d_control=d_control.numpy(),
             layers=np.array(layers),
             scales=np.array([scales[l] for l in layers]),
             **{f"v_{name}_L{l}": vecs[name][l].numpy()
                for name in arms for l in layers})
    dev_vecs = {name: {l: vecs[name][l].to("cuda:0") for l in layers}
                for name in arms}

    # ---- depth read + steering sweep ---------------------------------------
    try:
        from jlens.hooks import ActivationRecorder
    except ModuleNotFoundError:
        sys.path.insert(0, str(REPO / "sub/jacobian-lens"))
        from jlens.hooks import ActivationRecorder

    done: set = set()
    records: list[dict] = []
    depth_records: list[dict] = []
    if results_path.exists() and not args.smoke:
        prev = pd.read_parquet(results_path)
        records = prev.to_dict("records")
        done = set(prev["image_path"].unique())
        log(f"resuming: {len(done)} eval images already done")
        if depth_path.exists():
            depth_records = pd.read_parquet(depth_path).to_dict("records")

    t0 = time.time()
    n_forward = 0
    processed = 0
    for path in eval_paths:
        if path in done:
            continue
        inputs = build(path)
        inputs["use_cache"] = False
        seq_len = int(inputs["input_ids"].shape[1])
        if args.positions == "patches":
            positions = all_image_token_positions(inputs, image_token_id)
        else:
            positions = [seq_len - 1]

        # clean forward with recorder: baseline metrics + depth read.
        with ActivationRecorder(lens_model.layers, at=read_layers) as rec:
            with torch.no_grad():
                out = model(**inputs)
            acts = {i: rec.activations[i] for i in read_layers}
        base_logits = out.logits[0, -1].float()
        n_forward += 1
        base = score_polar_logits(base_logits, tok_pos, tok_neg)
        if lens is not None:
            depth_records.extend(depth_read_rows(
                lens, lens_model, acts, read_layers, tok_pos, tok_neg, path))
        del acts, out

        for layer in layers:
            for direction in arms:
                v = dev_vecs[direction][layer]
                for alpha in alphas:
                    if alpha == 0:
                        m = dict(base)
                    else:
                        add_vec = float(alpha) * scales[layer] * v
                        logits = steered_answer_logits(
                            model, lens_model, inputs, layer, add_vec,
                            positions)
                        n_forward += 1
                        m = score_polar_logits(logits, tok_pos, tok_neg)
                    records.append({
                        "image_path": path, "direction": direction,
                        "layer": int(layer), "alpha": float(alpha),
                        "scale_l": scales[layer], "scale_mode": args.scale,
                        "pullback": args.pullback,
                        "positions": args.positions,
                        "base_logit_diff": base["logit_diff"],
                        "base_argmax_pos": base["argmax_pos"], **m})
        processed += 1
        if processed % 10 == 0:
            pd.DataFrame(records).to_parquet(results_path, index=False)
            pd.DataFrame(depth_records).to_parquet(depth_path, index=False)
            el = time.time() - t0
            log(f"  {processed}/{len(eval_paths)} images | {n_forward} "
                f"forwards | {el / max(n_forward, 1):.2f}s/fwd | ckpt")

    df = pd.DataFrame(records)
    df.to_parquet(results_path, index=False)
    pd.DataFrame(depth_records).to_parquet(depth_path, index=False)
    log(f"wrote {results_path} ({len(df)} rows) and {depth_path} "
        f"({len(depth_records)} rows) in {time.time() - t0:.0f}s")

    # depth summary: mean p_pos/p_neg + agreement with the model's own answer.
    ddf = pd.DataFrame(depth_records)
    base_df = df[(df["alpha"] == 0) & (df["direction"] == arms[0])
                 & (df["layer"] == layers[0])]
    own = dict(zip(base_df["image_path"], base_df["argmax_pos"]))
    if len(ddf):
        ddf["own_argmax_pos"] = ddf["image_path"].map(own)
        ddf = ddf.dropna(subset=["own_argmax_pos"])
    depth_summary = {} if not len(ddf) else {
        str(l): {
            "mean_p_pos": round(float(g["p_pos"].mean()), 4),
            "mean_p_neg": round(float(g["p_neg"].mean()), 4),
            "agree_own": round(
                float((g["argmax_pos"] == g["own_argmax_pos"]).mean()), 4),
        }
        for l, g in ddf.groupby("layer")}

    summary = {
        "case": args.case, "words": CASE_WORDS[args.case],
        "token_ids": [tok_pos, tok_neg],
        "layers": layers, "alphas": alphas,
        "positions": args.positions, "scale_mode": args.scale,
        "pullback": args.pullback, "arms": arms,
        "n_calib": args.n_calib, "n_eval": len(eval_paths),
        "scales": scales,
        "scale_report": {str(l): scale_report[l] for l in layers},
        "pullback_errors": {
            name: {str(l): errs for l, errs in per.items()}
            for name, per in pb_report.items()},
        "lens": args.lens,
        "registry": str(canonical.registry_dir(args.case, args.kind)),
        "depth": depth_summary,
        "per_cell": summarize(
            df, *polar_variant_ids(proc.tokenizer, args.case)),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {summary_path}")
    print_dose_response(df, layers, alphas, arms)
    return 0


# ---------------------------------------------------------------------------
# G) Smoke
# ---------------------------------------------------------------------------
def run_smoke(args: argparse.Namespace) -> int:
    """2 calib + 2 eval images, 1 layer, alphas {-4,0,4}, with asserts."""
    torch.manual_seed(SEED)
    layers = parse_int_list(args.layers)[:1]
    layer = layers[0]
    log(f"SMOKE: case={args.case} layer={layer}")

    lens = load_lens(args.lens)
    if layer not in lens.jacobians:
        raise ValueError(f"layer {layer} not fitted ({lens.source_layers})")
    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(
        args.model_dir)
    tok_pos, tok_neg = polar_token_ids(proc.tokenizer, args.case)
    log(f"ASSERT single-token polar words OK ({tok_pos}, {tok_neg})")

    paths = load_images(args.case, args.kind)
    calib, ev = split_calibration_eval([{"p": p} for p in paths], 2, 2)
    build = make_build_inputs(proc, tmpl, args.case)

    d, scales, report = compute_direction_and_scales(
        model, lens_model, build, [r["p"] for r in calib],
        layers, tok_pos, tok_neg, scale_mode=args.scale)
    assert torch.isfinite(d).all(), "direction is non-finite"
    log(f"ASSERT direction finite OK | scales={scales} report={report}")

    gen = torch.Generator().manual_seed(SEED)
    d_control = torch.randn(lens.d_model, generator=gen)
    d_control = d_control / d_control.norm()
    pb: dict = {}
    vecs = build_layer_vectors(lens, d, d_control, layers,
                               pullback=args.pullback, report=pb)
    log(f"pull-back errors: {pb['signal']}")
    v = vecs["signal"][layer].to("cuda:0")

    inputs = build(ev[0]["p"])
    inputs["use_cache"] = False
    positions = [int(inputs["input_ids"].shape[1]) - 1]
    la = steered_answer_logits(model, lens_model, inputs, layer, None, positions)
    _ = steered_answer_logits(
        model, lens_model, inputs, layer, 4.0 * scales[layer] * v, positions)
    lb = steered_answer_logits(model, lens_model, inputs, layer, None, positions)
    assert torch.equal(la, lb), "hook removal failed"
    log("ASSERT hook removal OK (clean forwards bit-equal)")

    m = score_polar_logits(la, tok_pos, tok_neg)
    top = proc.tokenizer.decode([m["top_token_id"]])
    log(f"baseline read: p_pos={m['p_pos']:.3f} p_neg={m['p_neg']:.3f} "
        f"logit_diff={m['logit_diff']:+.3f} top_token={top!r}")
    assert m["p_pos"] + m["p_neg"] > 0.1, (
        f"polar words carry almost no mass (p_pos+p_neg="
        f"{m['p_pos'] + m['p_neg']:.4f}); the prompt does not put the answer "
        f"on the polar pair")
    log("ASSERT polar mass OK")
    log("SMOKE complete")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--coverage", action="store_true",
                    help="Structural gate only: direction + reconstruction "
                         "errors per case, no steering sweep.")
    ap.add_argument("--cases", default="all",
                    help="Coverage mode: comma list of cases, or 'all'.")
    ap.add_argument("--case", default=CASE_DEFAULT,
                    choices=list(canonical.CASES))
    ap.add_argument("--kind", default=KIND_DEFAULT,
                    choices=list(canonical.KINDS))
    ap.add_argument("--lens", default=LENS_DEFAULT)
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-images", type=int, default=60, help="eval images")
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--layers", default=DEFAULT_LAYERS)
    ap.add_argument("--alphas", default=DEFAULT_ALPHAS)
    ap.add_argument("--positions", choices=["answer", "patches"],
                    default="answer")
    ap.add_argument("--scale", choices=["raw", "centered", "centered_mean"],
                    default="centered",
                    help="centered_mean: the between-image spread of the "
                         "mean injected residual (use it with --positions "
                         "patches)")
    ap.add_argument("--arms", default="signal,control",
                    help="comma list from " + ",".join(ARMS))
    ap.add_argument("--grad-vectors", default=None,
                    help="grad_dirs.npz from research.monocle.grad_align (grad arm)")
    ap.add_argument("--no-lens", action="store_true",
                    help="skip the lens: axis and control_raw arms only "
                         "(needed for a model with no fitted lens)")
    ap.add_argument("--causal-image", action="store_true",
                    help="strip mm_token_type_ids: a causal mask over the "
                         "image tokens (gemma-4-12b ablation)")
    ap.add_argument("--image-mask", choices=list(IMAGE_MASK_MODES),
                    default=None,
                    help="how the image tokens attend to each other; "
                         "see install_image_mask")
    ap.add_argument("--pullback", choices=["transpose", "pinv"],
                    default="pinv",
                    help="pinv is the default here: it is the only form with "
                         "transport(v) ~ d, so it is the honest test of "
                         "'steering via the jacobian'.")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        return run_smoke(args)
    if args.coverage:
        return run_coverage(args)
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
