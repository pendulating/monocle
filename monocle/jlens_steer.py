"""Rung C of phase-4 monocle science: J-vector STEERING (the causal rung).

The Jacobian lens ``J_l = E[dh_final/dh_l]`` transports a layer-l residual
FORWARD into the final residual basis (``transport(h, l) = J_l @ h``). Its
transpose pulls a final-basis direction ``d`` BACK to layer l::

    v_l = J_l^T @ d

Adding ``v_l`` to the residual at layer l should causally push the model's
output toward ``d``. Rung A (activation probe) showed the safety label is
DECODABLE from the neutral-prompt residual; rung B (the lens) showed WHERE.
This rung asks whether that direction is USED: under the NEUTRAL prompt (which
never mentions safety) we inject the safety-label direction at various depths
and measure whether the model's OWN pairwise judgments shift toward its
production (safety-prompted) labels. "Decodable" -> "used".

Pipeline
--------
1. Direction (final basis). Under the PROD (safety) prompt, at the answer
   position (teacher-forced ``{"answer": "`` prefix, seq -1), take the recorded
   final-block residual ``h_final`` (a fresh grad leaf), compute
   ``logits = lens_model.unembed(h_final)`` (final RMSNorm + lm_head + logit
   softcap, all by autograd — never by hand), form the contrast scalar
   ``c = logits[tok_More] - logits[tok_Less]`` on the FIRST tokens of "More" /
   "Less", and set ``d = dc/dh_final``. This is the direction the model itself
   moves along when it says the label. Averaged (unit-normalised) over a small
   calibration set of PROD pairs.
2. Pull-back. ``v_l = J_l^T @ d`` per layer, unit-normalised. Orientation is
   pinned against ``lens.transport``: transport computes ``J_l @ h`` (it is
   ``h @ J_l.T``), so the pull-back is ``J_l.t() @ h`` and the invariant
   ``<d, transport(v_raw, l)> == ||v_raw||^2`` must hold (checked at runtime).
3. Injection. A forward hook on ``lens_model.layers[l]`` ADDS
   ``alpha * scale_l * v_l`` at the answer position (default) or at every
   image-token position of both images (``--positions patches``). ``scale_l``
   is the median ``||h_l||`` at the answer position over the calibration pairs,
   so ``alpha`` is in units of a typical residual norm. Exactly one hook at a
   time, always removed in a ``finally``.
4. Experiment. Over eval pairs DISJOINT from calibration (calibration = head,
   eval = the next block), under the NEUTRAL prompt, sweep ``layers x alphas``
   (one forward each) and read the model's OWN logits at the answer position.
   Restricted argmax over the 4 distinguishable label first-tokens
   {Much, Less, Same, More} (MuchLess/MuchMore share the "Much" token — verified
   & recorded). Metrics vs the production label: exact agreement on the
   collapsed 4 classes, ordinal ``1 - |Δ|/4`` on the collapsed 5-point scale
   (a "Much" argmax is resolved to the extreme matching the sign of
   ``p(More)-p(Less)``), and mean ``p(More-tok) - p(Less-tok)``. ``alpha=0`` is
   the unsteered baseline. A CONTROL direction (random unit vector, seed-777)
   run through the identical pipeline shows specificity.

Env: klara, .venv-nightly + LD_PRELOAD (see monocle/monocle.sub). No GPU work
happens on the CPU test venv — the pure helpers are dependency-injectable and
covered by tests/test_jlens_steer.py.

Usage:
    sbatch --time=3:00:00 monocle/monocle.sub monocle.jlens_steer --smoke
    sbatch --time=3:00:00 monocle/monocle.sub monocle.jlens_steer \\
        --n-pairs 100 --layers 24,36,42 --alphas -8,-4,-2,0,2,4,8
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

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import MODEL_DIR_DEFAULT  # noqa: E402
from monocle import canonical  # noqa: E402  (CPU-safe)

# ---- fixed recipe (must match the activation-probe / lens work) ------------
# Pairs and prompts come from the canonical run registry, through
# `monocle.canonical`. The pre-2026-08-11 build hardcoded the 2026-06-29 subway
# run and its superseded prompt, with a persona system turn and the
# images_then_text layout. See monocle/canonical.py.
CASE_DEFAULT = "subway_safety"
KIND_DEFAULT = "proxy"
LENS_DEFAULT = str(REPO / "outputs/_monocle/jlens/mm/gemma4_12b_lens.pt")
OUT_DIR_DEFAULT = str(REPO / "outputs/_monocle/steer")

FORCE_PREFIX = canonical.FORCE_PREFIX
ORDINAL = list(canonical.ORDINAL)
#: The DISTINGUISHABLE first-token classes. MuchLess/MuchMore both begin with
#: the "Much" token, so the restricted argmax collapses them. The registered
#: runs keep abstention on, so "NotSure" joins the set at run time — see
#: `label_first_tokens`.
COLLAPSED = ["Much", "Less", "Same", "More"]
SEED = 777

DEFAULT_LAYERS = "24,36,42"
DEFAULT_ALPHAS = "-8,-4,-2,0,2,4,8"


def log(msg: str) -> None:
    print(f"[jlens-steer] {msg}", flush=True)


# ---------------------------------------------------------------------------
# A) Pure, CPU-testable core
# ---------------------------------------------------------------------------
def parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.replace(" ", "").split(",") if x != ""]


def parse_float_list(spec: str) -> list[float]:
    return [float(x) for x in spec.replace(" ", "").split(",") if x != ""]


def pull_back_raw(jacobian: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Un-normalised pull-back of a final-basis direction ``d`` to a layer.

    ``jacobian`` is ``J_l`` with ``J_l[a, b] = dh_final[a]/dh_l[b]`` (rows index
    the final/target basis, cols the layer-l/source basis). Since transport is
    ``J_l @ h`` (see :func:`orientation_residual`), the correct pull-back is
    ``J_l^T @ d`` — i.e. ``jacobian.t() @ d``. Using ``jacobian @ d`` instead
    would be the WRONG orientation and fails :func:`orientation_residual`.
    """
    return jacobian.transpose(-1, -2).to(d.dtype) @ d


def orientation_residual(
    lens: Any, jacobian: torch.Tensor, d: torch.Tensor, layer: int,
) -> float:
    """Relative error of the pull-back orientation invariant.

    With ``v = J_l^T @ d`` and transport ``T(v) = J_l @ v`` we have
    ``<d, T(v)> = d^T J_l J_l^T d = (J_l^T d)·(J_l^T d) = ||v||^2``. This holds
    ONLY for the correct (transpose) orientation; the wrong orientation
    ``v = J_l @ d`` gives ``<d, T(v)> = d^T J_l J_l d != ||J_l d||^2`` in
    general. Returns ``|lhs - rhs| / |rhs|`` (should be ~0).
    """
    v_raw = pull_back_raw(jacobian, d)
    transported = lens.transport(v_raw, layer).to(d.dtype)
    lhs = float(torch.dot(d.flatten(), transported.flatten()))
    rhs = float(torch.dot(v_raw.flatten(), v_raw.flatten()))
    return abs(lhs - rhs) / (abs(rhs) + 1e-12)


def label_first_tokens(tokenizer, cfg=None) -> dict[str, int]:
    """First-token id of each distinguishable answer class.

    Uses ``tokenizer.encode(lbl, add_special_tokens=False)[0]`` exactly as the
    direction construction does. Validates that MuchLess and MuchMore share
    their first token (which justifies the collapse) and that the classes are
    mutually distinct. Returns ``{class: token_id}``.

    With ``cfg`` (a registered run's prompt config) the abstention class joins
    the set whenever the run enabled it — which the whole consolidated battery
    does. Without ``cfg`` only the 4 ordinal classes are returned, for the
    tokenizer-level unit tests.
    """
    def first(lbl: str) -> int:
        toks = tokenizer.encode(lbl, add_special_tokens=False)
        if not toks:
            raise ValueError(f"empty tokenization for {lbl!r}")
        return int(toks[0])

    much_less = first("MuchLess")
    much_more = first("MuchMore")
    if much_less != much_more:
        raise ValueError(
            f"MuchLess/MuchMore first tokens differ ({much_less} vs "
            f"{much_more}); the 4-class collapse is invalid for this tokenizer")
    classes = list(COLLAPSED)
    if cfg is not None and canonical.not_sure_enabled(cfg):
        classes.append(canonical.not_sure_label(cfg))
    ids = {cls: first(cls) for cls in classes}
    if len(set(ids.values())) != len(classes):
        raise ValueError(f"class first-tokens not distinct: {ids}")
    return ids


def collapse_label(label: str) -> str:
    """Production 5-point label -> 4-class collapsed name."""
    return "Much" if label in ("MuchLess", "MuchMore") else label


def collapsed_ordinal_index(cls: str, p_more: float, p_less: float) -> int:
    """5-point ordinal index (0..4) for a collapsed-class argmax.

    Less/Same/More map to 1/2/3. A "Much" argmax is ambiguous between the two
    extremes (both share the token); it is resolved to MuchMore (4) when the
    same forward leans More (``p_more >= p_less``) else MuchLess (0). This uses
    only information present in the very logits being scored.
    """
    if cls == "Less":
        return 1
    if cls == "Same":
        return 2
    if cls == "More":
        return 3
    if cls == "Much":
        return 4 if p_more >= p_less else 0
    raise ValueError(f"unknown collapsed class {cls!r}")


def score_answer_logits(
    logits: torch.Tensor,
    class_ids: dict[str, int],
    tok_more: int,
    tok_less: int,
    prod_label: str,
) -> dict:
    """Score one answer-position logit vector against a production label.

    ``logits`` is ``[vocab]`` (the model's own softcapped logits at the answer
    position). Returns a flat metrics dict (restricted argmax class, collapsed
    exact agreement, collapsed ordinal agreement, and the More/Less token
    probabilities).
    """
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    p_more = float(probs[tok_more])
    p_less = float(probs[tok_less])

    order = COLLAPSED
    class_logits = torch.tensor([float(logits[class_ids[c]]) for c in order])
    steered_class = order[int(class_logits.argmax())]

    steered_idx = collapsed_ordinal_index(steered_class, p_more, p_less)
    prod_idx = ORDINAL.index(prod_label)
    prod_class = collapse_label(prod_label)

    exact = float(steered_class == prod_class)
    ordinal = 1.0 - abs(steered_idx - prod_idx) / 4.0
    return {
        "steered_class": steered_class,
        "steered_idx": steered_idx,
        "prod_class": prod_class,
        "prod_idx": prod_idx,
        "exact_collapsed": exact,
        "ordinal_collapsed": ordinal,
        "p_more": p_more,
        "p_less": p_less,
        "p_more_minus_less": p_more - p_less,
    }


def contiguous_runs(positions: list[int]) -> list[tuple[int, int]]:
    """Split a sorted position list into contiguous ``(start, end)`` runs
    (inclusive). Two-image inputs give two runs; used by ``--positions patches``
    (``extract.image_token_positions`` intentionally raises on non-contiguity)."""
    if not positions:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = positions[0]
    for p in positions[1:]:
        if p == prev + 1:
            prev = p
            continue
        runs.append((start, prev))
        start = prev = p
    runs.append((start, prev))
    return runs


def all_image_token_positions(inputs: dict, image_token_id: int) -> list[int]:
    """Every image-token sequence position (both images), no contiguity check."""
    ids = inputs["input_ids"][0]
    pos = (ids == image_token_id).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        raise RuntimeError("zero image tokens in input_ids")
    return [int(p) for p in pos.tolist()]


def make_injection_hook(add_vec: torch.Tensor, positions: list[int]):
    """Forward hook that ADDS ``add_vec`` to the block output at ``positions``.

    Handles both tensor and tuple block outputs; never mutates the original
    output in place (clones first), and returns the replacement so the normal
    forward composes with it. ``positions`` are sequence indices in batch row 0.
    """
    def hook(module, inputs, output):
        is_tuple = not torch.is_tensor(output)
        tensor = output[0] if is_tuple else output
        new = tensor.clone()
        add = add_vec.to(new.dtype).to(new.device)
        for p in positions:
            new[0, p, :] = new[0, p, :] + add
        if is_tuple:
            return (new, *tuple(output[1:]))
        return new

    return hook


def split_calibration_eval(
    rows: list[dict], n_calib: int, n_eval: int,
) -> tuple[list[dict], list[dict]]:
    """Head ``n_calib`` = calibration; the NEXT ``n_eval`` = eval. Disjoint by
    construction; raises if the frame is too short."""
    need = n_calib + n_eval
    if len(rows) < need:
        raise ValueError(
            f"need {need} pairs ({n_calib} calib + {n_eval} eval) but frame "
            f"has {len(rows)}")
    return rows[:n_calib], rows[n_calib:need]


# ---------------------------------------------------------------------------
# B) Gemma / jlens glue (imported lazily so the CPU test venv stays clean)
# ---------------------------------------------------------------------------
def load_lens(path: str):
    """Load the fitted JacobianLens, tolerating the CPU venv (jlens not pip
    installed there) with the vendored-copy sys.path fallback."""
    try:
        from jlens import JacobianLens
    except ModuleNotFoundError:
        sys.path.insert(0, str(REPO / "sub/jacobian-lens"))
        from jlens import JacobianLens
    return JacobianLens.load(path)


def load_model_and_lens(model_dir: str):
    """gemma-4 via the verified monocle recipe + jlens wrapper (force_bos=False
    — the chat template emits its own <bos>; the default True would double-BOS
    the SHARED processor tokenizer). Returns
    ``(proc, model, tmpl, lens_model, image_token_id)``."""
    try:
        import jlens
    except ModuleNotFoundError:
        sys.path.insert(0, str(REPO / "sub/jacobian-lens"))
        import jlens

    from monocle import extract

    proc, model, tmpl = extract.load_model(model_dir)
    lens_model = jlens.from_hf(model, proc.tokenizer, force_bos=False)
    image_token_id = int(model.config.image_token_id)
    return proc, model, tmpl, lens_model, image_token_id


def load_conditions(case: str = CASE_DEFAULT, kind: str = KIND_DEFAULT):
    """{cond: prompt cfg} for PROD (direction) and NEUTRAL (causal test).

    Delegates to `canonical.build_conditions`, so both arms carry the
    registered run's frame — no system turn, interleaved_labels, abstention
    guidance — and differ ONLY in the question. The old build took prod from a
    conf YAML with a persona and neutral from the GEPA generic seed with a
    different persona, so its contrast moved two things at once.
    """
    return canonical.build_conditions(case, kind, ["prod", "neutral"])


def make_build_inputs(proc, tmpl, device: str = "cuda:0"):
    """Two-image chat-input builder (the ``_gemma4_unified_chat_template`` path
    from act_probe_smoke2), closing over the processor + template."""
    from PIL import Image

    def build(row: dict, cfg) -> dict:
        left, right = canonical.presented_images(row)
        imgs = [Image.open(left).convert("RGB"),
                Image.open(right).convert("RGB")]
        inputs = canonical.build_pair_inputs(
            proc, tmpl, imgs, row, cfg,
            force_prefix=FORCE_PREFIX, device=device)
        return dict(inputs)

    return build


def load_pairs(case: str = CASE_DEFAULT, kind: str = KIND_DEFAULT) -> list[dict]:
    """Supervision rows of a registered run, read through the registry.

    Keeps one presentation per canonical pair. Abstained rows are dropped here
    (unlike the readout study): the steering direction is the More-vs-Less
    contrast, which an abstention does not place on the scale.
    """
    df = canonical.load_results(case, kind, columns=[
        "pair_id", "repeat_idx", "presented_label",
        "presented_left_path", "presented_right_path"])
    df = df[(df["repeat_idx"] == 0) & df["presented_label"].isin(ORDINAL)]
    return df.reset_index(drop=True).to_dict("records")


# ---------------------------------------------------------------------------
# C) Direction + scale calibration (GPU)
# ---------------------------------------------------------------------------
def compute_direction_and_scales(
    model, lens_model, build, calib_rows: list[dict],
    prod_cfg,
    layers: list[int], tok_more: int, tok_less: int,
    device: str = "cuda:0",
    positions_mode: str = "answer",
    image_token_id: Optional[int] = None,
) -> tuple[torch.Tensor, dict[int, float]]:
    """Final-basis steering direction ``d`` (unit) + per-layer ``scale_l``.

    One PROD forward per calibration pair. At the answer position (seq -1):
      * ``d_i = d(logit_More - logit_Less)/dh_final`` via autograd through
        ``lens_model.unembed`` (final norm + lm_head + softcap); unit-normalise
        and average, then renormalise.

    ``scale_l`` is the median CENTERED residual norm ``||h_l - mean(h_l)||``
    over the calibration residuals at the positions that will be injected
    (``positions_mode``: "answer" = seq -1, "patches" = all image-token
    positions). The raw norm is the WRONG scale on Gemma: the massive-
    activation component (~250) is ~150x the pair-specific signal (~1.7 —
    see the activation-probe geometry), so raw-scaled injections obliterate
    the computation for signal and control alike (measured: a random control
    at alpha=2 rewrote the label distribution wholesale). Centered norms make
    alpha the dose in units of the *informative* signal.
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

    for row in calib_rows:
        inputs = build(row, prod_cfg)
        inputs["use_cache"] = False
        with ActivationRecorder(lens_model.layers, at=record_at) as rec:
            with torch.no_grad():
                model(**inputs)
            acts = {i: rec.activations[i] for i in record_at}

        # direction: grad of the More-Less contrast wrt the final residual.
        h_final = acts[final][0, -1].detach().float().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = lens_model.unembed(h_final).float()
            contrast = logits[tok_more] - logits[tok_less]
            contrast.backward()
        grad = h_final.grad.detach()
        d_i = grad / (grad.norm() + 1e-12)
        d_sum = d_i.clone() if d_sum is None else d_sum + d_i

        if positions_mode == "patches":
            pos = all_image_token_positions(inputs, image_token_id)
        else:
            pos = [int(inputs["input_ids"].shape[1]) - 1]
        for l in layers:
            resid_samples[l].append(acts[l][0, pos, :].float().cpu())

    assert d_sum is not None
    d = (d_sum / (d_sum.norm() + 1e-12)).float().cpu()
    scales: dict[int, float] = {}
    for l in layers:
        h = torch.cat(resid_samples[l], dim=0)          # [n_samples, d_model]
        centered = h - h.mean(dim=0, keepdim=True)
        scales[l] = float(centered.norm(dim=-1).median())
    return d, scales


def build_layer_vectors(
    lens, d_signal: torch.Tensor, d_control: torch.Tensor, layers: list[int],
) -> dict[str, dict[int, torch.Tensor]]:
    """Unit pull-back vectors ``v_l`` per direction per layer, and verify the
    transport orientation invariant on each."""
    out: dict[str, dict[int, torch.Tensor]] = {"signal": {}, "control": {}}
    for name, d in (("signal", d_signal), ("control", d_control)):
        for l in layers:
            J = lens.jacobians[l].float()
            resid = orientation_residual(lens, J, d, l)
            if resid > 1e-3:
                raise RuntimeError(
                    f"orientation invariant failed at layer {l} "
                    f"({name}): rel err {resid:.2e}")
            v_raw = pull_back_raw(J, d)
            out[name][l] = (v_raw / (v_raw.norm() + 1e-12)).float()
    return out


# ---------------------------------------------------------------------------
# D) Steered forward (GPU)
# ---------------------------------------------------------------------------
def steered_answer_logits(
    model, lens_model, inputs: dict, layer: int,
    add_vec: Optional[torch.Tensor], positions: list[int],
) -> torch.Tensor:
    """One forward with (optionally) an injection hook on ``layers[layer]``;
    returns the model's own logits at the answer position (seq -1), fp32.

    The hook is always removed in a ``finally`` so exactly one is live per
    forward and no state leaks to the next call.
    """
    handle = None
    try:
        if add_vec is not None:
            hook = make_injection_hook(add_vec, positions)
            handle = lens_model.layers[layer].register_forward_hook(hook)
        with torch.inference_mode():
            out = model(**{**inputs, "use_cache": False})
    finally:
        if handle is not None:
            handle.remove()
    logits = out.logits[0, -1].float()
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite answer logits")
    return logits


# ---------------------------------------------------------------------------
# E) Experiment driver
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """Aggregate long rows per (direction, layer, alpha)."""
    summary: dict = {}
    for (direction, layer, alpha), g in df.groupby(
            ["direction", "layer", "alpha"], sort=True):
        dist = g["steered_class"].value_counts().to_dict()
        summary[f"{direction}/L{layer}/a{alpha:+g}"] = {
            "direction": direction, "layer": int(layer), "alpha": float(alpha),
            "n": int(len(g)),
            "exact_collapsed": round(float(g["exact_collapsed"].mean()), 4),
            "ordinal_collapsed": round(float(g["ordinal_collapsed"].mean()), 4),
            "p_more_minus_less": round(float(g["p_more_minus_less"].mean()), 4),
            "label_dist": {k: int(v) for k, v in dist.items()},
        }
    return summary


def print_dose_response(df: pd.DataFrame, layers: list[int], alphas: list[float]) -> None:
    """Plain-text dose-response table per layer (signal vs control)."""
    for layer in layers:
        log("=" * 72)
        log(f"DOSE-RESPONSE  layer {layer}   (ordinal | exact | p_More-p_Less)")
        log(f"  {'alpha':>7} | {'signal':^26} | {'control':^26}")
        for alpha in alphas:
            cells = []
            for direction in ("signal", "control"):
                g = df[(df["layer"] == layer) & (df["alpha"] == alpha)
                       & (df["direction"] == direction)]
                if len(g):
                    cells.append(
                        f"{g['ordinal_collapsed'].mean():.3f} "
                        f"{g['exact_collapsed'].mean():.3f} "
                        f"{g['p_more_minus_less'].mean():+.3f}")
                else:
                    cells.append(" " * 26)
            tag = "  <- baseline" if alpha == 0 else ""
            log(f"  {alpha:>+7g} | {cells[0]:^26} | {cells[1]:^26}{tag}")


def run_experiment(args: argparse.Namespace) -> int:
    torch.manual_seed(SEED)
    device = "cuda:0"
    layers = parse_int_list(args.layers)
    alphas = parse_float_list(args.alphas)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "steer_results.parquet"
    summary_path = out_dir / "steer_summary.json"

    log(f"layers={layers} alphas={alphas} positions={args.positions} "
        f"n_calib={args.n_calib} n_pairs={args.n_pairs} smoke={args.smoke}")

    conditions = load_conditions(args.case, args.kind)
    prod_cfg, neutral_cfg = conditions["prod"], conditions["neutral"]

    lens = load_lens(args.lens)
    log(f"lens: {lens!r}")
    missing = [l for l in layers if l not in lens.jacobians]
    if missing:
        raise ValueError(
            f"layers {missing} not fitted; lens has {lens.source_layers}")

    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(
        args.model_dir)
    tokenizer = proc.tokenizer
    class_ids = label_first_tokens(tokenizer, prod_cfg)
    tok_more, tok_less = class_ids["More"], class_ids["Less"]
    log(f"label first-tokens: {class_ids}  (More={tok_more} Less={tok_less}; "
        f"MuchLess==MuchMore first token verified)")

    rows = load_pairs(args.case, args.kind)
    calib_rows, eval_rows = split_calibration_eval(
        rows, args.n_calib, args.n_pairs)
    log(f"pairs: {len(calib_rows)} calibration (head) + {len(eval_rows)} eval "
        f"(next block); disjoint")

    build = make_build_inputs(proc, tmpl, device)

    # ---- calibration: direction + scales -----------------------------------
    t0 = time.time()
    d_signal, scales = compute_direction_and_scales(
        model, lens_model, build, calib_rows, prod_cfg,
        layers, tok_more, tok_less, device,
        positions_mode=args.positions, image_token_id=image_token_id)
    if not torch.isfinite(d_signal).all():
        raise RuntimeError("non-finite steering direction")
    gen = torch.Generator().manual_seed(SEED)
    d_control = torch.randn(lens.d_model, generator=gen)
    d_control = d_control / d_control.norm()
    log(f"direction ||d||={float(d_signal.norm()):.3f} | scales="
        + " ".join(f"L{l}={scales[l]:.1f}" for l in layers)
        + f" | calib {time.time() - t0:.0f}s")

    vecs = build_layer_vectors(lens, d_signal, d_control, layers)
    log("pull-back orientation invariant passed for all layers/directions")

    # move injection vectors to device once
    dev_vecs = {name: {l: vecs[name][l].to(device) for l in layers}
                for name in ("signal", "control")}

    # ---- resume ------------------------------------------------------------
    done_pairs: set = set()
    prior_rows: list[dict] = []
    if results_path.exists() and not args.smoke:
        prev = pd.read_parquet(results_path)
        prior_rows = prev.to_dict("records")
        done_pairs = set(prev["pair_id"].unique())
        log(f"resuming: {len(done_pairs)} eval pairs already done")

    records: list[dict] = list(prior_rows)
    t0 = time.time()
    n_forward = 0
    processed = 0
    for pi, row in enumerate(eval_rows):
        if row["pair_id"] in done_pairs:
            continue
        inputs = build(row, neutral_cfg)
        inputs["use_cache"] = False
        seq_len = int(inputs["input_ids"].shape[1])
        if args.positions == "patches":
            positions = all_image_token_positions(inputs, image_token_id)
        else:
            positions = [seq_len - 1]
        prod_label = row["presented_label"]

        for layer in layers:
            for direction in ("signal", "control"):
                v = dev_vecs[direction][layer]
                for alpha in alphas:
                    add_vec = None if alpha == 0 else (
                        float(alpha) * scales[layer] * v)
                    logits = steered_answer_logits(
                        model, lens_model, inputs, layer, add_vec, positions)
                    n_forward += 1
                    m = score_answer_logits(
                        logits, class_ids, tok_more, tok_less, prod_label)
                    records.append({
                        "pair_id": row["pair_id"], "prod_label": prod_label,
                        "direction": direction, "layer": int(layer),
                        "alpha": float(alpha), "scale_l": scales[layer],
                        "positions": args.positions, **m})
        processed += 1
        if processed % 20 == 0:
            pd.DataFrame(records).to_parquet(results_path, index=False)
            el = time.time() - t0
            log(f"  {processed}/{len(eval_rows)} eval pairs | {n_forward} "
                f"forwards | {el / n_forward:.2f}s/fwd | ckpt -> "
                f"{results_path.name}")

    df = pd.DataFrame(records)
    df.to_parquet(results_path, index=False)
    log(f"wrote {results_path} ({len(df)} rows, {n_forward} steered forwards, "
        f"{time.time() - t0:.0f}s)")

    summary = {
        "layers": layers, "alphas": alphas, "positions": args.positions,
        "n_calib": args.n_calib, "n_eval": len(eval_rows),
        "scales": scales, "label_first_tokens": class_ids,
        "lens": args.lens, "case": args.case, "kind": args.kind,
        "registry": str(canonical.registry_dir(args.case, args.kind)),
        "per_cell": summarize(df),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"wrote {summary_path}")

    print_dose_response(df, layers, alphas)
    return 0


def run_smoke(args: argparse.Namespace) -> int:
    """2 calib + 4 eval, 1 layer, alphas {-4,0,4}. Asserts: direction finite,
    orientation invariant, and hook removal (a second clean forward reproduces
    the first bit-for-bit)."""
    torch.manual_seed(SEED)
    device = "cuda:0"
    layers = parse_int_list(args.layers)[:1]
    alphas = [-4.0, 0.0, 4.0]
    log(f"SMOKE: layer={layers} alphas={alphas} positions={args.positions}")

    conditions = load_conditions(args.case, args.kind)
    prod_cfg, neutral_cfg = conditions["prod"], conditions["neutral"]

    lens = load_lens(args.lens)
    missing = [l for l in layers if l not in lens.jacobians]
    if missing:
        raise ValueError(f"layers {missing} not fitted ({lens.source_layers})")

    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(
        args.model_dir)
    class_ids = label_first_tokens(proc.tokenizer, prod_cfg)
    tok_more, tok_less = class_ids["More"], class_ids["Less"]

    rows = load_pairs(args.case, args.kind)
    calib_rows, eval_rows = split_calibration_eval(rows, 2, 4)
    build = make_build_inputs(proc, tmpl, device)

    d_signal, scales = compute_direction_and_scales(
        model, lens_model, build, calib_rows, prod_cfg,
        layers, tok_more, tok_less, device,
        positions_mode=args.positions, image_token_id=image_token_id)
    assert torch.isfinite(d_signal).all(), "direction is non-finite"
    log(f"ASSERT direction finite OK | centered scales={scales}")

    gen = torch.Generator().manual_seed(SEED)
    d_control = torch.randn(lens.d_model, generator=gen)
    d_control = d_control / d_control.norm()
    layer = layers[0]
    resid = orientation_residual(lens, lens.jacobians[layer].float(), d_signal, layer)
    assert resid < 1e-3, f"orientation invariant failed: {resid:.2e}"
    log(f"ASSERT transport-orientation invariant OK (rel err {resid:.2e})")

    vecs = build_layer_vectors(lens, d_signal, d_control, layers)
    v = vecs["signal"][layer].to(device)

    # hook-removal check: clean forward A, one hooked forward, clean forward B.
    row = eval_rows[0]
    inputs = build(row, neutral_cfg)
    inputs["use_cache"] = False
    seq_len = int(inputs["input_ids"].shape[1])
    positions = ([seq_len - 1] if args.positions == "answer"
                 else all_image_token_positions(inputs, image_token_id))
    logits_a = steered_answer_logits(model, lens_model, inputs, layer, None, positions)
    _ = steered_answer_logits(
        model, lens_model, inputs, layer, 4.0 * scales[layer] * v, positions)
    logits_b = steered_answer_logits(model, lens_model, inputs, layer, None, positions)
    assert torch.equal(logits_a, logits_b), (
        "hook removal failed: clean forwards differ "
        f"(max |Δ|={float((logits_a - logits_b).abs().max()):.3e})")
    log("ASSERT hook removal OK (second clean forward == first, bit-for-bit)")

    # tiny end-to-end sweep so the scoring/records path is exercised.
    records = []
    for r in eval_rows:
        inp = build(r, neutral_cfg)
        inp["use_cache"] = False
        sl = int(inp["input_ids"].shape[1])
        pos = ([sl - 1] if args.positions == "answer"
               else all_image_token_positions(inp, image_token_id))
        for direction in ("signal", "control"):
            vv = vecs[direction][layer].to(device)
            for alpha in alphas:
                add = None if alpha == 0 else alpha * scales[layer] * vv
                lg = steered_answer_logits(model, lens_model, inp, layer, add, pos)
                m = score_answer_logits(lg, class_ids, tok_more, tok_less,
                                        r["presented_label"])
                records.append({"pair_id": r["pair_id"], "direction": direction,
                                "layer": layer, "alpha": float(alpha), **m})
    df = pd.DataFrame(records)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "steer_smoke.parquet", index=False)
    print_dose_response(df, layers, alphas)
    log(f"SMOKE complete: {len(df)} rows -> {out_dir / 'steer_smoke.parquet'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lens", default=LENS_DEFAULT)
    ap.add_argument("--case", default=CASE_DEFAULT,
                    choices=list(canonical.CASES),
                    help="Registered case (gemma-4-12b only).")
    ap.add_argument("--kind", default=KIND_DEFAULT,
                    choices=list(canonical.KINDS),
                    help="Registry kind; prefer proxy (no thinking).")
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-pairs", type=int, default=100, help="eval pairs")
    ap.add_argument("--n-calib", type=int, default=16, help="calibration pairs")
    ap.add_argument("--layers", default=DEFAULT_LAYERS)
    ap.add_argument("--alphas", default=DEFAULT_ALPHAS)
    ap.add_argument("--positions", choices=["answer", "patches"], default="answer")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        return run_smoke(args)
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
