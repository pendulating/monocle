"""Phase-4 rung B: the safety-workspace experiment (headline science).

When gemma-4-12B judges which of two subway-street photographs looks *safer*,
at what depth does the judgment enter the **verbalizable workspace** — the
Jacobian-lens readout — and does it ever enter under a prompt that never
mentions safety? This mechanistically completes the activation-probe result
([[concept-activation-probe]]): the safety judgment is linearly decodable from
L24 even under a blind prompt, but the model cannot *say* it. The J-lens asks
the sharper question: is it ever in the readout channel the model actually
verbalizes from?

The prompt recipe is copied verbatim from the activation probe
(`outputs/_actprobe/act_probe_smoke2.py`): three conditions (prod / axis /
neutral), a two-image chat build via `_gemma4_unified_chat_template`, and the
teacher-forced JSON prefix `{"answer": "` so the final position literally emits
the label token. Supervision + SEED-777 ordering come from the same subway
parquet, filtered the same way (repeat_idx==0, presented_label in ORDINAL,
`.head(N)`), so pairs align with the probe run.

Two parts:

  (a) DEPTH-RESOLVED ANSWER EMERGENCE.  One forward per (pair, condition) with
      jlens ActivationRecorder hooks on the fitted layers. At two read
      positions (label = -1, last = -1 - n_suffix) and for each of three lenses
      (wikitext / urban / mm), transport the residual with `unembed(J_l @ h)`
      and read: p(production label's first token), its rank, a restricted
      argmax over the 4 first-token label classes, and exact vocab-mass over
      safety- and brightness-words. Layer 47 = the model's own logits (exact).

  (b) PER-PATCH ANSWER-FEEDING MAPS (mm lens).  For a handful of pairs, split
      the two image blocks, and under the mm lens (whose Jacobian targets are
      later positions — the designated "which patches feed the answer"
      instrument) read per-patch safety-mass / brightness-mass / label-mass at
      every fitted layer, saved long and rendered as side-by-side heatmaps plus
      a prod-minus-neutral difference map.

The 4-way collapse (documented in the output metadata): the first tokens of
"MuchLess" and "MuchMore" collide on the single token "Much" (id 46003 on this
tokenizer), so the first-token classes are {Much*, Less, Same, More}. Ordinal
agreement is reported on this collapsed scale (see `collapsed_index`).

Environment: klara, `.venv-nightly` + LD_PRELOAD libstdc++ (monocle/monocle.sub).
NO GPU is needed for the CPU test-suite (tests/test_safety_workspace.py) —
every jlens / transformers / model import is deferred into the GPU code paths.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import canonical, extract, render  # noqa: E402  (CPU-safe)
from monocle import jlens_read  # noqa: E402  (jlens is imported lazily inside)
from monocle.validate import vocab_ids_exact  # noqa: E402

MODEL_DIR = "/share/pierson/matt/zoo/models/gemma-4-12B-it"
OUT_DEFAULT = REPO / "outputs/_monocle/safety_workspace"

# Pairs, labels, and prompts all come from the canonical run registry through
# `monocle.canonical` — never from a hardcoded parquet or a conf/prompt YAML.
# The pre-2026-08-11 build pointed at the 2026-06-29 subway run, whose prompt
# the consolidation superseded. See monocle/canonical.py.
DEFAULT_CASE = "subway_safety"
DEFAULT_KIND = "proxy"

LENS_WIKITEXT = str(REPO / "outputs/_monocle/jlens/gemma4_12b_lens.pt")
LENS_URBAN = str(REPO / "outputs/_monocle/jlens/urban/gemma4_12b_lens.pt")
LENS_MM = str(REPO / "outputs/_monocle/jlens/mm/gemma4_12b_lens.pt")
DEFAULT_LENSES = [LENS_WIKITEXT, LENS_URBAN, LENS_MM]

ORDINAL = list(canonical.ORDINAL)
FORCE_PREFIX = canonical.FORCE_PREFIX
SEED = 777
FITTED_LAYERS = [6, 12, 18, 24, 30, 36, 42, 46]
FINAL_LAYER = 47
MAP_LAYERS = [24, 36, 42]

# Exact-word vocab-mass probes (matched by display form via vocab_ids_exact).
SAFETY_WORDS = ["safe", "safety", "unsafe", "dangerous", "danger",
                "crime", "risky", "sketchy"]
BRIGHTNESS_WORDS = ["bright", "brightly", "dark", "darkness", "light",
                    "lit", "dim", "daylight", "shadow"]

# The 4 distinguishable first-token classes after the Much* collapse, in the
# order used for the collapsed-scale ordinal metric. Much* is placed at index 0
# (the extreme pole); both MuchLess and MuchMore map here — an information loss
# inherent to reading a single first token. The /4 normalization is retained
# from the original 5-point behavioral scale for continuity (so agreement lands
# in [0.25, 1.0]); this metric is a coarse diagnostic — exact 4-way agreement
# and p_prod_label are the primary readouts. Documented in the output metadata.
COLLAPSED_ORDER = ["Much*", "Less", "Same", "More"]

# The abstention class. The consolidated battery keeps abstention ON, so
# "NotSure" is a real first-token class the model can emit, and it must appear
# in the restricted argmax. It is NOT a point on the ordinal scale, so it stays
# out of COLLAPSED_ORDER and the ordinal metric returns NaN when either side
# abstains. Abstention rates vary a lot by case (road_quality 0.004, schools
# 0.518), so a case with a high rate needs the abstention column read first.
ABSTAIN = canonical.NOT_SURE


def log(m: str) -> None:
    print(f"[safety-ws] {m}", flush=True)


# ---------------------------------------------------------------------------
# Two-venv import fallback (jlens is installed in .venv-nightly, not .venv).
# ---------------------------------------------------------------------------
def _load_activation_recorder():
    try:
        from jlens.hooks import ActivationRecorder
    except ModuleNotFoundError:
        sys.path.insert(0, str(REPO / "sub/jacobian-lens"))
        from jlens.hooks import ActivationRecorder
    return ActivationRecorder


# ---------------------------------------------------------------------------
# Prompt conditions (verbatim from the activation probe recipe).
# ---------------------------------------------------------------------------
def build_conditions(
    case: str = DEFAULT_CASE, kind: str = DEFAULT_KIND,
    conds: Optional[list[str]] = None,
) -> dict[str, Any]:
    """{cond: prompt cfg} for the arms of part (a).

    Delegates to `canonical.build_conditions`, which varies ONLY the question
    text. The system turn stays absent, the layout stays interleaved_labels,
    and the abstention guidance still appends in every arm.
    """
    return canonical.build_conditions(case, kind, conds)


def build_pair_inputs(
    proc: Any, tmpl: str, images: list[Image.Image], row: dict, cfg: Any,
    device: str = "cuda:0",
) -> dict:
    """Two-image chat inputs on the canonical prompt path.

    Delegates to `canonical.build_pair_inputs`, so the token sequence matches
    the registered run. `tests/test_canonical_prompt.py` holds that identity.

    Warning: `images` must be in PRESENTED order (presented_left_path then
    presented_right_path), not image_path_a/b — the battery swaps half the
    rows.
    """
    return canonical.build_pair_inputs(
        proc, tmpl, images, row, cfg, force_prefix=FORCE_PREFIX, device=device)


def read_positions(seq_len: int, n_suffix: int) -> dict[str, int]:
    """Ordered {name: seq index}. label = -1 (emits the label token);
    last = -1 - n_suffix (last real prompt token before the JSON prefix)."""
    return {"label": seq_len - 1, "last": seq_len - 1 - n_suffix}


# ---------------------------------------------------------------------------
# Two-image patch-block geometry (replaces extract.image_token_positions, which
# asserts ONE contiguous block).
# ---------------------------------------------------------------------------
def contiguous_runs(positions: torch.Tensor) -> list[torch.Tensor]:
    """Split a 1-D LongTensor of positions into maximal contiguous runs.

    Pure splitter with no cardinality assertion — an interleaved singleton such
    as [0, 2, 3, 4] yields [[0], [2, 3, 4]]. `image_token_blocks` wraps this
    with the two-equal-blocks contract."""
    positions = torch.as_tensor(positions).long().flatten()
    if positions.numel() == 0:
        return []
    runs: list[torch.Tensor] = []
    start = 0
    for i in range(1, positions.numel()):
        if int(positions[i] - positions[i - 1]) != 1:
            runs.append(positions[start:i])
            start = i
    runs.append(positions[start:])
    return runs


def image_token_blocks(
    inputs: dict, image_token_id: int,
) -> list[torch.Tensor]:
    """Positions of the two image blocks, as [block_A, block_B].

    Contract: raises RuntimeError unless there are exactly two contiguous runs
    of image tokens of EQUAL length (the two-image replacement for
    extract.image_token_positions). One block, three-plus blocks, or ragged
    blocks all raise — a wrong split makes every downstream patch map garbage.
    """
    ids = inputs["input_ids"][0]
    pos = (ids == image_token_id).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        raise RuntimeError("zero image tokens — images never entered the seq")
    runs = contiguous_runs(pos)
    if len(runs) != 2:
        raise RuntimeError(
            f"expected exactly 2 image blocks, found {len(runs)} contiguous "
            f"runs (lengths {[int(r.numel()) for r in runs]})")
    if runs[0].numel() != runs[1].numel():
        raise RuntimeError(
            f"image blocks have unequal length: {int(runs[0].numel())} vs "
            f"{int(runs[1].numel())}")
    return runs


def square_grid(n_patches: int, inputs: Optional[dict] = None,
                orig_size: Optional[tuple[int, int]] = None) -> tuple[int, int]:
    """(rows, cols) for one image's patch grid. Prefers the perfect-square path
    (cyclomedia faces are 1024^2 -> 256 -> 16x16); falls back to
    extract.infer_grid only when the block is not square (needs inputs + size).
    """
    import math
    root = math.isqrt(n_patches)
    if root * root == n_patches:
        return root, root
    if inputs is None or orig_size is None:
        raise ValueError(
            f"{n_patches} patches is not a perfect square and no "
            f"inputs/orig_size given for aspect inference")
    grid = extract.infer_grid(inputs, n_patches, orig_size)
    return grid.n_rows, grid.n_cols


# ---------------------------------------------------------------------------
# Label first-token machinery + the Much* collapse.
# ---------------------------------------------------------------------------
def label_first_tokens(
    tokenizer: Any, cfg: Any,
) -> tuple[dict[str, int], dict[str, int], bool]:
    """First-token ids for the labels the registered run allowed.

    Delegates to `canonical.label_classes`, so the abstention label reaches the
    class set whenever the run enabled it. The pre-2026-08-11 build hardcoded
    the 5 ordinal labels and therefore could never read an abstention.

    Returns (label_first, class_ids, collision). `collision` records that
    MuchLess and MuchMore share the token "Much" — an information loss inherent
    to reading one first token.
    """
    label_first, class_ids, _ = canonical.label_classes(tokenizer, cfg)
    collision = label_first["MuchLess"] == label_first["MuchMore"]
    return label_first, class_ids, collision


def collapse_label(label: str) -> str:
    """Map an answer label onto its first-token class.

    MuchLess and MuchMore collapse to Much*. The abstention label passes
    through unchanged — it is its own class.
    """
    if label in ("MuchLess", "MuchMore"):
        return "Much*"
    return label


def collapsed_index(cls: str) -> int:
    return COLLAPSED_ORDER.index(cls)


def collapsed_ordinal_agreement(pred_cls: str, prod_label: str) -> float:
    """1 - |idx(pred) - idx(collapse(prod))| / 4 on the 4-way collapsed scale.

    Returns NaN when either side is the abstention class. An abstention is not
    a 0/"Same" judgment and must not be folded into the ordinal scale — the
    same rule `pairwise_vqa._score_labels` applies to the behavioural labels.
    """
    prod_cls = collapse_label(prod_label)
    if pred_cls == ABSTAIN or prod_cls == ABSTAIN:
        return float("nan")
    d = abs(collapsed_index(pred_cls) - collapsed_index(prod_cls))
    return 1.0 - d / 4.0


def vocab_mass_ids(tokenizer: Any, words: list[str]) -> list[int]:
    """Union of exact-display-form vocab ids over `words` (validate.vocab_ids_exact
    per word; substring matching is a known trap)."""
    ids: set[int] = set()
    for w in words:
        ids.update(vocab_ids_exact(tokenizer, w))
    return sorted(ids)


# ---------------------------------------------------------------------------
# Per-position metric primitives (pure, CPU-testable).
# ---------------------------------------------------------------------------
def restricted_argmax(logits_row: torch.Tensor, class_ids: dict[str, int]) -> str:
    """Argmax over only the 4 first-token class ids -> class name."""
    names = list(class_ids)
    idx = torch.tensor([class_ids[n] for n in names], device=logits_row.device)
    return names[int(torch.argmax(logits_row[idx]))]


def token_rank(logits_row: torch.Tensor, tid: int) -> int:
    """Rank of token `tid` in the full vocab (0 = argmax)."""
    return int((logits_row > logits_row[tid]).sum())


def mass(probs: torch.Tensor, ids: torch.Tensor) -> float:
    """Total probability mass on `ids` (a LongTensor on the same device)."""
    if ids.numel() == 0:
        return 0.0
    return float(probs[ids].sum())


def position_metrics(
    logits_row: torch.Tensor,
    prod_label: str,
    class_ids: dict[str, int],
    label_first: dict[str, int],
    safety_ids: Optional[torch.Tensor] = None,
    brightness_ids: Optional[torch.Tensor] = None,
) -> dict:
    """Label-class metrics for a single [vocab] logits row.

    The two vocab-mass probes are optional. Without them the returned dict
    omits `safety_mass` / `brightness_mass` — the open-vocabulary readout in
    `monocle.answer_tokens` wants the label metrics but tallies the tokens
    itself rather than pre-committing to a word list.
    """
    probs = torch.softmax(logits_row, dim=-1)
    prod_first = label_first[prod_label]
    argmax_cls = restricted_argmax(logits_row, class_ids)
    prod_cls = collapse_label(prod_label)
    out = {
        "p_prod_label": float(probs[prod_first]),
        "rank_prod_label": token_rank(logits_row, prod_first),
        "argmax_class": argmax_cls,
        "argmax_correct": bool(argmax_cls == prod_cls),
    }
    if safety_ids is not None:
        out["safety_mass"] = mass(probs, safety_ids)
    if brightness_ids is not None:
        out["brightness_mass"] = mass(probs, brightness_ids)
    return out


# ---------------------------------------------------------------------------
# Recording (adds a positions/logits variant ALONGSIDE
# jlens_read.record_patch_activations, which stays frozen and single-image).
# ---------------------------------------------------------------------------
def record_activations(
    model: Any, lens_model: Any, inputs: dict, layers: list[int],
    logit_positions: Optional[torch.Tensor] = None,
) -> tuple[dict[int, torch.Tensor], Optional[torch.Tensor]]:
    """One multimodal forward with jlens hooks on `layers` (block modules).

    Returns (activations {layer: [B, seq, d] full-sequence block output},
    logits) where `logits` is `out.logits[0, logit_positions, :]` fp32 when
    `logit_positions` is given, else None. Full-sequence activations let the
    caller slice ANY positions (the two read positions for part (a), the two
    image blocks for part (b)) from one forward.
    """
    ActivationRecorder = _load_activation_recorder()
    with ActivationRecorder(lens_model.layers, at=sorted(set(layers))) as rec:
        with torch.inference_mode():
            out = model(**inputs, use_cache=False)
        activations = {i: rec.activations[i].detach() for i in rec.activations}
    logits = None
    if logit_positions is not None:
        logits = out.logits[0, logit_positions, :].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite logits at requested positions")
    return activations, logits


# ---------------------------------------------------------------------------
# Checkpoint / resume for the long frame.
# ---------------------------------------------------------------------------
def completed_pair_ids(path: str | Path) -> set:
    """Set of pair_ids already fully written to the long parquet (empty if the
    file is absent). A pair is written only after all its rows are computed, so
    presence == done."""
    p = Path(path)
    if not p.exists():
        return set()
    return set(pd.read_parquet(p, columns=["pair_id"])["pair_id"].unique())


def write_long_frame(df: pd.DataFrame, path: str | Path) -> None:
    """Atomic parquet write (tmp + os.replace) so a wall-time kill mid-write
    never corrupts the checkpoint."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f"{p.stem}.tmp{os.getpid()}.parquet"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Heatmap rendering (PIL only).
# ---------------------------------------------------------------------------
def _seq_color(t: float) -> tuple[int, int, int]:
    """Sequential black-red-yellow ramp for t in [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        u = t / 0.5
        return int(255 * u), 0, 0
    u = (t - 0.5) / 0.5
    return 255, int(255 * u), 0


def _div_color(t: float) -> tuple[int, int, int]:
    """Diverging red(+)/blue(-) for t in [-1, 1]."""
    if t >= 0:
        return 235, 45, 45
    return 45, 95, 235


def render_heatmap(
    image: Image.Image, values: np.ndarray, n_rows: int, n_cols: int,
    *, scale: float, diverging: bool = False, dim: float = 0.45,
    title: Optional[str] = None,
) -> Image.Image:
    """Per-patch heatmap over a dimmed copy of `image` (row-major patches).

    Sequential mode colors each patch on a black-red-yellow ramp with alpha
    tracking value/scale; diverging mode colors sign (red +, blue -) with alpha
    tracking |value|/scale. `values` is length n_rows*n_cols, row-major.
    """
    base = image.convert("RGB")
    w, h = base.size
    cw, ch = w / n_cols, h / n_rows
    arr = (np.asarray(base, dtype=np.float32) * dim).clip(0, 255).astype(np.uint8)
    canvas = Image.fromarray(arr, "RGB").convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale = scale if scale > 1e-12 else 1.0
    for idx in range(min(len(values), n_rows * n_cols)):
        r, c = idx // n_cols, idx % n_cols
        v = float(values[idx])
        if diverging:
            t = float(np.clip(v / scale, -1.0, 1.0))
            color = _div_color(t)
            alpha = int(210 * abs(t))
        else:
            t = float(np.clip(v / scale, 0.0, 1.0))
            color = _seq_color(t)
            alpha = int(30 + 200 * t)
        x0, y0 = round(c * cw), round(r * ch)
        x1, y1 = round((c + 1) * cw), round((r + 1) * ch)
        draw.rectangle([x0, y0, x1, y1], fill=(*color, alpha))
    out = Image.alpha_composite(canvas, overlay).convert("RGB")
    if title:
        d = ImageDraw.Draw(out)
        font = render.get_font(max(12, w // 40))
        d.text((6, 4), title, fill=(255, 255, 255),
               font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def side_by_side(imgs: list[Image.Image], gap: int = 10) -> Image.Image:
    """Paste images left-to-right on a black strip separated by `gap` px."""
    imgs = [im.convert("RGB") for im in imgs]
    h = max(im.height for im in imgs)
    w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


# ---------------------------------------------------------------------------
# Sample selection.
# ---------------------------------------------------------------------------
def load_pairs(
    n_pairs: int, case: str = DEFAULT_CASE, kind: str = DEFAULT_KIND,
    seed: int = SEED, keep_abstentions: bool = True,
) -> list[dict]:
    """Supervision rows of a registered run, read through the registry.

    Keeps one presentation per canonical pair (`repeat_idx == 0`), then draws a
    seeded subsample. Abstained rows are KEPT by default: on a case such as
    schools they are more than half the run, and dropping them would silently
    restrict the study to the pairs the model was willing to answer.

    The returned rows carry `presented_left_path` / `presented_right_path`, so
    a caller loads the images in presented order.
    """
    df = canonical.load_results(case, kind, columns=[
        "pair_id", "repeat_idx", "presented_label",
        "presented_left_path", "presented_right_path"])
    df = df[df["repeat_idx"] == 0]
    if not keep_abstentions:
        df = df[df["presented_label"].isin(ORDINAL)]
    n_kept = len(df)
    if n_pairs < n_kept:
        df = df.sample(n=n_pairs, random_state=seed)
    n_abstain = int((df["presented_label"] == ABSTAIN).sum())
    log(f"{case}/{kind}: {len(df)}/{n_kept} pairs "
        f"({n_abstain} abstained = {n_abstain / max(len(df), 1):.1%})")
    return df.reset_index(drop=True).to_dict("records")


def lens_name(path: str) -> str:
    if "/mm/" in path:
        return "mm"
    if "/urban/" in path:
        return "urban"
    return "wikitext"


# ---------------------------------------------------------------------------
# Part (a): depth-resolved answer emergence.
# ---------------------------------------------------------------------------
def run_part_a(
    proc, model, tmpl, lens_model, lenses: dict[str, Any],
    conditions: dict[str, tuple[str, str]], rows: list[dict],
    class_ids: dict[str, int], label_first: dict[str, int],
    safety_ids: torch.Tensor, brightness_ids: torch.Tensor,
    layers: list[int], out_dir: Path, device: str,
    map_cache: dict, n_map_pairs: int, smoke: bool,
) -> pd.DataFrame:
    """One forward per (pair, cond); transport at both read positions under
    every lens; long frame written/rewritten every 50 pairs (resume-safe)."""
    n_suffix = len(proc.tokenizer.encode(FORCE_PREFIX, add_special_tokens=False))
    record_at = [l for l in layers if l != FINAL_LAYER]
    all_layers = sorted(set(layers) | {FINAL_LAYER})
    depth_pq = out_dir / "answer_depth.parquet"

    done = completed_pair_ids(depth_pq)
    prior = (pd.read_parquet(depth_pq) if depth_pq.exists() else
             pd.DataFrame())
    if done:
        log(f"resume: {len(done)} pairs already in {depth_pq.name}")

    new_rows: list[dict] = []
    processed_since_ckpt = 0
    t0 = time.time()
    for pi, row in enumerate(rows):
        pid = row["pair_id"]
        if pid in done:
            continue
        prod_label = row["presented_label"]
        left, right = canonical.presented_images(row)
        imgs = [Image.open(left).convert("RGB"),
                Image.open(right).convert("RGB")]
        for cond, cond_cfg in conditions.items():
            inputs = build_pair_inputs(proc, tmpl, imgs, row, cond_cfg, device)
            seq_len = int(inputs["input_ids"].shape[1])
            rp = read_positions(seq_len, n_suffix)
            positions = torch.tensor([rp["label"], rp["last"]], device=device)
            activations, final_logits = record_activations(
                model, lens_model, inputs, record_at, logit_positions=positions)

            if smoke and pi == 0 and cond == "prod":
                _smoke_asserts(inputs, activations, record_at, positions,
                               proc.tokenizer, n_suffix,
                               int(model.config.image_token_id))

            # Cache the first n_map_pairs pairs' prod/neutral recordings for
            # part (b), avoiding a second forward on the overlap.
            if pi < n_map_pairs and cond in ("prod", "neutral"):
                map_cache[(pi, cond)] = (activations, inputs)

            for lpath, lens in lenses.items():
                per_layer = jlens_read.lens_patch_logits(
                    lens, lens_model, activations, positions, final_logits,
                    layers=all_layers)
                for layer in all_layers:
                    logits2 = per_layer[layer]  # [2, vocab]: row0 label, row1 last
                    for j, posname in enumerate(("label", "last")):
                        m = position_metrics(
                            logits2[j], prod_label, class_ids, label_first,
                            safety_ids, brightness_ids)
                        new_rows.append({
                            "pair_id": pid, "cond": cond,
                            "lens": lens_name(lpath), "layer": layer,
                            "pos": posname, "prod_label": prod_label,
                            "prod_class": collapse_label(prod_label), **m})
            del activations, final_logits
        processed_since_ckpt += 1
        if (pi + 1) % (1 if smoke else 25) == 0:
            el = time.time() - t0
            n_new = pi + 1 - len(done)
            log(f"  {pi + 1}/{len(rows)} pairs | "
                f"{el / max(1, n_new):.2f}s/pair")
        if processed_since_ckpt >= 50:
            _flush(prior, new_rows, depth_pq)
            prior = pd.read_parquet(depth_pq)
            new_rows = []
            processed_since_ckpt = 0
    _flush(prior, new_rows, depth_pq)
    df = pd.read_parquet(depth_pq)
    log(f"part (a) done: {len(df)} rows over {df['pair_id'].nunique()} pairs "
        f"in {(time.time() - t0) / 60:.1f} min")
    return df


def _flush(prior: pd.DataFrame, new_rows: list[dict], path: Path) -> None:
    if not new_rows and prior.empty:
        return
    frames = [f for f in (prior, pd.DataFrame(new_rows)) if not f.empty]
    write_long_frame(pd.concat(frames, ignore_index=True), path)


def summarize_part_a(df: pd.DataFrame, meta: dict, out_dir: Path) -> dict:
    """Aggregate per (cond, lens, layer, pos) and write the summary JSON +
    a plain-text table to the log."""
    summary: dict = {"meta": meta, "by_group": []}
    grp = df.groupby(["cond", "lens", "layer", "pos"], sort=True)
    for (cond, lens, layer, pos), g in grp:
        # NaN marks a row where either side abstained. nanmean scores the
        # ordinal metric on the judged rows only, and n_ordinal says how many
        # those were — a plain mean would return NaN for the whole group.
        scored = np.array([
            collapsed_ordinal_agreement(pc, pl)
            for pc, pl in zip(g["argmax_class"], g["prod_label"])], dtype=float)
        n_ordinal = int(np.count_nonzero(~np.isnan(scored)))
        ord_agree = float(np.nanmean(scored)) if n_ordinal else float("nan")
        summary["by_group"].append({
            "cond": cond, "lens": lens, "layer": int(layer), "pos": pos,
            "n": int(len(g)),
            "mean_p_prod_label": float(g["p_prod_label"].mean()),
            "mean_rank_prod_label": float(g["rank_prod_label"].mean()),
            "argmax_agreement": float(g["argmax_correct"].mean()),
            "ordinal_agreement_collapsed": ord_agree,
            "n_ordinal": n_ordinal,
            # How often the lens READ is an abstention, and how often the run's
            # own answer was. These diverge, and on a high-abstention case the
            # agreement number cannot be read without them.
            "read_abstain_rate": float((g["argmax_class"] == ABSTAIN).mean()),
            "label_abstain_rate": float((g["prod_label"] == ABSTAIN).mean()),
            "mean_safety_mass": float(g["safety_mass"].mean()),
            "mean_brightness_mass": float(g["brightness_mass"].mean()),
        })
    out = out_dir / "answer_depth_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out}")
    _print_table(df)
    return summary


def _print_table(df: pd.DataFrame) -> None:
    """Plain-text depth table (read position) for the orchestrator's .out."""
    lab = df[df["pos"] == "label"]
    layers = sorted(lab["layer"].unique())
    log("=" * 78)
    log("PART (a) @label position — p(prod first-token) | 4way-agree | "
        "safetyMass | brightMass")
    for cond in [c for c in ("prod", "axis", "neutral")
                 if c in set(lab["cond"])]:
        for lens in sorted(lab["lens"].unique()):
            log(f"  {cond} / {lens}")
            log("    layer :  " + "  ".join(f"L{l:<3}" for l in layers))
            g = lab[(lab["cond"] == cond) & (lab["lens"] == lens)]
            def _cells(col, fmt):
                return "  ".join(
                    fmt.format(float(g[g["layer"] == l][col].mean()))
                    for l in layers)
            log("    p_prod:  " + _cells("p_prod_label", "{:.3f}"))
            log("    agree :  " + _cells("argmax_correct", "{:.3f}"))
            log("    safety:  " + _cells("safety_mass", "{:.3f}"))
            log("    bright:  " + _cells("brightness_mass", "{:.3f}"))
    log("=" * 78)


# ---------------------------------------------------------------------------
# Part (b): per-patch answer-feeding maps (mm lens).
# ---------------------------------------------------------------------------
def run_part_b(
    proc, model, tmpl, lens_model, mm_lens, rows: list[dict],
    class_ids: dict[str, int], safety_ids: torch.Tensor,
    brightness_ids: torch.Tensor, layers: list[int], map_layers: list[int],
    out_dir: Path, device: str, map_cache: dict, n_map_pairs: int, smoke: bool,
    case: str = DEFAULT_CASE, kind: str = DEFAULT_KIND,
) -> None:
    """Per-patch safety/brightness/label-mass maps under the mm lens, saved long
    + rendered as side-by-side heatmaps and a prod-minus-neutral difference."""
    record_at = [l for l in layers if l != FINAL_LAYER]
    label_ids = torch.tensor(sorted(set(class_ids.values())), device=device)
    image_token_id = int(model.config.image_token_id)
    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    conds = ["prod", "neutral"]
    conditions = build_conditions(case, kind, conds)

    for pi in range(min(n_map_pairs, len(rows))):
        row = rows[pi]
        pid = row["pair_id"]
        left, right = canonical.presented_images(row)
        imgs = [Image.open(left).convert("RGB"),
                Image.open(right).convert("RGB")]
        # per (cond, layer, slot) -> np.ndarray of safety_mass, for the diff map
        safety_by: dict[tuple[str, int, str], np.ndarray] = {}
        grids: dict[str, tuple[int, int]] = {}
        for cond in conds:
            if (pi, cond) in map_cache:
                activations, inputs = map_cache[(pi, cond)]
            else:
                inputs = build_pair_inputs(
                    proc, tmpl, imgs, row, conditions[cond], device)
                activations, _ = record_activations(
                    model, lens_model, inputs, record_at)
            blocks = image_token_blocks(inputs, image_token_id)
            if smoke:
                assert all(int(b.numel()) == 256 for b in blocks), \
                    f"expected two 256-patch blocks, got {[int(b.numel()) for b in blocks]}"
            positions = torch.cat(blocks).to(device)
            nA = int(blocks[0].numel())
            grids["A"] = square_grid(nA, inputs, imgs[0].size)
            grids["B"] = square_grid(int(blocks[1].numel()), inputs, imgs[1].size)

            # mm-lens transport at all patch positions, fitted layers only.
            per_layer = jlens_read.lens_patch_logits(
                mm_lens, lens_model, activations, positions,
                torch.empty(0, device=device), layers=record_at)
            rec_rows: list[dict] = []
            for layer in record_at:
                probs = torch.softmax(per_layer[layer], dim=-1)  # [nP, vocab]
                sm = probs[:, safety_ids].sum(dim=-1)
                bm = probs[:, brightness_ids].sum(dim=-1)
                lm = probs[:, label_ids].sum(dim=-1)
                for slot, sl in (("A", slice(0, nA)), ("B", slice(nA, None))):
                    sarr = sm[sl].float().cpu().numpy()
                    if layer in map_layers:
                        safety_by[(cond, layer, slot)] = sarr
                    for k in range(sarr.shape[0]):
                        rec_rows.append({
                            "pair_id": pid, "cond": cond, "image_slot": slot,
                            "patch_idx": k, "layer": layer,
                            "safety_mass": float(sarr[k]),
                            "brightness_mass": float(bm[sl][k]),
                            "p_label_tokens": float(lm[sl][k])})
            pd.DataFrame(rec_rows).to_parquet(
                maps_dir / f"{pid}_{cond}.parquet", index=False)

            # per-cond safety overlays
            for layer in map_layers:
                a = safety_by.get((cond, layer, "A"))
                b = safety_by.get((cond, layer, "B"))
                if a is None or b is None:
                    continue
                vmax = max(float(a.max()), float(b.max()), 1e-9)
                panel = side_by_side([
                    render_heatmap(imgs[0], a, *grids["A"], scale=vmax,
                                   title=f"A {cond} L{layer}"),
                    render_heatmap(imgs[1], b, *grids["B"], scale=vmax,
                                   title=f"B {cond} L{layer}")])
                panel.save(maps_dir / f"{pid}_{cond}_L{layer}_safety.png")

        # prod-minus-neutral difference maps
        for layer in map_layers:
            key = lambda c, s: safety_by.get((c, layer, s))
            if any(key(c, s) is None for c in conds for s in ("A", "B")):
                continue
            dA = key("prod", "A") - key("neutral", "A")
            dB = key("prod", "B") - key("neutral", "B")
            scale = max(float(np.abs(dA).max()), float(np.abs(dB).max()), 1e-9)
            panel = side_by_side([
                render_heatmap(imgs[0], dA, *grids["A"], scale=scale,
                               diverging=True, title=f"A prod-neutral L{layer}"),
                render_heatmap(imgs[1], dB, *grids["B"], scale=scale,
                               diverging=True, title=f"B prod-neutral L{layer}")])
            panel.save(maps_dir / f"{pid}_L{layer}_safety_diff.png")
        log(f"  part (b) pair {pi + 1}/{min(n_map_pairs, len(rows))} ({pid}) done")


# ---------------------------------------------------------------------------
# Smoke asserts (orchestrator-facing; GPU).
# ---------------------------------------------------------------------------
def _smoke_asserts(inputs, activations, record_at, positions, tokenizer,
                   n_suffix, image_token_id) -> None:
    ids = inputs["input_ids"][0]
    seq_len = int(inputs["input_ids"].shape[1])
    # two image blocks of 256
    blocks = image_token_blocks(inputs, image_token_id)
    assert [int(b.numel()) for b in blocks] == [256, 256], \
        f"expected two 256-patch blocks, got {[int(b.numel()) for b in blocks]}"
    # read positions sane
    assert int(positions[0]) == seq_len - 1, "label position must be -1"
    assert int(positions[1]) == seq_len - 1 - n_suffix, "last pos = -1-n_suffix"
    assert 0 <= int(positions[1]) < seq_len, "read positions in range"
    # position -1's input id decodes to the forced prefix's last char '"'
    dec = tokenizer.decode([int(ids[-1])])
    assert '"' in dec, f"position -1 should decode to the prefix quote, got {dec!r}"
    for layer in record_at:
        assert layer in activations, f"layer {layer} not recorded"
        assert torch.isfinite(activations[layer]).all(), \
            f"non-finite activations at layer {layer}"
    log(f"  [smoke] blocks={[int(b.numel()) for b in blocks]} seq_len={seq_len} "
        f"label@{int(positions[0])} last@{int(positions[1])} last_tok={dec!r} "
        f"activations finite OK")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=DEFAULT_CASE, choices=list(canonical.CASES),
                    help="Registered case to lens (gemma-4-12b only).")
    ap.add_argument("--kind", default=DEFAULT_KIND, choices=list(canonical.KINDS),
                    help=("Registry kind. Prefer proxy: the trace runs enable "
                          "thinking and sample, so the answer position sits "
                          "after a sampled reasoning block."))
    ap.add_argument("--n-pairs", type=int, default=300)
    ap.add_argument("--n-map-pairs", type=int, default=8)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help=("Part-(a) conditions (default prod+neutral, plus "
                          "axis when the case has a recovered phrase); "
                          "part (b) always uses prod+neutral."))
    ap.add_argument("--lenses", nargs="+", default=DEFAULT_LENSES,
                    help="Lens .pt paths (default wikitext, urban, mm).")
    ap.add_argument("--layers", nargs="+", type=int, default=FITTED_LAYERS,
                    help="Fitted source layers to transport (final 47 always added).")
    ap.add_argument("--map-layers", nargs="+", type=int, default=MAP_LAYERS)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true",
                    help="2 pairs, 1 map pair, tiny prints + asserts.")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    n_pairs = 2 if args.smoke else args.n_pairs
    n_map_pairs = 1 if args.smoke else args.n_map_pairs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    proc, model, tmpl = extract.load_model(args.model_dir, device=device)
    tokenizer = proc.tokenizer
    lens_model = jlens_read.wrap_for_unembed(model, tokenizer)

    lenses = {p: jlens_read.load_lens(p) for p in args.lenses}
    fitted_sets = {tuple(l.source_layers) for l in lenses.values()}
    assert len(fitted_sets) == 1, \
        f"lenses disagree on fitted layers: {fitted_sets}"
    shared_fitted = sorted(next(iter(fitted_sets)))
    log(f"loaded {len(lenses)} lenses; shared fitted layers {shared_fitted}")
    layers = sorted(set(args.layers) & set(shared_fitted))

    conditions = build_conditions(args.case, args.kind, args.conditions)
    label_first, class_ids, collision = label_first_tokens(
        tokenizer, conditions["prod"])
    log(f"label first-tokens: {label_first} | Much* collision={collision}")
    safety_ids = torch.tensor(vocab_mass_ids(tokenizer, SAFETY_WORDS),
                              device=device)
    brightness_ids = torch.tensor(vocab_mass_ids(tokenizer, BRIGHTNESS_WORDS),
                                  device=device)

    rows = load_pairs(n_pairs, args.case, args.kind)
    log(f"{len(rows)} pairs | conditions {list(conditions)} | "
        f"layers {layers} + final {FINAL_LAYER}")
    if args.kind == "trace":
        log("WARNING kind=trace enables thinking — the answer position sits "
            "after a sampled reasoning block, which confounds attribution")

    meta = {
        "case": args.case, "kind": args.kind,
        "registry": str(canonical.registry_dir(args.case, args.kind)),
        "question": canonical.user_text("<pair_id>", conditions["prod"]),
        "n_pairs": len(rows), "conditions": list(conditions),
        "lenses": {lens_name(p): p for p in args.lenses},
        "layers": layers, "final_layer": FINAL_LAYER,
        "read_positions": ["label (-1)", f"last (-1 - {len(proc.tokenizer.encode(FORCE_PREFIX, add_special_tokens=False))})"],
        "label_first_tokens": label_first,
        "label_first_display": {
            l: tokenizer.convert_ids_to_tokens(t)
            for l, t in label_first.items()},
        "much_collision": collision,
        "collapse": {
            "order": COLLAPSED_ORDER,
            "note": ("MuchLess and MuchMore share first token 'Much' -> class "
                     "Much* at index 0; ordinal agreement = 1-|Δ|/4 on this "
                     "4-way scale (see COLLAPSED_ORDER). Abstention is a 5th "
                     "class in the restricted argmax but sits OFF the ordinal "
                     "scale, so it scores NaN and n_ordinal counts the judged "
                     "rows. Diagnostic only; the primary readouts are "
                     "p_prod_label and the restricted argmax."),
        },
        "safety_words": SAFETY_WORDS, "brightness_words": BRIGHTNESS_WORDS,
        "n_safety_ids": int(safety_ids.numel()),
        "n_brightness_ids": int(brightness_ids.numel()),
        "recovered_axis": canonical.RECOVERED_AXIS.get(args.case),
        "abstention_class": ABSTAIN,
        "force_prefix": FORCE_PREFIX,
        "seed": SEED,
    }

    map_cache: dict = {}
    df = run_part_a(
        proc, model, tmpl, lens_model, lenses, conditions, rows, class_ids,
        label_first, safety_ids, brightness_ids, layers, out_dir, device,
        map_cache, n_map_pairs, args.smoke)
    summarize_part_a(df, meta, out_dir)

    # Part (b): mm lens (from --lenses if present, else the default mm path).
    mm_path = next((p for p in args.lenses if "/mm/" in p), LENS_MM)
    mm_lens = lenses.get(mm_path) or jlens_read.load_lens(mm_path)
    log(f"part (b): mm lens = {mm_path}")
    run_part_b(
        proc, model, tmpl, lens_model, mm_lens, rows, class_ids, safety_ids,
        brightness_ids, layers, args.map_layers, out_dir, device, map_cache,
        n_map_pairs, args.smoke, args.case, args.kind)

    log("safety-workspace experiment complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
