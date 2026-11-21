"""Causal patch maps: measure what each patch is worth, then draw it.

This is the RECOMMENDED figure path. It does not use a gradient.

Why not the gradient map
------------------------
`patch_map_causal` compared the 1-backward-pass map against the truth, on the
layers where patches still cause the judgment:

| Layer | sign accuracy, top 50 patches | r, top 50 |
|---|---|---|
| L12 | 0.56 | +0.13 |
| **L20** | **0.65** | **+0.43** |
| L24 | 0.57 | +0.15 |
| L28 | 0.46 | +0.12 |

Only L20 carries usable signal, and 0.65 is not enough to colour 256 cells.
Above L30 no estimator can work, because the patches cause nothing there.
So a figure that claims to show what the picture contributed must MEASURE it.

The method
----------
At layer L, replace one patch of the target with the same patch of a donor
image, run the rest forward, and record how far `c` moves. Do that for all
256 patches. The value in a cell is a real change in the judgment, in logit
units, not a score.

Donor averaging
---------------
A single donor answers "what if this patch were THAT patch". Averaging over
several donors answers the question a reader assumes a map answers: "how much
does this patch matter". The per-cell spread across donors is saved too, so
an unstable cell can be told from a solid one.

Reading the picture
-------------------
The DATA and the COLOUR run in opposite directions, on purpose.

| Layer | Direction |
|---|---|
| `values` in the parquet | contribution toward the POSITIVE label (`Safe`) |
| colour in the panel | coral is the NEGATIVE label (`Dangerous`) |

The data sign is the contribution, so it is the negative of the measured swap
delta: a patch that holds up `Safe` makes `c` fall when it is replaced.

The colour then turns that around again, because a reader of a safety map
expects red to mark the danger. `CORAL_SIGN` holds this one flip, and the
colour bar names both poles, so the picture cannot be read backwards.

Every panel shares one colour scale, so panels are comparable and the colour
bar is in logit units.
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

from monocle import jlens_read  # noqa: E402
from monocle import modality_order as mo  # noqa: E402
from monocle import patch_ablate as pab  # noqa: E402
from monocle import patch_attribution as pa  # noqa: E402
from monocle import patch_map_causal as pmc  # noqa: E402
from monocle import patch_swap as psw  # noqa: E402
from monocle import single as sg  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/causal_maps"
#: Patches still hold the judgment here (whole-image swap recovers about 1.0)
#: and it is the only depth where a gradient map carried signal.
DEFAULT_LAYER = 20
#: The picture paints the NEGATIVE label coral. Red for danger is what a
#: reader of a safety map expects, and a figure must not fight that. Only the
#: colour turns; the data column keeps the contribution sign.
CORAL_SIGN = -1.0


def log(m: str) -> None:
    print(f"[causal-maps] {m}", flush=True)


def grid_of(n_patches: int) -> tuple[int, int]:
    side = int(round(n_patches ** 0.5))
    if side * side != n_patches:
        raise ValueError(
            f"{n_patches} patches is not square; this renderer assumes a "
            "square grid, which every Cyclomedia face gives")
    return side, side


def overlay(
    image: Image.Image, values: np.ndarray, vmax: float, *,
    alpha: float = 0.55, upscale: int = 2, grid_lines: bool = True,
    coral_sign: float = CORAL_SIGN,
) -> Image.Image:
    """Draw the map on the photo, on a SHARED scale so panels compare.

    `pa.render_attrib` normalizes inside each panel, which is right for a
    single map and wrong for a sheet: it makes a weak map look as strong as a
    strong one. Here `vmax` comes from the whole set.

    `values` holds the contribution toward the positive label. `coral_sign`
    turns the colour over, so coral marks the negative label. Give it `+1.0`
    to paint the data sign directly.
    """
    from PIL import Image as PILImage, ImageDraw

    n_rows, n_cols = grid_of(values.size)
    norm = np.clip(coral_sign * values / vmax, -1.0, 1.0).reshape(n_rows, n_cols)
    W, H = image.size
    W, H = W * upscale, H * upscale
    base = image.convert("RGB").resize((W, H), PILImage.LANCZOS)
    cells = np.zeros((n_rows, n_cols, 3), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            cells[r, c] = pa._div_color(float(norm[r, c]))
    heat = PILImage.fromarray(cells).resize((W, H), PILImage.NEAREST)
    out = PILImage.blend(base, heat, alpha)
    if grid_lines:
        draw = ImageDraw.Draw(out)
        cw, ch = W / n_cols, H / n_rows
        for r in range(1, n_rows):
            draw.line([(0, r * ch), (W, r * ch)], fill=(255, 255, 255), width=1)
        for c in range(1, n_cols):
            draw.line([(c * cw, 0), (c * cw, H)], fill=(255, 255, 255), width=1)
    return out


def colour_bar(
    width: int, height: int, vmax: float, *,
    teal_label: str, coral_label: str,
) -> Image.Image:
    """A horizontal ramp that NAMES both poles, in logit units.

    An earlier bar read `change in the judgment`, which describes the swap
    delta. The cells carry the opposite sign of that, so a reader who took
    the bar at its word read every panel backwards. A bar that names the 2
    labels cannot go wrong in that way.
    """
    from PIL import Image as PILImage, ImageDraw

    bar = PILImage.new("RGB", (width, height), (255, 255, 255))
    px = bar.load()
    for x in range(width):
        t = 2.0 * x / max(width - 1, 1) - 1.0
        col = pa._div_color(t)
        for y in range(height - 16):
            px[x, y] = col
    draw = ImageDraw.Draw(bar)
    left = f"{teal_label}  {vmax:.2f}"
    right = f"{vmax:.2f}  {coral_label}"
    draw.text((2, height - 14), left, fill=(20, 20, 20))
    draw.text((width // 2 - 24, height - 14), "0  logits", fill=(20, 20, 20))
    draw.text((width - 7 * len(right) - 2, height - 14), right,
              fill=(20, 20, 20))
    return bar


def contact_sheet(
    panels: list[tuple[str, Image.Image]], vmax: float, cols: int = 5, *,
    teal_label: str, coral_label: str,
) -> Image.Image:
    """A labelled grid of panels, one shared colour bar, no title.

    The caption owns the title. Per-panel labels stay, because a reader has to
    tell the panels apart.
    """
    from PIL import Image as PILImage, ImageDraw

    if not panels:
        raise ValueError("no panels to draw")
    cell = 260
    lab = 18
    rows = (len(panels) + cols - 1) // cols
    W = cols * cell
    H = rows * (cell + lab) + 40
    sheet = PILImage.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(panels):
        r, c = divmod(i, cols)
        x, y = c * cell, r * (cell + lab)
        sheet.paste(img.resize((cell - 6, cell - 6), PILImage.LANCZOS),
                    (x + 3, y + 3))
        draw.text((x + 4, y + cell - 2), name, fill=(40, 40, 40))
    sheet.paste(colour_bar(min(W - 40, 480), 34, vmax,
                           teal_label=teal_label, coral_label=coral_label),
                (20, H - 38))
    return sheet


def map_for_image(
    model: Any, lens_model: Any, tgt: dict, donors: list[dict], layer: int,
    pos_id: int, neg_id: int, batch: int, group: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """(mean map over donors, spread over donors). Sign = CONTRIBUTION."""
    runs = []
    for don in donors:
        d = pmc.causal_map(
            model, lens_model, tgt["inputs"], layer, tgt["block"],
            don["h"][layer], tgt["c"], pos_id, neg_id, batch=batch,
            group=group)
        # A swap delta is the effect of REMOVING the patch. Flip it so coral
        # means "this patch pushes toward the positive label".
        runs.append(-d)
    arr = np.stack(runs)
    return arr.mean(0), arr.std(0)


#: How a cell is taken away. See `patch_ablate` for why `absent` is better.
REFERENCES = ("donor", "absent", "substitute")

#: What each reference writes into `summary.json`. The label must name the
#: edit that ran. A stale label made every `absent` map claim a donor swap.
METHOD_NOTE = {
    "donor": "RETIRED: per-patch residual swap at a layer, donor averaged",
    "absent": "per-patch removal at the input; no other photograph is read",
    "substitute": "per-patch pixel swap at the input, donor averaged",
}


def case_dir_for(
    out_dir: Path, case: str, group: int, reference: str = "donor",
    layout: str = "image_first", filler_kind: str = "neutral",
) -> Path:
    """Every (group, reference, layout, filler) set gets its own folder.

    A run must never overwrite the run it is compared against. The layout and
    the filler kind enter the name only when they are not the default, so
    every folder written before 2026-08-26 keeps its path.
    """
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}")
    name = case if group == 1 else f"{case}_g{group}"
    name = name if reference == "donor" else f"{name}_{reference}"
    name = name if layout == "image_first" else f"{name}_{layout}"
    return out_dir / (name if filler_kind == "neutral"
                      else f"{name}_{filler_kind}")


def map_by_absence(
    model: Any, lens_model: Any, tgt: dict, pos_id: int, neg_id: int,
    batch: int, group: int, image_token_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(map, spread). No donor, so the spread is 0 by construction."""
    d = pab.ablation_map(
        model, lens_model, tgt["inputs"], tgt["block"], tgt["c"],
        pos_id, neg_id, batch=batch, group=group,
        image_token_id=image_token_id)
    # An ablation delta is the effect of REMOVING the patch, so flip it.
    return -d, np.zeros_like(d)


def map_by_substitution(
    model: Any, lens_model: Any, tgt: dict, donors: list, pos_id: int,
    neg_id: int, batch: int, group: int, image_token_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(mean map over donors, spread over donors). Sign = CONTRIBUTION."""
    runs = []
    for px in donors:
        d = pab.substitution_map(
            model, lens_model, tgt["inputs"], tgt["block"], px, tgt["c"],
            pos_id, neg_id, batch=batch, group=group,
            image_token_id=image_token_id)
        runs.append(-d)
    arr = np.stack(runs)
    return arr.mean(0), arr.std(0)


def load_targets(
    case: str, proc: Any, model: Any, lens_model: Any, tmpl: str, *,
    n_images: int, layer: Optional[int], seed: int, device: str,
    reference: str, pos_id: int, neg_id: int, image_token_id: int,
    label_over: Optional[dict] = None, layout: str = "image_first",
    filler: Optional[str] = None,
) -> list[dict]:
    """The images, their patch positions, their `c`, and what each needs.

    A donor run also needs the residual at `layer`. An absence run needs
    neither the residual nor the layer, so it does not record them.
    """
    store = []
    for r in sg.load_single_rows(n_images, case, seed=seed):
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, case,
                                        overrides=label_over,
                                        device=device, layout=layout,
                                        filler=filler)
        block = pab.slots_of(inputs, image_token_id)
        item = {**r, "inputs": inputs, "block": block, "image": img}
        if reference == "donor":
            c, h = psw.contrast_and_residual(
                model, lens_model, inputs, layer, pos_id, neg_id)
            item.update(c=c, h={layer: h[block, :]})
        else:
            item["c"] = float(pab.contrast(
                model, lens_model, inputs, pos_id, neg_id)[0])
        store.append(item)
    return store


def run_case(
    case: str, proc: Any, model: Any, lens_model: Any, tmpl: str, *,
    n_images: int, n_donors: int, layer: Optional[int], batch: int, seed: int,
    out_dir: Path, device: str, group: int = 1, reference: str = "donor",
    label_pair: Optional[tuple] = None, question: Optional[str] = None,
    question_case: Optional[str] = None, allow_weak: bool = False,
    layout: str = "image_first", filler_kind: str = "neutral",
) -> pd.DataFrame:
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}")
    tok = proc.tokenizer
    label_over = ({"positive": label_pair[0], "negative": label_pair[1]}
                  if label_pair else {})
    if question:
        label_over["question"] = question
        log(f"the question is reworded: {question!r}")
    if question_case:
        # The whole rating of the other case, and not the question alone.
        # `subject` names the thing the question is about, so a swap of the
        # question alone leaves the prompt about subway entrances.
        label_over.update(sg.rating(question_case))
        log(f"the images come from {case}, the question from {question_case}")
    label_over = label_over or None
    if label_over:
        log(f"the answer words are {label_over.get('positive')!r} and "
            f"{label_over.get('negative')!r}")
    pos_id, neg_id = sg.contrast_ids(tok, case, label_over)
    image_token_id = int(model.config.image_token_id)
    rng = np.random.default_rng(seed)

    # A filler layout needs a block that matches the question in TOKENS, so
    # the arm holds the patch positions fixed. See monocle/modality_order.py.
    filler = None
    if layout in sg.FILLER_LAYOUTS:
        filler = mo.build_filler(
            tok, sg.prompt_text(case, label_over), filler_kind)
        log(f"layout {layout}: the {filler_kind} filler holds "
            f"{len(tok(filler, add_special_tokens=False)['input_ids'])} "
            f"tokens: {filler[:70]!r}")
    elif filler_kind != "neutral":
        raise ValueError(
            f"layout {layout!r} takes no filler, so --filler-kind "
            f"{filler_kind!r} would do nothing")
    elif layout != "image_first":
        log(f"layout {layout}: NOT the canonical order. This is a probe.")

    store = load_targets(
        case, proc, model, lens_model, tmpl, n_images=n_images, layer=layer,
        seed=seed, device=device, reference=reference, pos_id=pos_id,
        neg_id=neg_id, image_token_id=image_token_id,
        label_over=label_over, layout=layout, filler=filler)
    where = f"L{layer}" if reference == "donor" else "the input"
    log(f"{case}: {len(store)} images, {reference} reference at {where}, "
        f"c range {min(s['c'] for s in store):+.2f} to "
        f"{max(s['c'] for s in store):+.2f}")

    rows, maps, worths = [], [], []
    for i, tgt in enumerate(store):
        extra: dict = {}
        if reference == "donor":
            pool = [j for j in range(len(store)) if j != i]
            pick = rng.choice(pool, size=min(n_donors, len(pool)),
                              replace=False)
            donors = [store[int(j)] for j in pick]
            mean, spread = map_for_image(
                model, lens_model, tgt, donors, layer, pos_id, neg_id, batch,
                group=group)
            extra["n_donors"] = len(donors)
        else:
            # The controls must sit on the map's batch, or the whole and the
            # parts land on 2 different references. See `pab.controls`.
            ctl = pab.controls(
                model, lens_model, tgt["inputs"], tgt["block"], tgt["c"],
                pos_id, neg_id, image_token_id=image_token_id, batch=batch)
            pab.gate_rebuild(ctl)
            if reference == "absent":
                mean, spread = map_by_absence(
                    model, lens_model, tgt, pos_id, neg_id, batch, group,
                    image_token_id)
                extra["n_donors"] = 0
            else:
                pool = [j for j in range(len(store)) if j != i]
                pick = rng.choice(pool, size=min(n_donors, len(pool)),
                                  replace=False)
                px = [store[int(j)]["inputs"]["pixel_values"][0] for j in pick]
                mean, spread = map_by_substitution(
                    model, lens_model, tgt, px, pos_id, neg_id, batch, group,
                    image_token_id)
                extra["n_donors"] = len(px)
            extra["image_worth"] = ctl["image_worth"]
            extra["null_error"] = ctl["null_error"]
            extra["batch_offset"] = ctl["batch_offset"]
            extra["measure_batch"] = batch
            # A weak image is kept and flagged. Its ratio has a near-zero
            # denominator, so the sweep leaves it out of the median.
            extra["worth_ok"] = bool(ctl["worth_ok"])
            extra["parts_over_whole"] = float(
                mean.sum() / ctl["image_worth"]) if ctl["image_worth"] else \
                float("nan")
            worths.append(ctl["image_worth"])

        top = max(1, round(0.05 * mean.size))
        rows.append({
            "case": case, "layer": layer, "group": group,
            "reference": reference, "image_id": tgt["image_id"],
            "path": tgt["path"], "c": tgt["c"],
            "max_abs": float(np.max(np.abs(mean))),
            "top5pct_share": float(
                np.sum(np.sort(np.abs(mean))[-top:]) / np.sum(np.abs(mean))),
            "mean_spread": float(spread.mean()),
            "stability": float(np.mean(np.abs(mean)) / (spread.mean() + 1e-9)),
            **extra,
        })
        maps.append({"image_id": tgt["image_id"], "layer": layer,
                     "values": mean.tolist(), "spread": spread.tolist()})
        note = (f" whole={extra['image_worth']:+.2f} "
                f"parts/whole={extra['parts_over_whole']:+.2f}"
                f"{'' if extra['worth_ok'] else '  [weak image]'}"
                if reference == "absent" else "")
        log(f"  {i + 1}/{len(store)} {tgt['image_id']} "
            f"max|dc|={rows[-1]['max_abs']:.3f}{note}")

    df = pd.DataFrame(rows)
    case_dir = case_dir_for(out_dir, case, group, reference, layout,
                            filler_kind)
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(case_dir / "map_stats.parquet", index=False)
    pd.DataFrame(maps).to_parquet(case_dir / "maps.parquet", index=False)

    # The gate comes AFTER the write. It used to come before, and a battery
    # that failed the gate threw away every map it had just measured. The
    # gate still stops the job, so a bad run cannot pass in silence.
    if reference == "absent":
        # 1 weak image is data. Most of them means the ablation is dead.
        weak = sum(1 for w in worths if abs(w) < pab.WORTH_FLOOR)
        if weak:
            log(f"{weak} of {len(worths)} images are worth under "
                f"{pab.WORTH_FLOOR}; the sweep leaves them out of the median")
        if allow_weak:
            log("--allow-weak is set: the gate reports and does not stop.")
            # `gate_battery` raises SystemExit, which comes from BaseException
            # and NOT from Exception. An `except Exception` thus lets it go,
            # the process exits 1, and a complete run reads as FAILED. It also
            # breaks an `afterok` dependency. Catch SystemExit by name.
            try:
                pab.gate_battery(worths)
            except SystemExit as exc:
                log(f"GATE would have stopped this run: {exc}")
            except Exception as exc:  # noqa: BLE001
                log(f"GATE would have stopped this run: {exc}")
        else:
            pab.gate_battery(worths)

    draw_figures(case, case_dir,
                 [(t["image_id"], t["image"], np.array(m["values"]), t["c"])
                  for t, m in zip(store, maps)],
                 {"layer": layer, "seed": seed, "group": group,
                  "reference": reference,
                  "n_donors": n_donors if reference == "donor" else 0,
                  "cells": f"{group}x{group} patches, "
                           f"{48 * group} px of the model image"})
    return df


def draw_figures(
    case: str, case_dir: Path,
    items: list[tuple[str, Image.Image, np.ndarray, float]],
    meta: dict,
) -> float:
    """Write the panels, the sheet, and the summary. Returns `vmax`.

    The measurement and the drawing are separate, so the figures can be drawn
    again from the parquet when only the style changes. A GPU run is not
    needed to change a colour.
    """
    rating = sg.rating(case)
    panel_dir = case_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(panel_dir.glob("*.png"))
    for f in stale:
        f.unlink()
    if stale:
        log(f"removed {len(stale)} panels of the run before this one")

    # One scale for every panel: the 99th percentile over all maps.
    allv = np.concatenate([np.abs(v) for _, _, v, _ in items])
    vmax = float(np.percentile(allv, 99))
    panels = []
    for image_id, img, values, c in items:
        pic = overlay(img, values, vmax)
        pic.save(case_dir / "panels" / f"{image_id}.png")
        panels.append((f"{image_id}  c={c:+.1f}", pic))
    # Coral is the negative label, so teal takes the positive one.
    sheet = contact_sheet(panels, vmax, teal_label=rating["positive"],
                          coral_label=rating["negative"])
    sheet.save(case_dir / "contact_sheet.png")
    (case_dir / "summary.json").write_text(json.dumps({
        "case": case, "n_images": len(items), "vmax": vmax,
        "positive": rating["positive"], "negative": rating["negative"],
        "data_sign": f"values are the contribution toward {rating['positive']}",
        "colour_sign": f"coral is {rating['negative']}, teal is "
                       f"{rating['positive']}",
        "method": METHOD_NOTE[meta.get("reference", "donor")],
        **meta,
    }, indent=2))
    log(f"wrote {case_dir} (vmax {vmax:.3f})")
    return vmax


def rerender(case: str, case_dir: Path) -> float:
    """Draw the figures again from the parquet. No GPU, no model.

    Use this when only the style changes: a colour, a label, a scale. The
    measurement is the expensive part and it does not change.

    The stored `image_id` is NOT trusted. A run before the naming fix wrote
    the bare file stem, and a Cyclomedia stem is only the face letter, so
    several photographs share a name. The path is the ground truth, so the
    name is built from it again.
    """
    stats = pd.read_parquet(case_dir / "map_stats.parquet")
    maps = pd.read_parquet(case_dir / "maps.parquet")
    if len(stats) != len(maps):
        raise SystemExit(
            f"{case_dir}: {len(stats)} stat rows against {len(maps)} maps. "
            "The 2 files come from one loop, so they must agree.")
    ids = [sg.photograph_id(x) for x in stats["path"]]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{case_dir}: the photograph names still collide")
    items = []
    for i, path in enumerate(stats["path"]):
        items.append((ids[i], Image.open(path).convert("RGB"),
                      np.array(maps["values"].iloc[i]),
                      float(stats["c"].iloc[i])))
    stats = stats.assign(image_id=ids)
    stats.to_parquet(case_dir / "map_stats.parquet", index=False)
    maps.assign(image_id=ids).to_parquet(
        case_dir / "maps.parquet", index=False)
    # An absence run has no layer, so the column holds a null.
    raw = stats["layer"].iloc[0]
    layer = None if pd.isna(raw) else int(raw)
    ref = (stats["reference"].iloc[0] if "reference" in stats.columns
           else "donor")
    return draw_figures(case, case_dir, items,
                        {"layer": layer, "reference": ref,
                         "rerendered": True})


def additivity(
    case: str, out_dir: Path, group: int, reference: str = "donor",
) -> pd.DataFrame:
    """Does a group swap equal the sum of its parts? No GPU.

    Each cell of a `k` x `k` run is compared against the sum of the `k**2`
    fine cells below it. The 2 runs must share the seed, so they draw the same
    images and the same donors.

    | Ratio group / sum of parts | Reading |
    |---|---|
    | about 1 | the patches act on their own; the map decomposes `c` |
    | far below 1 | the patches repeat each other; the fine map double counts |
    | above 1 | the patches work together; the fine map misses the joint effect |
    """
    if group < 2:
        raise SystemExit("--additivity needs --group 2 or more")
    fine_dir = case_dir_for(out_dir, case, 1, reference)
    grp_dir = case_dir_for(out_dir, case, group, reference)
    fine = pd.read_parquet(fine_dir / "maps.parquet").set_index("image_id")
    grp = pd.read_parquet(grp_dir / "maps.parquet").set_index("image_id")
    shared = [i for i in grp.index if i in fine.index]
    if not shared:
        raise SystemExit(
            f"no image is in both {fine_dir.name} and {grp_dir.name}. "
            "The 2 runs need the same seed and the same image count.")

    rows = []
    for image_id in shared:
        f = np.array(fine.loc[image_id, "values"])
        g = np.array(grp.loc[image_id, "values"])
        blocks = pmc.patch_groups(f.size, group)
        if len(blocks) != g.size:
            raise SystemExit(
                f"{image_id}: {g.size} group cells against {len(blocks)} "
                f"blocks of a {f.size}-patch grid")
        parts = np.array([f[b].sum() for b in blocks])
        rows.append({
            "case": case, "image_id": image_id, "group": group,
            "reference": reference,
            "sum_parts": float(parts.sum()), "sum_group": float(g.sum()),
            "ratio": float(g.sum() / parts.sum()) if parts.sum() else np.nan,
            "pearson": float(np.corrcoef(parts, g)[0, 1]),
            "sign_agree": float(np.mean(np.sign(parts) == np.sign(g))),
            "mean_abs_parts": float(np.abs(parts).mean()),
            "mean_abs_group": float(np.abs(g).mean()),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(grp_dir / "additivity.parquet", index=False)
    log(f"{case} g{group} {reference}: {len(df)} images, "
        f"median |group| / |sum of parts| = "
        f"{(df.mean_abs_group / df.mean_abs_parts).median():.3f}, "
        f"median cell r = {df.pearson.median():.3f}, "
        f"median sign agreement = {df.sign_agree.median():.3f}")
    log(f"wrote {grp_dir / 'additivity.parquet'}")
    return df


def group_sweep(
    case: str, out_dir: Path, reference: str, groups: list[int],
) -> pd.DataFrame:
    """How big must a hole be before the parts sum to the whole? No GPU.

    Only the `absent` reference can answer this, because only it has a defined
    right answer. `image_worth` is `c` with the picture against `c` with no
    picture. If the cells were independent, they would sum to it.

    The sweep carries its own end point. At `group` = the grid side there is 1
    cell, and that cell IS the whole-image ablation, so `parts_over_whole`
    must read 1.000. A row that does not is a bug in the machinery, not a
    finding about the model.
    """
    if reference != "absent":
        raise SystemExit(
            "the group sweep needs --reference absent. A donor run has no "
            "whole-image number to compare the parts against.")
    rows = []
    for g in sorted(groups):
        d = case_dir_for(out_dir, case, g, reference)
        f = d / "map_stats.parquet"
        if not f.exists():
            log(f"no run at group {g} ({d.name}); skipped")
            continue
        st = pd.read_parquet(f)
        maps = pd.read_parquet(d / "maps.parquet")
        for i, image_id in enumerate(st["image_id"]):
            v = np.array(maps["values"].iloc[i])
            worth = float(st["image_worth"].iloc[i])
            rows.append({
                "case": case, "group": g, "image_id": image_id,
                "worth_ok": bool(abs(worth) >= pab.WORTH_FLOOR),
                "cells": int(v.size),
                "px": 48 * g,
                "c": float(st["c"].iloc[i]),
                "image_worth": worth,
                "sum_parts": float(v.sum()),
                "parts_over_whole": float(v.sum() / worth) if worth else
                np.nan,
                "max_abs": float(np.abs(v).max()),
                "max_share": float(np.abs(v).max() / abs(worth)) if worth
                else np.nan,
            })
    if not rows:
        raise SystemExit(f"no absence run of {case} on disk")

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / f"{case}_group_sweep.parquet", index=False)
    # A weak image has a near-zero denominator, so its ratio is noise times a
    # large number. The median runs over the images that carry a ratio.
    strong = df[df.worth_ok]
    weak = df.image_id.nunique() - strong.image_id.nunique()
    log(f"{case}: the group-size curve, {strong.image_id.nunique()} images"
        f"{f' ({weak} weak, left out)' if weak else ''}")
    log(f"{'group':>5} {'cells':>6} {'px':>5} {'parts/whole':>12} "
        f"{'strongest cell':>15}")
    for g, sub in strong.groupby("group"):
        log(f"{g:>5} {sub.cells.iloc[0]:>6} {sub.px.iloc[0]:>5} "
            f"{sub.parts_over_whole.median():>12.3f} "
            f"{sub.max_share.median():>15.3f}")

    # The end point is a machinery check, so it runs over EVERY image. A weak
    # image still has 1 cell that equals its own whole-image ablation.
    side = int(round(df[df.group == df.group.min()].cells.iloc[0] ** 0.5))
    end = df[df.group == side]
    if len(end):
        off = (end.parts_over_whole - 1.0).abs().max()
        state = "PASS" if off < 1e-3 else "FAIL"
        log(f"end point (group {side}, 1 cell): parts/whole must be 1.000, "
            f"worst error {off:.5f} — {state}")
        if state == "FAIL":
            raise SystemExit(
                "the 1-cell run does not reproduce the whole-image ablation. "
                "The grouping or the sign is wrong, so read no curve from it.")
    else:
        log(f"no group {side} run, so the curve has no end-point check")
    return df


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["subway_safety"],
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--n-donors", type=int, default=3)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--allow-weak", action="store_true",
                    help="report the weak-image gate, and do not stop on it")
    ap.add_argument("--question-case", default=None,
                    choices=list(sg.CASE_RATING),
                    help="take the whole question of another case")
    ap.add_argument("--question", default=None,
                    help="a question that replaces the case question")
    ap.add_argument("--labels", nargs=2, default=None,
                    metavar=("POSITIVE", "NEGATIVE"),
                    help="answer words that replace the case pair")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--group", type=int, nargs="+", default=[1],
                    metavar="K",
                    help="take K x K blocks of patches, not 1 patch. Give "
                         "several to sweep, e.g. --group 1 2 4 8 16. One "
                         "model load serves them all.")
    ap.add_argument("--reference", default="absent", choices=REFERENCES,
                    help="absent (the default): remove the patch at the "
                         "input. No other photograph is read. "
                         "substitute: the cell takes a donor's pixels; a "
                         "control, not a map. "
                         "donor: RETIRED, a residual swap at --layer")
    ap.add_argument("--layout", default="image_first", choices=sg.LAYOUTS,
                    help="the block order. image_first is production; the "
                         "others are probes and write their own folder. "
                         "See vlm-narratives-docs/modality-order.md")
    ap.add_argument("--filler-kind", default="neutral",
                    choices=list(mo.FILLER_KINDS),
                    help="the content of the filler block, for a layout that "
                         "has one. neutral: 1 sentence, repeated. decoy: "
                         "varied prose about an unrelated subject, which "
                         "separates the QUESTION from any informative text.")
    ap.add_argument("--rerender", action="store_true",
                    help="draw the figures again from the parquet; no GPU")
    ap.add_argument("--additivity", action="store_true",
                    help="compare a grouped run against the fine one; no GPU")
    ap.add_argument("--sweep", action="store_true",
                    help="report the group-size curve from the runs on "
                         "disk; no GPU")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # An absence run edits the input, so it has no layer to be wrong about.
    layer = args.layer if args.reference == "donor" else None
    if layer is not None and layer >= 30:
        raise SystemExit(
            f"L{args.layer} is above the causal window. The patches cause "
            "nothing there, so a causal map would be all noise. Use L20.")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted(set(int(g) for g in args.group))
    if args.sweep:
        for case in args.cases:
            group_sweep(case, out_dir, args.reference, groups)
        return 0
    if args.additivity:
        for case in args.cases:
            for g in groups:
                if g > 1:
                    additivity(case, out_dir, g, args.reference)
        return 0
    if args.rerender:
        for case in args.cases:
            for g in groups:
                rerender(case, case_dir_for(
                    out_dir, case, g, args.reference, args.layout,
                    args.filler_kind))
        log("figures redrawn.")
        return 0

    from monocle import attn_route as ar

    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    for case in args.cases:
        # The groups share 1 model load, and the same seed gives them the
        # same images, so the sweep compares like with like.
        for g in groups:
            run_case(case, proc, model, lens_model, tmpl,
                     n_images=args.n_images, n_donors=args.n_donors,
                     layer=layer, batch=args.batch, seed=args.seed,
                     out_dir=out_dir, device=args.device, group=g,
                     reference=args.reference,
                     label_pair=(tuple(args.labels) if args.labels
                                 else None),
                     question=args.question,
                     question_case=args.question_case,
                     allow_weak=args.allow_weak,
                     layout=args.layout, filler_kind=args.filler_kind)
    if len(groups) > 1 and args.reference == "absent":
        for case in args.cases:
            group_sweep(case, out_dir, args.reference, groups)
    log("causal maps complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
