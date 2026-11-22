"""The caption loop as a picture: Safe cells, all cells, Dangerous cells.

What this draws
---------------
One panel for each photograph. 3 columns and 2 rows.

| Column | The picture | The words under it |
|---|---|---|
| left | the whole photograph, only the Safe cells kept | what the model says it sees |
| middle | the causal map on the whole photograph | the same, with every cell kept |
| right | the whole photograph, only the Dangerous cells kept | the same |

Why the whole photograph stays
------------------------------
The model reads ONLY the kept quarter: `caption_loop` takes the other cells
away at the input. A reader cannot judge that from a picture of a quarter,
because the removed part is where the context is. So the panel keeps the
whole photograph and GHOSTS the cells the model did not get. The reader then
sees the same scene the map was built on, and sees which quarter of it made
the sentence.

The tint is the pole colour of the map ramp, so a column cannot be read
against the wrong label.

No GPU
------
The drawing reads `captions.parquet` and `maps.parquet`, both written by
`caption_loop`. Thus a change of style costs nothing:

    python -m research.monocle.caption_panel --dir outputs/_monocle/caption_loop/<run>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import causal_maps as cm  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import single as sg  # noqa: E402

#: The 2 ends of the diverging ramp `causal_maps` paints with.
TEAL = pa._div_color(-1.0)
CORAL = pa._div_color(1.0)

#: The 3 columns, in the order they are drawn.
COLUMNS = ("keep_safe", "whole_image", "keep_danger")

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def log(m: str) -> None:
    print(f"[caption-panel] {m}", flush=True)


def font(size: int, bold: bool = False) -> Any:
    """DejaVu when the node has it, else whatever PIL ships."""
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def wrap(text: str, fnt: Any, width: int, draw: ImageDraw.ImageDraw) -> list:
    """Greedy word wrap, measured in pixels and not in characters."""
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if draw.textlength(trial, font=fnt) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


def highlight(
    image: Image.Image, cells, n_cells: int, colour: tuple, *,
    upscale: int = 2, ghost: float = 0.45, keep_sat: float = 0.25,
    tint: float = 0.20, border: int = 2,
) -> Image.Image:
    """The whole photograph, with only `cells` kept and the rest ghosted.

    A removed cell is first drained of colour (`keep_sat` is how much of its
    own colour it keeps) and then faded toward white by `ghost`. It stays
    READABLE on purpose. The reader has to see the whole scene to judge the
    quarter the model was given, and a white hole hides exactly the context
    that makes the caption checkable.

    `tint` washes the kept cells with the pole colour, and a border of the
    same colour closes each one, so the kept quarter reads at a glance.
    """
    n_rows, n_cols = cm.grid_of(n_cells)
    keep = {int(c) for c in cells}
    W, H = image.size
    W, H = W * upscale, H * upscale
    base = image.convert("RGB").resize((W, H), Image.LANCZOS)

    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for c in keep:
        mask[c // n_cols, c % n_cols] = True
    big = np.array(Image.fromarray(
        mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127

    arr = np.array(base, dtype=np.float32)
    white = np.full_like(arr, 255.0)
    wash = np.zeros_like(arr)
    wash[:] = np.array(colour, dtype=np.float32)
    grey = arr.mean(axis=-1, keepdims=True).repeat(3, axis=-1)
    muted = grey * (1.0 - keep_sat) + arr * keep_sat
    out = np.where(big[..., None],
                   arr * (1.0 - tint) + wash * tint,
                   muted * (1.0 - ghost) + white * ghost)
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    if border:
        draw = ImageDraw.Draw(img)
        cw, ch = W / n_cols, H / n_rows
        for c in keep:
            r, k = c // n_cols, c % n_cols
            draw.rectangle([k * cw, r * ch, (k + 1) * cw - 1, (r + 1) * ch - 1],
                           outline=colour, width=border)
    return img


def column_images(
    image: Image.Image, values: np.ndarray, safe_cells, danger_cells,
    vmax: float,
) -> dict:
    """The 3 pictures, keyed by arm."""
    return {
        "keep_safe": highlight(image, safe_cells, values.size, TEAL),
        "whole_image": cm.overlay(image, values, vmax),
        "keep_danger": highlight(image, danger_cells, values.size, CORAL),
    }


#: What the bottom row shows.
READOUTS = ("rating", "thinking", "caption")


def verdict_block(
    draw: ImageDraw.ImageDraw, x: int, y: int, width: int,
    c: float, rating: dict, fonts: dict,
) -> int:
    """The production prompt's own answer, asked of the masked photograph.

    No text is generated, so nothing here can be a hallucination. The bar
    shows `sigmoid(c)`, which is the model's preference BETWEEN the 2 labels
    and needs no extra forward pass. It is not p(Safe) over the vocabulary.
    """
    p = 1.0 / (1.0 + np.exp(-c))
    safe_won = c >= 0
    won = rating["positive"] if safe_won else rating["negative"]
    # The WORD takes the colour of the verdict, not of the column. The header
    # above already names the column, so this way a reader sees at a glance
    # which way each of the 3 went.
    strength = max(p, 1.0 - p)
    # A logit contrast saturates fast. `sigmoid(6.4)` is 0.998, and a flat
    # `100%` claims a certainty the number does not carry.
    pct = ">99%" if strength >= 0.995 else f"{strength:.0%}"
    draw.text((x, y), f"{won}    {pct}",
              fill=(TEAL if safe_won else CORAL), font=fonts["big"])

    bar_y = y + 26
    h = 12
    split = int(round(width * p))
    draw.rectangle([x, bar_y, x + split, bar_y + h], fill=TEAL)
    draw.rectangle([x + split, bar_y, x + width, bar_y + h], fill=CORAL)
    draw.rectangle([x, bar_y, x + width, bar_y + h], outline=(150, 150, 150))

    draw.text((x, bar_y + h + 5), f"c = {c:+.2f} logits",
              fill=(60, 60, 60), font=fonts["num"])
    return bar_y + h + 22


def triptych(
    image_id: str, image: Image.Image, values: np.ndarray,
    safe_cells, danger_cells, captions: dict, c_text: dict, vmax: float, *,
    case: str, c_image: float, width: int = 380, gap: int = 14,
    max_lines: int = 6, readout: str = "rating",
    c_rate: Optional[dict] = None, closed: Optional[dict] = None,
) -> Image.Image:
    """One photograph: 3 pictures on top, and what the model said below them.

    | `readout` | The bottom row |
    |---|---|
    | `rating` | the production prompt, thought channel closed at once |
    | `thinking` | the same prompt with the channel OPEN, and the trace |
    | `caption` | an open question, and the sentence read back as text |

    `rating` and `thinking` are the same question. The template closes the
    thought channel at once for the first and leaves it open for the second,
    so the difference is reasoning and nothing else.

    `rating` generates nothing, so no sentence and no hallucination can enter
    it. `caption` says WHAT the model sees and pays for it with a fabricated
    weapon on about 1 photograph in 8.

    Under `caption` the text block is sized to the LONGEST of the 3, so a
    short set of sentences leaves no band of white.
    """
    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}")
    if readout in ("rating", "thinking") and not c_rate:
        raise ValueError(f"the {readout} readout needs a `c` for each arm")
    rating = sg.rating(case)
    heads = {
        "keep_safe": f"only the {rating['positive']} cells",
        "whole_image": "every cell",
        "keep_danger": f"only the {rating['negative']} cells",
    }
    tone = {"keep_safe": TEAL, "whole_image": (70, 70, 70),
            "keep_danger": CORAL}

    pics = column_images(image, values, safe_cells, danger_cells, vmax)
    thumbs = {k: v.resize((width, width), Image.LANCZOS)
              for k, v in pics.items()}

    f_head = font(15, bold=True)
    f_body = font(13)
    f_num = font(12, bold=True)
    f_big = font(17, bold=True)
    f_id = font(15, bold=True)
    fonts = {"big": f_big, "num": f_num}
    line_h = 17
    title_y = 8
    head_y = title_y + 24
    top = head_y + 21           # the headers must not sit on the title
    W = 3 * width + 4 * gap

    # Wrap first, so the text block is only as tall as the words need.
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    wrapped = {a: wrap(captions.get(a, ""), f_body, width - 6, probe)[:max_lines]
               for a in COLUMNS}
    n_lines = max(1, max(len(v) for v in wrapped.values()))
    text_top = top + width + 26
    if readout == "rating":
        text_h = 65
    elif readout == "thinking":
        text_h = 65 + n_lines * line_h + 8
    else:
        text_h = n_lines * line_h + 26
    H = text_top + text_h + 50

    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, title_y),
              f"{image_id}    the whole photograph is worth "
              f"{c_image:+.2f} logits", fill=(30, 30, 30), font=f_id)

    for i, arm in enumerate(COLUMNS):
        x = gap + i * (width + gap)
        draw.text((x, head_y), heads[arm], fill=tone[arm], font=f_head)
        sheet.paste(thumbs[arm], (x, top))
        draw.line([(x, text_top - 11), (x + width, text_top - 11)],
                  fill=tone[arm], width=2)
        if readout in ("rating", "thinking"):
            if arm not in c_rate:
                continue
            below = verdict_block(draw, x, text_top, width, c_rate[arm],
                                  rating, fonts)
            if readout == "thinking":
                if closed and not closed.get(arm, True):
                    # No answer position exists, so the number above is not a
                    # verdict. Say so on the panel, not only in the parquet.
                    draw.text((x, below), "the thought never closed",
                              fill=CORAL, font=f_num)
                    below += line_h
                for j, line in enumerate(wrapped[arm]):
                    draw.text((x, below + j * line_h), line,
                              fill=(90, 90, 90), font=f_body)
            continue
        for j, line in enumerate(wrapped[arm]):
            draw.text((x, text_top + j * line_h), line, fill=(35, 35, 35),
                      font=f_body)
        if arm in c_text:
            draw.text((x, text_top + n_lines * line_h + 7),
                      f"read back as text alone: {c_text[arm]:+.2f}",
                      fill=tone[arm], font=f_num)

    sheet.paste(cm.colour_bar(min(W - 2 * gap, 460), 34, vmax,
                              teal_label=rating["positive"],
                              coral_label=rating["negative"]),
                (gap, H - 42))
    return sheet


def load_run(run_dir: Path) -> tuple:
    """(captions, maps) for one caption-loop run."""
    caps = run_dir / "captions.parquet"
    maps = run_dir / "maps.parquet"
    for f in (caps, maps):
        if not f.exists():
            raise SystemExit(
                f"{f} is missing. `maps.parquet` is written by caption_loop "
                "from 2026-08-23; a run made before that has to be measured "
                "again.")
    return pd.read_parquet(caps), pd.read_parquet(maps)


def draw_run(
    run_dir: Path, *, strong_only: bool = True, readout: str = "rating",
) -> int:
    """One triptych for each photograph. Returns the count drawn."""
    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}")
    caps, maps = load_run(run_dir)
    need = {"rating": "c_rate", "thinking": "c_think",
            "caption": "c_text"}[readout]
    if need not in caps.columns:
        raise SystemExit(
            f"this run holds no `{need}`, so the {readout} readout cannot be "
            "drawn. Runs from before 2026-08-23 measured the caption only; "
            "use --readout caption, or measure the run again.")
    case = str(maps["case"].iloc[0])
    allv = np.concatenate([np.abs(np.asarray(v, dtype=float))
                           for v in maps["values"]])
    vmax = float(np.percentile(allv, 99))

    out = run_dir / f"triptych_{readout}"
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(out.glob("*.png")):
        f.unlink()

    drawn = 0
    for _, m in maps.iterrows():
        if strong_only and not bool(m["worth_ok"]):
            continue
        sub = caps[caps.image_id == m["image_id"]]
        cap = {a: str(sub[sub.arm == a].caption.iloc[0])
               for a in COLUMNS if (sub.arm == a).any()}
        ct = {a: float(sub[sub.arm == a].c_text.iloc[0])
              for a in COLUMNS if (sub.arm == a).any()}
        cr = ({a: float(sub[sub.arm == a][need].iloc[0])
               for a in COLUMNS if (sub.arm == a).any()}
              if need in caps.columns else None)
        cl = ({a: bool(sub[sub.arm == a].think_closed.iloc[0])
               for a in COLUMNS if (sub.arm == a).any()}
              if "think_closed" in caps.columns else None)
        if readout == "thinking":
            # The trace replaces the caption as the words on the panel.
            cap = {a: str(sub[sub.arm == a].reasoning.iloc[0])
                   for a in COLUMNS if (sub.arm == a).any()}
        img = Image.open(m["path"]).convert("RGB")
        panel = triptych(
            str(m["image_id"]), img, np.asarray(m["values"], dtype=float),
            m["safe_cells"], m["danger_cells"], cap, ct, vmax,
            case=case, c_image=float(m["c_image"]), readout=readout,
            c_rate=cr, closed=cl)
        panel.save(out / f"{m['image_id']}.png")
        drawn += 1
    log(f"wrote {drawn} {readout} triptychs to {out} (vmax {vmax:.3f})")
    return drawn


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True,
                    help="a caption-loop run directory, e.g. "
                         "outputs/_monocle/caption_loop/subway_safety_g2_k16")
    ap.add_argument("--all-images", action="store_true",
                    help="draw the weak photographs too")
    ap.add_argument("--readout", default="rating", choices=READOUTS,
                    help="rating (the default): the production prompt asked "
                         "of the masked photograph, which generates nothing. "
                         "caption: an open question and the sentence back")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    draw_run(Path(a.dir), strong_only=not a.all_images, readout=a.readout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
