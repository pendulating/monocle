"""Depth (LAYER-axis) visualizations for the Jacobian-lens monocle.

`monocle.jlens_read` produces, for one image, a *stack* of per-patch top-k
readouts — one score table per fitted layer (a sparse ascending set such as
``[6, 12, 18, 24, 30, 36, 42, 46, 47]``). This module turns that stack into
two depth views over the SAME base image:

    render_layer_gif       -> a list of PIL frames (one per layer) + save helper
    render_layer_scrubber  -> a single self-contained HTML page with a slider

Both reuse `monocle.render` for the word-cloud geometry/sizing so the drawing
stays identical to the single-layer overlay; this module only adds the layer
axis (frames / groups), the depth chrome (badge, progress bar, slider), and
the cross-layer score normalization described below.

Cross-layer normalization
--------------------------
`render.render_overlay` normalizes font size against a *robust* (5th..95th
percentile) min-max of whatever DataFrame it is handed — recomputed per call.
Handing it each layer independently would rescale every frame to its own
spread, so a dim early layer and a confident late layer would look the same
size: the depth signal would vanish.

We can't edit `render.py`. A plain affine rescale of the score column is a
no-op (render re-derives its bounds from whatever we pass), and a bare clamp
to a global range only pins the *lower* tail — each layer's own top patch
still saturates to full size, so the depth signal is still lost (measured:
every layer collapses to base_norm ~= 0.9). To make render's internal
5th..95th normalization reproduce the *global* scaling exactly, we pin BOTH
of its percentiles to the stack-wide robust range by:

  1. clamping every layer's `score` into the global robust range, and
  2. appending a small mass of non-drawing "anchor" rows at the global floor
     and ceiling. The anchors carry an out-of-grid `patch_idx`, which
     `render_overlay`/`render_svg` skip when drawing (their
     ``row >= n_rows`` guard) yet still include in the score array used for
     `_robust_bounds`. Enough anchor mass (~12% per tail) locks render's p5
     to the global floor and p95 to the global ceiling for every frame.

The net effect: a patch's font size becomes ``clip((score - global_lo) /
(global_hi - global_lo))`` in every frame, so dim early layers render small
and confident late layers render large — comparably across the stack. The
HTML scrubber, which we render ourselves, applies the same global range
directly via `_norm` and needs no anchors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# monocle.render lives beside us; mirror jlens_read's path bootstrap so this
# module imports whether it is loaded as a package member or a loose script.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import render  # noqa: E402

_SVG_FONT_FAMILY = "DejaVu Sans, sans-serif"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_layer_run(
    parquet_path: str | Path,
    meta_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, dict]:
    """Read a layered run's ``<id>.jlens.parquet`` and sibling meta JSON.

    The long DataFrame carries monocle's standard scoring columns plus a
    leading ``layer`` column. When ``meta_path`` is None it is inferred by
    swapping the ``.jlens.parquet`` suffix for ``.jlens.meta.json``.
    """
    pq = Path(parquet_path)
    df = pd.read_parquet(pq)
    if meta_path is not None:
        mp = Path(meta_path)
    else:
        name = pq.name
        if name.endswith(".jlens.parquet"):
            mp = pq.parent / (name[: -len(".jlens.parquet")] + ".jlens.meta.json")
        else:
            # Fall back to the generic single-suffix swap.
            mp = pq.parent / (pq.stem + ".meta.json")
    meta = json.loads(Path(mp).read_text())
    return df, meta


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _layer_list(df: pd.DataFrame, meta: dict) -> list[int]:
    """Ascending list of layers, from meta if present else the frame."""
    if meta.get("layers"):
        return sorted(int(x) for x in meta["layers"])
    return sorted(int(x) for x in df["layer"].unique())


def _global_bounds(df: pd.DataFrame) -> tuple[float, float]:
    """Robust (5th..95th pct) min-max over the WHOLE stack's rank-0 scores.

    Rank 0 is what drives per-patch font size in render's sizing core, so the
    cross-layer reference range is taken over rank-0 rows only; falls back to
    every row when a run has no rank column populated.
    """
    if "rank" in df.columns:
        r0 = df.loc[df["rank"] == 0, "score"].to_numpy(dtype=np.float32)
        if r0.size == 0:
            r0 = df["score"].to_numpy(dtype=np.float32)
    else:
        r0 = df["score"].to_numpy(dtype=np.float32)
    return render._robust_bounds(r0)


# --------------------------------------------------------------------------- #
# Animated GIF
# --------------------------------------------------------------------------- #
def _stamp_badge(frame: Image.Image, layer: int, idx: int, n_layers: int) -> None:
    """Draw an "Lxx" corner badge and a bottom-edge depth progress bar.

    Mutates ``frame`` (RGB) in place. The progress fraction is
    ``(idx + 1) / n_layers`` so the deepest frame fills the bar.
    """
    draw = ImageDraw.Draw(frame, "RGBA")
    w, h = frame.size
    label = f"L{int(layer):02d}"

    fs = max(14, int(round(min(w, h) * 0.045)))
    font = render.get_font(fs)
    pad = max(4, fs // 3)
    box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
    tw, th = box[2] - box[0], box[3] - box[1]

    x0 = y0 = pad
    x1 = x0 + tw + 2 * pad
    y1 = y0 + th + 2 * pad
    draw.rounded_rectangle([x0, y0, x1, y1], radius=pad, fill=(0, 0, 0, 190))
    # Offset by the glyph bbox origin so the label sits centered in the pill.
    draw.text(
        (x0 + pad - box[0], y0 + pad - box[1]),
        label,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0, 255),
    )

    bar_h = max(3, int(round(h * 0.012)))
    frac = (idx + 1) / max(1, n_layers)
    draw.rectangle([0, h - bar_h, w, h], fill=(255, 255, 255, 45))
    draw.rectangle([0, h - bar_h, int(round(w * frac)), h], fill=(255, 255, 255, 215))


def _pin_bounds(sub: pd.DataFrame, meta: dict, lo: float, hi: float) -> pd.DataFrame:
    """Clamp scores into [lo, hi] and append non-drawing anchor rows so
    ``render._robust_bounds`` locks to the global range (see module docstring).

    Anchors use an out-of-grid ``patch_idx`` (skipped by render's draw guard)
    and are added at ~12% mass per tail, enough to pull p5 to `lo` and p95 to
    `hi` regardless of the layer's own spread.
    """
    sub = sub.copy()
    sub["score"] = sub["score"].clip(lower=lo, upper=hi)
    n_rows, n_cols = render._grid_dims(meta)
    ghost_idx = n_rows * n_cols  # row == n_rows -> render skips drawing it
    m = max(2, int(np.ceil(0.12 * len(sub))))
    template = sub.iloc[0].to_dict()
    anchor_rows: list[dict] = []
    for val in (lo, hi):
        for _ in range(m):
            row = dict(template)
            row.update(patch_idx=ghost_idx, rank=0, token="", score=float(val))
            anchor_rows.append(row)
    return pd.concat([sub, pd.DataFrame(anchor_rows)], ignore_index=True)


def render_layer_gif(
    image: Image.Image,
    df: pd.DataFrame,
    meta: dict,
    *,
    k: int = 3,
    dim: float = 0.55,
    ms_per_frame: int = 800,
    loop: bool = True,
) -> list[Image.Image]:
    """One word-cloud frame per layer (ascending) over the SAME base image.

    Each layer's rows are pinned to the stack-wide global robust range (clamp
    + non-drawing anchors; see the module docstring) before being handed to
    ``render.render_overlay``, so font sizes are comparable frame-to-frame; a
    corner "Lxx" badge and a bottom depth bar are then stamped on.
    ``ms_per_frame`` and ``loop`` are carried for convenience but only take
    effect in ``save_layer_gif``.
    """
    layers = _layer_list(df, meta)
    lo, hi = _global_bounds(df)
    frames: list[Image.Image] = []
    for idx, layer in enumerate(layers):
        sub = _pin_bounds(df[df["layer"] == layer], meta, lo, hi)
        frame = render.render_overlay(image, sub, meta, k=k, dim=dim)
        _stamp_badge(frame, int(layer), idx, len(layers))
        frames.append(frame)
    return frames


def save_layer_gif(
    frames: list[Image.Image],
    path: str | Path,
    ms_per_frame: int = 800,
    loop: bool = True,
) -> Path:
    """Write ``frames`` as an animated GIF via PIL ``save_all``.

    Frames are full replacements, so each is quantized to its own adaptive
    256-color palette and written with GIF disposal method 2 (restore to
    background) to avoid inter-frame ghosting. ``loop=True`` loops forever;
    ``loop=False`` plays once. Returns the written path.
    """
    if not frames:
        raise ValueError("no frames to save")
    quant = [f.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256) for f in frames]
    out = Path(path)
    save_kwargs: dict = dict(
        save_all=True,
        append_images=quant[1:],
        duration=int(ms_per_frame),
        disposal=2,
        optimize=False,
    )
    if loop:
        save_kwargs["loop"] = 0  # 0 == loop indefinitely
    quant[0].save(out, format="GIF", **save_kwargs)
    return out


# --------------------------------------------------------------------------- #
# Interactive HTML scrubber
# --------------------------------------------------------------------------- #
def _layer_svg_groups(
    df: pd.DataFrame,
    meta: dict,
    layers: list[int],
    k: int,
    lo: float,
    hi: float,
) -> list[str]:
    """Build one ``<g class="layer">`` per layer, sized against [lo, hi].

    Reuses render's token selection (`_patch_tokens`), placement
    (`_patch_layout`) and normalization (`_norm`) helpers so the per-patch
    geometry is identical to the single-layer SVG — only the normalization
    bounds are pinned to the global range instead of being re-derived.
    """
    n_rows, n_cols = render._grid_dims(meta)
    cell_w = meta["_cell_w"]  # cell geometry stamped in by the caller
    cell_h = meta["_cell_h"]
    groups: list[str] = []
    for li, layer in enumerate(layers):
        sub = df[df["layer"] == layer]
        tokens_by_patch = render._patch_tokens(sub, k)

        # Per-patch tooltip detail (token / score / p_patch) in rank order.
        detail: dict[int, list[tuple[str, float, float]]] = {}
        for patch_idx, group in sub.groupby("patch_idx"):
            g = group.sort_values("rank")
            detail[int(patch_idx)] = [
                (str(r["token"]), float(r["score"]), float(r["p_patch"]))
                for _, r in g.iterrows()
            ]

        parts = [f'<g class="layer" data-layer="{int(layer)}" data-index="{li}">']
        for patch_idx, (toks, top_score) in tokens_by_patch.items():
            row = patch_idx // n_cols
            col = patch_idx % n_cols
            if row >= n_rows or col >= n_cols:
                continue
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            base_norm = render._norm(top_score, lo, hi)
            tip = "\n".join(
                f"{t}  score={s:.3f}  p_patch={p:.3f}" for t, s, p in detail.get(patch_idx, [])
            )
            parts.append("<g>")
            parts.append(f"<title>{escape(tip)}</title>")
            for item in render._patch_layout(toks, base_norm, cell_w, cell_h, min_font=6):
                a = item["alpha"] / 255.0
                y = cy + item["dy"]
                parts.append(
                    f'<text x="{cx:.2f}" y="{y:.2f}" font-size="{item["size"]}" '
                    f'text-anchor="middle" dominant-baseline="central" '
                    f'fill="#ffffff" fill-opacity="{a:.3f}" stroke="#000000" '
                    f'stroke-width="{item["stroke"]}" paint-order="stroke" '
                    f'stroke-opacity="{a:.3f}">{escape(item["token"])}</text>'
                )
            parts.append("</g>")
        parts.append("</g>")
        groups.append("\n".join(parts))
    return groups


def render_layer_scrubber(
    image: Image.Image,
    df: pd.DataFrame,
    meta: dict,
    *,
    k: int = 3,
    dim: float = 0.55,
) -> str:
    """Self-contained HTML page: base image + a slider over per-layer overlays.

    One SVG holds the base image (embedded as a base64 JPEG data URI), a dim
    rect, and a ``<g class="layer">`` per layer. A range slider, layer label,
    and play/pause button (vanilla JS, no external requests) toggle which
    layer group is active; the inactive groups fade out via a ~150ms opacity
    transition. Left/Right arrows step layers. Every top-k patch carries a
    ``<title>`` hover tooltip. Font sizing uses the stack-wide global robust
    range so layers are visually comparable.
    """
    base = image.convert("RGB")
    width, height = base.size
    layers = _layer_list(df, meta)
    lo, hi = _global_bounds(df)

    n_rows, n_cols = render._grid_dims(meta)
    meta = dict(meta)
    meta["_cell_w"] = width / n_cols
    meta["_cell_h"] = height / n_rows

    groups = _layer_svg_groups(df, meta, layers, k, lo, hi)
    dim_alpha = max(0.0, min(1.0, 1.0 - float(dim)))
    img_uri = render._jpeg_data_uri(base)

    svg_parts = [
        f'<svg id="scene" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'font-family="{escape(_SVG_FONT_FAMILY, {chr(34): "&quot;"})}">',
        f'<image x="0" y="0" width="{width}" height="{height}" '
        f'preserveAspectRatio="none" href="{img_uri}" />',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#000000" '
        f'fill-opacity="{dim_alpha:.3f}" />',
        *groups,
        "</svg>",
    ]
    svg = "\n".join(svg_parts)

    layers_json = json.dumps(layers)
    image_id = escape(str(meta.get("image_id", "")))
    n = len(layers)

    # NOTE: '{{' / '}}' are literal braces for the f-string; JS uses them.
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>monocle depth scrubber — {image_id}</title>
<style>
  :root {{ color-scheme: dark; }}
  html, body {{ margin: 0; background: #0c0d10; color: #e8e8ea;
    font-family: system-ui, -apple-system, "DejaVu Sans", sans-serif; }}
  .wrap {{ max-width: min(96vw, 1100px); margin: 0 auto; padding: 20px 16px 40px; }}
  h1 {{ font-size: 15px; font-weight: 600; letter-spacing: .02em;
    color: #b9c0cc; margin: 0 0 14px; }}
  .stage {{ display: flex; justify-content: center; }}
  #scene {{ width: 100%; height: auto; max-width: {width}px;
    border-radius: 8px; box-shadow: 0 6px 30px rgba(0,0,0,.55);
    background: #000; display: block; }}
  #scene .layer {{ opacity: 0; pointer-events: none;
    transition: opacity .15s ease; }}
  #scene .layer.active {{ opacity: 1; pointer-events: auto; }}
  .controls {{ display: flex; align-items: center; gap: 14px;
    max-width: {width}px; margin: 16px auto 0; }}
  button {{ background: #1c2230; color: #e8e8ea; border: 1px solid #333b4d;
    border-radius: 6px; padding: 7px 14px; font-size: 14px; cursor: pointer; }}
  button:hover {{ background: #262e40; }}
  input[type=range] {{ flex: 1; accent-color: #6ea8fe; }}
  .badge {{ font-variant-numeric: tabular-nums; font-weight: 600;
    min-width: 3.4em; text-align: center; padding: 6px 10px;
    background: #11141b; border: 1px solid #2a3140; border-radius: 6px; }}
  .hint {{ color: #6b7280; font-size: 12px; margin: 10px auto 0;
    max-width: {width}px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Jacobian-lens depth scrubber &middot; {image_id or "layered readout"}</h1>
  <div class="stage">
    {svg}
  </div>
  <div class="controls">
    <button id="play" aria-label="play/pause">&#9654; Play</button>
    <input id="slider" type="range" min="0" max="{n - 1}" value="0" step="1"
      aria-label="layer depth" />
    <span class="badge" id="label">L--</span>
  </div>
  <p class="hint">Drag the slider or use &larr;/&rarr; to step through layers.
    Deeper layers are further right. Hover a patch for its top-k tokens.</p>
</div>
<script>
(function() {{
  var layers = {layers_json};
  var groups = Array.prototype.slice.call(
    document.querySelectorAll('#scene .layer'));
  var slider = document.getElementById('slider');
  var label = document.getElementById('label');
  var playBtn = document.getElementById('play');
  var idx = 0, timer = null;

  function pad2(v) {{ return (v < 10 ? '0' : '') + v; }}
  function show(i) {{
    idx = Math.max(0, Math.min(layers.length - 1, i));
    for (var g = 0; g < groups.length; g++) {{
      groups[g].classList.toggle('active', g === idx);
    }}
    slider.value = idx;
    label.textContent = 'L' + pad2(layers[idx]);
  }}
  function stop() {{
    if (timer) {{ clearInterval(timer); timer = null; }}
    playBtn.innerHTML = '&#9654; Play';
  }}
  function play() {{
    if (timer) {{ stop(); return; }}
    playBtn.innerHTML = '&#10073;&#10073; Pause';
    timer = setInterval(function() {{
      var next = idx + 1;
      if (next >= layers.length) {{ next = 0; }}
      show(next);
    }}, 800);
  }}

  slider.addEventListener('input', function() {{ stop(); show(+slider.value); }});
  playBtn.addEventListener('click', play);
  document.addEventListener('keydown', function(e) {{
    if (e.key === 'ArrowLeft') {{ stop(); show(idx - 1); e.preventDefault(); }}
    else if (e.key === 'ArrowRight') {{ stop(); show(idx + 1); e.preventDefault(); }}
  }});

  show(0);
}})();
</script>
</body>
</html>
"""
    return html
