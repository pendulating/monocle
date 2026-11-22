"""Word-cloud overlay rendering for monocle patch predictions.

Consumes the long-format DataFrame from `monocle.scoring` (one row per
patch x rank) plus the sibling `<image_id>.meta.json` and paints, over each
image patch, the top-k tokens that patch would emit — rank 0 largest and
centered, lower ranks smaller and stacked around it. Font size tracks a
*robustly normalized* score (5th..95th-percentile min-max) so a single
runaway patch doesn't flatten the whole image to one size.

Two backends share one geometry/sizing core:
  render_overlay  -> a rasterized PIL image (dimmed base + white stroked text)
  render_svg      -> a self-contained SVG string (embedded JPEG + <text>,
                     per-patch <title> tooltips listing all top-k tokens)

The grid is n_rows x n_cols over the ORIGINAL image; patch (r, c) covers
[c*W/n_cols, r*H/n_rows, (c+1)*W/n_cols, (r+1)*H/n_rows]. See
vault/context/wiki/concept-activation-probe.md for the extraction recipe.
"""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Rank -> relative font scale (rank 0 is the full size). Beyond the table,
# ranks keep shrinking gently. Matches the spec's x0.6 / x0.45 examples.
_RANK_SCALE = (1.0, 0.6, 0.45, 0.38, 0.33)
# Rank -> text opacity (0..255); lower ranks are quieter.
_RANK_ALPHA = (255, 200, 165, 140, 120)

_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_SVG_FONT_FAMILY = "DejaVu Sans, sans-serif"

# ImageFont.truetype is expensive; cache by integer pixel size.
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}
# Scratch drawer for text measurement that does not touch a real canvas.
_scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))


def get_font(size: int) -> ImageFont.ImageFont:
    """Cached DejaVu font at `size` px, falling back to the bitmap default."""
    size = max(1, int(size))
    cached = _font_cache.get(size)
    if cached is not None:
        return cached
    font: ImageFont.ImageFont | None = None
    for path in _FONT_PATHS:
        try:
            font = ImageFont.truetype(path, size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    _font_cache[size] = font  # type: ignore[assignment]
    return font


def load_run(
    parquet_path: str | Path,
    meta_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, dict]:
    """Read a scoring run's parquet and its sibling `<image_id>.meta.json`.

    When `meta_path` is None the meta file is inferred by swapping the
    `.parquet` suffix for `.meta.json` next to it.
    """
    pq = Path(parquet_path)
    df = pd.read_parquet(pq)
    mp = Path(meta_path) if meta_path is not None else pq.with_suffix("").with_suffix(".meta.json")
    if not mp.exists():
        # `.with_suffix` above only strips one extension for names like
        # "foo.parquet"; guard the plain replacement too.
        alt = pq.parent / (pq.stem + ".meta.json")
        mp = alt if alt.exists() else mp
    meta = json.loads(Path(mp).read_text())
    return df, meta


def _measure_width(text: str, font: ImageFont.ImageFont, stroke_width: int) -> float:
    """Rendered width of `text` in px (via a scratch drawer, no canvas)."""
    if not text:
        return 0.0
    box = _scratch.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    return box[2] - box[0]


def _robust_bounds(scores: np.ndarray) -> tuple[float, float]:
    """5th/95th-percentile min-max bounds, clamp-safe against flat inputs."""
    if scores.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(scores, 5))
    hi = float(np.percentile(scores, 95))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _norm(value: float, lo: float, hi: float) -> float:
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _fit_text(
    text: str,
    max_size: float,
    min_font: int,
    cell_w: float,
) -> tuple[str, int, int]:
    """Shrink font until `text` fits `cell_w`; clip the string as a last resort.

    Returns (possibly-clipped text, chosen font size, stroke width).
    """
    avail = cell_w * 0.94
    size = max(min_font, int(round(max_size)))
    while size > min_font:
        stroke = 2 if size >= 16 else 1
        if _measure_width(text, get_font(size), stroke) <= avail:
            return text, size, stroke
        size -= 1
    # Floored at min_font: clip characters off the end until it fits.
    stroke = 2 if size >= 16 else 1
    font = get_font(size)
    while text and _measure_width(text, font, stroke) > avail:
        text = text[:-1]
    return text, size, stroke


def _patch_layout(
    tokens: list[str],
    base_norm: float,
    cell_w: float,
    cell_h: float,
    min_font: int,
) -> list[dict]:
    """Place a patch's top-k tokens: rank 0 centered, rest stacked around it.

    Returns dicts with token, size, stroke, dy (offset from cell center y),
    rank, alpha — x is always the cell center.
    """
    max0 = cell_h * 0.45
    base_size = min_font + base_norm * max(0.0, max0 - min_font)
    gap = max(1.0, cell_h * 0.03)

    placed: list[dict] = []
    top_edge = 0.0
    bot_edge = 0.0
    above_next = True
    for rank, token in enumerate(tokens):
        scale = _RANK_SCALE[rank] if rank < len(_RANK_SCALE) else _RANK_SCALE[-1] * 0.85
        target = max(float(min_font), base_size * scale)
        fitted, size, stroke = _fit_text(token, target, min_font, cell_w)
        if not fitted:
            continue
        half = size / 2.0
        if rank == 0:
            dy = 0.0
            top_edge = -half
            bot_edge = half
        elif above_next:
            dy = top_edge - gap - half
            top_edge = dy - half
            above_next = False
        else:
            dy = bot_edge + gap + half
            bot_edge = dy + half
            above_next = True
        placed.append({
            "token": fitted,
            "size": size,
            "stroke": stroke,
            "dy": dy,
            "rank": rank,
            "alpha": _RANK_ALPHA[rank] if rank < len(_RANK_ALPHA) else _RANK_ALPHA[-1],
        })
    return placed


def _grid_dims(meta: dict) -> tuple[int, int]:
    n_rows = int(meta["n_rows"])
    n_cols = int(meta["n_cols"])
    return n_rows, n_cols


def _select_image_rows(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Restrict to this image's rows when the frame carries an image_id."""
    if "image_id" in df.columns and meta.get("image_id") is not None:
        sub = df[df["image_id"] == meta["image_id"]]
        if not sub.empty:
            return sub
    return df


def _patch_tokens(df: pd.DataFrame, k: int) -> dict[int, tuple[list[str], float]]:
    """patch_idx -> (top-k display tokens in rank order, rank-0 score)."""
    out: dict[int, tuple[list[str], float]] = {}
    for patch_idx, group in df.groupby("patch_idx"):
        g = group.sort_values("rank")
        g = g[g["rank"] < k]
        if g.empty:
            continue
        tokens = [str(t) for t in g["token"].tolist()]
        top_score = float(g.iloc[0]["score"])
        out[int(patch_idx)] = (tokens, top_score)
    return out


def _auto_upscale(cell_w: float, cell_h: float, upscale: Optional[float]) -> float:
    """Resolve the base-image upscale factor.

    Explicit `upscale` wins. Otherwise, if cells are smaller than ~64 px,
    pick a factor that lifts the smaller cell dimension to >= 96 px, capped
    at 4x; cells already big enough are left untouched.
    """
    if upscale is not None:
        return max(1.0, float(upscale))
    cell = min(cell_w, cell_h)
    if cell >= 64.0:
        return 1.0
    return float(min(4.0, math.ceil(96.0 / max(cell, 1e-6) * 100) / 100))


def render_overlay(
    image: Image.Image,
    df: pd.DataFrame,
    meta: dict,
    *,
    k: int = 3,
    dim: float = 0.55,
    min_font: int = 9,
    upscale: Optional[float] = None,
    grid_lines: bool = True,
) -> Image.Image:
    """Rasterize the per-patch word-cloud overlay onto a dimmed copy of `image`.

    The base is darkened by `dim` (pixel multiply) so white text reads over
    bright regions. Font sizes follow a robust min-max normalization of
    `score` across all patches; small images are auto-upscaled so cells stay
    legible (see `upscale`). Returns an RGB image.
    """
    df = _select_image_rows(df, meta)
    n_rows, n_cols = _grid_dims(meta)

    base = image.convert("RGB")
    cell_w0 = base.width / n_cols
    cell_h0 = base.height / n_rows
    factor = _auto_upscale(cell_w0, cell_h0, upscale)
    if factor > 1.0:
        base = base.resize(
            (max(1, round(base.width * factor)), max(1, round(base.height * factor))),
            Image.LANCZOS,
        )

    width, height = base.size
    cell_w = width / n_cols
    cell_h = height / n_rows

    # Dim the base by pixel multiply.
    arr = (np.asarray(base, dtype=np.float32) * float(dim)).clip(0, 255).astype(np.uint8)
    canvas = Image.fromarray(arr, "RGB").convert("RGBA")

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if grid_lines:
        line = (255, 255, 255, 40)
        for c in range(1, n_cols):
            x = round(c * cell_w)
            draw.line([(x, 0), (x, height)], fill=line, width=1)
        for r in range(1, n_rows):
            y = round(r * cell_h)
            draw.line([(0, y), (width, y)], fill=line, width=1)

    tokens_by_patch = _patch_tokens(df, k)
    scores = df["score"].to_numpy(dtype=np.float32)
    lo, hi = _robust_bounds(scores)

    for patch_idx, (tokens, top_score) in tokens_by_patch.items():
        row = patch_idx // n_cols
        col = patch_idx % n_cols
        if row >= n_rows or col >= n_cols:
            continue
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h
        base_norm = _norm(top_score, lo, hi)
        for item in _patch_layout(tokens, base_norm, cell_w, cell_h, min_font):
            a = item["alpha"]
            draw.text(
                (cx, cy + item["dy"]),
                item["token"],
                font=get_font(item["size"]),
                fill=(255, 255, 255, a),
                anchor="mm",
                stroke_width=item["stroke"],
                stroke_fill=(0, 0, 0, a),
            )

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def _jpeg_data_uri(image: Image.Image, quality: int = 88) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def render_svg(
    image: Image.Image,
    df: pd.DataFrame,
    meta: dict,
    *,
    k: int = 3,
    dim: float = 0.55,
) -> str:
    """Self-contained SVG overlay: embedded JPEG, dim rect, per-patch text.

    Each patch's tokens are wrapped in a `<g>` carrying a `<title>` that
    lists every top-k token with its score and p_patch (a hover tooltip).
    Font sizing mirrors the raster path; text is rendered as vectors, so no
    upscaling is needed. All dynamic text is XML-escaped.
    """
    df = _select_image_rows(df, meta)
    n_rows, n_cols = _grid_dims(meta)

    base = image.convert("RGB")
    width, height = base.size
    cell_w = width / n_cols
    cell_h = height / n_rows

    tokens_by_patch = _patch_tokens(df, k)
    scores = df["score"].to_numpy(dtype=np.float32)
    lo, hi = _robust_bounds(scores)

    # Per-patch detail for tooltips: token/score/p_patch in rank order.
    detail: dict[int, list[tuple[str, float, float]]] = {}
    for patch_idx, group in df.groupby("patch_idx"):
        g = group.sort_values("rank")
        detail[int(patch_idx)] = [
            (str(r["token"]), float(r["score"]), float(r["p_patch"]))
            for _, r in g.iterrows()
        ]

    dim_alpha = max(0.0, min(1.0, 1.0 - float(dim)))
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{escape(_SVG_FONT_FAMILY, {chr(34): "&quot;"})}">',
        f'<image x="0" y="0" width="{width}" height="{height}" '
        f'preserveAspectRatio="none" href="{_jpeg_data_uri(base)}" />',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#000000" '
        f'fill-opacity="{dim_alpha:.3f}" />',
    ]

    for patch_idx, (tokens, top_score) in tokens_by_patch.items():
        row = patch_idx // n_cols
        col = patch_idx % n_cols
        if row >= n_rows or col >= n_cols:
            continue
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h
        base_norm = _norm(top_score, lo, hi)

        tip_lines = [
            f"{tok}  score={sc:.3f}  p_patch={pp:.3f}"
            for tok, sc, pp in detail.get(patch_idx, [])
        ]
        parts.append("<g>")
        parts.append(f"<title>{escape(chr(10).join(tip_lines))}</title>")
        for item in _patch_layout(tokens, base_norm, cell_w, cell_h, min_font=6):
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

    parts.append("</svg>")
    return "\n".join(parts)
