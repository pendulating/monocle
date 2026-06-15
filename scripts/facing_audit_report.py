#!/usr/bin/env python3
"""Visual sanity-check PDF for a facing-filtered cyclomedia parquet.

For each unique unit (library, scaffolding permit, etc.) in the parquet,
render one landscape PDF page containing up to ``--per-unit`` thumbnails
of the rows attributed to that unit, each captioned with the per-row
facing diagnostics (``attribution_confidence``, ``delta_bearing_deg``,
``distance_to_unit_ft``).

Rows within a unit are **sorted by confidence descending** by default so
the highest-confidence attributions surface first — i.e. the images
the downstream weighted sampler is most likely to actually pick.

Example (libraries):

    python scripts/facing_audit_report.py \\
        --parquet /share/pierson/matt/mllmsci/curation/facdb_libraries/cyclomedia_near_libraries_facing.parquet \\
        --out     /share/pierson/matt/mllmsci/machine-beholder/audits/libraries_facing_sample.pdf \\
        --unit-label library --unit-label-plural libraries
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, UnidentifiedImageError

# Reuse the typographic polish from the single-run report (side-effect import
# applies global rcParams).
sys.path.insert(0, str(Path(__file__).parent))
import pairwise_vqa_report  # noqa: F401, E402


REQUIRED_COLUMNS = (
    "image_path",
    "attribution_confidence",
    "delta_bearing_deg",
    "distance_to_unit_ft",
)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_unit(
    sub: pd.DataFrame,
    per_unit: int,
    sort: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Pick up to ``per_unit`` rows from a single unit's rows, ordered by ``sort``."""
    n = len(sub)
    if n <= per_unit:
        picked = sub.copy()
    elif sort == "random":
        idx = rng.choice(n, size=per_unit, replace=False)
        picked = sub.iloc[np.sort(idx)].copy()
    else:
        # For confidence-asc/-desc we want the N rows at the extreme of the
        # distribution — those are the most informative to audit.
        asc = sort == "confidence-asc"
        picked = (
            sub.sort_values("attribution_confidence", ascending=asc, kind="mergesort")
            .head(per_unit)
            .copy()
        )

    # Final display order: confidence-desc (highest first) so the top of the
    # grid matches the rows the weighted sampler is most likely to draw.
    return picked.sort_values(
        "attribution_confidence", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_thumb(path: Path, thumb_size: int) -> Optional[Image.Image]:
    """Open an image, downsample to ``thumb_size`` (square bbox), return RGB."""
    try:
        img = Image.open(path)
        img.load()
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return None
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------


def _render_unit_page(
    *,
    pdf: PdfPages,
    unit_uid: str,
    unit_name: str,
    picked: pd.DataFrame,
    total_rows: int,
    ncols: int,
    thumb_size: int,
    page_idx: int,
    page_total: int,
    unit_label: str,
) -> None:
    n = len(picked)
    nrows = max(1, math.ceil(n / ncols))

    # Landscape US-letter. Leave a band at the top for the header.
    fig = plt.figure(figsize=(14.0, 8.5))
    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.88, bottom=0.03, wspace=0.08, hspace=0.30
    )

    # Header
    mean_c = picked["attribution_confidence"].mean() if n else float("nan")
    min_c = picked["attribution_confidence"].min() if n else float("nan")
    max_c = picked["attribution_confidence"].max() if n else float("nan")
    header = f"{unit_name}"
    subheader = (
        f"unit_uid={unit_uid[:12]}…  ·  showing {n} of {total_rows} rows  ·  "
        f"confidence: min={min_c:.2f}  mean={mean_c:.2f}  max={max_c:.2f}"
    )
    page_stamp = f"{unit_label} {page_idx}/{page_total}"
    fig.text(
        0.02, 0.955, header, fontsize=16, fontweight="semibold", va="top", ha="left"
    )
    fig.text(0.02, 0.922, subheader, fontsize=10, color="#333333", va="top", ha="left")
    fig.text(
        0.98, 0.955, page_stamp, fontsize=9, color="#666666", va="top", ha="right"
    )

    for i in range(nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if i >= n:
            ax.set_visible(False)
            continue

        row = picked.iloc[i]
        img = _load_thumb(Path(row["image_path"]), thumb_size)
        if img is None:
            ax.text(
                0.5,
                0.5,
                "(image\nunavailable)",
                ha="center",
                va="center",
                fontsize=8,
                color="#aa0000",
                transform=ax.transAxes,
            )
            ax.set_facecolor("#f4f4f4")
        else:
            ax.imshow(img)

        conf = float(row["attribution_confidence"])
        dtheta = float(row["delta_bearing_deg"])
        dist = float(row["distance_to_unit_ft"])
        face = str(row.get("face", "?"))
        sid = str(row.get("sample_id", ""))[-16:]
        caption = (
            f"conf={conf:.2f}  Δθ={dtheta:.0f}°  d={dist:.0f}ft  face={face}\n{sid}"
        )
        ax.set_title(caption, fontsize=7, pad=3, color="#222222")

    pdf.savefig(fig, dpi=120)
    plt.close(fig)


def _render_cover(
    pdf: PdfPages,
    *,
    title: str,
    parquet_path: Path,
    n_units: int,
    n_rows: int,
    unit_label_plural: str,
    per_unit: int,
    sort: str,
    confidence_stats: dict,
) -> None:
    fig = plt.figure(figsize=(14.0, 8.5))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)

    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    lines = [
        f"# {title}",
        "",
        f"Parquet: {parquet_path}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        f"{n_units:,} {unit_label_plural}  ·  {n_rows:,} total facing rows",
        f"Up to {per_unit} rows/unit  ·  sampling: {sort}  ·  display order: confidence descending",
        "",
        "Confidence stats (all rows):",
        f"    min = {confidence_stats['min']:.4f}",
        f"   mean = {confidence_stats['mean']:.4f}",
        f"    p50 = {confidence_stats['p50']:.4f}",
        f"    max = {confidence_stats['max']:.4f}",
        "",
        "How to read each page:",
        "  • Header = unit name, unit_uid prefix, row count, per-unit confidence stats.",
        "  • Thumbnails sorted by attribution_confidence DESCENDING — highest",
        "    (most-likely-to-be-sampled) images at the top. These are the rows",
        "    the weighted pair sampler preferentially draws from.",
        "  • Caption: conf · Δθ (ray–bearing angle) · distance (ft) · cube face · sample_id tail.",
    ]

    # Render as monospace left-aligned.
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        family="DejaVu Sans Mono",
        fontsize=11,
        va="top",
        ha="left",
        transform=ax.transAxes,
    )
    pdf.savefig(fig, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--parquet", type=Path, required=True, help="Facing-filtered parquet.")
    p.add_argument("--out", type=Path, required=True, help="Output PDF path.")
    p.add_argument("--unit-col", type=str, default="unit_uid")
    p.add_argument("--unit-name-col", type=str, default="unit_name")
    p.add_argument("--unit-label", type=str, default="unit")
    p.add_argument("--unit-label-plural", type=str, default=None)
    p.add_argument(
        "--per-unit",
        type=int,
        default=20,
        help="Max rows per unit page. Default: 20.",
    )
    p.add_argument(
        "--sort",
        choices=("confidence-desc", "confidence-asc", "random"),
        default="confidence-desc",
        help="Within each unit, which rows to pick when more than --per-unit exist. "
        "Default: confidence-desc (show the highest-confidence rows — what the "
        "weighted pair sampler is most likely to draw).",
    )
    p.add_argument("--ncols", type=int, default=5, help="Thumbnails per row. Default: 5.")
    p.add_argument(
        "--thumb-size",
        type=int,
        default=320,
        help="Pixel bbox for each thumbnail. Default: 320.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="PDF cover title. Default derived from parquet filename.",
    )
    p.add_argument(
        "--limit-units",
        type=int,
        default=None,
        help="Cap the number of units rendered (for smoke testing).",
    )
    p.add_argument(
        "--order",
        choices=("name", "rowcount-desc", "confidence-desc"),
        default="name",
        help="Ordering of unit pages. Default: name (alphabetical).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    unit_label_plural = args.unit_label_plural or f"{args.unit_label}s"

    if not args.parquet.is_file():
        raise SystemExit(f"parquet not found: {args.parquet}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.parquet} …")
    df = pd.read_parquet(args.parquet)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Parquet is missing required columns: {missing}. "
            f"Expected at least {list(REQUIRED_COLUMNS)}."
        )
    if args.unit_col not in df.columns:
        raise SystemExit(f"unit column '{args.unit_col}' not in parquet")
    if args.unit_name_col not in df.columns:
        # Fall back to the uid.
        df[args.unit_name_col] = df[args.unit_col].astype(str)

    # Coerce the diagnostic columns to float just in case.
    for c in (
        "attribution_confidence",
        "delta_bearing_deg",
        "distance_to_unit_ft",
    ):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_rows = len(df)
    conf = df["attribution_confidence"].dropna()
    conf_stats = {
        "min": float(conf.min()) if len(conf) else float("nan"),
        "mean": float(conf.mean()) if len(conf) else float("nan"),
        "p50": float(conf.median()) if len(conf) else float("nan"),
        "max": float(conf.max()) if len(conf) else float("nan"),
    }

    # Build per-unit ordering.
    per_unit_stats = (
        df.groupby(args.unit_col, sort=False)
        .agg(
            unit_name=(args.unit_name_col, "first"),
            n_rows=(args.unit_col, "size"),
            mean_conf=("attribution_confidence", "mean"),
        )
        .reset_index()
    )
    if args.order == "name":
        per_unit_stats = per_unit_stats.sort_values("unit_name", kind="mergesort")
    elif args.order == "rowcount-desc":
        per_unit_stats = per_unit_stats.sort_values(
            ["n_rows", "unit_name"], ascending=[False, True], kind="mergesort"
        )
    elif args.order == "confidence-desc":
        per_unit_stats = per_unit_stats.sort_values(
            ["mean_conf", "unit_name"], ascending=[False, True], kind="mergesort"
        )
    per_unit_stats = per_unit_stats.reset_index(drop=True)

    if args.limit_units is not None:
        per_unit_stats = per_unit_stats.head(args.limit_units).reset_index(drop=True)

    n_units = len(per_unit_stats)
    rng = np.random.default_rng(args.seed)

    title = args.title or f"Facing audit: {args.parquet.stem}"
    print(f"Writing {args.out} ({n_units} {unit_label_plural} × up to {args.per_unit} rows) …")

    t0 = time.time()
    with PdfPages(args.out) as pdf:
        _render_cover(
            pdf,
            title=title,
            parquet_path=args.parquet,
            n_units=n_units,
            n_rows=n_rows,
            unit_label_plural=unit_label_plural,
            per_unit=args.per_unit,
            sort=args.sort,
            confidence_stats=conf_stats,
        )

        # Index units by uid for fast group pulls.
        grouped = dict(list(df.groupby(args.unit_col, sort=False)))

        for page_idx, row in enumerate(per_unit_stats.itertuples(index=False), start=1):
            uid = row.unit_uid if hasattr(row, "unit_uid") else getattr(row, args.unit_col)
            name = row.unit_name
            sub = grouped.get(uid)
            if sub is None:
                continue
            total_rows = len(sub)
            picked = _sample_unit(sub, args.per_unit, args.sort, rng)
            _render_unit_page(
                pdf=pdf,
                unit_uid=str(uid),
                unit_name=str(name),
                picked=picked,
                total_rows=total_rows,
                ncols=args.ncols,
                thumb_size=args.thumb_size,
                page_idx=page_idx,
                page_total=n_units,
                unit_label=args.unit_label,
            )

            if page_idx % 20 == 0 or page_idx == n_units:
                elapsed = time.time() - t0
                rate = page_idx / max(elapsed, 1e-6)
                eta = (n_units - page_idx) / max(rate, 1e-6)
                print(
                    f"  [{page_idx:>4}/{n_units}]  {rate:5.2f} units/s  "
                    f"elapsed={elapsed:6.1f}s  eta={eta:5.1f}s"
                )

    print(f"Done. Wrote {args.out}  ({args.out.stat().st_size / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    main()
