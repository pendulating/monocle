#!/usr/bin/env python3
"""Image-distribution preview PDF for a facing-filtered cyclomedia parquet.

Where ``facing_audit_report.py`` renders one thumbnail page *per unit* (a deep
per-unit audit), this report gives a *bird's-eye* view of how the images in a
curated dataset are distributed — counts, per-category breakdowns, facing-
diagnostic histograms, a geographic point map, and a small stratified montage
so you can eyeball what the imagery actually looks like.

It works for any curation family's ``*_facing.parquet`` (open-restaurants,
libraries, scaffolding, …). Pass ``--units-parquet`` to unlock the per-borough
and per-``license_type``/``factype`` breakdowns (joined on ``unit_uid`` → the
units parquet's ``uid``) plus an accurate coverage ratio (units with imagery /
total units in the curation).

Pages:
  1. Cover — headline counts, coverage ratio, facing-diagnostic summary stats.
  2. Distributions — images-per-unit hist, per-borough / per-category / per-face
     / per-dataset bars, confidence / distance / Δbearing histograms.
  3. Geographic map — every image's recording point, coloured by borough.
  4. Montage — a stratified random sample of actual images.

Example (open restaurants):

    python scripts/image_distribution_report.py \\
        --parquet curation/open_restaurants_all/cyclomedia_near_open_restaurants_facing.parquet \\
        --units-parquet curation/open_restaurants_all/open_restaurants.parquet \\
        --category-col license_type \\
        --unit-label restaurant \\
        --title "Open Restaurants · image distribution"
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

# Reuse the typographic polish (rcParams applied on import) + accent colours
# from the single-run report, and the robust thumbnail loader from the
# per-unit audit. Both live next to this file.
sys.path.insert(0, str(Path(__file__).parent))
from pairwise_vqa_report import ACCENT, ACCENT_WARM  # noqa: E402
from facing_audit_report import _load_thumb  # noqa: E402

# Stable, colour-blind-friendly palette for the five NYC boroughs (+ fallback).
BOROUGH_COLORS = {
    "MANHATTAN": "#4c78a8",
    "BROOKLYN": "#f58518",
    "QUEENS": "#54a24b",
    "BRONX": "#e45756",
    "STATEN ISLAND": "#b279a2",
}
_FALLBACK_COLORS = ["#9d755d", "#bab0ac", "#ff9da6", "#79706e", "#d3b484"]

# Facing-diagnostic columns we summarise/plot when present.
FACING_COLS = ("attribution_confidence", "delta_bearing_deg", "distance_to_unit_ft")


# ---------------------------------------------------------------------------
# Data loading / joining
# ---------------------------------------------------------------------------


def _load(
    parquet: Path,
    units_parquet: Optional[Path],
    unit_col: str,
    category_col: Optional[str],
) -> tuple[pd.DataFrame, Optional[int], list[str]]:
    """Read the facing parquet; optionally left-join borough/category from units.

    Returns ``(df, total_units, notes)`` where ``total_units`` is the row count
    of the units parquet (None if not supplied) and ``notes`` collects any
    soft warnings to surface on the cover page.
    """
    notes: list[str] = []
    df = pd.read_parquet(parquet)
    for c in FACING_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    total_units: Optional[int] = None
    if units_parquet is not None:
        units = pd.read_parquet(units_parquet)
        total_units = len(units)
        key = "uid" if "uid" in units.columns else unit_col
        keep = [key]
        for c in ("borough", category_col):
            if c and c in units.columns and c not in keep:
                keep.append(c)
        umap = units[keep].drop_duplicates(subset=[key]).rename(columns={key: unit_col})
        # Avoid clobbering columns that already exist on the facing frame.
        umap = umap[[unit_col] + [c for c in umap.columns
                                  if c != unit_col and c not in df.columns]]
        df = df.merge(umap, on=unit_col, how="left")

    if "borough" not in df.columns:
        notes.append("no 'borough' column (pass --units-parquet for borough breakdown)")
    if category_col and category_col not in df.columns:
        notes.append(f"no '{category_col}' column for the category breakdown")
    return df, total_units, notes


def _series_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {k: float("nan") for k in ("min", "mean", "p50", "p95", "max")}
    return {
        "min": float(s.min()), "mean": float(s.mean()), "p50": float(s.median()),
        "p95": float(s.quantile(0.95)), "max": float(s.max()),
    }


# ---------------------------------------------------------------------------
# Small plotting helpers
# ---------------------------------------------------------------------------


def _bar(ax, counts: pd.Series, *, title: str, color=ACCENT, rotate: int = 0,
         color_map: Optional[dict] = None) -> None:
    if counts.empty:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, color="#999999", fontsize=10)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    labels = [str(x) for x in counts.index]
    colors = ([color_map.get(str(l).upper(), color) for l in labels]
              if color_map else color)
    ax.bar(range(len(counts)), counts.values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("images")
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n{100*v/total:.0f}%", ha="center", va="bottom", fontsize=7,
                color="#333333")
    ax.margins(y=0.18)


def _hist(ax, s: pd.Series, *, title: str, xlabel: str, color=ACCENT,
          bins: int = 40, logy: bool = False) -> None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, color="#999999", fontsize=10)
        ax.set_title(title)
        return
    ax.hist(s.values, bins=bins, color=color, edgecolor="white", linewidth=0.4, alpha=0.9)
    med = float(s.median())
    ax.axvline(med, color=ACCENT_WARM, linestyle="--", linewidth=1.4)
    ax.text(med, ax.get_ylim()[1] * 0.92, f" median {med:.1f}", color=ACCENT_WARM,
            fontsize=8, ha="left", va="top")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("images" + (" (log)" if logy else ""))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _render_cover(pdf: PdfPages, *, title: str, parquet: Path, df: pd.DataFrame,
                  unit_col: str, category_col: Optional[str], total_units: Optional[int],
                  unit_label_plural: str, notes: list[str]) -> None:
    fig = plt.figure(figsize=(14.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    n_images = len(df)
    n_units = df[unit_col].nunique()
    per_unit = df.groupby(unit_col).size()

    lines = [
        f"# {title}", "",
        f"Parquet:   {parquet}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", "",
        f"{n_images:,} facing images  ·  {n_units:,} {unit_label_plural} with imagery",
    ]
    if total_units:
        cov = 100.0 * n_units / total_units
        lines.append(f"Coverage:  {n_units:,} / {total_units:,} curated {unit_label_plural} "
                     f"have ≥1 facing image  ({cov:.1f}%)")
    lines += [
        f"Images per {unit_label_plural[:-1] if unit_label_plural.endswith('s') else unit_label_plural}: "
        f"mean {per_unit.mean():.1f}  ·  p50 {per_unit.median():.0f}  ·  "
        f"p95 {per_unit.quantile(0.95):.0f}  ·  max {per_unit.max():,}",
        "",
    ]

    def _counts_block(header: str, col: str) -> None:
        if col not in df.columns:
            return
        vc = df[col].fillna("(none)").astype(str).value_counts()
        lines.append(f"{header}:")
        for k, v in vc.head(8).items():
            lines.append(f"    {k:<22} {v:>8,}  ({100*v/n_images:4.1f}%)")
        lines.append("")

    _counts_block("By borough", "borough")
    if category_col:
        _counts_block(f"By {category_col}", category_col)
    _counts_block("By catalog dataset", "dataset")
    _counts_block("By cube face", "face")

    present = [c for c in FACING_COLS if c in df.columns]
    if present:
        lines.append("Facing diagnostics (min / mean / p50 / p95 / max):")
        for c in present:
            st = _series_stats(df[c])
            lines.append(f"    {c:<24} {st['min']:8.2f} {st['mean']:8.2f} "
                         f"{st['p50']:8.2f} {st['p95']:8.2f} {st['max']:8.2f}")
        lines.append("")
    if notes:
        lines.append("Notes:")
        lines += [f"    • {n}" for n in notes]

    ax.text(0.0, 1.0, "\n".join(lines), family="DejaVu Sans Mono", fontsize=10.5,
            va="top", ha="left", transform=ax.transAxes)
    pdf.savefig(fig, dpi=120)
    plt.close(fig)


def _render_distributions(pdf: PdfPages, *, df: pd.DataFrame, unit_col: str,
                          category_col: Optional[str], unit_label_plural: str) -> None:
    fig = plt.figure(figsize=(14.0, 8.5))
    fig.suptitle("Distributions", fontsize=15, fontweight="semibold", x=0.02, ha="left")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.08, wspace=0.28, hspace=0.45)

    gs = fig.add_gridspec(2, 3)

    # 1) images per unit
    per_unit = df.groupby(unit_col).size()
    ax = fig.add_subplot(gs[0, 0])
    _hist(ax, per_unit, title=f"Images per {unit_label_plural}",
          xlabel="images attributed to a unit", logy=True, bins=40)

    # 2) per borough (or dataset fallback)
    ax = fig.add_subplot(gs[0, 1])
    if "borough" in df.columns:
        _bar(ax, df["borough"].fillna("(none)").astype(str).value_counts(),
             title="Images by borough", rotate=30, color_map=BOROUGH_COLORS)
    elif "dataset" in df.columns:
        _bar(ax, df["dataset"].astype(str).value_counts(), title="Images by catalog dataset",
             rotate=30)
    else:
        _bar(ax, pd.Series(dtype=int), title="Images by borough")

    # 3) per category (license_type / factype / …)
    ax = fig.add_subplot(gs[0, 2])
    if category_col and category_col in df.columns:
        _bar(ax, df[category_col].fillna("(none)").astype(str).value_counts(),
             title=f"Images by {category_col}", rotate=20, color=ACCENT_WARM)
    elif "face" in df.columns:
        _bar(ax, df["face"].astype(str).value_counts().sort_index(),
             title="Images by cube face", color=ACCENT_WARM)
    else:
        _bar(ax, pd.Series(dtype=int), title=f"Images by {category_col or 'category'}")

    # 4-6) facing-diagnostic histograms (or a face bar if diagnostics absent)
    panels = [
        ("attribution_confidence", "attribution_confidence", ACCENT, False),
        ("distance_to_unit_ft", "distance to unit centroid (ft)", ACCENT, False),
        ("delta_bearing_deg", "Δ bearing (face ray vs unit) [deg]", ACCENT, False),
    ]
    for j, (col, xlabel, color, logy) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j])
        if col in df.columns:
            _hist(ax, df[col], title=col, xlabel=xlabel, color=color, logy=logy)
        elif col == "attribution_confidence" and "face" in df.columns:
            _bar(ax, df["face"].astype(str).value_counts().sort_index(),
                 title="Images by cube face", color=color)
        else:
            _hist(ax, pd.Series(dtype=float), title=col, xlabel=xlabel)

    pdf.savefig(fig, dpi=120)
    plt.close(fig)


def _render_map(pdf: PdfPages, *, df: pd.DataFrame, unit_label_plural: str) -> None:
    if not {"latitude", "longitude"} <= set(df.columns):
        return
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    ok = lat.notna() & lon.notna()
    if ok.sum() == 0:
        return
    lat, lon = lat[ok], lon[ok]

    fig = plt.figure(figsize=(14.0, 8.5))
    fig.suptitle("Geographic distribution of recording points", fontsize=15,
                 fontweight="semibold", x=0.02, ha="left")
    ax = fig.add_subplot(1, 1, 1)

    boro = df.loc[ok, "borough"].astype(str).str.upper() if "borough" in df.columns else None
    if boro is not None:
        from matplotlib.lines import Line2D
        extra = iter(_FALLBACK_COLORS)
        handles: list = []
        for name, grp in boro.groupby(boro):
            color = BOROUGH_COLORS.get(name, next(extra, "#888888"))
            ax.scatter(lon[grp.index], lat[grp.index], s=5, c=color, alpha=0.35,
                       linewidths=0)
            handles.append(Line2D([], [], marker="o", linestyle="", markersize=7,
                                  markerfacecolor=color, markeredgecolor="none",
                                  label=f"{name.title()} ({len(grp):,})"))
        ax.legend(handles=handles, loc="upper left", fontsize=9, title="borough")
    else:
        ax.scatter(lon, lat, s=5, c=ACCENT, alpha=0.35, linewidths=0)

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    # Correct the aspect so NYC isn't horizontally squashed.
    ax.set_aspect(1.0 / math.cos(math.radians(float(lat.mean()))))
    ax.set_title(f"{len(lat):,} facing images", fontsize=11)
    ax.grid(True, linewidth=0.3, color="#dddddd")
    pdf.savefig(fig, dpi=140)
    plt.close(fig)


def _render_montage(pdf: PdfPages, *, df: pd.DataFrame, n: int, ncols: int,
                    thumb_size: int, seed: int, stratify_col: Optional[str],
                    name_col: str) -> None:
    if "image_path" not in df.columns or n <= 0:
        return
    rng = np.random.default_rng(seed)

    # Stratified-ish sample: round-robin across strata so the montage spans
    # boroughs/datasets rather than over-representing the densest one.
    if stratify_col and stratify_col in df.columns:
        groups = [g.sample(frac=1.0, random_state=int(rng.integers(1e9)))
                  for _, g in df.groupby(stratify_col)]
        picked_idx: list = []
        gi = 0
        while len(picked_idx) < min(n, len(df)) and groups:
            g = groups[gi % len(groups)]
            take = g.index[(gi // len(groups))] if (gi // len(groups)) < len(g) else None
            if take is not None:
                picked_idx.append(take)
            gi += 1
            if gi > len(df) * 2:
                break
        sample = df.loc[picked_idx]
    else:
        sample = df.sample(n=min(n, len(df)), random_state=seed)

    sample = sample.head(n)
    nrows = max(1, math.ceil(len(sample) / ncols))
    per_page = ncols * 4  # 4 rows of thumbnails per landscape page
    pages = max(1, math.ceil(len(sample) / per_page))

    for p in range(pages):
        chunk = sample.iloc[p * per_page:(p + 1) * per_page]
        rows_here = max(1, math.ceil(len(chunk) / ncols))
        fig = plt.figure(figsize=(14.0, 8.5))
        fig.suptitle(f"Image preview montage ({p+1}/{pages})", fontsize=15,
                     fontweight="semibold", x=0.02, ha="left")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03,
                            wspace=0.06, hspace=0.30)
        for i in range(rows_here * ncols):
            ax = fig.add_subplot(rows_here, ncols, i + 1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i >= len(chunk):
                ax.set_visible(False)
                continue
            row = chunk.iloc[i]
            img = _load_thumb(Path(row["image_path"]), thumb_size)
            if img is None:
                ax.text(0.5, 0.5, "(image\nunavailable)", ha="center", va="center",
                        fontsize=8, color="#aa0000", transform=ax.transAxes)
                ax.set_facecolor("#f4f4f4")
            else:
                ax.imshow(img)
            name = str(row.get(name_col, ""))[:26]
            face = str(row.get("face", "?"))
            boro = str(row.get("borough", "")).title()
            cap = f"{name}\n{boro} · face {face}"
            ax.set_title(cap, fontsize=7, pad=3, color="#222222")
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
    p.add_argument("--parquet", type=Path, required=True,
                   help="Facing-filtered cyclomedia parquet.")
    p.add_argument("--units-parquet", type=Path, default=None,
                   help="Curation units parquet (e.g. open_restaurants.parquet). "
                        "Joined on unit_uid → uid to add borough / category + coverage ratio.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PDF. Default: <parquet_dir>/<stem>_distribution.pdf.")
    p.add_argument("--unit-col", type=str, default="unit_uid")
    p.add_argument("--unit-name-col", type=str, default="unit_name")
    p.add_argument("--unit-label", type=str, default="unit")
    p.add_argument("--unit-label-plural", type=str, default=None)
    p.add_argument("--category-col", type=str, default=None,
                   help="Units-parquet column to break images down by "
                        "(e.g. license_type, factype). Optional.")
    p.add_argument("--montage", type=int, default=40,
                   help="Number of sample images in the preview montage. 0 to skip. Default: 40.")
    p.add_argument("--ncols", type=int, default=5, help="Montage thumbnails per row. Default: 5.")
    p.add_argument("--thumb-size", type=int, default=300, help="Montage thumbnail bbox px. Default: 300.")
    p.add_argument("--stratify-col", type=str, default="dataset",
                   help="Column to round-robin the montage sample across. Default: dataset.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--title", type=str, default=None,
                   help="Cover title. Default derived from the parquet filename.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    unit_label_plural = args.unit_label_plural or f"{args.unit_label}s"

    if not args.parquet.is_file():
        raise SystemExit(f"parquet not found: {args.parquet}")
    if args.units_parquet is not None and not args.units_parquet.is_file():
        raise SystemExit(f"units parquet not found: {args.units_parquet}")
    out = args.out or args.parquet.with_name(f"{args.parquet.stem}_distribution.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.parquet} …")
    df, total_units, notes = _load(args.parquet, args.units_parquet,
                                   args.unit_col, args.category_col)
    if args.unit_col not in df.columns:
        raise SystemExit(f"unit column '{args.unit_col}' not in parquet")
    if args.unit_name_col not in df.columns:
        df[args.unit_name_col] = df[args.unit_col].astype(str)

    title = args.title or f"Image distribution: {args.parquet.stem}"
    print(f"Writing {out} ({len(df):,} images, {df[args.unit_col].nunique():,} "
          f"{unit_label_plural}) …")

    t0 = time.time()
    with PdfPages(out) as pdf:
        _render_cover(pdf, title=title, parquet=args.parquet, df=df,
                      unit_col=args.unit_col, category_col=args.category_col,
                      total_units=total_units, unit_label_plural=unit_label_plural,
                      notes=notes)
        _render_distributions(pdf, df=df, unit_col=args.unit_col,
                              category_col=args.category_col,
                              unit_label_plural=unit_label_plural)
        _render_map(pdf, df=df, unit_label_plural=unit_label_plural)
        _render_montage(pdf, df=df, n=args.montage, ncols=args.ncols,
                        thumb_size=args.thumb_size, seed=args.seed,
                        stratify_col=args.stratify_col, name_col=args.unit_name_col)

    print(f"Done. Wrote {out}  ({out.stat().st_size / (1024*1024):.1f} MB, "
          f"{time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
