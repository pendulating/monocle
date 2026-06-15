#!/usr/bin/env python3
"""Generate a markdown summary report for a urbanpairvqa run.

Reads a pairwise VQA stage output parquet plus its companion ``pairs.parquet``
(for unit identity) and emits a self-contained markdown report with embedded
images:

  * Ordinal label distribution (counts, proportions, entropy, position bias)
  * Reasoning-trace length statistics + histogram
  * Word cloud over the captured reasoning traces
  * TrueSkill ratings per unit, sorted from "most" to "least" of the attribute,
    with a sorted error-bar chart and top/bottom-N tables
  * Stitched Cyclomedia pair images for the top-N / bottom-N units plus a
    random inspection sample
  * Optional PDF export (pandoc + xelatex) with all images baked in

Designed to work for any pairwise VQA run (libraries, PUMAs, tracts, ...);
pass ``--attribute`` for nicer headings.

Example:

    python scripts/pairwise_vqa_report.py \\
        multirun/2026-04-21_URBANPAIRVQA/21-31-18/0/outputs/pairwise/libraries_mvp_20260421_213129.parquet \\
        --attribute maintained --title "Libraries MVP" --pdf
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trueskill
import yaml
from PIL import Image
from wordcloud import STOPWORDS, WordCloud

# Global matplotlib polish. Applied once at import so every plot inherits it.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "axes.titlecolor": "#111111",
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.titlepad": 12,
    "axes.labelsize": 11,
    "axes.labelpad": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "text.color": "#111111",
    "font.family": "DejaVu Sans",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
})

ORDINAL_ORDER = ["MuchLess", "Less", "Same", "More", "MuchMore"]
ORDINAL_COLORS = ["#762a83", "#af8dc3", "#d9d9d9", "#7fbf7b", "#1b7837"]
ACCENT = "#2b6cb0"  # primary accent for single-series plots
ACCENT_WARM = "#dd8452"  # secondary accent

# Thinking-mode filler + task-specific words that would dominate the cloud
# without saying much. Extend via --extra-stopwords.
DEFAULT_DOMAIN_STOPWORDS = {
    "image", "images", "imagea", "imageb", "photo", "photograph", "photographs",
    "building", "buildings", "facade", "facades",
    "left", "right", "first", "second",
    "much", "less", "more", "same", "muchless", "muchmore",
    "maintained", "maintenance", "condition",
    "wait", "hmm", "okay", "ok", "well", "let", "lets", "like", "ll",
    "actually", "maybe", "probably", "possibly", "seems", "looks", "looking",
    "note", "noted", "also", "think", "thinking", "compare", "comparison",
    "attribute", "label", "answer",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "output_parquet",
        type=Path,
        help="Stage output parquet (with pair_id, relative_label, relative_score, model_reasoning).",
    )
    p.add_argument(
        "--pairs-parquet",
        type=Path,
        default=None,
        help="Companion pairs.parquet with unit identity. Default: sibling 'pairs.parquet'.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown path. Default: <output_parquet_dir>/<stem>.report.md.",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Report title. Default derived from parquet file name.",
    )
    p.add_argument(
        "--attribute",
        type=str,
        default="more of the attribute",
        help='Attribute phrase for the ranking (e.g., "maintained"). Default: "more of the attribute".',
    )
    p.add_argument(
        "--unit-label",
        type=str,
        default="unit",
        help='Singular noun for the ranked entity in headings (e.g., "library"). Default: "unit".',
    )
    p.add_argument(
        "--unit-label-plural",
        type=str,
        default=None,
        help='Plural noun for the ranked entity (e.g., "libraries"). Default: unit-label + "s".',
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top/bottom units to show in the rankings tables. Default: 20.",
    )
    p.add_argument(
        "--draw-prob",
        type=float,
        default=0.05,
        help="TrueSkill draw probability. Default: 0.05 (matches wealth.ipynb).",
    )
    p.add_argument(
        "--min-comparisons",
        type=int,
        default=1,
        help="Minimum comparisons for a unit to appear in top/bottom tables. Default: 1.",
    )
    p.add_argument(
        "--extra-stopwords",
        type=str,
        default="",
        help="Comma-separated extra stopwords for the word cloud.",
    )
    p.add_argument(
        "--top-bottom-image-n",
        type=int,
        default=5,
        help="Number of top + bottom units to embed example pair images for. Default: 5.",
    )
    p.add_argument(
        "--random-image-n",
        type=int,
        default=10,
        help="Number of random pairs to embed for spot inspection. Default: 10.",
    )
    p.add_argument(
        "--image-scale",
        type=float,
        default=0.5,
        help="Uniform scale factor for embedded pair images (source is 1024x1024). Default: 0.5.",
    )
    p.add_argument(
        "--image-quality",
        type=int,
        default=85,
        help="JPEG quality for embedded pair images (1-95). Default: 85.",
    )
    p.add_argument(
        "--image-seed",
        type=int,
        default=1234,
        help="RNG seed for example/random pair selection. Default: 1234.",
    )
    p.add_argument(
        "--pdf",
        action="store_true",
        help="Also render the markdown to PDF via pandoc (requires pandoc + a PDF engine).",
    )
    p.add_argument(
        "--pdf-engine",
        type=str,
        default="xelatex",
        help="pandoc --pdf-engine value. Default: xelatex.",
    )
    p.add_argument(
        "--pdf-orientation",
        choices=("landscape", "portrait"),
        default="landscape",
        help="PDF page orientation. Default: landscape.",
    )
    p.add_argument(
        "--hydra-config",
        type=Path,
        default=None,
        help=(
            "Path to the run's resolved Hydra config.yaml (for prompt capture). "
            "Default: walk up from the output parquet to find .hydra/config.yaml."
        ),
    )
    # ---- Zone-geometry aggregation ----
    # For runs with no unit identity in pairs.parquet (e.g. image-mode
    # sterility), spatially join each image's point to a containing polygon
    # and rate the *zone* via TrueSkill instead. See the wiki's
    # concept-trueskill "Geographic area (PUMA, tract)" recipe.
    p.add_argument(
        "--zone-geojson",
        type=Path,
        default=None,
        help="Polygon file (GeoJSON/any geopandas-readable). Enables zone aggregation.",
    )
    p.add_argument(
        "--zone-id-column",
        type=str,
        default=None,
        help="Zone property used as the rated unit id (e.g. 'geoid'). Required with --zone-geojson.",
    )
    p.add_argument(
        "--zone-name-column",
        type=str,
        default=None,
        help=(
            "Zone property/properties used as the display name. Comma-separated "
            "values are joined with ' · '. Default: the id column."
        ),
    )
    p.add_argument(
        "--lat-col-a",
        type=str,
        default="latitude_a",
        help="pairs.parquet latitude column for side A. Default: latitude_a.",
    )
    p.add_argument(
        "--lon-col-a",
        type=str,
        default="longitude_a",
        help="pairs.parquet longitude column for side A. Default: longitude_a.",
    )
    p.add_argument(
        "--lat-col-b",
        type=str,
        default="latitude_b",
        help="pairs.parquet latitude column for side B. Default: latitude_b.",
    )
    p.add_argument(
        "--lon-col-b",
        type=str,
        default="longitude_b",
        help="pairs.parquet longitude column for side B. Default: longitude_b.",
    )
    p.add_argument(
        "--point-crs",
        type=str,
        default="EPSG:4326",
        help="CRS of the lat/lon point columns. Default: EPSG:4326.",
    )
    # When pairs.parquet has no usable lat/lon (e.g. metadata columns never
    # populated), resolve coordinates from a separate lookup keyed by a
    # per-side id present in pairs.parquet (default sample_id_a/sample_id_b).
    p.add_argument(
        "--coords-parquet",
        type=Path,
        default=None,
        help="Optional lookup parquet mapping an id to lat/lon (when pairs.parquet lacks coords).",
    )
    p.add_argument(
        "--coords-id-column",
        type=str,
        default="sample_id",
        help="Id column in --coords-parquet. Default: sample_id.",
    )
    p.add_argument(
        "--coords-lat-column",
        type=str,
        default="latitude",
        help="Latitude column in --coords-parquet. Default: latitude.",
    )
    p.add_argument(
        "--coords-lon-column",
        type=str,
        default="longitude",
        help="Longitude column in --coords-parquet. Default: longitude.",
    )
    p.add_argument(
        "--id-col-a",
        type=str,
        default="sample_id_a",
        help="pairs.parquet column with side-A id for the coords lookup. Default: sample_id_a.",
    )
    p.add_argument(
        "--id-col-b",
        type=str,
        default="sample_id_b",
        help="pairs.parquet column with side-B id for the coords lookup. Default: sample_id_b.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_merged(output_pq: Path, pairs_pq: Optional[Path]) -> tuple[pd.DataFrame, bool]:
    """Merge stage output with the pairs manifest. Returns (df, has_units)."""
    out = pd.read_parquet(output_pq)
    required = {"pair_id", "relative_label", "relative_score", "model_reasoning"}
    missing = required - set(out.columns)
    if missing:
        raise SystemExit(
            f"Output parquet {output_pq} missing required columns: {sorted(missing)}"
        )

    pairs_path = pairs_pq or (output_pq.parent / "pairs.parquet")
    if not pairs_path.exists():
        return out, False
    pairs = pd.read_parquet(pairs_path)

    unit_cols = [c for c in ("unit_uid_a", "unit_uid_b", "unit_name_a", "unit_name_b") if c in pairs.columns]
    if not {"unit_uid_a", "unit_uid_b"}.issubset(unit_cols):
        return out, False

    merge_cols = ["pair_id", "is_swapped", *unit_cols]
    merge_cols = [c for c in merge_cols if c in pairs.columns]
    pairs_slim = pairs[merge_cols].copy()
    # Output parquet already has is_swapped; drop duplicates to avoid suffixed cols.
    if "is_swapped" in out.columns and "is_swapped" in pairs_slim.columns:
        pairs_slim = pairs_slim.drop(columns=["is_swapped"])
    merged = out.merge(pairs_slim, on="pair_id", how="left", validate="one_to_one")
    return merged, True


def _apply_zone_aggregation(
    df: pd.DataFrame,
    output_pq: Path,
    pairs_pq: Optional[Path],
    *,
    zone_geojson: Path,
    zone_id_column: str,
    zone_name_column: Optional[str],
    lat_col_a: str,
    lon_col_a: str,
    lat_col_b: str,
    lon_col_b: str,
    point_crs: str,
    coords_parquet: Optional[Path] = None,
    coords_id_column: str = "sample_id",
    coords_lat_column: str = "latitude",
    coords_lon_column: str = "longitude",
    id_col_a: str = "sample_id_a",
    id_col_b: str = "sample_id_b",
) -> tuple[pd.DataFrame, int, str]:
    """Spatially join each side's image point to a containing polygon and set
    ``unit_uid_*`` / ``unit_name_*`` from the zone, so the existing TrueSkill
    path rates zones instead of images.

    Returns ``(df, n_unmatched_pairs, provenance)`` where ``df`` has the four
    unit columns assigned (overwriting any existing ones) and ``provenance`` is
    a one-line human-readable description for the report overview.
    """
    import geopandas as gpd  # local import: only needed for zone aggregation

    if not zone_geojson.exists():
        raise SystemExit(f"--zone-geojson not found: {zone_geojson}")

    pairs_path = pairs_pq or (output_pq.parent / "pairs.parquet")
    if not pairs_path.exists():
        raise SystemExit(
            f"Zone aggregation needs lat/lon from pairs.parquet, not found: {pairs_path}"
        )
    pairs = pd.read_parquet(pairs_path)
    if "pair_id" not in pairs.columns:
        raise SystemExit(f"pairs.parquet {pairs_path} has no 'pair_id' column.")

    zones = gpd.read_file(zone_geojson)
    if zones.crs is None:
        # NYC Open Data GeoJSON exports are unprojected WGS84 lon/lat.
        zones = zones.set_crs("EPSG:4326")
    if zone_id_column not in zones.columns:
        raise SystemExit(
            f"--zone-id-column '{zone_id_column}' not in zone properties: {sorted(zones.columns)}"
        )

    name_cols = [c.strip() for c in (zone_name_column or "").split(",") if c.strip()]
    for c in name_cols:
        if c not in zones.columns:
            raise SystemExit(
                f"--zone-name-column '{c}' not in zone properties: {sorted(zones.columns)}"
            )
    if name_cols:
        zones["__zone_name__"] = (
            zones[name_cols].astype(str).agg(" · ".join, axis=1)
        )
    else:
        zones["__zone_name__"] = zones[zone_id_column].astype(str)
    zones["__zone_uid__"] = zones[zone_id_column].astype(str)
    zones_slim = zones[["__zone_uid__", "__zone_name__", "geometry"]]

    # Resolve per-side coordinates. Prefer an explicit coords lookup (keyed by
    # a per-side id in pairs.parquet); otherwise read lat/lon from pairs.
    def _side_coords(side: str, id_col: str, lat_col: str, lon_col: str) -> pd.DataFrame:
        if coords_parquet is not None:
            if id_col not in pairs.columns:
                raise SystemExit(
                    f"pairs.parquet has no id column '{id_col}' for the coords lookup."
                )
            lut = pd.read_parquet(
                coords_parquet,
                columns=[coords_id_column, coords_lat_column, coords_lon_column],
            )
            lut = lut.rename(
                columns={
                    coords_id_column: id_col,
                    coords_lat_column: "lat",
                    coords_lon_column: "lon",
                }
            ).drop_duplicates(id_col)
            lut[id_col] = lut[id_col].astype(str)
            s = pairs[["pair_id", id_col]].copy()
            s[id_col] = s[id_col].astype(str)
            s = s.merge(lut, on=id_col, how="left")[["pair_id", "lat", "lon"]]
        else:
            for c in (lat_col, lon_col):
                if c not in pairs.columns:
                    raise SystemExit(
                        f"pairs.parquet has no '{c}'; supply --coords-parquet to "
                        "resolve coordinates from a lookup instead."
                    )
            s = pairs[["pair_id", lat_col, lon_col]].rename(
                columns={lat_col: "lat", lon_col: "lon"}
            )
        s["lat"] = pd.to_numeric(s["lat"], errors="coerce")
        s["lon"] = pd.to_numeric(s["lon"], errors="coerce")
        return s.assign(side=side)

    # One sjoin over the full point set (geopandas' spatial index keeps this
    # cheap; images are reused across many pairs but de-duping is optional).
    long = pd.concat(
        [
            _side_coords("a", id_col_a, lat_col_a, lon_col_a),
            _side_coords("b", id_col_b, lat_col_b, lon_col_b),
        ],
        ignore_index=True,
    )
    n_no_coord = int(long["lat"].isna().sum() + long["lon"].isna().sum())
    if n_no_coord:
        print(f"[WARN] {n_no_coord:,} pair-sides have no resolvable coordinate.")
    long = long.dropna(subset=["lat", "lon"])
    pts = gpd.GeoDataFrame(
        long,
        geometry=gpd.points_from_xy(long["lon"], long["lat"]),
        crs=point_crs,
    ).to_crs(zones_slim.crs)
    joined = gpd.sjoin(pts, zones_slim, how="left", predicate="within")
    # A point exactly on a shared boundary can match >1 polygon; keep the first.
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined[["pair_id", "side", "__zone_uid__", "__zone_name__"]]

    # Reshape side a/b into one row per pair via a self-merge (pivot_table
    # silently drops these object columns even with aggfunc="first").
    side_a = (
        joined[joined["side"] == "a"]
        .drop(columns="side")
        .rename(columns={"__zone_uid__": "unit_uid_a", "__zone_name__": "unit_name_a"})
        .drop_duplicates("pair_id")
    )
    side_b = (
        joined[joined["side"] == "b"]
        .drop(columns="side")
        .rename(columns={"__zone_uid__": "unit_uid_b", "__zone_name__": "unit_name_b"})
        .drop_duplicates("pair_id")
    )
    wide = side_a.merge(side_b, on="pair_id", how="outer")

    df = df.drop(
        columns=[c for c in ("unit_uid_a", "unit_uid_b", "unit_name_a", "unit_name_b") if c in df.columns]
    )
    df = df.merge(wide, on="pair_id", how="left")

    n_unmatched = int(
        df["unit_uid_a"].isna().sum() + df["unit_uid_b"].isna().sum()
    )
    n_zones = pd.concat([df["unit_uid_a"], df["unit_uid_b"]]).dropna().nunique()
    provenance = (
        f"{zone_geojson.name} (id=`{zone_id_column}`"
        + (f", name=`{zone_name_column}`" if zone_name_column else "")
        + f", {n_zones:,} zones touched)"
    )
    return df, n_unmatched, provenance


# ---------------------------------------------------------------------------
# Hydra config discovery (for prompt capture)
# ---------------------------------------------------------------------------


def _load_run_config(output_pq: Path, override: Optional[Path]) -> Optional[dict]:
    """Find the run's resolved Hydra config.yaml. Returns parsed dict or None."""
    if override is not None:
        if not override.exists():
            print(f"[WARN] --hydra-config {override} not found; skipping prompt capture.")
            return None
        with open(override) as f:
            return yaml.safe_load(f)
    for parent in output_pq.parents:
        candidate = parent / ".hydra" / "config.yaml"
        if candidate.exists():
            with open(candidate) as f:
                return yaml.safe_load(f)
    return None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["relative_label"].value_counts(dropna=False)
    rows = []
    total = len(df)
    for label in ORDINAL_ORDER:
        n = int(counts.get(label, 0))
        rows.append({"label": label, "count": n, "proportion": n / total if total else 0.0})
    # Catch anything unexpected (e.g. nulls)
    for label, n in counts.items():
        if label not in ORDINAL_ORDER:
            rows.append(
                {
                    "label": f"(other) {label!r}",
                    "count": int(n),
                    "proportion": int(n) / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _label_entropy(dist: pd.DataFrame) -> float:
    p = dist["proportion"].to_numpy()
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _position_bias(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "presented_label" not in df.columns or "presented_order" not in df.columns:
        return None
    tab = (
        df.groupby("presented_order")["presented_label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    return tab.reindex(columns=ORDINAL_ORDER, fill_value=0.0).round(4)


def _reasoning_lengths(df: pd.DataFrame) -> pd.DataFrame:
    r = df["model_reasoning"].fillna("").astype(str)
    chars = r.str.len()
    words = r.str.split().map(len)
    return pd.DataFrame({"chars": chars, "words": words, "is_empty": r.str.len() == 0})


# ---------------------------------------------------------------------------
# TrueSkill
# ---------------------------------------------------------------------------


def _compute_trueskill(df: pd.DataFrame, draw_prob: float) -> pd.DataFrame:
    """Iterate 1v1 updates in row order. ``relative_score`` already reflects
    the canonical A-vs-B comparison (inverted for swapped pairs upstream)."""
    env = trueskill.TrueSkill(draw_probability=draw_prob)
    ratings: dict[str, trueskill.Rating] = defaultdict(env.create_rating)
    name_map: dict[str, str] = {}

    # Subset to valid rows.
    work = df.dropna(subset=["unit_uid_a", "unit_uid_b", "relative_score"]).copy()
    work["unit_uid_a"] = work["unit_uid_a"].astype(str)
    work["unit_uid_b"] = work["unit_uid_b"].astype(str)
    if "unit_name_a" in work.columns:
        work["unit_name_a"] = work["unit_name_a"].fillna("").astype(str)
    if "unit_name_b" in work.columns:
        work["unit_name_b"] = work["unit_name_b"].fillna("").astype(str)
    work = work[work["unit_uid_a"] != work["unit_uid_b"]]

    for row in work.itertuples(index=False):
        a = row.unit_uid_a
        b = row.unit_uid_b
        if "unit_name_a" in work.columns:
            name_map.setdefault(a, getattr(row, "unit_name_a", "") or a)
        if "unit_name_b" in work.columns:
            name_map.setdefault(b, getattr(row, "unit_name_b", "") or b)
        score = int(row.relative_score)
        ra, rb = ratings[a], ratings[b]
        if score > 0:
            ra, rb = env.rate_1vs1(ra, rb, drawn=False)  # A wins
        elif score < 0:
            rb, ra = env.rate_1vs1(rb, ra, drawn=False)  # B wins
        else:
            ra, rb = env.rate_1vs1(ra, rb, drawn=True)
        ratings[a] = ra
        ratings[b] = rb

    counts = (
        pd.concat([work["unit_uid_a"], work["unit_uid_b"]])
        .value_counts()
        .to_dict()
    )

    rows = [
        {
            "unit_uid": uid,
            "unit_name": name_map.get(uid, uid) or uid,
            "mu": r.mu,
            "sigma": r.sigma,
            "ts_point_estimate": r.mu,
            "ts_conservative": r.mu - 3.0 * r.sigma,
            "n_comparisons": int(counts.get(uid, 0)),
        }
        for uid, r in ratings.items()
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("ts_conservative", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_label_distribution(dist: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    mask = dist["label"].isin(ORDINAL_ORDER)
    sub = dist[mask].set_index("label").reindex(ORDINAL_ORDER)
    bars = ax.bar(
        sub.index,
        sub["count"],
        color=ORDINAL_COLORS,
        edgecolor="#444444",
        linewidth=0.6,
        width=0.68,
    )
    total = int(sub["count"].sum())
    for bar, n in zip(bars, sub["count"]):
        pct = (n / total * 100.0) if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(n):,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
        )
    ax.set_ylabel("count")
    ax.set_title("Ordinal label distribution  ·  canonical A-vs-B")
    ax.set_ylim(0, sub["count"].max() * 1.20 if total else 1)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_reasoning_length(lengths: pd.DataFrame, out_path: Path) -> None:
    nonzero = lengths.loc[~lengths["is_empty"], "words"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if nonzero.empty:
        ax.text(0.5, 0.5, "No reasoning captured", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(nonzero, bins=40, color=ACCENT, edgecolor="white", linewidth=0.6, alpha=0.9)
        ax.axvline(
            nonzero.median(),
            color="#c0392b",
            linestyle="--",
            linewidth=1.2,
            label=f"median = {int(nonzero.median())}",
        )
        ax.axvline(
            nonzero.mean(),
            color=ACCENT_WARM,
            linestyle=":",
            linewidth=1.2,
            label=f"mean = {nonzero.mean():.0f}",
        )
        ax.legend(loc="upper right")
    ax.set_xlabel("reasoning length (words, non-empty traces)")
    ax.set_ylabel("count")
    ax.set_title("Reasoning-trace length distribution")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _tokenize_for_cloud(texts: pd.Series, stopwords: set[str]) -> str:
    buf: list[str] = []
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        for tok in TOKEN_RE.findall(t.lower()):
            if tok in stopwords or len(tok) < 3:
                continue
            buf.append(tok)
    return " ".join(buf)


def _plot_wordcloud(reasoning: pd.Series, out_path: Path, extra_stopwords: set[str]) -> Optional[int]:
    stopwords = set(STOPWORDS) | {s.lower() for s in DEFAULT_DOMAIN_STOPWORDS} | extra_stopwords
    text = _tokenize_for_cloud(reasoning, stopwords)
    if not text:
        return None
    wc = WordCloud(
        width=1600,
        height=700,
        background_color="white",
        colormap="viridis",
        max_words=200,
        collocations=True,
        prefer_horizontal=0.9,
        relative_scaling=0.45,
        min_font_size=8,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return len(text.split())


def _plot_trueskill_ranking(
    ratings: pd.DataFrame,
    out_path: Path,
    attribute: str,
    unit_label_plural: str,
    highlight_n: int = 5,
) -> None:
    if ratings.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No TrueSkill ratings computed", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    sorted_r = ratings.sort_values("ts_point_estimate", ascending=False).reset_index(drop=True)
    n = len(sorted_r)
    x = np.arange(n)
    mu = sorted_r["ts_point_estimate"].to_numpy()
    sigma = sorted_r["sigma"].to_numpy()

    # Color top/bottom highlight vs body.
    h = min(highlight_n, n // 3) if n else 0
    colors = np.full(n, "#9bb4d1", dtype=object)  # muted body
    if h > 0:
        colors[:h] = "#1b7837"  # top in green
        colors[-h:] = "#762a83"  # bottom in purple

    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.errorbar(
        x,
        mu,
        yerr=sigma,
        fmt="none",
        ecolor="#cccccc",
        elinewidth=0.5,
        capsize=0,
        alpha=0.7,
    )
    ax.scatter(x, mu, s=10, c=list(colors), alpha=0.9, linewidths=0)
    # Mean reference line.
    ax.axhline(float(np.mean(mu)), color="#888888", linestyle="--", linewidth=0.8, alpha=0.7, label=f"mean μ = {np.mean(mu):.2f}")
    # Annotate top/bottom counts.
    if h > 0:
        ax.annotate(
            f"top {h}",
            xy=(x[h - 1], mu[:h].min()),
            xytext=(6, -2),
            textcoords="offset points",
            fontsize=9,
            color="#1b7837",
            va="top",
        )
        ax.annotate(
            f"bottom {h}",
            xy=(x[-h], mu[-h:].max()),
            xytext=(-6, 2),
            textcoords="offset points",
            fontsize=9,
            color="#762a83",
            va="bottom",
            ha="right",
        )
    ax.set_xlabel(f"rank  (most → least {attribute})")
    ax.set_ylabel(r"TrueSkill $\mu$  (error bars = $\pm 1\sigma$)")
    ax.set_title(f"TrueSkill ratings  ·  {n} {unit_label_plural}  ·  sorted by μ")
    ax.set_xlim(-1, n)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_trueskill_distribution(ratings: pd.DataFrame, out_path: Path) -> None:
    if ratings.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    axes[0].hist(
        ratings["ts_point_estimate"], bins=30, color=ACCENT, edgecolor="white", linewidth=0.6, alpha=0.9
    )
    axes[0].set_xlabel(r"TrueSkill $\mu$")
    axes[0].set_ylabel("count")
    axes[0].set_title(r"Distribution of $\mu$  ·  skill")
    axes[1].hist(
        ratings["sigma"], bins=30, color=ACCENT_WARM, edgecolor="white", linewidth=0.6, alpha=0.9
    )
    axes[1].set_xlabel(r"TrueSkill $\sigma$")
    axes[1].set_title(r"Distribution of $\sigma$  ·  uncertainty")
    for ax in axes:
        ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Example pair images
# ---------------------------------------------------------------------------


def _stitch_pair_jpeg(
    path_a: Path,
    path_b: Path,
    out_path: Path,
    *,
    scale: float,
    quality: int,
) -> Optional[tuple[int, int]]:
    """Side-by-side stitch of the canonical A|B pair, resized + JPEG-compressed."""
    try:
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
    except (FileNotFoundError, OSError):
        return None

    def _resize(im: Image.Image) -> Image.Image:
        w = max(1, int(im.width * scale))
        h = max(1, int(im.height * scale))
        return im.resize((w, h), Image.LANCZOS)

    img_a = _resize(img_a)
    img_b = _resize(img_b)
    h = max(img_a.height, img_b.height)
    canvas = Image.new("RGB", (img_a.width + img_b.width, h), (255, 255, 255))
    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b, (img_a.width, 0))
    canvas.save(out_path, "JPEG", quality=quality, optimize=True)
    return canvas.size


def _pair_side_label(row: pd.Series, unit_uid: str) -> str:
    if str(row.get("unit_uid_a", "")) == unit_uid:
        return "A"
    if str(row.get("unit_uid_b", "")) == unit_uid:
        return "B"
    return "?"


def _select_top_bottom_pairs(
    df: pd.DataFrame,
    ratings: pd.DataFrame,
    n: int,
    seed: int,
) -> list[dict]:
    """One example pair per top-N and bottom-N unit (by TrueSkill point estimate)."""
    if ratings.empty or n <= 0:
        return []
    rng = np.random.default_rng(seed)
    sorted_r = ratings.sort_values("ts_point_estimate", ascending=False).reset_index(drop=True)
    top = sorted_r.head(n)
    bot = sorted_r.tail(n).iloc[::-1]

    selections: list[dict] = []
    for section_label, subset in (("top", top), ("bottom", bot)):
        for rank, rec in enumerate(subset.itertuples(index=False), start=1):
            uid = str(rec.unit_uid)
            cand = df[(df["unit_uid_a"].astype(str) == uid) | (df["unit_uid_b"].astype(str) == uid)]
            if cand.empty:
                continue
            row = cand.iloc[int(rng.integers(0, len(cand)))]
            selections.append(
                {
                    "section": section_label,
                    "rank": rank,
                    "unit_uid": uid,
                    "unit_name": rec.unit_name,
                    "ts_point_estimate": float(rec.ts_point_estimate),
                    "n_comparisons": int(rec.n_comparisons),
                    "row": row,
                }
            )
    return selections


def _select_random_pairs(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if df.empty or n <= 0:
        return df.iloc[0:0]
    rng = np.random.default_rng(seed)
    k = min(n, len(df))
    idx = rng.choice(len(df), size=k, replace=False)
    return df.iloc[sorted(idx.tolist())].reset_index(drop=True)


def _render_pair_images(
    selections: list[dict],
    random_pairs: pd.DataFrame,
    images_dir: Path,
    *,
    scale: float,
    quality: int,
) -> tuple[list[dict], list[dict]]:
    """Write stitched JPEGs for each selected pair; return records with image rel paths."""
    pairs_dir = images_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[dict] = []
    for sel in selections:
        row = sel["row"]
        uid_short = sel["unit_uid"][:10]
        fname = f"{sel['section']}_{sel['rank']:02d}_{uid_short}.jpg"
        out = pairs_dir / fname
        size = _stitch_pair_jpeg(
            Path(row["image_path_a"]),
            Path(row["image_path_b"]),
            out,
            scale=scale,
            quality=quality,
        )
        if size is None:
            continue
        rendered.append({**sel, "image_path": out, "image_size": size})

    rendered_rand: list[dict] = []
    for i, row in enumerate(random_pairs.itertuples(index=False)):
        fname = f"random_{i:02d}.jpg"
        out = pairs_dir / fname
        size = _stitch_pair_jpeg(
            Path(row.image_path_a),
            Path(row.image_path_b),
            out,
            scale=scale,
            quality=quality,
        )
        if size is None:
            continue
        rendered_rand.append(
            {
                "index": i,
                "pair_id": str(getattr(row, "pair_id", "")),
                "unit_name_a": str(getattr(row, "unit_name_a", "") or ""),
                "unit_name_b": str(getattr(row, "unit_name_b", "") or ""),
                "unit_uid_a": str(getattr(row, "unit_uid_a", "") or ""),
                "unit_uid_b": str(getattr(row, "unit_uid_b", "") or ""),
                "relative_label": str(row.relative_label),
                "relative_score": int(row.relative_score),
                "presented_order": str(getattr(row, "presented_order", "")),
                "image_path": out,
                "image_size": size,
            }
        )
    return rendered, rendered_rand


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------


def _export_pdf(md_path: Path, pdf_engine: str, *, landscape: bool = True) -> Optional[Path]:
    """Render the markdown to PDF via pandoc. Returns the PDF path on success."""
    if shutil.which("pandoc") is None:
        print("[WARN] pandoc not found on PATH; skipping PDF export.")
        return None
    pdf_path = md_path.with_suffix(".pdf")
    cmd = [
        "pandoc",
        md_path.name,
        "-o",
        pdf_path.name,
        "--pdf-engine",
        pdf_engine,
        "--from",
        "markdown+raw_tex+yaml_metadata_block",
        "--toc",
        "--toc-depth=2",
        "--number-sections",
        # Page geometry.
        "-V",
        "papersize=letter",
        "-V",
        f"geometry:{'landscape,' if landscape else ''}margin=0.55in,top=0.7in,bottom=0.7in",
        # Typography.
        "-V",
        "fontsize=11pt",
        "-V",
        "linestretch=1.18",
        "-V",
        "mainfont=DejaVu Serif",
        "-V",
        "sansfont=DejaVu Sans",
        "-V",
        "monofont=DejaVu Sans Mono",
        # Links (basic xcolor names to avoid dvipsnames dependency).
        "-V",
        "colorlinks=true",
        "-V",
        "linkcolor=blue",
        "-V",
        "urlcolor=blue",
        "-V",
        "toccolor=black",
        # Header + table polish.
        "-V",
        (
            "header-includes="
            "\\usepackage{url}"
            "\\renewcommand{\\arraystretch}{1.15}"
            "\\usepackage{fancyhdr}"
            "\\pagestyle{fancy}"
            "\\fancyhf{}"
            "\\fancyhead[L]{\\footnotesize\\nouppercase\\leftmark}"
            "\\fancyhead[R]{\\footnotesize\\thepage}"
            "\\renewcommand{\\headrulewidth}{0.3pt}"
            "\\renewcommand{\\footrulewidth}{0pt}"
            "\\setlength{\\parskip}{0.35em}"
        ),
    ]
    try:
        subprocess.run(cmd, cwd=md_path.parent, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] pandoc failed (engine={pdf_engine}):")
        print(exc.stderr[-2000:] if exc.stderr else "(no stderr)")
        return None
    return pdf_path


# ---------------------------------------------------------------------------
# Markdown assembly
# ---------------------------------------------------------------------------


def _md_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float) and not math.isnan(v):
                cells.append(float_fmt.format(v))
            elif v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_report(
    *,
    df: pd.DataFrame,
    has_units: bool,
    ratings: pd.DataFrame,
    label_dist: pd.DataFrame,
    lengths: pd.DataFrame,
    wordcloud_tokens: Optional[int],
    position_bias: Optional[pd.DataFrame],
    example_pairs: list[dict],
    random_pairs: list[dict],
    images_dir: Path,
    out_md: Path,
    title: str,
    attribute: str,
    unit_label: str,
    unit_label_plural: str,
    top_n: int,
    min_comparisons: int,
    source_parquet: Path,
    run_config: Optional[dict],
    zone_provenance: Optional[str] = None,
) -> None:
    entropy = _label_entropy(label_dist)
    total = len(df)
    captured = int((~lengths["is_empty"]).sum())
    capture_rate = captured / total if total else 0.0

    rel = lambda p: Path(images_dir.name) / p.name  # noqa: E731
    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%B %d, %Y")

    # Image sizing tuned for letter-landscape with 0.55" margins.
    IMG_WIDE = "width=86%"    # big charts (ranking, distributions, wordcloud)
    IMG_STD = "width=70%"     # medium charts (bar, histogram)
    IMG_PAIR = "width=72%"    # stitched pair images (1024x512 aspect, centered)

    def _newpage() -> None:
        lines.append("\\newpage")
        lines.append("")

    def _centered_image(rel_path: Path, alt: str, width_attr: str) -> list[str]:
        # Pandoc markdown: a paragraph with only an image becomes a figure with
        # caption = alt text. Wrap in a centering div so landscape pages look tidy.
        return [
            "::: {.center data-latex=\"\"}",
            f"![{alt}]({rel_path}){{ {width_attr} }}",
            ":::",
            "",
        ]

    lines: list[str] = []

    # ---- YAML title block (pandoc metadata) ----
    subtitle = f"Pairwise VQA  ·  ordinal analysis & TrueSkill ranking  ·  {unit_label_plural}"
    lines.append("---")
    lines.append(f'title: "{title}"')
    lines.append(f'subtitle: "{subtitle}"')
    lines.append(f'date: "{date_str}"')
    lines.append("---")
    lines.append("")

    # ---- Run overview ----
    # Keep the Source/Generated/Attribute metadata on this page instead of
    # letting it float off the end of the TOC. \path{} from `url` lets the
    # long parquet path break at /, _, . so it wraps instead of overflowing.
    _newpage()
    lines.append("# Run overview")
    lines.append("")
    lines.append(
        f"**Source:** \\path{{{source_parquet}}}  \n"
        f"**Generated:** {now_utc.isoformat(timespec='seconds')} &nbsp;·&nbsp; "
        f"**Attribute:** *{attribute}*"
    )
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| :--- | ---: |")
    lines.append(f"| rows | {total:,} |")
    if has_units:
        unique = pd.concat([df["unit_uid_a"], df["unit_uid_b"]]).astype(str).nunique()
        lines.append(f"| distinct {unit_label_plural} compared | {unique:,} |")
    if zone_provenance:
        lines.append(f"| zone aggregation | {zone_provenance} |")
    lines.append(f"| label entropy (nats) | {entropy:.4f} |")
    lines.append(f"| reasoning capture rate | {capture_rate:.1%} ({captured:,} / {total:,}) |")
    if wordcloud_tokens is not None:
        lines.append(f"| word-cloud tokens (after stopwords) | {wordcloud_tokens:,} |")
    lines.append("")

    # ---- Prompt (from resolved Hydra config) ----
    if run_config is not None:
        prompt_cfg = (run_config.get("prompt") or {})
        model_cfg = (run_config.get("model") or {})
        sampler_cfg = (run_config.get("pair_sampler") or {})

        sys_prompt = (prompt_cfg.get("system") or "").strip()
        user_template = (prompt_cfg.get("user_template") or "").strip()
        schema = (prompt_cfg.get("structured_output") or {}).get("json_schema") or {}
        model_src = str(model_cfg.get("model_source") or "")
        model_name = Path(model_src).name if model_src else ""

        if sys_prompt or user_template or schema:
            _newpage()
            lines.append("# Prompt")
            lines.append("")
            meta_bits: list[str] = []
            if model_name:
                meta_bits.append(f"**Model:** `{model_name}`")
            if sampler_cfg:
                meta_bits.append(
                    f"**Sampling:** mode=`{sampler_cfg.get('mode','?')}`, "
                    f"counterbalance=`{sampler_cfg.get('counterbalance_mode','?')}`, "
                    f"seed=`{sampler_cfg.get('pair_seed','?')}`, "
                    f"max_pairs=`{sampler_cfg.get('max_pairs','?')}`"
                )
            if meta_bits:
                lines.append(" &nbsp;·&nbsp; ".join(meta_bits))
                lines.append("")

            if sys_prompt:
                lines.append("## System prompt")
                lines.append("")
                lines.append("```")
                lines.append(sys_prompt)
                lines.append("```")
                lines.append("")
            if user_template:
                lines.append("## User template")
                lines.append("")
                lines.append("```")
                lines.append(user_template)
                lines.append("```")
                lines.append("")
            if schema:
                lines.append("## Structured-output schema")
                lines.append("")
                lines.append("```yaml")
                lines.append(yaml.safe_dump(schema, sort_keys=False).rstrip())
                lines.append("```")
                lines.append("")

    # ---- Label distribution ----
    _newpage()
    lines.append("# Ordinal label distribution")
    lines.append("")
    lines.append(
        "Counts of the canonical A-vs-B ordinal judgment. "
        "`MuchLess` / `MuchMore` are the strongest signals; `Same` means the VLM read them as equivalent."
    )
    lines.append("")
    display_dist = label_dist.copy()
    display_dist["proportion"] = display_dist["proportion"].map(lambda p: f"{p:.4f}")
    display_dist = display_dist.rename(columns={
        "label": "Label",
        "count": "Count",
        "proportion": "Proportion",
    })
    lines.append(_md_table(display_dist))
    lines.append("")
    lines.extend(_centered_image(rel(images_dir / "label_distribution.png"), "Ordinal label distribution", IMG_STD))

    if position_bias is not None:
        lines.append("## Presented-order label proportions")
        lines.append("")
        lines.append(
            "_Position-bias diagnostic: proportion of each **presented** (un-de-swapped) label "
            "by presentation order. Large asymmetries suggest the VLM has a left/right preference._"
        )
        lines.append("")
        disp = position_bias.reset_index().rename(columns={
            "index": "Presented order",
            "presented_order": "Presented order",
        })
        lines.append(_md_table(disp, float_fmt="{:.4f}"))
        lines.append("")

    # ---- Reasoning length ----
    _newpage()
    lines.append("# Reasoning-trace length")
    lines.append("")
    nonzero = lengths.loc[~lengths["is_empty"]]
    if nonzero.empty:
        lines.append("_No non-empty reasoning traces captured._")
        lines.append("")
    else:
        lines.append(
            f"{captured:,} of {total:,} rows ({capture_rate:.1%}) have a captured `model_reasoning` trace. "
            "Empty traces most often come from first-batch thinking drops."
        )
        lines.append("")
        stats = nonzero[["chars", "words"]].describe().round(2)
        stats = stats.reset_index().rename(columns={
            "index": "Statistic",
            "chars": "Characters",
            "words": "Words",
        })
        lines.append(_md_table(stats))
        lines.append("")
        lines.extend(_centered_image(rel(images_dir / "reasoning_length.png"), "Reasoning length histogram", IMG_STD))

    # ---- Word cloud ----
    _newpage()
    lines.append("# Reasoning word cloud")
    lines.append("")
    if wordcloud_tokens is None:
        lines.append("_No tokens remain after stopword filtering._")
        lines.append("")
    else:
        lines.append(
            "Top terms in captured reasoning traces after stopword + domain-filler filtering. "
            "Useful for sanity-checking what the VLM is *actually* reasoning about."
        )
        lines.append("")
        lines.extend(_centered_image(rel(images_dir / "wordcloud.png"), "Reasoning word cloud", IMG_WIDE))

    # ---- TrueSkill ranking ----
    _newpage()
    lines.append(f"# TrueSkill ranking")
    lines.append("")
    if ratings.empty:
        lines.append("_No TrueSkill ratings — is this a run with unit identity in `pairs.parquet`?_")
        lines.append("")
    else:
        n_units = len(ratings)
        # `ts_point_estimate` is identical to `mu`; drop it so we don't show the same column twice.
        summary_stats = (
            ratings[["mu", "sigma", "ts_conservative", "n_comparisons"]]
            .describe()
            .round(3)
            .reset_index()
            .rename(columns={
                "index": "Statistic",
                "mu": "μ",
                "sigma": "σ",
                "ts_conservative": "μ − 3σ",
                "n_comparisons": "# comparisons",
            })
        )
        lines.append(
            f"{n_units:,} {unit_label_plural} rated with `trueskill.rate_1vs1` (draws allowed). "
            f"Ordered below by **μ − 3σ** (conservative rating) — the ordinal score asked "
            f"“more of *{attribute}*?”, so higher = more."
        )
        lines.append("")
        lines.append(_md_table(summary_stats))
        lines.append("")
        lines.extend(_centered_image(rel(images_dir / "trueskill_ranking.png"), "TrueSkill ranking", IMG_WIDE))
        lines.extend(_centered_image(rel(images_dir / "trueskill_distributions.png"), "TrueSkill μ and σ distributions", IMG_WIDE))

        filt = ratings[ratings["n_comparisons"] >= min_comparisons]
        top = filt.head(top_n)
        bottom = filt.tail(top_n).iloc[::-1]

        unit_header = unit_label[:1].upper() + unit_label[1:]
        rename_map = {
            "rank": "Rank",
            "unit_name": unit_header,
            "ts_point_estimate": "μ",
            "ts_conservative": "μ − 3σ",
            "sigma": "σ",
            "n_comparisons": "# comparisons",
        }

        def _fmt(sub: pd.DataFrame) -> pd.DataFrame:
            # Dropped `unit_uid` — it wastes a column on the landscape page.
            keep = ["unit_name", "ts_point_estimate", "ts_conservative", "sigma", "n_comparisons"]
            keep = [c for c in keep if c in sub.columns]
            out = sub[keep].reset_index(drop=True).round(3)
            out.insert(0, "rank", np.arange(1, len(out) + 1))
            return out.rename(columns=rename_map)

        _newpage()
        lines.append(f"# Top {len(top)} most {attribute}")
        lines.append("")
        lines.append(_md_table(_fmt(top)))
        lines.append("")

        _newpage()
        lines.append(f"# Bottom {len(bottom)} (least {attribute})")
        lines.append("")
        lines.append(_md_table(_fmt(bottom)))
        lines.append("")

    # ---- Example pair images: top/bottom ranked units ----
    if example_pairs:
        top_examples = [e for e in example_pairs if e["section"] == "top"]
        bot_examples = [e for e in example_pairs if e["section"] == "bottom"]

        def _emit_examples(bucket: list[dict], header: str, highlight: str) -> None:
            if not bucket:
                return
            _newpage()
            lines.append(f"# {header}")
            lines.append("")
            lines.append(
                f"One randomly-drawn pair per {unit_label} from the **{highlight}** of the ranking. "
                f"Side **A** (left) / **B** (right); `relative_label` is the canonical A-vs-B judgment "
                "(already de-swapped)."
            )
            lines.append("")
            for idx, e in enumerate(bucket):
                row = e["row"]
                side = _pair_side_label(row, e["unit_uid"])
                a_name = row.get("unit_name_a", "") or ""
                b_name = row.get("unit_name_b", "") or ""
                rel_path = Path(images_dir.name) / "pairs" / e["image_path"].name
                if idx > 0:
                    _newpage()
                lines.append(
                    f"## #{e['rank']}  ·  {e['unit_name']}"
                )
                lines.append("")
                lines.append(
                    f"μ = **{e['ts_point_estimate']:.2f}**  ·  "
                    f"n = **{e['n_comparisons']}**  ·  "
                    f"highlighted side = **{side}**"
                )
                lines.append("")
                lines.append(
                    f"**A:** {a_name} &nbsp;·&nbsp; **B:** {b_name}  \n"
                    f"**label:** `{row['relative_label']}` (score {int(row['relative_score'])}) &nbsp;·&nbsp; "
                    f"**pair_id:** `{row['pair_id']}`"
                )
                lines.append("")
                lines.extend(
                    _centered_image(rel_path, f"{e['unit_name']} example pair", IMG_PAIR)
                )

        _emit_examples(
            top_examples,
            f"Example pairs — top {len(top_examples)} most {attribute}",
            "top",
        )
        _emit_examples(
            bot_examples,
            f"Example pairs — bottom {len(bot_examples)} (least {attribute})",
            "bottom",
        )

    # ---- Random inspection sample ----
    if random_pairs:
        _newpage()
        lines.append(f"# Random inspection sample  ·  N = {len(random_pairs)}")
        lines.append("")
        lines.append(
            "Randomly sampled pairs with the VLM's ordinal judgment, for spot inspection of label quality."
        )
        lines.append("")
        for i, r in enumerate(random_pairs):
            rel_path = Path(images_dir.name) / "pairs" / r["image_path"].name
            if i > 0:
                _newpage()
            lines.append(
                f"## Pair `{r['pair_id']}`  ·  `{r['relative_label']}` (score {r['relative_score']})"
            )
            lines.append("")
            caption_parts = [f"**A:** {r['unit_name_a']}"]
            if r["unit_name_b"]:
                caption_parts.append(f"**B:** {r['unit_name_b']}")
            if r["presented_order"]:
                caption_parts.append(f"**presented:** {r['presented_order']}")
            lines.append("  ·  ".join(caption_parts))
            lines.append("")
            lines.extend(
                _centered_image(rel_path, f"Random pair {r['index']}", IMG_PAIR)
            )

    out_md.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    if not args.output_parquet.exists():
        raise SystemExit(f"Output parquet not found: {args.output_parquet}")

    df, has_units = _load_merged(args.output_parquet, args.pairs_parquet)

    zone_provenance: Optional[str] = None
    if args.zone_geojson is not None:
        if not args.zone_id_column:
            raise SystemExit("--zone-id-column is required when --zone-geojson is given.")
        df, n_unmatched, zone_provenance = _apply_zone_aggregation(
            df,
            args.output_parquet,
            args.pairs_parquet,
            zone_geojson=args.zone_geojson,
            zone_id_column=args.zone_id_column,
            zone_name_column=args.zone_name_column,
            lat_col_a=args.lat_col_a,
            lon_col_a=args.lon_col_a,
            lat_col_b=args.lat_col_b,
            lon_col_b=args.lon_col_b,
            point_crs=args.point_crs,
            coords_parquet=args.coords_parquet,
            coords_id_column=args.coords_id_column,
            coords_lat_column=args.coords_lat_column,
            coords_lon_column=args.coords_lon_column,
            id_col_a=args.id_col_a,
            id_col_b=args.id_col_b,
        )
        has_units = True
        print(f"Zone aggregation:  {zone_provenance}")
        if n_unmatched:
            print(
                f"[WARN] {n_unmatched:,} pair-sides fell outside all zones "
                "(dropped from TrueSkill)."
            )

    out_md = args.out or args.output_parquet.with_suffix(".report.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    images_dir = out_md.parent / f"{out_md.stem}_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    title = args.title or f"Pairwise VQA report: {args.output_parquet.stem}"
    unit_label_plural = args.unit_label_plural or f"{args.unit_label}s"

    label_dist = _label_distribution(df)
    lengths = _reasoning_lengths(df)
    position_bias = _position_bias(df)

    _plot_label_distribution(label_dist, images_dir / "label_distribution.png")
    _plot_reasoning_length(lengths, images_dir / "reasoning_length.png")

    extra_sw = {s.strip().lower() for s in args.extra_stopwords.split(",") if s.strip()}
    wordcloud_tokens = _plot_wordcloud(
        df["model_reasoning"], images_dir / "wordcloud.png", extra_sw
    )

    if has_units:
        ratings = _compute_trueskill(df, draw_prob=args.draw_prob)
        _plot_trueskill_ranking(
            ratings,
            images_dir / "trueskill_ranking.png",
            args.attribute,
            unit_label_plural,
        )
        _plot_trueskill_distribution(ratings, images_dir / "trueskill_distributions.png")
    else:
        ratings = pd.DataFrame()

    # Stitched Cyclomedia pair images: one per top/bottom unit + a random sample.
    example_selections = (
        _select_top_bottom_pairs(df, ratings, args.top_bottom_image_n, seed=args.image_seed)
        if has_units
        else []
    )
    random_sample_df = _select_random_pairs(df, args.random_image_n, seed=args.image_seed)
    example_pairs, random_pairs = _render_pair_images(
        example_selections,
        random_sample_df,
        images_dir,
        scale=args.image_scale,
        quality=args.image_quality,
    )

    run_config = _load_run_config(args.output_parquet, args.hydra_config)

    _write_report(
        df=df,
        has_units=has_units,
        ratings=ratings,
        label_dist=label_dist,
        lengths=lengths,
        wordcloud_tokens=wordcloud_tokens,
        position_bias=position_bias,
        example_pairs=example_pairs,
        random_pairs=random_pairs,
        images_dir=images_dir,
        out_md=out_md,
        title=title,
        attribute=args.attribute,
        unit_label=args.unit_label,
        unit_label_plural=unit_label_plural,
        top_n=args.top_n,
        min_comparisons=args.min_comparisons,
        source_parquet=args.output_parquet.resolve(),
        run_config=run_config,
        zone_provenance=zone_provenance,
    )

    print(f"Wrote report:      {out_md}")
    print(f"Images directory:  {images_dir}")
    print(f"Pair images:       {len(example_pairs)} ranked + {len(random_pairs)} random")

    if args.pdf:
        pdf_path = _export_pdf(
            out_md,
            args.pdf_engine,
            landscape=(args.pdf_orientation == "landscape"),
        )
        if pdf_path is not None:
            print(f"Wrote PDF:         {pdf_path}")


if __name__ == "__main__":
    main()
