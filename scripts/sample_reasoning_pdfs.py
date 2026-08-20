#!/usr/bin/env python3
"""Render PDF samples of pairwise-VQA reasoning traces with the paired images.

For a pairwise VQA stage output parquet (the kind produced by
``dagspaces.urbanpairvqa``), draw a reproducible random sample of pairs and
write one PDF per pair containing:

  * a **prompt page** — the exact ``system`` + ``user_template`` prompt the
    model was given, verbatim from the prompt config (rendered once, at the
    top of each file; once at the very top of the ``--combined`` report);
  * a cover page — the two images **in presented order** (left = "Image A",
    right = "Image B", matching how the reasoning text refers to them),
    plus the metadata and the model's final verdict;
  * following pages — the full ``model_reasoning`` thinking trace, wrapped
    and paginated.

Only rows with a non-empty reasoning trace are eligible by default (a thinking
model occasionally spends its whole token budget mid-trace and emits no final
JSON; pass ``--include-empty`` to sample from all rows regardless).

The exact prompt is auto-resolved from the parquet filename
(restaurants / schools / libraries -> the matching
``dagspaces/urbanpairvqa/conf/prompt/*.yaml``); override with ``--prompt-file``.

Examples
--------
Restaurants run (12 pairs)::

    python scripts/sample_reasoning_pdfs.py \\
        multirun/2026-06-04_URBANPAIRVQA/17-28-28/0/outputs/pairwise/restaurants_mvp_20260604_172839.parquet \\
        -n 12 --title "Restaurants — Qwen3.5-9B thinking" \\
        --out reports/pairwise/restaurants_reasoning_pdfs

Schools run, only decisive (non-"Same") verdicts, plus a combined PDF::

    python scripts/sample_reasoning_pdfs.py \\
        multirun/2026-06-04_URBANPAIRVQA/17-28-42/0/outputs/pairwise/schools_mvp_20260604_172848.parquet \\
        -n 12 --decisive-only --combined \\
        --title "Schools — Qwen3.5-9B thinking" \\
        --out reports/pairwise/schools_reasoning_pdfs
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# Verdict labels, ordered from "strongly prefer B" to "strongly prefer A".
LABEL_ORDER = ["MuchLess", "Less", "Same", "More", "MuchMore"]

# Prompt-config dir and the keyword -> prompt-stem map used to auto-resolve the
# exact prompt from a parquet filename.
PROMPT_DIR = Path(__file__).resolve().parents[1] / "dagspaces/urbanpairvqa/conf/prompt"
# Warning: the schools prompt changed on 2026-08-13. A parquet from before that
# date used `deprecated/pairwise_school_send_child_ordinal`, so the text below
# is the WRONG prompt for it. Pass `--prompt-file` to render an older schools
# run, and point it at the file in `prompt/deprecated/`.
PROMPT_BY_KEYWORD = {
    "restaurant": "pairwise_restaurant_eat_at_ordinal",
    "school": "pairwise_school_better_ordinal",
    "librar": "pairwise_library_maintained_ordinal",
    "street": "pairwise_street_photography_ordinal",
    "subway": "pairwise_subway_safety_ordinal",
}

# --- Geo enrichment ---------------------------------------------------------
# For each image we resolve its neighborhood (NTA), census tract, and the
# tract's ACS median household income. lat/lon comes from the Cyclomedia catalog
# (universal across datasets); the 2020 census-tract polygons already carry the
# NTA name, and the tract geoid keys into ACS 2024 5-yr S1901. Mirrors the join
# in scripts/pairwise_socioeconomic_regression.py.
REPO = Path(__file__).resolve().parents[1]
TRACT_GEOJSON = REPO / "data/geo/2020_Census_Tracts_20260304.geojson"
INCOME_CSV = REPO / "data/demo/ct/ACSST5Y2024.S1901-Data.csv"
_DATASET_RE = re.compile(r"/raw/([^/]+)/")


def _presented_sids(row: pd.Series) -> tuple[str | None, str | None]:
    """(left_sample_id, right_sample_id) in PRESENTED order.

    ``is_swapped`` flips which canonical side (a/b) is shown on the left, so key
    off the actually-displayed path rather than assuming left == sample_id_a.
    """
    a, b = row.get("sample_id_a"), row.get("sample_id_b")
    left_path = row.get("presented_left_path")
    if left_path is not None and left_path == row.get("image_path_b"):
        return b, a
    return a, b


def _load_geo_map(sample: pd.DataFrame) -> dict[str, dict]:
    """sample_id -> {'borough','nta','tract','median_income'} for the sampled pairs.

    Best-effort: any failure (missing geo deps/files, no catalog hit) prints a
    warning and returns ``{}`` so the PDFs still render without the geo lines.
    """
    sids: set[str] = set()
    for c in ("sample_id_a", "sample_id_b"):
        if c in sample.columns:
            sids |= set(sample[c].dropna().astype(str))
    if not sids:
        return {}
    try:
        import sys

        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))  # make `dagspaces` importable when run as a script
        import geopandas as gpd
        import polars as pl

        from dagspaces.common.cyclomedia_catalog import CyclomediaCatalog
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] geo enrichment unavailable ({e}); skipping NTA/tract/income.")
        return {}
    try:
        # datasets present in the sample -> prune the catalog scan
        datasets: set[str] = set()
        for c in ("image_path_a", "image_path_b",
                  "presented_left_path", "presented_right_path"):
            if c in sample.columns:
                for p in sample[c].dropna().astype(str):
                    m = _DATASET_RE.search(p)
                    if m:
                        datasets.add(m.group(1))
        cat = CyclomediaCatalog()
        ll = (
            cat.scan(datasets=datasets or None)
            .select(["sample_id", "latitude", "longitude"])
            .filter(pl.col("sample_id").is_in(list(sids)))
            .collect()
            .to_pandas()
            .dropna(subset=["latitude", "longitude"])
        )
        if ll.empty:
            print("[WARN] no catalog lat/lon for the sampled images; skipping geo.")
            return {}
        tr = gpd.read_file(TRACT_GEOJSON)[["geoid", "boroname", "ntaname", "geometry"]].copy()
        tr["geoid"] = tr["geoid"].astype(str)
        pts = gpd.GeoDataFrame(
            ll, geometry=gpd.points_from_xy(ll["longitude"], ll["latitude"]), crs=4326
        )
        j = gpd.sjoin(pts, tr, how="left", predicate="within")
        inc = pd.read_csv(INCOME_CSV, header=1, dtype=str, low_memory=False)
        g = inc["Geography"].str.extract(r"US(\d{11})$")[0]
        med = pd.to_numeric(
            inc["Estimate!!Households!!Median income (dollars)"], errors="coerce"
        )
        income = pd.DataFrame({"geoid": g, "median_income": med.values}).dropna(subset=["geoid"])
        j = j.merge(income, on="geoid", how="left")
        out: dict[str, dict] = {}
        for _, r in j.drop_duplicates("sample_id").iterrows():
            out[str(r["sample_id"])] = {
                "borough": None if pd.isna(r.get("boroname")) else str(r["boroname"]),
                "nta": None if pd.isna(r.get("ntaname")) else str(r["ntaname"]),
                "tract": None if pd.isna(r.get("geoid")) else str(r["geoid"]),
                "median_income": None if pd.isna(r.get("median_income")) else float(r["median_income"]),
                "lat": None if pd.isna(r.get("latitude")) else float(r["latitude"]),
                "lon": None if pd.isna(r.get("longitude")) else float(r["longitude"]),
            }
        print(f"Geo-enriched {len(out)}/{len(sids)} images (NTA / tract / median HH income).")
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] geo enrichment failed ({e}); skipping NTA/tract/income.")
        return {}


def _fmt_geo(sid: str | None, geo: dict[str, dict]) -> str:
    """One-line 'Borough · NTA · tract <geoid> · median HH income $X' for a sample."""
    g = geo.get(str(sid)) if geo else None
    if not g:
        return "—"
    inc = (
        f"${int(g['median_income']):,}"
        if g.get("median_income") is not None
        else "n/a"
    )
    return "  ·  ".join([
        g.get("borough") or "?",
        g.get("nta") or "?",
        f"tract {g['tract']}" if g.get("tract") else "tract ?",
        f"median HH income {inc}",
    ])


# Marker colors for the two presented images (also used on the minimap).
_A_COLOR = "#1f77b4"  # Image A / presented left
_B_COLOR = "#d62728"  # Image B / presented right


@lru_cache(maxsize=1)
def _city_basemap():
    """Borough outlines (dissolved from the 2020 tracts), EPSG:2263. Cached once
    per run; returns None if the geo stack/file is unavailable (minimap skipped)."""
    try:
        import sys

        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import geopandas as gpd

        tr = gpd.read_file(TRACT_GEOJSON)[["boroname", "geometry"]]
        return tr.dissolve(by="boroname").reset_index().to_crs(2263)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] minimap basemap unavailable ({e}); skipping minimaps.")
        return None


def _draw_minimap(fig, geo: dict[str, dict], left_sid, right_sid) -> None:
    """Small NYC map on the cover marking where Image A and Image B were taken."""
    base = _city_basemap()
    if base is None or not geo:
        return
    ga = geo.get(str(left_sid)) or {}
    gb = geo.get(str(right_sid)) or {}
    if ga.get("lat") is None or gb.get("lat") is None:
        return
    try:
        import geopandas as gpd

        pts = gpd.GeoDataFrame(
            {"lbl": ["A", "B"]},
            geometry=gpd.points_from_xy(
                [ga["lon"], gb["lon"]], [ga["lat"], gb["lat"]]
            ),
            crs=4326,
        ).to_crs(2263)
        ax = fig.add_axes([0.575, 0.05, 0.38, 0.34])
        base.plot(ax=ax, facecolor="#ececec", edgecolor="#9a9a9a", linewidth=0.5)
        # Rasterize the borough polygon (its merged coastline has ~100k vertices;
        # as vector it bloats every cover page to MBs). Do this before the markers
        # so the A/B points + connector stay crisp vector. rasterized artists are
        # flattened at save-dpi only in the PDF backend.
        for coll in ax.collections:
            coll.set_rasterized(True)
        xs = list(pts.geometry.x)
        ys = list(pts.geometry.y)
        ax.plot(xs, ys, color="#666", lw=0.8, ls="--", zorder=2)  # A–B connector
        for (x, y), lbl, col in zip(zip(xs, ys), ["A", "B"], [_A_COLOR, _B_COLOR]):
            ax.scatter([x], [y], s=70, c=col, edgecolor="white",
                       linewidth=0.9, zorder=3)
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 4),
                        fontsize=10, fontweight="bold", color=col, zorder=4)
        ax.set_title("where in NYC (A = left, B = right)", fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] minimap draw failed ({e}).")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]) if __doc__ else None,
    )
    p.add_argument("output_parquet", type=Path, help="Pairwise VQA stage output parquet.")
    p.add_argument("-n", "--n", type=int, default=12, help="Number of pairs to sample. Default: 12.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for the per-pair PDFs. "
        "Default: <parquet_dir>/<stem>.reasoning_pdfs/",
    )
    p.add_argument("--seed", type=int, default=1234, help="RNG seed for the sample. Default: 1234.")
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Header label printed on every PDF (e.g. dataset + model). "
        "Default: the parquet stem.",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Prompt YAML to embed verbatim (system + user_template). "
        "Default: auto-resolved from the parquet filename. "
        "Pass 'none' to skip the prompt page.",
    )
    p.add_argument(
        "--include-empty",
        action="store_true",
        help="Allow rows with an empty/missing reasoning trace into the sample.",
    )
    p.add_argument(
        "--decisive-only",
        action="store_true",
        help="Only sample pairs whose verdict is not 'Same'.",
    )
    p.add_argument(
        "--no-geo",
        action="store_true",
        help="Skip the per-image NTA / census tract / median-household-income lookup.",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Also write a single combined PDF with all sampled pairs.",
    )
    p.add_argument(
        "--max-image-px",
        type=int,
        default=1100,
        help="Downscale images so the longest side is at most this many px. Default: 1100.",
    )
    p.add_argument(
        "--wrap",
        type=int,
        default=100,
        help="Character width to wrap text at. Default: 100.",
    )
    p.add_argument(
        "--lines-per-page",
        type=int,
        default=58,
        help="Reasoning/text lines per page. Default: 58.",
    )
    return p.parse_args()


def _resolve_prompt_file(arg: Path | None, parquet: Path) -> Path | None:
    """Pick the prompt YAML: explicit arg, else keyword-match the parquet name."""
    if arg is not None:
        if str(arg).lower() == "none":
            return None
        if not arg.exists():
            raise SystemExit(f"--prompt-file not found: {arg}")
        return arg
    stem = parquet.stem.lower()
    for keyword, prompt_stem in PROMPT_BY_KEYWORD.items():
        if keyword in stem:
            candidate = PROMPT_DIR / f"{prompt_stem}.yaml"
            if candidate.exists():
                return candidate
    print(f"[WARN] could not auto-resolve a prompt config from '{parquet.stem}'; "
          "no prompt page will be rendered (pass --prompt-file to set one).")
    return None


def _load_prompt(path: Path) -> tuple[str, str]:
    """Return (system, user_template) verbatim from a prompt YAML."""
    cfg = yaml.safe_load(path.read_text()) or {}
    return str(cfg.get("system", "")).rstrip(), str(cfg.get("user_template", "")).rstrip()


def _parse_verdict(raw: str) -> tuple[str | None, str | None]:
    """Best-effort pull (answer_label, explanation) from a model_response JSON blob."""
    if not raw or not str(raw).strip():
        return None, None
    text = str(raw)
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if isinstance(obj, dict):
        label = obj.get("answer")
        expl = obj.get("explanation") or obj.get("reasoning") or obj.get("rationale")
        return (str(label) if label is not None else None,
                str(expl) if expl is not None else None)
    return None, None


def _load_image(path: str | None, max_px: int) -> Image.Image | None:
    if not path:
        return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    long_side = max(img.size)
    if long_side > max_px:
        scale = max_px / long_side
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    return img


def _interpret(label: str | None) -> str:
    """Human-readable gloss of the ordinal label (A relative to B)."""
    gloss = {
        "MuchLess": "strongly prefer Image B",
        "Less": "prefer Image B",
        "Same": "no preference (about equal)",
        "More": "prefer Image A",
        "MuchMore": "strongly prefer Image A",
    }
    return gloss.get(str(label), "—")


def _text_block_pages(pdf: PdfPages, header: str, body: str, wrap: int,
                      lines_per_page: int) -> None:
    """Render an arbitrary text body across as many monospace pages as needed."""
    raw_lines: list[str] = []
    for para in str(body).splitlines():
        if not para.strip():
            raw_lines.append("")
            continue
        raw_lines.extend(textwrap.wrap(para, width=wrap) or [""])
    if not raw_lines:
        raw_lines = ["(empty)"]

    total_pages = (len(raw_lines) + lines_per_page - 1) // lines_per_page
    for pi in range(total_pages):
        chunk = raw_lines[pi * lines_per_page:(pi + 1) * lines_per_page]
        suffix = f"  (p.{pi + 1}/{total_pages})" if total_pages > 1 else ""
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.06, 0.975, f"{header}{suffix}", fontsize=9, fontweight="bold", va="top")
        fig.text(0.06, 0.945, "\n".join(chunk), fontsize=7.5, family="monospace",
                 va="top", ha="left", linespacing=1.25)
        pdf.savefig(fig)
        plt.close(fig)


def _prompt_page(pdf: PdfPages, title: str, system: str, user_template: str,
                 wrap: int, lines_per_page: int) -> None:
    body = (
        "================ SYSTEM ================\n"
        f"{system}\n\n"
        "================ USER TEMPLATE ================\n"
        f"{user_template}\n"
    )
    _text_block_pages(pdf, f"{title}  —  exact prompt", body, wrap, lines_per_page)


def _cover_page(pdf: PdfPages, row: pd.Series, title: str, max_px: int,
                geo: dict[str, dict]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)

    left = _load_image(row.get("presented_left_path"), max_px)
    right = _load_image(row.get("presented_right_path"), max_px)
    # sample_id of the image actually shown on each side (respects is_swapped).
    left_sid, right_sid = _presented_sids(row)

    ax_l = fig.add_axes([0.04, 0.50, 0.44, 0.40])
    ax_r = fig.add_axes([0.52, 0.50, 0.44, 0.40])
    for ax, img, cap, sid, col in (
        (ax_l, left, "Image A  (presented left)", left_sid, _A_COLOR),
        (ax_r, right, "Image B  (presented right)", right_sid, _B_COLOR),
    ):
        ax.axis("off")
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "image unavailable", ha="center", va="center",
                    fontsize=10, color="crimson")
        ax.set_title(f"{cap}\n{sid}", fontsize=9, color=col)

    label = row.get("presented_label")
    vlabel, expl = _parse_verdict(row.get("model_response"))
    verdict = vlabel or label

    lines = [
        f"pair_id:          {row.get('pair_id')}",
        f"Image A (left):   {left_sid}",
        f"    {_fmt_geo(left_sid, geo)}",
        f"Image B (right):  {right_sid}",
        f"    {_fmt_geo(right_sid, geo)}",
        f"presented_order:  {row.get('presented_order')}    is_swapped: {row.get('is_swapped')}",
        f"presented_label:  {label}    relative_label: {row.get('relative_label')}",
        "",
        f"VERDICT:  {verdict}   ->   {_interpret(verdict)}",
    ]
    if expl:
        lines += ["", "Model explanation:", textwrap.fill(expl, width=104)]

    fig.text(0.04, 0.46, "\n".join(lines), fontsize=8.5, family="monospace",
             va="top", ha="left", wrap=True)
    fig.text(0.04, 0.015,
             "Exact prompt is on the prompt page (top of this file / report). "
             "Reasoning trace follows on the next page(s).",
             fontsize=7, style="italic", color="dimgray", va="bottom")
    _draw_minimap(fig, geo, left_sid, right_sid)
    pdf.savefig(fig)
    plt.close(fig)


def _render_pair(pdf: PdfPages, row: pd.Series, title: str,
                 args: argparse.Namespace, geo: dict[str, dict]) -> None:
    _cover_page(pdf, row, title, args.max_image_px, geo)
    _text_block_pages(pdf, f"{row.get('pair_id')}  —  reasoning trace",
                      row.get("model_reasoning") or "", args.wrap, args.lines_per_page)


def main() -> None:
    args = _parse_args()
    if not args.output_parquet.exists():
        raise SystemExit(f"Parquet not found: {args.output_parquet}")

    df = pd.read_parquet(args.output_parquet)
    if "model_reasoning" not in df.columns:
        raise SystemExit(f"{args.output_parquet} has no 'model_reasoning' column.")

    title = args.title or args.output_parquet.stem

    prompt_file = _resolve_prompt_file(args.prompt_file, args.output_parquet)
    system = user_template = ""
    if prompt_file is not None:
        system, user_template = _load_prompt(prompt_file)
        print(f"Embedding exact prompt from {prompt_file}")

    def _prompt(pdf: PdfPages) -> None:
        if prompt_file is not None:
            _prompt_page(pdf, title, system, user_template, args.wrap, args.lines_per_page)

    eligible = df
    if not args.include_empty:
        reasoning = df["model_reasoning"].fillna("").astype(str).str.strip()
        eligible = eligible[reasoning.str.len() > 0]
    if args.decisive_only and "presented_label" in eligible.columns:
        eligible = eligible[eligible["presented_label"].astype(str) != "Same"]
    if eligible.empty:
        raise SystemExit("No eligible rows after filtering "
                         "(try --include-empty / drop --decisive-only).")

    k = min(args.n, len(eligible))
    if k < args.n:
        print(f"[WARN] only {k} eligible rows (requested {args.n}).")
    sample = eligible.sample(n=k, random_state=args.seed).reset_index(drop=True)

    geo: dict[str, dict] = {} if args.no_geo else _load_geo_map(sample)

    out_dir = args.out or args.output_parquet.with_suffix("").with_name(
        f"{args.output_parquet.stem}.reasoning_pdfs"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in sample.iterrows():
        pid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(row.get("pair_id") or f"row{i}"))
        pdf_path = out_dir / f"{i:02d}_{pid}.pdf"
        with PdfPages(pdf_path) as pdf:
            _prompt(pdf)              # exact prompt once, at the top of the file
            _render_pair(pdf, row, title, args, geo)
        print(f"  wrote {pdf_path}")

    if args.combined:
        combined = out_dir / f"_combined_{k}.pdf"
        with PdfPages(combined) as pdf:
            _prompt(pdf)              # exact prompt once, at the very top of the report
            for _, row in sample.iterrows():
                _render_pair(pdf, row, title, args, geo)
        print(f"  wrote {combined}  (combined, {k} pairs)")

    print(f"Done: {k} pair PDF(s) -> {out_dir}")


if __name__ == "__main__":
    main()
