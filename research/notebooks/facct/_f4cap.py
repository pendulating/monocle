"""Helpers for the F4 caption notebook (`f4_captions.py`).

The caption prompt asks each model: "Look and write an Instagram caption for
what's in front of me." It runs on 2 image sets (`CASES`): `caption` shows a
restaurant storefront, and `street_caption` shows a street view (D133). The caption
analysis by NTA (`research/scripts/f4/caption_nta_analysis.py`, D101 to D108) gives 1
row for each (model, NTA) with the entropy of the caption words and of the
hashtags. This module reads that output, tests it against the ACS features
of the NTA, and gives the captions and hashtags of a group of NTAs.

Where the data is
-----------------
The data root comes from `_f4` (the pool by default, `F4_DATA=canonical` for
the canonical batch). The caption, hashtag and NTA files of a case are the
`inputs` of `<root>/<case dir>/nta_analysis/manifest.json`, thus the notebook
reads the same files as the entropies. A case is available when that
manifest exists (`available_cases`).

| Item | Rule |
|------|------|
| Caption | a quotable `caption` span of the x2 caption extraction; the texts with `copy_suspect` are dropped, as in the analysis script |
| Hashtag | 1 count for each tag in each answer (a looping answer repeats its tags) |
| Caption text | lower case, 1 space (`clean` of the analysis script) |
| Caption word | the `WORD` pattern of the analysis script |
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _f4 as F  # noqa: E402
import _style as S  # noqa: E402
import _traces as T  # noqa: E402

__version__ = "1.10.0"

# case -> (the folder of its caption analysis under the data root, the name of 1 image)
CASES = {"caption": ("captions", "restaurant"), "street_caption": ("street_captions", "street view")}
ALL = "all_models"
GROUPS = (*F.MODELS, ALL)
# The rarefied entropies: 1 sample size `k` for each model over all NTAs,
# thus the number of captions in an NTA does not drive the value.
MEASURES = {"word_entropy_rare": "caption words", "hashtag_entropy_rare": "hashtags"}
FEATURE_LABELS = {
    "pct_black_nh": "% Black, non-Hispanic",
    "pct_white_nh": "% white, non-Hispanic",
    "pct_asian_nh": "% Asian, non-Hispanic",
    "pct_bachelors": "% bachelor's degree",
    "median_household_income": "median household income",
}
KINDS = ("hashtags", "caption words", "captions")
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def _clean(text: object) -> str:
    return " ".join(str(text).lower().split())


def nta_dir(case: str = "caption") -> Path:
    """Give the folder of the caption analysis by NTA of `case`, under the data root."""
    return F.F4 / CASES[case][0] / "nta_analysis"


def available_cases() -> List[str]:
    """Give the cases whose caption analysis exists under the data root."""
    return [c for c in CASES if (nta_dir(c) / "manifest.json").exists()]


def manifest(case: str = "caption") -> dict:
    """Give the manifest of the caption analysis of `case`."""
    return json.loads((nta_dir(case) / "manifest.json").read_text())


def _input(stem: str, case: str = "caption") -> Path:
    """Give the input file of the caption analysis of `case` whose name holds `stem`."""
    for p in manifest(case)["inputs"]:
        if stem in Path(p).name:
            return F.REPO_ROOT / p
    raise FileNotFoundError(f"no input with {stem!r} in {nta_dir(case) / 'manifest.json'}")


def load_measures(case: str = "caption") -> pd.DataFrame:
    """Give the measures of each (model, NTA), with `all_models` for the 6 models together."""
    return pd.read_parquet(nta_dir(case) / "nta_measures.parquet")


def load_acs() -> pd.DataFrame:
    """Give the ACS features of the NTAs that have residents."""
    acs = pd.read_parquet(F.ACS_PATH)
    return acs[acs.population > 0].reset_index(drop=True)


def load_ntas(case: str = "caption") -> pd.DataFrame:
    """Give the NTA of each image: `image_idx`, `nta`, `NTAName`, `BoroName`, `place`.

    `place` is the restaurant name (`caption`) or the street view id
    (`street_caption`).
    """
    if case == "caption":
        n = pd.read_parquet(_input("restaurants_nta", case), columns=["image_idx", "NTA2020", "NTAName", "BoroName", "dba"])
        return n.rename(columns={"NTA2020": "nta", "dba": "place"})
    n = pd.read_parquet(_input("street_1k", case), columns=["image_idx", "nta2020", "nta_name", "borough", "sample_id"])
    return n.rename(columns={"nta2020": "nta", "nta_name": "NTAName", "borough": "BoroName", "sample_id": "place"})


def load_captions(case: str = "caption") -> pd.DataFrame:
    """Give 1 row for each caption, with the same filters as the analysis script.

    Columns: `model`, `doc_id`, `image_idx`, `nta`, `NTAName`, `BoroName`,
    `place`, `caption` (as written) and `text` (lower case, 1 space).
    """
    cap = pd.read_parquet(_input("captions", case))
    if "copy_suspect" in cap:
        cap = cap[~cap.copy_suspect.astype(bool)]
    cap = cap[(cap.extraction_class == "caption") & cap.is_quotable.fillna(False)]
    cap = cap[["model", "doc_id", "image_idx", "extraction_text"]].rename(columns={"extraction_text": "caption"})
    cap = cap.merge(load_ntas(case), on="image_idx", how="left", validate="many_to_one")
    cap["text"] = cap.caption.map(_clean)
    return cap.reset_index(drop=True)


def load_hashtags(case: str = "caption") -> pd.DataFrame:
    """Give 1 row for each (answer, tag), with the NTA. `tag` is the lower-case tag; `hashtag` is 1 written form."""
    tags = pd.read_parquet(_input("hashtags", case), columns=["doc_id", "model", "image_idx", "hashtag", "tag"])
    tags = tags.drop_duplicates(["doc_id", "tag"])
    return tags.merge(load_ntas(case), on="image_idx", how="left", validate="many_to_one").reset_index(drop=True)


def eligible_ntas(measures: pd.DataFrame, acs: pd.DataFrame, min_restaurants: int) -> pd.DataFrame:
    """Give the NTAs with residents and `min_restaurants` images or more, with their features.

    The column `restaurants` of the analysis counts the images of the NTA:
    restaurant storefronts, or street views for `street_caption`.
    """
    rest = measures[measures.model == ALL].set_index("nta")[["restaurants", "NTAName", "BoroName"]]
    rest = rest[rest.restaurants >= min_restaurants]
    out = acs.set_index("nta2020")[list(F.NTA_FEATURES)].join(rest, how="inner")
    out.index.name = "nta"
    return out


def entropy_correlations(measures: pd.DataFrame, acs: pd.DataFrame, min_restaurants: int = 20,
                         alpha: float = 0.05) -> pd.DataFrame:
    """Correlate the caption entropies of each model with the ACS features of the NTA.

    1 test for each (model, measure, feature): the Spearman correlation over
    the NTAs (`scipy.stats.spearmanr`), as test 2 of the traces notebook
    (section 4.10). The NTAs are those of `eligible_ntas`. `p_bonf` is `p`
    times the number of tests in the table (Bonferroni). `sign` is +1 or -1
    when `p_bonf` < `alpha`, else 0.
    """
    from scipy.stats import spearmanr

    feats = eligible_ntas(measures, acs, min_restaurants)
    rows = []
    for g in GROUPS:
        part = measures[measures.model == g].set_index("nta")
        for m in MEASURES:
            for f in F.NTA_FEATURES:
                both = pd.concat([part[m], feats[f]], axis=1, join="inner").dropna()
                rho, p = spearmanr(both[m], both[f]) if len(both) > 2 else (np.nan, np.nan)
                rows.append({"model": g, "measure": m, "feature": f, "ntas": len(both),
                             "rho": float(rho), "p": float(p)})
    out = pd.DataFrame(rows)
    out["p_bonf"] = (out.p * len(out)).clip(upper=1.0)
    out["sign"] = np.where(out.p_bonf < alpha, np.sign(out.rho), 0).astype(int)
    out["rho"] = out.rho.round(3)
    out.attrs.update(alpha=alpha, min_restaurants=min_restaurants, tests=len(out))
    return out


def plot_correlations(cor: pd.DataFrame, title: Optional[str] = None):
    """Draw the Spearman rho of each (model, feature), 1 panel for each measure.

    A cell with a significant test (`sign` not 0) shows its rho in bold with
    a star. Return the figure.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(MEASURES), figsize=(4.2 * len(MEASURES) + 1.2, 3.8), squeeze=False,
                             sharey=True, layout="constrained")
    for ax, m in zip(axes[0], MEASURES):
        t = cor[cor.measure == m]
        rho = t.pivot(index="model", columns="feature", values="rho").reindex(index=list(GROUPS),
                                                                               columns=list(F.NTA_FEATURES))
        sig = t.pivot(index="model", columns="feature", values="sign").reindex_like(rho)
        im = ax.imshow(rho.to_numpy(float), cmap=S.CMAP_DIV, vmin=-1, vmax=1, aspect="auto")
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                v = rho.iat[i, j]
                if pd.isna(v):
                    continue
                star = sig.iat[i, j] != 0
                ax.text(j, i, f"{v:+.2f}" + ("*" if star else ""), ha="center", va="center", fontsize=7,
                        fontweight="bold" if star else "normal", color=S.EDGE)
        ax.set_xticks(range(rho.shape[1]), [FEATURE_LABELS[f] for f in rho.columns], rotation=35, ha="right",
                      fontsize=7)
        ax.set_yticks(range(rho.shape[0]), [g.replace("_", " ") for g in rho.index], fontsize=7)
        ax.tick_params(axis="y", labelleft=ax is axes[0][0])
        ax.set_title(MEASURES[m], fontsize=9, color=S.EDGE)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="Spearman rho")
    if title:
        fig.suptitle(title, fontsize=10, color=S.EDGE)
    fig.patch.set_facecolor(S.PAPER)
    return fig


def plot_entropy_scatter(measures: pd.DataFrame, acs: pd.DataFrame, model: str, measure: str, feature: str,
                         min_restaurants: int = 20):
    """Draw the entropy of 1 model against 1 ACS feature, 1 dot for each NTA. Return the figure."""
    import matplotlib.pyplot as plt

    feats = eligible_ntas(measures, acs, min_restaurants)
    t = measures[measures.model == model].set_index("nta")[[measure]].join(feats[[feature]], how="inner").dropna()
    color = F.MODEL_COLORS.get(model, S.EDGE)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.scatter(t[feature], t[measure], s=14, color=color, edgecolor=S.EDGE, linewidth=0.3,
               marker=F.MODEL_MARKERS.get(model, "o"))
    ax.set_xlabel(FEATURE_LABELS[feature])
    ax.set_ylabel(f"rarefied entropy, {MEASURES[measure]} (bits)")
    ax.set_title(f"{model.replace('_', ' ')}: {len(t)} NTAs", fontsize=9, color=S.EDGE)
    fig.patch.set_facecolor(S.PAPER)
    fig.tight_layout()
    return fig


def nta_group(feats: pd.DataFrame, feature: str, side: str, pct: float) -> Tuple[pd.DataFrame, float]:
    """Give the NTAs at the `side` end ("bottom" or "top") of `feature`, `pct` percent of the NTAs.

    The percentile is over the NTAs of `feats` (each NTA counts 1, not its
    population). Return the group, sorted by the feature, and the cut value.
    """
    v = feats[feature].dropna()
    if side == "bottom":
        cut = float(v.quantile(pct / 100))
        g = feats.loc[v[v <= cut].index]
    else:
        cut = float(v.quantile(1 - pct / 100))
        g = feats.loc[v[v >= cut].index]
    return g.sort_values(feature, ascending=(side == "bottom")), cut


def _units(kind: str, captions: pd.DataFrame, hashtags: pd.DataFrame,
           stopwords: frozenset) -> pd.DataFrame:
    """Give 1 row for each (answer, item) of `kind`, with the NTA: columns `doc_id`, `nta`, `item`."""
    if kind == "hashtags":
        return hashtags[["doc_id", "nta", "tag"]].rename(columns={"tag": "item"}).assign(item=lambda d: "#" + d.item)
    if kind == "captions":
        return captions[["doc_id", "nta", "text"]].rename(columns={"text": "item"}).drop_duplicates()
    words = captions[["doc_id", "nta", "text"]].copy()
    words["item"] = words.text.map(lambda s: [w for w in WORD.findall(s) if w not in stopwords])
    return words.explode("item").dropna(subset=["item"])[["doc_id", "nta", "item"]].drop_duplicates()


def group_items(kind: str, captions: pd.DataFrame, hashtags: pd.DataFrame, group: Sequence[str],
                rest: Sequence[str], model: str = ALL, sort: str = "share", n: int = 30,
                stopwords: frozenset = frozenset(), min_count: int = 5, case: str = "caption") -> pd.DataFrame:
    """Compare the items (hashtags, caption words, or whole captions) of a group of NTAs with the rest.

    The unit is 1 caption answer: an item counts 1 time for each answer that
    holds it. The denominator of a share is the answers of the selected
    model(s) in the NTAs: images x models, because each model answers each
    image 1 time. `z` is the log-odds z of the group against the rest, with
    an informative Dirichlet prior (`_traces.distinctive_scores`, Monroe,
    Colaresi and Quinn 2008); it is empty for an item with fewer than
    `min_count` answers in total. `sort` is "share" (the most frequent items
    in the group) or "z" (the items that the group uses more than the rest).
    """
    cap = captions if model == ALL else captions[captions.model == model]
    tag = hashtags if model == ALL else hashtags[hashtags.model == model]
    u = _units(kind, cap, tag, stopwords)
    models = len(F.MODELS) if model == ALL else 1
    images = load_ntas(case).groupby("nta").size()
    n_group = int(images.reindex(list(group)).fillna(0).sum()) * models
    n_rest = int(images.reindex(list(rest)).fillna(0).sum()) * models
    in_g = u[u.nta.isin(group)].item.value_counts()
    in_r = u[u.nta.isin(rest)].item.value_counts()
    z = T.distinctive_scores(Counter(in_g.to_dict()), Counter(in_r.to_dict()), min_count=min_count)
    out = pd.DataFrame({"group_answers": in_g}).join(pd.DataFrame({"rest_answers": in_r}), how="outer").fillna(0)
    out = out[out.group_answers > 0].astype(int)
    out["group_pct"] = (100 * out.group_answers / max(n_group, 1)).round(2)
    out["rest_pct"] = (100 * out.rest_answers / max(n_rest, 1)).round(2)
    out["z"] = pd.Series(z).reindex(out.index).round(2)
    out = out.sort_values("z" if sort == "z" else "group_answers", ascending=False, na_position="last")
    out.index.name = kind.rstrip("s") if kind != "captions" else "caption"
    out.attrs.update(group_answers=n_group, rest_answers=n_rest)
    return out.head(n).reset_index()


def caption_sample(captions: pd.DataFrame, group: Sequence[str], model: str = ALL, n: int = 20,
                   seed: int = 0) -> pd.DataFrame:
    """Give `n` random captions of the group of NTAs, with the model, the NTA and the place of the image."""
    c = captions[captions.nta.isin(group)]
    if model != ALL:
        c = c[c.model == model]
    if c.empty:
        return pd.DataFrame(columns=["model", "NTAName", "BoroName", "place", "caption"])
    s = c.sample(min(n, len(c)), random_state=seed)
    return s[["model", "NTAName", "BoroName", "place", "caption"]].reset_index(drop=True)


# ── the hashtag map ───────────────────────────────────────────────────
NTA_SHAPES = F.REPO_ROOT / "data" / "geo" / "nynta2020_26b" / "nynta2020.shp"
CM_FONT = f"{S.CM_FONT_DIR}/cmunrm.otf"


def nta_top_tags(hashtags: pd.DataFrame, n: int = 3, min_count: int = 5, min_in_nta: int = 3,
                 unit: str = "image") -> pd.DataFrame:
    """Give the `n` most distinctive hashtags of each NTA: the highest log-odds z against the other NTAs.

    `unit` is "image" (the default) or "answer". With "image", a tag counts
    1 time for each image of the NTA, if 1 or more models write it. Thus a
    tag that 1 image gets from all 6 models counts 1, not 6. With "answer",
    the count is the answers (`load_hashtags` keeps 1 count for each tag in
    each answer). `z` is `_traces.distinctive_scores` of the NTA against all
    the other NTAs together (Monroe, Colaresi and Quinn 2008), on the same
    unit. A tag must have a positive z, `min_count` units in the city, and
    `min_in_nta` units in the NTA. An NTA can have fewer than `n` tags. The
    column `count` gives the units in the NTA, and `share` gives them as a
    fraction of the units of the NTA (images, or answers).
    """
    if unit not in ("image", "answer"):
        raise ValueError(f"unit must be 'image' or 'answer', not {unit!r}")
    if unit == "image":
        hashtags = hashtags.drop_duplicates(["nta", "image_idx", "tag"])
        size = hashtags.groupby("nta")["image_idx"].nunique()
    else:
        size = hashtags.groupby("nta")["doc_id"].nunique()
    counts = hashtags.groupby(["nta", "tag"]).size()
    city = Counter(hashtags["tag"].value_counts().to_dict())
    rows = []
    for nta, g in counts.groupby(level=0):
        here = Counter(g.droplevel(0).to_dict())
        z = T.distinctive_scores(here, city - here, min_count=min_count)
        top = sorted(((t, s) for t, s in z.items() if s > 0 and here[t] >= min_in_nta), key=lambda x: -x[1])[:n]
        rows += [{"nta": nta, "rank": i + 1, "tag": t, "z": s, "count": here[t], "share": here[t] / size[nta]}
                 for i, (t, s) in enumerate(top)]
    out = pd.DataFrame(rows)
    out.attrs.update(unit=unit, min_count=min_count, min_in_nta=min_in_nta,
                     models=sorted(hashtags["model"].unique()))
    return out


def nta_cartogram(has_tags: Sequence[bool], empty_weight: Optional[float] = 0.6, simplify_ft: float = 200.0,
                  min_part: float = 0.02, iters: int = 100, tol: float = 0.05, shapes=None,
                  weight: Optional[Sequence[float]] = None):
    """Give the NTA polygons warped so that each NTA with tags has the same area (a cartogram).

    The street case holds 50 or 51 images in each NTA, so equal area gives
    equal space to equal evidence. Manhattan grows and Staten Island shrinks.
    An NTA with no tags (`has_tags` False: parks, airports, cemeteries) gets
    `empty_weight` of that area. The method is Dougenik, Chrisman and
    Niemeyer (1985): each iteration moves every vertex by the same force
    field, so the shared edges stay shared. `shapely.coverage_simplify`
    first smooths the shores and edges by `simplify_ft`, and the islands
    smaller than `min_part` of the median NTA drop out. The loop stops when
    the mean area error is below `tol`, or after `iters` iterations.
    `shapes` gives other polygons (for example the community districts), and
    `weight` gives each polygon a relative area in place of 1 (the NTAs with
    tags in a district); `has_tags` and `empty_weight` still apply. With
    `empty_weight` None, a polygon with no tags has no force of its own: it
    only follows the warp of its neighbours. A large empty polygon (Jamaica
    Bay, JFK) then does not collapse into spikes.
    """
    import geopandas as gpd
    import shapely

    ntas = gpd.read_file(NTA_SHAPES) if shapes is None else shapes
    geoms = shapely.coverage_simplify(ntas.geometry.values, simplify_ft)
    floor = min_part * np.median(shapely.area(geoms))
    geoms = np.array([shapely.multipolygons([p for p in shapely.get_parts(g) if p.area >= floor]) or g
                      for g in geoms])
    base = np.ones(len(ntas)) if weight is None else np.asarray(weight, dtype=float)
    active = np.asarray(has_tags, dtype=bool) | (empty_weight is not None)
    weight = np.where(np.asarray(has_tags, dtype=bool), base, empty_weight or 0.0)
    total = shapely.area(geoms).sum()
    for _ in range(iters):
        area = shapely.area(geoms)
        want = area.copy()  # a passive polygon wants the area that it has: no force
        want[active] = (total - area[~active].sum()) * weight[active] / weight[active].sum()
        err = np.maximum(area, want) / np.minimum(area, want)
        if (err - 1).mean() < tol:
            break
        cx, cy = shapely.get_x(shapely.centroid(geoms)), shapely.get_y(shapely.centroid(geoms))
        radius = np.sqrt(area / np.pi)
        mass = np.sqrt(want / np.pi) - radius
        reduce = 1 / (1 + err.mean())
        xy = shapely.get_coordinates(geoms)
        dx, dy = xy[:, :1] - cx, xy[:, 1:] - cy
        d = np.hypot(dx, dy) + 1e-9
        force = np.where(d > radius, mass * radius / d, mass * (d / radius) ** 2 * (4 - 3 * d / radius))
        move = force * reduce / d
        geoms = shapely.set_coordinates(geoms, xy + np.c_[(move * dx).sum(1), (move * dy).sum(1)])
    # A warped edge can cross itself; the repair can then leave lines. Keep only the polygons.
    def polygonal(g):
        parts = [p for p in shapely.get_parts(shapely.get_parts(shapely.make_valid(g))) if p.geom_type == "Polygon"]
        return shapely.multipolygons([p for p in parts if p.area >= floor] or parts)

    geoms = [polygonal(g) for g in geoms]
    return ntas.set_geometry(gpd.GeoSeries(geoms, crs=ntas.crs)), float((err - 1).mean())


def _nta_mask(geom, x0: float, y1: float, scale: float):
    """Give the word-cloud mask of 1 NTA (0 inside, 255 outside) and its pixel offset on the map."""
    from PIL import Image, ImageDraw

    bx0, by0, bx1, by1 = geom.bounds
    ox, oy = int((bx0 - x0) * scale), int((y1 - by1) * scale)
    w, h = int((bx1 - bx0) * scale) + 2, int((by1 - by0) * scale) + 2
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)

    def ring(coords):
        return [((x - x0) * scale - ox, (y1 - y) * scale - oy) for x, y in coords]

    for poly in getattr(geom, "geoms", [geom]):
        draw.polygon(ring(poly.exterior.coords), fill=0)
        for hole in poly.interiors:
            draw.polygon(ring(hole.coords), fill=255)
    return np.array(img), (ox, oy)


def _cloud_words(wc, ox: int, oy: int) -> List[dict]:
    """Give the words that `wordcloud` placed, as the text start, baseline and size in canvas pixels.

    `wordcloud` draws each word with Pillow, with the top of the ink at the
    corner that it stores. Thus the baseline of a level word is 1 ink height
    (`top`) below the corner. A word turned by 90 degrees reads from the
    bottom up: its baseline is `top` right of the corner, and its text starts
    at the bottom of its box.
    """
    from PIL import Image, ImageDraw, ImageFont

    draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    out = []
    for (word, z), size, (row, col), turn, _ in wc.layout_:
        font = ImageFont.truetype(wc.font_path, size)
        top = -font.getbbox(word, anchor="ls")[1]
        x, y = col + ox, row + oy
        if turn is None:
            out.append({"word": word, "z": z, "px": size, "rotation": 0, "x": x, "y": y + top})
        else:
            box = draw.textbbox((0, 0), word, font=ImageFont.TransposedFont(font, orientation=turn))
            out.append({"word": word, "z": z, "px": size, "rotation": 90, "x": x + top, "y": y + box[3]})
    return out


def plot_tag_map(top: pd.DataFrame, ntas=None, width_px: int = 7000, width_in: float = 7.0, min_pt: float = 1.0,
                 max_pts: Sequence[float] = (11.0, 8.0, 6.0, 4.5, 3.5), seed: int = 20261029,
                 border: Optional[str] = None, tries: int = 1, floor: Optional[Dict[str, float]] = None,
                 ax=None, fill: Optional[pd.Series] = None, vmax: Optional[float] = None, prefix: str = "#",
                 ceiling: Optional[Dict[str, float]] = None, **fill_kw):
    """Draw a word cloud of the top hashtags inside each NTA, on 1 map of NYC.

    `wordcloud` fits the words of each NTA in a pixel mask of the polygon
    (`width_px` over the city), from the highest z down. A tag that does not
    fit at `min_pt` drops out, and the fit runs again without it. The fit
    runs for each largest size in `max_pts` and keeps the layout with the
    most tags (the larger size on a tie). If no layout fits all the tags,
    the fit tries more turned words and `tries` random placements, in the 3
    smallest sizes of `max_pts` only.
    Thus a large NTA shows all its tags in `top` and
    a small NTA shows fewer, in smaller type. The figure then draws each
    placed word as vector text (CMU Serif, the font of the fit), at the same
    place and size. Sizes are in points at `width_in` inches. `ntas` gives
    the polygons (for example `nta_cartogram`); the default is the NTA file.
    Their key is the column `NTA2020`. `border` draws the polygon edges in
    that colour; the default is a thin white line. `floor` gives a lower
    `min_pt` to some keys. With `ax`, the map goes in that axes, and
    `width_in` must be the width of the axes; the return is then the axes.
    `fill`, `vmax` and `fill_kw` colour the polygons (`draw_tag_map`); the fit
    does not use them. `prefix` comes before each word: "#" for the hashtags,
    "" for other words (for example `other_city_top`). `ceiling` gives a
    largest size to some keys (`count_ceiling`), thus a large polygon does
    not make its words large.
    """
    import geopandas as gpd
    from wordcloud import WordCloud

    ntas = gpd.read_file(NTA_SHAPES) if ntas is None else ntas
    x0, y0, x1, y1 = ntas.total_bounds
    scale = width_px / (x1 - x0)
    pt_per_px = width_in * 72 / width_px
    words, shown = [], {}
    for r in ntas[ntas["NTA2020"].isin(top["nta"])].itertuples():
        mask, (ox, oy) = _nta_mask(r.geometry, x0, y1, scale)
        shown[r.NTA2020] = 0
        lo = (floor or {}).get(r.NTA2020, min_pt)
        sizes = [m for m in max_pts if m >= lo] + ([lo] if lo < min(max_pts) else [])
        top_pt = (ceiling or {}).get(r.NTA2020)
        if top_pt is not None:
            sizes = [m for m in sizes if m <= top_pt] or [max(top_pt, lo)]
        if min(mask.shape) < 2 * lo / pt_per_px:
            continue
        freqs = {f"{prefix}{t.tag}": float(t.z) for t in top[top["nta"] == r.NTA2020].itertuples()}
        best = None
        # From the largest type down; then, in small type only, more turned words and other random placements.
        plan = [(0, 0.85, m) for m in sizes]
        plan += [(k, h, m) for k in range(tries) for h in (0.85, 0.5) for m in sizes[-3:] if (k, h) != (0, 0.85)]
        for k, turn, max_pt in plan:
            wc = WordCloud(font_path=CM_FONT, mask=mask, max_words=len(freqs), prefer_horizontal=turn,
                           min_font_size=int(lo / pt_per_px), max_font_size=int(max_pt / pt_per_px),
                           relative_scaling=0.6, margin=2, random_state=seed + k)
            todo = dict(freqs)
            while todo:
                wc.generate_from_frequencies(todo)
                if len(wc.layout_) == len(todo):
                    break
                # `wordcloud` stops at the first tag that does not fit: drop it and fit the rest.
                del todo[sorted(todo, key=lambda t: -todo[t])[len(wc.layout_)]]
            if best is None or len(wc.layout_) > len(best.layout_):
                best = wc
            if len(best.layout_) == len(freqs):
                break
        shown[r.NTA2020] = len(best.layout_)
        words += [dict(w, nta=r.NTA2020) for w in _cloud_words(best, ox, oy)]

    out = draw_tag_map(top, ntas, pd.DataFrame(words), scale, width_in, border=border, ax=ax, fill=fill, vmax=vmax,
                       **fill_kw)
    out._tag_counts = pd.Series(shown)  # the tags that fit in each NTA, for the caption
    out._min_pt, out._floor = min_pt, dict(floor or {})
    return out


def draw_tag_map(top: pd.DataFrame, ntas, words: pd.DataFrame, scale: float, width_in: float,
                 border: Optional[str] = None, ax=None, fill: Optional[pd.Series] = None,
                 vmax: Optional[float] = None, fill_label: str = "P(caption names its NTA or borough)",
                 fill_color: Optional[str] = None):
    """Draw the words of a `plot_tag_map` fit on the polygons `ntas`.

    `words` are in canvas pixels, and `scale` is the pixels for 1 map unit.
    Without `ax`, the figure is `width_in` wide and the axes fill it. With
    `ax`, `width_in` must be the width of the axes, and the axes box must be
    as tall as the map or taller (the extra height goes to the y limits).
    The return is the figure,
    or the axes, with the words and the points for 1 pixel.

    `fill` gives a value for each key of `ntas` (for example the column `p`
    of `place_rate`). The polygons then take the colour of `place_cmap`, from
    0 to `vmax` (the default is the largest value), and a colour bar with
    `fill_label` goes in the top left corner (New Jersey). A polygon with no
    value stays white. Give 1 `vmax` to maps that the reader compares.
    `fill_color` is the dark end of the ramp (`place_cmap`).
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import shapely
    from matplotlib.font_manager import FontProperties

    x0, y0, x1, y1 = ntas.total_bounds
    pt_per_px = width_in * 72 / (scale * (x1 - x0))
    if ax is None:
        # The axes fill the figure, so 1 canvas pixel is `pt_per_px` points on the page.
        fig = plt.figure(figsize=(width_in, width_in * (y1 - y0) / (x1 - x0)))
        ax = fig.add_axes((0, 0, 1, 1))
        out = fig
    else:
        out = ax
    has = ntas["NTA2020"].isin(top["nta"])
    if fill is None:
        ntas[has].plot(ax=ax, facecolor=S.PAL["warmgrey"], alpha=0.18, edgecolor="white", linewidth=0.3)
    else:
        from matplotlib.colors import Normalize

        value = ntas["NTA2020"].map(fill)
        norm = Normalize(0, float(value.max()) if vmax is None else vmax, clip=True)
        cmap = place_cmap(fill_color)
        ntas[value.notna()].plot(ax=ax, color=cmap(norm(value.dropna().to_numpy(float))), edgecolor="white",
                                 linewidth=0.3)
        cax = ax.inset_axes((0.03, 0.93, 0.28, 0.015))
        bar = plt.colorbar(plt.cm.ScalarMappable(norm, cmap), cax=cax, orientation="horizontal")
        bar.outline.set_linewidth(0.3)
        bar.ax.tick_params(labelsize=5, width=0.3, length=2, pad=1)
        bar.ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:g}%")
        bar.set_label(fill_label, fontsize=5.5, labelpad=2)
        bar.ax.xaxis.set_label_position("top")
    if border is not None:
        ntas.boundary.plot(ax=ax, color=border, linewidth=0.4)
    with plt.rc_context({"hatch.linewidth": 0.25, "hatch.color": S.PAL["warmgrey"]}):
        ntas[~has].plot(ax=ax, facecolor="none", edgecolor=S.PAL["warmgrey"], linewidth=0.2, hatch="//////")
    # The NTA edges do not meet exactly: close the gaps (20 ft) before the city outline.
    # The city has no holes, so the outline keeps only the outer ring of each part.
    city = shapely.get_parts(ntas.geometry.buffer(20).union_all().buffer(-20))
    city = city[shapely.area(city) >= 1e-4 * shapely.area(city).sum()]  # no scraps of a repaired edge
    city = gpd.GeoSeries(shapely.polygons(shapely.get_exterior_ring(city)), crs=ntas.crs)
    city.boundary.plot(ax=ax, color=S.EDGE, linewidth=0.5)
    font = FontProperties(fname=CM_FONT)
    for w in words.itertuples():
        ax.text(x0 + w.x / scale, y1 - w.y / scale, w.word, fontproperties=font,
                fontsize=w.px * pt_per_px, rotation=w.rotation, rotation_mode="anchor",
                ha="left", va="baseline", color=S.EDGE, zorder=3)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    out._words, out._pt_per_px, out._scale, out._width_in = words, pt_per_px, scale, width_in
    out._ntas = ntas
    return out


def redraw_tag_map(fig, top: pd.DataFrame, **kw):
    """Draw the words of a fitted map (`plot_tag_map`) again with no new fit, for example with an other `fill`.

    `kw` goes to `draw_tag_map`.
    """
    return draw_tag_map(top, fig._ntas, fig._words, fig._scale, fig._width_in, **kw)


def district_shapes(empty_weight: float = 0.6):
    """Give the community districts (CDTA) as polygons, keyed in `NTA2020` for `plot_tag_map`.

    `weight` is the relative area of a district on the cartogram: 1 for each
    of its NTAs with street views, and `empty_weight` for each of its other
    NTAs (parks, airports, cemeteries). `n_ntas` counts the NTAs with street views.
    """
    import geopandas as gpd

    ntas = gpd.read_file(NTA_SHAPES)
    street = ntas["NTA2020"].isin(load_ntas("street_caption")["nta"])
    ntas = ntas.assign(n_ntas=street.astype(int), weight=np.where(street, 1.0, empty_weight))
    cd = ntas.dissolve("CDTA2020", aggfunc={"n_ntas": "sum", "weight": "sum", "CDTAName": "first",
                                            "BoroName": "first"}).reset_index()
    # The NTA edges do not meet exactly: close the gaps (50 ft), or they stay as thin holes in a district.
    cd["geometry"] = cd.buffer(50).buffer(-50)
    return cd.rename(columns={"CDTA2020": "NTA2020"})


def district_hashtags(hashtags: pd.DataFrame) -> pd.DataFrame:
    """Give the hashtags with the community district (CDTA) of the image in the column `nta`."""
    import geopandas as gpd

    cd = gpd.read_file(NTA_SHAPES).set_index("NTA2020")["CDTA2020"]
    return hashtags.assign(nta=hashtags["nta"].map(cd))


def fit_district_map(top: pd.DataFrame, min_pt: float = 4.0, boost: float = 1.3, rounds: int = 14,
                     max_pts: Sequence[float] = (16, 13, 11, 9, 8, 7, 6, 5, 4.5, 4), border: Optional[str] = None,
                     tries: int = 4, floors: Sequence[float] = (3.5, 3.0, 2.5), width_in: float = 7.0,
                     ax=None, fill: Optional[pd.Series] = None, vmax: Optional[float] = None):
    """Draw the district cartogram so that every district shows all its tags.

    The first cartogram gives each district an area in proportion to its NTAs
    with street views (`district_shapes`). Each round then multiplies the
    weight of each district that shows fewer tags than it has by `boost`, and
    warps and fits again. The space is near full at `min_pt`, so the rounds
    can trade space between neighbours: the map keeps the round with the
    most tags. A district that is still short then gets the next lower floor
    of `floors`, until all its tags fit. The figure keeps the factor of each
    district in `fig._boost` and the lower floors in `fig._floor`. Sizes are
    in points at `width_in` inches. With `ax` (`width_in` wide), the map goes
    in that axes, and the return is the axes, with the same data. `fill` and
    `vmax` colour the districts of the final map (`draw_tag_map`).
    """
    import matplotlib.pyplot as plt

    cd = district_shapes()
    need = top.groupby("nta").size()
    factor = pd.Series(1.0, index=cd["NTA2020"])
    best = None
    for _ in range(rounds):
        warped, _err = nta_cartogram(cd["n_ntas"] > 0, empty_weight=None, shapes=cd,
                                     weight=cd["weight"] * factor.values)
        fig = plot_tag_map(top, ntas=warped, width_in=width_in, min_pt=min_pt, max_pts=max_pts, border=border,
                           tries=tries)
        shown = int(fig._tag_counts.sum())
        if best is None or shown > best[0]:
            if best is not None:
                plt.close(best[1])
            best = (shown, fig, warped, factor.copy())
        else:
            plt.close(fig)
        short = fig._tag_counts < need.reindex(fig._tag_counts.index)
        if not short.any():
            break
        factor[short[short].index] *= boost
    _, fig, warped, factor = best
    floor: Dict[str, float] = {}
    for lo in floors:
        short = fig._tag_counts < need.reindex(fig._tag_counts.index)
        if not short.any():
            break
        floor.update({k: lo for k in short[short].index})
        plt.close(fig)
        fig = plot_tag_map(top, ntas=warped, width_in=width_in, min_pt=min_pt, max_pts=max_pts, border=border,
                           tries=tries, floor=floor)
    if ax is not None or fill is not None:
        out = draw_tag_map(top, warped, fig._words, fig._scale, width_in, border=border, ax=ax, fill=fill, vmax=vmax)
        out._tag_counts, out._min_pt, out._floor = fig._tag_counts, fig._min_pt, fig._floor
        plt.close(fig)
        fig = out
    fig._boost = factor
    return fig


LAYOUT_KEYS = ("words", "ntas", "scale", "width_in", "tag_counts", "min_pt", "floor", "boost")


def _fit_layout(job: Tuple[pd.DataFrame, dict]) -> dict:
    """Fit 1 district map in a worker process, and give its layout (`LAYOUT_KEYS`), not the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top, kw = job
    fig = fit_district_map(top, **kw)
    out = {k: getattr(fig, "_" + k) for k in LAYOUT_KEYS}
    plt.close(fig)
    return out


def fit_model_district_maps(hashtags: pd.DataFrame, models: Sequence[str] = F.THINKING_MODELS, n: int = 5,
                            min_images: int = 5, workers: int = 4, **fit_kw) -> Dict[str, Tuple[pd.DataFrame, dict]]:
    """Fit 1 district map for each model, with the hashtags of that model only.

    The selection is `nta_top_tags` on the images (`min_images` in the
    district). Each fit is `fit_district_map` with `fit_kw` (give `width_in`
    as the panel width). The fits run in `workers` processes at the same
    time. The return maps each model to its tags and its layout, for
    `plot_model_district_maps`.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    cd = district_hashtags(hashtags)
    tops = {m: nta_top_tags(cd[cd["model"] == m], n=n, min_in_nta=min_images) for m in models}
    with ProcessPoolExecutor(min(workers, len(models)), mp_context=get_context("spawn")) as ex:
        layouts = list(ex.map(_fit_layout, [(tops[m], fit_kw) for m in models]))
    return {m: (tops[m], L) for m, L in zip(models, layouts)}


def plot_model_district_maps(fits: Dict[str, Tuple[pd.DataFrame, dict]], gap_in: float = 0.15,
                             title_pt: float = 9.0, border: Optional[str] = None,
                             fill: Optional[Dict[str, pd.Series]] = None, vmax: Optional[float] = None):
    """Draw the maps of `fit_model_district_maps` as small multiples, 2 in each row, with 1 title for each model.

    Each panel is as wide as its fit (`width_in`), thus the type has the
    sizes of the fit. The figure keeps each panel in `fig._panels`, with the
    data of the caption. `fill` gives the values of each model
    (`draw_tag_map`); all the panels use 1 `vmax`, the largest value by default.
    """
    import matplotlib.pyplot as plt

    if fill is not None and vmax is None:
        vmax = float(max(f.max() for f in fill.values()))
    panel = max(L["width_in"] for _, L in fits.values())
    aspect = max((lambda b: (b[3] - b[1]) / (b[2] - b[0]))(L["ntas"].total_bounds) for _, L in fits.values())
    rows = -(-len(fits) // 2)
    head = 2.0 * title_pt / 72
    width = 2 * panel + gap_in
    height = rows * (head + panel * aspect) + (rows - 1) * gap_in
    fig = plt.figure(figsize=(width, height))
    fig._panels = {}
    for i, (m, (top, L)) in enumerate(fits.items()):
        r, c = divmod(i, 2)
        top_in = r * (head + panel * aspect + gap_in) + head
        ax = fig.add_axes((c * (panel + gap_in) / width, 1 - (top_in + panel * aspect) / height,
                           L["width_in"] / width, panel * aspect / height))
        draw_tag_map(top, L["ntas"], L["words"], L["scale"], L["width_in"], border=border, ax=ax,
                     fill=None if fill is None else fill[m], vmax=vmax)
        ax._tag_counts, ax._min_pt, ax._floor, ax._boost = L["tag_counts"], L["min_pt"], L["floor"], L["boost"]
        ax.set_title(f"({'abcdefgh'[i]}) {m}", fontsize=title_pt, loc="left", pad=title_pt * 0.4)
        fig._panels[m] = ax
    return fig


def model_maps_caption(fits: Dict[str, Tuple[pd.DataFrame, dict]], fig) -> str:
    """Give the caption of `plot_model_district_maps`, with the counts of the tags that fit in each panel."""
    top0 = next(iter(fits.values()))[0]
    k = int(max(t["rank"].max() for t, _ in fits.values()))
    parts = []
    for i, (m, (top, _)) in enumerate(fits.items()):
        ax = fig._panels[m]
        shown = ax._tag_counts
        need = top.groupby("nta").size().reindex(shown.index)
        txt = (f"({'abcdefgh'[i]}) {m}: {int(shown.sum())} of {len(top)} tags fit, and {int((shown == need).sum())} "
               f"of {len(need)} districts show all their tags")
        if (ax._boost > 1).any():
            txt += f"; {int((ax._boost > 1).sum())} districts get more area (up to {ax._boost.max():.1f} times)"
        if ax._floor:
            txt += f"; {len(ax._floor)} districts use a lower floor (down to {min(ax._floor.values()):g} pt)"
        parts.append(txt)
    share = min(100 * t["share"].min() for t, _ in fits.values())
    return (f"The most distinctive hashtags of the street captions in each community district, for each model "
            f"alone: {len(fits)} maps, 1 for each model. For each district, up to {k} hashtags with the "
            f"highest log-odds z against the other districts (informative Dirichlet prior), from the answers of "
            f"that model only. The count is images. A tag must be on {top0.attrs['min_count']} images in the city "
            f"and {top0.attrs['min_in_nta']} images in the district; a tag that is shown is on {share:.1f}% or more "
            f"of the images of its district. The word size shows the z inside the district; sizes do not compare "
            f"across districts or maps. Each map is a cartogram (Dougenik, Chrisman and Niemeyer 1985): the area of "
            f"a district starts in proportion to its NTAs with street views, plus 0.6 of that area for each NTA "
            f"with no street views. A district whose tags do not fit at {ax._min_pt:g} pt gets more area, and then "
            f"a lower floor, so the 4 maps have different warps. Shores are simplified. " + ". ".join(parts) + ".")


def place_caption(rate: pd.DataFrame, unit: str = "NTA") -> str:
    """Give the caption sentences of the fill of a map, from `place_rate`."""
    us = "NTAs" if unit == "NTA" else "community districts"
    what = {"caption": "a caption", "answer": "an answer", "image": "an image"}[rate.attrs.get("unit", "caption")]
    alias = ", ".join(sorted({a for v in PLACE_ALIASES.values() for a in v} | {a for v in BORO_ALIASES.values()
                                                                                for a in v}))
    return (f" The fill shows the probability that {what} of the {unit} names its place: the NTA of the image "
            f"(the NTA name, each part of it between the hyphens, or an alias: {alias}), or its borough. "
            f"Hashtags do not count, and a caption that names an other place counts as no. The ramp is 1 hue, "
            f"with equal steps in OKLab. Over all {us}, {100 * rate['hits'].sum() / rate['n'].sum():.2f}% name "
            f"the place; the {unit} values run from {100 * rate['p'].min():.1f}% to {100 * rate['p'].max():.1f}%.")


def tag_map_caption(top: pd.DataFrame, fig, cartogram: bool = False, unit: str = "NTA",
                    rate: Optional[pd.DataFrame] = None) -> str:
    """Give the caption of `plot_tag_map`, with the counts of the tags that fit.

    `unit` is "NTA" or "district" (the community districts of `district_shapes`).
    `rate` (`place_rate`) adds the sentences of the fill.
    """
    shown = fig._tag_counts
    n = top.groupby("nta").size().reindex(shown.index)
    k = int(top["rank"].max())
    pt = fig._words["px"] * fig._pt_per_px
    u = unit if unit == "NTA" else "community district"
    us = "NTAs" if unit == "NTA" else "districts"
    if not cartogram:
        warp = ""
    elif unit == "NTA":
        warp = (" The map is a cartogram (Dougenik, Chrisman and Niemeyer 1985): each NTA with tags has the "
                "same area, because each holds 50 or 51 street views; an NTA with no tags has 0.6 of that area. "
                "Shores are simplified.")
    else:
        warp = (" The map is a cartogram (Dougenik, Chrisman and Niemeyer 1985): the area of a district is in "
                "proportion to its NTAs with street views (50 or 51 street views each), plus 0.6 of that area for "
                "each NTA with no street views (parks, airports, cemeteries). Shores are simplified.")
        boost = getattr(fig, "_boost", None)
        if boost is not None and (boost > 1).any():
            warp += (f" {int((boost > 1).sum())} districts get more area (up to {boost.max():.1f} times), "
                     f"so that all their tags fit.")
        lower = getattr(fig, "_floor", {})
        if lower:
            names = ", ".join(f"{k} ({v:g} pt)" for k, v in sorted(lower.items()))
            warp += f" The smallest type is lower in {len(lower)} districts, so that all their tags fit: {names}."
    models = top.attrs.get("models", F.MODELS)
    who = f"for the {len(models)} models together" if len(models) > 1 else f"from {models[0]} only"
    if top.attrs.get("unit", "answer") == "image":
        count = (f"The count is images: a tag counts 1 time for each image, if 1 or more of the models write it. "
                 f"A tag must be on {top.attrs['min_count']} images in the city and "
                 f"{top.attrs['min_in_nta']} images in the {u}")
    else:
        count = (f"The count is answers. A tag must be in {top.attrs.get('min_count', 5)} answers in the city and "
                 f"{top.attrs.get('min_in_nta', 3)} answers in the {u}")
    if "share" in top:
        what = "images" if top.attrs.get("unit") == "image" else "answers"
        count += f"; a tag that is shown is in {100 * top['share'].min():.1f}% or more of the {what} of its {u}"
    return (f"The most distinctive hashtags of the street captions in each {u}, {who}. "
            f"For each {u}, up to {k} hashtags with the highest log-odds z against the other {us} "
            f"(informative Dirichlet prior). {count}. "
            f"The word size shows the z inside the {u}; sizes do not compare across {us}. "
            f"A tag that does not fit in its {u} at {getattr(fig, '_min_pt', pt.min()):g} pt drops out: "
            f"{int(shown.sum())} of the "
            f"{len(top)} tags fit, and {int((shown == n).sum())} of the {len(n)} {us} show all {k}. "
            f"Hatched {us} have no tag that passes the filter." + warp
            + ("" if rate is None else place_caption(rate, unit)))


# ── the place names in the captions ──────────────────────────────────
# A caption names its place when it names the NTA of the image, or the borough
# of that NTA. The names of an NTA are its NTAName without the "(North)" part,
# each part between the hyphens (the parts of HYPHENATED stay whole), and
# PLACE_ALIASES. "NYC" and "New York" name the city, not a place: they do not
# count. A name never comes from a name with its compass word removed, because
# "East New York" then gives "New York".
HYPHENATED = ("Bedford-Stuyvesant", "Co-op City")
PLACE_ALIASES = {
    "Bedford-Stuyvesant": ("bed-stuy",),
    "Downtown Brooklyn": ("downtown bk",),
    "East Williamsburg": ("williamsburg",),
    "South Williamsburg": ("williamsburg",),
    "East Flatbush": ("flatbush",),
    "East Harlem": ("harlem", "spanish harlem", "el barrio"),
    "West Harlem": ("harlem",),
    "Harlem": ("harlem",),
    "East Midtown": ("midtown",),
    "Midtown South": ("midtown",),
    "Financial District": ("fidi",),
    "Upper East Side": ("ues",),
    "Upper West Side": ("uws",),
    "Long Island City": ("lic",),
    "Old Astoria": ("astoria",),
}
# The hashtag pattern of `research/scripts/f4/caption_hashtags.py`.
HASHTAG = re.compile(r"(?<![\w&])#(?=\w*[^\W\d_])\w+")
BORO_ALIASES = {"Brooklyn": ("bk", "bklyn"), "Bronx": ("bx",), "Staten Island": ("staten",)}
# The ramp of the place map: 1 hue (teal), from light to dark. The steps are
# equal in OKLab, thus equal steps in the value look equal. The dark end stays
# light enough for the black words of the map.
PLACE_RAMP_L = (0.96, 0.58)


def _name_parts(name: str) -> List[str]:
    """Give the names in 1 NTAName: the whole name and each part between the hyphens."""
    name = re.sub(r"\s*\([^)]*\)", "", name)
    keep = {h.replace("-", "\0"): h for h in HYPHENATED}
    for k, h in keep.items():
        name = name.replace(h, k)
    parts = [p.strip().replace("\0", "-") for p in name.split("-")]
    whole = "-".join(parts)
    return list(dict.fromkeys([whole, *parts]))


def _name_pattern(names: Sequence[str]) -> re.Pattern:
    """Give 1 regex for `names`: whole words, a space or hyphen optional, the apostrophe optional."""
    alts = []
    for n in sorted(set(n.lower() for n in names), key=len, reverse=True):
        p = re.escape(n).replace(r"\ ", " ").replace(r"\-", "-").replace(r"\.", ".")
        p = re.sub(r"[ -]", r"[\\s-]?", p)
        p = p.replace("'", "['’]?").replace("st.", r"(?:st\.?|saint)")
        alts.append(p)
    return re.compile(r"(?<![\w#])(?:" + "|".join(alts) + r")(?!\w)")


def place_names(ntas: pd.DataFrame) -> pd.DataFrame:
    """Give the names of each NTA and of its borough: columns `nta`, `nta_names`, `boro_names`.

    `ntas` has the columns `nta`, `NTAName` and `BoroName` (`load_ntas`).
    """
    rows = []
    for r in ntas.drop_duplicates("nta").itertuples():
        names = _name_parts(r.NTAName)
        names += [a for n in names for a in PLACE_ALIASES.get(n, ())]
        rows.append({"nta": r.nta, "nta_names": list(dict.fromkeys(names)),
                     "boro_names": [r.BoroName, *BORO_ALIASES.get(r.BoroName, ())]})
    return pd.DataFrame(rows)


def flag_place_names(captions: pd.DataFrame) -> pd.DataFrame:
    """Add to each caption the flags `names_nta`, `names_boro` and `names_place` (either one).

    The test reads the caption text without its hashtags, thus `#brooklyn`
    does not count. The names are those of the NTA of the image
    (`place_names`): a caption that names an other place gets False.
    """
    text = captions["text"].map(lambda s: HASHTAG.sub(" ", s).replace("’", "'"))
    names = place_names(captions[["nta", "NTAName", "BoroName"]]).set_index("nta")
    out = captions.copy()
    for col, src in (("names_nta", "nta_names"), ("names_boro", "boro_names")):
        pats = {k: _name_pattern(v) for k, v in names[src].items()}
        out[col] = [bool(pats[k].search(t)) for k, t in zip(captions["nta"], text)]
    out["names_place"] = out["names_nta"] | out["names_boro"]
    return out


def place_rate(flagged: pd.DataFrame, unit: str = "caption", model: str = ALL) -> pd.DataFrame:
    """Give the probability that a caption of each NTA names its place, from `flag_place_names`.

    `unit` is "caption" (the default: each caption span counts 1), "answer"
    (an answer counts 1 if 1 or more of its captions name the place) or
    "image" (the same for the answers of all the models on 1 image). The
    group key is the column `nta`: give `district_hashtags(flagged)` for the
    community districts. `p` is `hits / n`; `p_nta` and `p_boro` split it.
    """
    f = flagged if model == ALL else flagged[flagged["model"] == model]
    cols = ["names_nta", "names_boro", "names_place"]
    if unit == "answer":
        f = f.groupby(["nta", "doc_id"])[cols].any().reset_index()
    elif unit == "image":
        f = f.groupby(["nta", "image_idx"])[cols].any().reset_index()
    elif unit != "caption":
        raise ValueError(f"unit must be 'caption', 'answer' or 'image', not {unit!r}")
    g = f.groupby("nta")
    out = pd.DataFrame({"n": g.size(), "hits": g["names_place"].sum(), "p": g["names_place"].mean(),
                        "p_nta": g["names_nta"].mean(), "p_boro": g["names_boro"].mean()})
    out.attrs.update(unit=unit, model=model)
    return out


# ── the city names in the captions ───────────────────────────────────
# All the street views are in NYC. A caption names the city when it names NYC
# (CITY_NAMES); "east new york" (an NTA) and "new york ave" (a street) do not
# count. The borough and the NTA names do not count here (`flag_place_names`).
CITY_NAMES = ("nyc", "new york city", "new york", "ny", "big apple", "gotham", "the city that never sleeps")
CITY_PATTERN = re.compile(r"(?<![\w#])(?<!east )(?:" + "|".join(re.escape(n) for n in CITY_NAMES)
                          + r")(?![\w])(?! ?(?:ave|avenue)\b)")
# Other places that a caption can give as the place of a street view. A
# region outside NYC (New Jersey, Connecticut) counts; Long Island does not,
# because Brooklyn and Queens are on it. The list comes from a gazetteer
# search of the street captions (geonamescache, cities over 100,000), with the
# hits checked by hand. Each name maps to 1 label. Amsterdam (Amsterdam Ave)
# and Providence (a common word) are not in the list. "sf" after a number is
# square feet.
OTHER_CITIES = {
    "chicago": "Chicago", "chi-town": "Chicago", "chitown": "Chicago",
    "boston": "Boston", "philadelphia": "Philadelphia", "philly": "Philadelphia",
    "baltimore": "Baltimore", "washington dc": "Washington DC", "washington d.c.": "Washington DC",
    "dc": "Washington DC", "d.c.": "Washington DC", "pittsburgh": "Pittsburgh", "toronto": "Toronto",
    "montreal": "Montreal", "jersey city": "Jersey City", "hoboken": "Hoboken", "newark": "Newark",
    "new jersey": "New Jersey", "connecticut": "Connecticut", "albany": "Albany",
    "los angeles": "Los Angeles", "san francisco": "San Francisco", "sf": "San Francisco",
    "seattle": "Seattle", "portland": "Portland", "miami": "Miami", "atlanta": "Atlanta",
    "detroit": "Detroit", "houston": "Houston", "dallas": "Dallas", "austin": "Austin",
    "new orleans": "New Orleans", "las vegas": "Las Vegas", "london": "London", "paris": "Paris",
    "berlin": "Berlin", "rome": "Rome", "stockholm": "Stockholm",
    "copenhagen": "Copenhagen", "seoul": "Seoul", "tokyo": "Tokyo", "havana": "Havana",
}
# A name that a sign or a street gives is not a claim about the place: the
# name must not come before 1 of these words. Each pattern below is also not a claim.
NAME_OF_THING = ("ave", "avenue", "st", "street", "blvd", "rd", "road", "pl", "place", "loop", "restaurant",
                 "bakery", "grill", "deli", "pizza", "fashion", "plane", "university", "college", "health",
                 "hall", "burger", "burgers", "wings", "bound", "elevators", "works", "style", "styles",
                 "market", "nice", "skyline", "&", "and")
NOT_A_CLAIM = re.compile(
    r"rome wasn'?t built|roads lead to rome|not (?:quite |exactly |really )?(?:in )?(?:{n})(?!\w)"
    r"|/\s*(?:{n})|(?:{n})\s*/|(?:{n}) or |\bor (?:{n})|a (?:little )?(?:slice|piece|taste) of|a little '?(?:{n})"
    r"|manifest|trip to|coming for you|pretend|squint|next stop|towards? (?:{n})|corner of (?:{n})"
    r"|new york, (?:{n})|alone london|e\.g\.|\[insert|\[your".format(n="|".join(re.escape(k) for k in OTHER_CITIES)))
OTHER_PATTERN = re.compile(r"(?<![\w#-])(?<!\d )(?<!\dk )(" + "|".join(
    re.escape(k) for k in sorted(OTHER_CITIES, key=len, reverse=True))
                           + r")(?![\w-])(?!(?:'s)? (?:" + "|".join(re.escape(w) for w in NAME_OF_THING) + r")(?!\w))")


def flag_city_names(captions: pd.DataFrame) -> pd.DataFrame:
    """Add to each caption `names_city` (it names NYC), `names_other` (it gives an other city) and `other`.

    `other` is the label of the first other city (`OTHER_CITIES`), or None.
    A caption that has a NOT_A_CLAIM pattern ("not quite Paris", "nyc/philly",
    "a slice of Seoul") gets no other city. The test reads the caption text
    without its hashtags. A caption can name NYC and an other city.
    """
    text = captions["text"].map(lambda s: HASHTAG.sub(" ", s).replace("’", "'"))
    out = captions.copy()
    out["names_city"] = [bool(CITY_PATTERN.search(t)) for t in text]
    other = [None if NOT_A_CLAIM.search(t) else OTHER_PATTERN.search(t) for t in text]
    out["other"] = [OTHER_CITIES[m.group(1)] if m else None for m in other]
    out["names_other"] = out["other"].notna()
    return out


def other_city_top(flagged: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Give the other cities of each NTA in the form of `nta_top_tags`, for `plot_tag_map(..., prefix="")`.

    The count is images: a city counts 1 time for each image of the NTA, if 1
    or more captions on it give that city. `z` is the count, thus the word
    size shows the count. Up to `n` cities for each NTA, the most frequent
    first; a tie goes in the order of the name.
    """
    o = flagged[flagged["names_other"]].drop_duplicates(["nta", "image_idx", "other"])
    size = flagged.groupby("nta")["image_idx"].nunique()
    c = o.groupby(["nta", "other"]).size().rename("count").reset_index() \
        .sort_values(["nta", "count", "other"], ascending=[True, False, True])
    c["rank"] = c.groupby("nta").cumcount() + 1
    c = c[c["rank"] <= n].rename(columns={"other": "tag"})
    out = c.assign(z=c["count"].astype(float), share=c["count"] / c["nta"].map(size))
    out = out[["nta", "rank", "tag", "z", "count", "share"]].reset_index(drop=True)
    out.attrs.update(unit="image", models=sorted(flagged["model"].unique()))
    return out


def count_ceiling(top: pd.DataFrame, pt_1: float = 4.5, step: float = 2.0) -> Dict[str, float]:
    """Give the largest type size of each key for `plot_tag_map`: `pt_1` for a top count of 1, `step` more for each 1 more.

    Thus the size of the largest word shows the count of the key on the
    whole map, not the area of the polygon.
    """
    return {k: pt_1 + step * (c - 1) for k, c in top.groupby("nta")["count"].max().items()}


def other_city_map_caption(top: pd.DataFrame, fig, rate: pd.DataFrame, flagged: pd.DataFrame) -> str:
    """Give the caption of the other-city map: `plot_tag_map` of `other_city_top`, filled by `rate`."""
    shown = fig._tag_counts
    need = top.groupby("nta").size().reindex(shown.index)
    return (f"The other cities that the street captions give, in each NTA, for the "
            f"{len(top.attrs['models'])} models together. All the images are in NYC. In each NTA, up to "
            f"{int(top['rank'].max())} cities, the most frequent first. The word size shows the number of images "
            f"with 1 or more captions that give the city: the largest word of an NTA is 4.5 pt for 1 image and 2 pt "
            f"more for each 1 more, or smaller if it does not fit. {int(shown.sum())} of the {len(top)} names fit, and "
            f"{int((shown == need).sum())} of the {len(need)} NTAs show all their cities. Hatched NTAs have no "
            f"caption that gives an other city." + city_caption(rate, flagged))


def flag_rate(flagged: pd.DataFrame, col: str, unit: str = "caption", model: str = ALL) -> pd.DataFrame:
    """Give the probability that a caption of each NTA has the flag `col`: columns `n`, `hits`, `p`.

    `unit` and `model` are as in `place_rate`. The group key is the column `nta`.
    """
    f = flagged if model == ALL else flagged[flagged["model"] == model]
    if unit in ("answer", "image"):
        f = f.groupby(["nta", "doc_id" if unit == "answer" else "image_idx"])[[col]].any().reset_index()
    elif unit != "caption":
        raise ValueError(f"unit must be 'caption', 'answer' or 'image', not {unit!r}")
    g = f.groupby("nta")[col]
    out = pd.DataFrame({"n": g.size(), "hits": g.sum(), "p": g.mean()})
    out.attrs.update(unit=unit, model=model, flag=col)
    return out


def other_cities(flagged: pd.DataFrame) -> pd.DataFrame:
    """Give the other cities of each NTA, from `flag_city_names`: columns `nta`, `other`, `captions`."""
    o = flagged[flagged["names_other"]]
    return o.groupby(["nta", "other"]).size().rename("captions").reset_index() \
        .sort_values(["nta", "captions"], ascending=[True, False])


def city_caption(rate: pd.DataFrame, flagged: pd.DataFrame, unit: str = "NTA") -> str:
    """Give the caption sentences of a city fill, from `flag_rate` on `names_city` or `names_other`."""
    us = "NTAs" if unit == "NTA" else "community districts"
    what = {"caption": "a caption", "answer": "an answer has a caption that",
            "image": "an image gets a caption that"}[rate.attrs["unit"]]
    if rate.attrs["flag"] == "names_city":
        claim = (f"names the city: {', '.join(repr(n) for n in CITY_NAMES)} (not \"east new york\" or "
                 f"\"new york ave\"). Borough and NTA names do not count")
    else:
        top = flagged.loc[flagged["names_other"], "other"].value_counts()
        claim = (f"gives an other city as the place of the street view ({len(OTHER_CITIES)} names; all the images "
                 f"are in NYC). A name that comes before a street or shop word (\"boston rd\", \"chicago's pizza\"), "
                 f"a hedge or a comparison (\"nyc or philly\", \"not quite paris\") and a template blank "
                 f"(\"[insert city, e.g., philly]\") do not count. The most frequent: "
                 + ", ".join(f"{k} ({v})" for k, v in top.head(5).items()) + " captions")
    what = what.replace(" has a", f" of the {unit} has a").replace(" gets a", f" of the {unit} gets a") \
        if rate.attrs["unit"] != "caption" else f"{what} of the {unit}"
    return (f" The fill shows the probability that {what} {claim}. Hashtags do not count. "
            f"{int(rate['hits'].sum())} of {int(rate['n'].sum())} have the flag, and {int((rate['hits'] == 0).sum())} "
            f"of {len(rate)} {us} have none. The ramp is 1 hue, with equal steps in OKLab.")


def place_cmap(color: Optional[str] = None, n: int = 256):
    """Give a sequential ramp of 1 hue, with equal steps in OKLab (Ottosson 2020): light to dark.

    The lightness runs over `PLACE_RAMP_L`, and the chroma grows from 0 to
    the chroma of `color` (the default is the teal of the palette).
    """
    from matplotlib.colors import ListedColormap, to_rgb

    def lin(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def gamma(c):
        c = np.clip(c, 0, 1)
        return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)

    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    _, a, b = m2 @ np.cbrt(m1 @ lin(np.array(to_rgb(color or S.PAL["teal"]))))
    t = np.linspace(0, 1, n)[:, None]
    lab = np.c_[PLACE_RAMP_L[0] + t[:, 0] * (PLACE_RAMP_L[1] - PLACE_RAMP_L[0]), t * a, t * b]
    rgb = gamma((np.linalg.inv(m1) @ (np.linalg.inv(m2) @ lab.T) ** 3).T)
    return ListedColormap(rgb, name="place")


def place_vmax(*rates: pd.DataFrame, q: float = 1.0) -> float:
    """Give 1 top for the ramp of 1 or more maps: the `q` quantile of `p`, rounded up to 0.5%."""
    p = pd.concat([r["p"] for r in rates])
    return float(np.ceil(p.quantile(q) * 200) / 200) or 0.005


def save_figure(fig, name: str, title: str, caption: str, tags: Sequence[str]) -> List[Path]:
    """Write 1 figure as PDF and PNG, with its caption data as JSON (~/style/proper-plotting.md)."""
    out = F.REPO_ROOT / "research" / "notebooks" / "facct" / "figures" / F.FIGURES
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", dpi=300)
    meta = {"plot-title": title, "plot-caption": caption, "plot-latex-label": "fig:" + name.replace("_", "-"),
            "plot-tags": list(tags)}
    paths.append(out / f"{name}.json")
    paths[-1].write_text(json.dumps(meta, indent=2) + "\n")
    return paths
