"""Does more reasoning make the model more right?

The IC report says how complex each trace is. The validation notebooks say
whether the model agrees with an outside measurement. This module joins the 2,
at the level of a single PAIR, and asks whether the traces the model reasoned
hardest about are the ones it got right.

The join, and why it is where it is
-----------------------------------
There is no per-unit proxy. `_proxies` puts each proxy on the map and
aggregates it to a polygon, and `_geography` does the same for the model units.
The 2 sides meet at the POLYGON, thus the ground truth of a pair is:

    the polygon of unit A holds a higher proxy value than the polygon of unit B

A pair whose 2 units sit in 1 polygon carries no proxy difference, thus it
drops out. So does an abstention, and so does `Same`.

    proxy_gap  = proxy(polygon of A) - proxy(polygon of B)
    model_dir  = sign of `relative_score`, which already states A against B
    agrees     = the 2 signs match

Warning: this is a coarse target. A unit can differ from its polygon, thus a
pair can be marked wrong when the model read the 2 buildings correctly and the
neighbourhoods disagree. Read the agreement rate as a floor.

The confound you must not skip
------------------------------
A model reasons longest about a HARD pair, and a hard pair is one where the 2
areas sit close together. Complexity and difficulty therefore move together,
and a raw comparison of "code 5 against code 3" measures the difficulty, not
the reasoning. Thus every table here splits on the size of the proxy gap.
`agreement_by_gap` is the number to read; `agreement_by_code` alone is not.

Warning: the code and the judgment come from the SAME run, thus this shows an
association and never a cause.

See `vlm-narratives-docs/ic-ingredient-extraction.md`.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import _canonical as C
import _geography as geo
import _ic
import _style as S

__version__ = "1.0.0"

HERE = Path(__file__).resolve().parent
REPO_ROOT = geo.REPO_ROOT

# case -> (prompt folder, export prefix, curation root, unit table)
#
# The unit table is NOT always `facilities.parquet`. The subway case keeps its
# entrances in their own file, and the restaurants case keeps 1 row for each
# inspected establishment. These names come from the `UNIT_TABLE` of each
# validation notebook and must stay in step with it.
#
# Only a UNIT-mode case is here. Road quality and street photography pair
# random images, thus their pairs manifest holds no `unit_uid`, and a pair has
# no unit to place in a polygon. They need the camera position instead, which
# is a different join and is not in this version.
UNIT_CASES: Dict[str, Tuple[str, str, str, str]] = {
    "subway_safety": ("subway", "subway", "curation/subway_entrances_all",
                      "entrances.parquet"),
    "libraries": ("libraries", "libraries", "curation/facdb_libraries",
                  "facilities.parquet"),
    "schools": ("schools", "schools", "curation/facdb_schools_k_12",
                "facilities.parquet"),
    "parks_plazas": ("parks", "parks", "curation/facdb_parks_plazas",
                     "facilities.parquet"),
    "restaurants": ("restaurants", "restaurants",
                    "curation/dohmh_restaurants_inspected_all",
                    "restaurants_aggregated.parquet"),
}

# The 2 image-mode cases. A pair here is 2 random street-level shots, thus
# there is no unit and no FacDB position. The CAMERA position of each shot goes
# into the polygon instead, which the pairs manifest already holds.
#
# Note: `street_photography` has no outside measurement, thus it holds no proxy
# export and it reaches this test only if one is ever written for it. The entry
# stays so the case is named rather than silently absent.
IMAGE_CASES: Dict[str, Tuple[str, str]] = {
    "road_quality": ("road", "road_quality"),
    "street_photography": ("street_photography", "street_photography"),
}

DEFAULT_LAYER = "community_district"


def _newest(pattern: str) -> Optional[str]:
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def proxy_by_key(case: str, layer: str = DEFAULT_LAYER,
                 proxy: Optional[str] = None) -> pd.DataFrame:
    """Read the proxy the case notebook exported, for 1 layer and 1 proxy.

    The values arrive oriented "higher is better": `_proxies` applies the sign
    before it exports, thus crime density is already negated here.
    """
    folder, prefix = _folder_prefix(case)
    path = _newest(str(HERE / folder / "outputs" / f"{prefix}_proxy_v*.parquet"))
    if not path:
        raise FileNotFoundError(
            f"{case}: no proxy export. Run {folder}/{prefix}_validation.py first.")
    df = pd.read_parquet(path)
    df = df[df["layer"] == layer]
    if "proxy" in df.columns:
        names = sorted(df["proxy"].dropna().unique())
        proxy = proxy or names[0]
        df = df[df["proxy"] == proxy]
    else:
        proxy = proxy or prefix
    key = geo.LAYERS[layer]["key"]
    out = df[[key, "proxy_mean"]].dropna().rename(columns={key: "poly_key"})
    out["proxy"] = proxy
    return out.drop_duplicates("poly_key").reset_index(drop=True)


def _folder_prefix(case: str) -> Tuple[str, str]:
    """The prompt folder and the export prefix of a case, whatever its mode."""
    if case in UNIT_CASES:
        folder, prefix, _, _ = UNIT_CASES[case]
        return folder, prefix
    if case in IMAGE_CASES:
        return IMAGE_CASES[case]
    raise KeyError(f"{case}: not a case this module can link")


def linkable_cases() -> List[str]:
    """Every case the link can reach, unit mode first."""
    return list(UNIT_CASES) + list(IMAGE_CASES)


def proxies_for(case: str) -> List[str]:
    """Name every proxy the case exported. Several cases hold more than 1."""
    folder, prefix = _folder_prefix(case)
    path = _newest(str(HERE / folder / "outputs" / f"{prefix}_proxy_v*.parquet"))
    if not path:
        return []
    df = pd.read_parquet(path)
    if "proxy" in df.columns:
        return sorted(df["proxy"].dropna().unique())
    return [prefix]


def unit_polygons(case: str, layer: str = DEFAULT_LAYER) -> pd.DataFrame:
    """Give each unit the polygon it sits in.

    The position comes from the FacDB facility table, NOT from the pairs
    manifest: the manifest holds the camera, which sits up to 80 ft away and
    can fall in the neighbour polygon.
    """
    import geopandas as gpd

    _, _, curation_root, unit_table = UNIT_CASES[case]
    units = geo.load_units(curation_root, filename=unit_table)
    poly = geo.load_layer(layer)
    key = geo.LAYERS[layer]["key"]
    joined = gpd.sjoin(units[["unit_uid", "geometry"]], poly[[key, "geometry"]],
                       how="inner", predicate="within")
    return (joined[["unit_uid", key]]
            .rename(columns={key: "poly_key"})
            .drop_duplicates("unit_uid")
            .reset_index(drop=True))


def image_polygons(pairs: pd.DataFrame, layer: str = DEFAULT_LAYER) -> pd.DataFrame:
    """Give each SHOT of a pair the polygon its camera sits in.

    An image-mode pair holds no unit, thus the camera position is the only
    position there is. It is the true position of the shot, unlike in unit
    mode, where the camera sits up to 80 ft from the building it looks at.

    Returns the pair frame with `poly_a` and `poly_b` added.
    """
    import geopandas as gpd

    poly = geo.load_layer(layer)
    key = geo.LAYERS[layer]["key"]
    out = pairs.copy()
    for side in ("a", "b"):
        cols = [f"latitude_{side}", f"longitude_{side}"]
        part = out[["pair_id"] + cols].dropna()
        pts = gpd.GeoDataFrame(
            part,
            geometry=gpd.points_from_xy(part[f"longitude_{side}"],
                                        part[f"latitude_{side}"]),
            crs="EPSG:4326",
        ).to_crs(geo.WORKING_CRS)
        joined = gpd.sjoin(pts[["pair_id", "geometry"]], poly[[key, "geometry"]],
                           how="inner", predicate="within")
        joined = (joined[["pair_id", key]]
                  .rename(columns={key: f"poly_{side}"})
                  .drop_duplicates("pair_id"))
        out = out.merge(joined, on="pair_id", how="left")
    return out


def pair_table(case: str, codes: pd.DataFrame, layer: str = DEFAULT_LAYER,
               proxy: Optional[str] = None) -> pd.DataFrame:
    """Build 1 row for each usable pair: the code, the gap, and the verdict.

    Args:
        codes: The output of `_ic.codes`, for this case or for every case.

    Returns:
        pair_id, ic_code, the components, `proxy_gap`, `model_dir`, `agrees`.
    """
    run = next((r for r in C.runs(kind="trace", case=case)), None)
    if run is None:
        raise FileNotFoundError(f"{case}: no registered trace run")

    part = codes[codes["case"] == case].copy()
    if part.empty:
        raise ValueError(f"{case}: the code table holds no row for this case")

    image_mode = case in IMAGE_CASES
    pair_cols = (["pair_id", "latitude_a", "latitude_b", "longitude_a", "longitude_b"]
                 if image_mode else ["pair_id", "unit_uid_a", "unit_uid_b"])
    pairs = pd.read_parquet(run.pairs_link, columns=pair_cols)
    labels = pd.read_parquet(run.results_link,
                             columns=["pair_id", "relative_label", "relative_score"])

    part["pair_id"] = part["doc_id"].astype(str)
    pairs["pair_id"] = pairs["pair_id"].astype(str)
    labels["pair_id"] = labels["pair_id"].astype(str)

    df = (part.merge(labels, on="pair_id", how="left", suffixes=("", "_run"))
              .merge(pairs, on="pair_id", how="left"))

    prx = proxy_by_key(case, layer, proxy)
    if image_mode:
        # The camera position places the shot; there is no unit to look up.
        df = image_polygons(df, layer)
        for side in ("a", "b"):
            df = df.merge(
                prx.rename(columns={"poly_key": f"poly_{side}",
                                    "proxy_mean": f"proxy_{side}"}).drop(columns="proxy"),
                on=f"poly_{side}", how="left")
    else:
        place = unit_polygons(case, layer).merge(prx, on="poly_key", how="inner")
        for side in ("a", "b"):
            df = df.merge(
                place.rename(columns={"unit_uid": f"unit_uid_{side}",
                                      "poly_key": f"poly_{side}",
                                      "proxy_mean": f"proxy_{side}"}).drop(columns="proxy"),
                on=f"unit_uid_{side}", how="left")

    df["proxy_gap"] = df["proxy_a"] - df["proxy_b"]
    # `relative_score` already states A against B, thus no sign flip belongs here.
    score = pd.to_numeric(df.get("relative_score"), errors="coerce")
    df["model_dir"] = np.sign(score)

    keep = (df["poly_a"].notna() & df["poly_b"].notna()
            & (df["poly_a"] != df["poly_b"])
            & df["proxy_gap"].notna() & (df["proxy_gap"] != 0)
            & df["model_dir"].notna() & (df["model_dir"] != 0))
    df = df[keep].copy()
    df["agrees"] = (np.sign(df["proxy_gap"]) == df["model_dir"])
    df["abs_gap"] = df["proxy_gap"].abs()
    df["proxy"] = prx["proxy"].iloc[0] if len(prx) else ""
    df["layer"] = layer
    return df.reset_index(drop=True)


def coverage(case: str, codes: pd.DataFrame, pairs: pd.DataFrame) -> Dict[str, object]:
    """Say how many traces reached the test, and where the rest went.

    A reader must see this before an agreement rate. The test keeps a minority
    of the traces, and a silent drop would look like a sample.
    """
    part = codes[codes["case"] == case]
    return {
        "case": case,
        "traces": int(len(part)),
        "usable_pairs": int(len(pairs)),
        "kept": round(len(pairs) / max(1, len(part)), 3),
        "proxy": pairs["proxy"].iloc[0] if len(pairs) else "",
        "layer": pairs["layer"].iloc[0] if len(pairs) else "",
    }


def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """A confidence interval that behaves at a small n, unlike the normal one."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def agreement_by_code(pairs: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """The agreement rate at each code, with a Wilson interval.

    Warning: read `agreement_by_gap` beside this. A hard pair draws a longer
    trace, thus a low rate at a high code can be the difficulty and not the
    reasoning.
    """
    rows = []
    for code, part in pairs.groupby("ic_code"):
        n, k = len(part), int(part["agrees"].sum())
        lo, hi = _wilson(k, n)
        rows.append({"ic_code": int(code), "pairs": n, "agreement": round(k / n, 4),
                     "lo": round(lo, 4), "hi": round(hi, 4),
                     "median_gap": round(float(part["abs_gap"].median()), 4)})
    cols = ["ic_code", "pairs", "agreement", "lo", "hi", "median_gap"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    return out[out["pairs"] >= min_n].reset_index(drop=True)


def agreement_by_gap(pairs: pd.DataFrame, quantiles: int = 4,
                     split_code: int = 5, min_n: int = 30) -> pd.DataFrame:
    """The comparison that controls for difficulty.

    The pairs split into `quantiles` bands by the size of the proxy gap, and
    each band splits again on the code. Inside a band the 2 groups face a
    comparable task, thus a difference between them is about the reasoning.

    `split_code` is the rung that integration starts at: 5 and above means a
    justified weighing, 4 and below means no link or an unjustified one.
    """
    df = pairs.copy()
    df["gap_band"] = pd.qcut(df["abs_gap"], quantiles, labels=False, duplicates="drop")
    df["integrated_code"] = df["ic_code"] >= split_code
    rows = []
    for (band, high), part in df.groupby(["gap_band", "integrated_code"]):
        n, k = len(part), int(part["agrees"].sum())
        if n < min_n:
            continue
        lo, hi = _wilson(k, n)
        rows.append({
            "gap_band": int(band),
            "gap_range": f"{part['abs_gap'].min():.3g}-{part['abs_gap'].max():.3g}",
            "group": f"code >= {split_code}" if high else f"code < {split_code}",
            "pairs": n, "agreement": round(k / n, 4),
            "lo": round(lo, 4), "hi": round(hi, 4),
        })
    cols = ["gap_band", "gap_range", "group", "pairs", "agreement", "lo", "hi"]
    if not rows:
        # No band holds `min_n` pairs. That is a real answer for a small case,
        # thus give an empty frame with the columns a caller expects.
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(["gap_band", "group"]).reset_index(drop=True)


def contrast(pairs: pd.DataFrame, split_code: int = 5,
             quantiles: int = 4) -> pd.DataFrame:
    """1 number for each case: the difference inside a gap band, pooled.

    The pooled difference weights each band by its size, thus a band that holds
    few pairs cannot carry the answer.
    """
    table = agreement_by_gap(pairs, quantiles=quantiles, split_code=split_code)
    if table.empty:
        return pd.DataFrame()
    wide = table.pivot_table(index="gap_band", columns="group",
                             values=["agreement", "pairs"])
    cols = [c for c in wide["agreement"].columns]
    if len(cols) < 2:
        return pd.DataFrame()
    low, high = f"code < {split_code}", f"code >= {split_code}"
    if low not in cols or high not in cols:
        return pd.DataFrame()
    weight = wide["pairs"][[low, high]].min(axis=1)
    diff = wide["agreement"][high] - wide["agreement"][low]
    ok = weight.notna() & diff.notna()
    if not ok.any():
        return pd.DataFrame()
    pooled = float((diff[ok] * weight[ok]).sum() / weight[ok].sum())
    return pd.DataFrame([{
        "bands": int(ok.sum()),
        "pairs_low": int(wide["pairs"][low][ok].sum()),
        "pairs_high": int(wide["pairs"][high][ok].sum()),
        "pooled_difference": round(pooled, 4),
    }])


def plot_agreement_by_gap(pairs: pd.DataFrame, case: str = "",
                          quantiles: int = 4, split_code: int = 5):
    """2 lines over the gap bands: the complex traces and the simple ones."""
    import matplotlib.pyplot as plt

    table = agreement_by_gap(pairs, quantiles=quantiles, split_code=split_code)
    if table.empty:
        raise ValueError("no band holds enough pairs")
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    colours = {f"code >= {split_code}": S.PAL["blue"], f"code < {split_code}": S.PAL["coral"]}
    for group, part in table.groupby("group"):
        ax.errorbar(part["gap_band"], part["agreement"],
                    yerr=[part["agreement"] - part["lo"], part["hi"] - part["agreement"]],
                    marker="o", markersize=3.5, linewidth=1.2, capsize=2,
                    color=colours.get(group, S.PAL["slate"]), label=group)
    ax.axhline(0.5, color=S.EDGE, linewidth=0.6, linestyle=":")
    ax.set_xticks(sorted(table["gap_band"].unique()))
    ax.set_xlabel("proxy gap, small to large")
    ax.set_ylabel("agreement with the proxy")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- the sweep


def link_all(codes: pd.DataFrame, layer: str = DEFAULT_LAYER,
             cases: Optional[Sequence[str]] = None,
             split_code: int = 5, quantiles: int = 4) -> Tuple[pd.DataFrame, Dict]:
    """Run the test for every case and every proxy it holds.

    Returns:
        (summary, tables). The summary holds 1 row for each case and proxy.
        `tables` maps "<case>|<proxy>" to the pair frame behind that row, so a
        number can be read back to the pairs under it.
    """
    rows: List[Dict[str, object]] = []
    tables: Dict[str, pd.DataFrame] = {}
    for case in (cases or [c for c in linkable_cases() if c in set(codes["case"])]):
        names = proxies_for(case)
        if not names:
            # Name the case rather than drop it. Street photography holds no
            # outside measurement, thus it has no proxy export, and a silent
            # absence would read as a case that failed.
            rows.append({"case": case, "proxy": "", "usable_pairs": 0,
                         "note": "no proxy export; this case has no outside measurement"})
            continue
        for proxy in names:
            try:
                pairs = pair_table(case, codes, layer=layer, proxy=proxy)
            except (FileNotFoundError, ValueError, KeyError) as exc:
                rows.append({"case": case, "proxy": proxy, "usable_pairs": 0,
                             "note": f"{type(exc).__name__}: {exc}"})
                continue
            if pairs.empty:
                rows.append({"case": case, "proxy": proxy, "usable_pairs": 0,
                             "note": "no pair crosses 2 polygons with a proxy"})
                continue
            tables[f"{case}|{proxy}"] = pairs
            cov = coverage(case, codes, pairs)
            diff = contrast(pairs, split_code=split_code, quantiles=quantiles)
            rows.append({
                "case": case,
                "proxy": proxy,
                "traces": cov["traces"],
                "usable_pairs": cov["usable_pairs"],
                "kept": cov["kept"],
                "agreement": round(float(pairs["agrees"].mean()), 4),
                "share_code_ge": round(float((pairs["ic_code"] >= split_code).mean()), 4),
                "pooled_difference": (float(diff["pooled_difference"].iloc[0])
                                      if not diff.empty else float("nan")),
                "note": "",
            })
    return pd.DataFrame(rows), tables


def export(summary: pd.DataFrame, tables: Dict[str, pd.DataFrame], out_dir: Path,
           split_code: int = 5, quantiles: int = 4) -> List[str]:
    """Write the summary, the per-case bands, and 1 figure for each case."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    if not summary.empty:
        summary.to_csv(out_dir / "ic_link_summary.csv", index=False)
        written.append("ic_link_summary.csv")
    bands = []
    for key, pairs in tables.items():
        case, proxy = key.split("|", 1)
        table = agreement_by_gap(pairs, quantiles=quantiles, split_code=split_code)
        if table.empty:
            continue
        table.insert(0, "proxy", proxy)
        table.insert(0, "case", case)
        bands.append(table)
        name = f"ic_link_{case}_{proxy}.png"
        try:
            fig = plot_agreement_by_gap(pairs, case=case, quantiles=quantiles,
                                        split_code=split_code)
        except ValueError:
            continue
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)
        written.append(name)
    if bands:
        pd.concat(bands, ignore_index=True).to_csv(
            out_dir / "ic_link_bands.csv", index=False)
        written.append("ic_link_bands.csv")
    return written
