"""The proxies that the parks case and the plazas case add at version 1.2.0.

`_proxies.py` holds the shared builders. This module holds the builders that
only these 2 cases use, so the shared file stays small and this file stays
under the owner of the 2 cases.

What this module adds
---------------------
| Case | Proxy | Measures | Orientation |
|------|-------|----------|-------------|
| Parks | `pip_condition_acceptable_rate` | The structure of the site | Higher is better |
| Parks | `park_acreage` | The size of the site | NONE. Size is not quality. |
| Plazas | `plaza_bid_partner` | The type of the maintenance partner | Higher is better |
| Plazas | `bid_membership` | A business improvement district covers it | Higher is better |
| Plazas | `dob_ecb_violations` | Open orders against the tax lot | Higher is better (negated) |

The condition metric needs no code here. `_proxies.PIP_METRICS` already holds
`condition_acceptable_rate`, and `_proxies.parks_pip_proxy` already builds it.
The parks notebook selected the cleanliness metric only, so the export never
carried the second one. Version 1.2.0 exports both.

Why 2 PIP metrics and not 1
---------------------------
The model answers "which one is better maintained". PIP answers 2 questions
about the same park, and they are different visual constructs:

- `cleanliness` reads litter, glass, and graffiti. It is the surface today.
- `overall_condition` reads the structure and the landscape. It is the fabric.

A park can be clean and broken, or dirty and sound. The 2 ratings disagree on
the parks themselves, thus 1 of them cannot stand for both.

Orientation
-----------
Every value that leaves this module is oriented "higher is better", with 1
exception that the table above names. `park_acreage` measures SIZE. A positive
correlation on that row says that the model calls a LARGER park better, which
is a finding to state and never an agreement to score. Add `park_acreage` to
`_results_table.UNORIENTED_PROXIES`.

Credentials
-----------
The SODA key lives in `.env` and `_proxies` reads it. Never print it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

import _proxies as px

__version__ = "1.0.0"

# --------------------------------------------------------------------------
# The dataset ids
# --------------------------------------------------------------------------
#: DPR Parks Properties. The same source that supplies the acquisition year.
PARKS_PROPERTIES_RESOURCE = "enfh-gkve"

#: Parks Inspection Program — Inspections. The parent of both PIP metrics.
PIP_INSPECTIONS_RESOURCE = "yg3y-7juh"

#: Parks Inspection Program — Feature Ratings. See `pip_feature_status`.
PIP_FEATURES_RESOURCE = "5mma-5n3h"

#: NYC DOT Pedestrian Plazas, polygon. It carries the maintenance partner.
DOT_PLAZA_RESOURCE = "k5k6-6jex"

#: Business Improvement Districts, polygon. 76 districts.
BID_RESOURCE = "7jdm-inj8"

#: DOB violations and ECB violations. Both key on borough, block, and lot.
DOB_VIOLATION_RESOURCE = "3h2n-5cm9"
ECB_VIOLATION_RESOURCE = "6bgk-3dad"

#: The column names of the BID polygon source. The publisher exported them
#: from a shapefile, thus the names carry no meaning and this map supplies it.
BID_COLUMNS = {
    "f_all_bi_1": "bid_borough",
    "f_all_bi_2": "bid_name",
    "f_all_bi_6": "bid_assessment",
    "f_all_bi_7": "bid_budget",
    "year_found": "bid_year_founded",
}

#: How far a plaza point may sit from a BID polygon and still count as inside.
#:
#: Warning: a strict `within` test undercounts the DOT plazas. A BID boundary
#: follows the sidewalks and the lots, and a DOT plaza sits in the ROADBED
#: between them, thus the plaza point falls a few feet outside the district
#: that pays to clean it. Measured 2026-08-31 over the 504 plaza units:
#:
#: | Tolerance | POPS | DOT plazas | Special districts |
#: |-----------|------|------------|-------------------|
#: | 0 ft (within) | 176/392 | 5/92 | 0/20 |
#: | 50 ft | 179/392 | 34/92 | 0/20 |
#: | 100 ft | 185/392 | 43/92 | 0/20 |
#: | 250 ft | 227/392 | 49/92 | 0/20 |
#:
#: 100 ft is a roadbed width. 250 ft reaches across a whole avenue and starts
#: to count a plaza that the district does not touch.
BID_TOLERANCE_FT = 100.0

#: How far a plaza unit point may sit from a DOT plaza polygon.
#:
#: A DOT plaza is small, and the FacDB point is a geocoded address, thus the
#: point can miss the polygon by a sidewalk width. Measured 2026-08-31 over the
#: 92 DOT plaza units: a strict `within` test reaches 83, 100 ft reaches all
#: 92, and 250 ft adds nothing more. Thus 100 ft is the smallest tolerance that
#: keeps the set whole.
DOT_PLAZA_TOLERANCE_FT = 100.0

#: The first year of the violation window.
VIOLATION_SINCE_YEAR = 2021

#: How many tax blocks go into 1 SODA request. The `where` clause holds the
#: blocks of the plazas, thus a large list makes a long URL.
BLOCK_CHUNK = 100


# --------------------------------------------------------------------------
# Parks — acreage
# --------------------------------------------------------------------------
#: Warning: acreage carries NO orientation. It is the size of a park and not
#: its quality. A positive correlation says that the model calls a LARGER park
#: better. State that as a finding. Do not read it as agreement.
ACREAGE_LABEL = "Park acreage (DPR Parks Properties)"


def park_acreage(min_acres: float = 0.0) -> pd.DataFrame:
    """Read the acreage of each DPR property.

    The Parks Properties layer carries `acres` beside the shape, thus this
    needs no new network call: `_proxies._parks_polygons` already caches the
    layer on disk. All 2,059 properties carry a value.

    Returns `proxy_key` (the DPR `gispropnum`) and `proxy_value` (acres).

    Args:
        min_acres: Drop a property under this size. The default keeps all.
    """
    g = px._parks_polygons()
    out = pd.DataFrame({
        "proxy_key": g["gispropnum"].astype(str).str.strip(),
        "proxy_value": pd.to_numeric(g["acres"], errors="coerce"),
    })
    out = out.dropna(subset=["proxy_key", "proxy_value"])
    out = out[out["proxy_value"] > min_acres]
    # A property id can repeat across the shape rows. Keep the largest, which
    # is the record that the property name points at.
    return (out.sort_values("proxy_value", ascending=False)
               .drop_duplicates("proxy_key").reset_index(drop=True))


def park_acreage_points():
    """The acreage of each park, on the map, ready for `aggregate_proxy`.

    This uses the same point layer that the PIP proxy uses, thus a park lands
    in exactly the polygon that its inspection rate lands in.
    """
    shapes = px.load_parks_properties()
    return shapes.merge(park_acreage(), on="proxy_key", how="inner")


# --------------------------------------------------------------------------
# Parks — the PIP feature ratings, and why they do not reach a park
# --------------------------------------------------------------------------
#: The features that the Feature Ratings source rates, with the row count of
#: each, measured 2026-08-31. The source rates the parts of a park that the
#: task names: benches, paved surfaces, sidewalks, and play equipment.
PIP_FEATURE_ROWS = {
    "Litter": 137_012, "Graffiti": 136_973, "Glass": 136_020,
    "Paved Surfaces": 125_484, "Trees": 123_527, "Sidewalks": 110_415,
    "Weeds": 103_910, "Fences": 99_745, "Benches": 93_678, "Lawns": 76_881,
    "Horticultural Areas": 65_837, "Play Equipment": 65_631,
    "Safety Surface": 50_854, "Ice": 34_314, "Athletic Fields": 16_884,
    "Water Bodies": 8_754, "Trails": 8_680,
}

#: The features that describe the FABRIC of a park, not its surface today.
#: A feature rate over this set would be the structural companion of the
#: cleanliness rate. `pip_feature_status` says why it cannot be built today.
PIP_STRUCTURAL_FEATURES = (
    "Benches", "Paved Surfaces", "Sidewalks", "Play Equipment",
    "Fences", "Safety Surface", "Trails", "Athletic Fields",
)


def pip_feature_status(use_cache: bool = False) -> Dict[str, Any]:
    """Report whether a feature-level PIP rate can be built today.

    The Feature Ratings source holds 1,408,115 rows of `feature`, `rating`, and
    1 numeric key. It carries NO `prop_id`. The only table that maps that key
    to a park is the Inspections source, and the city serves that source EMPTY
    today: a plain request returns the header row and no data.

    Thus a feature rate cannot reach a park, and this module builds none. The
    cleanliness rate and the condition rate come from a disk cache that a
    2026-08-12 pull filled, so those 2 metrics still work.

    Returns a dict that names the counts and the blocker. Read it in the
    notebook and print it, so a reader sees the reason and not a silent gap.
    """
    status: Dict[str, Any] = {
        "feature_resource": PIP_FEATURES_RESOURCE,
        "inspection_resource": PIP_INSPECTIONS_RESOURCE,
        "feature_rows": None,
        "inspection_rows": None,
        "blocked": True,
        "reason": "",
    }
    try:
        feat = px.fetch({"resource_id": PIP_FEATURES_RESOURCE,
                         "index": "pip-features"},
                        use_cache=use_cache, select="count(*) as n")
        status["feature_rows"] = int(feat["n"].iloc[0]) if len(feat) else 0
    except Exception as exc:  # the network, not the logic
        status["reason"] = f"feature count failed: {type(exc).__name__}"
        return status
    try:
        insp = px.fetch({"resource_id": PIP_INSPECTIONS_RESOURCE,
                         "index": "pip-inspections"},
                        use_cache=use_cache, select="count(*) as n")
        status["inspection_rows"] = int(insp["n"].iloc[0]) if len(insp) else 0
    except Exception as exc:
        status["reason"] = f"inspection count failed: {type(exc).__name__}"
        return status

    if status["inspection_rows"]:
        status["blocked"] = False
        status["reason"] = (
            "The Inspections source holds rows again. A feature rate is now "
            "possible: join the feature key to `inspection_id`, then take "
            "`prop_id` and roll the zones up to the parent site.")
    else:
        status["reason"] = (
            "The Feature Ratings source carries no `prop_id`. Its only key is "
            "the inspection id, and the Inspections source returns 0 rows "
            "today, thus a feature rating cannot reach a park.")
    return status


# --------------------------------------------------------------------------
# Plazas — the maintenance partner of a DOT pedestrian plaza
# --------------------------------------------------------------------------
#: The 3 partner types. A DOT plaza always names a partner, thus "no partner"
#: is not one of them. See `partner_type`.
PARTNER_TYPES = ("bid", "city_agency", "community_org")

PARTNER_TYPE_LABEL = {
    "bid": "A business improvement district",
    "city_agency": "A city agency (DOT or Parks)",
    "community_org": "A community group, a conservancy, or a school",
}

#: The words that name a city agency in the `partner` field.
CITY_AGENCY_WORDS = ("NYC DEPARTMENT OF", "DEPARTMENT OF TRANSPORTATION",
                     "DEPARTMENT OF PARKS")

#: The words that name a business improvement district on their own.
BID_WORDS = ("BUSINESS IMPROVEMENT DISTRICT", "IMPROVEMENT DISTRICT")

#: The 2 districts that the DOT field writes with another name than the BID
#: register does. The key is the DOT text and the value is the register name.
#: Without this map the substring test misses them.
#:
#: - The register writes "Union Square Partnership"; DOT writes "Union Sq".
#: - The register writes "Flatiron 23rd Street Partnership"; the district
#:   joined the NoMad district in 2023 and DOT writes the new name.
PARTNER_BID_ALIASES = {
    "UNION SQ PARTNERSHIP": "UNION SQUARE PARTNERSHIP",
    "FLATIRON NOMAD PARTNERSHIP": "FLATIRON 23RD STREET PARTNERSHIP",
}


def _norm_org(value: Any) -> str:
    """Put an organization name into a form that 2 sources share."""
    s = re.sub(r"\s+", " ", str(value).upper())
    return re.sub(r"[^A-Z0-9 ]", " ", s).strip()


def bid_names() -> List[str]:
    """The normalized names of the 76 business improvement districts."""
    b = bid_polygons()
    return sorted({_norm_org(v) for v in b["bid_name"].dropna() if str(v).strip()})


def partner_type(partner: Any, known_bids: Optional[Sequence[str]] = None) -> str:
    """Give 1 `partner` value its type.

    The test reads the BID register first, so the classification comes from
    data and not from a guess about a word. "Village Alliance" is a district
    and "North Brooklyn Parks Alliance" is not, and no keyword separates them.

    | Order | Test | Type |
    |-------|------|------|
    | 1 | The name holds a registered BID name, or an alias, or the words "improvement district", or the token BID | `bid` |
    | 2 | The name holds a city agency | `city_agency` |
    | 3 | Everything else | `community_org` |

    Warning: a partner can name 2 organizations, such as "NYC Department of
    Parks and Recreation/Union Sq Partnership". The order above gives it
    `bid`, because the district money reaches the plaza whatever else does.

    Warning: the type reads the FUNDING that the register can name. It does
    not rank the partners. A conservancy raises money like a district and this
    rule calls it `community_org`, because the BID register does not hold it.
    """
    text = _norm_org(partner)
    if not text:
        return "community_org"
    names = list(known_bids) if known_bids is not None else bid_names()
    if re.search(r"\bBID\b", text):
        return "bid"
    if any(w in text for w in BID_WORDS):
        return "bid"
    if any(alias in text for alias in PARTNER_BID_ALIASES):
        return "bid"
    for name in names:
        if len(name) >= 6 and name in text:
            return "bid"
    if any(w in text for w in CITY_AGENCY_WORDS):
        return "city_agency"
    return "community_org"


def dot_plazas(use_cache: bool = True):
    """Read the DOT pedestrian plaza polygons with the maintenance partner.

    93 rows. Every row names a partner, thus the field cannot say "nobody
    maintains this plaza". It says WHO maintains it.
    """
    import geopandas as gpd

    import _geography as geo

    entry = {"resource_id": DOT_PLAZA_RESOURCE, "index": "dot-plazas"}
    df = px.fetch(entry, use_cache=use_cache,
                  select="the_geom,plazaname,partner,boroname,borocd")
    if df.empty:
        return df
    geom = df["the_geom"].apply(px._to_shape)
    g = gpd.GeoDataFrame(df.drop(columns=["the_geom"]), geometry=list(geom),
                         crs="EPSG:4326")
    g = g[g.geometry.notna()].to_crs(geo.WORKING_CRS).copy()
    names = bid_names()
    g["partner_type"] = [partner_type(p, names) for p in g["partner"]]
    return g.reset_index(drop=True)


def attach_dot_plaza(units_gdf,
                     max_distance_ft: float = DOT_PLAZA_TOLERANCE_FT
                     ) -> pd.DataFrame:
    """Give each plaza unit the DOT plaza polygon that holds it.

    A unit inside a polygon takes that plaza. A unit outside every polygon
    takes the nearest plaza within `max_distance_ft`. This is the rule that
    `_proxies.attach_park_property_id` uses for a park.

    Returns `unit_uid`, `plazaname`, `partner`, and `partner_type`, with 1 row
    for each unit that reached a plaza.
    """
    import geopandas as gpd

    import _geography as geo

    poly = dot_plazas()
    if poly is None or len(poly) == 0:
        return pd.DataFrame(columns=["unit_uid", "plazaname", "partner",
                                     "partner_type"])
    keep = ["plazaname", "partner", "partner_type", "geometry"]
    pts = units_gdf.to_crs(geo.WORKING_CRS)[["unit_uid", "geometry"]].copy()

    inside = gpd.sjoin(pts, poly[keep], how="left", predicate="within")
    inside = inside.drop_duplicates("unit_uid")
    hit = inside.dropna(subset=["plazaname"])
    miss = pts[~pts.unit_uid.isin(hit.unit_uid)]
    if len(miss) and max_distance_ft > 0:
        near = gpd.sjoin_nearest(miss, poly[keep], max_distance=max_distance_ft,
                                 how="left").drop_duplicates("unit_uid")
        hit = pd.concat([hit, near.dropna(subset=["plazaname"])],
                        ignore_index=True)
    cols = ["unit_uid", "plazaname", "partner", "partner_type"]
    return hit[cols].drop_duplicates("unit_uid").reset_index(drop=True)


#: The FacDB source of a DOT pedestrian plaza. Only these units carry a
#: maintenance partner, thus only these units may take a partner value.
DOT_PLAZA_SOURCE = "dot_pedplazas"


def plaza_partner_values(units_gdf, facilities: pd.DataFrame,
                         max_distance_ft: float = DOT_PLAZA_TOLERANCE_FT
                         ) -> pd.DataFrame:
    """The BID-partner flag of each DOT pedestrian plaza unit.

    `proxy_value` is 1.0 when a business improvement district maintains the
    plaza and 0.0 when a city agency or a community group does. A district
    raises an assessment on its members and spends it on the street, thus this
    reads as maintenance MONEY and not as a measured condition.

    **Warning: this covers the DOT plazas only, and the code enforces it.** A
    POPS plaza and a special district space carry no partner field in any city
    source. Without the filter the nearest-plaza join gives 28 POPS units the
    partner of a DOT plaza beside them, which is a false attribution.

    **Warning: do not build "the share with a named partner".** Every DOT
    plaza names one, thus that share is 1.0 inside the DOT set and 0.0 outside
    it, and it measures which source a plaza came from and nothing else.
    """
    src = facilities[facilities["datasource"] == DOT_PLAZA_SOURCE]
    only = units_gdf[units_gdf["unit_uid"].isin(set(src["uid"]))]
    j = attach_dot_plaza(only, max_distance_ft=max_distance_ft)
    if j.empty:
        return pd.DataFrame(columns=["unit_uid", "proxy_value", "partner",
                                     "partner_type"])
    out = j.copy()
    out["proxy_value"] = (out["partner_type"] == "bid").astype(float)
    return out[["unit_uid", "proxy_value", "partner", "partner_type"]]


# --------------------------------------------------------------------------
# Plazas — business improvement district membership
# --------------------------------------------------------------------------
def bid_polygons(use_cache: bool = True):
    """Read the 76 business improvement district polygons.

    A BID raises an assessment on the property owners inside its boundary and
    spends it on supplemental cleaning, on graffiti removal, and on public
    space upkeep. Thus membership measures maintenance MONEY.

    Warning: the publisher exported the columns from a shapefile, so the names
    carry no meaning. `BID_COLUMNS` renames them.
    """
    import geopandas as gpd

    import _geography as geo

    entry = {"resource_id": BID_RESOURCE, "index": "bid"}
    df = px.fetch(entry, use_cache=use_cache,
                  select="the_geom," + ",".join(BID_COLUMNS))
    if df.empty:
        return df
    geom = df["the_geom"].apply(px._to_shape)
    g = gpd.GeoDataFrame(df.drop(columns=["the_geom"]), geometry=list(geom),
                         crs="EPSG:4326")
    g = g[g.geometry.notna()].to_crs(geo.WORKING_CRS)
    g = g.rename(columns=BID_COLUMNS)
    for col in ("bid_assessment", "bid_budget"):
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")
    return g.reset_index(drop=True)


def plaza_bid_values(units_gdf,
                     tolerance_ft: float = BID_TOLERANCE_FT) -> pd.DataFrame:
    """Flag each plaza unit inside a business improvement district.

    `proxy_value` is 1.0 inside a district and 0.0 outside it. Every plaza unit
    gets a value, thus this proxy covers the whole case — the first plaza proxy
    that does.

    Warning: read `BID_TOLERANCE_FT` before you change the tolerance. A strict
    `within` test finds 5 of the 92 DOT plazas, because a DOT plaza sits in the
    roadbed and the district boundary follows the sidewalk.

    Returns `unit_uid`, `proxy_value`, `bid_name`, and `bid_budget`.
    """
    import geopandas as gpd

    import _geography as geo

    poly = bid_polygons()
    pts = units_gdf.to_crs(geo.WORKING_CRS)[["unit_uid", "geometry"]].copy()
    if poly is None or len(poly) == 0:
        out = pts[["unit_uid"]].copy()
        out["proxy_value"] = 0.0
        out["bid_name"] = None
        out["bid_budget"] = float("nan")
        return out

    keep = ["bid_name", "bid_budget", "geometry"]
    if tolerance_ft > 0:
        hit = gpd.sjoin_nearest(pts, poly[keep], max_distance=tolerance_ft,
                                how="left")
    else:
        hit = gpd.sjoin(pts, poly[keep], how="left", predicate="within")
    hit = hit.drop_duplicates("unit_uid")

    out = pts[["unit_uid"]].merge(
        hit[["unit_uid", "bid_name", "bid_budget"]], on="unit_uid", how="left")
    out["proxy_value"] = out["bid_name"].notna().astype(float)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Plazas — DOB and ECB violations at the tax lot
# --------------------------------------------------------------------------
#: Warning: a violation names the tax LOT, not the plaza surface. A POPS plaza
#: shares its lot with the tower that had to build it, thus most orders on that
#: lot describe an elevator, a boiler, or a facade. A large tower collects more
#: orders than a small one for that reason alone.
#:
#: Read this row as "how much trouble the owner of this lot is in", which is a
#: capacity signal for upkeep. Do not read it as a rating of the plaza.
VIOLATION_LABEL = "DOB and ECB violations at the tax lot (negated)"


def _bbl_parts(bbl: pd.Series) -> pd.DataFrame:
    """Split a 10-digit BBL into borough, block, and lot."""
    s = bbl.astype(str)
    return pd.DataFrame({"boro": s.str[0], "block": s.str[1:6],
                         "lot": s.str[6:]})


def clean_bbl(values) -> pd.Series:
    """Return the usable BBLs of a column, and NaN for the rest.

    A usable BBL is 10 digits and starts with the borough code 1 to 5. This
    repeats the rule that `_proxies.pops_vintage` applies to the same column:
    the FacDB parquet writes the BBL as a float, thus the text carries a `.0`
    that must go before any join.
    """
    s = (pd.Series(values).astype(str)
         .str.replace(r"\.0$", "", regex=True).str.strip())
    return s.where(s.str.match(r"^[1-5]\d{9}$"))


def plaza_bbls(facilities: pd.DataFrame) -> pd.DataFrame:
    """The tax lot of each plaza unit that carries one.

    Returns `unit_uid` and `bbl`. Measured 2026-08-31 over the 504 plaza units:
    419 carry a usable BBL (390 of 392 POPS, 19 of 20 special district spaces,
    and 10 of 92 DOT plazas). A DOT plaza sits in the roadbed, which owns no
    tax lot, thus its BBL is absent by construction and not by an error.
    """
    f = facilities.copy()
    out = pd.DataFrame({
        "unit_uid": f["uid"].astype(str),
        "bbl": clean_bbl(f["bbl"]),
    })
    return out.dropna(subset=["bbl"]).reset_index(drop=True)


def _violation_counts(resource: str, bbls: Sequence[str], since_year: int,
                      select_extra: str = "", use_cache: bool = True
                      ) -> pd.DataFrame:
    """Count the violation rows of one source at a set of tax lots.

    Warning: filter on the server. DOB violations hold 2.5 million rows and ECB
    violations hold 1.8 million. This sends the borough, the tax blocks, and
    the first date to the server, and it keeps only the exact lots afterward.

    The `where` clause names BLOCKS and not lots, because the 2 sources pad the
    lot differently: an ECB lot is 4 or 5 characters and a DOB lot is 5. A
    block filter is exact in both, and the lot test then runs on this side.
    """
    parts = _bbl_parts(pd.Series(sorted(set(bbls))))
    want = set(sorted(set(bbls)))
    frames: List[pd.DataFrame] = []
    entry = {"resource_id": resource, "index": f"viol-{resource}"}
    since = f"{int(since_year)}0101"

    for boro, grp in parts.groupby("boro"):
        blocks = sorted(set(grp["block"]))
        for i in range(0, len(blocks), BLOCK_CHUNK):
            chunk = blocks[i:i + BLOCK_CHUNK]
            quoted = ",".join(f"'{b}'" for b in chunk)
            where = (f"boro='{boro}' AND block in ({quoted}) "
                     f"AND issue_date >= '{since}'")
            df = px.fetch(entry, use_cache=use_cache,
                          select=f"boro,block,lot,issue_date{select_extra}",
                          where=where)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["bbl", "n"])
    raw = pd.concat(frames, ignore_index=True)
    # Build the BBL on this side, where the padding is under our control.
    lot = pd.to_numeric(raw["lot"], errors="coerce")
    block = pd.to_numeric(raw["block"], errors="coerce")
    ok = lot.notna() & block.notna()
    raw = raw[ok].copy()
    raw["bbl"] = (raw["boro"].astype(str).str.strip()
                  + block[ok].astype(int).astype(str).str.zfill(5)
                  + lot[ok].astype(int).astype(str).str.zfill(4))
    raw = raw[raw["bbl"].isin(want)]
    return raw.reset_index(drop=True)


def violations_by_bbl(bbls: Sequence[str],
                      since_year: int = VIOLATION_SINCE_YEAR,
                      use_cache: bool = True) -> pd.DataFrame:
    """Count the DOB and the ECB orders against each tax lot.

    Returns `bbl`, `n_dob`, `n_ecb`, `n_dob_open`, and `n_violations`. A lot
    with no order gets a 0 row, thus a plaza never drops out for a clean record.

    Args:
        since_year: The first year of the window. The count covers
            `since_year`-01-01 to the pull date.
    """
    keys = sorted({str(b) for b in bbls})
    if not keys:
        return pd.DataFrame(columns=["bbl", "n_dob", "n_ecb", "n_dob_open",
                                     "n_violations"])

    dob = _violation_counts(DOB_VIOLATION_RESOURCE, keys, since_year,
                            select_extra=",violation_category",
                            use_cache=use_cache)
    ecb = _violation_counts(ECB_VIOLATION_RESOURCE, keys, since_year,
                            select_extra=",ecb_violation_status",
                            use_cache=use_cache)

    out = pd.DataFrame({"bbl": keys})
    if len(dob):
        agg = dob.groupby("bbl").size().rename("n_dob").reset_index()
        open_mask = dob["violation_category"].astype(str).str.upper().str.contains(
            "ACTIVE", na=False)
        opn = (dob[open_mask].groupby("bbl").size()
               .rename("n_dob_open").reset_index())
        out = out.merge(agg, on="bbl", how="left").merge(opn, on="bbl", how="left")
    else:
        out["n_dob"] = 0.0
        out["n_dob_open"] = 0.0
    if len(ecb):
        agg = ecb.groupby("bbl").size().rename("n_ecb").reset_index()
        out = out.merge(agg, on="bbl", how="left")
    else:
        out["n_ecb"] = 0.0

    for col in ("n_dob", "n_ecb", "n_dob_open"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)
    out["n_violations"] = out["n_dob"] + out["n_ecb"]
    return out.reset_index(drop=True)


def plaza_violation_values(facilities: pd.DataFrame,
                           since_year: int = VIOLATION_SINCE_YEAR,
                           use_cache: bool = True) -> pd.DataFrame:
    """The negated violation count of each plaza unit that carries a tax lot.

    A violation marks a lot that failed an inspection, thus this carries a sign
    of -1 and leaves the module oriented "higher is better". A positive
    correlation means the model and the record agree.

    Warning: read `VIOLATION_LABEL` first. The order names the LOT, not the
    plaza surface.

    Returns `unit_uid`, `proxy_value`, `bbl`, `n_dob`, `n_ecb`, `n_dob_open`,
    and `n_violations`.
    """
    keys = plaza_bbls(facilities)
    if keys.empty:
        return pd.DataFrame(columns=["unit_uid", "proxy_value", "bbl"])
    counts = violations_by_bbl(keys["bbl"].unique(), since_year=since_year,
                               use_cache=use_cache)
    out = keys.merge(counts, on="bbl", how="left")
    for col in ("n_dob", "n_ecb", "n_dob_open", "n_violations"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)
    out["proxy_value"] = -out["n_violations"]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# The shared aggregation of a unit-keyed value
# --------------------------------------------------------------------------
def unit_values_by_layer(values: pd.DataFrame, facilities: pd.DataFrame,
                         layer: str, min_units: int = 1) -> pd.DataFrame:
    """Aggregate a per-unit proxy value into 1 geography layer.

    The value of a polygon is the MEAN over its units, and `n_units_proxy`
    counts them. A 0/1 value thus aggregates to a SHARE, which is what the BID
    row and the partner row report.

    This mirrors `_proxies.vintage_by_layer`, so a unit-keyed proxy and a
    vintage land in the same polygons by the same rule.

    Args:
        values: `unit_uid` and `proxy_value`.
        facilities: The FacDB table that carries `uid`, `latitude`, and
            `longitude`.
    """
    if values is None or values.empty:
        return pd.DataFrame()
    pos = facilities[["uid", "latitude", "longitude"]].rename(
        columns={"uid": "unit_uid"})
    pos["unit_uid"] = pos["unit_uid"].astype(str)
    v = values.copy()
    v["unit_uid"] = v["unit_uid"].astype(str)
    j = v.merge(pos, on="unit_uid", how="inner").dropna(
        subset=["latitude", "longitude", "proxy_value"])
    if j.empty:
        return pd.DataFrame()
    pts = px.points_from_latlon(
        j.rename(columns={"unit_uid": "proxy_key"}))
    return px.aggregate_proxy(pts, layer, min_units=min_units)


def coverage(values: pd.DataFrame, facilities: pd.DataFrame,
             label: str) -> Dict[str, Any]:
    """How much of a unit set 1 proxy reaches, by facility source.

    Read this before a correlation. A proxy that covers a fifth of the units
    describes that fifth, and the notebook must say so.
    """
    have = set(values["unit_uid"].astype(str)) if len(values) else set()
    f = facilities[["uid", "datasource"]].copy()
    f["uid"] = f["uid"].astype(str)
    f["has"] = f["uid"].isin(have)
    by = f.groupby("datasource")["has"].agg(units="size", reached="sum")
    by["rate"] = (by["reached"] / by["units"]).round(3)
    return {
        "proxy": label,
        "units": int(len(f)),
        "reached": int(f["has"].sum()),
        "rate": round(float(f["has"].mean()), 4) if len(f) else 0.0,
        "by_source": by.reset_index(),
    }
