"""Query API for the Cyclomedia catalog.

Usage:

    from dagspaces.common.cyclomedia_catalog import CyclomediaCatalog
    import geopandas as gpd

    cat = CyclomediaCatalog()
    cd5 = gpd.read_file("data/geo/nyc_community_districts.geojson").query("boro_cd == 105")
    df = cat.query(
        within=cd5,
        between=("2025-05-01", "2025-08-01"),
        faces={"F", "B", "L", "R"},
        datasets=["manhattan_2025_1k"],
    )
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from typing import Any, Iterable, Optional, Union
from zoneinfo import ZoneInfo

import polars as pl
import polars_st as st
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.wkb import dumps as wkb_dumps

__all__ = ["CyclomediaCatalog", "DEFAULT_CATALOG_ROOT"]

log = logging.getLogger(__name__)

DEFAULT_CATALOG_ROOT = "/share/ju/cyclomedia/catalog/v1"

_TimeLike = Union[str, dt.date, dt.datetime]


def _to_wgs84_union_wkb(polygons: Any) -> bytes:
    """Accept a GeoDataFrame, GeoSeries, shapely geometry, or iterable thereof;
    return a single WKB payload (EPSG:4326)."""
    # GeoPandas path
    try:
        import geopandas as gpd
    except ImportError:
        gpd = None  # type: ignore

    if gpd is not None and isinstance(polygons, (gpd.GeoDataFrame, gpd.GeoSeries)):
        g = polygons
        if g.crs is None:
            log.warning("CyclomediaCatalog.query: `within` has no CRS; assuming EPSG:4326")
        elif str(g.crs).lower() not in ("epsg:4326", "4326"):
            g = g.to_crs("EPSG:4326")
        geom = unary_union(list(g.geometry))
    elif isinstance(polygons, BaseGeometry):
        geom = polygons
    else:
        # assume iterable of shapely geometries
        geom = unary_union(list(polygons))
    return wkb_dumps(geom)


_CATALOG_TZ = ZoneInfo("America/New_York")


def _coerce_datetime(v: _TimeLike) -> dt.datetime:
    """Accept str | date | datetime; return a tz-aware datetime in US/Eastern.

    The catalog stores `recordedAt` as tz-aware US/Eastern, so the literals
    used in `is_between` must also be tz-aware. Naive inputs are assumed to
    be local to US/Eastern.
    """
    if isinstance(v, dt.datetime):
        return v if v.tzinfo is not None else v.replace(tzinfo=_CATALOG_TZ)
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day, tzinfo=_CATALOG_TZ)
    s = v.strip()
    try:
        parsed = dt.datetime.fromisoformat(s)
    except ValueError:
        parsed = dt.datetime.strptime(s, "%Y-%m-%d")
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=_CATALOG_TZ)


class CyclomediaCatalog:
    """Lazy Polars query interface over the partitioned catalog parquet."""

    def __init__(self, root: str = DEFAULT_CATALOG_ROOT) -> None:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise ValueError(f"Catalog root not found: {root}")
        self.root = root
        self._dataset_glob = os.path.join(root, "by_dataset", "**", "*.parquet")

    # -- basic introspection --------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        import json
        p = os.path.join(self.root, "manifest.json")
        if not os.path.isfile(p):
            return {}
        with open(p, "r") as f:
            return json.load(f)

    def datasets(self) -> list[str]:
        """Return dataset names found on disk (from hive partition dirs)."""
        root = os.path.join(self.root, "by_dataset")
        if not os.path.isdir(root):
            return []
        out = []
        for name in sorted(os.listdir(root)):
            if name.startswith("dataset="):
                out.append(name[len("dataset="):])
        return out

    # -- core scan ------------------------------------------------------------

    def scan(
        self,
        datasets: Optional[Iterable[str]] = None,
        years: Optional[Iterable[int]] = None,
    ) -> pl.LazyFrame:
        """Return a lazy Polars frame over the (optionally partition-pruned) dataset.

        Partition pruning is done by Polars when the filter is pushed down; for
        maximum punt-through we pass a glob that targets only the requested
        dataset/year hive dirs when those are given.
        """
        lf = pl.scan_parquet(self._dataset_glob, hive_partitioning=True)
        if datasets is not None:
            lf = lf.filter(pl.col("dataset").is_in(list(datasets)))
        if years is not None:
            lf = lf.filter(pl.col("year").is_in([int(y) for y in years]))
        return lf

    # -- the main query method -----------------------------------------------

    def query(
        self,
        within: Any = None,
        between: Optional[tuple[_TimeLike, _TimeLike]] = None,
        faces: Optional[Iterable[str]] = None,
        datasets: Optional[Iterable[str]] = None,
        years: Optional[Iterable[int]] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> pl.DataFrame:
        """Return rows matching all provided filters.

        Args:
            within: GeoDataFrame / GeoSeries / shapely geometry / iterable of geoms.
                Any CRS; reprojected to EPSG:4326 before the spatial check.
            between: (start, end) for `recordedAt`. Strings or datetimes.
            faces: subset of {"F","B","L","R","U","D"}.
            datasets: list of dataset names to restrict to.
            years: list of ints; restricts the year partition.
            columns: if given, only select these columns.

        Returns a `pl.DataFrame`. Call `.to_pandas()` if the caller wants pandas.
        """
        lf = self.scan(datasets=datasets, years=years)

        if faces is not None:
            faces_list = [f.upper() for f in faces]
            lf = lf.filter(pl.col("face").cast(pl.Utf8).is_in(faces_list))

        if between is not None:
            t0, t1 = between
            lf = lf.filter(
                pl.col("recordedAt").is_between(_coerce_datetime(t0), _coerce_datetime(t1))
            )

        if within is not None:
            poly_wkb = _to_wgs84_union_wkb(within)
            lf = lf.filter(
                st.from_wkb(pl.col("geom_wkb")).st.within(st.from_wkb(pl.lit(poly_wkb)))
            )

        if columns is not None:
            lf = lf.select(list(columns))

        return lf.collect()

    # -- drop-in replacement for create_cyclomedia_dataset.py ---------------

    def build_inference_parquet(
        self,
        output_path: str,
        **query_kwargs: Any,
    ) -> pl.DataFrame:
        """Materialize a query and write to parquet. Returns the DataFrame."""
        df = self.query(**query_kwargs)
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.write_parquet(output_path)
        log.info("CyclomediaCatalog: wrote %d rows to %s", df.height, output_path)
        return df
