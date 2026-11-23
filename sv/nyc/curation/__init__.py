"""Curation: spatial sub-dataset bootstraps over the Cyclomedia catalog.

Each sub-dataset assembles a geographic mask (GeoJSON / polygon) and feeds it
into :class:`sv.cyclomedia_catalog.CyclomediaCatalog.query` to
carve out a curated parquet of Cyclomedia rows.

The first sub-module is :mod:`sv.nyc.curation.permits` — DOB
scaffold/shed permits through a date cutoff, buffered by N feet.
"""

#: The Socrata host of NYC Open Data. The framework reader
#: ``dagspaces.sources.opendata.socrata.fetch_socrata`` gets the rows.
NYC_OPEN_DATA = "data.cityofnewyork.us"
#: The Socrata host of NY State Open Data (the subway entrances).
NY_STATE_OPEN_DATA = "data.ny.gov"
