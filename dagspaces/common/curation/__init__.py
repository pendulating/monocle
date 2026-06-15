"""Curation: spatial sub-dataset bootstraps over the Cyclomedia catalog.

Each sub-dataset assembles a geographic mask (GeoJSON / polygon) and feeds it
into :class:`dagspaces.common.cyclomedia_catalog.CyclomediaCatalog.query` to
carve out a curated parquet of Cyclomedia rows.

The first sub-module is :mod:`dagspaces.common.curation.permits` — DOB
scaffold/shed permits through a date cutoff, buffered by N feet.
"""
