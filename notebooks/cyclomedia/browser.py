"""Cyclomedia NYC street-view browser.

Interactive explorer over the materialised Cyclomedia catalog
(``/share/ju/cyclomedia/catalog/v1``): 31.5M cube faces / 5.24M recordings
across all five boroughs, March-November 2025.

What you can do:
    1. Filter the city (borough, month, depth availability).
    2. Click the citywide density map to pick an area.
    3. Find the recordings nearest that point.
    4. View a recording: metadata, the six cube faces, an equirectangular panorama.
    5. View its depth maps: per-face, unfolded, and stitched into a depth panorama.
    6. Drop into raw SQL against the catalog.

Run with:
    marimo edit notebooks/cyclomedia/browser.py

Heads-up on depth units: the depth maps decode to a raw 16-bit *code*, and
`depth.to_metres()` converts it to Euclidean range along the pixel ray as
`(code - 16384) / 250`. The colour scales in this notebook are nonetheless a
*relative* percentile stretch over the raw codes, because a shared relative
scale reads better across faces -- so don't read the colour bar as metres even
though the underlying numbers are metric.
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _intro():
    import marimo as mo

    mo.md(
        """
        # Cyclomedia NYC browser

        **5.24M recordings · 31.5M cube faces · all five boroughs · Mar-Nov 2025**

        Every recording is a 360 degree panorama stored as six 90 degree cube faces
        (`F`/`R`/`B`/`L`/`U`/`D`) plus a matching depth map per face. The cube is
        **north-referenced**: the `F` face always points true north, regardless of
        which way the vehicle was driving.

        > **Depth decodes to metres.** The depth PNGs pack a 16-bit code with
        > `range_m = (code - 16384) / 250` — Euclidean range along the pixel ray.
        > The colour scales below are still a *relative* percentile stretch,
        > because that reads better across faces; use `depth.to_metres()` when
        > you want the numbers.
        """
    )
    return (mo,)


@app.cell
def _setup(mo):
    import functools
    import io
    import os
    import sys

    # dagspaces is not installed into the venv; it resolves off the repo root.
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from PIL import Image

    from dagspaces.common.cyclomedia import catalog as cat
    from dagspaces.common.cyclomedia import cubemap as cube
    from dagspaces.common.cyclomedia import depth as dep

    con = cat.connect()

    # The recording-level index (one row per recording) is derived from the
    # face-level catalog and cached to parquet. Building it scans all 31.5M rows
    # and takes ~10s; after that it loads in about a second.
    index_path = cat.build_recording_index(con)
    n_recordings = cat.load_recording_index(con)

    def png(arr):
        """numpy array -> PNG bytes, for mo.image()."""
        buf = io.BytesIO()
        Image.fromarray(np.asarray(arr)).save(buf, format="PNG")
        return buf.getvalue()

    @functools.lru_cache(maxsize=16)
    def load_cube(rec_dir: str, kind: str):
        """Cached face loader. NFS reads are ~200ms per recording per kind."""
        return cube.load_faces(rec_dir, kind)

    mo.md(
        f"""
        Catalog connected — **{n_recordings:,}** recordings indexed.
        <br>Index: `{index_path}`
        """
    )
    return cat, con, cube, dep, go, load_cube, np, pd, png


@app.cell(hide_code=True)
def _filters_ui(con, mo):
    boroughs = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT borough FROM recordings ORDER BY 1"
        ).fetchall()
    ]
    months = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT month FROM recordings WHERE month IS NOT NULL ORDER BY 1"
        ).fetchall()
    ]

    borough_ui = mo.ui.multiselect(
        options=boroughs, value=boroughs, label="Borough"
    )
    month_ui = mo.ui.range_slider(
        start=min(months), stop=max(months), value=[min(months), max(months)],
        step=1, label="Month (2025)", show_value=True,
    )
    depth_ui = mo.ui.checkbox(value=False, label="Only recordings with depth maps")

    mo.vstack([
        mo.md("## 1 · Filter"),
        mo.hstack([borough_ui, month_ui, depth_ui], justify="start", gap=2),
    ])
    return borough_ui, depth_ui, month_ui


@app.cell(hide_code=True)
def _filters_apply(borough_ui, con, depth_ui, mo, month_ui):
    _clauses = []
    if borough_ui.value:
        _quoted = ", ".join(f"'{b}'" for b in borough_ui.value)
        _clauses.append(f"borough IN ({_quoted})")
    else:
        # An empty multiselect means "nothing", not "everything" — say so rather
        # than silently showing the whole city.
        _clauses.append("false")
    _clauses.append(f"month BETWEEN {month_ui.value[0]} AND {month_ui.value[1]}")
    if depth_ui.value:
        _clauses.append("has_depth")

    where_sql = " AND ".join(_clauses)
    n_filtered = int(
        con.execute(f"SELECT count(*) FROM recordings WHERE {where_sql}").fetchone()[0]
    )

    mo.md(
        f"**{n_filtered:,}** recordings match. <br>`WHERE {where_sql}`"
        if n_filtered
        else f"**No recordings match.** `WHERE {where_sql}`"
    )
    return n_filtered, where_sql


@app.cell(hide_code=True)
def _overview_ui(mo):
    cell_ui = mo.ui.dropdown(
        options={"100 m": 100, "250 m": 250, "500 m": 500, "1 km": 1000},
        value="250 m",
        label="Grid cell",
    )
    mo.vstack([
        mo.md("""
        ## 2 · Citywide map

        Recordings binned into a density grid — 5.2M individual points will not
        render, so the overview aggregates. **Box- or lasso-select** a region
        (drag on the map) to set the query point for the next section.
        """),
        cell_ui,
    ])
    return (cell_ui,)


@app.cell(hide_code=True)
def _overview_map(cat, cell_ui, con, go, mo, n_filtered, where_sql):
    if n_filtered:
        grid_df = cat.overview_grid(con, cell_m=cell_ui.value, where=where_sql)
    else:
        grid_df = None

    if grid_df is None or grid_df.empty:
        overview = mo.md("*No data to map — widen the filters above.*")
        overview_map = None
    else:
        _fig = go.Figure(
            go.Scattermap(
                lat=grid_df["latitude"],
                lon=grid_df["longitude"],
                mode="markers",
                marker=dict(
                    size=7,
                    color=grid_df["n"],
                    colorscale="Viridis",
                    opacity=0.75,
                    colorbar=dict(title="recordings<br>per cell"),
                ),
                customdata=grid_df[["n", "borough"]],
                hovertemplate=(
                    "%{customdata[1]}<br>%{customdata[0]} recordings"
                    "<br>%{lat:.5f}, %{lon:.5f}<extra></extra>"
                ),
            )
        )
        _fig.update_layout(
            map=dict(style="carto-positron", center=dict(lat=40.71, lon=-73.94), zoom=9.3),
            margin=dict(l=0, r=0, t=0, b=0),
            height=560,
            dragmode="select",
        )
        overview_map = mo.ui.plotly(_fig)
        overview = overview_map
    overview
    return (overview_map,)


@app.cell(hide_code=True)
def _selection(mo, overview_map):
    # A box/lasso selection on the overview gives us the cells the user dragged
    # over; their centroid becomes the default query point.
    sel_points = overview_map.value if overview_map is not None else None

    if sel_points:
        sel_lat = sum(p["lat"] for p in sel_points) / len(sel_points)
        sel_lon = sum(p["lon"] for p in sel_points) / len(sel_points)
        sel_note = mo.md(
            f"Selected **{len(sel_points)}** grid cells — centroid "
            f"**{sel_lat:.5f}, {sel_lon:.5f}**."
        )
    else:
        # Times Square, so the notebook shows something real on a cold start.
        sel_lat, sel_lon = 40.75800, -73.98550
        sel_note = mo.md(
            "*No map selection — defaulting to Times Square. "
            "Drag a box on the map above, or type coordinates below.*"
        )
    sel_note
    return sel_lat, sel_lon


@app.cell(hide_code=True)
def _locate_ui(mo, sel_lat, sel_lon):
    # Seeded from the map selection, but freely editable — typing here overrides
    # the map.
    lat_ui = mo.ui.number(start=40.4, stop=41.0, step=1e-5, value=round(sel_lat, 5), label="Lat")
    lon_ui = mo.ui.number(start=-74.3, stop=-73.6, step=1e-5, value=round(sel_lon, 5), label="Lon")
    radius_ui = mo.ui.slider(25, 1000, value=150, step=25, label="Radius (m)", show_value=True)
    k_ui = mo.ui.slider(5, 200, value=40, step=5, label="Max results", show_value=True)

    mo.vstack([
        mo.md("""
        ## 3 · Find nearby recordings

        The nearest recordings to the query point. Coordinates are seeded from the
        map selection above; edit them to override.
        """),
        mo.hstack([lat_ui, lon_ui, radius_ui, k_ui], justify="start", gap=2),
    ])
    return k_ui, lat_ui, lon_ui, radius_ui


@app.cell(hide_code=True)
def _nearby(cat, con, go, k_ui, lat_ui, lon_ui, mo, radius_ui, where_sql):
    near_df = cat.nearest_recordings(
        con,
        lat=lat_ui.value,
        lon=lon_ui.value,
        k=k_ui.value,
        radius_m=radius_ui.value,
        where=where_sql,
    )

    if near_df.empty:
        nearby_view = mo.md(
            f"*No recordings within {radius_ui.value} m of "
            f"({lat_ui.value:.5f}, {lon_ui.value:.5f}) under the current filters.*"
        )
    else:
        _fig = go.Figure()
        _fig.add_trace(
            go.Scattermap(
                lat=near_df["latitude"],
                lon=near_df["longitude"],
                mode="markers",
                marker=dict(
                    size=11,
                    color=near_df["dist_m"],
                    colorscale="Plasma",
                    colorbar=dict(title="dist (m)"),
                ),
                text=near_df["recording_id"],
                customdata=near_df[["dist_m", "recordedAt"]],
                hovertemplate=(
                    "<b>%{text}</b><br>%{customdata[0]:.1f} m"
                    "<br>%{customdata[1]}<extra></extra>"
                ),
                name="recordings",
            )
        )
        _fig.add_trace(
            go.Scattermap(
                lat=[lat_ui.value],
                lon=[lon_ui.value],
                mode="markers",
                marker=dict(size=16, color="red", symbol="circle"),
                name="query point",
                hovertemplate="query point<extra></extra>",
            )
        )
        _fig.update_layout(
            map=dict(
                style="carto-positron",
                center=dict(lat=lat_ui.value, lon=lon_ui.value),
                zoom=17,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=460,
            showlegend=False,
        )
        nearby_view = mo.ui.plotly(_fig)
    nearby_view
    return (near_df,)


@app.cell(hide_code=True)
def _pick_ui(mo, near_df):
    _cols = [
        "recording_id", "dist_m", "borough", "recordedAt",
        "n_faces", "has_depth", "dataset", "group",
    ]
    if near_df.empty:
        pick_ui = None
        pick_view = mo.md("*Nothing to pick.*")
    else:
        _table = near_df[_cols].copy()
        _table["dist_m"] = _table["dist_m"].round(1)
        pick_ui = mo.ui.table(
            _table,
            selection="single",
            initial_selection=[0],
            page_size=8,
            label="Pick a recording to view",
        )
        pick_view = pick_ui
    pick_view
    return (pick_ui,)


@app.cell(hide_code=True)
def _resolve(cat, mo, near_df, pick_ui):
    # mo.ui.table hands back the selected row(s) as a DataFrame; fall back to the
    # closest recording if the selection is empty.
    rec = None
    if pick_ui is not None and not near_df.empty:
        _sel = pick_ui.value
        _rid = (
            _sel["recording_id"].iloc[0]
            if _sel is not None and len(_sel)
            else near_df["recording_id"].iloc[0]
        )
        rec = near_df[near_df["recording_id"] == _rid].iloc[0]

    if rec is None:
        rec_dir = None
        rec_header = mo.md("## 4 · Viewer\n\n*Select a recording above.*")
    else:
        rec_dir = cat.recording_dir(rec["dataset"], rec["group"], rec["recording_id"])
        rec_header = mo.vstack([
            mo.md(f"## 4 · Viewer — `{rec['recording_id']}`"),
            mo.hstack([
                mo.stat(label="Borough", value=str(rec["borough"])),
                mo.stat(label="Recorded", value=str(rec["recordedAt"])[:19]),
                mo.stat(label="Distance", value=f"{rec['dist_m']:.1f} m"),
                mo.stat(label="Faces", value=f"{int(rec['n_faces'])}/6"),
                mo.stat(
                    label="Camera height",
                    value=f"{rec['groundLevelOffset']:.2f} m",
                ),
                mo.stat(
                    label="Drive heading",
                    value=f"{rec['recorderDirection']:.0f}°",
                ),
            ], justify="start", gap=1),
            mo.md(f"`{rec_dir}`"),
        ])
    rec_header
    return (rec_dir,)


@app.cell(hide_code=True)
def _rgb_view(cube, load_cube, mo, png, rec_dir):
    if rec_dir is None:
        rgb_faces = None
        rgb_view = mo.md("")
    else:
        rgb_faces = load_cube(rec_dir, "rgb")
        if not rgb_faces:
            rgb_view = mo.md("*No RGB faces found on disk for this recording.*")
        else:
            _cross = cube.cube_cross(rgb_faces)
            _pano = cube.cube_to_equirect(rgb_faces, width=2048)
            rgb_view = mo.vstack([
                mo.md(
                    "### Cube faces\n"
                    "Unfolded cross — `U` on top, then `L F R B` around the "
                    "horizon, `D` below. `F` points **north**."
                ),
                mo.image(png(_cross), width=980),
                mo.md(
                    "### Equirectangular panorama\n"
                    "Stitched from the six faces. Left edge is **north (0°)**, "
                    "running east across to 360°; the middle row is the horizon."
                ),
                mo.image(png(_pano), width=980),
            ])
    rgb_view
    return (rgb_faces,)


@app.cell(hide_code=True)
def _depth_intro(mo):
    mo.md("""
    ## 5 · Depth maps

    Each face has a matching depth render on the identical cube geometry, so
    depth and RGB line up pixel for pixel.

    > **The codes convert to metres.** The PNGs pack a 16-bit code
    > (`code = R*256 + G`, with `0` = *no return* — sky, or beyond the depth
    > model), and `depth.to_metres()` maps it to Euclidean range along the
    > pixel ray: `range_m = (code - 16384) / 250`, a 4 mm quantum spanning
    > 0–196.6 m. The scale was measured from known camera baselines and is flat
    > from 19 m to 52 m; a facade decodes planar to ~4 cm RMS.
    >
    > Note the catalog's `groundLevelOffset` is **not** the rendered camera
    > height — anchoring on it biases the scale by ~2%.

    The colour scale below is still a *relative* percentile stretch over the raw
    codes, shared across all faces so they stay comparable with each other.
    Black = no return.
    """)
    return


@app.cell(hide_code=True)
def _depth_view(cube, dep, load_cube, mo, np, png, rec_dir):
    if rec_dir is None:
        depth_codes = None
        depth_view = mo.md("*Select a recording above.*")
    else:
        _raw = load_cube(rec_dir, "depth")
        depth_codes = dep.decode_depth_faces(_raw) if _raw else None

        if not depth_codes:
            depth_view = mo.md("*No depth maps on disk for this recording.*")
        else:
            # One stretch across every face, so the six faces (and the panorama)
            # share a scale. Per-face stretching would make them incomparable.
            _valid = np.concatenate(
                [c[c > dep.NO_RETURN] for c in depth_codes.values()]
            )
            _vmin, _vmax = np.percentile(_valid, [1, 99])

            _cross = cube.cube_cross(
                {
                    f: dep.colorize_depth(c, vmin=_vmin, vmax=_vmax)
                    for f, c in depth_codes.items()
                }
            )
            # Stitch the raw codes, then colourise once — see cube_to_equirect.
            _pano_code = cube.cube_to_equirect(depth_codes, width=2048)
            _pano = dep.colorize_depth(_pano_code, vmin=_vmin, vmax=_vmax)

            depth_view = mo.vstack([
                mo.md(
                    f"Shared colour scale: code **{_vmin:.0f}** (near) → "
                    f"**{_vmax:.0f}** (far). Black = no return."
                ),
                mo.image(png(_cross), width=980),
                mo.md("### Depth panorama"),
                mo.image(png(_pano), width=980),
            ])
    depth_view
    return (depth_codes,)


@app.cell(hide_code=True)
def _depth_stats(dep, depth_codes, mo, pd):
    if not depth_codes:
        stats_view = mo.md("")
    else:
        _rows = []
        for _f in ("F", "R", "B", "L", "U", "D"):
            if _f not in depth_codes:
                continue
            _s = dep.depth_stats(depth_codes[_f])
            _rows.append({
                "face": _f,
                "valid %": round(_s["pct_valid"], 1),
                "no-return %": round(_s["pct_no_return"], 1),
                "p1": int(_s.get("p1", 0)),
                "p50": int(_s.get("p50", 0)),
                "p99": int(_s.get("p99", 0)),
                "max": int(_s.get("max", 0)),
            })
        stats_view = mo.vstack([
            mo.md(
                "### Per-face depth codes\n"
                "`U` is mostly sky, so its no-return share is high; `D` sees only "
                "ground, so it is fully valid over a narrow range."
            ),
            mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=6),
        ])
    stats_view
    return


@app.cell(hide_code=True)
def _face_compare_ui(mo):
    face_ui = mo.ui.dropdown(
        options=["F", "R", "B", "L", "U", "D"], value="F", label="Face"
    )
    mo.vstack([mo.md("### Side by side — RGB vs depth, full resolution"), face_ui])
    return (face_ui,)


@app.cell(hide_code=True)
def _face_compare(dep, depth_codes, face_ui, mo, np, png, rgb_faces):
    _f = face_ui.value
    if not rgb_faces or _f not in rgb_faces:
        compare_view = mo.md("*Face unavailable.*")
    else:
        _panels = [mo.vstack([mo.md(f"**{_f} · RGB**"), mo.image(png(rgb_faces[_f]), width=470)])]
        if depth_codes and _f in depth_codes:
            _c = depth_codes[_f]
            _valid = np.concatenate([c[c > dep.NO_RETURN] for c in depth_codes.values()])
            _vmin, _vmax = np.percentile(_valid, [1, 99])
            _panels.append(
                mo.vstack([
                    mo.md(f"**{_f} · depth** (relative)"),
                    mo.image(png(dep.colorize_depth(_c, vmin=_vmin, vmax=_vmax)), width=470),
                ])
            )
        compare_view = mo.hstack(_panels, justify="start", gap=1)
    compare_view
    return


@app.cell(hide_code=True)
def _sql_ui(mo):
    sql_ui = mo.ui.text_area(
        value=(
            "SELECT borough, count(*) AS n, min(recordedAt) AS first_seen,\n"
            "       max(recordedAt) AS last_seen\n"
            "FROM recordings\n"
            "GROUP BY 1 ORDER BY n DESC"
        ),
        label="",
        rows=6,
        full_width=True,
    )
    run_ui = mo.ui.run_button(label="Run query")
    mo.vstack([
        mo.md("""
        ## 6 · SQL

        `recordings` is the recording-level index (one row each). For face-level
        rows — `image_path`, `bearing`, `file_size` — query the catalog parquet
        directly with
        `read_parquet('/share/ju/cyclomedia/catalog/v1/by_dataset/**/*.parquet', hive_partitioning=1)`.
        """),
        sql_ui,
        run_ui,
    ])
    return run_ui, sql_ui


@app.cell(hide_code=True)
def _sql_run(con, mo, run_ui, sql_ui):
    if not run_ui.value:
        sql_out = mo.md("*Press **Run query**.*")
    else:
        try:
            sql_out = mo.ui.table(con.execute(sql_ui.value).fetchdf(), page_size=15)
        except Exception as exc:  # surface the DB error rather than blanking out
            sql_out = mo.md(f"```\n{type(exc).__name__}: {exc}\n```")
    sql_out
    return


if __name__ == "__main__":
    app.run()
