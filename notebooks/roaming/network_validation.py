"""Exploratory validation of the urbanroamvqa board-first street network.

Marimo notebook that loads the canonical board graph (conf/graph/board_25m.yaml,
config-fingerprinted cache built by scripts/build_roam_board_graph.py) and
audits the properties the roaming game depends on:

    1. QA summary — connectivity (must be exactly 1 component), size, coverage
    2. Degree distribution — dead ends / corridors / intersections
    3. Pitch uniformity — edge lengths vs the 25m target
    4. Spatial coverage — board nodes vs raw recording positions
    5. Face alignment — how well compass-fixed faces (F=N, R=E, B=S, L=W)
       match actual street bearings (Manhattan grid is rotated ~29 degrees)
    6. Legal-move audit — menu sizes, ambiguous faces, unreachable edges
    7. Interactive close-up map — inspect any intersection's edges and faces
    8. Random-walk simulation — 500 VLM-free walks over the real mechanics

Run with:  marimo edit notebooks/roaming/network_validation.py
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Board-First Street Network Validation

    The roaming dagspace plays a *board game* over the NYC street network:
    a **uniform**, **walkable**, **100% connected** board built from OSM,
    with Cyclomedia imagery attached to board positions (not the other way
    around). This notebook is the exploratory validation of that board.
    """)
    return


@app.cell
def _():
    import os
    import sys

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    import math

    import matplotlib.pyplot as plt
    import numpy as np
    from omegaconf import OmegaConf

    return OmegaConf, REPO_ROOT, math, np, os, plt


@app.cell
def _(OmegaConf, REPO_ROOT, os):
    # Load the graph exactly the way the pipeline does: same config -> same
    # fingerprint -> instant cache hit. A rebuild here means the cache is stale.
    _conf_dir = os.path.join(REPO_ROOT, "dagspaces", "urbanroamvqa", "conf")
    graph_cfg = OmegaConf.load(os.path.join(_conf_dir, "graph", "board_25m.yaml"))
    data_cfg = OmegaConf.load(os.path.join(_conf_dir, "data", "cyclomedia_manhattan_2025.yaml"))
    graph_cfg.metadata_parquet = str(data_cfg.parquet_path)

    from dagspaces.urbanroamvqa.graph.builder import build_street_graph
    from dagspaces.urbanroamvqa.graph.street_graph import (
        FACE_BEARING_DEG,
        HORIZONTAL_FACES,
        compute_graph_diagnostics,
    )

    graph = build_street_graph(str(data_cfg.parquet_path), graph_cfg)
    diag = compute_graph_diagnostics(graph)
    return FACE_BEARING_DEG, HORIZONTAL_FACES, data_cfg, diag, graph, graph_cfg


@app.cell
def _(graph, np):
    # Arrays reused by every section
    node_ids = list(graph.adjacency.keys())
    degrees = np.array([len(graph.adjacency[r]) for r in node_ids])
    node_lats = np.array([graph.coords[r][0] for r in node_ids])
    node_lons = np.array([graph.coords[r][1] for r in node_ids])
    edge_dists = np.array(
        [nb.distance_m for nbs in graph.adjacency.values() for nb in nbs]
    )
    return degrees, edge_dists, node_ids, node_lats, node_lons


@app.cell
def _(diag, graph_cfg, mo):
    # --- 1. QA summary ---------------------------------------------------
    _spacing = float(graph_cfg.spacing_m)
    _checks = [
        ("Exactly 1 connected component", diag["graph/n_components"] == 1,
         f"{int(diag['graph/n_components'])} component(s)"),
        ("No isolated nodes", diag["graph/n_isolated"] == 0,
         f"{int(diag['graph/n_isolated'])} isolated"),
        ("Median pitch within 40% of target", abs(diag["graph/edge_m_median"] - _spacing) <= 0.4 * _spacing,
         f"median {diag['graph/edge_m_median']:.1f}m vs target {_spacing:.0f}m"),
        ("Degree mean in walkable range [1.8, 3.2]", 1.8 <= diag["graph/degree_mean"] <= 3.2,
         f"mean degree {diag['graph/degree_mean']:.2f}"),
    ]
    _rows = "\n".join(
        f"| {'✅' if ok else '❌'} | {name} | {detail} |" for name, ok, detail in _checks
    )
    mo.md(
        f"""
        ## 1. QA Summary

        | | Check | Observed |
        |---|---|---|
        {_rows}

        **{int(diag['graph/n_nodes']):,} nodes** · {int(diag['graph/n_edges']):,} edges ·
        {int(diag['graph/n_dead_ends']):,} dead ends · {int(diag['graph/n_intersections']):,} intersections (degree > 2) ·
        pitch p10/p50/p90 = {diag['graph/edge_m_p10']:.0f} / {diag['graph/edge_m_median']:.0f} / {diag['graph/edge_m_p90']:.0f} m
        """
    )
    return


@app.cell
def _(degrees, mo, np, plt):
    # --- 2. Degree distribution -------------------------------------------
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _vals, _counts = np.unique(degrees, return_counts=True)
    _axes[0].bar(_vals, _counts, color="steelblue")
    _axes[0].set_xlabel("degree")
    _axes[0].set_ylabel("nodes")
    _axes[0].set_title("Node degree distribution")
    for _v, _c in zip(_vals, _counts):
        _axes[0].text(_v, _c, f"{_c:,}", ha="center", va="bottom", fontsize=8)

    _kinds = {
        "dead end (1)": int((degrees == 1).sum()),
        "corridor (2)": int((degrees == 2).sum()),
        "intersection (3+)": int((degrees >= 3).sum()),
    }
    _axes[1].pie(_kinds.values(), labels=_kinds.keys(), autopct="%1.1f%%",
                 colors=["#cc6666", "#88aacc", "#66aa66"])
    _axes[1].set_title("Node roles")
    _fig.tight_layout()
    mo.vstack([
        mo.md("## 2. Degree Distribution\n\nA healthy board is mostly degree-2 "
              "corridors with degree-3/4 intersections; dead ends should be "
              "real cul-de-sacs, not coverage artifacts."),
        _fig,
    ])
    return


@app.cell
def _(edge_dists, graph_cfg, mo, np, plt):
    # --- 3. Pitch uniformity ------------------------------------------------
    _spacing = float(graph_cfg.spacing_m)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _axes[0].hist(edge_dists, bins=80, color="steelblue", edgecolor="none")
    _axes[0].axvline(_spacing, color="crimson", ls="--", label=f"target {_spacing:.0f}m")
    _axes[0].axvline(float(graph_cfg.max_contracted_edge_m), color="orange", ls=":",
                     label=f"contract cap {float(graph_cfg.max_contracted_edge_m):.0f}m")
    _axes[0].set_xlabel("edge length (m)")
    _axes[0].set_ylabel("edges")
    _axes[0].set_title("Edge pitch distribution")
    _axes[0].legend()

    _sorted = np.sort(edge_dists)
    _axes[1].plot(_sorted, np.linspace(0, 1, len(_sorted)))
    _axes[1].axvline(_spacing, color="crimson", ls="--")
    _axes[1].set_xlabel("edge length (m)")
    _axes[1].set_ylabel("CDF")
    _axes[1].set_title("Pitch CDF")
    _fig.tight_layout()

    _over_cap = float((edge_dists > float(graph_cfg.max_contracted_edge_m)).mean())
    mo.vstack([
        mo.md(f"## 3. Pitch Uniformity\n\nEvery step should feel like one board "
              f"square (~{_spacing:.0f}m). Edges above the contract cap: "
              f"**{100 * _over_cap:.2f}%** (should be 0 — the builder never creates them)."),
        _fig,
    ])
    return


@app.cell
def _(data_cfg, mo, node_lats, node_lons, np, plt):
    # --- 4. Spatial coverage: board vs raw recordings ----------------------
    import pandas as pd

    _recs = (
        pd.read_parquet(str(data_cfg.parquet_path), columns=["recording_id", "lat", "lon"])
        .drop_duplicates("recording_id")
    )
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    _sample = _recs.sample(min(60_000, len(_recs)), random_state=0)
    _axes[0].scatter(_sample["lon"], _sample["lat"], s=0.3, alpha=0.25, color="gray")
    _axes[0].set_title(f"Raw recordings ({len(_recs):,}, sampled)")
    _axes[1].scatter(node_lons, node_lats, s=0.3, alpha=0.4, color="darkgreen")
    _axes[1].set_title(f"Board nodes ({len(node_lats):,})")
    for _ax in _axes:
        _ax.set_aspect(1.0 / np.cos(np.radians(40.75)))
        _ax.set_xlabel("lon")
    _axes[0].set_ylabel("lat")
    _fig.tight_layout()
    mo.vstack([
        mo.md("## 4. Spatial Coverage\n\nThe board should trace the full street "
              "grid with uniform density — no dual-lane doubling, no dense "
              "capture-rate clumps, and gaps only where imagery truly has none."),
        _fig,
    ])
    return (pd,)


@app.cell
def _(FACE_BEARING_DEG, graph, mo, node_ids, np, plt):
    # --- 5. Face alignment with street bearings -----------------------------
    # Faces are compass-fixed (F=N, R=E, B=S, L=W). For every directed edge,
    # measure the angular gap to the nearest face bearing. Resolution fails
    # past the 45-degree tolerance; the Manhattan grid (~29 degrees off true
    # north) should land well inside it.
    _gaps = []
    for _rid in node_ids:
        for _nb in graph.adjacency[_rid]:
            _g = min(
                min(abs(_nb.bearing_deg - f) % 360, 360 - abs(_nb.bearing_deg - f) % 360)
                for f in FACE_BEARING_DEG.values()
            )
            _gaps.append(_g)
    _gaps_arr = np.array(_gaps)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(_gaps_arr, bins=45, range=(0, 45), color="steelblue")
    _ax.set_xlabel("edge bearing gap to nearest face (deg)")
    _ax.set_ylabel("directed edges")
    _ax.set_title("Face / street alignment (must be < 45)")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"## 5. Face Alignment\n\nMax gap observed: "
              f"**{_gaps_arr.max():.1f} degrees** (tolerance 45). "
              f"p95 = {np.percentile(_gaps_arr, 95):.1f} degrees. The ~29-degree "
              f"bump is the rotated Manhattan grid — expected."),
        _fig,
    ])
    return


@app.cell
def _(HORIZONTAL_FACES, graph, mo, node_ids, np, plt):
    # --- 6. Legal-move audit -------------------------------------------------
    # The menu shown to the agent = legal_faces(). Audit menu sizes and how many
    # edges are shadowed (two edges mapping to the same face: only the
    # closer-bearing one is reachable).
    _menu_sizes = []
    _shadowed = 0
    for _rid in node_ids:
        _legal = graph.legal_faces(_rid)
        _menu_sizes.append(len(_legal))
        _resolved = {
            graph.resolve_face_to_neighbor(_rid, _f).recording_id
            for _f in _legal
        }
        _shadowed += len(graph.adjacency[_rid]) - len(_resolved)
    _menu_arr = np.array(_menu_sizes)

    _fig, _ax = plt.subplots(figsize=(7, 3.5))
    _vals, _counts = np.unique(_menu_arr, return_counts=True)
    _ax.bar(_vals, _counts, color="seagreen")
    _ax.set_xlabel("legal faces per node (no exclusions)")
    _ax.set_ylabel("nodes")
    _ax.set_xticks(list(range(0, len(HORIZONTAL_FACES) + 1)))
    _fig.tight_layout()

    _n_zero = int((_menu_arr == 0).sum())
    _total_edges = sum(len(v) for v in graph.adjacency.values())
    mo.vstack([
        mo.md(f"## 6. Legal-Move Audit\n\n"
              f"Nodes with **zero** legal moves: **{_n_zero}** (must be 0). "
              f"Shadowed edges (unreachable because a same-face sibling is "
              f"closer in bearing): **{_shadowed:,}** of {_total_edges:,} "
              f"directed edges ({100 * _shadowed / max(1, _total_edges):.2f}%)."),
        _fig,
    ])
    return


@app.cell
def _(mo, node_lats, node_lons):
    # --- 7. Interactive close-up ----------------------------------------------
    lat_input = mo.ui.number(
        value=round(float(node_lats.mean()), 5), label="center lat", step=0.001
    )
    lon_input = mo.ui.number(
        value=round(float(node_lons.mean()), 5), label="center lon", step=0.001
    )
    radius_input = mo.ui.slider(100, 1000, value=300, label="radius (m)")
    mo.vstack([
        mo.md("## 7. Intersection Close-Up\n\nPan the board anywhere in "
              "Manhattan; nodes are colored by degree (red = dead end, "
              "blue = corridor, green = intersection)."),
        mo.hstack([lat_input, lon_input, radius_input]),
    ])
    return lat_input, lon_input, radius_input


@app.cell
def _(
    graph,
    lat_input,
    lon_input,
    math,
    node_ids,
    node_lats,
    node_lons,
    np,
    radius_input,
):
    import folium

    _clat, _clon = float(lat_input.value), float(lon_input.value)
    _radius_deg = float(radius_input.value) / 111_000.0
    _mask = (np.abs(node_lats - _clat) < _radius_deg) & (
        np.abs(node_lons - _clon) < _radius_deg / math.cos(math.radians(_clat))
    )
    _local_ids = [node_ids[i] for i in np.where(_mask)[0]]

    closeup_map = folium.Map(location=[_clat, _clon], zoom_start=17, tiles="cartodbpositron")
    _deg_color = lambda d: "#cc3333" if d == 1 else ("#3366cc" if d == 2 else "#33aa33")
    _local_set = set(_local_ids)
    for _rid in _local_ids:
        _lat, _lon = graph.coords[_rid]
        _nbs = graph.adjacency[_rid]
        folium.CircleMarker(
            [_lat, _lon], radius=3, color=_deg_color(len(_nbs)), fill=True,
            tooltip=f"{_rid} deg={len(_nbs)} yaw={graph.yaw_degrees[_rid]:.0f}",
        ).add_to(closeup_map)
        for _nb in _nbs:
            if _nb.recording_id in _local_set and _rid < _nb.recording_id:
                _nlat, _nlon = graph.coords[_nb.recording_id]
                folium.PolyLine(
                    [[_lat, _lon], [_nlat, _nlon]], weight=2, opacity=0.6,
                    tooltip=f"{_nb.distance_m:.0f}m @ {_nb.bearing_deg:.0f}°",
                ).add_to(closeup_map)
    closeup_map
    return (folium,)


@app.cell
def _(graph, mo, node_ids, np, pd):
    # --- 8. Random-walk simulation (no VLM) -----------------------------------
    # Drive the REAL mechanics (legal_faces -> resolve -> arrival_face) with a
    # uniform-random policy. If random walks survive to max_steps, the board
    # cannot strand the VLM agent for mechanical reasons.
    _rng = np.random.default_rng(42)
    _N_WALKS, _MAX_STEPS = 500, 40
    _records = []
    for _w in range(_N_WALKS):
        _rid = node_ids[int(_rng.integers(0, len(node_ids)))]
        _backtrack, _dist, _steps, _term = "", 0.0, 0, "max_steps"
        _seen = {_rid}
        for _ in range(_MAX_STEPS):
            _legal = graph.legal_faces(_rid, exclude_face=_backtrack)
            if not _legal:
                _term = "dead_end"
                break
            _face = _legal[int(_rng.integers(0, len(_legal)))]
            _nb = graph.resolve_face_to_neighbor(_rid, _face)
            if _nb is None:
                _term = "resolve_failed"  # must never happen: menu == legal moves
                break
            _backtrack = graph.arrival_face(_rid, _nb.recording_id)
            _rid = _nb.recording_id
            _dist += _nb.distance_m
            _steps += 1
            _seen.add(_rid)
        _records.append({"steps": _steps, "distance_m": _dist,
                         "unique_nodes": len(_seen), "termination": _term})
    walks_df = pd.DataFrame(_records)

    _term_counts = walks_df["termination"].value_counts().to_dict()
    mo.vstack([
        mo.md(f"## 8. Random-Walk Simulation\n\n{_N_WALKS} walks x {_MAX_STEPS} steps "
              f"with the real mechanics and a random policy.\n\n"
              f"- Terminations: `{_term_counts}` — `resolve_failed` must be absent; "
              f"`dead_end` should be ~0 (turnaround is always offered)\n"
              f"- Mean distance: **{walks_df['distance_m'].mean():.0f}m**, "
              f"mean unique nodes: {walks_df['unique_nodes'].mean():.1f} / "
              f"{walks_df['steps'].mean():.1f} steps"),
        mo.ui.table(walks_df.describe().round(2).reset_index(), selection=None),
    ])
    return (walks_df,)


@app.cell
def _(folium, graph, node_ids, np):
    # --- 9. Sample walk trajectories on the map --------------------------------
    _rng = np.random.default_rng(7)
    walk_map = folium.Map(
        location=[float(np.mean([graph.coords[r][0] for r in node_ids[:1000]])),
                  float(np.mean([graph.coords[r][1] for r in node_ids[:1000]]))],
        zoom_start=13, tiles="cartodbpositron",
    )
    _palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    for _w in range(6):
        _rid = node_ids[int(_rng.integers(0, len(node_ids)))]
        _backtrack = ""
        _path = [graph.coords[_rid]]
        for _ in range(40):
            _legal = graph.legal_faces(_rid, exclude_face=_backtrack)
            if not _legal:
                break
            _face = _legal[int(_rng.integers(0, len(_legal)))]
            _nb = graph.resolve_face_to_neighbor(_rid, _face)
            if _nb is None:
                break
            _backtrack = graph.arrival_face(_rid, _nb.recording_id)
            _rid = _nb.recording_id
            _path.append(graph.coords[_rid])
        folium.PolyLine(_path, color=_palette[_w % len(_palette)], weight=3,
                        opacity=0.8, tooltip=f"walk {_w}: {len(_path) - 1} steps").add_to(walk_map)
        folium.CircleMarker(_path[0], radius=5, color=_palette[_w % len(_palette)],
                            fill=True, tooltip=f"walk {_w} start").add_to(walk_map)
    walk_map
    return


@app.cell
def _(diag, mo, walks_df):
    # --- 10. Verdict ------------------------------------------------------------
    _ok = (
        diag["graph/n_components"] == 1
        and diag["graph/n_isolated"] == 0
        and (walks_df["termination"] == "resolve_failed").sum() == 0
    )
    mo.md(
        f"""
        ## 10. Verdict

        {"✅ **Board is game-ready**: single component, no isolated nodes, and the menu never lies — every offered move resolves." if _ok else "❌ **Board has problems** — see the failing sections above."}
        """
    )
    return


if __name__ == "__main__":
    app.run()
