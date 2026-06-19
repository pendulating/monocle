---
title: "Street Graph Navigation"
category: concept
created: 2026-04-06
updated: 2026-06-09
tags:
  - concept
  - graph
  - navigation
  - osmnx
  - roaming
---

# Street Graph Navigation

How UrbanRoamVQA models street networks for multi-step traversal with vision-language models.

## Overview

UrbanRoamVQA simulates a street-level "roaming" agent that navigates through a city by choosing which direction to move at each step. The agent views panoramic imagery at each location and decides which face (N/E/S/W) to follow. The underlying street network is modeled as a `StreetGraph` adjacency structure.

**Design principle (since 2026-06-09): the board comes first.** The canonical builder (`graph_type: board`) constructs a uniform, 100%-connected board from the OSM street network and attaches imagery to board positions afterwards — think Monopoly over the NYC street grid. The legacy builders inverted this (nodes = image coordinates, edges inferred), which inherited every coverage gap and dual-lane artifact of the capture vehicle; they remain only for comparison experiments.

## Core Data Structures

### Key File: `dagspaces/urbanroamvqa/graph/street_graph.py`

### StreetGraph

The primary graph dataclass with three dictionaries:

| Field | Type | Description |
|-------|------|-------------|
| `adjacency` | `Dict[str, List[Neighbor]]` | Recording-level adjacency list. Keys are recording IDs, values are lists of neighboring recordings. |
| `coords` | `Dict[str, Tuple[float, float]]` | Mapping from `recording_id` to `(lat, lon)` coordinates. |
| `yaw_degrees` | `Dict[str, float]` | Camera yaw (heading) per recording in degrees. |

### Neighbor

Each neighbor in the adjacency list carries:

| Field | Type | Description |
|-------|------|-------------|
| `recording_id` | `str` | The neighboring recording's ID |
| `distance_m` | `float` | Distance from source to neighbor in meters |
| `bearing_deg` | `float` | Absolute bearing from source to neighbor (0-360 degrees) |

### Face System

Cyclomedia panoramas are divided into four horizontal faces. Empirically (verified 2026-04-22 against NYC imagery) these are rendered in a **globally-oriented absolute frame** — F=North, R=East, B=South, L=West — independent of the vehicle's direction of travel:

```
FACE_BEARING_DEG = {
    "F": 0.0,    # North
    "R": 90.0,   # East
    "B": 180.0,  # South
    "L": 270.0,  # West
}
```

**Resolved 2026-06-09:** `StreetGraph` now carries `face_frame` (default `"absolute"`), and all face math goes through `face_bearing(recording_id, face)`, which ignores recorder yaw in the absolute frame — matching the catalog-side fix in [[cyclomedia-catalog]]. `face_frame: "relative"` restores yaw-offset behavior for vehicle-oriented datasets. Prompt/stitch labels follow the frame (North/East/South/West vs Forward/Right/Behind/Left).

The constant `HORIZONTAL_FACES = ("F", "R", "B", "L")` enumerates valid face choices.

## Key Methods

### `neighbors(recording_id) -> List[Neighbor]`

Returns the list of neighbors for a recording, or an empty list if the recording is not in the graph.

### `resolve_face_to_neighbor(recording_id, face, bearing_tolerance_deg=45.0) -> Optional[Neighbor]`

Maps a face choice to the best matching neighbor:

1. Compute the absolute bearing for the face: `target = (yaw + FACE_BEARING_DEG[face]) % 360`
2. Find the neighbor whose bearing is closest to `target` and within `bearing_tolerance_deg`
3. Return that neighbor, or `None` if no neighbor is within tolerance

This is the core navigation function -- when the VLM chooses "R" (turn right), this resolves which neighboring recording that corresponds to.

### `arrival_face(from_id, to_id) -> str`

Returns the **backtrack face** at the destination: the face whose bearing points back toward the origin. Choosing it would retrace the move just made. Returns `""` when either node is unknown (no exclusion).

> Until 2026-06-09 this returned the *forward* face (direction of travel) and `available_faces` excluded it — agents could U-turn but never continue straight. Fixed: the backtrack face is excluded instead.

### `available_faces(arrival_face) -> List[str]`

Returns the horizontal faces excluding the backtrack face (`""` excludes nothing). Prevents immediate backtracking.

### `legal_faces(recording_id, exclude_face, bearing_tolerance_deg, exclude_ids) -> List[str]`

The move menu shown to the agent: faces that actually resolve to a neighbor, minus the backtrack face and `exclude_ids` (visited nodes when revisits are off). If only the backtrack face is resolvable, it is returned alone — walks turn around at dead ends instead of dying. **The menu always equals the set of legal moves.**

### `connected_components()` / `compute_graph_diagnostics(graph)`

Connectivity and board-quality metrics (components, degree distribution, pitch percentiles, dead ends, intersections) — logged by the orchestrator with every run and used by the board builder's QA gate.

## Bearing Math

### `_normalize_bearing(deg) -> float`

Normalizes any bearing to the `[0, 360)` range via modulo.

### `_bearing_diff(a, b) -> float`

Computes the smallest angular difference between two bearings (0-180 degrees). Handles the wrap-around at 0/360.

### `_face_for_bearing(bearing_deg, yaw_deg) -> str`

Returns the face whose absolute bearing (reference yaw + face offset) is closest to a given bearing. Used by `arrival_face()`, which passes yaw 0 in the absolute frame.

## Graph Building

### Key File: `dagspaces/urbanroamvqa/graph/builder.py`

### `build_street_graph(metadata_parquet, graph_cfg, use_osmnx=True) -> StreetGraph`

Main entry point for graph construction. Dispatches to a strategy based on `graph_cfg.graph_type`:

| Strategy | `graph_type` | Description |
|----------|-------------|-------------|
| **Board** | `"board"` | **Canonical** (`board_builder.py`). Board-first: OSM network → consolidate intersections → largest component → discretize at uniform `spacing_m` → attach imagery (greedy unique nearest + heading alignment) → contract imageless nodes (bounded by `max_contracted_edge_m`) → QA gate asserts exactly 1 component. |
| **Trajectory** | `"trajectory"` | Vehicle-pass chains + Street Smart-scored cross-pass edges (all 4 directions) + component bridging (warns honestly if components remain). |
| **KNN / OSMnx-constrained** | default | KNN by geodesic distance, filtered to same/adjacent OSM edges. Raises if osmnx is missing rather than silently degrading. Yaw normalized via `recorderDirection`. |
| **OSM-projected** | `"osm"` | Snap recordings to OSM edges, connect along edges and at shared intersection nodes. |
| **H3** | `"h3"` | One recording per occupied hex cell, adjacency from hex neighbors. |
| **Intersection** | `"intersection"` | OSM-based, subsamples recordings per edge, cross-street connections at OSM nodes. |

All builders cache via `graph_cfg.precomputed_path` using `graph/cache.py`: pickles carry a config fingerprint, so a changed config or input parquet rebuilds instead of silently loading a stale graph.

**Why board-first matters:** every legacy strategy makes recordings the nodes, so the board inherits imagery gaps (broken connectivity), ~5m capture pitch and dual-lane duplicates (non-uniform), and provides no connectivity guarantee. The board builder guarantees uniformity and single-component connectivity by construction and fails loudly (QA gate) when imagery coverage can't support the requested extent.

## Seed Sampling

### Key File: `dagspaces/urbanroamvqa/samplers/seed_sampler.py`

Walk starting points are selected by `sample_seed_recordings()`:

| Strategy | Description |
|----------|-------------|
| `random` | Uniform random selection from eligible recordings (those with at least `min_neighbors` neighbors) |
| `spatial_stratified` | Spatially stratified sampling to ensure geographic coverage |
| `manual` | Explicit list of recording IDs provided via `manual_seeds` parameter |

Each seed row includes:
- `seed_recording_id` -- starting recording
- `walk_id` -- unique walk identifier
- `seed_face` -- backtrack face excluded on the first step (`""` = show all legal faces, the default)
- `lat`, `lon` -- seed coordinates

## Walk Execution Flow

1. **Seed sampling** -- select starting recordings
2. **Legal moves** -- `legal_faces()` builds the menu: resolvable faces minus backtrack/visited
3. **Graph navigation** -- the VLM views the legal-face panels and chooses one
4. **Face resolution** -- `resolve_face_to_neighbor()` maps the chosen face to a concrete neighbor
5. **Arrival computation** -- `arrival_face()` records the backtrack face at the new location
6. **Repeat** until max steps, agent stop, or no legal moves remain

## See Also

- [[urban-roam-vqa]] -- the UrbanRoamVQA dagspace overview
