---
title: "UrbanRoamVQA — Agent Street Traversal"
category: dagspace
created: 2026-04-06
updated: 2026-06-09
tags:
  - dagspace
  - roaming
  - agent
  - navigation
  - street-graph
  - vlm-agent
---

# UrbanRoamVQA — Agent Street Traversal

UrbanRoamVQA is the dagspace for **VLM-driven agent navigation through urban street networks**. A vision-language model acts as an agent that "walks" through a city by viewing panoramic face images at each location, choosing a direction, and moving to the corresponding neighbor in a street graph. This enables exploration-based urban analysis from persona-driven perspectives (tourist, safety auditor, urban planner, etc.).

## Purpose

- Simulate agent-driven walks through street networks using VLM decision-making
- Present the panoramic face images of the **legal moves** (1-4 panels) at each step for direction selection
- Traverse a **board-first street graph**: a uniform, 100%-connected board built from the OSM street network with imagery attached to board positions (see [[concept-street-graph]])
- Support multiple agent personas via prompt configuration
- Produce walk trace datasets for spatial behavior analysis

## Key Files

| File | Role |
|------|------|
| `dagspaces/urbanroamvqa/cli.py` | Hydra CLI entry point |
| `dagspaces/urbanroamvqa/orchestrator.py` | DAG execution engine; defines `RoamingVQARunner(StageRunner)`, diagnostics |
| `dagspaces/urbanroamvqa/stages/roaming_vqa.py` | Core roaming stage: `run_roaming_vqa_stage()`, `RoamingStepper`, walk state management |
| `dagspaces/urbanroamvqa/graph/street_graph.py` | `StreetGraph` dataclass, `Neighbor`, face bearing math, `resolve_face_to_neighbor`, `legal_faces`, `compute_graph_diagnostics` |
| `dagspaces/urbanroamvqa/graph/board_builder.py` | **Canonical** board-first builder (`graph_type: board`): OSM board + imagery attachment + QA gate |
| `dagspaces/urbanroamvqa/graph/builder.py` | `build_street_graph()` dispatcher + legacy image-coordinate strategies |
| `dagspaces/urbanroamvqa/graph/cache.py` | Config-fingerprinted pickle cache for prebuilt graphs |
| `dagspaces/urbanroamvqa/samplers/seed_sampler.py` | `sample_walk_seeds()` with random, spatial_stratified, manual strategies |
| `dagspaces/urbanroamvqa/config_schema.py` | Roaming-specific config dataclasses |

## Walk Simulation Loop

The core loop runs iteratively, with each step requiring a VLM inference call:

```
1. sample_walk_seeds(graph, n_walks, strategy)
   -> Select starting recording_ids

2. For step_n in range(max_steps):
   a. RoamingStepper.prepare_step_batch()
      -> For each active walk:
         - legal_faces(): faces that resolve to a real neighbor, excluding
           the backtrack face (offered alone only at dead ends) and visited
           neighbors when allow_revisits=false
         - Drop faces whose image files are missing (walk dies only if NONE remain)
         - _stitch_faces(paths, labels, max_height) -> composite image (1-4 panels)
         - Render prompt with persona, history, direction choices
      -> Return DataFrame of VQA inference rows

   b. run_vqa_stage(batch, cfg)  [reuses UrbanVQA engine]
      -> VLM sees the composite of legal moves, chooses direction

   c. RoamingStepper.advance_from_answers(answers_df)
      -> Parse VLM answer to extract face choice
      -> resolve_face_to_neighbor(recording_id, face)
         -> Convert face to absolute bearing via yaw + offset
         -> Find closest neighbor within bearing_tolerance
      -> Update walk state: move to neighbor, increment step_n
      -> Check termination conditions

   d. save_checkpoint() [periodic]

3. Collect all walk traces -> output parquet
```

## StreetGraph

The `StreetGraph` dataclass in `dagspaces/urbanroamvqa/graph/street_graph.py` represents recording-level adjacency in the street network.

### Data Structure

```python
@dataclass
class Neighbor:
    recording_id: str       # Adjacent recording
    distance_m: float       # Haversine distance in meters
    bearing_deg: float      # Absolute bearing from source to neighbor (0-360)

@dataclass
class StreetGraph:
    adjacency: Dict[str, List[Neighbor]]         # recording_id -> neighbors
    coords: Dict[str, Tuple[float, float]]        # recording_id -> (lat, lon)
    yaw_degrees: Dict[str, float]                 # recording_id -> camera yaw
```

### Face Bearing System

Each panoramic recording has 4 horizontal faces (F, R, B, L). Bearing interpretation depends on `StreetGraph.face_frame`:

| Face | Bearing | `absolute` frame (default) | `relative` frame |
|------|---------|---------------------------|------------------|
| `F` | 0 degrees | North | Camera's forward direction |
| `R` | 90 degrees | East | Camera's right |
| `B` | 180 degrees | South | Camera's rear |
| `L` | 270 degrees | West | Camera's left |

**Cyclomedia NYC cube faces are compass-fixed** (camera orientation ~0 degrees on 100% of rows; verified 2026-04-22 in [[cyclomedia-catalog]]), so the default `face_frame: absolute` ignores recorder yaw in face math. Set `graph.face_frame: relative` only for vehicle-oriented panoramas.

### Face-to-Neighbor Resolution

`resolve_face_to_neighbor(recording_id, face, bearing_tolerance_deg=45.0)`:

1. Compute the face's absolute bearing via `face_bearing()` (frame-aware)
2. For each neighbor, compute angular difference to target bearing
3. Return the closest neighbor within `bearing_tolerance_deg` (default 45 degrees)
4. Return `None` if no neighbor is close enough

`legal_faces(recording_id, exclude_face, bearing_tolerance_deg, exclude_ids)` builds the move menu shown to the agent: only faces that resolve to a real neighbor, excluding the backtrack face (returned alone when it's the only way out of a dead end) and any `exclude_ids` (visited nodes when revisits are off). **The menu always equals the set of legal moves** — the agent can never pick a direction that doesn't exist.

**Per-step guided decoding (added 2026-06-09):** each batch row carries a `guided_decoding` payload — the prompt's `structured_output.json_schema` with `chosen_face.enum` narrowed to that step's legal faces (`RoamingStepper._guided_payload_for_faces`). The urbanvqa preprocess honors row-level payloads over the cfg-level schema (`vqa.py _make_preprocess`), and `run_vllm_inference`'s single-process path already passes per-row `SamplingParams` lists to `llm.generate()`. Illegal faces are therefore *unrepresentable* in the model output. Caveats: (a) before this, roaming's `prompt.structured_output` never reached vLLM at all (vqa.py only reads `sampling_params_vqa.structured_output`); (b) the cfg-level path collapses enum-bearing schemas to a bare `choice` constraint — the per-row payload deliberately bypasses that; (c) the DP-worker path (`model.concurrency > 1`) uses the first row's sampling params — roaming pins `concurrency: 1`.

`arrival_face(from_id, to_id)` returns the **backtrack face** at `to_id` (the face pointing back toward `from_id`); it is excluded on the next step. `""` means no exclusion (walk seeds).

## Graph Building

`build_street_graph()` in `dagspaces/urbanroamvqa/graph/builder.py` constructs a StreetGraph from recording metadata. The canonical strategy is **board** (`graph/board_builder.py`); the rest build the network from image coordinates and are kept for comparison experiments only.

| Strategy | Config `graph_type` | Description |
|----------|-------------------|-------------|
| **Board** | `"board"` | **Canonical.** Board-first: OSM network → consolidate intersections → largest component → discretize edges at uniform `spacing_m` → attach imagery to board nodes (greedy unique nearest, heading-aligned) → contract imageless nodes → QA gate asserts exactly 1 connected component. Uniform, walkable, 100% connected by construction. |
| **Trajectory** | `"trajectory"` | Reconstructs capture vehicle passes from `recordedAt` timestamps, chains sequential recordings, scores cross-pass edges with the Street Smart formula (all 4 directions), bridges components (warns if any remain beyond `bridge_radius_m`). |
| **KNN** | `"knn"` | K-nearest-neighbors via `scipy.cKDTree`; OSM-edge-constrained by default (raises if osmnx missing — no silent fallback) |
| **OSM Projected** | `"osm"` | OSMNX-projected graph with snapping to nearest road |
| **H3** | `"h3"` | H3 hexagonal grid-based adjacency |
| **Intersection** | `"intersection"` | Intersection-node-based adjacency from OSMNX |

### Board Builder (`graph_type: board`)

`build_board_graph()` in `dagspaces/urbanroamvqa/graph/board_builder.py`:

1. **OSM backbone** — `graph_from_place` (or cached graphml), project, `consolidate_intersections` (merges divided roads / dual carriageways, `consolidate_tolerance_m`), convert to undirected, keep largest component
2. **Discretize** — every edge geometry interpolated at `spacing_m` (default 25m); board nodes = OSM junctions + interpolated points; edges carry street distance
3. **Attach imagery** — globally-greedy unique assignment of recordings to board nodes within `max_snap_dist_m`, scored by distance + heading-alignment penalty (`heading_penalty_m`); each recording used at most once
4. **Contract imageless nodes** — neighbors reconnected through them with summed street distance; reconnections longer than `max_contracted_edge_m` are skipped (no teleport steps); board re-trimmed to largest component
5. **QA gate** — raises unless the final graph is exactly 1 connected component and retains ≥ `min_main_component_frac` of imaged nodes

The final StreetGraph is keyed by `recording_id` (image lookup unchanged), but coords are board positions and bearings follow street geometry. Caches are config-fingerprinted (`graph/cache.py`) — a changed config invalidates the pickle automatically.

**Canonical Manhattan board (built 2026-06-09** via `scripts/build_roam_board_graph.py`, rebuilt same day after the geometry-orientation fix): 36,317 nodes, 39,827 edges, 1 component, 99.2% imagery coverage, pitch p10/p50/p90 = 23/25/27m, 3,657 intersections, 104 dead ends. Artifacts: graph cache `data/graphs/roamvqa_board_25m_manhattan_2025_1.pkl`, OSM cache `data/osm/manhattan_2025_drive.graphml` (both referenced by `conf/graph/board_25m.yaml`). Validation: 500 random walks over the real mechanics all reach max steps (no dead ends, no resolve failures, mean step 24.7m); 2.1% of directed edges are face-shadowed at degree>4 consolidated intersections (the 4-face menu can't express more than 4 moves — known limitation). Exploratory validation lives in the marimo notebook `notebooks/roaming/network_validation.py` (supersedes `network_validation.ipynb`).

> **Geometry-orientation pitfall (fixed 2026-06-09):** `ox.convert.to_undirected` leaves edge geometry direction arbitrary relative to the `(u, v)` order that `G.edges()` yields — 53% of Manhattan edges were flipped, so chain interpolation anchored at `u` produced junction hops spanning the whole edge (up to 2.4km) and ~7k same-bearing shadowed corridor nodes. `_discretize_edges` now re-anchors the geometry at `u` before interpolating (regression test: `test_flipped_geometry_is_reanchored`).

### Trajectory Graph (Street Smart-Style)

The trajectory graph builder (`build_trajectory_graph()`) reverse-engineers how the Cyclomedia Street Smart web viewer computes navigation between panoramas. Instead of spatial heuristics, it uses the actual capture sequence:

1. **Pass segmentation:** Sort recordings by `recordedAt`, split into vehicle passes when time gap > 10s, distance > 100m, or heading change > 90 degrees
2. **Along-pass edges:** Chain consecutive recordings within each pass (forward/backward). These are ~5m apart at ~0.5-1s intervals.
3. **Intersection edges:** Find recordings from different passes within 15m that have heading difference > 30 degrees (turn options at crossings)

**Required columns:** `recordedAt`, `recorderDirection` (from Cyclomedia WFS catalog). See [[concept-streetsmart-api]] for details on the API.

**Reference:** `dagspaces/urbanroamvqa/graph/STREETSMART_API_REFERENCE.md`

### Graph Parameters (board)

| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| Spacing | `graph.spacing_m` | 25 | Target board pitch in meters |
| Network type | `graph.osm_network_type` | `drive` | OSM network (imagery is roadway-captured) |
| Consolidate tolerance | `graph.consolidate_tolerance_m` | 15 | Merge intersection clusters within this radius |
| Face frame | `graph.face_frame` | `absolute` | `absolute` (compass-fixed faces) or `relative` (yaw-offset) |
| Max snap distance | `graph.max_snap_dist_m` | 30 | Max recording-to-board-node attach distance |
| Heading penalty | `graph.heading_penalty_m` | 10 | Score penalty for street-misaligned recordings |
| Max contracted edge | `graph.max_contracted_edge_m` | 100 | Longest reconnection allowed through imageless gaps |
| Min component frac | `graph.min_main_component_frac` | 0.9 | QA gate: min fraction of imaged nodes retained |
| Bearing tolerance | `graph.bearing_tolerance_deg` | 45.0 | Max angular difference for face-to-neighbor matching |
| Precomputed path | `graph.precomputed_path` | None | Config-fingerprinted pickle cache |

(Legacy builders keep their own params — `k_neighbors`, `max_radius_m`, `pass_*`, `subsample_spacing_m`, etc. — see the respective `conf/graph/*.yaml`.)

### Graph Config Files (`dagspaces/urbanroamvqa/conf/graph/`)

| Config | Description |
|--------|-------------|
| `board_25m.yaml` | **Default/canonical.** Board-first, 25m pitch, QA-gated connectivity |
| `trajectory.yaml` | Trajectory-based, Street Smart-style (comparison only) |
| `osmnx_knn_10.yaml` | OSMNX-constrained KNN, k=10 (comparison only) |
| `osm_25m.yaml` | OSM-projected, 25m subsampling (comparison only) |
| `h3_res12.yaml` | H3 grid at resolution 12 (comparison only) |
| `intersection_25m.yaml` | Intersection-based, 25m subsampling (comparison only) |

## Seed Sampling

`sample_walk_seeds()` in `dagspaces/urbanroamvqa/samplers/seed_sampler.py` selects starting locations for walks.

| Strategy | Description |
|----------|-------------|
| `"random"` | Uniform random selection from eligible recordings (those with >= `min_neighbors` neighbors) |
| `"spatial_stratified"` | Grid-based spatial stratification for geographic diversity; divides bounding box into cells and samples from each |
| `"manual"` | Explicit list of recording_ids provided via `manual_seeds` |

**Output DataFrame columns:** `walk_id`, `seed_recording_id`, `seed_face`, `lat`, `lon`

## Walk State

### WalkStep Dataclass

Each step in a walk records:

| Field | Type | Description |
|-------|------|-------------|
| `step_n` | int | Step number within the walk |
| `recording_id` | str | Current recording location |
| `arrival_face` | str | Backtrack face (points back where the agent came from; "" at seeds) |
| `faces_shown` | List[str] | Legal faces presented to the VLM (1-4) |
| `face_chosen` | str | Face selected by the VLM |
| `reasoning` | str | VLM's reasoning text |
| `lat`, `lon` | float | Geographic coordinates |
| `bearing_deg` | float | Bearing to next location |
| `next_recording_id` | str | Recording moved to |
| `distance_m` | float | Distance traveled in meters |
| `termination_reason` | str | Why the walk ended (if applicable) |
| `answer_raw` | str | Raw VLM response |

### WalkState Dataclass

| Field | Type | Description |
|-------|------|-------------|
| `walk_id` | str | Unique walk identifier (e.g., `walk_000042`) |
| `current_recording_id` | str | Current location |
| `current_arrival_face` | str | Face of arrival at current location |
| `step_n` | int | Current step count |
| `active` | bool | Whether the walk is still ongoing |
| `history` | List[WalkStep] | Full step history |
| `visited` | set | Set of visited recording_ids |

### Checkpoint Support

Walks can be checkpointed to disk and resumed:

- `save_checkpoint(path, walks, step_iter, max_steps)` -- atomic write via temp file + rename
- `load_checkpoint(path)` -- returns checkpoint dict or None if corrupt/missing
- Checkpoint format: JSON with version, step_iter, max_steps, timestamp, serialized walk states

### Termination Conditions

Walks terminate when:

- `max_steps` reached
- `dead_end`: no legal moves at all (with revisits allowed)
- `no_unvisited_moves`: every legal move leads to a visited node (revisits disallowed) — illegal moves are filtered from the menu, never offered and then punished
- `missing_images`: none of the legal faces have image files on disk (single missing faces are dropped from the menu, not fatal)
- `stop`: agent chose to stop (independent mode)
- `revisit_blocked` / post-hoc `dead_end`: safety nets in `advance_from_answers` that should not trigger now that the menu is pre-filtered

## Image Stitching

`_stitch_faces(paths, labels, max_height=512)` creates a 1-4 panel composite image:

1. Load the legal-face images as RGB PIL Images
2. Scale each to `max_height` preserving aspect ratio
3. Horizontally concatenate onto a white canvas
4. Overlay direction labels (North/East/South/West in the absolute frame) with white text and black outline
5. Return as numpy array

The VLM sees a single composite image with labeled panels and must choose one direction.

## Diagnostics

The orchestrator (`_roaming_diagnostics()`) computes walk quality metrics:

| Metric | Description |
|--------|-------------|
| `total_steps` | Total steps across all walks |
| `n_walks` | Number of walks completed |
| `walk_length_mean/median/std/min/max` | Walk length distribution |
| `unique_recordings_visited` | Number of distinct locations visited |
| `revisit_rate` | Fraction of steps that revisit a previous location |
| `face_pref/{F,R,B,L}` | Direction preference distribution |
| `total_distance_mean/median_m` | Total walk distance statistics |
| `termination/{reason}` | Distribution of walk termination reasons |

Graph-level board-quality metrics (`compute_graph_diagnostics()` in `street_graph.py`, merged into the same diagnostics table):

| Metric | Description |
|--------|-------------|
| `graph/n_nodes`, `graph/n_edges` | Board size |
| `graph/n_components`, `graph/largest_component_frac` | Connectivity (must be 1 / 1.0 for board graphs) |
| `graph/degree_mean/median/max` | Degree distribution |
| `graph/n_isolated`, `graph/n_dead_ends`, `graph/n_intersections` | Topology counts |
| `graph/edge_m_median/p10/p90` | Pitch uniformity |

The orchestrator also runs `_validate_config()` before any expensive work (parquet existence, termination mode, max_steps, bearing tolerance range).

## Configuration

### Roaming Config

| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| N walks | `roaming.n_walks` | varies | Number of walks to simulate |
| Max steps | `roaming.max_steps` | 10 | Maximum steps per walk |
| Termination mode | `roaming.termination_mode` | `"fixed"` | How walks end |
| Allow revisits | `roaming.allow_revisits` | `true` | Can agent revisit locations |
| Include history | `roaming.include_history_in_prompt` | `true` | Show walk history in prompt |
| History max steps | `roaming.history_max_steps` | 5 | Max history steps in prompt |
| Stitch max height | `roaming.stitch_max_height` | 512 | Composite image height |

### Prompt Configs (`dagspaces/urbanroamvqa/conf/prompt/`)

| Config | Persona |
|--------|---------|
| `tourist.yaml` | Tourist exploring the city |
| `tourist_independent.yaml` | Independent tourist with free exploration |
| `safety_auditor.yaml` | Safety auditor inspecting hazards |
| `accessibility_surveyor.yaml` | Accessibility surveyor checking mobility |
| `greenery_seeker.yaml` | Agent seeking green spaces and vegetation |
| `urban_planner.yaml` | Urban planner assessing infrastructure |

### Pipeline Configs (`dagspaces/urbanroamvqa/conf/pipeline/`)

| Config | Description |
|--------|-------------|
| `roam_tourist_fixed_10.yaml` | Tourist persona, 10 fixed-length walks |
| `roam_tourist_independent_20.yaml` | Independent tourist, 20 walks |

### Data Config

| Config | Description |
|--------|-------------|
| `conf/data/cyclomedia_manhattan_2025.yaml` | Cyclomedia Manhattan 2025 recording metadata |

## Related Pages

- [[architecture]] -- overall pipeline architecture
- [[concept-street-graph]] -- street graph construction and traversal algorithms
- [[urban-vqa]] -- core VQA dagspace (roaming reuses `run_vqa_stage`)
- [[urban-pair-vqa]] -- pairwise dagspace (shares image stitching pattern)
- [[urban-ocr]] -- OCR dagspace
- [[urban-embed]] -- embedding dagspace
- [[shared-infrastructure]] -- common modules
