"""Unit tests for urbanroamvqa — street graph, stepper, checkpointing, and config validation."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from PIL import Image

from dagspaces.urbanroamvqa.graph.street_graph import (
    FACE_BEARING_DEG,
    HORIZONTAL_FACES,
    Neighbor,
    StreetGraph,
    _bearing_diff,
    _face_for_bearing,
    _normalize_bearing,
)
from dagspaces.urbanroamvqa.stages.roaming_vqa import (
    RoamingStepper,
    WalkState,
    WalkStep,
    _walk_state_from_dict,
    _walk_state_to_dict,
    _walk_step_from_dict,
    _walk_step_to_dict,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_simple_graph() -> StreetGraph:
    """3-node triangle graph with explicit bearings.

    Layout (N=0 is at top):
        N0 (lat=40.0, lon=-74.0, yaw=0)
        N1 (lat=40.001, lon=-74.0, yaw=0)  — north of N0
        N2 (lat=40.0, lon=-73.999, yaw=90) — east of N0
    """
    adjacency = {
        "N0": [
            Neighbor(recording_id="N1", distance_m=111.0, bearing_deg=0.0),   # due north
            Neighbor(recording_id="N2", distance_m=88.0,  bearing_deg=90.0),  # due east
        ],
        "N1": [
            Neighbor(recording_id="N0", distance_m=111.0, bearing_deg=180.0),
            Neighbor(recording_id="N2", distance_m=140.0, bearing_deg=135.0),
        ],
        "N2": [
            Neighbor(recording_id="N0", distance_m=88.0,  bearing_deg=270.0),
            Neighbor(recording_id="N1", distance_m=140.0, bearing_deg=315.0),
        ],
    }
    coords = {
        "N0": (40.000, -74.000),
        "N1": (40.001, -74.000),
        "N2": (40.000, -73.999),
    }
    yaw_degrees = {
        "N0": 0.0,
        "N1": 0.0,
        "N2": 90.0,
    }
    return StreetGraph(adjacency=adjacency, coords=coords, yaw_degrees=yaw_degrees)


def _make_stepper_cfg(
    max_steps: int = 5,
    termination_mode: str = "fixed",
    system_prompt: str = "test",
    user_template: str = "",
    bearing_tolerance: float = 45.0,
    include_history: bool = False,
    structured_schema: dict | None = None,
) -> "OmegaConf":
    prompt_cfg = {
        "system": system_prompt,
        "user_template": user_template,
    }
    if structured_schema is not None:
        prompt_cfg["structured_output"] = {
            "enabled": True,
            "json_schema": structured_schema,
        }
    return OmegaConf.create({
        "roaming": {
            "max_steps": max_steps,
            "termination_mode": termination_mode,
            "allow_revisits": True,
            "include_history_in_prompt": include_history,
            "history_max_steps": 3,
            "stitch_max_height": 64,
        },
        "graph": {
            "bearing_tolerance_deg": bearing_tolerance,
        },
        "prompt": prompt_cfg,
    })


_ROAM_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_face": {"type": "string", "enum": ["F", "R", "B", "L"]},
        "reasoning": {"type": "string"},
        "stop": {"type": "boolean"},
    },
    "required": ["chosen_face", "reasoning"],
    "additionalProperties": True,
}


def _make_walks(recording_id: str = "N0", arrival_face: str = "") -> List[WalkState]:
    return [WalkState(walk_id="w0", current_recording_id=recording_id, current_arrival_face=arrival_face)]


# ---------------------------------------------------------------------------
# 1. StreetGraph helper functions
# ---------------------------------------------------------------------------

class TestNormalizeBearing:
    """Tests for _normalize_bearing."""

    def test_zero(self):
        assert _normalize_bearing(0.0) == pytest.approx(0.0)

    def test_full_circle(self):
        assert _normalize_bearing(360.0) == pytest.approx(0.0)

    def test_negative(self):
        assert _normalize_bearing(-90.0) == pytest.approx(270.0)

    def test_over_360(self):
        assert _normalize_bearing(450.0) == pytest.approx(90.0)

    def test_large_negative(self):
        assert _normalize_bearing(-361.0) == pytest.approx(359.0)

    def test_identity(self):
        assert _normalize_bearing(180.0) == pytest.approx(180.0)


class TestBearingDiff:
    """Tests for _bearing_diff."""

    def test_same_bearing(self):
        assert _bearing_diff(45.0, 45.0) == pytest.approx(0.0)

    def test_opposite_bearings(self):
        assert _bearing_diff(0.0, 180.0) == pytest.approx(180.0)

    def test_wraparound(self):
        """350° vs 10° should give 20°, not 340°."""
        assert _bearing_diff(350.0, 10.0) == pytest.approx(20.0)

    def test_orthogonal(self):
        assert _bearing_diff(0.0, 90.0) == pytest.approx(90.0)

    def test_symmetric(self):
        assert _bearing_diff(10.0, 350.0) == pytest.approx(_bearing_diff(350.0, 10.0))

    def test_negative_inputs(self):
        """Inputs may be negative bearings from calculations."""
        assert _bearing_diff(-90.0, 270.0) == pytest.approx(0.0)


class TestFaceForBearing:
    """Tests for _face_for_bearing."""

    def test_forward_face_yaw_zero(self):
        """When yaw=0, bearing 0° (north) maps to Forward."""
        assert _face_for_bearing(0.0, yaw_deg=0.0) == "F"

    def test_right_face_yaw_zero(self):
        """When yaw=0, bearing 90° (east) maps to Right."""
        assert _face_for_bearing(90.0, yaw_deg=0.0) == "R"

    def test_behind_face_yaw_zero(self):
        """When yaw=0, bearing 180° (south) maps to Behind."""
        assert _face_for_bearing(180.0, yaw_deg=0.0) == "B"

    def test_left_face_yaw_zero(self):
        """When yaw=0, bearing 270° (west) maps to Left."""
        assert _face_for_bearing(270.0, yaw_deg=0.0) == "L"

    def test_forward_face_rotated_yaw(self):
        """When yaw=90, absolute forward is east (90°). Bearing 90° maps to F."""
        assert _face_for_bearing(90.0, yaw_deg=90.0) == "F"

    def test_right_face_rotated_yaw(self):
        """When yaw=90, R = 90+90=180°. Bearing 180° maps to R."""
        assert _face_for_bearing(180.0, yaw_deg=90.0) == "R"

    def test_ambiguous_boundary_resolves_deterministically(self):
        """Bearing exactly midway between two faces picks one consistently."""
        face = _face_for_bearing(45.0, yaw_deg=0.0)
        assert face in HORIZONTAL_FACES


# ---------------------------------------------------------------------------
# 2. StreetGraph methods
# ---------------------------------------------------------------------------

class TestStreetGraphResolveFace:
    """Tests for StreetGraph.resolve_face_to_neighbor."""

    def setup_method(self):
        self.graph = _make_simple_graph()

    def test_resolve_forward_north(self):
        """N0 has yaw=0; Forward (0°) should find N1 at bearing 0°."""
        result = self.graph.resolve_face_to_neighbor("N0", "F", bearing_tolerance_deg=45.0)
        assert result is not None
        assert result.recording_id == "N1"

    def test_resolve_right_east(self):
        """N0 has yaw=0; Right (90°) should find N2 at bearing 90°."""
        result = self.graph.resolve_face_to_neighbor("N0", "R", bearing_tolerance_deg=45.0)
        assert result is not None
        assert result.recording_id == "N2"

    def test_returns_none_when_no_neighbor_in_tolerance(self):
        """Behind face for N0 (180°) has no neighbor in that direction."""
        result = self.graph.resolve_face_to_neighbor("N0", "B", bearing_tolerance_deg=10.0)
        assert result is None

    def test_returns_none_for_unknown_recording_id(self):
        result = self.graph.resolve_face_to_neighbor("NONEXISTENT", "F")
        assert result is None

    def test_returns_none_for_unknown_face(self):
        result = self.graph.resolve_face_to_neighbor("N0", "X")
        assert result is None

    def test_returns_none_when_no_neighbors(self):
        empty_graph = StreetGraph(
            adjacency={"lone": []},
            coords={"lone": (40.0, -74.0)},
            yaw_degrees={"lone": 0.0},
        )
        result = empty_graph.resolve_face_to_neighbor("lone", "F")
        assert result is None

    def test_wide_tolerance_finds_neighbor(self):
        """With a very tight bearing (facing slightly off), wide tolerance still resolves."""
        result = self.graph.resolve_face_to_neighbor("N0", "F", bearing_tolerance_deg=90.0)
        assert result is not None

    def test_zero_tolerance_exact_match(self):
        """Zero tolerance requires exact bearing match."""
        result = self.graph.resolve_face_to_neighbor("N0", "F", bearing_tolerance_deg=0.0)
        # N1 is at exactly 0° bearing and Forward = 0° for N0 yaw=0; exact match expected
        assert result is not None
        assert result.recording_id == "N1"


class TestFaceFrame:
    """Cyclomedia NYC faces are compass-fixed (absolute frame); yaw-relative
    resolution remains available via face_frame='relative'."""

    def _graph(self, face_frame: str) -> StreetGraph:
        adjacency = {
            "X": [
                Neighbor(recording_id="NORTH", distance_m=20.0, bearing_deg=0.0),
                Neighbor(recording_id="EAST", distance_m=20.0, bearing_deg=90.0),
            ],
        }
        coords = {"X": (40.0, -74.0), "NORTH": (40.0002, -74.0), "EAST": (40.0, -73.9998)}
        # Vehicle was driving east (yaw=90) — must not affect absolute faces
        return StreetGraph(adjacency=adjacency, coords=coords,
                           yaw_degrees={"X": 90.0, "NORTH": 90.0, "EAST": 90.0},
                           face_frame=face_frame)

    def test_absolute_frame_ignores_yaw(self):
        graph = self._graph("absolute")
        result = graph.resolve_face_to_neighbor("X", "F")
        assert result is not None and result.recording_id == "NORTH"
        result = graph.resolve_face_to_neighbor("X", "R")
        assert result is not None and result.recording_id == "EAST"

    def test_relative_frame_applies_yaw(self):
        graph = self._graph("relative")
        # yaw=90: Forward points east
        result = graph.resolve_face_to_neighbor("X", "F")
        assert result is not None and result.recording_id == "EAST"
        # Left (270 offset) points north
        result = graph.resolve_face_to_neighbor("X", "L")
        assert result is not None and result.recording_id == "NORTH"

    def test_face_bearing_per_frame(self):
        assert self._graph("absolute").face_bearing("X", "R") == pytest.approx(90.0)
        assert self._graph("relative").face_bearing("X", "R") == pytest.approx(180.0)

    def test_default_frame_is_absolute(self):
        graph = StreetGraph()
        assert graph.face_frame == "absolute"


class TestStreetGraphArrivalFace:
    """Tests for StreetGraph.arrival_face (the backtrack face at the new node)."""

    def setup_method(self):
        self.graph = _make_simple_graph()

    def test_arrival_face_returns_string(self):
        face = self.graph.arrival_face("N0", "N1")
        assert face in HORIZONTAL_FACES

    def test_arrival_face_moving_north_is_behind(self):
        """Moving from N0 to N1 (northward), the backtrack face at N1 points
        south (back toward N0). N1 has yaw=0, so south is Behind ('B').
        Excluding it lets the agent continue forward but not retrace its step.
        """
        face = self.graph.arrival_face("N0", "N1")
        assert face == "B"

    def test_arrival_face_returns_empty_for_missing_coords(self):
        """When either node is unknown, no face is excluded ('')."""
        graph_missing = StreetGraph(
            adjacency={},
            coords={"A": (40.0, -74.0)},
            yaw_degrees={"A": 0.0},
        )
        face = graph_missing.arrival_face("A", "MISSING")
        assert face == ""

    def test_arrival_face_round_trip(self):
        """N0->N1 gives 'B' at N1 (south, toward N0); N1->N0 gives 'F' at N0
        (north, toward N1). They must differ."""
        face_fwd = self.graph.arrival_face("N0", "N1")
        face_rev = self.graph.arrival_face("N1", "N0")
        assert face_fwd == "B"
        assert face_rev == "F"


class TestStreetGraphAvailableFaces:
    """Tests for StreetGraph.available_faces."""

    def setup_method(self):
        self.graph = _make_simple_graph()

    def test_excludes_arrival_face(self):
        """available_faces excludes only the (backtrack) arrival face."""
        for arrival in HORIZONTAL_FACES:
            result = self.graph.available_faces(arrival)
            assert arrival not in result

    def test_returns_three_faces(self):
        """Returns exactly 3 faces for a real arrival face."""
        for arrival in HORIZONTAL_FACES:
            result = self.graph.available_faces(arrival)
            assert len(result) == 3

    def test_empty_arrival_returns_all_faces(self):
        """'' (walk seed) excludes nothing — all 4 faces available."""
        result = self.graph.available_faces("")
        assert result == list(HORIZONTAL_FACES)

    def test_all_returned_faces_are_valid(self):
        """All returned faces are members of HORIZONTAL_FACES."""
        result = self.graph.available_faces("F")
        assert set(result) <= set(HORIZONTAL_FACES)

    def test_no_duplicates(self):
        result = self.graph.available_faces("R")
        assert len(result) == len(set(result))


class TestStreetGraphLegalFaces:
    """Tests for StreetGraph.legal_faces — the menu must equal the legal moves."""

    def setup_method(self):
        self.graph = _make_simple_graph()

    def test_only_resolvable_faces_returned(self):
        """N0 (yaw=0) has neighbors due north (F) and due east (R) only."""
        result = self.graph.legal_faces("N0")
        assert result == ["F", "R"]

    def test_backtrack_face_excluded(self):
        result = self.graph.legal_faces("N0", exclude_face="F")
        assert result == ["R"]

    def test_visited_neighbors_excluded(self):
        result = self.graph.legal_faces("N0", exclude_ids={"N2"})
        assert result == ["F"]

    def test_dead_end_turnaround(self):
        """When the backtrack face is the only resolvable move, it is offered
        alone instead of stranding the walk."""
        linear = _build_linear_graph(2)  # N0 <-> N1 only
        result = linear.legal_faces("N0", exclude_face="F")
        assert result == ["F"]

    def test_dead_end_with_visited_backtrack_is_empty(self):
        """With revisits disallowed, a dead end whose only exit was visited
        yields no legal moves."""
        linear = _build_linear_graph(2)
        result = linear.legal_faces("N0", exclude_face="F", exclude_ids={"N1"})
        assert result == []


class TestGraphDiagnostics:
    """Tests for compute_graph_diagnostics."""

    def test_single_component_triangle(self):
        from dagspaces.urbanroamvqa.graph.street_graph import compute_graph_diagnostics

        diag = compute_graph_diagnostics(_make_simple_graph())
        assert diag["graph/n_nodes"] == 3
        assert diag["graph/n_components"] == 1
        assert diag["graph/largest_component_frac"] == pytest.approx(1.0)
        assert diag["graph/n_isolated"] == 0

    def test_linear_chain_dead_ends(self):
        from dagspaces.urbanroamvqa.graph.street_graph import compute_graph_diagnostics

        diag = compute_graph_diagnostics(_build_linear_graph(5))
        assert diag["graph/n_dead_ends"] == 2
        assert diag["graph/n_intersections"] == 0
        assert diag["graph/n_components"] == 1

    def test_disconnected_components_detected(self):
        from dagspaces.urbanroamvqa.graph.street_graph import compute_graph_diagnostics

        adjacency = {
            "A": [Neighbor(recording_id="B", distance_m=10.0, bearing_deg=0.0)],
            "B": [Neighbor(recording_id="A", distance_m=10.0, bearing_deg=180.0)],
            "C": [],
        }
        coords = {"A": (40.0, -74.0), "B": (40.0001, -74.0), "C": (41.0, -74.0)}
        graph = StreetGraph(adjacency=adjacency, coords=coords,
                            yaw_degrees={k: 0.0 for k in coords})
        diag = compute_graph_diagnostics(graph)
        assert diag["graph/n_components"] == 2
        assert diag["graph/n_isolated"] == 1
        assert diag["graph/largest_component_frac"] == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# 3. Answer parsing
# ---------------------------------------------------------------------------

class TestParseAnswer:
    """Tests for RoamingStepper._parse_answer."""

    def setup_method(self):
        graph = _make_simple_graph()
        walks = _make_walks()
        cfg = _make_stepper_cfg()
        self.stepper = RoamingStepper(graph, walks, cfg)
        self.available = ["F", "R", "L"]

    def test_valid_json_no_stop(self):
        raw = json.dumps({"chosen_face": "R", "reasoning": "nice view", "stop": False})
        face, reasoning, stop = self.stepper._parse_answer(raw, self.available)
        assert face == "R"
        assert reasoning == "nice view"
        assert stop is False

    def test_valid_json_with_stop(self):
        raw = json.dumps({"chosen_face": "F", "reasoning": "done", "stop": True})
        face, reasoning, stop = self.stepper._parse_answer(raw, self.available)
        assert face == "F"
        assert reasoning == "done"
        assert stop is True

    def test_markdown_wrapped_json(self):
        raw = "```json\n{\"chosen_face\": \"L\", \"reasoning\": \"quiet street\"}\n```"
        face, reasoning, stop = self.stepper._parse_answer(raw, self.available)
        assert face == "L"
        assert stop is False

    def test_markdown_no_lang_tag(self):
        raw = "```\n{\"chosen_face\": \"R\"}\n```"
        face, _reasoning, _stop = self.stepper._parse_answer(raw, self.available)
        assert face == "R"

    def test_malformed_json_face_in_text(self):
        """If JSON parsing fails, a face letter found in the text is used as fallback."""
        raw = "I choose R because the street looks interesting."
        face, reasoning, stop = self.stepper._parse_answer(raw, self.available)
        assert face == "R"
        assert stop is False
        assert "parse_fallback" in reasoning

    def test_completely_invalid_falls_back_to_first_available(self):
        # Use a string with no face letters (F/R/L) when uppercased.
        # "hello world" contains "R" in "WORLD", so use digits instead.
        raw = "5 + 5 = 10"
        face, _reasoning, _stop = self.stepper._parse_answer(raw, self.available)
        assert face == self.available[0]

    def test_face_not_in_available_falls_back(self):
        """chosen_face not in available_faces is ignored; fallback applies."""
        raw = json.dumps({"chosen_face": "B", "reasoning": "going back"})
        face, _reasoning, _stop = self.stepper._parse_answer(raw, self.available)
        # "B" is not in available=["F","R","L"], so fallback first element
        assert face == self.available[0]

    def test_case_insensitive_face_parsing(self):
        """JSON value "r" (lowercase) should be uppercased and matched."""
        raw = json.dumps({"chosen_face": "r", "reasoning": "lowercase"})
        face, _reasoning, _stop = self.stepper._parse_answer(raw, self.available)
        assert face == "R"

    def test_stop_defaults_to_false_when_missing(self):
        raw = json.dumps({"chosen_face": "F", "reasoning": "moving"})
        _face, _reasoning, stop = self.stepper._parse_answer(raw, self.available)
        assert stop is False


# ---------------------------------------------------------------------------
# 4. Checkpoint round-trip
# ---------------------------------------------------------------------------

class TestWalkStepSerialization:
    """Tests for _walk_step_to_dict / _walk_step_from_dict."""

    def _make_step(self) -> WalkStep:
        return WalkStep(
            step_n=3,
            recording_id="REC_42",
            arrival_face="L",
            faces_shown=["F", "R", "L"],
            face_chosen="R",
            reasoning="clear path ahead",
            lat=40.712,
            lon=-74.006,
            bearing_deg=87.5,
            next_recording_id="REC_43",
            distance_m=15.3,
            termination_reason=None,
            answer_raw='{"chosen_face": "R"}',
        )

    def test_round_trip_preserves_all_fields(self):
        original = self._make_step()
        d = _walk_step_to_dict(original)
        restored = _walk_step_from_dict(d)

        assert restored.step_n == original.step_n
        assert restored.recording_id == original.recording_id
        assert restored.arrival_face == original.arrival_face
        assert restored.faces_shown == original.faces_shown
        assert restored.face_chosen == original.face_chosen
        assert restored.reasoning == original.reasoning
        assert restored.lat == pytest.approx(original.lat)
        assert restored.lon == pytest.approx(original.lon)
        assert restored.bearing_deg == pytest.approx(original.bearing_deg)
        assert restored.next_recording_id == original.next_recording_id
        assert restored.distance_m == pytest.approx(original.distance_m)
        assert restored.termination_reason == original.termination_reason
        assert restored.answer_raw == original.answer_raw

    def test_round_trip_with_none_fields(self):
        step = WalkStep(step_n=0, recording_id="X", arrival_face="F", faces_shown=["F", "R", "L"])
        d = _walk_step_to_dict(step)
        restored = _walk_step_from_dict(d)
        assert restored.face_chosen is None
        assert restored.reasoning is None
        assert restored.bearing_deg is None
        assert restored.next_recording_id is None
        assert restored.distance_m is None
        assert restored.termination_reason is None
        assert restored.answer_raw is None

    def test_serialized_form_is_json_serializable(self):
        step = self._make_step()
        d = _walk_step_to_dict(step)
        # Must not raise
        encoded = json.dumps(d)
        assert isinstance(encoded, str)


class TestWalkStateSerialization:
    """Tests for _walk_state_to_dict / _walk_state_from_dict."""

    def _make_state(self) -> WalkState:
        ws = WalkState(
            walk_id="walk_000001",
            current_recording_id="REC_10",
            current_arrival_face="R",
            step_n=2,
            active=True,
        )
        ws.history = [
            WalkStep(step_n=0, recording_id="REC_00", arrival_face="F", faces_shown=["F", "R", "L"],
                     face_chosen="F", reasoning="first step"),
            WalkStep(step_n=1, recording_id="REC_10", arrival_face="F", faces_shown=["F", "R", "B"],
                     face_chosen="R", reasoning="second step"),
        ]
        ws.visited = {"REC_00", "REC_10"}
        return ws

    def test_round_trip_basic_fields(self):
        ws = self._make_state()
        d = _walk_state_to_dict(ws)
        restored = _walk_state_from_dict(d)

        assert restored.walk_id == ws.walk_id
        assert restored.current_recording_id == ws.current_recording_id
        assert restored.current_arrival_face == ws.current_arrival_face
        assert restored.step_n == ws.step_n
        assert restored.active == ws.active

    def test_round_trip_history_length(self):
        ws = self._make_state()
        d = _walk_state_to_dict(ws)
        restored = _walk_state_from_dict(d)
        assert len(restored.history) == len(ws.history)

    def test_round_trip_visited_set(self):
        ws = self._make_state()
        d = _walk_state_to_dict(ws)
        restored = _walk_state_from_dict(d)
        assert restored.visited == ws.visited

    def test_visited_serialized_as_sorted_list(self):
        """visited must be a list in the dict (not a set) for JSON compatibility."""
        ws = self._make_state()
        d = _walk_state_to_dict(ws)
        assert isinstance(d["visited"], list)
        assert d["visited"] == sorted(ws.visited)

    def test_empty_history_round_trip(self):
        ws = WalkState(walk_id="w0", current_recording_id="R0", current_arrival_face="F")
        d = _walk_state_to_dict(ws)
        restored = _walk_state_from_dict(d)
        assert restored.history == []
        assert restored.visited == set()


class TestCheckpointIO:
    """Tests for save_checkpoint / load_checkpoint."""

    def _make_walks_dict(self) -> Dict[str, WalkState]:
        ws = WalkState(walk_id="w0", current_recording_id="R0", current_arrival_face="F", step_n=1)
        ws.history = [
            WalkStep(step_n=0, recording_id="R0", arrival_face="F", faces_shown=["F", "R", "L"],
                     face_chosen="F", reasoning="moving"),
        ]
        ws.visited = {"R0"}
        return {ws.walk_id: ws}

    def test_save_load_round_trip(self, tmp_path):
        walks = self._make_walks_dict()
        checkpoint_path = str(tmp_path / "ckpt.json")
        save_checkpoint(checkpoint_path, walks, step_iter=3, max_steps=10)

        data = load_checkpoint(checkpoint_path)
        assert data is not None
        assert data["version"] == 1
        assert data["step_iter"] == 3
        assert data["max_steps"] == 10
        assert "w0" in data["walks"]

    def test_load_restores_walk_state(self, tmp_path):
        walks = self._make_walks_dict()
        checkpoint_path = str(tmp_path / "ckpt.json")
        save_checkpoint(checkpoint_path, walks, step_iter=2, max_steps=5)

        data = load_checkpoint(checkpoint_path)
        restored_ws = _walk_state_from_dict(data["walks"]["w0"])
        assert restored_ws.walk_id == "w0"
        assert restored_ws.step_n == 1
        assert "R0" in restored_ws.visited
        assert len(restored_ws.history) == 1

    def test_missing_checkpoint_returns_none(self, tmp_path):
        result = load_checkpoint(str(tmp_path / "does_not_exist.json"))
        assert result is None

    def test_corrupt_json_returns_none(self, tmp_path):
        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{this is not json", encoding="utf-8")
        result = load_checkpoint(str(corrupt))
        assert result is None

    def test_wrong_version_returns_none(self, tmp_path):
        bad = tmp_path / "bad_version.json"
        bad.write_text(json.dumps({"version": 99, "step_iter": 0, "max_steps": 5, "walks": {}}))
        result = load_checkpoint(str(bad))
        assert result is None

    def test_atomic_write_no_tmp_file_after_save(self, tmp_path):
        """The .tmp file must not persist after a successful save."""
        walks = self._make_walks_dict()
        checkpoint_path = str(tmp_path / "ckpt.json")
        save_checkpoint(checkpoint_path, walks, step_iter=0, max_steps=5)

        tmp_file = checkpoint_path + ".tmp"
        assert not os.path.exists(tmp_file), ".tmp file should be removed after atomic rename"
        assert os.path.exists(checkpoint_path), "Final checkpoint file must exist"

    def test_checkpoint_has_timestamp(self, tmp_path):
        walks = self._make_walks_dict()
        checkpoint_path = str(tmp_path / "ckpt.json")
        save_checkpoint(checkpoint_path, walks, step_iter=0, max_steps=5)

        data = load_checkpoint(checkpoint_path)
        assert "timestamp" in data
        assert isinstance(data["timestamp"], float)
        assert data["timestamp"] > 0.0

    def test_multiple_walks_preserved(self, tmp_path):
        ws1 = WalkState(walk_id="w1", current_recording_id="R1", current_arrival_face="F")
        ws2 = WalkState(walk_id="w2", current_recording_id="R2", current_arrival_face="R")
        walks = {ws1.walk_id: ws1, ws2.walk_id: ws2}
        checkpoint_path = str(tmp_path / "ckpt.json")
        save_checkpoint(checkpoint_path, walks, step_iter=1, max_steps=10)

        data = load_checkpoint(checkpoint_path)
        assert "w1" in data["walks"]
        assert "w2" in data["walks"]


# ---------------------------------------------------------------------------
# 5. Config validation
# ---------------------------------------------------------------------------

class TestValidateConfig:
    """Tests for _validate_config from urbanroamvqa.orchestrator."""

    # Import lazily to avoid heavy import at module load
    @staticmethod
    def _validate(cfg):
        from dagspaces.urbanroamvqa.orchestrator import _validate_config
        _validate_config(cfg)

    def _make_valid_cfg(self, tmp_parquet: str) -> "OmegaConf":
        """Build a minimal valid config pointing at a real parquet."""
        return OmegaConf.create({
            "data": {
                "parquet_path": tmp_parquet,
                "image_root": "/some/root",
            },
            "roaming": {
                "max_steps": 5,
                "termination_mode": "fixed",
                "n_walks": 10,
            },
            "graph": {
                "bearing_tolerance_deg": 45.0,
            },
        })

    def _write_minimal_parquet(self, path: str) -> None:
        """Write a parquet with the minimum required columns."""
        df = pd.DataFrame({
            "recording_id": ["R1"],
            "latitude": [40.0],
            "longitude": [-74.0],
            "yaw_deg": [0.0],
        })
        df.to_parquet(path, index=False)

    def test_valid_config_passes(self, tmp_path):
        parquet = str(tmp_path / "meta.parquet")
        self._write_minimal_parquet(parquet)
        cfg = self._make_valid_cfg(parquet)
        # Must not raise
        self._validate(cfg)

    def test_missing_parquet_path_fails(self, tmp_path):
        parquet = str(tmp_path / "meta.parquet")
        self._write_minimal_parquet(parquet)
        cfg = self._make_valid_cfg(parquet)
        OmegaConf.update(cfg, "data.parquet_path", "", merge=True)
        with pytest.raises(ValueError, match="parquet_path is not set"):
            self._validate(cfg)

    def test_nonexistent_parquet_fails(self, tmp_path):
        cfg = self._make_valid_cfg(str(tmp_path / "missing.parquet"))
        with pytest.raises(ValueError, match="does not exist"):
            self._validate(cfg)

    def test_invalid_termination_mode_fails(self, tmp_path):
        parquet = str(tmp_path / "meta.parquet")
        self._write_minimal_parquet(parquet)
        cfg = self._make_valid_cfg(parquet)
        OmegaConf.update(cfg, "roaming.termination_mode", "unknown_mode", merge=True)
        with pytest.raises(ValueError, match="termination_mode"):
            self._validate(cfg)

    def test_max_steps_below_one_fails(self, tmp_path):
        parquet = str(tmp_path / "meta.parquet")
        self._write_minimal_parquet(parquet)
        cfg = self._make_valid_cfg(parquet)
        OmegaConf.update(cfg, "roaming.max_steps", 0, merge=True)
        with pytest.raises(ValueError, match="max_steps"):
            self._validate(cfg)

    def test_bearing_tolerance_out_of_range_warns(self, tmp_path):
        """bearing_tolerance_deg outside 5-170 should be flagged in the error message."""
        parquet = str(tmp_path / "meta.parquet")
        self._write_minimal_parquet(parquet)
        cfg = self._make_valid_cfg(parquet)
        OmegaConf.update(cfg, "graph.bearing_tolerance_deg", 2.0, merge=True)
        with pytest.raises(ValueError, match="bearing_tolerance_deg"):
            self._validate(cfg)

    def test_missing_data_section_fails(self, tmp_path):
        parquet = str(tmp_path / "meta.parquet")
        self._write_minimal_parquet(parquet)
        cfg = self._make_valid_cfg(parquet)
        # Remove the data section entirely
        cfg_dict = OmegaConf.to_container(cfg, resolve=False)
        del cfg_dict["data"]
        cfg_no_data = OmegaConf.create(cfg_dict)
        with pytest.raises(ValueError, match="Missing 'data'"):
            self._validate(cfg_no_data)

    def test_missing_roaming_section_fails(self, tmp_path):
        parquet = str(tmp_path / "meta.parquet")
        self._write_minimal_parquet(parquet)
        cfg = self._make_valid_cfg(parquet)
        cfg_dict = OmegaConf.to_container(cfg, resolve=False)
        del cfg_dict["roaming"]
        cfg_no_roaming = OmegaConf.create(cfg_dict)
        with pytest.raises(ValueError, match="Missing 'roaming'"):
            self._validate(cfg_no_roaming)

    def test_independent_termination_mode_passes(self, tmp_path):
        parquet = str(tmp_path / "meta.parquet")
        self._write_minimal_parquet(parquet)
        cfg = self._make_valid_cfg(parquet)
        OmegaConf.update(cfg, "roaming.termination_mode", "independent", merge=True)
        # Must not raise
        self._validate(cfg)


# ---------------------------------------------------------------------------
# 6. Prompt rendering
# ---------------------------------------------------------------------------

class TestRenderPrompt:
    """Tests for RoamingStepper._render_prompt."""

    def setup_method(self):
        self.graph = _make_simple_graph()

    def test_system_prompt_not_in_user_prompt(self):
        """The system prompt must NOT bleed into the rendered user prompt."""
        walks = _make_walks()
        cfg = _make_stepper_cfg(system_prompt="You are a tourist guide.")
        stepper = RoamingStepper(self.graph, walks, cfg)
        walk = stepper.walks["w0"]
        prompt = stepper._render_prompt(walk, ["F", "R", "L"])
        assert "You are a tourist guide" not in prompt

    def test_available_faces_present_in_prompt(self):
        """The rendered prompt should reference the available direction faces."""
        walks = _make_walks()
        cfg = _make_stepper_cfg()
        stepper = RoamingStepper(self.graph, walks, cfg)
        walk = stepper.walks["w0"]
        prompt = stepper._render_prompt(walk, ["F", "R", "L"])
        assert "F" in prompt
        assert "R" in prompt
        assert "L" in prompt

    def test_custom_user_template_rendered(self):
        """A custom user_template variable is expanded correctly."""
        walks = _make_walks()
        template = "Step {{ step_n }}: go to {{ available_faces }}"
        cfg = _make_stepper_cfg(user_template=template)
        stepper = RoamingStepper(self.graph, walks, cfg)
        walk = stepper.walks["w0"]
        prompt = stepper._render_prompt(walk, ["F", "R"])
        assert "Step 0" in prompt
        assert "F, R" in prompt

    def test_history_not_included_when_disabled(self):
        """With include_history_in_prompt=False, no history block in prompt."""
        walks = _make_walks()
        cfg = _make_stepper_cfg(include_history=False)
        stepper = RoamingStepper(self.graph, walks, cfg)
        walk = stepper.walks["w0"]
        # Add a fake history step
        walk.history = [
            WalkStep(step_n=0, recording_id="N0", arrival_face="F", faces_shown=["F", "R", "L"],
                     face_chosen="F", reasoning="test", lat=40.0, lon=-74.0)
        ]
        prompt = stepper._render_prompt(walk, ["R", "B", "L"])
        assert "Recent history" not in prompt

    def test_default_prompt_template_used_when_none_set(self):
        """When user_template is empty, the built-in default template is used."""
        walks = _make_walks()
        cfg = _make_stepper_cfg(user_template="")
        stepper = RoamingStepper(self.graph, walks, cfg)
        walk = stepper.walks["w0"]
        prompt = stepper._render_prompt(walk, ["F", "R", "L"])
        # The default template has this phrase
        assert "Available directions" in prompt


# ---------------------------------------------------------------------------
# 7. Smoke test — full walk execution without inference
# ---------------------------------------------------------------------------

def _build_linear_graph(n: int, yaw: float = 0.0) -> StreetGraph:
    """Build a simple north-south linear chain: N0 -> N1 -> ... -> N{n-1}.

    Each node is at (40.0 + i*0.001, -74.0).  Neighbors face north (bearing 0)
    and south (bearing 180) between consecutive nodes.  All yaw_degrees = yaw.
    """
    ids = [f"N{i}" for i in range(n)]
    coords = {nid: (40.0 + i * 0.001, -74.0) for i, nid in enumerate(ids)}
    yaw_degrees = {nid: yaw for nid in ids}

    adjacency: Dict[str, List[Neighbor]] = {}
    for i, nid in enumerate(ids):
        nbs: List[Neighbor] = []
        if i > 0:
            nbs.append(Neighbor(recording_id=ids[i - 1], distance_m=111.0, bearing_deg=180.0))
        if i < n - 1:
            nbs.append(Neighbor(recording_id=ids[i + 1], distance_m=111.0, bearing_deg=0.0))
        adjacency[nid] = nbs

    return StreetGraph(adjacency=adjacency, coords=coords, yaw_degrees=yaw_degrees)


def _create_dummy_images(tmp_path: Path, graph: StreetGraph, faces=("F", "R", "B", "L")) -> str:
    """Write small 64x64 PNG images for every (node, face) pair. Returns image_root path."""
    image_root = tmp_path / "images"
    image_root.mkdir()
    for node_id in graph.coords:
        for face in faces:
            img = Image.new("RGB", (64, 64), color=(128, 64, 32))
            img.save(image_root / f"{node_id}_{face}.jpg", "JPEG")
    return str(image_root)


def test_smoke_skip_inference(tmp_path):
    """Full walk execution using skip_inference=True (deterministic debug mode)."""
    from dagspaces.urbanroamvqa.stages.roaming_vqa import run_roaming_vqa_stage

    n_nodes = 5
    n_walks = 2
    max_steps = 3

    graph = _build_linear_graph(n_nodes)
    image_root = _create_dummy_images(tmp_path, graph)

    # Build seeds DataFrame manually (mirrors what sample_walk_seeds produces)
    seeds = pd.DataFrame([
        {"walk_id": "walk_000000", "seed_recording_id": "N0", "seed_face": ""},
        {"walk_id": "walk_000001", "seed_recording_id": "N2", "seed_face": ""},
    ])

    checkpoint_dir = str(tmp_path / "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    cfg = OmegaConf.create({
        "roaming": {
            "max_steps": max_steps,
            "termination_mode": "fixed",
            "allow_revisits": True,
            "include_history_in_prompt": False,
            "history_max_steps": 3,
            "stitch_max_height": 64,
            "checkpoint_dir": checkpoint_dir,
        },
        "graph": {
            "bearing_tolerance_deg": 45.0,
        },
        "prompt": {
            "system": "You are a navigator.",
            "user_template": "",
        },
        "data": {
            "image_root": image_root,
            "image_pattern": "{recording_id}_{face}.jpg",
        },
        "runtime": {
            "skip_inference": True,
        },
        "model": {
            "model_source": "test/model",
            "batch_size": 1,
            "concurrency": 1,
        },
        "sampling_params_vqa": {
            "temperature": 0.0,
            "max_tokens": 100,
        },
    })

    traces = run_roaming_vqa_stage(seeds, cfg, graph=graph)

    # --- Output shape assertions ---
    assert isinstance(traces, pd.DataFrame), "run_roaming_vqa_stage must return a DataFrame"

    expected_cols = {
        "walk_id", "step_n", "recording_id", "arrival_face",
        "faces_shown", "face_chosen", "reasoning", "lat", "lon",
        "next_recording_id", "distance_m", "termination_reason", "answer_raw",
    }
    assert expected_cols.issubset(set(traces.columns)), (
        f"Missing columns: {expected_cols - set(traces.columns)}"
    )

    # Both walks should appear in the traces
    assert "walk_000000" in traces["walk_id"].values
    assert "walk_000001" in traces["walk_id"].values

    # Each walk should have at least 1 step and no more than max_steps
    for wid in ["walk_000000", "walk_000001"]:
        walk_steps = traces[traces["walk_id"] == wid]
        assert len(walk_steps) >= 1, f"{wid} should have at least 1 step"
        assert len(walk_steps) <= max_steps, f"{wid} exceeded max_steps={max_steps}"

    # step_n should be monotonically increasing within each walk
    for wid in traces["walk_id"].unique():
        steps = traces[traces["walk_id"] == wid]["step_n"].tolist()
        assert steps == sorted(steps), f"step_n not monotonic for {wid}: {steps}"

    # --- Checkpoint lifecycle ---
    checkpoint_file = os.path.join(checkpoint_dir, "roaming_checkpoint.json")
    finished_file = checkpoint_file.replace(".json", ".finished.json")
    # After successful completion, .json should be renamed to .finished.json
    assert os.path.exists(finished_file), "Finished checkpoint file must exist after completion"
    assert not os.path.exists(checkpoint_file), "Active checkpoint must be renamed on completion"


def test_smoke_empty_seeds(tmp_path):
    """Running with zero seeds produces an empty trace DataFrame with correct columns."""
    from dagspaces.urbanroamvqa.stages.roaming_vqa import run_roaming_vqa_stage

    graph = _build_linear_graph(3)
    image_root = _create_dummy_images(tmp_path, graph)

    seeds = pd.DataFrame(columns=["walk_id", "seed_recording_id", "seed_face"])

    cfg = OmegaConf.create({
        "roaming": {
            "max_steps": 3,
            "termination_mode": "fixed",
            "allow_revisits": True,
            "include_history_in_prompt": False,
            "history_max_steps": 3,
            "stitch_max_height": 64,
        },
        "graph": {"bearing_tolerance_deg": 45.0},
        "prompt": {"system": "", "user_template": ""},
        "data": {"image_root": image_root, "image_pattern": "{recording_id}_{face}.jpg"},
        "runtime": {"skip_inference": True},
        "model": {"model_source": "test/model", "batch_size": 1, "concurrency": 1},
        "sampling_params_vqa": {"temperature": 0.0, "max_tokens": 100},
    })

    traces = run_roaming_vqa_stage(seeds, cfg, graph=graph)
    assert isinstance(traces, pd.DataFrame)
    assert traces.empty or len(traces) == 0


# ---------------------------------------------------------------------------
# 8. Builder column normalization (optional, requires real parquet)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists("data/cyclomedia/manhattan_2025_1_1k_scratch.parquet"),
    reason="Cyclomedia data not available",
)
def test_column_normalization_real_data():
    """Column aliases (lat, lon, yawDegrees) are normalized in build_street_graph."""
    from dagspaces.urbanroamvqa.graph.builder import build_street_graph

    parquet = "data/cyclomedia/manhattan_2025_1_1k_scratch.parquet"
    df = pd.read_parquet(parquet)
    columns = set(df.columns)

    # Verify that the parquet uses alias columns so the test is meaningful.
    # The Cyclomedia parquet has both "latitude" (all None) and "lat" (actual data),
    # so we check for the presence of aliases rather than absence of canonical names.
    uses_aliases = "lat" in columns or "lon" in columns or "yawDegrees" in columns
    if not uses_aliases:
        pytest.skip("Parquet already uses canonical column names; alias test not applicable")

    graph_cfg = OmegaConf.create({"k_neighbors": 3, "max_radius_m": 100.0})
    graph = build_street_graph(parquet, graph_cfg, use_osmnx=False)

    assert len(graph.coords) > 0, "Graph must have at least one node"
    assert len(graph.adjacency) > 0, "Graph must have adjacency entries"

    # Spot-check that coordinates and yaw values are numeric
    sample_id = next(iter(graph.coords))
    lat, lon = graph.coords[sample_id]
    assert math.isfinite(lat) and math.isfinite(lon)
    yaw = graph.yaw_degrees.get(sample_id, 0.0)
    assert math.isfinite(yaw)


# ---------------------------------------------------------------------------
# 9. Stepper advance_from_answers integration
# ---------------------------------------------------------------------------

class TestStepperAdvanceFromAnswers:
    """Integration tests for the advance/step loop (no inference required)."""

    def setup_method(self):
        self.graph = _make_simple_graph()

    def test_advance_moves_walk_to_next_node(self):
        """When the VLM picks a valid face, the walk advances to the next recording."""
        walks = [WalkState(walk_id="w0", current_recording_id="N0", current_arrival_face="F")]
        cfg = _make_stepper_cfg(max_steps=5)
        stepper = RoamingStepper(self.graph, walks, cfg)
        walk = stepper.walks["w0"]

        # Manually add a history step as prepare_step_batch would
        step = WalkStep(step_n=0, recording_id="N0", arrival_face="F", faces_shown=["R", "B", "L"])
        walk.history.append(step)
        walk.visited.add("N0")

        # Simulate answers: pick the Right face, which maps to N2 (bearing 90°)
        answers = pd.DataFrame([{
            "walk_id": "w0",
            "sample_id": "w0_step0",
            "answer": json.dumps({"chosen_face": "R", "reasoning": "east street"}),
        }])
        stepper.advance_from_answers(answers)

        assert walk.current_recording_id == "N2"
        assert walk.step_n == 1

    def test_advance_deactivates_walk_on_dead_end(self):
        """If the chosen face resolves to no neighbor, the walk is marked inactive."""
        walks = [WalkState(walk_id="w0", current_recording_id="N0", current_arrival_face="F")]
        cfg = _make_stepper_cfg(max_steps=5, bearing_tolerance=1.0)  # very tight tolerance
        stepper = RoamingStepper(self.graph, walks, cfg)
        walk = stepper.walks["w0"]

        step = WalkStep(step_n=0, recording_id="N0", arrival_face="F", faces_shown=["B", "R", "L"])
        walk.history.append(step)
        walk.visited.add("N0")

        # Behind face has no neighbor at 180° within 1° tolerance for N0
        answers = pd.DataFrame([{
            "walk_id": "w0",
            "sample_id": "w0_step0",
            "answer": json.dumps({"chosen_face": "B", "reasoning": "going back"}),
        }])
        stepper.advance_from_answers(answers)

        assert walk.active is False
        assert walk.history[-1].termination_reason == "dead_end"

    def test_advance_stop_signal_in_independent_mode(self):
        """In independent mode, stop=true in the answer terminates the walk."""
        walks = [WalkState(walk_id="w0", current_recording_id="N0", current_arrival_face="F")]
        cfg = _make_stepper_cfg(max_steps=10, termination_mode="independent")
        stepper = RoamingStepper(self.graph, walks, cfg)
        walk = stepper.walks["w0"]

        step = WalkStep(step_n=0, recording_id="N0", arrival_face="F", faces_shown=["F", "R", "L"])
        walk.history.append(step)
        walk.visited.add("N0")

        answers = pd.DataFrame([{
            "walk_id": "w0",
            "sample_id": "w0_step0",
            "answer": json.dumps({"chosen_face": "F", "reasoning": "arrived", "stop": True}),
        }])
        stepper.advance_from_answers(answers)

        assert walk.active is False
        assert walk.history[-1].termination_reason == "stop"

    def test_advance_max_steps_deactivates_walk(self):
        """Walk is deactivated when step_n reaches max_steps."""
        walks = [WalkState(walk_id="w0", current_recording_id="N0", current_arrival_face="F",
                           step_n=4)]  # one step away from max
        cfg = _make_stepper_cfg(max_steps=5)
        stepper = RoamingStepper(self.graph, walks, cfg)
        walk = stepper.walks["w0"]

        step = WalkStep(step_n=4, recording_id="N0", arrival_face="F", faces_shown=["F", "R", "L"])
        walk.history.append(step)
        walk.visited.add("N0")

        answers = pd.DataFrame([{
            "walk_id": "w0",
            "sample_id": "w0_step4",
            "answer": json.dumps({"chosen_face": "F", "reasoning": "forward"}),
        }])
        stepper.advance_from_answers(answers)

        assert walk.active is False
        assert walk.history[-1].termination_reason == "max_steps"


# ---------------------------------------------------------------------------
# 10. all_traces output structure
# ---------------------------------------------------------------------------

class TestAllTraces:
    """Tests for RoamingStepper.all_traces."""

    def test_empty_walks_returns_empty_df_with_correct_columns(self):
        graph = _make_simple_graph()
        stepper = RoamingStepper(graph, [], _make_stepper_cfg())
        df = stepper.all_traces()
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        expected_cols = {
            "walk_id", "step_n", "recording_id", "arrival_face", "faces_shown",
            "face_chosen", "reasoning", "lat", "lon", "bearing_deg",
            "next_recording_id", "distance_m", "termination_reason", "answer_raw",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_traces_has_one_row_per_step(self):
        graph = _make_simple_graph()
        ws = WalkState(walk_id="w0", current_recording_id="N0", current_arrival_face="F")
        ws.history = [
            WalkStep(step_n=0, recording_id="N0", arrival_face="F", faces_shown=["F", "R", "L"],
                     face_chosen="F"),
            WalkStep(step_n=1, recording_id="N1", arrival_face="F", faces_shown=["R", "B", "L"],
                     face_chosen="R"),
        ]
        stepper = RoamingStepper(graph, [ws], _make_stepper_cfg())
        df = stepper.all_traces()
        assert len(df) == 2
        assert list(df["step_n"]) == [0, 1]

    def test_faces_shown_serialized_as_comma_joined_string(self):
        graph = _make_simple_graph()
        ws = WalkState(walk_id="w0", current_recording_id="N0", current_arrival_face="F")
        ws.history = [
            WalkStep(step_n=0, recording_id="N0", arrival_face="F", faces_shown=["F", "R", "L"]),
        ]
        stepper = RoamingStepper(graph, [ws], _make_stepper_cfg())
        df = stepper.all_traces()
        assert df.iloc[0]["faces_shown"] == "F,R,L"


# ---------------------------------------------------------------------------
# 11. Board-first builder (pure functions, no OSM network access)
# ---------------------------------------------------------------------------

def _make_osm_like_grid(n: int = 3, spacing_deg: float = 0.001):
    """Build an n x n grid MultiGraph mimicking osmnx output: nodes carry
    x/y (lon/lat) attrs, edges carry length in meters."""
    import networkx as nx
    from dagspaces.urbanroamvqa.graph.board_builder import _haversine_m

    G = nx.MultiGraph()
    for r in range(n):
        for c in range(n):
            G.add_node(r * n + c, x=-74.0 + c * spacing_deg, y=40.0 + r * spacing_deg)

    def _add(u, v):
        d = _haversine_m(G.nodes[u]["y"], G.nodes[u]["x"], G.nodes[v]["y"], G.nodes[v]["x"])
        G.add_edge(u, v, length=d)

    for r in range(n):
        for c in range(n):
            node = r * n + c
            if c + 1 < n:
                _add(node, node + 1)
            if r + 1 < n:
                _add(node, node + n)
    return G


class TestBoardDiscretize:
    """Tests for board_builder._discretize_edges."""

    def test_board_is_connected_and_uniform(self):
        import networkx as nx
        from dagspaces.urbanroamvqa.graph.board_builder import _discretize_edges

        G = _make_osm_like_grid(3)  # edges ~85-111m
        board = _discretize_edges(G, spacing_m=30.0)

        assert nx.is_connected(board)
        # Junctions plus interpolated interior nodes
        assert board.number_of_nodes() > G.number_of_nodes()
        # Pitch stays near the target spacing
        dists = [d["dist"] for _, _, d in board.edges(data=True)]
        assert all(10.0 <= d <= 60.0 for d in dists)

    def test_all_nodes_have_coords(self):
        from dagspaces.urbanroamvqa.graph.board_builder import _discretize_edges

        board = _discretize_edges(_make_osm_like_grid(2), spacing_m=30.0)
        for _, data in board.nodes(data=True):
            assert math.isfinite(data["lat"]) and math.isfinite(data["lon"])

    def test_flipped_geometry_is_reanchored(self):
        """to_undirected leaves geometry direction arbitrary: a v->u geometry
        must not create junction hops spanning the whole edge (regression for
        the 2026-06-09 Manhattan build: 53% flipped edges -> 2.4km 'steps')."""
        import networkx as nx
        from shapely.geometry import LineString
        from dagspaces.urbanroamvqa.graph.board_builder import (
            _discretize_edges,
            _haversine_m,
        )

        G = nx.MultiGraph()
        G.add_node(0, x=-74.0, y=40.0)
        G.add_node(1, x=-74.0, y=40.002)  # ~222m due north
        length = _haversine_m(40.0, -74.0, 40.002, -74.0)
        # Geometry deliberately runs from node 1 down to node 0 (flipped)
        flipped_geom = LineString([(-74.0, 40.002), (-74.0, 40.0)])
        G.add_edge(0, 1, length=length, geometry=flipped_geom)

        board = _discretize_edges(G, spacing_m=30.0)
        dists = [d["dist"] for _, _, d in board.edges(data=True)]
        assert max(dists) < 45.0, f"flipped geometry produced a long hop: {max(dists):.0f}m"
        assert nx.is_connected(board)


class TestBoardAttach:
    """Tests for board_builder._attach_recordings."""

    def _board(self):
        from dagspaces.urbanroamvqa.graph.board_builder import _discretize_edges

        return _discretize_edges(_make_osm_like_grid(3), spacing_m=30.0)

    def test_unique_assignment_at_exact_positions(self):
        from dagspaces.urbanroamvqa.graph.board_builder import _attach_recordings

        board = self._board()
        nodes = list(board.nodes)
        lats = np.array([board.nodes[n]["lat"] for n in nodes])
        lons = np.array([board.nodes[n]["lon"] for n in nodes])
        yaws = np.zeros(len(nodes))

        assigned = _attach_recordings(board, lats, lons, yaws, max_snap_dist_m=10.0)
        # One recording placed exactly at each node: full unique coverage
        assert len(assigned) == len(nodes)
        assert len(set(assigned.values())) == len(assigned)

    def test_far_recordings_not_assigned(self):
        from dagspaces.urbanroamvqa.graph.board_builder import _attach_recordings

        board = self._board()
        lats = np.array([41.0])  # ~100km away
        lons = np.array([-74.0])
        assigned = _attach_recordings(board, lats, lons, np.zeros(1), max_snap_dist_m=30.0)
        assert assigned == {}

    def test_each_recording_used_at_most_once(self):
        from dagspaces.urbanroamvqa.graph.board_builder import _attach_recordings

        board = self._board()
        nodes = list(board.nodes)
        # Single recording near two adjacent nodes: only one node gets it
        lat = board.nodes[nodes[0]]["lat"]
        lon = board.nodes[nodes[0]]["lon"]
        assigned = _attach_recordings(
            board, np.array([lat]), np.array([lon]), np.zeros(1), max_snap_dist_m=100.0,
        )
        assert len(assigned) == 1


class TestBoardContract:
    """Tests for board_builder._contract_imageless."""

    def _chain_board(self):
        """a - x - b chain with 10m edges; x will be imageless."""
        import networkx as nx

        board = nx.Graph()
        board.add_node("a", lat=40.0, lon=-74.0)
        board.add_node("x", lat=40.0001, lon=-74.0)
        board.add_node("b", lat=40.0002, lon=-74.0)
        board.add_edge("a", "x", dist=10.0)
        board.add_edge("x", "b", dist=10.0)
        return board

    def test_contraction_reconnects_with_summed_distance(self):
        from dagspaces.urbanroamvqa.graph.board_builder import _contract_imageless

        board = self._chain_board()
        n = _contract_imageless(board, assigned={"a": 0, "b": 1}, max_contracted_edge_m=100.0)
        assert n == 1
        assert "x" not in board
        assert board.has_edge("a", "b")
        assert board["a"]["b"]["dist"] == pytest.approx(20.0)

    def test_contraction_respects_max_edge_length(self):
        from dagspaces.urbanroamvqa.graph.board_builder import _contract_imageless

        board = self._chain_board()
        _contract_imageless(board, assigned={"a": 0, "b": 1}, max_contracted_edge_m=15.0)
        # 20m reconnect exceeds the 15m cap: no teleport edge is created
        assert not board.has_edge("a", "b")

    def test_imageless_chain_collapses(self):
        import networkx as nx
        from dagspaces.urbanroamvqa.graph.board_builder import _contract_imageless

        board = nx.Graph()
        chain = ["a", "x1", "x2", "x3", "b"]
        for i, node in enumerate(chain):
            board.add_node(node, lat=40.0 + i * 0.0001, lon=-74.0)
        for u, v in zip(chain[:-1], chain[1:]):
            board.add_edge(u, v, dist=10.0)

        n = _contract_imageless(board, assigned={"a": 0, "b": 1}, max_contracted_edge_m=100.0)
        assert n == 3
        assert board.has_edge("a", "b")
        assert board["a"]["b"]["dist"] == pytest.approx(40.0)
        assert nx.is_connected(board)


class TestPerRowGuidedDecoding:
    """Per-step guided decoding: the JSON schema's chosen_face enum is
    narrowed to each row's legal faces, so illegal moves are unrepresentable."""

    def _prepare(self, tmp_path, structured_schema):
        graph = _make_simple_graph()
        image_root = _create_dummy_images(tmp_path, graph)
        cfg = _make_stepper_cfg(structured_schema=structured_schema)
        stepper = RoamingStepper(graph, _make_walks("N0", arrival_face=""), cfg)
        data_cfg = OmegaConf.create({
            "image_root": image_root,
            "image_pattern": "{recording_id}_{face}.jpg",
        })
        return stepper, stepper.prepare_step_batch(data_cfg)

    def test_enum_narrowed_to_faces_shown(self, tmp_path):
        stepper, batch = self._prepare(tmp_path, _ROAM_SCHEMA)
        assert len(batch) == 1
        row = batch.iloc[0]
        # N0 (yaw=0) has neighbors due north (F) and due east (R)
        faces_shown = stepper.walks["w0"].history[-1].faces_shown
        assert faces_shown == ["F", "R"]
        guided = row["guided_decoding"]
        assert isinstance(guided, dict)
        schema = guided["json"]
        assert schema["properties"]["chosen_face"]["enum"] == faces_shown

    def test_rest_of_schema_preserved(self, tmp_path):
        _, batch = self._prepare(tmp_path, _ROAM_SCHEMA)
        schema = batch.iloc[0]["guided_decoding"]["json"]
        assert schema["properties"]["reasoning"] == {"type": "string"}
        assert schema["properties"]["stop"] == {"type": "boolean"}
        assert schema["required"] == ["chosen_face", "reasoning"]

    def test_base_schema_not_mutated_across_rows(self, tmp_path):
        stepper, _ = self._prepare(tmp_path, _ROAM_SCHEMA)
        assert stepper.structured_base["properties"]["chosen_face"]["enum"] == ["F", "R", "B", "L"]
        # A second payload for a different menu is independent
        p1 = stepper._guided_payload_for_faces(["B"])
        p2 = stepper._guided_payload_for_faces(["F", "L"])
        assert p1["json"]["properties"]["chosen_face"]["enum"] == ["B"]
        assert p2["json"]["properties"]["chosen_face"]["enum"] == ["F", "L"]

    def test_no_schema_means_no_column(self, tmp_path):
        _, batch = self._prepare(tmp_path, structured_schema=None)
        assert "guided_decoding" not in batch.columns


class TestVqaPerRowGuidedOverride:
    """run_vqa_stage preprocess honors a per-row guided_decoding payload."""

    def _make_cfg(self, structured_output=None):
        sp = {"temperature": 0.1, "max_tokens": 8, "stop": []}
        if structured_output is not None:
            sp["structured_output"] = structured_output
        return OmegaConf.create({
            "model": {"model_source": "test/tiny-text", "multimodal": False},
            "prompt": {"system": "sys", "user_template": "{{prompt}}"},
            "sampling_params_vqa": sp,
        })

    def test_row_payload_overrides_cfg_schema(self):
        from dagspaces.urbanvqa.stages.vqa import _make_preprocess

        cfg = self._make_cfg(structured_output=_ROAM_SCHEMA)
        preprocess = _make_preprocess(cfg)
        row_guided = {"json": {"type": "object", "properties": {
            "chosen_face": {"type": "string", "enum": ["F", "R"]}}}}
        result = preprocess({"sample_id": "s1", "prompt": "q",
                             "guided_decoding": row_guided})
        assert result["sampling_params"]["guided_decoding"] == row_guided

    def test_falls_back_to_cfg_schema_without_row_payload(self):
        from dagspaces.urbanvqa.stages.vqa import _make_preprocess

        cfg = self._make_cfg(structured_output=_ROAM_SCHEMA)
        preprocess = _make_preprocess(cfg)
        result = preprocess({"sample_id": "s1", "prompt": "q"})
        # cfg-level path goes through _build_guided_decoding_config, which
        # collapses enum-bearing schemas to a choice constraint
        assert result["sampling_params"]["guided_decoding"] == {"choice": ["F", "R", "B", "L"]}

    def test_no_guided_decoding_when_neither_present(self):
        from dagspaces.urbanvqa.stages.vqa import _make_preprocess

        preprocess = _make_preprocess(self._make_cfg())
        result = preprocess({"sample_id": "s1", "prompt": "q"})
        assert "guided_decoding" not in result["sampling_params"]


class TestBuildBoardGraphEndToEnd:
    """Full build_board_graph run against a synthetic OSM backbone."""

    def _build(self, tmp_path, monkeypatch, precomputed_path=None):
        import dagspaces.urbanroamvqa.graph.board_builder as bb

        G = _make_osm_like_grid(4)
        monkeypatch.setattr(bb, "_load_osm_backbone", lambda cfg: G)

        # One recording exactly at each junction, heading north
        rows = [
            {
                "recording_id": f"R{i}",
                "latitude": data["y"],
                "longitude": data["x"],
                "recorderDirection": 0.0,
            }
            for i, (_n, data) in enumerate(G.nodes(data=True))
        ]
        parquet = str(tmp_path / "meta.parquet")
        pd.DataFrame(rows).to_parquet(parquet, index=False)

        cfg = OmegaConf.create({
            "graph_type": "board",
            "spacing_m": 40.0,
            "max_snap_dist_m": 60.0,
            "heading_penalty_m": 10.0,
            "max_contracted_edge_m": 300.0,
            "min_main_component_frac": 0.5,
            "yaw_column": "recorderDirection",
            "precomputed_path": precomputed_path,
        })
        return bb.build_board_graph(parquet, cfg)

    def test_single_connected_component_keyed_by_recording_id(self, tmp_path, monkeypatch):
        from dagspaces.urbanroamvqa.graph.street_graph import compute_graph_diagnostics

        graph = self._build(tmp_path, monkeypatch)
        # All 16 junction recordings survive; imageless interior nodes contracted
        assert len(graph.adjacency) == 16
        assert all(rid.startswith("R") for rid in graph.adjacency)
        diag = compute_graph_diagnostics(graph)
        assert diag["graph/n_components"] == 1
        assert diag["graph/n_isolated"] == 0
        # Grid corners have degree 2, edges 3, center 4
        assert diag["graph/degree_max"] == 4

    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        cache_path = str(tmp_path / "board.pkl")
        graph1 = self._build(tmp_path, monkeypatch, precomputed_path=cache_path)
        assert os.path.exists(cache_path)
        graph2 = self._build(tmp_path, monkeypatch, precomputed_path=cache_path)
        assert set(graph2.adjacency.keys()) == set(graph1.adjacency.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
