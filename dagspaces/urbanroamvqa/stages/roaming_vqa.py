"""Roaming VQA stage — agent-driven urban navigation via VLM face selection."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from ..graph.street_graph import FACE_BEARING_DEG, HORIZONTAL_FACES, StreetGraph, _normalize_bearing
from ..graph.builder import build_street_graph
from ..samplers.seed_sampler import sample_walk_seeds
from dagspaces.urbanvqa.stages.vqa import run_vqa_stage


# --- Image stitching (adapted from urbanpairvqa) ---

_COMPASS_LABELS = {"F": "Forward", "R": "Right", "B": "Behind", "L": "Left"}


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _stitch_three(
    paths: List[str],
    labels: List[str],
    max_height: int = 512,
) -> np.ndarray:
    """Stitch 3 face images horizontally with direction labels.

    Args:
        paths: List of 3 image file paths.
        labels: List of 3 compass direction labels (e.g. ["Left", "Forward", "Right"]).
        max_height: Target height for all images.

    Returns:
        numpy array of the stitched composite image.
    """
    images = []
    for path in paths:
        img = _load_rgb(path)
        scale = max_height / float(max(1, img.height))
        new_size = (max(1, int(img.width * scale)), max_height)
        img = img.resize(new_size)
        images.append(img)

    total_width = sum(img.width for img in images)
    combined = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    x_offset = 0
    for img, label in zip(images, labels):
        combined.paste(img, (x_offset, 0))
        # Draw label text as simple overlay
        try:
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except (IOError, OSError):
                font = ImageFont.load_default()
            # White text with black outline for visibility
            tx = x_offset + 10
            ty = 10
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        draw.text((tx + dx, ty + dy), label, fill=(0, 0, 0), font=font)
            draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
        except ImportError:
            pass  # PIL without ImageDraw; labels will be in the prompt instead
        x_offset += img.width

    return np.asarray(combined)


# --- Walk state ---

@dataclass
class WalkStep:
    step_n: int
    recording_id: str
    arrival_face: str
    faces_shown: List[str]
    face_chosen: Optional[str] = None
    reasoning: Optional[str] = None
    lat: float = 0.0
    lon: float = 0.0
    bearing_deg: Optional[float] = None
    next_recording_id: Optional[str] = None
    distance_m: Optional[float] = None
    termination_reason: Optional[str] = None
    answer_raw: Optional[str] = None


@dataclass
class WalkState:
    walk_id: str
    current_recording_id: str
    current_arrival_face: str
    step_n: int = 0
    active: bool = True
    history: List[WalkStep] = field(default_factory=list)
    visited: set = field(default_factory=set)


# --- Checkpoint support ---

def _walk_step_to_dict(step: WalkStep) -> Dict[str, Any]:
    return {
        "step_n": step.step_n,
        "recording_id": step.recording_id,
        "arrival_face": step.arrival_face,
        "faces_shown": step.faces_shown,
        "face_chosen": step.face_chosen,
        "reasoning": step.reasoning,
        "lat": step.lat,
        "lon": step.lon,
        "bearing_deg": step.bearing_deg,
        "next_recording_id": step.next_recording_id,
        "distance_m": step.distance_m,
        "termination_reason": step.termination_reason,
        "answer_raw": step.answer_raw,
    }


def _walk_step_from_dict(d: Dict[str, Any]) -> WalkStep:
    return WalkStep(
        step_n=d["step_n"],
        recording_id=d["recording_id"],
        arrival_face=d["arrival_face"],
        faces_shown=d.get("faces_shown", []),
        face_chosen=d.get("face_chosen"),
        reasoning=d.get("reasoning"),
        lat=d.get("lat", 0.0),
        lon=d.get("lon", 0.0),
        bearing_deg=d.get("bearing_deg"),
        next_recording_id=d.get("next_recording_id"),
        distance_m=d.get("distance_m"),
        termination_reason=d.get("termination_reason"),
        answer_raw=d.get("answer_raw"),
    )


def _walk_state_to_dict(ws: WalkState) -> Dict[str, Any]:
    return {
        "walk_id": ws.walk_id,
        "current_recording_id": ws.current_recording_id,
        "current_arrival_face": ws.current_arrival_face,
        "step_n": ws.step_n,
        "active": ws.active,
        "history": [_walk_step_to_dict(s) for s in ws.history],
        "visited": sorted(ws.visited),
    }


def _walk_state_from_dict(d: Dict[str, Any]) -> WalkState:
    return WalkState(
        walk_id=d["walk_id"],
        current_recording_id=d["current_recording_id"],
        current_arrival_face=d["current_arrival_face"],
        step_n=d.get("step_n", 0),
        active=d.get("active", True),
        history=[_walk_step_from_dict(s) for s in d.get("history", [])],
        visited=set(d.get("visited", [])),
    )


def save_checkpoint(
    checkpoint_path: str,
    walks: Dict[str, WalkState],
    step_iter: int,
    max_steps: int,
) -> None:
    """Write checkpoint to disk (atomic via temp file + rename)."""
    data = {
        "version": 1,
        "step_iter": step_iter,
        "max_steps": max_steps,
        "timestamp": time.time(),
        "walks": {wid: _walk_state_to_dict(ws) for wid, ws in walks.items()},
    }
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint from disk. Returns None if not found or corrupt."""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != 1:
            print(f"[checkpoint] Unknown version {data.get('version')}, ignoring", flush=True)
            return None
        return data
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"[checkpoint] Corrupt checkpoint, ignoring: {exc}", flush=True)
        return None


# --- Roaming stepper ---

class RoamingStepper:
    def __init__(
        self,
        graph: StreetGraph,
        walks: List[WalkState],
        cfg: DictConfig,
    ):
        self.graph = graph
        self.walks = {w.walk_id: w for w in walks}
        self.cfg = cfg
        self._image_lookup: Dict[tuple, str] = {}

        roaming_cfg = getattr(cfg, "roaming", {})
        self.max_steps = int(getattr(roaming_cfg, "max_steps", 10))
        self.termination_mode = str(getattr(roaming_cfg, "termination_mode", "fixed"))
        self.allow_revisits = bool(getattr(roaming_cfg, "allow_revisits", True))
        self.include_history = bool(getattr(roaming_cfg, "include_history_in_prompt", True))
        self.history_max_steps = int(getattr(roaming_cfg, "history_max_steps", 5))
        self.stitch_max_height = int(getattr(roaming_cfg, "stitch_max_height", 512))

        graph_cfg = getattr(cfg, "graph", {})
        self.bearing_tolerance = float(getattr(graph_cfg, "bearing_tolerance_deg", 45.0))

    @property
    def active_walks(self) -> List[WalkState]:
        return [w for w in self.walks.values() if w.active]

    def prepare_step_batch(self, data_cfg: Any) -> pd.DataFrame:
        """Build a DataFrame of VQA inference rows for all active walks.

        Each row has: walk_id, image (stitched 3-face composite), prompt, sample_id.
        """
        rows = []
        metadata_parquet = str(getattr(data_cfg, "parquet_path", ""))

        for walk in self.active_walks:
            rid = walk.current_recording_id
            arrival_face = walk.current_arrival_face
            available_faces = self.graph.available_faces(arrival_face)

            if not available_faces:
                walk.active = False
                if walk.history:
                    walk.history[-1].termination_reason = "dead_end"
                continue

            # Build face image paths
            face_paths = []
            face_labels = []
            valid = True
            for face in available_faces:
                # Construct image path from recording_id + face
                img_path = self._resolve_face_image_path(rid, face, data_cfg)
                if not img_path or not os.path.exists(img_path):
                    valid = False
                    break
                face_paths.append(img_path)
                face_labels.append(_COMPASS_LABELS.get(face, face))

            if not valid:
                walk.active = False
                if walk.history:
                    walk.history[-1].termination_reason = "missing_images"
                continue

            # Stitch images
            stitched = _stitch_three(face_paths, face_labels, self.stitch_max_height)

            # Build prompt
            prompt = self._render_prompt(walk, available_faces)

            # Record step info
            coords = self.graph.coords.get(rid, (0.0, 0.0))
            step = WalkStep(
                step_n=walk.step_n,
                recording_id=rid,
                arrival_face=arrival_face,
                faces_shown=available_faces,
                lat=coords[0],
                lon=coords[1],
            )
            walk.history.append(step)
            walk.visited.add(rid)

            rows.append({
                "walk_id": walk.walk_id,
                "sample_id": f"{walk.walk_id}_step{walk.step_n}",
                "image": stitched,
                "prompt": prompt,
            })

        return pd.DataFrame(rows)

    def advance_from_answers(self, answers_df: pd.DataFrame) -> None:
        """Parse VLM answers and advance walks."""
        if answers_df is None or answers_df.empty:
            return

        answer_map: Dict[str, Dict[str, Any]] = {}
        for _, row in answers_df.iterrows():
            wid = str(row.get("walk_id", row.get("sample_id", ""))).split("_step")[0]
            if "_step" in str(row.get("sample_id", "")):
                wid = str(row["sample_id"]).rsplit("_step", 1)[0]
            elif "walk_id" in row:
                wid = str(row["walk_id"])
            answer_map[wid] = dict(row)

        for walk in list(self.active_walks):
            ans_row = answer_map.get(walk.walk_id)
            if not ans_row:
                walk.active = False
                if walk.history:
                    walk.history[-1].termination_reason = "no_answer"
                continue

            raw_answer = str(ans_row.get("answer", ""))
            current_step = walk.history[-1] if walk.history else None
            if current_step is None:
                walk.active = False
                continue

            current_step.answer_raw = raw_answer

            # Parse JSON response
            chosen_face, reasoning, stop = self._parse_answer(raw_answer, current_step.faces_shown)
            current_step.face_chosen = chosen_face
            current_step.reasoning = reasoning

            # Check stop signal (independent mode)
            if stop and self.termination_mode == "independent":
                walk.active = False
                current_step.termination_reason = "stop"
                walk.step_n += 1
                continue

            # Resolve face to neighbor
            neighbor = self.graph.resolve_face_to_neighbor(
                walk.current_recording_id, chosen_face, self.bearing_tolerance
            )

            if neighbor is None:
                walk.active = False
                current_step.termination_reason = "dead_end"
                walk.step_n += 1
                continue

            # Check revisits
            if not self.allow_revisits and neighbor.recording_id in walk.visited:
                walk.active = False
                current_step.termination_reason = "revisit_blocked"
                walk.step_n += 1
                continue

            # Record movement
            current_step.next_recording_id = neighbor.recording_id
            current_step.distance_m = neighbor.distance_m
            current_step.bearing_deg = neighbor.bearing_deg

            # Advance
            walk.current_recording_id = neighbor.recording_id
            walk.current_arrival_face = self.graph.arrival_face(
                current_step.recording_id, neighbor.recording_id
            )
            walk.step_n += 1

            # Check max steps
            if walk.step_n >= self.max_steps:
                walk.active = False
                # The termination_reason goes on the NEXT implicit step or we mark last
                current_step.termination_reason = "max_steps"

    def all_traces(self) -> pd.DataFrame:
        """Flatten all walk histories into output trace DataFrame."""
        rows = []
        for walk in self.walks.values():
            for step in walk.history:
                rows.append({
                    "walk_id": walk.walk_id,
                    "step_n": step.step_n,
                    "recording_id": step.recording_id,
                    "arrival_face": step.arrival_face,
                    "faces_shown": ",".join(step.faces_shown),
                    "face_chosen": step.face_chosen,
                    "reasoning": step.reasoning,
                    "lat": step.lat,
                    "lon": step.lon,
                    "bearing_deg": step.bearing_deg,
                    "next_recording_id": step.next_recording_id,
                    "distance_m": step.distance_m,
                    "termination_reason": step.termination_reason,
                    "answer_raw": step.answer_raw,
                })
        if not rows:
            return pd.DataFrame(columns=[
                "walk_id", "step_n", "recording_id", "arrival_face", "faces_shown",
                "face_chosen", "reasoning", "lat", "lon", "bearing_deg",
                "next_recording_id", "distance_m", "termination_reason", "answer_raw",
            ])
        return pd.DataFrame(rows)

    def _resolve_face_image_path(self, recording_id: str, face: str, data_cfg: Any) -> Optional[str]:
        """Resolve image file path for a recording face.

        Strategy:
        1. If image_root is set, construct path from pattern.
        2. Otherwise, look up from the parquet-based image lookup table.
        """
        image_root = str(getattr(data_cfg, "image_root", ""))
        image_pattern = str(getattr(data_cfg, "image_pattern", "{recording_id}_{face}.jpg"))

        if image_root:
            filename = image_pattern.format(recording_id=recording_id, face=face)
            return os.path.join(image_root, filename)

        # Parquet-based lookup (populated by _build_image_lookup)
        return self._image_lookup.get((recording_id, face))

    def _build_image_lookup(self, data_cfg: Any) -> None:
        """Build a lookup table from metadata parquet mapping (recording_id, face) -> image_path."""
        parquet_path = str(getattr(data_cfg, "parquet_path", ""))
        if not parquet_path or not os.path.exists(parquet_path):
            self._image_lookup: Dict[tuple, str] = {}
            return

        df = pd.read_parquet(parquet_path, columns=["recording_id", "face", "image_path"])
        self._image_lookup = {}
        for _, row in df.iterrows():
            key = (str(row["recording_id"]), str(row["face"]))
            self._image_lookup[key] = str(row["image_path"])

    def _render_prompt(self, walk: WalkState, available_faces: List[str]) -> str:
        """Render the Jinja2 user prompt template for a step.

        Note: The system prompt is NOT included here. It is injected
        separately by urbanvqa's preprocess_simple via cfg.prompt.system,
        which wraps it as a proper {"role": "system"} message.
        """
        prompt_cfg = getattr(self.cfg, "prompt", {})
        user_template = str(getattr(prompt_cfg, "user_template", ""))

        if not user_template:
            user_template = self._default_prompt_template()

        # Build compass direction descriptions
        yaw = self.graph.yaw_degrees.get(walk.current_recording_id, 0.0)
        directions = []
        for i, face in enumerate(available_faces):
            abs_bearing = _normalize_bearing(yaw + FACE_BEARING_DEG[face])
            compass = self._bearing_to_compass(abs_bearing)
            label = _COMPASS_LABELS.get(face, face)
            panel = ["left panel", "center panel", "right panel"][i] if i < 3 else f"panel {i+1}"
            directions.append(f"- {face} ({label}, {compass}, {panel})")

        direction_text = "\n".join(directions)

        # History summary
        history_text = ""
        if self.include_history and walk.history:
            recent = walk.history[-self.history_max_steps:]
            history_lines = []
            for step in recent:
                if step.face_chosen:
                    history_lines.append(f"Step {step.step_n}: chose {step.face_chosen} at ({step.lat:.4f}, {step.lon:.4f})")
            if history_lines:
                history_text = "Recent history:\n" + "\n".join(history_lines)

        try:
            import jinja2

            tmpl = jinja2.Template(user_template)
            rendered = tmpl.render(
                step_n=walk.step_n,
                directions=direction_text,
                history=history_text,
                available_faces=", ".join(available_faces),
                walk_id=walk.walk_id,
                termination_mode=self.termination_mode,
            )
        except ImportError:
            rendered = user_template.replace("{{ directions }}", direction_text)
            rendered = rendered.replace("{{ history }}", history_text)
            rendered = rendered.replace("{{ available_faces }}", ", ".join(available_faces))
            rendered = rendered.replace("{{ step_n }}", str(walk.step_n))

        return rendered

    @staticmethod
    def _default_prompt_template() -> str:
        return (
            "You are exploring a city on foot. You see three street views arranged left to right.\n\n"
            "Available directions:\n{{ directions }}\n\n"
            "{% if history %}{{ history }}\n\n{% endif %}"
            "Choose which direction to walk by selecting one face ({{ available_faces }}).\n"
            "{% if termination_mode == 'independent' %}"
            "You may also choose to STOP if you feel you've reached an interesting destination.\n"
            "{% endif %}"
            'Respond with JSON: {"chosen_face": "<F|R|B|L>", "reasoning": "<brief reason>", "stop": false}\n'
        )

    @staticmethod
    def _bearing_to_compass(bearing: float) -> str:
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int(round(bearing / 45.0)) % 8
        return dirs[idx]

    def _parse_answer(
        self, raw: str, available_faces: List[str]
    ) -> tuple:
        """Parse structured JSON answer. Returns (chosen_face, reasoning, stop)."""
        chosen_face = available_faces[0]  # fallback
        reasoning = ""
        stop = False

        try:
            # Try to extract JSON from the answer
            text = raw.strip()
            # Handle markdown code blocks
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        text = part
                        break

            data = json.loads(text)
            face = str(data.get("chosen_face", "")).strip().upper()
            if face in available_faces:
                chosen_face = face
            reasoning = str(data.get("reasoning", ""))
            stop = bool(data.get("stop", False))
        except (json.JSONDecodeError, AttributeError, KeyError):
            # Try to find a face letter in the raw text
            raw_upper = raw.upper().strip()
            for face in available_faces:
                if face in raw_upper:
                    chosen_face = face
                    break
            reasoning = f"parse_fallback: {raw[:200]}"

        return chosen_face, reasoning, stop


# --- Retry helper ---

def _run_vqa_with_retry(
    batch_df: pd.DataFrame,
    local_cfg: DictConfig,
    max_retries: int = 2,
    backoff_s: float = 5.0,
) -> pd.DataFrame:
    """Call run_vqa_stage with retry on failure.

    On exhausted retries, returns a DataFrame with empty answers so that
    advance_from_answers can deactivate the affected walks gracefully
    instead of crashing the entire experiment.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1 + max_retries):
        try:
            answers = run_vqa_stage(batch_df, local_cfg)
            if hasattr(answers, "to_pandas"):
                answers = answers.to_pandas()
            # Rejoin walk_id from sample_id
            if "walk_id" not in answers.columns and "sample_id" in answers.columns:
                answers["walk_id"] = answers["sample_id"].apply(
                    lambda x: str(x).rsplit("_step", 1)[0]
                )
            return answers
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = backoff_s * (2 ** attempt)
                print(
                    f"[roaming_vqa] Inference failed (attempt {attempt + 1}/{1 + max_retries}): "
                    f"{type(exc).__name__}: {exc}. Retrying in {wait:.0f}s...",
                    flush=True,
                )
                time.sleep(wait)
            else:
                print(
                    f"[roaming_vqa] Inference failed after {1 + max_retries} attempts: "
                    f"{type(exc).__name__}: {exc}. "
                    f"Returning empty answers — affected walks will be deactivated.",
                    flush=True,
                )

    # Return empty answers so walks get deactivated with "no_answer"
    return pd.DataFrame(columns=["sample_id", "walk_id", "answer"])


# --- Main entry point ---

def _resolve_checkpoint_path(cfg: DictConfig) -> Optional[str]:
    """Determine checkpoint file path from config."""
    roaming_cfg = getattr(cfg, "roaming", {})
    checkpoint_dir = getattr(roaming_cfg, "checkpoint_dir", None)
    if checkpoint_dir is None:
        runtime_cfg = getattr(cfg, "runtime", {})
        checkpoint_dir = getattr(runtime_cfg, "output_dir", None)
    if checkpoint_dir is None:
        return None
    return os.path.join(str(checkpoint_dir), "roaming_checkpoint.json")


def run_roaming_vqa_stage(
    seeds_df: pd.DataFrame,
    cfg: DictConfig,
    graph: Optional[StreetGraph] = None,
) -> pd.DataFrame:
    """Execute the roaming VQA stage with checkpoint/resume support.

    Args:
        seeds_df: DataFrame from sample_walk_seeds() with walk_id, seed_recording_id, seed_face.
        cfg: Full Hydra config.
        graph: Optional prebuilt StreetGraph.

    Returns:
        Trace DataFrame with all walk steps.
    """
    # Build or load graph
    if graph is None:
        graph_cfg = getattr(cfg, "graph", {})
        data_cfg = getattr(cfg, "data", {})
        metadata_parquet = str(
            getattr(graph_cfg, "metadata_parquet", "") or
            getattr(data_cfg, "metadata_parquet", "") or
            getattr(data_cfg, "parquet_path", "")
        )
        graph = build_street_graph(metadata_parquet, graph_cfg)

    roaming_cfg = getattr(cfg, "roaming", {})
    max_steps = int(getattr(roaming_cfg, "max_steps", 10))
    start_step = 0

    # --- Checkpoint resume ---
    checkpoint_path = _resolve_checkpoint_path(cfg)
    resumed = False

    if checkpoint_path:
        ckpt_data = load_checkpoint(checkpoint_path)
        if ckpt_data is not None:
            # Restore walk states from checkpoint
            walks = [
                _walk_state_from_dict(wd)
                for wd in ckpt_data["walks"].values()
            ]
            start_step = ckpt_data["step_iter"] + 1
            n_active = sum(1 for w in walks if w.active)
            print(
                f"[roaming_vqa] Resumed from checkpoint at step {ckpt_data['step_iter']}: "
                f"{len(walks)} walks ({n_active} active), continuing from step {start_step}",
                flush=True,
            )
            resumed = True

    if not resumed:
        # Initialize walk states from seeds
        walks = []
        for _, row in seeds_df.iterrows():
            ws = WalkState(
                walk_id=str(row["walk_id"]),
                current_recording_id=str(row["seed_recording_id"]),
                current_arrival_face=str(row.get("seed_face", "F")),
            )
            walks.append(ws)

    stepper = RoamingStepper(graph, walks, cfg)

    # Build image lookup from metadata
    data_cfg = getattr(cfg, "data", {})
    stepper._build_image_lookup(data_cfg)

    # Pre-flight: verify image resolution works for seed recordings
    image_root = str(getattr(data_cfg, "image_root", ""))
    if not image_root and not stepper._image_lookup:
        raise ValueError(
            "Cannot resolve images: data.image_root is empty and no image_path "
            "column found in the metadata parquet. Set data.image_root or ensure "
            "the parquet contains recording_id, face, and image_path columns."
        )

    if not resumed:
        n_checked, n_found = 0, 0
        for ws in walks[:min(5, len(walks))]:
            for face in ("F", "R", "L"):
                path = stepper._resolve_face_image_path(ws.current_recording_id, face, data_cfg)
                n_checked += 1
                if path and os.path.exists(path):
                    n_found += 1
        if n_checked > 0 and n_found == 0:
            sample_rid = walks[0].current_recording_id if walks else "?"
            sample_path = stepper._resolve_face_image_path(sample_rid, "F", data_cfg)
            raise FileNotFoundError(
                f"Pre-flight image check failed: 0/{n_checked} sample images found on disk. "
                f"Example path: {sample_path}. Check data.image_root or image_path values in parquet."
            )
        if n_checked > 0:
            print(f"[roaming_vqa] Pre-flight image check: {n_found}/{n_checked} sample images found", flush=True)

    runtime_cfg = getattr(cfg, "runtime", {})
    skip_inference = bool(getattr(runtime_cfg, "skip_inference", False))
    step_max_retries = int(getattr(roaming_cfg, "step_max_retries", 2))
    step_retry_backoff_s = float(getattr(roaming_cfg, "step_retry_backoff_s", 5.0))

    print(f"[roaming_vqa] Starting {len(walks)} walks, max {max_steps} steps, "
          f"mode={stepper.termination_mode}"
          f"{f', resuming from step {start_step}' if start_step > 0 else ''}",
          flush=True)

    for step_iter in range(start_step, max_steps):
        active = stepper.active_walks
        if not active:
            print(f"[roaming_vqa] All walks terminated at step {step_iter}", flush=True)
            break

        print(f"[roaming_vqa] Step {step_iter}: {len(active)} active walks", flush=True)

        # Prepare batch
        batch_df = stepper.prepare_step_batch(data_cfg)
        if batch_df.empty:
            print(f"[roaming_vqa] No valid batches at step {step_iter}, stopping", flush=True)
            break

        # Run VQA inference
        if skip_inference:
            # Deterministic debug mode: pick random face
            answers = batch_df.copy()
            rng = np.random.default_rng(42 + step_iter)
            dummy_answers = []
            for _, row in batch_df.iterrows():
                wid = str(row["walk_id"])
                walk = stepper.walks[wid]
                faces = walk.history[-1].faces_shown if walk.history else ["F", "R", "L"]
                chosen = faces[int(rng.integers(0, len(faces)))]
                dummy_answers.append(json.dumps({"chosen_face": chosen, "reasoning": "debug", "stop": False}))
            answers["answer"] = dummy_answers
        else:
            # Configure for single stitched image.
            # Override user_template to "{{prompt}}" so run_vqa_stage passes
            # through the already-rendered prompt instead of replacing it
            # with the raw Jinja2 template.
            model_cfg = OmegaConf.to_container(cfg, resolve=False)
            local_cfg = OmegaConf.create(model_cfg)
            OmegaConf.update(local_cfg, "model.engine_kwargs.limit_mm_per_prompt.image", 1, merge=True)
            OmegaConf.update(local_cfg, "prompt.user_template", "{{prompt}}", merge=True)

            answers = _run_vqa_with_retry(
                batch_df, local_cfg, max_retries=step_max_retries, backoff_s=step_retry_backoff_s,
            )

        stepper.advance_from_answers(answers)

        # Save checkpoint after each step
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            save_checkpoint(checkpoint_path, stepper.walks, step_iter, max_steps)

    # Clean up checkpoint on successful completion
    if checkpoint_path and os.path.exists(checkpoint_path):
        finished_path = checkpoint_path.replace(".json", ".finished.json")
        os.replace(checkpoint_path, finished_path)
        print(f"[roaming_vqa] Run complete, checkpoint moved to {finished_path}", flush=True)

    traces = stepper.all_traces()
    print(f"[roaming_vqa] Completed: {len(traces)} total steps across "
          f"{traces['walk_id'].nunique() if not traces.empty else 0} walks", flush=True)
    return traces
