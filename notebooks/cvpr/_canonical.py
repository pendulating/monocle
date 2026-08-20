"""The canonical run registry — the single ground truth for every CVPR figure.

A notebook must NOT choose its own runs. The battery has run 3 times with 3
different prompts, and W&B holds all of them, thus a notebook that queries the
network can silently read a run that answers a question the paper no longer
asks. This module reads `canonical_data/manifest.json` instead, which names
exactly 1 run for each cell of the grid:

    kind   x  case  x  model
    proxy     7        2      -> the label-only runs behind the results table
    trace     7        2      -> the thinking runs behind the word figures

`scripts/register_canonical_runs.py` writes the registry and the symlinks under
`canonical_data/`. That tool also verifies them, and every export script calls
`verify_or_raise()` here before it writes a file.

To move the paper to a new battery:

1. Run the new sweeps.
2. `python scripts/register_canonical_runs.py register --stage-root '<glob>'`
3. `python scripts/register_canonical_runs.py verify`
4. Run the export scripts again.

Nothing else selects a run. A notebook that needs the old behaviour can ask
`_provenance.discover_runs(..., source="wandb")`, but no paper figure may.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__version__ = "1.0.0"

HERE = Path(__file__).resolve().parent
REGISTRY_DIR = HERE / "canonical_data"
MANIFEST = REGISTRY_DIR / "manifest.json"
SCHEMA = 1

# The environment variable that turns the guard off. It exists for a debug
# session only. A paper figure must never run with it set.
ALLOW_UNVERIFIED = os.environ.get("CVPR_ALLOW_UNVERIFIED") == "1"


@dataclass
class CanonicalRun:
    """One registered run. The fields mirror the manifest."""

    kind: str
    case: str
    model: str            # short name, e.g. gemma-4-12b
    model_config: str     # the Hydra model string, e.g. gemma-4-12b/instruct
    sweep: str
    pipeline: str
    stage_dir: str
    results_path: str
    pairs_path: str
    created_at: str
    rows: int = 0
    trace_rows: int = 0
    not_sure_rate: float = float("nan")
    question: str = ""
    image_layout: str = ""
    results_bytes: int = 0
    results_sha256: str = ""
    problems: List[str] = field(default_factory=list)
    link: str = ""

    @property
    def link_dir(self) -> Path:
        return REGISTRY_DIR / self.link

    @property
    def results_link(self) -> str:
        """The path a notebook should read: the symlink, not the target.

        Read through the link and the provenance of a figure names the
        registry. Read the target and it names a run directory, which tells a
        reader nothing about whether that run is the canonical one.
        """
        return str(self.link_dir / "results.parquet")

    @property
    def pairs_link(self) -> str:
        return str(self.link_dir / "pairs.parquet")

    @property
    def run_id(self) -> str:
        """A stable id that stands where a W&B id used to stand."""
        return f"canon:{self.kind}:{self.case}:{self.model}"


def load(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the manifest. Raises when it is absent or holds another schema."""
    p = Path(path or MANIFEST)
    if not p.exists():
        raise FileNotFoundError(
            f"no canonical registry at {p}.\n"
            f"Run: python scripts/register_canonical_runs.py register")
    blob = json.loads(p.read_text())
    if blob.get("schema") != SCHEMA:
        raise ValueError(
            f"registry schema {blob.get('schema')} != {SCHEMA}; "
            f"update _canonical.py or register again")
    return blob


def runs(kind: Optional[str] = None,
         case: Optional[str] = None,
         model: Optional[str] = None) -> List[CanonicalRun]:
    """Return the registered runs, filtered.

    Args:
        kind: `proxy`, `trace`, or None for both.
        case: A SUBSTRING of the case name, which is what the notebooks pass.
            `parks` thus matches `parks_plazas`, and an empty string matches
            every case.
        model: The short model name.
    """
    out: List[CanonicalRun] = []
    for rec in load()["runs"]:
        r = CanonicalRun(**{k: v for k, v in rec.items()
                            if k in CanonicalRun.__dataclass_fields__})
        if kind and r.kind != kind:
            continue
        if case and case.lower() not in r.case.lower():
            continue
        if model and r.model != model:
            continue
        out.append(r)
    out.sort(key=lambda r: (r.kind, r.case, r.model))
    return out


def verify_or_raise(quick: bool = True) -> None:
    """Stop unless every registered file is on disk and unchanged.

    This is the gate. An export script calls it BEFORE it writes anything, so a
    stale or broken registry cannot reach the paper.

    `quick=True` tests that each link resolves and that the size matches, which
    costs milliseconds. `quick=False` hashes each file again.
    """
    if ALLOW_UNVERIFIED:
        print("[canonical] WARNING: CVPR_ALLOW_UNVERIFIED=1, the gate is off")
        return
    blob = load()
    problems: List[str] = []
    for rec in blob["runs"]:
        tag = f"{rec['kind']}/{rec['case']}/{rec['model']}"
        link = REGISTRY_DIR / rec["link"] / "results.parquet"
        if not link.exists():
            problems.append(f"{tag}: the link does not resolve")
            continue
        target = Path(os.path.realpath(link))
        if str(target) != rec["results_path"]:
            problems.append(f"{tag}: the link moved to {target}")
        elif target.stat().st_size != rec["results_bytes"]:
            problems.append(f"{tag}: the file changed size")
    if not quick:
        import subprocess
        rc = subprocess.call([
            "python", str(HERE.parents[1] / "scripts" / "register_canonical_runs.py"),
            "verify"])
        if rc != 0:
            problems.append("the full verify failed; see the output above")
    if problems:
        raise RuntimeError(
            "the canonical registry does not match the disk:\n  "
            + "\n  ".join(problems)
            + "\nRegister again: python scripts/register_canonical_runs.py register")


def summary() -> str:
    """A 1-line description for a figure caption or a log."""
    blob = load()
    return (f"canonical registry of {blob['generated_at']} "
            f"({len(blob['runs'])} runs, commit {blob.get('git_commit', '')[:12]})")


def provenance_rows() -> List[Dict[str, Any]]:
    """The registry as plain rows, for a provenance table in an export."""
    return [
        {"kind": r.kind, "case": r.case, "model": r.model_config,
         "sweep": r.sweep, "rows": r.rows, "not_sure_rate": r.not_sure_rate,
         "image_layout": r.image_layout, "question": r.question,
         "results_path": r.results_path, "sha256": r.results_sha256,
         "problems": "; ".join(r.problems)}
        for r in runs()
    ]
