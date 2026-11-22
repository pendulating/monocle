#!/usr/bin/env python
"""Register the canonical runs of the pairwise battery, and guard them.

Why this exists
---------------
A notebook that finds its own runs on W&B picks whatever the network gives it
that day. The battery has now run 3 times with 3 different prompts, thus the
network holds runs that must never mix. This tool names ONE run for each cell
of the grid, links it into `research/notebooks/cvpr/canonical_data/`, and records what
that file held on the day of registration.

    kind   x  arm  x  case  x  model
    proxy    canonical   7        2   -> the label-only runs the results table reads
    trace    canonical   7        2   -> the thinking runs the word figures read
    proxy    noabstain   7        2   -> the forced-choice ablation
    proxy    nocity      7        2   -> the ablation with no place name

The ARM is what keeps an ablation out of a canonical cell. Every arm runs the
same 2 model strings, thus the model alone cannot tell them apart. The arm
comes from what the run DID, which the resolved Hydra config records.

The registry is the ground truth. `research/notebooks/cvpr/_canonical.py` reads it, and
`_provenance.discover_runs` reads that, so no notebook chooses a run again.

The gate
--------
`verify` re-reads every registered file and compares it with the manifest: the
symlink resolves, the size and the SHA-256 match, the row count matches, the
grid is complete, and the 2 models of a case asked the SAME question. A change
to any of these means the paper numbers no longer come from the registered
runs, thus `verify` exits 1 and the export scripts stop.

Usage
-----
    python research/scripts/register_canonical_runs.py register \
        --stage-root 'multirun/2026-08-14_URBANPAIRVQA/21-2*' --force
    python research/scripts/register_canonical_runs.py verify
    python research/scripts/register_canonical_runs.py show

Run it from the canonical venv (`.venv-mllmsci-vllm025cu129`).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

__version__ = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_DIR = REPO_ROOT / "research" / "notebooks" / "cvpr" / "canonical_data"
MANIFEST = REGISTRY_DIR / "manifest.json"
SCHEMA = 2

# The 7 cases of the consolidated battery, in the order the paper uses.
CASES: Tuple[str, ...] = (
    "subway_safety", "libraries", "schools", "road_quality",
    "parks_plazas", "restaurants", "street_photography",
)

# The 2 canonical raters. The key is the Hydra model string, and the value is
# the (kind, short name) pair. A model string that is absent here is not
# canonical, thus this map is what keeps a probe run out of the registry.
MODEL_KINDS: Dict[str, Tuple[str, str]] = {
    "qwen3.5-9b/instruct": ("proxy", "qwen3.5-9b"),
    "gemma-4-12b/instruct": ("proxy", "gemma-4-12b"),
    "qwen3.5-9b/instruct_thinking": ("trace", "qwen3.5-9b"),
    "gemma-4-12b/instruct_thinking": ("trace", "gemma-4-12b"),
}

KINDS: Tuple[str, ...] = ("proxy", "trace")
MODELS: Tuple[str, ...] = ("gemma-4-12b", "qwen3.5-9b")

# The arms of the battery. `canonical` is the prompt the paper reports; the
# others change ONE thing each and are read against it.
#
# | Arm         | What changes                                              |
# |-------------|-----------------------------------------------------------|
# | `canonical` | nothing — the registered prompt, abstention allowed       |
# | `noabstain` | the decoder cannot emit `NotSure`                         |
# | `nocity`    | the prompt does not name the city                         |
# | `textfirst` | the prompt text comes BEFORE the images                   |
# | `matched`   | nothing — the canonical prompt on the required layout,    |
# |             | over the 100,000-pair table of `nocity` and `textfirst`   |
# | `paraphrase`| the subway-safety user text reworded (subway only, 25k)   |
# | `persona`   | an expert system turn added (subway only, 25k)            |
#
# `matched` is a CONTROL, not an ablation. The canonical gemma-4-12b cell ran
# `interleaved_labels` at 1,000,000 pairs and shares 91.7% of the `pair_id`
# rows of the 100,000-pair arms. Read the gemma half of `nocity` and
# `textfirst` against `matched`. See research/docs/prompt-ablations.md.
#
# Warning: an arm is NOT a model and NOT a case. Do not read a number of one
# arm beside a number of another without saying which arm it comes from.
ARMS: Tuple[str, ...] = ("canonical", "noabstain", "nocity", "textfirst",
                         "matched", "paraphrase", "persona")

# The arms that hold 1 case on purpose. The robustness battery of 2026-09-04
# (`research/docs/robustness-battery-rerun-plan.md`, tier B) re-runs
# the subway-safety proxies under 2 wording arms at 25,000 pairs. The other
# 6 cases were never meant to run there, thus the grid must not ask for them.
ARM_CASES: Dict[str, Tuple[str, ...]] = {
    "paraphrase": ("subway_safety",),
    "persona": ("subway_safety",),
}

# The arms that change the QUESTION TEXT on purpose. The question gate ("the
# runs of a case asked 1 question") does not apply to them: a paraphrase that
# the gate refuses cannot exist. `persona` keeps the user text and adds a
# system turn, thus it is not in this list.
QUESTION_PROBE_ARMS: Tuple[str, ...] = ("paraphrase",)

# The arms that change the image layout ON PURPOSE. `REQUIRED_LAYOUT` does not
# apply to them: a layout probe that the layout gate refuses cannot exist.
#
# `matched` joined this list on 2026-09-02, when the canonical layout went back
# to `interleaved_labels`. That arm holds the canonical prompt on
# `images_then_labels`, thus it is now a layout probe and not a control.
LAYOUT_PROBE_ARMS: Tuple[str, ...] = ("textfirst", "matched")

# The layouts that put the prompt before the images. A run with one of them is
# the `textfirst` arm, whatever else it does.
TEXT_FIRST_LAYOUTS: Tuple[str, ...] = ("text_first", "text_first_labels",
                                       "text_first_interleaved")

# The pipeline suffixes that name an arm. `case_of` takes them off, thus
# `pairwise_libraries_mvp_nocity` and `pairwise_libraries_mvp` share a case.
# `_matched` inherits its base case and overrides NOTHING: the suffix is the
# only signal, because the run did what the canonical run did.
ARM_SUFFIXES: Tuple[str, ...] = ("nocity", "matched")

# The arms that hold 1 rater on purpose. `matched` exists because the gemma
# control ran another layout; the qwen control ran the same layout as its
# arms, thus qwen needs no matched control and the grid must not ask for it.
ARM_MODELS: Dict[str, Tuple[str, ...]] = {"matched": ("gemma-4-12b",)}

#: The ONLY image layout a gemma-4-12b run may register with.
#:
#: "Image A:", image A, "Image B:", image B, prompt
#:
#: Decision of 2026-09-02: the canonical layout is `interleaved_labels`. The
#: whole 1,000,000-pair canonical gemma battery and the trace runs ran it, and
#: on the greedy path it abstains 6 to 16 points LESS than
#: `images_then_labels` on the 5 cases that abstain (measured pair for pair on
#: about 100,900 shared pairs for each case, 2026-09-02). Text must still sit
#: BETWEEN the two images: the arch reads two adjacent image blocks as ONE
#: image (55 of 70 traces, 2026-08-20). This layout keeps that rule.
#:
#: From 2026-08-20 to 2026-09-02 this constant said `images_then_labels`,
#: because the gemma-4 card asks for an image to open the message. The
#: `noabstain`, `nocity` and `textfirst` gemma arms ran under that rule and
#: thus carry a recorded layout problem until they run again. See
#: vlm-narratives-docs/gemma-4-best-practices.md.
REQUIRED_LAYOUT = "interleaved_labels"


# The smallest run each kind may hold. A smaller run is a smoke test.
#
# The proxy floor went 100,000 -> 1,000,000 on 2026-08-20, when the battery
# moved to `million_proxy_*`. The floor is what enforces the move: a
# `looks_proxy_*` run holds 110,000 rows, thus it now fails to register and
# cannot come back into a paper figure by accident.
# The `nocity` arm ran at 100,000 pairs on purpose, thus it carries its own
# floor. A floor that is too high would refuse a run that is complete.
MIN_ROWS: Dict[Tuple[str, str], int] = {
    ("proxy", "canonical"): 1_000_000,
    ("proxy", "noabstain"): 1_000_000,
    ("proxy", "nocity"):      100_000,
    ("proxy", "textfirst"):   100_000,
    ("proxy", "matched"):     100_000,
    ("proxy", "paraphrase"):   25_000,
    ("proxy", "persona"):      25_000,
    ("trace", "canonical"):    10_000,
}
MIN_ROWS_DEFAULT = 10_000


def min_rows(kind: str, arm: str) -> int:
    return MIN_ROWS.get((kind, arm), MIN_ROWS_DEFAULT)

# The columns a notebook needs. A file without them is a defect, not data.
NEED_RESULTS = ("pair_id", "relative_label", "relative_score", "presented_label")
NEED_TRACE = ("model_reasoning",)


def case_of(pipeline: str) -> str:
    """Turn `pairwise_subway_safety_mvp` into `subway_safety`.

    An ablation pipeline carries an arm suffix, such as
    `pairwise_subway_safety_mvp_nocity`. The case is the same case, thus the
    suffix comes off here and the arm carries the difference.
    """
    s = re.sub(r"^pairwise_", "", pipeline or "")
    s = re.sub(r"_(" + "|".join(ARM_SUFFIXES) + r")$", "", s)
    return re.sub(r"_(mvp|ordinal|large)$", "", s) or "unknown"


def read_config(stage_dir: Path) -> Dict[str, Any]:
    """Read the resolved Hydra config of a run, or an empty dict."""
    f = stage_dir / ".hydra" / "config.yaml"
    if not f.exists():
        return {}
    try:
        return yaml.safe_load(f.read_text()) or {}
    except Exception:
        return {}


def arm_of(pipeline: str, cfg: Dict[str, Any]) -> str:
    """Name the arm a run belongs to, from what the run DID.

    Do NOT read the sweep name for this. A sweep can be renamed or copied, and
    the resolved config is what the GPU actually ran.

    | Arm         | Signal                                              |
    |-------------|-----------------------------------------------------|
    | `nocity`    | the pipeline name ends in `_nocity`                 |
    | `noabstain` | `prompt.structured_output.allow_not_sure` is false  |
    | `textfirst` | `prompt.image_layout` puts the prompt first         |
    | `matched`   | the pipeline name ends in `_matched`                |
    | `paraphrase`| the user text opens "The first image is Image A"   |
    | `persona`   | the prompt carries a system turn                    |
    | `canonical` | no signal                                           |

    A run that carries 2 signals gets a joined name, such as
    `noabstain+nocity`. That arm then stands alone in the grid, which is what
    should happen: it is neither ablation on its own.
    """
    flags = []
    prompt = cfg.get("prompt") or {}
    so = (prompt.get("structured_output") or {})
    if so.get("allow_not_sure") is False:
        flags.append("noabstain")
    if (pipeline or "").endswith("_nocity"):
        flags.append("nocity")
    if (pipeline or "").endswith("_matched"):
        flags.append("matched")
    if str(prompt.get("image_layout") or "") in TEXT_FIRST_LAYOUTS:
        flags.append("textfirst")
    user = str(prompt.get("user_template") or "").lstrip()
    if user.startswith("The first image is Image A"):
        flags.append("paraphrase")
    if prompt.get("system") is not None:
        flags.append("persona")
    return "+".join(flags) if flags else "canonical"


def read_overrides(stage_dir: Path) -> Dict[str, str]:
    """Read a Hydra `overrides.yaml` into a flat dict."""
    f = stage_dir / ".hydra" / "overrides.yaml"
    if not f.exists():
        return {}
    out: Dict[str, str] = {}
    for raw in (yaml.safe_load(f.read_text()) or []):
        if isinstance(raw, str) and "=" in raw:
            k, _, v = raw.partition("=")
            out[k.strip().lstrip("+~")] = v.strip()
    return out


def read_question(stage_dir: Path) -> str:
    """Read the question the run asked, from the RESOLVED Hydra config.

    A case name is not a question. The schools case asked 2 different questions
    in 4 days, and both runs carry the case name `schools`.
    """
    f = stage_dir / ".hydra" / "config.yaml"
    if not f.exists():
        return ""
    try:
        cfg = yaml.safe_load(f.read_text()) or {}
        template = (cfg.get("prompt") or {}).get("user_template", "") or ""
    except Exception:
        return ""
    for line in template.splitlines():
        s = line.strip()
        if s.endswith("?"):
            return s
    return ""


def read_layout(stage_dir: Path) -> str:
    """Read `prompt.image_layout`, which gemma-4-12b needs set correctly."""
    f = stage_dir / ".hydra" / "config.yaml"
    if not f.exists():
        return ""
    try:
        cfg = yaml.safe_load(f.read_text()) or {}
        return str((cfg.get("prompt") or {}).get("image_layout") or "images_then_text")
    except Exception:
        return ""


def sha256_of(path: Path, quick: bool = False) -> str:
    """Hash a file. `quick` hashes the first and last 4 MiB only."""
    h = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as fh:
        if quick and size > 8 << 20:
            h.update(fh.read(4 << 20))
            fh.seek(-(4 << 20), os.SEEK_END)
            h.update(fh.read())
            return "q:" + h.hexdigest()
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


@dataclass
class Candidate:
    """One run that the scan found on disk."""

    kind: str
    arm: str              # canonical, noabstain, nocity
    case: str
    model: str            # the short name, e.g. gemma-4-12b
    model_config: str     # the Hydra model string
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

    @property
    def cell(self) -> Tuple[str, str, str, str]:
        return (self.kind, self.arm, self.case, self.model)

    @property
    def link_name(self) -> str:
        return f"{self.kind}/{self.arm}/{self.case}__{self.model}"


def scan(roots: List[str]) -> List[Candidate]:
    """Find every stage directory under the roots and read what it holds."""
    # Absolute paths only. A symlink resolves against ITS OWN directory, thus a
    # repo-relative target would point at a path under `canonical_data/`.
    stage_dirs: List[Path] = []
    for root in roots:
        for hit in glob.glob(root):
            p = Path(hit).resolve()
            if (p / ".hydra").is_dir():
                stage_dirs.append(p)
            stage_dirs += [d.parent.resolve() for d in p.glob("*/.hydra") if d.is_dir()]
    out: List[Candidate] = []
    for sd in sorted(set(stage_dirs)):
        ov = read_overrides(sd)
        pipeline, model_cfg = ov.get("pipeline"), ov.get("model")
        if not pipeline or model_cfg not in MODEL_KINDS:
            continue
        kind, model = MODEL_KINDS[model_cfg]
        cfg = read_config(sd)
        out_dir = sd / "outputs" / "pairwise"
        results = [p for p in out_dir.glob("*.parquet") if p.name != "pairs.parquet"]
        if not results:
            continue
        newest = max(results, key=lambda p: p.stat().st_mtime)
        pairs = out_dir / "pairs.parquet"
        c = Candidate(
            kind=kind, arm=arm_of(pipeline, cfg),
            case=case_of(pipeline), model=model, model_config=model_cfg,
            sweep=ov.get("sweep", ""), pipeline=pipeline, stage_dir=str(sd),
            results_path=str(newest), pairs_path=str(pairs),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S",
                                     time.localtime(newest.stat().st_mtime)),
            question=read_question(sd), image_layout=read_layout(sd),
        )
        if not pairs.exists():
            c.problems.append("no pairs.parquet")
        out.append(c)
    return out


def inspect(c: Candidate, quick: bool = False) -> Candidate:
    """Read the parquet of one candidate and fill in its facts and problems."""
    p = Path(c.results_path)
    df = pd.read_parquet(p)
    c.rows = len(df)
    c.results_bytes = p.stat().st_size
    c.results_sha256 = sha256_of(p, quick=quick)

    missing = [col for col in NEED_RESULTS if col not in df.columns]
    if missing:
        c.problems.append(f"missing columns: {missing}")
    floor = min_rows(c.kind, c.arm)
    if c.rows < floor:
        c.problems.append(f"only {c.rows} rows, under the {floor} floor")
    if "relative_label" in df.columns:
        lab = df["relative_label"].astype(str)
        c.not_sure_rate = float((lab == "NotSure").mean())
        # A run where 1 label takes almost everything carries no ordering.
        top = lab.value_counts(normalize=True)
        if len(top) and top.iloc[0] > 0.98:
            c.problems.append(
                f"degenerate: {top.index[0]} takes {100 * top.iloc[0]:.1f}% of rows")
    if c.kind == "trace":
        for col in NEED_TRACE:
            if col not in df.columns:
                c.problems.append(f"missing column: {col}")
        if "model_reasoning" in df.columns:
            tr = df["model_reasoning"].astype(str).str.len() > 20
            c.trace_rows = int(tr.sum())
            if tr.mean() < 0.95:
                c.problems.append(
                    f"only {100 * tr.mean():.1f}% of rows carry a trace")
    if not c.question:
        c.problems.append("no question in the resolved config")
    if (c.model == "gemma-4-12b" and c.arm not in LAYOUT_PROBE_ARMS
            and c.image_layout != REQUIRED_LAYOUT):
        # See REQUIRED_LAYOUT. `interleaved_labels` binds the images but opens
        # the message with text, thus it fails the modality-order rule.
        c.problems.append(f"gemma layout is {c.image_layout!r}, "
                          f"not {REQUIRED_LAYOUT!r}")

    # The join the notebooks depend on must work.
    if Path(c.pairs_path).exists():
        try:
            pairs = pd.read_parquet(c.pairs_path, columns=["pair_id"])
            head = df[["pair_id"]].head(2000)
            merged = head.merge(pairs, on="pair_id", how="left", validate="one_to_one")
            if merged["pair_id"].isna().any() or len(merged) != len(head):
                c.problems.append("results do not join 1-to-1 to pairs.parquet")
        except Exception as exc:
            c.problems.append(f"pair join failed: {type(exc).__name__}: {exc}")
    return c


def choose(cands: List[Candidate]) -> Tuple[Dict[Tuple[str, str, str], Candidate], List[str]]:
    """Keep 1 candidate for each cell. Report a cell that holds 2 clean runs."""
    by_cell: Dict[Tuple[str, str, str], List[Candidate]] = {}
    for c in cands:
        by_cell.setdefault(c.cell, []).append(c)
    chosen: Dict[Tuple[str, str, str], Candidate] = {}
    notes: List[str] = []
    for cell, group in by_cell.items():
        clean = [c for c in group if not c.problems]
        pick = clean or group
        if len(pick) > 1:
            pick = sorted(pick, key=lambda c: c.created_at)
            notes.append(
                f"{cell}: {len(pick)} runs match; took the newest "
                f"({Path(pick[-1].results_path).name})")
        chosen[cell] = pick[-1]
    return chosen, notes


def missing_cells(cells: set) -> List[str]:
    """Name every hole in the grid of every arm that the registry holds.

    The `canonical` arm must hold BOTH kinds: the paper reads its proxy runs
    and its trace runs. Another arm must only be complete in the kinds it
    already has, thus a proxy-only ablation does not ask for trace runs it was
    never meant to produce. An arm in `ARM_MODELS` holds only the raters named
    there.
    """
    out: List[str] = []
    for arm in sorted({c[1] for c in cells}):
        kinds = (KINDS if arm == "canonical"
                 else sorted({c[0] for c in cells if c[1] == arm}))
        models = ARM_MODELS.get(arm, MODELS)
        cases = ARM_CASES.get(arm, CASES)
        for kind in kinds:
            for case in cases:
                for model in models:
                    if (kind, arm, case, model) not in cells:
                        out.append(f"missing run: {kind} / {arm} / {case} / {model}")
    return out


def grid_problems(chosen: Dict[Tuple[str, str, str, str], Candidate]) -> List[str]:
    """Name every missing cell and every question that 2 runs do not share."""
    out: List[str] = missing_cells(set(chosen))
    # The question must not move BETWEEN arms either. An ablation changes the
    # prompt, thus this is the test that it changed only what it was meant to.
    for case in CASES:
        qs = {c.question for k, c in chosen.items()
              if k[2] == case and c.question and k[1] not in QUESTION_PROBE_ARMS}
        if len(qs) > 1:
            out.append(f"{case}: the runs asked {len(qs)} different questions: {sorted(qs)}")
    return out


#: The 3 links a cell directory holds. `prune_registry` removes a directory
#: only when it holds these and nothing else, thus it can never delete data.
CELL_LINKS: Tuple[str, ...] = ("results.parquet", "pairs.parquet", "stage")


def prune_registry(wanted: List[str]) -> List[str]:
    """Remove every cell directory that the new registry does not name.

    Schema 1 put a cell at `<kind>/<case>__<model>`, and schema 2 puts it at
    `<kind>/<arm>/<case>__<model>`. Without this step the old directories stay
    on disk and point at runs that the manifest no longer names, which is the
    exact confusion the registry exists to remove.

    A directory goes only when every entry in it is one of `CELL_LINKS`. A
    directory that holds anything else is not ours, thus it stays.
    """
    keep = {(REGISTRY_DIR / w) for w in wanted}
    removed: List[str] = []
    for cell in sorted(REGISTRY_DIR.rglob("*"), key=lambda x: -len(x.parts)):
        if not cell.is_dir() or cell in keep:
            continue
        names = {c.name for c in cell.iterdir()}
        if names and not names <= set(CELL_LINKS):
            continue
        for c in cell.iterdir():
            c.unlink()
        cell.rmdir()
        removed.append(str(cell.relative_to(REGISTRY_DIR)))
    return removed


def write_registry(chosen: Dict[Tuple[str, str, str, str], Candidate],
                   notes: List[str]) -> None:
    """Write the symlink tree and the manifest."""
    for cell in chosen:
        (REGISTRY_DIR / cell[0] / cell[1]).mkdir(parents=True, exist_ok=True)
    runs = []
    for cell in sorted(chosen):
        c = chosen[cell]
        d = REGISTRY_DIR / c.link_name
        d.mkdir(parents=True, exist_ok=True)
        for name, target in (("results.parquet", c.results_path),
                             ("pairs.parquet", c.pairs_path),
                             ("stage", c.stage_dir)):
            link = d / name
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(target, link)
        rec = asdict(c)
        rec["link"] = c.link_name
        runs.append(rec)
    gone = prune_registry([c.link_name for c in chosen.values()])
    for g in gone:
        print(f"[register] pruned stale link dir: {g}")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "schema": SCHEMA,
        "tool_version": __version__,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "cases": list(CASES),
        "models": list(MODELS),
        "kinds": list(KINDS),
        "arms": sorted({c[1] for c in chosen}),
        "notes": notes,
        "runs": runs,
    }, indent=1) + "\n")


def load_manifest() -> Dict[str, Any]:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"no canonical registry at {MANIFEST}. "
            f"Run: python research/scripts/register_canonical_runs.py register")
    blob = json.loads(MANIFEST.read_text())
    if blob.get("schema") != SCHEMA:
        raise ValueError(f"registry schema {blob.get('schema')} != {SCHEMA}")
    return blob


def cmd_register(args: argparse.Namespace) -> int:
    cands = scan(args.stage_root)
    print(f"[register] {len(cands)} candidate runs under {args.stage_root}")
    if not cands:
        print("[register] nothing to do")
        return 1
    for c in cands:
        inspect(c, quick=args.quick)
        flag = "  PROBLEM: " + "; ".join(c.problems) if c.problems else ""
        print(f"  {c.kind:<6} {c.arm:<10} {c.case:<19} {c.model:<12} "
              f"rows={c.rows:>7} NotSure={c.not_sure_rate:5.1%} "
              f"{Path(c.results_path).name}{flag}")
    chosen, notes = choose(cands)
    for n in notes:
        print(f"[register] note: {n}")
    problems = grid_problems(chosen)
    bad = {k: v for k, v in chosen.items() if v.problems}
    if problems:
        for p in problems:
            print(f"[register] GRID PROBLEM: {p}")
    if bad and not args.force:
        print("[register] refused: some chosen runs carry problems. "
              "Fix them, or pass --force to register anyway.")
        return 1
    if problems and not args.force:
        print("[register] refused: the grid is incomplete. Pass --force to register anyway.")
        return 1
    write_registry(chosen, notes + problems)
    print(f"[register] wrote {len(chosen)} runs to {MANIFEST.relative_to(REPO_ROOT)}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    blob = load_manifest()
    runs = blob["runs"]
    problems: List[str] = []
    print(f"[verify] registry of {blob['generated_at']}, {len(runs)} runs")
    for rec in runs:
        link = REGISTRY_DIR / rec["link"] / "results.parquet"
        tag = f"{rec['kind']}/{rec.get('arm', 'canonical')}/{rec['case']}/{rec['model']}"
        if not link.exists():
            problems.append(f"{tag}: the link {link} does not resolve")
            continue
        p = Path(os.path.realpath(link))
        if str(p) != rec["results_path"]:
            problems.append(f"{tag}: the link points at {p}, not the registered file")
        size = p.stat().st_size
        if size != rec["results_bytes"]:
            problems.append(f"{tag}: size {size} != registered {rec['results_bytes']}")
            continue
        if not args.quick:
            digest = sha256_of(p, quick=rec["results_sha256"].startswith("q:"))
            if digest != rec["results_sha256"]:
                problems.append(f"{tag}: the file changed since registration")
                continue
        if args.rows:
            n = len(pd.read_parquet(p, columns=["pair_id"]))
            if n != rec["rows"]:
                problems.append(f"{tag}: {n} rows != registered {rec['rows']}")
    have = {(r["kind"], r.get("arm", "canonical"), r["case"], r["model"])
            for r in runs}
    problems += missing_cells(have)
    for case in CASES:
        qs = {r["question"] for r in runs
              if r["case"] == case and r["question"]
              and r.get("arm", "canonical") not in QUESTION_PROBE_ARMS}
        if len(qs) > 1:
            problems.append(f"{case}: the runs asked {len(qs)} different questions")
    for rec in runs:
        if (rec["model"] == "gemma-4-12b"
                and rec.get("arm", "canonical") not in LAYOUT_PROBE_ARMS
                and rec["image_layout"] != REQUIRED_LAYOUT):
            problems.append(
                f"{rec['kind']}/{rec.get('arm', 'canonical')}/{rec['case']}: "
                f"gemma layout is {rec['image_layout']!r}, not "
                f"{REQUIRED_LAYOUT!r} — re-run this case")
    # A problem recorded at registration is a WARNING here, not a failure. The
    # operator saw it and accepted it, and the manifest keeps the record. A
    # silent acceptance is what this tool exists to prevent, thus print it.
    for rec in runs:
        for note in rec.get("problems", []):
            print(f"[verify] warning: {rec['kind']}/{rec.get('arm', 'canonical')}"
                  f"/{rec['case']}/{rec['model']}: {note}")
    if problems:
        for p in problems:
            print(f"[verify] FAIL: {p}")
        print(f"[verify] {len(problems)} problems")
        return 1
    print("[verify] OK: every run resolves, matches its hash, and the grid is complete")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    blob = load_manifest()
    df = pd.DataFrame(blob["runs"])
    if "arm" not in df.columns:
        df["arm"] = "canonical"
    cols = ["kind", "arm", "case", "model", "sweep", "rows", "trace_rows",
            "not_sure_rate", "image_layout", "question"]
    df = df[[c for c in cols if c in df.columns]].sort_values(
        ["kind", "arm", "case", "model"])
    with pd.option_context("display.width", 200, "display.max_colwidth", 46):
        print(df.to_string(index=False))
    print(f"\nregistered {blob['generated_at']} at commit {blob.get('git_commit', '')[:12]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="scan runs and write the registry")
    r.add_argument("--stage-root", nargs="+",
                   default=["multirun/2026-08-14_URBANPAIRVQA/21-2*"],
                   help="globs of monitor directories or stage directories")
    r.add_argument("--quick", action="store_true",
                   help="hash the first and last 4 MiB only")
    r.add_argument("--force", action="store_true",
                   help="register even with problems or an incomplete grid")
    r.set_defaults(func=cmd_register)

    v = sub.add_parser("verify", help="check the registry against the disk")
    v.add_argument("--quick", action="store_true", help="skip the hash test")
    v.add_argument("--rows", action="store_true", help="read each parquet and count rows")
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("show", help="print the registry")
    s.set_defaults(func=cmd_show)

    args = ap.parse_args()
    os.chdir(REPO_ROOT)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
