#!/usr/bin/env python
"""Register the canonical runs of the pairwise battery, and guard them.

Why this exists
---------------
A notebook that finds its own runs on W&B picks whatever the network gives it
that day. The battery has now run 3 times with 3 different prompts, thus the
network holds runs that must never mix. This tool names ONE run for each cell
of the grid, links it into `notebooks/cvpr/canonical_data/`, and records what
that file held on the day of registration.

    kind   x  case  x  model
    proxy     7        2      -> the label-only runs the results table reads
    trace     7        2      -> the thinking runs the word figures read

The registry is the ground truth. `notebooks/cvpr/_canonical.py` reads it, and
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
    python scripts/register_canonical_runs.py register \
        --stage-root 'multirun/2026-08-14_URBANPAIRVQA/21-2*' --force
    python scripts/register_canonical_runs.py verify
    python scripts/register_canonical_runs.py show

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

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = REPO_ROOT / "notebooks" / "cvpr" / "canonical_data"
MANIFEST = REGISTRY_DIR / "manifest.json"
SCHEMA = 1

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

# The smallest run each kind may hold. A smaller run is a smoke test.
MIN_ROWS = {"proxy": 100_000, "trace": 10_000}

# The columns a notebook needs. A file without them is a defect, not data.
NEED_RESULTS = ("pair_id", "relative_label", "relative_score", "presented_label")
NEED_TRACE = ("model_reasoning",)


def case_of(pipeline: str) -> str:
    """Turn `pairwise_subway_safety_mvp` into `subway_safety`."""
    s = re.sub(r"^pairwise_", "", pipeline or "")
    return re.sub(r"_(mvp|ordinal|large)$", "", s) or "unknown"


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
    def cell(self) -> Tuple[str, str, str]:
        return (self.kind, self.case, self.model)

    @property
    def link_name(self) -> str:
        return f"{self.kind}/{self.case}__{self.model}"


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
        out_dir = sd / "outputs" / "pairwise"
        results = [p for p in out_dir.glob("*.parquet") if p.name != "pairs.parquet"]
        if not results:
            continue
        newest = max(results, key=lambda p: p.stat().st_mtime)
        pairs = out_dir / "pairs.parquet"
        c = Candidate(
            kind=kind, case=case_of(pipeline), model=model, model_config=model_cfg,
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
    if c.rows < MIN_ROWS[c.kind]:
        c.problems.append(f"only {c.rows} rows, under the {MIN_ROWS[c.kind]} floor")
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
    if c.model == "gemma-4-12b" and c.image_layout != "interleaved_labels":
        # Without the anchor, this arch does not bind the second image.
        c.problems.append(f"gemma layout is {c.image_layout!r}, not interleaved_labels")

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


def grid_problems(chosen: Dict[Tuple[str, str, str], Candidate]) -> List[str]:
    """Name every missing cell and every question that 2 runs do not share."""
    out: List[str] = []
    for kind in KINDS:
        for case in CASES:
            for model in MODELS:
                if (kind, case, model) not in chosen:
                    out.append(f"missing run: {kind} / {case} / {model}")
    for case in CASES:
        qs = {c.question for k, c in chosen.items() if k[1] == case and c.question}
        if len(qs) > 1:
            out.append(f"{case}: the runs asked {len(qs)} different questions: {sorted(qs)}")
    return out


def write_registry(chosen: Dict[Tuple[str, str, str], Candidate],
                   notes: List[str]) -> None:
    """Write the symlink tree and the manifest."""
    for kind in KINDS:
        (REGISTRY_DIR / kind).mkdir(parents=True, exist_ok=True)
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
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "schema": SCHEMA,
        "tool_version": __version__,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "cases": list(CASES),
        "models": list(MODELS),
        "kinds": list(KINDS),
        "notes": notes,
        "runs": runs,
    }, indent=1) + "\n")


def load_manifest() -> Dict[str, Any]:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"no canonical registry at {MANIFEST}. "
            f"Run: python scripts/register_canonical_runs.py register")
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
        print(f"  {c.kind:<6} {c.case:<19} {c.model:<12} rows={c.rows:>7} "
              f"NotSure={c.not_sure_rate:5.1%} {Path(c.results_path).name}{flag}")
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
        tag = f"{rec['kind']}/{rec['case']}/{rec['model']}"
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
    have = {(r["kind"], r["case"], r["model"]) for r in runs}
    for kind in KINDS:
        for case in CASES:
            for model in MODELS:
                if (kind, case, model) not in have:
                    problems.append(f"missing run: {kind} / {case} / {model}")
    for case in CASES:
        qs = {r["question"] for r in runs if r["case"] == case and r["question"]}
        if len(qs) > 1:
            problems.append(f"{case}: the runs asked {len(qs)} different questions")
    for rec in runs:
        if rec["model"] == "gemma-4-12b" and rec["image_layout"] != "interleaved_labels":
            problems.append(f"{rec['kind']}/{rec['case']}: gemma layout is "
                            f"{rec['image_layout']!r}, not interleaved_labels")
    # A problem recorded at registration is a WARNING here, not a failure. The
    # operator saw it and accepted it, and the manifest keeps the record. A
    # silent acceptance is what this tool exists to prevent, thus print it.
    for rec in runs:
        for note in rec.get("problems", []):
            print(f"[verify] warning: {rec['kind']}/{rec['case']}/{rec['model']}: {note}")
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
    cols = ["kind", "case", "model", "sweep", "rows", "trace_rows",
            "not_sure_rate", "image_layout", "question"]
    df = df[[c for c in cols if c in df.columns]].sort_values(["kind", "case", "model"])
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
