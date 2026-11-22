#!/usr/bin/env python
"""Curate the figures and tables of the CVPR notebooks.

The notebooks write their figures and tables into many folders, and over time
the folders collect old versions, runs that a gate marked stale, and copies in
scratch folders. This script does 4 things, in this order:

1. **Import.** Copy each file of an import folder (`--import <dir>`) into
   `research/notebooks/cvpr/master/figures/`. A file of the same name that is there
   goes into the archive first.
2. **Archive the stale files.** A file is stale when:
   - its path holds `stale` (a `stale_pre_looks/` folder, a `_stale` suffix);
   - a newer version of the same name sits in the same folder
     (`name_v1.0.0.csv` against `name_v1.1.0.csv`);
   - the import replaced it.
   A superseded version whose name, or name tail, a `.py` file under
   `notebooks/`, `scripts/` or `dagspaces/` names is never archived, because
   code reads it.
3. **Label the paper candidates.** Each figure of `PAPER_CANDIDATES` gets the
   prefix `paper_`, with its `.json` sidecar and its `.pdf` sibling. A
   candidate in a stale folder moves up into the case's `figures/` folder.
   A later export writes the name without the prefix; a rerun of this script
   then replaces the `paper_` copy with the new file.
4. **Index.** Write `research/notebooks/cvpr/figure_index.csv`: every figure, its
   folder, its notebook tag, whether it is a paper candidate, whether the
   paper holds it now, and whether it has a sidecar.

The archive is 1 `.tar.gz` under `data/archive/`, with a manifest. Nothing is
lost: `tar -xzf <archive> -C /share/ju/matt/mllmsci` puts every file back.

Use:
    python research/scripts/curate_notebook_figures.py                      # dry run: print the plan
    python research/scripts/curate_notebook_figures.py --apply
    python research/scripts/curate_notebook_figures.py --apply --import <scratch dir> --import <scratch dir>
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import shutil
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

__version__ = "1.0.0"

ROOT = Path("/share/ju/matt/mllmsci")
CVPR = ROOT / "research" / "notebooks" / "cvpr"
MASTER_FIGURES = CVPR / "master" / "figures"
PAPER = ROOT / "research" / "papers" / "cvpr27_machine-perception-public-spaces"
ARCHIVE_DIR = ROOT / "data" / "archive"

PREFIX = "paper_"
FIGURE_EXT = {".png", ".pdf", ".svg"}

# The folders this script curates. A cache folder is not on the list.
CURATED = [
    CVPR / "master" / "figures",
    CVPR / "master" / "outputs",
    CVPR / "master" / "outputs" / "explore",
    CVPR / "master" / "outputs" / "judge_vocab",
] + [CVPR / case / kind
     for case in ("schools", "road", "restaurants", "libraries", "subway", "parks", "plazas",
                  "street_photography")
     for kind in ("figures", "outputs", "figures/stale_pre_looks")]

# The figures that could go into the paper. The first group is in the paper
# now (research/papers/cvpr27_machine-perception-public-spaces/figures, 2026-10-01).
# The second group holds the headline figure of each version 3 analysis.
# A name without a suffix matches the .png, the .pdf, and the .json.
IN_PAPER = [
    "corpus_cubemap", "cue_hierarchy_agreement", "cue_words_facility_ladder",
    "trace_channels_inference", "trace_channels_abstention",
    "ic_perspectives_gemma_by_qwen27_v3_dimension_extractor_agreement",
    "ic_perspectives_gemma_by_qwen27_v3_inference_extractor_agreement",
    "ic_perspectives_gemma_by_qwen27_v3_perspective_extractor_agreement",
    "proxy_overview_v1.1.0",
    "street_photography_income_map_census_tract",
    "libraries_win_block", "schools_win_block", "subway_safety_win_block", "road_quality_win_bars",
    "libraries_win_block_column", "schools_win_block_column", "subway_safety_win_block_column",
    "libraries_distinctive", "road_quality_distinctive", "schools_distinctive",
    "subway_safety_distinctive",
]
V3_HEADLINES = [
    "trace_channels_abstention_arms",
    "ic_perspectives_qwen_by_gemma31_v3_dimension_extractor_agreement",
    "ic_perspectives_qwen_by_gemma31_v3_inference_extractor_agreement",
    "ic_perspectives_qwen_by_gemma31_v3_perspective_extractor_agreement",
    "ic_perspectives_gemma_by_qwen27_v3_perspective_ends",
    "ic_perspectives_qwen_by_gemma31_v3_perspective_ends",
    "ic_perspectives_gemma_by_qwen27_v3_dimension_ends",
    "ic_perspectives_qwen_by_gemma31_v3_dimension_ends",
    "ic_perspectives_gemma_by_qwen27_v3_inference_ends",
    "ic_perspectives_qwen_by_gemma31_v3_inference_ends",
    "ic_grounding_gemma_by_qwen27_v3_nesting",
    "ic_grounding_qwen_by_gemma31_v3_nesting",
    "ic_differentiation_arms_share_pseudo",
    "ic_differentiation_arms_revised_given_recon",
    "ic_differentiation_gemma_by_qwen27_v3_dismissal_about",
    "ic_differentiation_qwen_by_gemma31_v3_dismissal_about",
    "aesthetic_consistency_gemma_by_qwen27_v3_partner_effect",
    "aesthetic_consistency_qwen_by_gemma31_v3_partner_effect",
    "aesthetic_consistency_gemma_by_qwen27_v3_valence_grid",
    "aesthetic_consistency_qwen_by_gemma31_v3_valence_grid",
    "reasoning_vs_proxy_gemma_by_qwen27_v3_contrasts",
    "reasoning_vs_proxy_qwen_by_gemma31_v3_contrasts",
    "judge_vocab_dimension", "judge_vocab_inference", "judge_vocab_perspective",
]
PAPER_CANDIDATES = IN_PAPER + V3_HEADLINES

_VERSION = re.compile(r"^(?P<base>.*?)_v(?P<ver>\d+(?:\.\d+)*)$")


def _ver_key(ver: str) -> Tuple[int, ...]:
    parts = [int(p) for p in ver.split(".")]
    return tuple(parts + [0] * (3 - len(parts)))


def pinned_names() -> Set[str]:
    """Every file name, or file-name tail, that a `.py` file names literally.

    A name built by an f-string (`f"{case}_proxy_v1.1.0.parquet"`) leaves its
    tail (`_proxy_v1.1.0.parquet`) in the source. `is_pinned` matches tails.
    """
    names: Set[str] = set()
    pat = re.compile(r"[A-Za-z0-9_\-.]+\.(?:parquet|csv|tex|png|pdf|json|html)")
    for top in ("notebooks", "scripts", "dagspaces"):
        for path in (ROOT / top).rglob("*.py"):
            if "__pycache__" in path.parts or ".venv" in str(path):
                continue
            try:
                names.update(pat.findall(path.read_text(errors="ignore")))
            except OSError:
                continue
    return names


def is_pinned(path: Path, pinned: Set[str]) -> bool:
    return any(path.name.endswith(p) for p in pinned)


def paper_basenames() -> Set[str]:
    out: Set[str] = set()
    for sub in ("figures", "tables"):
        base = PAPER / sub
        if base.exists():
            out.update(p.name for p in base.rglob("*") if p.is_file())
    return out


def files_of(folders: Sequence[Path]) -> List[Path]:
    out = []
    for d in folders:
        if d.is_dir():
            out.extend(sorted(p for p in d.iterdir() if p.is_file()))
    return out


class Plan:
    def __init__(self) -> None:
        self.archive: Dict[Path, str] = {}       # path -> reason
        self.copies: List[Tuple[Path, Path]] = []
        self.renames: List[Tuple[Path, Path]] = []

    def to_archive(self, path: Path, reason: str) -> None:
        self.archive.setdefault(path, reason)


def plan_imports(plan: Plan, imports: Sequence[Path]) -> Set[Path]:
    """Copy the import folders into master/figures; archive what they replace."""
    landed: Set[Path] = set()
    for src_dir in imports:
        for src in sorted(p for p in src_dir.iterdir() if p.is_file()):
            dst = MASTER_FIGURES / src.name
            if dst.exists():
                plan.to_archive(dst, f"replaced by the import from {src_dir}")
            labelled = MASTER_FIGURES / (PREFIX + src.name)
            if labelled.exists():
                plan.to_archive(labelled, f"replaced by the import from {src_dir}")
            plan.copies.append((src, dst))
            landed.add(dst)
    return landed


def plan_stale(plan: Plan, pinned: Set[str], keep: Set[str]) -> None:
    """Archive stale-marked files and superseded versions."""
    for path in files_of(CURATED):
        stem = path.stem[len(PREFIX):] if path.stem.startswith(PREFIX) else path.stem
        # The pin test does not apply here: code writes these names, but no
        # code reads a stale folder or a `_stale` file (checked 2026-10-01).
        if stem in keep:
            continue
        rel = str(path.relative_to(CVPR))
        if "stale" in rel.lower():
            plan.to_archive(path, "marked stale (a stale folder or a _stale name)")
    groups: Dict[Tuple[Path, str, str], List[Tuple[str, Path]]] = defaultdict(list)
    for path in files_of(CURATED):
        m = _VERSION.match(path.stem)
        if m:
            groups[(path.parent, m.group("base"), path.suffix)].append((m.group("ver"), path))
    for (_, base, ext), items in groups.items():
        if len(items) < 2:
            continue
        # Equal versions ("v1" and "v1.0.0") sort by the time of the write.
        items.sort(key=lambda t: (_ver_key(t[0]), t[1].stat().st_mtime))
        newest = items[-1][0]
        for ver, path in items[:-1]:
            if is_pinned(path, pinned) or f"{base}_v{ver}" in keep:
                continue
            plan.to_archive(path, f"superseded by {base}_v{newest}{ext}")


def _candidate_files(name: str, extra: Sequence[Path] = ()) -> List[Path]:
    """The .png/.pdf/.svg/.json files of 1 candidate name, without the prefix."""
    out: List[Path] = []
    for path in files_of(CURATED) + list(extra):
        if path.stem == name and (path.suffix in FIGURE_EXT or path.suffix == ".json") and path not in out:
            out.append(path)
    return out


def plan_labels(plan: Plan, landed: Set[Path]) -> None:
    """Give each paper candidate the prefix; lift it out of a stale folder."""
    for name in PAPER_CANDIDATES:
        for path in _candidate_files(name, sorted(landed)):
            # A landed path is in the archive plan for its OLD content; the
            # import puts the new file there before the rename.
            if path in plan.archive and path not in landed:
                continue
            parent = path.parent.parent if path.parent.name.startswith("stale") else path.parent
            dst = parent / (PREFIX + path.name)
            if dst.exists() and dst not in plan.archive:
                plan.to_archive(dst, f"replaced by the newer {path.name}")
            plan.renames.append((path, dst))


def execute(plan: Plan, stamp: str) -> Optional[Path]:
    archive_path = None
    if plan.archive:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archive_path = ARCHIVE_DIR / f"notebook_outputs_{stamp}.tar.gz"
        manifest = ARCHIVE_DIR / f"notebook_outputs_{stamp}_manifest.csv"
        with tarfile.open(archive_path, "w:gz") as tar:
            for path in sorted(plan.archive):
                tar.add(path, arcname=str(path.relative_to(ROOT)))
        with open(manifest, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "reason", "bytes"])
            for path, reason in sorted(plan.archive.items()):
                w.writerow([str(path.relative_to(ROOT)), reason, path.stat().st_size])
        # Check the archive before anything leaves the tree.
        with tarfile.open(archive_path, "r:gz") as tar:
            held = set(tar.getnames())
        missing = [p for p in plan.archive if str(p.relative_to(ROOT)) not in held]
        if missing:
            raise SystemExit(f"the archive misses {len(missing)} files; nothing was removed")
        for path in plan.archive:
            path.unlink()
    for src, dst in plan.copies:
        shutil.copy2(src, dst)
    for src, dst in plan.renames:
        if not src.exists():
            continue
        src.replace(dst)
        if dst.suffix == ".json":
            try:
                meta = json.loads(dst.read_text())
            except ValueError:
                continue
            prov = meta.get("provenance")
            if isinstance(prov, dict) and "file" in prov:
                prov["file"] = str(Path(prov["file"]).with_name(PREFIX + Path(prov["file"]).name))
                dst.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    # Remove the folders the curation left empty.
    for d in CURATED:
        if d.is_dir() and d.name.startswith("stale") and not any(d.iterdir()):
            d.rmdir()
    return archive_path


def write_index() -> Path:
    in_paper = paper_basenames()
    rows = []
    for path in files_of(CURATED):
        if path.suffix not in FIGURE_EXT:
            continue
        stem = path.stem[len(PREFIX):] if path.stem.startswith(PREFIX) else path.stem
        rows.append({
            "path": str(path.relative_to(CVPR)),
            "folder": str(path.parent.relative_to(CVPR)),
            "tag": stem.split("_")[0],
            "paper_candidate": path.stem.startswith(PREFIX),
            "in_paper_now": (stem + path.suffix) in in_paper,
            "has_sidecar": path.with_suffix(".json").exists(),
        })
    out = CVPR / "figure_index.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["path"])
        w.writeheader()
        w.writerows(rows)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="do it; without this flag, print the plan")
    ap.add_argument("--import", dest="imports", action="append", default=[], type=Path,
                    help="a folder whose files go into research/notebooks/cvpr/master/figures (repeatable)")
    args = ap.parse_args()

    plan = Plan()
    landed = plan_imports(plan, args.imports)
    keep = set(PAPER_CANDIDATES)
    plan_stale(plan, pinned_names(), keep)
    plan_labels(plan, landed)

    by_reason: Dict[str, int] = defaultdict(int)
    for reason in plan.archive.values():
        by_reason[reason.split(" from ")[0].split(" by ")[0]] += 1
    print(f"[curate] import: {len(plan.copies)} files into {MASTER_FIGURES.relative_to(ROOT)}")
    print(f"[curate] archive: {len(plan.archive)} files")
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"           {n:4d}  {reason}")
    print(f"[curate] label: {len(plan.renames)} files get the prefix {PREFIX!r}")
    for src, dst in plan.renames:
        print(f"           {src.relative_to(CVPR)} -> {dst.relative_to(CVPR)}")
    if not args.apply:
        for path, reason in sorted(plan.archive.items()):
            print(f"  archive  {path.relative_to(CVPR)}  ({reason})")
        print("[curate] dry run: nothing changed. Add --apply.")
        return 0
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = execute(plan, stamp)
    index = write_index()
    if archive_path:
        print(f"[curate] archive written: {archive_path.relative_to(ROOT)}")
    print(f"[curate] index written: {index.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
