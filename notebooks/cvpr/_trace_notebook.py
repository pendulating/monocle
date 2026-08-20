"""The body of a per-prompt trace notebook.

Every `<case>_traces.py` notebook is the same report over a different case.
This module holds that report, so a notebook is a title, a case name, and a
call. A change here reaches all of them at once.

The pattern follows the validation-by-proxy notebooks: 1 notebook for each
prompt, each one load-bearing for the paper, each one stating its provenance.

Why a case notebook still reads every case
------------------------------------------
The `distinctive` score asks what THIS prompt says that the others do not.
Thus it needs the other prompts as a background. A notebook therefore counts
every case and draws only its own. The background is named in the report, so a
reader knows what the comparison is against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

import _provenance as P
import _traces as T

__version__ = "1.0.0"

# The sweep directories that hold thinking runs W&B cannot resolve on its own.
# `scan_sweep_dir` reads these from disk.
#
# Warning: since 2026-08-17 the default source is the canonical registry, and
# the registry ignores this list. It applies to `source="wandb"` alone. Do not
# add a sweep here to make it reach a figure — register it instead:
#   python scripts/register_canonical_runs.py register --stage-root '<glob>'
EXTRA_SWEEP_DIRS = (
    "multirun/2026-08-13_URBANPAIRVQA/01-37-12",
)


def figures_dir(notebook_file: str | Path) -> Path:
    """Give the figures directory of the calling notebook.

    Every artifact of a prompt lives inside that prompt's folder:
    `notebooks/cvpr/<prompt>/figures/` for images, and `outputs/` beside it for
    tables. Thus this resolves against the NOTEBOOK, not against this module —
    this module sits in the parent and serves every prompt, so its own
    `__file__` would send each notebook's figures to one shared directory.
    """
    return Path(notebook_file).resolve().parent / "figures"


def discover(
    extra_sweep_dirs: Sequence[str] = EXTRA_SWEEP_DIRS,
    min_date: Optional[str] = P.CONSOLIDATION_DATE,
    refresh: bool = False,
) -> List[T.TraceRun]:
    """Find every usable thinking run.

    The default source is the canonical registry, which holds 1 thinking run
    for each case and model (`_canonical.py`). Thus `extra_sweep_dirs`,
    `min_date`, and `refresh` do nothing unless you ask for the W&B source.
    They stay in the signature for that path.
    """
    runs = T.discover_trace_runs(
        only_finished=True,
        require_trace=True,
        extra_sweep_dirs=list(extra_sweep_dirs),
        min_date=min_date,
        refresh=refresh,
    )
    return [r for r in runs if r.case in T.BATTERY_CASES and r.is_readable]


def groups_for_case(counts: Dict[str, T.Counter], case: str) -> List[str]:
    """Name the groups that belong to 1 case.

    A case is usually 1 group. It becomes 2 when the case asked 2 questions,
    for example the schools case across the prompt change of 2026-08-13.
    """
    return sorted(k for k in counts if k == case or k.startswith(f"{case} | "))


def word_table(
    counts: Dict[str, T.Counter],
    groups: Sequence[str],
    mode: str = "distinctive",
    max_words: int = 120,
) -> pd.DataFrame:
    """Give the numbers behind the cloud, so a value can be quoted."""
    frames = []
    for g in groups:
        w = T.cloud_weights(counts, g, mode=mode, max_words=max_words)
        frames.append(pd.DataFrame({
            "group": g,
            "word": list(w),
            "weight": list(w.values()),
            "count": [counts[g].get(k, 0) for k in w],
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def export(
    case: str,
    counts: Dict[str, T.Counter],
    groups: Sequence[str],
    runs: Sequence[T.TraceRun],
    out_dir: Path,
    mode: str = "distinctive",
    max_words: int = 120,
) -> List[str]:
    """Write 1 PNG for each group, plus the word table and the provenance.

    Args:
        out_dir: Where to write. Pass `figures_dir(__file__)` from a notebook,
            so the files land in that prompt's folder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for g in groups:
        w = T.cloud_weights(counts, g, mode=mode, max_words=max_words)
        if not w:
            continue
        # safe_name, never the raw group: a group can carry the question, and 2
        # long questions can share a prefix.
        p = out_dir / f"{T.safe_name(g)}_{mode}.png"
        T.make_cloud(w, width=1600, height=900).to_file(str(p))
        written.append(p.name)

    wt = word_table(counts, groups, mode=mode, max_words=max_words)
    if not wt.empty:
        wt.to_csv(out_dir / f"{case}_words_{mode}.csv", index=False)
        written.append(f"{case}_words_{mode}.csv")

    case_runs = [r for r in runs if r.case == case]
    if case_runs:
        T.runs_table(case_runs).to_csv(out_dir / f"{case}_provenance.csv", index=False)
        written.append(f"{case}_provenance.csv")
    return written
