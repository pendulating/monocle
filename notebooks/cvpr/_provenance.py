"""Run discovery and provenance for the CVPR validation notebooks.

Every notebook in `notebooks/cvpr/` is load-bearing for the paper. Thus each one
must state exactly which runs it read, and a reader must be able to get those
runs again. This module does that job.

**Do not hard-code run directories in a notebook, and do not let a notebook
choose a run.** `discover_runs` reads the canonical registry
(`canonical_data/manifest.json`, see `_canonical.py`), which names exactly 1 run
for each case and model. The W&B chain below stays available with
`source="wandb"`: use it to FIND a run that you then register, never to build a
paper figure.

The provenance chain
--------------------
A W&B stage run does not record the case name or the output path. This module
resolves them:

1. `wandb-metadata.json` → `args[0]` gives the submitit job directory.
2. Its parent directory is the stage directory.
3. `<stage_dir>/.hydra/overrides.yaml` gives the pipeline and the model.
4. `<stage_dir>/outputs/pairwise/` holds the results parquet and `pairs.parquet`.

`wandb-metadata.json` also gives the git commit and the Python executable, so
each row of the provenance table names the code and the environment that made it.

Data facts
----------
- The results parquet holds the labels. It has NO unit or geographic columns.
- `pairs.parquet` holds `unit_uid_a/b`, `latitude_a/b`, and `longitude_a/b`.
  Join the two on `pair_id`.
- The pairs latitude is the camera position, not the unit position. Use the
  FacDB `facilities.parquet` for the true unit position.

Run the notebooks from `.venv-mllmsci-vllm025cu129`. It has marimo,
geopandas, and wordcloud.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

__version__ = "1.1.0"

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "urbanekg")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "URBANPAIRVQA")
REPO_ROOT = Path(__file__).resolve().parents[2]

# The canonical rater set. Match the string exactly. A substring test is wrong
# here: it accepts `gemma-4-12b/instruct_thinking`, which is a different config
# with its own sampling parameters, so its labels are not comparable.
CANONICAL_MODELS = ("qwen3.5-9b/instruct", "gemma-4-12b/instruct")

# The date of the CURRENT battery. On 2026-08-11 the battery became 7 cases, a
# minimal prompt with no persona, abstention always on, and greedy decoding. On
# 2026-08-14 every prompt moved to "looks like" wording, which asks about the
# IMAGE and not about the place, and the gemma runs gained
# `image_layout=interleaved_labels`. An earlier run answers a different
# question, thus the notebooks must not mix the two.
#
# Note: since the canonical registry arrived, this date no longer selects the
# runs. `canonical_data/manifest.json` does. The date stays as a second guard
# for the paths that still read W&B.
CONSOLIDATION_DATE = "2026-08-14"

# The sweep configs that produce a canonical run. This list now guards the W&B
# path only: the registry decides what is canonical, and it holds `looks_proxy_*`
# runs alone.
#
# `looks_proxy_*` came on 2026-08-14 with the "looks like" prompts. A prompt now
# asks what the image SHOWS, not what is true of the place. The 2 older names
# stay for a reader of a 2026-08-11..14 run; CONSOLIDATION_DATE already keeps
# those runs out of a figure.
CANONICAL_SWEEPS = (
    "looks_proxy_qwen9b",
    "looks_proxy_gemma12b",
    "canonical_qwen9b",
    "canonical_gemma12b",
)

# The 5-point ordinal scale. `NotSure` is absent on purpose: an abstention is
# not a zero, so it must not enter the mean.
ORDINAL_SCORE = {"MuchLess": -2, "Less": -1, "Same": 0, "More": 1, "MuchMore": 2}
NOT_SURE = "NotSure"


@dataclass
class RunRecord:
    """One canonical run, with the provenance that the paper must cite."""

    wandb_id: str
    wandb_name: str
    wandb_url: str
    state: str
    created_at: str
    git_commit: Optional[str]
    executable: Optional[str]
    host: Optional[str]
    stage_dir: Optional[str]
    pipeline: Optional[str]
    model: Optional[str]
    results_path: Optional[str]
    pairs_path: Optional[str]
    sweep: Optional[str] = None
    total_rows: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """True when the notebook can read both parquets from this run."""
        return bool(
            self.results_path
            and self.pairs_path
            and os.path.exists(self.results_path)
            and os.path.exists(self.pairs_path)
        )

    @property
    def is_canonical(self) -> bool:
        """True when this run belongs to the consolidated battery.

        3 conditions must hold together. The model must match the canonical set
        exactly. The run must start on or after the consolidation date. The run
        must come from a canonical sweep.
        """
        if (self.model or "") not in CANONICAL_MODELS:
            return False
        if (self.created_at or "") < CONSOLIDATION_DATE:
            return False
        return (self.sweep or "") in CANONICAL_SWEEPS


def _parse_overrides(path: Path) -> Dict[str, str]:
    """Read a Hydra `overrides.yaml` into a flat dict."""
    out: Dict[str, str] = {}
    try:
        import yaml

        items = yaml.safe_load(path.read_text()) or []
    except Exception:
        return out
    for raw in items:
        if isinstance(raw, str) and "=" in raw:
            k, _, v = raw.partition("=")
            out[k.strip().lstrip("+~")] = v.strip()
    return out


def _resolve_stage_dir(metadata: Dict[str, Any]) -> Optional[Path]:
    """Get the stage directory from the submitit argument in the W&B metadata.

    `args[0]` points at `<stage_dir>/.slurm_jobs/<node>`. The stage directory is
    2 levels above it.
    """
    args = metadata.get("args") or []
    if not args:
        return None
    p = Path(str(args[0]))
    if ".slurm_jobs" in p.parts:
        i = p.parts.index(".slurm_jobs")
        return Path(*p.parts[:i])
    return None


def _find_parquets(stage_dir: Path) -> Dict[str, Optional[str]]:
    """Find the results parquet and the pairs manifest in a stage directory."""
    out_dir = stage_dir / "outputs" / "pairwise"
    if not out_dir.is_dir():
        return {"results_path": None, "pairs_path": None}
    pairs = out_dir / "pairs.parquet"
    results = [p for p in out_dir.glob("*.parquet") if p.name != "pairs.parquet"]
    # A run writes 1 results parquet. If a re-run left more, take the newest.
    newest = max(results, key=lambda p: p.stat().st_mtime) if results else None
    return {
        "results_path": str(newest) if newest else None,
        "pairs_path": str(pairs) if pairs.exists() else None,
    }


SCAN_CACHE = REPO_ROOT / "notebooks" / "cvpr" / ".trace_cache" / "run_scan.json"

# How long a cached scan stays good. A sweep that lands after the scan needs a
# refresh, thus keep this short enough that a forgotten cache cannot hide a run
# for a whole day.
SCAN_TTL_SECONDS = 6 * 3600


def scan_runs(
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    limit: int = 400,
    cache_dir: Optional[str] = None,
    refresh: bool = False,
) -> List[RunRecord]:
    """Read every pairwise run from W&B and resolve its stage directory.

    This does the network work for `discover_runs`. It applies **no** case
    filter, **no** canonical filter, and **no** state filter, so 1 scan serves
    every notebook. `discover_runs` applies those 3 filters to the result.

    Warning: keep the state filter out of this function. Put it in, and a
    notebook that asks for `only_finished=False` needs its own 20-minute scan,
    because the cache key no longer matches the one the other notebooks built.

    Warning: this reads the network and downloads 1 metadata file for each run,
    thus it costs about 20 minutes. The result goes to `SCAN_CACHE`. Every
    later call inside `SCAN_TTL_SECONDS` reads that file instead. Pass
    `refresh=True` to make it read the network again, which you must do after a
    new sweep lands.
    """
    key = f"{entity}/{project}|limit={limit}"
    if not refresh and SCAN_CACHE.exists():
        try:
            blob = json.loads(SCAN_CACHE.read_text())
            fresh = (time.time() - blob.get("stamp", 0)) < SCAN_TTL_SECONDS
            if blob.get("key") == key and fresh:
                return [RunRecord(**d) for d in blob["runs"]]
        except Exception:
            pass

    import wandb

    api = wandb.Api()
    cache = Path(cache_dir or (REPO_ROOT / "notebooks" / "cvpr" / ".wandb_cache"))
    cache.mkdir(parents=True, exist_ok=True)

    filters: Dict[str, Any] = {"jobType": "pairwise_vqa"}

    records: List[RunRecord] = []
    for run in api.runs(f"{entity}/{project}", filters=filters,
                        order="-created_at", per_page=50):
        if len(records) >= limit:
            break
        meta: Dict[str, Any] = {}
        try:
            f = run.file("wandb-metadata.json").download(
                replace=True, root=str(cache / run.id)
            )
            meta = json.load(open(f.name))
        except Exception:
            pass

        stage_dir = _resolve_stage_dir(meta)
        ov = _parse_overrides(stage_dir / ".hydra" / "overrides.yaml") if stage_dir else {}
        pipeline = ov.get("pipeline")
        if not pipeline:
            continue

        paths = _find_parquets(stage_dir) if stage_dir else {"results_path": None, "pairs_path": None}
        rec = RunRecord(
            wandb_id=run.id,
            wandb_name=run.name,
            wandb_url=run.url,
            state=run.state,
            created_at=str(run.created_at),
            git_commit=(meta.get("git") or {}).get("commit"),
            executable=meta.get("executable"),
            host=meta.get("host"),
            stage_dir=str(stage_dir) if stage_dir else None,
            pipeline=pipeline,
            model=ov.get("model"),
            sweep=ov.get("sweep"),
            total_rows=run.summary.get("pairwise/results/total_rows"),
            **paths,
        )
        if not rec.is_usable:
            rec.notes.append("parquet missing on disk")
        records.append(rec)

    SCAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        SCAN_CACHE.write_text(json.dumps(
            {"key": key, "stamp": time.time(),
             "runs": [asdict(r) for r in records]}, indent=1,
        ))
    except Exception:
        pass
    return records


# Where `discover_runs` gets its runs.
#   canonical: read `canonical_data/manifest.json`. This is the default and the
#              only source a paper figure may use.
#   wandb:     query the network, the old behaviour. Use it to FIND a run that
#              you then register, never to build a figure.
RUN_SOURCE = os.environ.get("CVPR_RUN_SOURCE", "canonical")


def _records_from_registry(case: str, only_canonical: bool) -> List[RunRecord]:
    """Build RunRecords from the canonical registry.

    The record points at the SYMLINK under `canonical_data/`, not at the run
    directory. Thus a provenance table names the registry, and a reader can see
    at once whether a figure used a canonical run.
    """
    import _canonical as C

    C.verify_or_raise()
    out: List[RunRecord] = []
    for r in C.runs(kind="proxy" if only_canonical else None, case=case or None):
        out.append(RunRecord(
            wandb_id=r.run_id,
            wandb_name=f"{r.kind}:{r.case}:{r.model}",
            wandb_url="",
            state="finished",
            created_at=r.created_at,
            git_commit=None,
            executable=None,
            host=None,
            stage_dir=r.stage_dir,
            pipeline=r.pipeline,
            model=r.model_config,
            results_path=r.results_link,
            pairs_path=r.pairs_link,
            sweep=r.sweep,
            total_rows=r.rows,
            notes=list(r.problems),
        ))
    out.sort(key=lambda r: r.created_at or "", reverse=True)
    return out


def discover_runs(
    case: str,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    only_finished: bool = True,
    only_canonical: bool = True,
    limit: int = 400,
    cache_dir: Optional[str] = None,
    refresh: bool = False,
    source: Optional[str] = None,
) -> List[RunRecord]:
    """Find every run of one case, newest first.

    Args:
        case: A substring of the pipeline name, for example `schools`. An empty
            string keeps every case.
        only_finished: Keep only runs that W&B marks `finished`. Set it to False
            to see runs that are still in progress.
        only_canonical: Keep only the canonical rater set.
        cache_dir: Where to put the downloaded `wandb-metadata.json` files.
        refresh: Read the network again instead of the cached scan.

    Note: the network work happens in `scan_runs`, which caches its result.
    Thus the 7 case notebooks share 1 scan, and only the first one waits.
    """
    if (source or RUN_SOURCE) == "canonical":
        # The registry already holds 1 run for each cell, so the filters below
        # have nothing left to drop. Return it as it is.
        return _records_from_registry(case, only_canonical)

    records = scan_runs(
        entity=entity, project=project,
        limit=limit, cache_dir=cache_dir, refresh=refresh,
    )
    out: List[RunRecord] = []
    for rec in records:
        if only_finished and rec.state != "finished":
            continue
        if case and case.lower() not in (rec.pipeline or "").lower():
            continue
        if only_canonical and not rec.is_canonical:
            continue
        out.append(rec)
        if len(out) >= limit:
            break
    return out


def provenance_table(records: List[RunRecord]) -> pd.DataFrame:
    """Build the provenance table that the paper cites."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(r) for r in records])
    df["notes"] = df["notes"].apply(lambda x: "; ".join(x) if x else "")
    df["usable"] = [r.is_usable for r in records]
    cols = ["model", "pipeline", "sweep", "state", "created_at", "total_rows",
            "usable", "wandb_id", "git_commit", "stage_dir", "wandb_url", "notes"]
    return df[[c for c in cols if c in df.columns]]


def load_run(record: RunRecord, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read one run and join the labels to the unit columns.

    The results parquet has no unit or geographic columns, so this joins it to
    `pairs.parquet` on `pair_id`.
    """
    if not record.is_usable:
        raise FileNotFoundError(f"run {record.wandb_id} has no readable parquet")
    res = pd.read_parquet(
        record.results_path,
        columns=columns or ["pair_id", "relative_label", "relative_score"],
    )
    keep = ["pair_id", "unit_uid_a", "unit_uid_b", "unit_name_a", "unit_name_b"]
    pairs = pd.read_parquet(record.pairs_path, columns=keep)
    df = res.merge(pairs, on="pair_id", how="left", validate="one_to_one")
    df["model"] = record.model
    df["wandb_id"] = record.wandb_id
    return df


def load_run_images(record: RunRecord) -> pd.DataFrame:
    """Read an IMAGE-mode run and attach the position of each image.

    Road quality and street photography pair random street-level shots, so
    their `pairs.parquet` holds NO `unit_uid` column. `load_run` therefore
    fails on them. This reads `sample_id` and the coordinates instead.
    """
    if not record.is_usable:
        raise FileNotFoundError(f"run {record.wandb_id} has no readable parquet")
    res = pd.read_parquet(
        record.results_path, columns=["pair_id", "relative_label", "relative_score"]
    )
    keep = ["pair_id", "sample_id_a", "sample_id_b",
            "latitude_a", "latitude_b", "longitude_a", "longitude_b"]
    pairs = pd.read_parquet(record.pairs_path, columns=keep)
    df = res.merge(pairs, on="pair_id", how="left", validate="one_to_one")
    df["model"] = record.model
    df["wandb_id"] = record.wandb_id
    return df


def score_images(df: pd.DataFrame, min_comparisons: int = 1) -> pd.DataFrame:
    """Give each image a mean score, and keep its position.

    Warning: an image carries far fewer comparisons than a curated unit. A
    library sat in about 693 comparisons, because 236 libraries shared 110,000
    pairs. An image-mode run spreads the same 110,000 pairs over a 500,000-image
    manifest, so most images appear once or twice.

    Thus a single image score is noisy, and `min_comparisons` defaults to 1. The
    geography step carries the precision here: a polygon holds many images, even
    though each image is weak. Never read 1 image score on its own.
    """
    lab = df["relative_label"].astype(str)
    score = lab.map(ORDINAL_SCORE)
    ok = score.notna()

    a = pd.DataFrame({
        "sample_id": df.loc[ok, "sample_id_a"],
        "latitude": df.loc[ok, "latitude_a"],
        "longitude": df.loc[ok, "longitude_a"],
        "score": score[ok],
    })
    b = pd.DataFrame({
        "sample_id": df.loc[ok, "sample_id_b"],
        "latitude": df.loc[ok, "latitude_b"],
        "longitude": df.loc[ok, "longitude_b"],
        "score": -score[ok],
    })
    long = pd.concat([a, b], ignore_index=True).dropna(subset=["sample_id"])

    g = long.groupby("sample_id").agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        mean_score=("score", "mean"),
        n_comparisons=("score", "size"),
    ).reset_index()

    absent = pd.concat([
        pd.DataFrame({"sample_id": df["sample_id_a"], "abstain": lab.eq(NOT_SURE)}),
        pd.DataFrame({"sample_id": df["sample_id_b"], "abstain": lab.eq(NOT_SURE)}),
    ], ignore_index=True).dropna(subset=["sample_id"])
    rate = absent.groupby("sample_id")["abstain"].mean().rename("abstention_rate")
    g = g.merge(rate, on="sample_id", how="left")

    return g[g.n_comparisons >= min_comparisons].reset_index(drop=True)


def score_units(df: pd.DataFrame, min_comparisons: int = 5) -> pd.DataFrame:
    """Give each unit a mean score from its pairwise labels.

    `relative_label` always states side A against side B, so the orchestrator has
    already undone the presentation order. For a unit on side B, the sign flips.

    An abstention (`NotSure`) becomes NaN and drops out. It is not a zero.

    Args:
        min_comparisons: Drop a unit with fewer comparisons than this. A unit
            with 1 or 2 comparisons gives a noisy mean.
    """
    lab = df["relative_label"].astype(str)
    score = lab.map(ORDINAL_SCORE)
    ok = score.notna()

    a = pd.DataFrame({
        "unit_uid": df.loc[ok, "unit_uid_a"],
        "unit_name": df.loc[ok, "unit_name_a"],
        "score": score[ok],
    })
    b = pd.DataFrame({
        "unit_uid": df.loc[ok, "unit_uid_b"],
        "unit_name": df.loc[ok, "unit_name_b"],
        "score": -score[ok],
    })
    long = pd.concat([a, b], ignore_index=True).dropna(subset=["unit_uid"])

    g = long.groupby("unit_uid").agg(
        unit_name=("unit_name", "first"),
        mean_score=("score", "mean"),
        sd_score=("score", "std"),
        n_comparisons=("score", "size"),
    ).reset_index()

    # The abstention rate belongs to the unit too, so count it before the drop.
    absent = pd.concat([
        pd.DataFrame({"unit_uid": df["unit_uid_a"], "abstain": lab.eq(NOT_SURE)}),
        pd.DataFrame({"unit_uid": df["unit_uid_b"], "abstain": lab.eq(NOT_SURE)}),
    ], ignore_index=True).dropna(subset=["unit_uid"])
    rate = absent.groupby("unit_uid")["abstain"].mean().rename("abstention_rate")
    g = g.merge(rate, on="unit_uid", how="left")

    return g[g.n_comparisons >= min_comparisons].reset_index(drop=True)
