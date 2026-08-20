"""Turn the reasoning traces of a pairvqa run into typed, grounded extractions.

The stage reads a pairvqa results parquet, sends each `model_reasoning` text
through LangExtract, and writes 1 row for each extraction. The output joins back
to the source on `pair_id`.

Warning: only a THINKING run holds a trace. The canonical battery runs greedy
with `max_tokens=128`, thus its `model_reasoning` column is empty. See
`vlm-narratives-docs/reasoning-trace-analysis.md`.

Shards and resume
-----------------
The stage cuts the trace table into row ranges and writes 1 parquet for each
range under `chunks/`. A rerun skips a range whose chunk exists. Thus a job that
the walltime stops keeps everything it finished.

See `vlm-narratives-docs/langextract-trace-extraction.md`.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from dagspaces.common.langextract_backend import (
    ExtractionSpec,
    VLLMEngine,
    VLLMLanguageModel,
    annotated_to_rows,
    extract_documents,
    spec_from_config,
    validate_examples,
)

STAGE_NAME = "trace_extract"

# The default columns to carry from the source parquet into every output row.
# `presented_label` is the one that makes an extraction answerable: it lets a
# reader ask which cues go with which judgment.
DEFAULT_KEEP_COLUMNS = ("presented_label", "presented_score", "relative_label")

# A trace shorter than this is a stub, not a judgment. A cut-off generation and
# an empty column both land here.
DEFAULT_MIN_CHARS = 200

DEFAULT_SHARD_ROWS = 2000


def case_of_parquet(name: str) -> str:
    """Read the case name out of a results file name.

    `subway_safety_mvp_20260813_013722.parquet` gives `subway_safety`. This
    mirrors `notebooks/cvpr/_traces.case_of_parquet`, so a stage output and a
    notebook agree on the name.
    """
    stem = os.path.basename(str(name))
    stem = re.sub(r"\.parquet$", "", stem)
    stem = re.sub(r"\.presplit$", "", stem)
    stem = re.sub(r"_\d{8}_\d{6}$", "", stem)
    stem = re.sub(r"^pairwise_", "", stem)
    stem = re.sub(r"_(mvp|ordinal|large)$", "", stem)
    return stem or "unknown"


def source_provenance(results_path: str) -> Dict[str, str]:
    """Name the run that wrote a results parquet.

    A results parquet sits at `<stage_dir>/outputs/pairwise/<file>.parquet`, and
    `<stage_dir>/.hydra/overrides.yaml` names the pipeline, the model, and the
    sweep. An older stage directory holds no Hydra record, and then this returns
    what it can.
    """
    out: Dict[str, str] = {"stage_dir": "", "pipeline": "", "judge_model": "", "sweep": ""}
    # Follow a symlink first. The canonical registry
    # (`notebooks/cvpr/canonical_data/`) links each run as
    # `<kind>/<case>__<model>/results.parquet`, and 3 directories above THAT
    # link is `notebooks/cvpr`, which holds no Hydra record. The real file
    # still sits at `<stage_dir>/outputs/pairwise/<name>.parquet`.
    real_path = os.path.realpath(str(results_path))
    stage_dir = os.path.dirname(os.path.dirname(os.path.dirname(real_path)))
    out["stage_dir"] = stage_dir
    overrides = os.path.join(stage_dir, ".hydra", "overrides.yaml")
    if not os.path.exists(overrides):
        return out
    try:
        import yaml

        entries = yaml.safe_load(open(overrides).read()) or []
    except Exception:
        return out
    for entry in entries:
        text = str(entry)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.lstrip("+~").strip()
        if key == "pipeline":
            out["pipeline"] = value.strip()
        elif key == "model":
            out["judge_model"] = value.strip()
        elif key in ("sweep", "+sweep"):
            out["sweep"] = value.strip()
    return out


def load_traces(
    cfg: Any, section: str = "trace_extract"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read the traces of the source run, and report what the filters dropped.

    `section` names the config group that holds the source. `ic_extract` reads
    the same settings under its own name.
    """
    tcfg = getattr(cfg, section, {})
    results_path = str(getattr(tcfg, "results_path", "") or "")
    if not results_path:
        raise ValueError(f"{section}.results_path is not set")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"no results parquet at {results_path}")

    reasoning_col = str(getattr(tcfg, "reasoning_column", "model_reasoning"))
    id_col = str(getattr(tcfg, "id_column", "pair_id"))
    keep = list(getattr(tcfg, "keep_columns", DEFAULT_KEEP_COLUMNS) or [])
    min_chars = int(getattr(tcfg, "min_chars", DEFAULT_MIN_CHARS) or 0)

    df = pd.read_parquet(results_path)
    if reasoning_col not in df.columns:
        raise ValueError(
            f"{results_path} has no column {reasoning_col!r}. "
            "A label-only run keeps no trace."
        )
    if id_col not in df.columns:
        raise ValueError(f"{results_path} has no column {id_col!r}")

    columns = [id_col, reasoning_col] + [c for c in keep if c in df.columns]
    df = df[columns].copy()
    df[reasoning_col] = df[reasoning_col].fillna("").astype(str)

    total = len(df)
    lengths = df[reasoning_col].str.len()
    df = df[lengths >= min_chars].reset_index(drop=True)

    stats = {
        "source_rows": int(total),
        "traces_kept": int(len(df)),
        "traces_dropped_short": int(total - len(df)),
        "trace_chars_mean": float(lengths.mean()) if total else 0.0,
        "trace_chars_max": int(lengths.max()) if total else 0,
    }
    if len(df) == 0:
        raise ValueError(
            f"{results_path} holds no trace of {min_chars} characters or more. "
            "This is a label-only run."
        )
    return df, stats


def _shard(df: pd.DataFrame, cfg: Any, section: str = "trace_extract") -> pd.DataFrame:
    """Keep the part of the table that belongs to this job.

    A case of 11,000 traces costs about 9 hours on 1 GPU, and the work is
    decode-bound. It splits perfectly: no trace needs another trace. Thus a
    Hydra multirun over `shard_index` runs N jobs on N GPUs, and the wall clock
    falls by N.

    ```bash
    python -m dagspaces.urbanpairvqa.cli -m pipeline=extract_traces \\
        trace_extract.results_path=... \\
        trace_extract.shard_count=8 \\
        trace_extract.shard_index=0,1,2,3,4,5,6,7
    ```

    The stride keeps each shard mixed. A contiguous cut would put the long
    traces of 1 region in 1 job, and that job would then run much longer than
    the rest.
    """
    tcfg = getattr(cfg, section, {})
    count = int(getattr(tcfg, "shard_count", 1) or 1)
    index = int(getattr(tcfg, "shard_index", 0) or 0)
    if count <= 1:
        return df
    if not 0 <= index < count:
        raise ValueError(f"shard_index must be in [0, {count}), got {index}")
    out = df.iloc[index::count].reset_index(drop=True)
    print(
        f"[{section}] shard {index + 1} of {count}: {len(out)} traces of {len(df)}",
        flush=True,
    )
    return out


def _sample(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Cut the table down for a smoke run.

    The sample is random, not the head. The head of a pairs table is 1 corner of
    the city. The seed keeps a rerun on the same rows.
    """
    sample_n = getattr(getattr(cfg, "runtime", {}), "sample_n", None)
    if not sample_n or int(sample_n) >= len(df):
        return df
    seed = int(getattr(getattr(cfg, "pair_sampler", {}), "pair_seed", 777) or 777)
    return df.sample(n=int(sample_n), random_state=seed).reset_index(drop=True)


class _StubModel(VLLMLanguageModel):
    """Answer with no extraction. Use it to test the path without a GPU."""

    def __init__(self) -> None:
        super().__init__(
            lambda prompts: ['{"extractions": []}'] * len(prompts),
            stage_name=STAGE_NAME,
        )


def run_trace_extract_stage(
    cfg: Any, output_dir: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Extract from every trace of the source run.

    Args:
        cfg: The stage config. `cfg.extract` holds the schema, and
            `cfg.trace_extract` holds the source and the shard size.
        output_dir: Where to write the chunk parquets.

    Returns:
        (rows, metadata). `rows` is the long extraction table.
    """
    tcfg = getattr(cfg, "trace_extract", {})
    results_path = str(getattr(tcfg, "results_path", "") or "")
    reasoning_col = str(getattr(tcfg, "reasoning_column", "model_reasoning"))
    id_col = str(getattr(tcfg, "id_column", "pair_id"))
    shard_rows = int(getattr(tcfg, "shard_rows", DEFAULT_SHARD_ROWS) or DEFAULT_SHARD_ROWS)

    spec = spec_from_config(getattr(cfg, "extract"))

    # An example that cannot align teaches a shape the model cannot copy. Stop
    # before the engine loads, not after.
    issues = validate_examples(spec)
    if issues:
        raise ValueError(
            "the examples of "
            f"{spec.name}/{spec.schema_version} do not align:\n  "
            + "\n  ".join(issues)
        )

    df, stats = load_traces(cfg)
    # Sample first, then shard. The other order gives each shard its own sample
    # of a different size, and no 2 shards then cover the same set.
    df = _sample(df, cfg)
    df = _shard(df, cfg)
    # `traces_kept` counts what the filter left. `traces_extracted` counts what
    # this job read, which is smaller under `runtime.sample_n`. Every rate below
    # divides by the second one.
    stats["traces_extracted"] = int(len(df))

    # A trace longer than the buffer splits into 2 chunks, and the second chunk
    # cannot see image A. Keep the count visible rather than silent.
    over = int((df[reasoning_col].str.len() > spec.max_char_buffer).sum())
    stats["traces_over_buffer"] = over
    if over:
        print(
            f"[{STAGE_NAME}] {over} of {len(df)} traces are longer than "
            f"max_char_buffer={spec.max_char_buffer}, thus they split",
            flush=True,
        )

    provenance = source_provenance(results_path)
    case = str(getattr(tcfg, "case", "") or "") or case_of_parquet(results_path)
    judge_model = (
        str(getattr(tcfg, "judge_model", "") or "") or provenance.get("judge_model", "")
    )

    skip_inference = bool(getattr(getattr(cfg, "runtime", {}), "skip_inference", False))
    extractor_model = (
        "stub" if skip_inference else str(getattr(cfg.model, "model_source", "unknown"))
    )

    print(
        f"[{STAGE_NAME}] case={case} judge={judge_model} "
        f"traces={len(df)} schema={spec.name}/{spec.schema_version} "
        f"classes={spec.classes}",
        flush=True,
    )

    chunk_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    engine: Optional[VLLMEngine] = None
    frames: List[pd.DataFrame] = []
    t_start = time.time()
    try:
        model: VLLMLanguageModel
        if skip_inference:
            print(f"[{STAGE_NAME}] runtime.skip_inference — the stub answers", flush=True)
            model = _StubModel()
        else:
            engine = VLLMEngine(cfg, stage_name=STAGE_NAME)
            debug_answers = int(getattr(getattr(cfg, "extract"), "debug_answers", 0) or 0)
            if debug_answers:
                engine.debug_answers_path = os.path.join(output_dir, "debug_answers.jsonl")
                engine.debug_answers_left = debug_answers
                print(
                    f"[{STAGE_NAME}] the first {debug_answers} raw answers go to "
                    f"{engine.debug_answers_path}",
                    flush=True,
                )
            model = engine.as_language_model(spec)

        for start in range(0, len(df), shard_rows):
            end = min(start + shard_rows, len(df))
            chunk_path = os.path.join(chunk_dir, f"extractions_{start:07d}_{end:07d}.parquet")
            if os.path.exists(chunk_path):
                print(f"[{STAGE_NAME}] shard {start}-{end} exists, skipped", flush=True)
                frames.append(pd.read_parquet(chunk_path))
                continue

            shard = df.iloc[start:end]
            t0 = time.time()
            annotated = extract_documents(
                shard[reasoning_col].tolist(),
                shard[id_col].astype(str).tolist(),
                spec,
                model,
            )

            # Read the id, never the position. `lx.extract` gives no order
            # promise.
            by_id = {str(row[id_col]): row for _, row in shard.iterrows()}
            rows: List[Dict[str, Any]] = []
            for doc in annotated:
                source = by_id.get(str(doc.document_id))
                base: Dict[str, Any] = {
                    "case": case,
                    "judge_model": judge_model,
                    "sweep": provenance.get("sweep", ""),
                    "source_results_path": results_path,
                    id_col: doc.document_id,
                }
                if source is not None:
                    for col in shard.columns:
                        if col in (id_col, reasoning_col):
                            continue
                        base[col] = source[col]
                rows.extend(annotated_to_rows(doc, spec, extractor_model, base=base))

            chunk = pd.DataFrame(rows)
            chunk.to_parquet(chunk_path, index=False)
            frames.append(chunk)
            print(
                f"[{STAGE_NAME}] shard {start}-{end}: {len(chunk)} rows "
                f"in {time.time() - t0:.1f}s → {os.path.basename(chunk_path)}",
                flush=True,
            )
    finally:
        if engine is not None:
            engine.shutdown()

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if engine is not None:
        stats["answers_truncated"] = int(engine.truncated)
        stats["answers_repaired"] = int(engine.repaired)
    metadata = _metadata(out, stats, spec, case, judge_model, extractor_model)
    metadata["duration_s"] = round(time.time() - t_start, 1)
    metadata["shard_index"] = int(getattr(tcfg, "shard_index", 0) or 0)
    metadata["shard_count"] = int(getattr(tcfg, "shard_count", 1) or 1)
    metadata.update({f"source_{k}": v for k, v in provenance.items()})
    return out, metadata


def _metadata(
    out: pd.DataFrame,
    stats: Dict[str, Any],
    spec: ExtractionSpec,
    case: str,
    judge_model: str,
    extractor_model: str,
) -> Dict[str, Any]:
    """Report what the stage produced, and how much of it is grounded."""
    meta: Dict[str, Any] = dict(stats)
    meta.update(
        {
            "case": case,
            "judge_model": judge_model,
            "extractor_model": extractor_model,
            "schema_name": spec.name,
            "schema_version": spec.schema_version,
            "rows": int(len(out)),
        }
    )
    if out.empty:
        return meta

    real = out[out["extraction_class"].notna()]
    meta["extractions"] = int(len(real))
    meta["traces_with_no_extraction"] = int(len(out) - len(real))
    meta["extractions_per_trace"] = round(
        float(len(real)) / max(1, int(stats.get("traces_extracted", 1))), 2
    )
    if len(real):
        # An unaligned extraction is a defect, not data. Keep the rate visible.
        grounded = real["alignment_status"].notna()
        meta["grounded_rate"] = round(float(grounded.mean()), 4)
        meta["exact_rate"] = round(
            float((real["alignment_status"] == "match_exact").mean()), 4
        )
        # The rate that an analysis may use. A `match_lesser` row holds a
        # composed sentence, and its offsets point at a fragment of it.
        if "is_quotable" in real.columns:
            meta["quotable_rate"] = round(float(real["is_quotable"].mean()), 4)
        for klass, count in real["extraction_class"].value_counts().items():
            meta[f"class/{klass}"] = int(count)
    return meta
