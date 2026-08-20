"""Extract the Integrative Complexity ingredients from the traces of a run.

The stage reads a pairvqa results parquet, sends each `model_reasoning` text to
the extractor under a hand-written JSON schema, and writes 1 row for each
ingredient. A later step turns those rows into the IC codes and the score.

This stage does NOT use LangExtract. See `dagspaces/common/ic_schema.py` for
the reason: the schema needs real enums, real bounds, and nested sub-quotes,
and a schema derived from examples gives none of the three.

Warning: only a THINKING run holds a trace. The canonical battery runs greedy
with `max_tokens=128`, thus its `model_reasoning` column is empty.

Shards and resume
-----------------
The stage cuts the trace table into row ranges and writes 1 parquet for each
range under `chunks/`. A rerun skips a range whose chunk exists. Thus a job that
the walltime stops keeps everything it finished.

See `vlm-narratives-docs/ic-ingredient-extraction.md`.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from dagspaces.common import ic_schema as S
from dagspaces.urbanpairvqa.stages.trace_extract import (
    _sample,
    _shard,
    case_of_parquet,
    load_traces,
    source_provenance,
)

STAGE_NAME = "ic_extract"

DEFAULT_SHARD_ROWS = 2000

# How many prompts reach `generate()` at one time.
DEFAULT_BATCH_SIZE = 256


def build_prompts(
    traces: List[str], examples: Optional[List[Dict[str, Any]]],
    order: Optional[List[str]] = None,
) -> List[str]:
    """Render 1 prompt for each trace, in 1 ingredient order."""
    return [S.build_prompt(text, examples=examples, order=order) for text in traces]


def rows_for_batch(
    traces: List[str],
    doc_ids: List[str],
    answers: List[str],
    finish_reasons: List[str],
    bases: List[Dict[str, Any]],
    order: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Turn a batch of raw answers into output rows.

    Returns:
        (rows, counters). The counters name the 3 ways an answer fails: the cap
        cut it, the parser could not read it, or it held no ingredient.
    """
    rows: List[Dict[str, Any]] = []
    counters = {"cut": 0, "unparsed": 0, "repaired": 0, "no_ingredient": 0}

    for trace, doc_id, answer, reason, base in zip(
        traces, doc_ids, answers, finish_reasons, bases
    ):
        obj, error, repaired = S.parse_answer(answer)
        cut = reason == "length"
        counters["cut"] += int(cut)
        counters["repaired"] += int(repaired)
        if obj is None:
            counters["unparsed"] += 1

        # A repair can lose a whole array, thus a missing key would look like a
        # real zero. Mark the row so a count of absence can drop it.
        note = error
        if not note and repaired:
            note = "repaired: the token cap cut the answer"

        row_base = dict(base)
        row_base["doc_id"] = str(doc_id)
        row_base["answer_truncated"] = cut
        row_base["answer_repaired"] = repaired

        made = S.ingredient_rows(trace, obj, base=row_base, parse_error=note,
                                 order=order)
        if len(made) == 1 and made[0]["ingredient_type"] is None and obj is not None:
            counters["no_ingredient"] += 1
        rows.extend(made)
    return rows, counters


def run_ic_extract_stage(
    cfg: Any, output_dir: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Extract the IC ingredients from every trace of the source run.

    Args:
        cfg: The stage config. `cfg.ic_extract` holds the source, the shard
            size, and whether to use the few-shot examples.
        output_dir: Where to write the chunk parquets.

    Returns:
        (rows, metadata). `rows` is the long ingredient table.
    """
    icfg = getattr(cfg, "ic_extract", {})
    results_path = str(getattr(icfg, "results_path", "") or "")
    reasoning_col = str(getattr(icfg, "reasoning_column", "model_reasoning"))
    id_col = str(getattr(icfg, "id_column", "pair_id"))
    shard_rows = int(getattr(icfg, "shard_rows", DEFAULT_SHARD_ROWS) or DEFAULT_SHARD_ROWS)
    batch_size = int(getattr(icfg, "batch_size", DEFAULT_BATCH_SIZE) or DEFAULT_BATCH_SIZE)
    max_tokens = int(getattr(icfg, "max_tokens", S.DEFAULT_MAX_TOKENS) or S.DEFAULT_MAX_TOKENS)
    temperature = float(getattr(icfg, "temperature", 0.0) or 0.0)
    use_examples = bool(getattr(icfg, "use_examples", True))
    # The ingredient ORDER, which is also the schema version. v2 is the corpus
    # of 2026-08-18. v3 moves `dismissal` from 5th to 3rd and exists to test
    # whether that position, and not the trace, sets its rate.
    order = list(S.order_for(str(getattr(icfg, "schema_order", "v2") or "v2")))
    schema_version = S.version_for(order)

    # An example whose span is not in its own trace teaches a shape the model
    # cannot copy. Stop before the engine loads, not after.
    examples: Optional[List[Dict[str, Any]]] = None
    if use_examples:
        examples = S.load_examples(str(getattr(icfg, "examples_path", "") or "") or None)
        issues = _bad_examples(examples)
        if issues:
            raise ValueError(
                f"the examples of {S.SCHEMA_NAME}/{S.SCHEMA_VERSION} do not "
                "match their own traces:\n  " + "\n  ".join(issues)
            )

    df, stats = load_traces(cfg, section="ic_extract")
    # Sample first, then shard. The other order gives each shard its own sample
    # of a different size, and no 2 shards then cover the same set.
    df = _sample(df, cfg)
    df = _shard(df, cfg, section="ic_extract")
    stats["traces_extracted"] = int(len(df))

    # Follow a symlink before you read a name from the path. Every canonical
    # registry link is called `results.parquet`, thus `case_of_parquet` would
    # return "results" and every row of the corpus would carry the wrong case.
    real_path = os.path.realpath(results_path) if results_path else results_path
    provenance = source_provenance(real_path)
    case = str(getattr(icfg, "case", "") or "") or case_of_parquet(real_path)
    judge_model = (
        str(getattr(icfg, "judge_model", "") or "") or provenance.get("judge_model", "")
    )

    skip_inference = bool(getattr(getattr(cfg, "runtime", {}), "skip_inference", False))
    extractor_model = (
        "stub" if skip_inference else str(getattr(cfg.model, "model_source", "unknown"))
    )

    print(
        f"[{STAGE_NAME}] case={case} judge={judge_model} traces={len(df)} "
        f"schema={S.SCHEMA_NAME}/{schema_version} order={'-'.join(order)} "
        f"examples={len(examples) if examples else 0} max_tokens={max_tokens}",
        flush=True,
    )

    chunk_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    guided = S.json_schema(order)
    engine = None
    frames: List[pd.DataFrame] = []
    totals = {"cut": 0, "unparsed": 0, "repaired": 0, "no_ingredient": 0}
    t_start = time.time()

    try:
        if not skip_inference:
            from dagspaces.common.vllm_structured import VLLMEngine

            engine = VLLMEngine(cfg, stage_name=STAGE_NAME)
            # `S.parse_answer` repairs a cut answer AND reports the repair for
            # each row. The engine repair would hide that, thus turn it off.
            engine.repair_fn = None
            debug_answers = int(getattr(icfg, "debug_answers", 0) or 0)
            if debug_answers:
                engine.debug_answers_path = os.path.join(output_dir, "debug_answers.jsonl")
                engine.debug_answers_left = debug_answers
                print(
                    f"[{STAGE_NAME}] the first {debug_answers} raw answers go to "
                    f"{engine.debug_answers_path}",
                    flush=True,
                )

        for start in range(0, len(df), shard_rows):
            end = min(start + shard_rows, len(df))
            chunk_path = os.path.join(chunk_dir, f"ingredients_{start:07d}_{end:07d}.parquet")
            if os.path.exists(chunk_path):
                print(f"[{STAGE_NAME}] rows {start}-{end} exist, skipped", flush=True)
                frames.append(pd.read_parquet(chunk_path))
                continue

            shard = df.iloc[start:end]
            t0 = time.time()
            rows: List[Dict[str, Any]] = []

            for at in range(0, len(shard), batch_size):
                batch = shard.iloc[at : at + batch_size]
                traces = batch[reasoning_col].tolist()
                doc_ids = batch[id_col].astype(str).tolist()
                bases = []
                for _, source in batch.iterrows():
                    base: Dict[str, Any] = {
                        "case": case,
                        "judge_model": judge_model,
                        "sweep": provenance.get("sweep", ""),
                        # The REAL file, so a reader can compare this corpus
                        # with the canonical registry. `source_link_path` keeps
                        # the path the run was given, which is the registry
                        # link when the launch used one.
                        "source_results_path": real_path,
                        "source_link_path": results_path,
                        "extractor_model": extractor_model,
                        "few_shot": bool(examples),
                        id_col: str(source[id_col]),
                    }
                    for col in batch.columns:
                        if col in (id_col, reasoning_col):
                            continue
                        base[col] = source[col]
                    bases.append(base)

                if engine is None:
                    answers = ['{"dimensions": [], "verdicts": []}'] * len(traces)
                    reasons = ["stop"] * len(traces)
                else:
                    answers = engine.generate(
                        build_prompts(traces, examples, order),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        guided_json=guided,
                    )
                    reasons = list(engine.last_finish_reasons)

                made, counters = rows_for_batch(traces, doc_ids, answers, reasons,
                                                bases, order)
                rows.extend(made)
                for key, value in counters.items():
                    totals[key] += value

            chunk = pd.DataFrame(rows)
            chunk.to_parquet(chunk_path, index=False)
            frames.append(chunk)
            print(
                f"[{STAGE_NAME}] rows {start}-{end}: {len(chunk)} ingredients "
                f"in {time.time() - t0:.1f}s → {os.path.basename(chunk_path)}",
                flush=True,
            )
    finally:
        if engine is not None:
            engine.shutdown()

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    stats.update({f"answers_{k}": int(v) for k, v in totals.items()})
    metadata = _metadata(out, stats, case, judge_model, extractor_model,
                         bool(examples), schema_version)
    metadata["duration_s"] = round(time.time() - t_start, 1)
    metadata["shard_index"] = int(getattr(icfg, "shard_index", 0) or 0)
    metadata["shard_count"] = int(getattr(icfg, "shard_count", 1) or 1)
    metadata.update({f"source_{k}": v for k, v in provenance.items()})
    return out, metadata


def _bad_examples(examples: List[Dict[str, Any]]) -> List[str]:
    """Name every example span that is not in its own trace."""
    issues: List[str] = []
    for example in examples:
        rows = S.ingredient_rows(example["trace"], example["answer"])
        for row in rows:
            if row["ingredient_type"] is None:
                issues.append(f"{example.get('pair_id')}: the answer holds nothing")
            elif not row["quote_found"]:
                issues.append(
                    f"{example.get('pair_id')}: {row['ingredient_type']} "
                    f"quote not in the trace: {row['quote']!r}"
                )
            elif row["n_sub_quotes"] != row["n_sub_quotes_found"]:
                issues.append(
                    f"{example.get('pair_id')}: {row['ingredient_type']} "
                    "holds a sub-quote that is not in the trace"
                )
    return issues


def _metadata(
    out: pd.DataFrame,
    stats: Dict[str, Any],
    case: str,
    judge_model: str,
    extractor_model: str,
    few_shot: bool,
    schema_version: str = S.SCHEMA_VERSION,
) -> Dict[str, Any]:
    """Report what the stage produced, and how well it is grounded."""
    meta: Dict[str, Any] = dict(stats)
    meta.update(
        {
            "case": case,
            "judge_model": judge_model,
            "extractor_model": extractor_model,
            "few_shot": few_shot,
            "schema_name": S.SCHEMA_NAME,
            "schema_version": schema_version,
            "rows": int(len(out)),
        }
    )
    if out.empty:
        return meta

    real = out[out["ingredient_type"].notna()]
    traces = max(1, int(stats.get("traces_extracted", 1)))
    meta["ingredients"] = int(len(real))
    meta["traces_with_no_ingredient"] = int(len(out) - len(real))
    meta["ingredients_per_trace"] = round(float(len(real)) / traces, 2)
    if len(real):
        # A quote no search finds is a defect, not data. This is the number to
        # read first: it replaces the LangExtract alignment rate.
        meta["quote_found_rate"] = round(float(real["quote_found"].mean()), 4)
        meta["exact_rate"] = round(float((real["quote_method"] == "exact").mean()), 4)
        subs = int(real["n_sub_quotes"].sum())
        meta["sub_quotes"] = subs
        if subs:
            meta["sub_quote_found_rate"] = round(
                float(real["n_sub_quotes_found"].sum()) / subs, 4
            )
        for kind, count in real["ingredient_type"].value_counts().items():
            meta[f"type/{kind}"] = int(count)
            meta[f"per_trace/{kind}"] = round(float(count) / traces, 3)
    return meta
