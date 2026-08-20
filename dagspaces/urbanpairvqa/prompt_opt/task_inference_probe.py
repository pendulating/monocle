"""Direct task-inference probe + axis-ceiling reference evals.

Two cheap controls for the GEPA prompt-retranslation program, sharing one
in-process engine (no GEPA search involved):

1. **Task-inference probe** — can the model name the hidden question at all,
   given N in-context (caption, label) records? Separates channel limits from
   search limits: if direct inference fails too, the caption channel does not
   carry the task; if it succeeds where GEPA converged elsewhere, GEPA's
   local search is the bottleneck. Run for both caption styles.

2. **Axis ceilings** — evaluate the frozen axis-slot scaffold on the exact
   GEPA val split with (a) the TRUE task attribute, (b) the 'visually complex
   and busy' style-attractor axis, (c) the generic seed axis. (a) is the
   number an axis search is chasing; if (a) ≈ (b), val score cannot certify
   semantic recovery and arms must be judged by whether they NAME the axis.

Run on a GPU node under .venv-nightly (see scripts/gepa_probe.sub):

  python -m dagspaces.urbanpairvqa.prompt_opt.task_inference_probe \\
      --tasks subway,restaurants,schools --outdir outputs/gepa_pairwise/probe_X
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from dagspaces.urbanpairvqa.prompt_opt.gepa_pairwise import (
    DEFAULT_MODEL_DIR,
    DEFAULT_SEED_AXIS,
    DESCRIBE_STYLES,
    PairwiseGEPAAdapter,
    PairwiseTaskEngine,
    REPO,
    build_frames,
    _ordinal_score,
)

LOG = logging.getLogger(__name__)

# Supervision parquets = the same production runs the GEPA sweeps distill;
# true_axis paraphrases the production question as a bare attribute phrase
# usable in "Which photograph looks more {axis}?".
TASKS: Dict[str, Dict[str, Any]] = {
    "subway": {
        "parquet": REPO / "multirun/2026-06-29_URBANPAIRVQA/14-37-17/0/outputs/"
                          "pairwise/subway_safety_mvp_20260629_143729.parquet",
        "true_axis": "safe for commuters and residents",
    },
    "street": {
        "parquet": REPO / "multirun/2026-06-29_URBANPAIRVQA/14-37-17/0/outputs/"
                          "pairwise/street_photography_mvp_20260629_143729.parquet",
        "true_axis": "appealing as a location for a street photography photoshoot",
    },
    "restaurants": {
        "parquet": REPO / "multirun/2026-06-23_URBANPAIRVQA/14-38-25/0/outputs/"
                          "pairwise/restaurants_mvp_20260623_143838.parquet",
        "true_axis": "appealing as a place to eat",
    },
    "schools": {
        "parquet": REPO / "multirun/2026-06-17_URBANPAIRVQA/17-06-54/0/outputs/"
                          "pairwise/schools_mvp_20260617_170705.parquet",
        "true_axis": "appealing as a school to send your child to",
    },
    "libraries": {
        "parquet": REPO / "multirun/2026-07-10_URBANPAIRVQA/23-51-43/0/outputs/"
                          "pairwise/libraries_mvp_20260710_235151.parquet",
        "true_axis": "well-maintained",
    },
}
COMPLEXITY_AXIS = "visually complex and busy"


def _build_guess_prompt(records: Sequence[Mapping[str, str]]) -> str:
    """In-context task-inference prompt. Task-neutral framing only: the sole
    semantics channel is the captions + labels themselves."""
    lines = [
        f"Below are {len(records)} records. Each contains an observer's "
        "description of a pair of street-level photographs (Image A and "
        "Image B), followed by a label on the scale MuchLess, Less, Same, "
        "More, MuchMore. The labels were produced by a vision model that saw "
        "the two photographs themselves (not the descriptions) and answered "
        "ONE fixed question about every pair, always interpreted as Image A "
        "relative to Image B.",
        "",
    ]
    for i, rec in enumerate(records, 1):
        lines.append(f"[{i}] Description: {rec['caption']}")
        lines.append(f"[{i}] Label: {rec['label']}")
        lines.append("")
    lines.append(
        "Infer the hidden question. State your single best guess for the "
        "question on a line starting with 'QUESTION:', then briefly note "
        "which regularities in the records support your guess.")
    return "\n".join(lines)


def _eval_axis(adapter: PairwiseGEPAAdapter, valset: List[Mapping[str, Any]],
               axis: str) -> Dict[str, float]:
    batch = adapter.evaluate(valset, {"axis": axis}, capture_traces=False)
    n = max(len(batch.scores), 1)
    return {
        "ordinal": sum(batch.scores) / n,
        "exact": sum(t["exact"] for t in batch.outputs) / n,
        "n": len(batch.scores),
    }


def _constant_baseline(valset: Sequence[Mapping[str, Any]],
                       label: str = "More") -> Dict[str, float]:
    scores = [_ordinal_score(label, str(r["expected_answer"])) for r in valset]
    exact = [float(str(r["expected_answer"]) == label) for r in valset]
    n = max(len(valset), 1)
    return {"ordinal": sum(scores) / n, "exact": sum(exact) / n, "n": len(valset)}


def probe_task(engine: PairwiseTaskEngine, name: str, spec: Mapping[str, Any],
               args: argparse.Namespace, outdir: Path) -> Dict[str, Any]:
    trainset, valset = build_frames(Path(spec["parquet"]), args.train_n,
                                    args.val_n, args.seed)
    rng = random.Random(args.seed)
    examples = rng.sample(trainset, min(args.probe_n, len(trainset)))
    result: Dict[str, Any] = {"task": name, "n_examples": len(examples),
                              "true_axis": spec["true_axis"]}

    # 1) task-inference probe, both caption styles
    result["guesses"] = {}
    for style in sorted(DESCRIBE_STYLES):
        captions = engine.describe(examples, style=style)
        records = [{"caption": c, "label": str(r["expected_answer"]),
                    "sample_id": r["sample_id"]}
                   for r, c in zip(examples, captions)]
        cap_path = outdir / f"{name}_probe_captions_{style}.jsonl"
        with cap_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        prompt = _build_guess_prompt(records)
        guesses = [engine.reflect(prompt).strip() for _ in range(args.guesses)]
        result["guesses"][style] = guesses
        for i, g in enumerate(guesses):
            head = next((ln for ln in g.splitlines() if "QUESTION:" in ln), g[:200])
            LOG.info("[%s/%s] guess %d: %s", name, style, i + 1, head.strip())

    # 2) axis ceilings on the exact GEPA val split
    adapter = PairwiseGEPAAdapter(engine, candidate_mode="axis")
    result["axis_ceilings"] = {}
    for tag, axis in [("true_axis", spec["true_axis"]),
                      ("complexity_axis", COMPLEXITY_AXIS),
                      ("seed_axis", DEFAULT_SEED_AXIS)]:
        scores = _eval_axis(adapter, valset, axis)
        result["axis_ceilings"][tag] = {"axis": axis, **scores}
        LOG.info("[%s] axis ceiling %s (%r): ordinal=%.4f exact=%.4f",
                 name, tag, axis, scores["ordinal"], scores["exact"])
    result["constant_more"] = _constant_baseline(valset)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="subway,restaurants,schools",
                    help=f"comma list from: {','.join(TASKS)}")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--probe-n", type=int, default=50,
                    help="in-context records per task-inference prompt")
    ap.add_argument("--guesses", type=int, default=5,
                    help="sampled question guesses per caption style")
    ap.add_argument("--guess-temperature", type=float, default=0.8)
    ap.add_argument("--train-n", type=int, default=2000,
                    help="match the GEPA runs so splits are identical")
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--outdir", type=Path,
                    default=REPO / "outputs/gepa_pairwise/probe")
    args = ap.parse_args()

    names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in names if t not in TASKS]
    if unknown:
        raise SystemExit(f"unknown tasks: {unknown}; choose from {list(TASKS)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    engine = PairwiseTaskEngine(
        args.model_dir,
        max_model_len=args.max_model_len,
        reflection_temperature=args.guess_temperature,
    )

    results = [probe_task(engine, name, TASKS[name], args, args.outdir)
               for name in names]
    summary = {
        "tasks": names,
        "probe_n": args.probe_n,
        "guesses_per_style": args.guesses,
        "guess_temperature": args.guess_temperature,
        "model_dir": args.model_dir,
        "seed": args.seed,
        "results": results,
    }
    (args.outdir / "probe_results.json").write_text(
        json.dumps(summary, indent=2, default=str))
    LOG.info("probe done -> %s", args.outdir / "probe_results.json")

    # same vLLM 0.23 non-daemon-thread workaround as gepa_pairwise
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
