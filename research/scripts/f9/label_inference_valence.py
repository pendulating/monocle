#!/usr/bin/env python
"""F9: add a valence to the social inference spans of the F9 answers.

The `inference` class of the cue schemas has a `kind` and no valence. Only
`visual_evidence` has one. The figure of `research/notebooks/facct/_trace_ses.py` needs
the valence of 5 kinds: upkeep, wealth, demographic, class, safety. This
script labels each of those spans `good`, `bad` or `neutral` with the extractor
of the spans (DeepSeek-V4-Flash, no thinking, temperature 0, a JSON schema).

| Step | Node | Output in `data/f4_traces/subj/inference_valence/` |
|------|------|----------------------------------------------------|
| `prepare` | CPU | `input.parquet`: 1 row for each span, with the text around it. `texts.parquet`: the text table of the other outcomes |
| `label` | 1 B300 | `<tag>/chunk_*.parquet`, then `<tag>.parquet` (`span_id`, `valence`, `finish_reason`) |

`label` writes 1 chunk at a time and skips a chunk that exists, thus a requeue
continues the run. `--order reversed` lists the 3 labels in the opposite order:
the test-retest of a span sample.

Use (the GPU step goes through `label_inference_valence.sbatch`):
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f9/label_inference_valence.py prepare
    F9V_ARGS="--tag smoke --limit 600" sbatch --export=ALL --exclude=g3-compute-17 \
        research/scripts/f9/label_inference_valence.sbatch

See `research/docs/f9-inference-valence.md`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

__version__ = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = REPO_ROOT / "data" / "f4_traces" / "subj" / "inference_valence"
MODEL_SOURCE = "/share/ju/matt/zoo/models/DeepSeek-V4-Flash-0731"

SOCIAL_KINDS = ("upkeep", "wealth", "demographic", "class", "safety")
LABELS = ("good", "bad", "neutral")
WINDOW = 300
CHUNK = 4000
SCHEMA = {"type": "object", "properties": {"valence": {"type": "string", "enum": list(LABELS)}},
          "required": ["valence"], "additionalProperties": False}
# case -> (arm dir, input case, the thing that the image shows, the question of the F9 prompt)
CASES: Dict[str, tuple] = {
    "restaurant": ("cues", "restaurant_subj_x2", "restaurant", "Does this look like a nice restaurant to eat at?"),
    "street": ("cues_street", "street_subj_x2", "street", "Would this street make a good spot for taking a photo?"),
}
# The text of the `kind` values in the schemas.
KIND_TEXT = {
    "upkeep": "how well the place is kept",
    "wealth": "price or wealth (high-end, cheap)",
    "demographic": "who lives, works, or eats there",
    "class": "social class",
    "safety": "safety",
}
DEFINITIONS = {
    "good": "good: the text presents the claim as a good thing about the place. It counts for the place.",
    "bad": "bad: the text presents the claim as a bad thing about the place. It counts against the place.",
    "neutral": "neutral: the text states the claim with no approval and no disapproval, or the text gives both sides.",
}
EXAMPLES = (
    ("The bright, tidy front suggests [[the owners take care of the place]].", "good"),
    ("With a wine list like that, [[this is clearly an upscale spot]], perfect for a date.", "good"),
    ("Peeling paint and a broken sign: [[it looks neglected]].", "bad"),
    ("The bars on the windows make me think [[the area feels unsafe at night]], so I would not stop here.", "bad"),
    ("The menu board suggests [[prices are in the moderate range]].", "neutral"),
    ("Nothing says who comes here, but [[it probably serves office workers]] from the towers nearby.", "neutral"),
)


def prompt(row, order: Sequence[str] = LABELS) -> str:
    """Give the prompt of 1 span. The span sits between [[ and ]] in the text around it."""
    _, _, thing, question = CASES[row["case"]]
    part = "final answer" if row["field"] == "response" else "reasoning"
    defs = "\n".join(f"- {DEFINITIONS[v]}" for v in order)
    shots = "\n".join(f'Text: {t}\nAnswer: {{"valence": "{v}"}}' for t, v in EXAMPLES)
    return (
        f'A vision model looked at a photograph of a {thing}. The question was: "{question}"\n\n'
        f"Below is a part of its {part}. The words between [[ and ]] are a claim of the model about "
        f"{KIND_TEXT[row['kind']]}.\n\n"
        f"How does the text present this claim?\n{defs}\n\n"
        "Read the stance of the text, not your own view of the claim.\n\n"
        f"Examples:\n{shots}\n\n"
        f"Text: {row['before']}[[{row['extraction_text']}]]{row['after']}\n"
        'Give the JSON with the key "valence".'
    )


def _context(sp: pd.DataFrame, texts: pd.Series) -> pd.DataFrame:
    """Add the text before and after each span.

    `offset_ok` is False when the offsets do not hold the exact span text (about
    1%, mostly normalized characters). The text around the span stays usable.
    """
    full = sp["doc_id"].map(texts)
    before, after, ok = [], [], []
    for t, a, b, s in zip(full, sp["char_start"], sp["char_end"], sp["extraction_text"]):
        a, b = int(a), int(b)
        ok.append(isinstance(t, str) and t[a:b] == s)
        if not isinstance(t, str):
            before.append("")
            after.append("")
            continue
        before.append(t[max(0, a - WINDOW):a])
        after.append(t[b:b + WINDOW])
    return sp.assign(before=before, after=after, offset_ok=ok)


def prepare() -> None:
    """Write the spans to label and the text table of the other 3 outcomes."""
    import analyze_subjective as A

    OUT.mkdir(parents=True, exist_ok=True)
    inputs, tables = [], []
    for case, (arm, name, _, _) in CASES.items():
        raw = pd.read_parquet(A.X2 / arm / f"f4_{name}_extractions.parquet")
        raw = raw[~raw.copy_suspect.astype(bool)].assign(case=case, prompt="f9")
        sp = A.spans_of(raw)
        t = A.text_valence(raw, sp)
        inf = sp[sp.extraction_class == "inference"]
        t = t.merge(inf[inf.kind == "location"].groupby("doc_id").size().rename("n_location"),
                    left_on="doc_id", right_index=True, how="left")
        unc = sp[sp.extraction_class == "uncertainty"].groupby("doc_id").size().rename("n_uncertainty")
        t = t.merge(unc, left_on="doc_id", right_index=True, how="left")
        t[["n_location", "n_uncertainty"]] = t[["n_location", "n_uncertainty"]].fillna(0).astype(int)
        tables.append(t)
        source = pd.read_parquet(A.INPUT / f"f4_{name}.parquet", columns=["doc_id", "text"]).set_index("doc_id")["text"]
        keep = ["doc_id", "case", "model", "field", "image_idx", "kind", "extraction_text", "char_start", "char_end"]
        social = _context(inf[inf.kind.isin(SOCIAL_KINDS)][keep], source)
        print(f"[prepare] {case}: {len(t):,} texts, {len(social):,} spans, offsets hold {social.offset_ok.mean():.4f}",
              flush=True)
        inputs.append(social)
    spans = pd.concat(inputs, ignore_index=True)
    spans.insert(0, "span_id", range(len(spans)))
    spans.to_parquet(OUT / "input.parquet", index=False)
    pd.concat(tables, ignore_index=True).to_parquet(OUT / "texts.parquet", index=False)
    print(spans.groupby(["case", "field", "kind"]).size().unstack("kind", fill_value=0).to_string())


def _engine(seqs: int):
    from omegaconf import OmegaConf

    from dagspaces.common.vllm_structured import VLLMEngine

    cfg = OmegaConf.create({"model": {
        "model_source": MODEL_SOURCE, "model_family": "deepseek_v4", "multimodal": False, "thinking_mode": "off",
        "chat_template_kwargs": {"enable_thinking": False},
        "engine_kwargs": {"max_model_len": 4096, "max_num_seqs": seqs, "tensor_parallel_size": 1,
                          "trust_remote_code": False, "kv_cache_dtype": "fp8"},
    }})
    return VLLMEngine(cfg, stage_name="inference_valence")


def _parse(answer: str) -> Optional[str]:
    try:
        v = json.loads(answer).get("valence")
    except (ValueError, AttributeError):
        return None
    return v if v in LABELS else None


def label(tag: str, order: Sequence[str], limit: int, seqs: int, seed: int) -> None:
    """Label the spans (or a random sample of `limit` spans), 1 chunk at a time."""
    spans = pd.read_parquet(OUT / "input.parquet")
    if limit:
        spans = spans.sample(n=min(limit, len(spans)), random_state=seed)
    spans = spans.sort_values("span_id").reset_index(drop=True)
    folder = OUT / tag
    folder.mkdir(parents=True, exist_ok=True)
    print(f"[label] {tag}: {len(spans):,} spans, order {list(order)}, version {__version__}", flush=True)
    engine = None
    try:
        for start in range(0, len(spans), CHUNK):
            path = folder / f"chunk_{start:07d}.parquet"
            if path.exists():
                continue
            if engine is None:
                engine = _engine(seqs)
            t0 = time.time()
            part = spans.iloc[start:start + CHUNK]
            answers = engine.generate([prompt(r, order) for _, r in part.iterrows()],
                                      temperature=0.0, max_tokens=24, guided_json=SCHEMA)
            out = pd.DataFrame({"span_id": part["span_id"].values, "valence": [_parse(a) for a in answers],
                                "finish_reason": engine.last_finish_reasons})
            out.to_parquet(path.with_suffix(".tmp"), index=False)
            path.with_suffix(".tmp").rename(path)
            print(f"[label] {start + len(part):,}/{len(spans):,} in {time.time() - t0:.1f}s, "
                  f"unparsed {int(out.valence.isna().sum())}", flush=True)
    finally:
        if engine is not None:
            engine.shutdown()
    done = pd.concat([pd.read_parquet(p) for p in sorted(folder.glob("chunk_*.parquet"))], ignore_index=True)
    done.to_parquet(OUT / f"{tag}.parquet", index=False)
    print(f"[label] wrote {tag}.parquet: {len(done):,} rows", done.valence.value_counts(dropna=False).to_dict(), flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("prepare")
    lab = sub.add_parser("label")
    lab.add_argument("--tag", default="full")
    lab.add_argument("--order", choices=("listed", "reversed"), default="listed")
    lab.add_argument("--limit", type=int, default=0)
    lab.add_argument("--seqs", type=int, default=256)
    lab.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.step == "prepare":
        prepare()
    else:
        label(args.tag, LABELS[::-1] if args.order == "reversed" else LABELS, args.limit, args.seqs, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
