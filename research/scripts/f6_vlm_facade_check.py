#!/usr/bin/env python
"""F6: ask Gemma-4-12B-it if the best view of each restaurant shows a facade.

Geometry cannot see a truck in front of a storefront. This step asks a model
about the pixels of each best view (`f6_select_storefront_views.py
render-best`), and keeps only the restaurants with a "Yes".

The rules come from the decisions in `research/docs/facct-runbook.md`
(D63 to D69).

| Rule | Value | Decision |
|------|-------|----------|
| Model | Gemma-4-12B-it (a model of the study; the circularity is accepted) | D63, D64 |
| Question | the text of `QUESTION`, verbatim | D63, D65 |
| Decoding | structured choice of `Yes` or `No`, temperature 0, thinking off, no system prompt | D63, D69 |
| Views | only the best view of each restaurant | D67 |
| A "No" | the restaurant leaves; a random restaurant with a "Yes" replaces it | D67 |
| Target | 10,000 different restaurants; 2 restaurants can share 1 photo | D66, D68 |

The loop of D63 (draw, drop the "No", draw again until all are "Yes") gives a
uniform random set of 10,000 restaurants from the restaurants with a "Yes".
This script asks every best view one time and then draws the 10,000 from that
pool. The two give the same distribution, and this one also gives the size of
the pool.

Choices of the implementation:
- The image comes before the question in the user turn, as the Gemma-4 model
  card recommends.
- P(Yes) is `exp(lp_yes) / (exp(lp_yes) + exp(lp_no))` at the first answer
  position. `Yes` (id 10784) and `No` (id 3771) are each 1 token
  (`python -m research.monocle.labels Yes No`). The label is the decoded text, not a
  cut on P(Yes).

Use:
    sbatch research/scripts/f6_vlm_facade_check.sub                       # ask  (GPU, 1 A6000 on klara)
    sbatch research/scripts/f6_vlm_facade_check_g3.sub                    # ask  (GPU, the B300 on g3)
    $PY research/scripts/f6_vlm_facade_check.py draw --n 10000            # draw (CPU)
    $PY research/scripts/f6_vlm_facade_check.py review --n 100            # Label Studio tasks of the drawn set
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

import polars as pl

ROOT = "/share/ju/matt/mllmsci"
sys.path.insert(0, ROOT)

# The selection of f6_select_storefront_views.py 2.0.0 (D73 to D84).
SEL_DIR = os.path.join(ROOT, "data/f6_restaurant_depiction/v4")
MODEL = "/share/ju/matt/zoo/models/gemma-4-12B-it"
QUESTION = "Is a restaurant facade or food restaurant signage visible in this image?"   # D63, verbatim
CHOICES = ["Yes", "No"]                                                                  # D69
SEED = 20261023

__version__ = "1.0.0"


def _views(sel_dir: str) -> pl.DataFrame:
    parts = sorted(glob.glob(os.path.join(sel_dir, "views", "index_*.parquet")))
    if not parts:
        raise SystemExit(f"no views index under {sel_dir}/views; run render-best first")
    return pl.concat([pl.read_parquet(p) for p in parts]).sort("uid")


def _gpu_name() -> str | None:
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def cmd_ask(args: argparse.Namespace) -> int:
    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    from dagspaces.common.model_registry import resolve_model_source
    from dagspaces.common.stage_utils import ensure_dotenv

    ensure_dotenv()   # server.env names the node-local model mirror
    views = _views(args.sel_dir).filter(pl.col("view_path").is_not_null())
    out_dir = os.path.join(args.sel_dir, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    chunks = [views.slice(i, args.chunk) for i in range(0, views.height, args.chunk)]
    todo = [(k, c) for k, c in enumerate(chunks)
            if not os.path.isfile(os.path.join(out_dir, f"part_{k:04d}.parquet"))]
    print(f"[ask] {views.height:,} views, {len(chunks)} chunks, {len(todo)} to do")
    if not todo:
        return 0

    llm = LLM(model=resolve_model_source(MODEL, stage_name="f6_facade_check"),
              max_model_len=2048, max_num_seqs=args.max_num_seqs, tensor_parallel_size=1,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True, seed=0)
    tok = llm.get_tokenizer()
    ids = {c: tok.encode(c, add_special_tokens=False) for c in CHOICES}
    if any(len(v) != 1 for v in ids.values()):
        raise SystemExit(f"a choice is not 1 token: {ids}")
    yes_id, no_id = ids["Yes"][0], ids["No"][0]
    sp = SamplingParams(temperature=0.0, max_tokens=4, logprobs=20,
                        structured_outputs=StructuredOutputsParams(choice=CHOICES))

    for k, chunk in todo:
        convs = []
        for path in chunk["view_path"].to_list():
            with Image.open(path) as im:
                img = im.convert("RGB")
            convs.append([{"role": "user", "content": [
                {"type": "image_pil", "image_pil": img},
                {"type": "text", "text": QUESTION},
            ]}])
        outs = llm.chat(convs, sp, use_tqdm=False,
                        chat_template_kwargs={"enable_thinking": False})
        rows = []
        for o in outs:
            g = o.outputs[0]
            first = g.logprobs[0] if g.logprobs else {}
            lp_yes = first[yes_id].logprob if yes_id in first else None
            lp_no = first[no_id].logprob if no_id in first else None
            p_yes = None
            if lp_yes is not None and lp_no is not None and math.isfinite(lp_yes) and math.isfinite(lp_no):
                m = max(lp_yes, lp_no)
                p_yes = math.exp(lp_yes - m) / (math.exp(lp_yes - m) + math.exp(lp_no - m))
            rows.append({"answer": g.text.strip(), "lp_yes": lp_yes, "lp_no": lp_no, "p_yes": p_yes,
                         "first_token_id": g.token_ids[0] if g.token_ids else None})
        part = pl.concat([chunk.select("uid", "recording_id", "view_path"),
                          pl.DataFrame(rows, schema={"answer": pl.Utf8, "lp_yes": pl.Float64,
                                                     "lp_no": pl.Float64, "p_yes": pl.Float64,
                                                     "first_token_id": pl.Int64})], how="horizontal")
        tmp = os.path.join(out_dir, f"part_{k:04d}.parquet.tmp")
        part.write_parquet(tmp)
        os.replace(tmp, os.path.join(out_dir, f"part_{k:04d}.parquet"))
        print(f"[ask] chunk {k}: {part.height} views, {(part['answer'] == 'Yes').sum()} Yes", flush=True)

    manifest = {"script": os.path.relpath(__file__, ROOT), "version": __version__, "step": "ask",
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": MODEL, "question": QUESTION, "choices": CHOICES,
                "sampling": {"temperature": 0.0, "max_tokens": 4, "logprobs": 20, "thinking": False},
                "token_ids": {"Yes": yes_id, "No": no_id}, "n_views": views.height,
                "gpu": _gpu_name(), "max_num_seqs": args.max_num_seqs, "chunk": args.chunk}
    with open(os.path.join(out_dir, "ask_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return 0


def cmd_draw(args: argparse.Namespace) -> int:
    out_dir = os.path.join(args.sel_dir, args.out_name)
    views = _views(args.sel_dir)
    ans = pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(os.path.join(out_dir, "part_*.parquet")))])
    rendered = views.filter(pl.col("view_path").is_not_null())
    if ans.height != rendered.height:
        raise SystemExit(f"{ans.height:,} answers for {rendered.height:,} views; finish `ask` first")
    best = pl.read_parquet(os.path.join(args.sel_dir, "best.parquet")).select("uid", "camis", "dba", "borough")
    pool = ans.filter(pl.col("answer") == "Yes").join(best, on="uid", how="left")
    if pool.height < args.n:
        raise SystemExit(f"only {pool.height:,} restaurants have a Yes; the target is {args.n:,}")
    drawn = pool.sort("uid").sample(n=args.n, seed=args.seed, shuffle=True)
    drawn.write_parquet(os.path.join(out_dir, f"drawn_{args.n}.parquet"))

    by_b = (best.join(ans.select("uid", "answer"), on="uid", how="left")
            .group_by("borough").agg(pl.len().alias("restaurants"),
                                     (pl.col("answer") == "Yes").sum().alias("yes"),
                                     (pl.col("answer") == "No").sum().alias("no"))
            .sort("borough"))
    manifest = {"script": os.path.relpath(__file__, ROOT), "version": __version__, "step": "draw",
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "restaurants": best.height, "rendered": rendered.height,
                "not_rendered": views.height - rendered.height,
                "yes": int((ans["answer"] == "Yes").sum()), "no": int((ans["answer"] == "No").sum()),
                "other_answer": int((~ans["answer"].is_in(CHOICES)).sum()),
                "n": args.n, "seed": args.seed,
                "distinct_recordings_drawn": drawn["recording_id"].n_unique(),
                "by_borough_pool": by_b.to_dicts(),
                "by_borough_drawn": dict(drawn["borough"].value_counts().iter_rows())}
    with open(os.path.join(out_dir, f"draw_{args.n}_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))
    return 0


LABEL_CONFIG = """<View>
  <Header value="$dba"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <Choices name="verdict" toName="image" choice="single" required="true" showInline="true">
    <Choice value="Pass"/>
    <Choice value="Fail"/>
  </Choices>
  <TextArea name="notes" toName="image" placeholder="Notes" rows="3" editable="true" maxSubmissions="1"/>
</View>
"""


def cmd_review(args: argparse.Namespace) -> int:
    """Draw --n restaurants of the drawn set for a check by hand (Q63).

    The tasks show the DBA name and the view, as in the version 3 project. They
    do not show the answer or P(Yes): every task is a Yes. `sample.parquet`
    keeps P(Yes) for the analysis.
    """
    out_dir = os.path.join(args.sel_dir, args.out_name)
    drawn = pl.read_parquet(os.path.join(out_dir, f"drawn_{args.pool}.parquet"))
    sample = drawn.sort("uid").sample(n=args.n, seed=args.seed, shuffle=True)
    rev_dir = os.path.join(out_dir, f"review{args.n}")
    os.makedirs(rev_dir, exist_ok=True)
    root = os.path.join(ROOT, "data/f6_restaurant_depiction")   # the Label Studio local-files root
    tasks = [{"data": {"image": f"/data/local-files/?d={os.path.relpath(r['view_path'], root)}",
                       "uid": r["uid"], "recording_id": r["recording_id"], "dba": r["dba"]}}
             for r in sample.iter_rows(named=True)]
    sample.write_parquet(os.path.join(rev_dir, "sample.parquet"))
    with open(os.path.join(rev_dir, "tasks.json"), "w") as fh:
        json.dump(tasks, fh, indent=1)
    with open(os.path.join(rev_dir, "label_config.xml"), "w") as fh:
        fh.write(LABEL_CONFIG)
    manifest = {"script": os.path.relpath(__file__, ROOT), "version": __version__, "step": "review",
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "pool": f"drawn_{args.pool}.parquet", "n": args.n, "seed": args.seed,
                "by_borough": dict(sample["borough"].value_counts().iter_rows()),
                "p_yes_quantiles": {q: sample["p_yes"].quantile(q) for q in (0.05, 0.25, 0.5)}}
    with open(os.path.join(rev_dir, "review_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("--sel-dir", "--v3-dir", default=SEL_DIR)
    a.add_argument("--chunk", type=int, default=512)
    a.add_argument("--max-num-seqs", type=int, default=64)
    a.add_argument("--out-name", default="facade_check",
                   help="the output folder under --sel-dir; 1 folder holds the answers of 1 GPU type")
    d = sub.add_parser("draw")
    d.add_argument("--sel-dir", "--v3-dir", default=SEL_DIR)
    d.add_argument("--n", type=int, default=10000)
    d.add_argument("--seed", type=int, default=SEED)
    d.add_argument("--out-name", default="facade_check")
    r = sub.add_parser("review")
    r.add_argument("--sel-dir", "--v3-dir", default=SEL_DIR)
    r.add_argument("--out-name", default="facade_check_b300")
    r.add_argument("--pool", type=int, default=10000)
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    return {"ask": cmd_ask, "draw": cmd_draw, "review": cmd_review}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
