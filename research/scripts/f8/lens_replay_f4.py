#!/usr/bin/env python3
"""F8.4: replay the stored F4 answers of gemma-4-31b and read the logit lens.

The F4 run (vLLM, thinking on, the gemma4 reasoning parser) kept the text of
each answer and not the token ids. This script builds the completion again and
teacher-forces it through the HF model in one forward pass. A hook reads the
output of each block. At each position that predicts a completion token, and
at each layer, the plain logit lens gives a distribution over the vocabulary:
`unembed(norm(h_l))`, with the final logit softcap of the model (jlens
HFLensModel.unembed, the `logit` lens of research/scripts/whatami/lens_replay.py).

The completion that the model wrote (thinking on, chat_template.jinja of the
model, vllm/parser/gemma4.py):

    <|channel>thought\\n{reasoning}<channel|>{response}<turn|>

The parser removes the `<|channel>` and `<channel|>` tokens and the
`thought\\n` label, and strips the whitespace at the ends of each field.
Thus the script tries the stored text first. When the token count does not
agree with the served `completion_tokens`, it tries whitespace at the 3
boundaries (end of reasoning, start of response, end of response) and keeps
the first variant whose count agrees. The column `ws_fix` records the
variant. A count that does not agree after this has a different tokenization
(the sampler can write a token sequence that the tokenizer does not make).
The replay then uses the canonical tokens, and `count_diff` is not 0.

The checks, logged for each chunk and written to answers.parquet:

| Check | Pass |
|---|---|
| HF prompt tokens == served prompt_tokens | always (the job stops if not) |
| replayed completion tokens == completion_tokens | most answers; off by 1 is logged |
| final-layer top 1 == written token | below 100%, because the sampler used T = 1.0 |
| lens at the last layer == model logits (top 1) | about 100% |

The selection (written once to <out-root>/selection.parquet, fixed seed): the
gemma-4-31b restaurant answers that are not cut; the 100 images of the
lowest-income NTAs and the 100 of the highest-income NTAs (NTA from
f4_restaurants_nta.parquet, median_household_income from nta_acs.parquet).
Ties in income are broken at random with the seed. The smoke takes the first
N/2 images of each group in the seeded order.

The output, one parquet file for each chunk of answers, in <out-root>/<run>/:

| Column | Type | Content |
|---|---|---|
| image_idx | int32 | the image (and the answer: 1 answer for each image) |
| pos | int32 | 0 is the last prompt position, it predicts `<|channel>` |
| target_id | int32 | the token at pos + 1 |
| field | int8 | 0 delimiter, 1 reasoning, 2 response (the field of target_id) |
| topk_ids | fixed [layers][k] int32 | the top-k tokens at each stored layer |
| topk_logprob | fixed [layers][k] float32 | their log-probabilities |
| target_rank | fixed [layers] int32 | the rank of target_id (0 = top) |
| target_logprob | fixed [layers] float32 | its log-probability |

Usage (as a GPU job, research/scripts/f8/lens_job.sbatch):
    python research/scripts/f8/lens_replay_f4.py --select-only            # CPU
    python research/scripts/f8/lens_replay_f4.py --run smoke --limit 20
    python research/scripts/f8/lens_replay_f4.py --run figure
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research/scripts/whatami"))
sys.path.insert(0, str(REPO / "sub/jacobian-lens"))

TRACES = REPO / "data/f4_traces/pooled/f4_traces_1k.parquet"
NTA = REPO / "data/f4_traces/pooled/f4_restaurants_nta.parquet"
ACS = REPO / "data/f4_traces/nta_acs.parquet"
MODEL_DIR = "/share/ju/matt/zoo/models/Gemma-4-31B-it"
OUT_ROOT = REPO / "multirun/f8/lens"
MODEL, CASE = "gemma-4-31b", "restaurant"
SEED = 20261028
N_GROUP = 100

#: Whitespace that the parser can strip at a boundary. The order sets the
#: preference: the stored text first.
WS = ["", "\n", "\n\n", " "]
FIELD_DELIM, FIELD_REASONING, FIELD_RESPONSE = 0, 1, 2


def log(msg: str) -> None:
    print(f"[f8-lens] {msg}", flush=True)


# ── the selection ──────────────────────────────────────────────────────
def make_selection(path: Path) -> pd.DataFrame:
    """The 100 lowest-income and the 100 highest-income NTA images (not cut)."""
    tr = pd.read_parquet(TRACES, columns=["model", "case", "image_idx", "sample_id", "image_path", "cut",
                                          "prompt_tokens", "completion_tokens"])
    tr = tr[(tr.model == MODEL) & (tr.case == CASE) & (~tr.cut)]
    nta = pd.read_parquet(NTA, columns=["image_idx", "NTA2020", "NTAName", "BoroName"])
    acs = pd.read_parquet(ACS, columns=["nta2020", "median_household_income", "income_topcoded"])
    df = tr.merge(nta, on="image_idx").merge(acs, left_on="NTA2020", right_on="nta2020", how="left")
    df = df.dropna(subset=["median_household_income"])
    rng = np.random.default_rng(SEED)
    # A random key breaks the ties in income (the topcode holds 182 images).
    df["tie"] = rng.random(len(df))
    df = df.sort_values(["median_household_income", "tie"]).reset_index(drop=True)
    low = df.head(N_GROUP).assign(group="low")
    high = df.iloc[::-1].head(N_GROUP).assign(group="high")
    sel = pd.concat([low, high], ignore_index=True)
    sel["order"] = sel.groupby("group").cumcount()
    sel = sel[["image_idx", "sample_id", "image_path", "group", "order", "NTA2020", "NTAName", "BoroName",
               "median_household_income", "income_topcoded", "prompt_tokens", "completion_tokens"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    sel.to_parquet(path, index=False)
    log(f"selection: {len(sel)} images -> {path}")
    for g, d in sel.groupby("group"):
        log(f"  {g}: income {d.median_household_income.min():.0f}..{d.median_household_income.max():.0f}; "
            f"NTAs {d.NTAName.value_counts().to_dict()}")
    return sel


# ── the completion ─────────────────────────────────────────────────────
def build_completion(tok, reasoning: str, response: str, served: int):
    """Token ids, field codes and the whitespace fix of one completion.

    Returns (ids, fields, ws_fix, count_diff). `fields[i]` is the field of
    ids[i], from the character offsets of the fast tokenizer.
    """
    best = None
    for a, b, c in itertools.product(WS, WS, WS):
        r_txt = reasoning + a
        p_txt = b + response + c
        head = "<|channel>thought\n"
        text = head + r_txt + "<channel|>" + p_txt + "<turn|>"
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        n = len(enc.input_ids)
        if best is None:
            best = (text, enc, (a, b, c), head, r_txt)
        if n == served:
            best = (text, enc, (a, b, c), head, r_txt)
            break
        if (a, b, c) == ("", "", "") and abs(n - served) > 8:
            # A large gap is a different tokenization, not stripped whitespace.
            break
    text, enc, fix, head, r_txt = best
    r0 = len(head)
    r1 = r0 + len(reasoning)
    p0 = r0 + len(r_txt) + len("<channel|>") + len(fix[1])
    p1 = p0 + len(response)
    fields = []
    for s, _e in enc.offset_mapping:
        if r0 <= s < r1:
            fields.append(FIELD_REASONING)
        elif p0 <= s < p1:
            fields.append(FIELD_RESPONSE)
        else:
            fields.append(FIELD_DELIM)
    ws = "|".join(repr(x)[1:-1] for x in fix)
    return list(enc.input_ids), np.array(fields, np.int8), ws, len(enc.input_ids) - served


# ── the model ──────────────────────────────────────────────────────────
def load(model_dir: str):
    import torch
    import jlens
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_dir)
    model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    model.requires_grad_(False)
    lm = jlens.from_hf(model, proc.tokenizer, force_bos=False)
    tmpl = (Path(model_dir) / "chat_template.jinja").read_text()
    log(f"loaded {model_dir} in {time.time() - t0:.0f}s: {type(model).__name__}, "
        f"n_layers={lm.n_layers}, d_model={lm.d_model}, softcap={lm._logit_softcap}")
    return proc, model, lm, tmpl


def build_prompt(proc, tmpl: str, image_path: str, user_text: str) -> dict:
    """The served prompt: thinking on, one user turn, the image then the text."""
    from lens_common import open_image

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                    chat_template=tmpl, enable_thinking=True)
    out = proc(text=[text], images=[[open_image(image_path)]], return_tensors="pt", add_special_tokens=False)
    return dict(out.to("cuda:0"))


def read_answer(model, lm, inputs: dict, prompt_len: int, layers: list[int], k: int):
    """One teacher-forced pass; the logit lens at each stored layer."""
    import torch
    from jlens.hooks import ActivationRecorder

    ids = inputs["input_ids"][0]
    seq = int(ids.shape[0])
    pos = torch.arange(prompt_len - 1, seq - 1, device=ids.device)
    targets = ids[pos + 1]
    with torch.no_grad():
        with ActivationRecorder(lm.layers, at=layers) as rec:
            out = model(**inputs, use_cache=False)
            hidden = {l: rec.activations[l][0, pos] for l in layers}
        model_top1 = out.logits[0, pos].argmax(-1)
        del out
        n, L = int(pos.numel()), len(layers)
        vocab = int(lm.unembed(hidden[layers[-1]][:1]).shape[-1])
        per_call = max(1, int(8e9 // (n * vocab * 4)))
        top_ids = np.empty((n, L, k), np.int32)
        top_lp = np.empty((n, L, k), np.float32)
        t_rank = np.empty((n, L), np.int32)
        t_lp = np.empty((n, L), np.float32)
        for c in range(0, L, per_call):
            part = list(range(c, min(L, c + per_call)))
            X = torch.stack([hidden[layers[j]] for j in part])
            lp = torch.log_softmax(lm.unembed(X.flatten(0, 1)).float(), dim=-1).view(len(part), n, -1)
            v, i = lp.topk(k, dim=-1)
            tl = lp.gather(2, targets[None, :, None].expand(len(part), n, 1)).squeeze(2)
            rank = (lp > tl[..., None]).sum(2)
            top_ids[:, part] = i.permute(1, 0, 2).cpu().numpy()
            top_lp[:, part] = v.permute(1, 0, 2).cpu().numpy()
            t_rank[:, part] = rank.T.cpu().numpy()
            t_lp[:, part] = tl.T.cpu().numpy()
            del lp, X
        agree_model = float((torch.from_numpy(top_ids[:, -1, 0]).to(model_top1.device) == model_top1).float().mean())
    return (pos - (prompt_len - 1)).cpu().numpy(), targets.cpu().numpy(), (top_ids, top_lp, t_rank, t_lp), agree_model


def _fixed(arr: np.ndarray):
    import pyarrow as pa
    flat = pa.array(arr.reshape(-1))
    for size in reversed(arr.shape[1:]):
        flat = pa.FixedSizeListArray.from_arrays(flat, size)
    return flat


def write_chunk(path: Path, rows: list[tuple]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cat = lambda j: np.concatenate([r[4][j] for r in rows])  # noqa: E731
    n_of = [len(r[1]) for r in rows]
    t = pa.table({
        "image_idx": pa.array(np.repeat([r[0] for r in rows], n_of).astype(np.int32)),
        "pos": pa.array(np.concatenate([r[1] for r in rows]).astype(np.int32)),
        "target_id": pa.array(np.concatenate([r[2] for r in rows]).astype(np.int32)),
        "field": pa.array(np.concatenate([r[3] for r in rows]).astype(np.int8)),
        "topk_ids": _fixed(cat(0)),
        "topk_logprob": _fixed(cat(1)),
        "target_rank": _fixed(cat(2)),
        "target_logprob": _fixed(cat(3)),
    })
    tmp = path.with_suffix(".tmp")
    pq.write_table(t, tmp, compression="zstd")
    tmp.rename(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="figure", help="output subdirectory: smoke or figure")
    ap.add_argument("--limit", type=int, default=0, help="N answers: the first N/2 of each group")
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--chunk-answers", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--layers", default="all", help="'all', or a comma list of block indices")
    ap.add_argument("--image-root", default=None, help="staged copies as <sample_id>.jpg")
    ap.add_argument("--model-dir", default=os.environ.get("F8_MODEL_DIR") or MODEL_DIR)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    sel_path = out_root / "selection.parquet"
    sel = pd.read_parquet(sel_path) if sel_path.exists() else make_selection(sel_path)
    if args.select_only:
        return 0
    if args.limit:
        half = args.limit // 2
        sel = sel[sel.order < half]
    sel = sel.sort_values(["order", "group"]).reset_index(drop=True)

    import torch

    proc, model, lm, tmpl = load(args.model_dir)
    tok = proc.tokenizer
    layers = list(range(lm.n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    if layers[-1] != lm.n_layers - 1:
        raise ValueError("the stored layers must include the last block")
    tr = pd.read_parquet(TRACES, columns=["model", "case", "image_idx", "prompt", "reasoning", "response",
                                          "prompt_tokens", "completion_tokens", "cut"])
    tr = tr[(tr.model == MODEL) & (tr.case == CASE)].set_index("image_idx")

    out_dir = out_root / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps({
        "model": MODEL, "model_dir": args.model_dir, "case": CASE, "lens": "logit (final norm, lm_head, softcap)",
        "layers": layers, "k": args.k, "n_answers": len(sel), "seed": SEED, "traces": str(TRACES),
        "completion_format": "<|channel>thought\\n{reasoning}<channel|>{response}<turn|>",
        "ws_variants": WS, "positions": "every position that predicts a completion token",
    }, indent=2))
    chunks = [sel.iloc[i:i + args.chunk_answers] for i in range(0, len(sel), args.chunk_answers)]
    for ci, chunk in enumerate(chunks):
        part = out_dir / f"part-{ci:04d}.parquet"
        meta_part = out_dir / f"answers-{ci:04d}.parquet"
        if part.exists() and meta_part.exists():
            continue
        t_chunk = time.time()
        rows, metas = [], []
        for s in chunk.itertuples():
            t0 = time.time()
            r = tr.loc[s.image_idx]
            assert not r.cut
            src = s.image_path
            if args.image_root and (Path(args.image_root) / f"{s.sample_id}.jpg").exists():
                src = str(Path(args.image_root) / f"{s.sample_id}.jpg")
            prompt = build_prompt(proc, tmpl, src, r.prompt)
            plen = int(prompt["input_ids"].shape[1])
            if plen != int(r.prompt_tokens):
                raise ValueError(f"image {s.image_idx}: HF prompt {plen} tokens, served {r.prompt_tokens}")
            comp, fields, ws_fix, diff = build_completion(tok, r.reasoning, r.response, int(r.completion_tokens))
            from lens_common import append_tokens
            inputs = append_tokens(prompt, comp)
            pos, tgt, arrs, agree_model = read_answer(model, lm, inputs, plen, layers, args.k)
            assert len(tgt) == len(comp) and (tgt == np.array(comp)).all()
            rows.append((int(s.image_idx), pos, tgt, fields, arrs))
            final_top1 = arrs[2][:, -1] == 0
            metas.append({
                "image_idx": int(s.image_idx), "group": s.group, "hf_prompt_tokens": plen,
                "served_prompt_tokens": int(r.prompt_tokens), "replay_completion_tokens": len(comp),
                "served_completion_tokens": int(r.completion_tokens), "count_diff": int(diff), "ws_fix": ws_fix,
                "n_reasoning": int((fields == FIELD_REASONING).sum()), "n_response": int((fields == FIELD_RESPONSE).sum()),
                "n_delim": int((fields == FIELD_DELIM).sum()),
                "final_top1_rate": float(final_top1.mean()),
                "final_top1_rate_reasoning": float(final_top1[fields == FIELD_REASONING].mean()),
                "final_top1_rate_response": float(final_top1[fields == FIELD_RESPONSE].mean()),
                "lens_vs_model_top1": agree_model, "seconds": time.time() - t0,
            })
            del inputs, prompt
            torch.cuda.empty_cache()
        write_chunk(part, rows)
        m = pd.DataFrame(metas)
        m.to_parquet(meta_part, index=False)
        log(f"chunk {ci + 1}/{len(chunks)}: {len(m)} answers in {time.time() - t_chunk:.0f}s "
            f"({m.seconds.mean():.1f}s each); count agrees {int((m.count_diff == 0).sum())}/{len(m)}, "
            f"off by 1 {int((m.count_diff.abs() == 1).sum())}, ws fixes {int((m.ws_fix != '||').sum())}; "
            f"final top-1 = written {m.final_top1_rate.mean():.3f}; lens = model logits {m.lens_vs_model_top1.mean():.4f}")
    allm = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("answers-*.parquet"))], ignore_index=True)
    allm.to_parquet(out_dir / "answers.parquet", index=False)
    log(f"done: {len(allm)} answers; count agrees {int((allm.count_diff == 0).sum())}, "
        f"|diff|=1 {int((allm.count_diff.abs() == 1).sum())}, other {int((allm.count_diff.abs() > 1).sum())}; "
        f"final top-1 = written {allm.final_top1_rate.mean():.3f}; {allm.seconds.mean():.1f}s per answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
