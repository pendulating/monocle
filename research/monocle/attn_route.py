"""Where the answer position looks, at the global blocks.

`patch_attribution` infers that the model routes each image block to its LAST
patch: the gradient there is 117x every other patch at L46. That is an
inference from a gradient. This module records the attention itself.

Why block 47 is the one to read
-------------------------------
`c` reads `h_47[answer]` and nothing else. For a PATCH the only surviving path
to `c` is block 47's attention from the answer position back to that patch.
Every other route is closed.

Block 47 is also a GLOBAL block, and in gemma-4-12b a global block is a
different mechanism from a sliding one:

    num_global_key_value_heads = 1     one KV head, shared by 16 query heads
    attention_k_eq_v = true            no v_proj — the keys ARE the values
    global_head_dim = 512              twice the sliding width

One shared key/value set means the 16 heads read the SAME vectors and differ
only by their query. Thus the attention row is unusually simple to read.

See `research/docs/gemma4-12b-architecture.md`.

Output
------
`attn_route.parquet`, one row per (pair, block, head, position group), with
the attention mass the answer position puts on that group.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import canonical  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import safety_workspace as sw  # noqa: E402

OUT_DEFAULT = REPO / "outputs/_monocle/attn_route"
BOI_TOKEN_ID = 255999
EOI_TOKEN_ID = 258882
GLOBAL_BLOCKS = [5, 11, 17, 23, 29, 35, 41, 47]
# Two sliding blocks for contrast, and every global block.
BLOCKS = sorted(set(GLOBAL_BLOCKS) | {42, 46})


def log(m: str) -> None:
    print(f"[attn-route] {m}", flush=True)


def load_eager(model_dir: str, device: str):
    """Load with eager attention so the weights are returned.

    sdpa and flash-attention never materialize the attention matrix, so
    `output_attentions=True` is silently useless with them.
    """
    from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration

    from dagspaces.common.model_registry import resolve_model_source
    from dagspaces.common.vllm_inference import _gemma4_unified_chat_template

    is_g4u, tmpl = _gemma4_unified_chat_template(model_dir)
    if not is_g4u or not tmpl:
        raise RuntimeError(f"{model_dir} is not gemma4_unified")
    # 24 GB of weights off NFS is minutes when the mount is busy, and the
    # pairvqa arrays keep it busy. The mirror is used only when its marker
    # names this source, so a missing mirror falls back to the zoo path.
    src = resolve_model_source(model_dir, stage_name="attn-route")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(src)
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        src, dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager")
    model.eval()
    log(f"model loaded eager in {time.time() - t0:.0f}s "
        f"from {src}")
    return proc, model, tmpl


def group_positions(
    blocks: list[torch.Tensor], seq_len: int,
    input_ids: Optional[torch.Tensor] = None,
) -> dict:
    """Named position groups over the sequence.

    `A_last` / `B_last` are the final patch of each image block — the cell the
    gradient says the model aggregates into.

    `boi` and `eoi` get their own groups. They wrap each image block, and
    `mm_token_type_ids` marks them as TEXT, so they attend causally over the
    whole bidirectional island behind them. That is the shape of a pooling
    token, and an earlier version of this module hid them inside `text`.
    """
    a, b = [t.tolist() for t in blocks]
    # The groups must PARTITION the sequence, or the summary double counts.
    # `*_rest` therefore drops both the first and the last patch, since both
    # are reported on their own.
    groups = {
        "A_last": [a[-1]], "B_last": [b[-1]],
        "A_first": [a[0]], "B_first": [b[0]],
        "A_rest": a[1:-1], "B_rest": b[1:-1],
    }
    img = set(a) | set(b)
    delim: list[int] = []
    if input_ids is not None:
        ids = input_ids.tolist()
        boi = [i for i, t in enumerate(ids) if t == BOI_TOKEN_ID]
        eoi = [i for i, t in enumerate(ids) if t == EOI_TOKEN_ID]
        if boi:
            groups["boi"] = boi
        if eoi:
            groups["eoi"] = eoi
        delim = boi + eoi
    rest = set(img) | set(delim)
    groups["text"] = [i for i in range(seq_len) if i not in rest]
    return groups


def check_partition(groups: dict, seq_len: int) -> None:
    """Every position belongs to exactly 1 group, or the summary lies."""
    seen: list[int] = []
    for pos in groups.values():
        seen.extend(pos)
    if len(seen) != len(set(seen)):
        raise RuntimeError("position groups overlap")
    if len(seen) != seq_len:
        raise RuntimeError(f"groups cover {len(seen)} of {seq_len} positions")


@torch.no_grad()
def query_attention(
    model: Any, inputs: dict, queries: dict[str, int], blocks: list[int],
) -> dict[str, dict[int, torch.Tensor]]:
    """Attention FROM each named query position, per block.

    The answer position tells us how the model reads the image at the end.
    A query at `A_last` tells us how that patch FILLS UP, which is the claim
    that the answer row cannot test. Vision is bidirectional, so `A_last` can
    read every patch of its own block.
    """
    out = model(**inputs, use_cache=False, output_attentions=True,
                logits_to_keep=1)
    att = out.attentions
    if att is None or att[0] is None:
        raise RuntimeError(
            "no attention weights returned — the model is not using eager "
            "attention")
    return {name: {b: att[b][0, :, pos, :].float().cpu() for b in blocks}
            for name, pos in queries.items()}


def rows_for(
    att: dict[int, torch.Tensor], groups: dict, base: dict,
    query: str = "answer",
) -> list[dict]:
    rows = []
    for block, a in att.items():
        n_heads = a.shape[0]
        for name, pos in groups.items():
            if not pos:
                continue
            mass = a[:, pos].sum(-1)
            for h in range(n_heads):
                rows.append({**base, "query": query, "block": block,
                             "head": h, "group": name,
                             "n_pos": len(pos), "mass": float(mass[h])})
    return rows


def run_case(
    case: str, kind: str, proc: Any, model: Any, tmpl: str, *,
    n_pairs: int, blocks: list[int], out_dir: Path, device: str, seed: int,
) -> pd.DataFrame:
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    cfg = canonical.build_conditions(case, kind, ["prod"])["prod"]
    image_token_id = int(model.config.image_token_id)
    n_suffix = len(proc.tokenizer.encode(
        canonical.FORCE_PREFIX, add_special_tokens=False))
    rows: list[dict] = []
    for i, row in enumerate(sw.load_pairs(n_pairs, case, kind, seed=seed)):
        left, right = canonical.presented_images(row)
        imgs = [Image.open(left).convert("RGB"),
                Image.open(right).convert("RGB")]
        inputs = canonical.build_pair_inputs(
            proc, tmpl, imgs, row, cfg, device=device)
        seq_len = int(inputs["input_ids"].shape[1])
        answer_pos = sw.read_positions(seq_len, n_suffix)["label"]
        blk = sw.image_token_blocks(inputs, image_token_id)
        groups = group_positions(blk, seq_len, inputs["input_ids"][0])
        if i == 0:
            check_partition(groups, seq_len)
        # A_mid is the CONTROL. Without it, "the last patch reads its block
        # broadly at blocks 0-9" says nothing: an ordinary patch may do the
        # same, and then the breadth is not evidence of aggregation.
        queries = {"answer": answer_pos,
                   "A_last": int(blk[0][-1]), "B_last": int(blk[1][-1]),
                   "A_mid": int(blk[0][len(blk[0]) // 2]),
                   "B_mid": int(blk[1][len(blk[1]) // 2])}
        att = query_attention(model, inputs, queries, blocks)
        base = {"case": case, "pair_id": row["pair_id"]}
        for qname, per_block in att.items():
            rows.extend(rows_for(per_block, groups, base, query=qname))
        if (i + 1) % 5 == 0:
            log(f"{case}: {i + 1} pairs")
    df = pd.DataFrame(rows)
    df.to_parquet(case_dir / "attn_route.parquet", index=False)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean attention mass for each (block, group), summed over heads."""
    per = (df.groupby(["case", "pair_id", "query", "block", "group"])
             .mass.sum().reset_index())
    n_heads = df["head"].nunique()
    per["mass"] /= n_heads          # back to a per-head average
    return per.groupby(["query", "block", "group"]).mass.mean().unstack()


def summary_json(s: pd.DataFrame) -> dict:
    """Nest the (query, block) index, since JSON keys cannot be tuples.

    `DataFrame.to_dict` puts the index tuple 1 level DOWN, under each column,
    so converting the outer keys does not help.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for (query, block), row in s.iterrows():
        out.setdefault(str(query), {})[str(block)] = {
            g: float(v) for g, v in row.items() if pd.notna(v)}
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["subway_safety"],
                    choices=list(canonical.CASES))
    ap.add_argument("--kind", default="proxy", choices=list(canonical.KINDS))
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--blocks", nargs="+", type=int, default=BLOCKS,
                    help="Use --blocks -1 for every block, 0 to 47.")
    ap.add_argument("--seed", type=int, default=pa.SEED)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    proc, model, tmpl = load_eager(args.model_dir, args.device)
    for case in args.cases:
        blocks = (list(range(48)) if args.blocks == [-1]
                  else sorted(set(args.blocks)))
        df = run_case(case, args.kind, proc, model, tmpl,
                      n_pairs=args.n_pairs, blocks=blocks,
                      out_dir=out_dir, device=args.device, seed=args.seed)
        s = summarize(df)
        cols = [c for c in ["A_last", "B_last", "A_first", "B_first",
                            "A_rest", "B_rest", "boi", "eoi", "text"]
                if c in s.columns]
        for qname in s.index.get_level_values("query").unique():
            q = s.xs(qname, level="query")
            log("=" * 82)
            log(f"CASE {case} — attention mass FROM {qname} "
                f"(per head, {df['pair_id'].nunique()} pairs)")
            log("  block " + "".join(f"{c:>9}" for c in cols))
            for b in q.index:
                log(f"  {b:>5} " + "".join(
                    f"{q.loc[b, c]:>9.4f}" if c in q.columns else f"{'-':>9}"
                    for c in cols))
        s.reset_index().to_parquet(
            out_dir / case / "attn_summary.parquet", index=False)
        (out_dir / case / "summary.json").write_text(
            json.dumps(summary_json(s), indent=2))
    log("attention routing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
