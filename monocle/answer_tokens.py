"""What the answer position is disposed to say, at every depth.

This is the open-vocabulary readout. `safety_workspace.py` reads the answer
position too, but it only logs what it was told to look for: the production
label's probability, a restricted argmax, and the mass on two fixed word lists.
It never records WHICH tokens the transport actually carries. This module does.

    logits_l[answer] = unembed( J_l @ h_l[answer] )    l in the fitted layers
    logits_47[answer] = the model's own logits         (exact, no transport)

then top-k over the full vocabulary at every layer.

The question it answers
-----------------------
"Given a query such as 'which station entrance looks safer?', what tokens are
in the J-space BEFORE the final output?"

Read the L42-L47 window. Two earlier results bound what the rest can mean:

  - Under the production prompt the label token carries no mass at all through
    L36 (0.000), then 0.066 at L42, 0.254 at L46, 0.770 at L47. Before L42 the
    answer is not in the channel.
  - Layers at or below L30 are corpus-dominated. The wikitext and urban lenses
    emit near-disjoint token sets there (top-10 Jaccard about 0). An early
    layer reads out its fitting corpus, not the image.

So a token at L6 is an artefact of the instrument. A token at L42 is a finding.

Outputs (per case, under `--out-dir`)
-------------------------------------
| File | One row per |
|---|---|
| `answer_tokens.parquet` | (pair, cond, pos, layer, rank) — the top-k tokens |
| `answer_labels.parquet` | (pair, cond, pos, layer) — the label-class metrics |
| `summary.json` | the corpus aggregation: most frequent tokens per layer |

Both frames carry `case`, so the per-case files concatenate.

Prompts, pairs, and labels all come from the canonical run registry through
`monocle.canonical`. See vlm-narratives-docs/canonical-lens-prompt-path.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import canonical, extract, scoring  # noqa: E402
from monocle import jlens_read  # noqa: E402  (jlens imported lazily inside)
from monocle import safety_workspace as sw  # noqa: E402

MODEL_DIR = "/share/pierson/matt/zoo/models/gemma-4-12B-it"
OUT_DEFAULT = REPO / "outputs/_monocle/answer_tokens"

#: The reference instrument. The wikitext lens is the one whose readout
#: semantics match a same-position read; the mm lens is the right transport for
#: "which patches feed the answer" (that is step 3, not this module).
LENS_DEFAULT = str(REPO / "outputs/_monocle/jlens/gemma4_12b_lens.pt")

FITTED_LAYERS = [6, 12, 18, 24, 30, 36, 42, 46]
FINAL_LAYER = 47

#: Read positions. `label` emits the answer token. `last` is the final real
#: prompt token, before the forced JSON prefix.
POSITIONS = ("label", "last")

DEFAULT_TOPK = 20
SEED = 777


def log(m: str) -> None:
    print(f"[answer-tokens] {m}", flush=True)


# ---------------------------------------------------------------------------
# Top-k extraction (pure; CPU-testable)
# ---------------------------------------------------------------------------
def topk_tokens(
    logits_row: torch.Tensor,
    tokenizer: Any,
    k: int,
    word_mask: Optional[torch.Tensor] = None,
) -> list[dict]:
    """The k highest-probability tokens of one [vocab] logits row.

    Returns raw top-k over the WHOLE vocabulary — no filter. At the answer
    position the leading tokens are often JSON syntax or a label, and hiding
    them would misrepresent the channel. `word_mask` only annotates: each row
    gets `is_word`, so a readable view is a query (`df[df.is_word]`), not a
    different run.

    `prob` is the softmax over the full vocabulary, so probabilities are
    comparable across layers and across the word/non-word split.
    """
    if logits_row.ndim != 1:
        raise ValueError(f"expected a 1-D logits row, got {tuple(logits_row.shape)}")
    probs = torch.softmax(logits_row.float(), dim=-1)
    top = torch.topk(probs, k)
    ids = top.indices.tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    out = []
    for rank, (tid, tok) in enumerate(zip(ids, tokens)):
        out.append({
            "rank": rank,
            "token_id": int(tid),
            "token": tok,
            "display": scoring.display_form(tok if tok is not None else ""),
            "prob": float(probs[tid]),
            "logit": float(logits_row[tid]),
            "is_word": bool(word_mask[tid]) if word_mask is not None else None,
        })
    return out


def cached_token_mask(
    tokenizer: Any, vocab: int, cache_dir: Path,
) -> torch.Tensor:
    """`scoring.build_token_mask`, cached to disk.

    The mask is a pure function of the tokenizer, but it walks all 262,144
    vocabulary entries in Python and takes minutes on a contended node. Every
    shard of a scale run would otherwise pay that before its first pair. The
    cache key is the vocab size plus the tokenizer class name; a mismatch
    rebuilds rather than returning a stale mask.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{type(tokenizer).__name__}_{vocab}"
    p = cache_dir / f"token_mask_{key}.pt"
    if p.is_file():
        mask = torch.load(p)
        if mask.numel() == vocab:
            log(f"token mask from cache ({p.name})")
            return mask
        log(f"cached mask has {mask.numel()} entries, not {vocab} — rebuilding")
    t0 = time.time()
    mask = scoring.build_token_mask(tokenizer, vocab)
    log(f"built token mask in {time.time() - t0:.0f}s -> {p.name}")
    tmp = p.with_suffix(f".tmp{os.getpid()}")
    torch.save(mask, tmp)
    os.replace(tmp, p)
    return mask


def entropy(logits_row: torch.Tensor) -> float:
    """Shannon entropy in nats of the full-vocabulary distribution.

    A layer whose readout is diffuse has high entropy; the channel narrowing
    with depth is exactly the effect this study looks for.
    """
    logp = torch.log_softmax(logits_row.float(), dim=-1)
    return float(-(logp.exp() * logp).sum())


def top_mass(logits_row: torch.Tensor, k: int) -> float:
    """Probability mass held by the top k tokens — a concentration measure."""
    probs = torch.softmax(logits_row.float(), dim=-1)
    return float(torch.topk(probs, k).values.sum())


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------
def parse_shard(spec: Optional[str]) -> tuple[int, int]:
    """'i/n' -> (i, n), validated. None -> (0, 1)."""
    if not spec:
        return 0, 1
    try:
        i_s, n_s = spec.split("/")
        i, n = int(i_s), int(n_s)
    except ValueError as exc:
        raise ValueError(f"bad --shard {spec!r}; expected 'i/n'") from exc
    if n < 1 or not (0 <= i < n):
        raise ValueError(f"bad --shard {spec!r}; need 0 <= i < n and n >= 1")
    return i, n


def shard_rows(rows: list, shard: int, n_shards: int) -> list:
    """Strided slice, so every shard sees the same case and label mix."""
    return rows[shard::n_shards]


# ---------------------------------------------------------------------------
# Checkpoint parts — constant-cost resume
# ---------------------------------------------------------------------------
def part_path(base: Path, n: int) -> Path:
    return base.parent / f"{base.stem}.part{n:04d}.parquet"


def write_part(rows: list[dict], base: Path, n: int) -> None:
    """Write one checkpoint part: only the rows since the last checkpoint.

    The first implementation re-concatenated every prior row and rewrote the
    whole parquet each time. Checkpoint cost then grew with the frame, and the
    measured rate drifted from 1.51 to 1.88 s/pair over 350 pairs (job 203559).
    A part file makes the cost constant.
    """
    if not rows:
        return
    sw.write_long_frame(pd.DataFrame(rows), part_path(base, n))


def read_parts(base: Path) -> pd.DataFrame:
    """Every checkpoint part plus the merged file, concatenated."""
    frames = []
    if base.exists():
        frames.append(pd.read_parquet(base))
    for p in sorted(base.parent.glob(f"{base.stem}.part*.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def merge_parts(base: Path) -> pd.DataFrame:
    """Fold the parts into the final file, then remove them.

    The merged file is written atomically before any part is unlinked, so a
    kill during the merge never loses a completed pair.
    """
    df = read_parts(base)
    if len(df):
        sw.write_long_frame(df, base)
        for p in sorted(base.parent.glob(f"{base.stem}.part*.parquet")):
            p.unlink()
    return df


def completed_pairs(base: Path) -> set:
    """pair_ids already written, across the merged file and every part."""
    done: set = set()
    paths = ([base] if base.exists() else []) + sorted(
        base.parent.glob(f"{base.stem}.part*.parquet"))
    for p in paths:
        done |= set(pd.read_parquet(p, columns=["pair_id"])["pair_id"].unique())
    return done


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------
def run_case(
    case: str, kind: str, proc, model, tmpl, lens, lens_model,
    *, n_pairs: int, conds: Optional[list[str]], layers: list[int],
    k: int, out_dir: Path, device: str, word_mask: torch.Tensor,
    positions: list[str], shard: int, n_shards: int, seed: int, smoke: bool,
    force_prefix: str = canonical.FORCE_PREFIX,
) -> dict:
    """Read one case and write its two long frames. Resumes by pair_id."""
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".shard{shard}of{n_shards}" if n_shards > 1 else ""
    tok_pq = case_dir / f"answer_tokens{suffix}.parquet"
    lab_pq = case_dir / f"answer_labels{suffix}.parquet"

    conditions = canonical.build_conditions(case, kind, conds)
    tokenizer = proc.tokenizer
    label_first, class_ids, _ = canonical.label_classes(
        tokenizer, conditions["prod"])

    rows = sw.load_pairs(n_pairs, case, kind, seed=seed)
    rows = shard_rows(rows, shard, n_shards)
    done = completed_pairs(tok_pq)
    if done:
        log(f"{case}: resume — {len(done)} pairs already written")
    n_part = len(list(tok_pq.parent.glob(f"{tok_pq.stem}.part*.parquet")))

    record_at = [l for l in layers if l != FINAL_LAYER]
    all_layers = sorted(set(layers) | {FINAL_LAYER})
    n_suffix = len(tokenizer.encode(force_prefix, add_special_tokens=False))

    new_tok: list[dict] = []
    new_lab: list[dict] = []
    t0 = time.time()
    n_done = 0
    for row in rows:
        pid = row["pair_id"]
        if pid in done:
            continue
        prod_label = row["presented_label"]
        left, right = canonical.presented_images(row)
        imgs = [Image.open(left).convert("RGB"),
                Image.open(right).convert("RGB")]

        for cond, cfg in conditions.items():
            inputs = canonical.build_pair_inputs(
                proc, tmpl, imgs, row, cfg, force_prefix=force_prefix,
                device=device)
            seq_len = int(inputs["input_ids"].shape[1])
            rp = sw.read_positions(seq_len, n_suffix)
            pos_idx = torch.tensor([rp[p] for p in positions], device=device)
            activations, final_logits = sw.record_activations(
                model, lens_model, inputs, record_at, logit_positions=pos_idx)

            per_layer = jlens_read.lens_patch_logits(
                lens, lens_model, activations, pos_idx, final_logits,
                layers=all_layers)

            for layer in all_layers:
                rows_l = per_layer[layer]  # [n_pos, vocab]
                for j, posname in enumerate(positions):
                    lr = rows_l[j]
                    base = {"case": case, "cond": cond, "pair_id": pid,
                            "pos": posname, "layer": int(layer)}
                    for t in topk_tokens(lr, tokenizer, k, word_mask):
                        new_tok.append({**base, **t})
                    # No word-list probes here: this module tallies the
                    # tokens the transport actually carries.
                    m = sw.position_metrics(
                        lr, prod_label, class_ids, label_first)
                    new_lab.append({
                        **base, "prod_label": prod_label,
                        "prod_class": sw.collapse_label(prod_label),
                        "entropy": entropy(lr), "top_mass": top_mass(lr, k),
                        **m})
        n_done += 1
        if smoke or n_done % 25 == 0:
            rate = (time.time() - t0) / max(n_done, 1)
            log(f"  {case}: {n_done} pairs | {rate:.2f}s/pair")
        # Checkpoint every 25 pairs — a pre-emption then costs at most 25.
        # Only the new rows are written, so the cost does not grow.
        if n_done % 25 == 0:
            write_part(new_tok, tok_pq, n_part)
            write_part(new_lab, lab_pq, n_part)
            new_tok, new_lab = [], []
            n_part += 1

    write_part(new_tok, tok_pq, n_part)
    write_part(new_lab, lab_pq, n_part)
    tok_df = merge_parts(tok_pq)
    lab_df = merge_parts(lab_pq)
    log(f"{case}: {len(tok_df)} token rows, {len(lab_df)} label rows "
        f"-> {case_dir}")
    return summarize_case(case, tok_df, lab_df, conditions, case_dir, suffix)


# ---------------------------------------------------------------------------
# Corpus aggregation — the direct answer to "what tokens are in there"
# ---------------------------------------------------------------------------
def top_tokens_by_layer(
    tok_df: pd.DataFrame, cond: str, pos: str = "label",
    n: int = 15, words_only: bool = False,
) -> dict[int, list[dict]]:
    """The most frequent top-1 tokens per layer, over the corpus.

    A per-pair top-k answers "what did THIS image make it say". The corpus
    tally answers "what does this layer say at all", which is the question the
    study asks.
    """
    df = tok_df[(tok_df["cond"] == cond) & (tok_df["pos"] == pos)
                & (tok_df["rank"] == 0)]
    if words_only:
        df = df[df["is_word"] == True]  # noqa: E712 (pandas mask)
    out: dict[int, list[dict]] = {}
    for layer, g in df.groupby("layer"):
        counts = Counter(g["display"])
        total = max(len(g), 1)
        out[int(layer)] = [
            {"token": t, "count": int(c), "frac": c / total,
             "mean_prob": float(g.loc[g["display"] == t, "prob"].mean())}
            for t, c in counts.most_common(n)]
    return out


def summarize_case(
    case: str, tok_df: pd.DataFrame, lab_df: pd.DataFrame,
    conditions: dict, case_dir: Path, suffix: str = "",
) -> dict:
    """Per-(cond, pos, layer) label metrics plus the corpus token tally."""
    by_group = []
    if len(lab_df):
        for (cond, pos, layer), g in lab_df.groupby(["cond", "pos", "layer"]):
            scored = np.array([
                sw.collapsed_ordinal_agreement(pc, pl)
                for pc, pl in zip(g["argmax_class"], g["prod_label"])],
                dtype=float)
            n_ord = int(np.count_nonzero(~np.isnan(scored)))
            by_group.append({
                "cond": cond, "pos": pos, "layer": int(layer), "n": int(len(g)),
                "mean_p_prod_label": float(g["p_prod_label"].mean()),
                "median_rank_prod_label": float(g["rank_prod_label"].median()),
                "argmax_agreement": float(g["argmax_correct"].mean()),
                "ordinal_agreement_collapsed":
                    float(np.nanmean(scored)) if n_ord else float("nan"),
                "n_ordinal": n_ord,
                "read_abstain_rate": float((g["argmax_class"] == canonical.NOT_SURE).mean()),
                "label_abstain_rate": float((g["prod_label"] == canonical.NOT_SURE).mean()),
                "mean_entropy": float(g["entropy"].mean()),
                "mean_top_mass": float(g["top_mass"].mean()),
            })
    summary = {
        "case": case,
        "conditions": list(conditions),
        "question": canonical.user_text("<pair_id>", conditions["prod"]),
        "n_pairs": int(lab_df["pair_id"].nunique()) if len(lab_df) else 0,
        "by_group": by_group,
        "top_tokens": {
            cond: {
                "all": top_tokens_by_layer(tok_df, cond),
                "words_only": top_tokens_by_layer(tok_df, cond, words_only=True),
            } for cond in conditions
        } if len(tok_df) else {},
    }
    p = case_dir / f"summary{suffix}.json"
    p.write_text(json.dumps(summary, indent=2))
    log(f"{case}: wrote {p}")
    return summary


def print_case_table(summary: dict) -> None:
    """The L42-L47 window, per condition, at the label position."""
    log("=" * 78)
    log(f"CASE {summary['case']} — top-1 readout at the answer position")
    for cond, blocks in summary.get("top_tokens", {}).items():
        log(f"  {cond}")
        for layer in sorted(blocks["all"], key=int):
            toks = blocks["all"][layer][:6]
            cells = " ".join(
                f"{t['token']!r}:{t['frac']:.2f}" for t in toks)
            log(f"    L{layer:<3} {cells}")
    rows = [r for r in summary["by_group"] if r["pos"] == "label"]
    if rows:
        log("  label-class metrics @label")
        conds = sorted({r["cond"] for r in rows})
        layers = sorted({r["layer"] for r in rows})
        log("    layer : " + " ".join(f"L{l:<5}" for l in layers))
        for cond in conds:
            by_l = {r["layer"]: r for r in rows if r["cond"] == cond}
            log(f"    {cond[:7]:<7}" + " ".join(
                f"{by_l[l]['mean_p_prod_label']:.3f} " for l in layers))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=list(canonical.CASES),
                    choices=list(canonical.CASES),
                    help="Registered cases to read (default: all seven).")
    ap.add_argument("--kind", default="proxy", choices=list(canonical.KINDS),
                    help="Prefer proxy; trace enables thinking.")
    ap.add_argument("--n-pairs", type=int, default=1000)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="Default prod+neutral, plus axis where one exists.")
    ap.add_argument("--lens", default=LENS_DEFAULT,
                    help="Fitted lens .pt (default: the wikitext reference).")
    ap.add_argument("--layers", nargs="+", type=int, default=FITTED_LAYERS)
    ap.add_argument("--positions", nargs="+", default=list(POSITIONS),
                    choices=list(POSITIONS))
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--force-prefix", default="natural",
                    choices=["natural", "compact"],
                    help=("Teacher-forced JSON prefix. 'natural' (default) "
                          "matches the newline+indent the model itself "
                          "writes. 'compact' reproduces the rung-B run and "
                          "reads a flat zero on most cases — see "
                          "canonical.FORCE_PREFIX_COMPACT."))
    ap.add_argument("--shard", default=None, metavar="i/n",
                    help="Strided shard of the pair list, for array jobs.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true",
                    help="2 pairs per case, verbose.")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    shard, n_shards = parse_shard(args.shard)
    force_prefix = (canonical.FORCE_PREFIX if args.force_prefix == "natural"
                    else canonical.FORCE_PREFIX_COMPACT)
    n_pairs = 2 if args.smoke else args.n_pairs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "trace":
        log("WARNING kind=trace enables thinking — the answer position sits "
            "after a sampled reasoning block")

    proc, model, tmpl = extract.load_model(args.model_dir, device=args.device)
    tokenizer = proc.tokenizer
    lens_model = jlens_read.wrap_for_unembed(model, tokenizer)
    lens = jlens_read.load_lens(args.lens)
    layers = sorted(set(args.layers) & set(lens.source_layers))
    log(f"lens {args.lens} | layers {layers} + final {FINAL_LAYER}")

    vocab = int(model.config.text_config.vocab_size
                if hasattr(model.config, "text_config")
                else model.config.vocab_size)
    word_mask = cached_token_mask(tokenizer, vocab, REPO / "cache/monocle")
    log(f"vocab {vocab} | {int(word_mask.sum())} word-like tokens")

    summaries = {}
    for case in args.cases:
        s = run_case(
            case, args.kind, proc, model, tmpl, lens, lens_model,
            n_pairs=n_pairs, conds=args.conditions, layers=layers,
            k=args.topk, out_dir=out_dir, device=args.device,
            word_mask=word_mask, positions=args.positions,
            shard=shard, n_shards=n_shards, seed=args.seed, smoke=args.smoke,
            force_prefix=force_prefix)
        summaries[case] = s
        print_case_table(s)

    idx = out_dir / (f"index.shard{shard}of{n_shards}.json" if n_shards > 1
                     else "index.json")
    idx.write_text(json.dumps({
        "cases": args.cases, "kind": args.kind, "lens": args.lens,
        "force_prefix": force_prefix, "force_prefix_mode": args.force_prefix,
        "layers": layers, "final_layer": FINAL_LAYER, "topk": args.topk,
        "n_pairs": n_pairs, "seed": args.seed,
        "shard": shard, "n_shards": n_shards,
        "positions": args.positions,
        "n_pairs_read": {c: s["n_pairs"] for c, s in summaries.items()},
    }, indent=2))
    log(f"wrote {idx}")
    log("answer-token readout complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
