#!/usr/bin/env python3
"""Summaries of the whatami lens replay: what each layer suggests.

Reads multirun/whatami/lens_replay/<model>/<lens>/part-*.parquet (lens_replay.py)
one file at a time, and writes to multirun/whatami/lens_report/<model>/<lens>/:

| File | Content |
|---|---|
| layer_top1.csv | for each layer, the 25 tokens most often at top 1, over all answer positions |
| layer_top1_pos0.csv | the same at pos 0, the position that chooses the first answer token |
| settle_layer.csv | for each layer, how many answer tokens settle there: the token is top 1 from this layer to the last |
| place_latent.csv | for each place term, depth band (early, middle, late third of the layers below the last) and top1/top5: the share of answers that SAY the term, and the share where one of its tokens is in the top 1 / top 5 at any position in the band |
| summary.json | counts, and the files read |

A place term counts as latent in an answer, for a band, when one of its tokens
is in the top 1 (or top 5) at any position, at any layer of the band. Read it
against the control cities: there are many slots, thus a stray hit is common,
above all in the early layers. "York" and the borough
names are single tokens in these vocabularies (with or without a leading
space); a term that is not one token is skipped and logged. "New" alone is
not a term: too many answers use it in other senses.

Usage:
    python research/scripts/whatami/lens_report.py --model gemma-4-12b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import lens_common as lc

REPLAY = lc.REPO / "multirun/whatami/lens_replay"
REPORT = lc.REPO / "multirun/whatami/lens_report"
RUNS = lc.REPO / "multirun/whatami/sample_1k"

#: Term -> the strings whose tokens stand for it. The answer text "says" the
#: term when it contains the first string.
PLACES = {
    "York": ["York"], "NYC": ["NYC"], "Manhattan": ["Manhattan"], "Brooklyn": ["Brooklyn"],
    "Queens": ["Queens"], "Bronx": ["Bronx"], "Staten": ["Staten"],
    "Philadelphia": ["Philadelphia"], "Boston": ["Boston"], "Chicago": ["Chicago"],
    "London": ["London"], "Paris": ["Paris"], "Toronto": ["Toronto"],
}


#: Cities that the probe images do not show. Their latent rate is the chance
#: level: any position x any layer x top 5 gives many slots for a stray hit.
CONTROLS = {"Philadelphia", "Boston", "Chicago", "London", "Paris", "Toronto"}


def bands(n_layers: int) -> dict[str, tuple[int, int]]:
    """Early, middle and late thirds of the layers below the last: [lo, hi)."""
    m = n_layers - 1
    a, b = round(m / 3), round(2 * m / 3)
    return {"early": (0, a), "middle": (a, b), "late": (b, m)}


def single_token_ids(tok, word: str) -> list[int]:
    """Ids of `word` and ` word` that encode to one token."""
    out = []
    for w in (word, " " + word):
        ids = tok(w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            out.append(int(ids[0]))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(lc.MODELS))
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--replay-root", default=str(REPLAY))
    ap.add_argument("--out-root", default=str(REPORT))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    spec = lc.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(spec.path)
    vocab = len(tok)
    answers = pd.read_parquet(RUNS / args.model / "responses.parquet", columns=["image_idx", "repeat", "response"])
    said = {term: dict(zip(zip(answers.image_idx, answers.repeat), answers.response.str.contains(words[0])))
            for term, words in PLACES.items()}
    place_ids = {t: single_token_ids(tok, w[0]) for t, w in PLACES.items()}
    skipped = [t for t, ids in place_ids.items() if not ids]
    if skipped:
        lc.log(f"not one token in this vocabulary, skipped: {skipped}")
    place_ids = {t: ids for t, ids in place_ids.items() if ids}

    root = Path(args.replay_root) / args.model
    lenses = sorted(p.name for p in root.iterdir() if p.is_dir())
    for lens in lenses:
        parts = sorted((root / lens).glob("part-*.parquet"))
        if not parts:
            continue
        top1 = top1_pos0 = settle = None
        # (term, band, depth) -> the answers with a hit; depth is top1 or top5.
        latent: dict[tuple, set] = {}
        keys: set = set()
        n_rows = 0
        for part in parts:
            t = pq.read_table(part)
            ids = t.column("topk_ids").combine_chunks().flatten().flatten().to_numpy()
            rank = t.column("target_rank").combine_chunks().flatten().to_numpy()
            n = t.num_rows
            n_layers = rank.size // n
            k = ids.size // (n * n_layers)
            ids = ids.reshape(n, n_layers, k)
            rank = rank.reshape(n, n_layers)
            pos = t.column("pos").to_numpy()
            img = t.column("image_idx").to_numpy()
            rep = t.column("repeat").to_numpy()
            if top1 is None:
                top1 = np.zeros((n_layers, vocab), np.int64)
                top1_pos0 = np.zeros((n_layers, vocab), np.int64)
                settle = np.zeros(n_layers + 1, np.int64)
            first = ids[:, :, 0]
            for l in range(n_layers):
                col = first[:, l]
                ok = col >= 0
                top1[l] += np.bincount(col[ok], minlength=vocab)[:vocab]
                c0 = col[ok & (pos == 0)]
                top1_pos0[l] += np.bincount(c0, minlength=vocab)[:vocab]
            # Settle layer: the first layer from which the rank stays 0 to the last.
            valid = rank >= 0
            at_top = (rank == 0) | ~valid  # a layer with no readout does not break the run
            tail = np.flip(np.cumprod(np.flip(at_top, 1), 1), 1).astype(bool)
            settled = np.where(tail[:, -1] & (rank[:, -1] == 0), n_layers - tail.sum(1), n_layers)
            settle += np.bincount(settled, minlength=n_layers + 1)
            for band, (lo, hi) in bands(n_layers).items():
                sub = ids[:, lo:hi, :]
                for term, tids in place_ids.items():
                    m = np.isin(sub, tids)
                    for depth, hit in (("top5", m.any(axis=(1, 2))), ("top1", m[:, :, 0].any(axis=1))):
                        latent.setdefault((term, band, depth), set()).update(
                            zip(img[hit].tolist(), rep[hit].tolist()))
            keys.update(zip(img.tolist(), rep.tolist()))
            n_rows += n

        out = Path(args.out_root) / args.model / lens
        out.mkdir(parents=True, exist_ok=True)
        dec = lambda i: tok.convert_ids_to_tokens(int(i))  # noqa: E731
        for name, arr in (("layer_top1", top1), ("layer_top1_pos0", top1_pos0)):
            rows = []
            for l in range(arr.shape[0]):
                tot = arr[l].sum()
                for i in np.argsort(arr[l])[::-1][: args.top]:
                    if arr[l, i] == 0:
                        break
                    rows.append({"layer": l, "token": dec(i), "token_id": int(i),
                                 "count": int(arr[l, i]), "share": float(arr[l, i] / max(tot, 1))})
            pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
        pd.DataFrame({"layer": list(range(len(settle) - 1)) + ["never"], "n_tokens": settle}).to_csv(
            out / "settle_layer.csv", index=False)
        n_ans = len(keys)
        rows = []
        for (term, band, depth), lat in sorted(latent.items()):
            s = sum(bool(said[term].get(k_, False)) for k_ in keys)
            rows.append({"term": term, "control": term in CONTROLS, "band": band,
                         "layers": "%d-%d" % (bands(top1.shape[0])[band][0], bands(top1.shape[0])[band][1] - 1),
                         "depth": depth, "answers": n_ans, "said": s / n_ans, "latent": len(lat) / n_ans,
                         "latent_not_said": sum(1 for k_ in lat if not said[term].get(k_, False)) / n_ans})
        pd.DataFrame(rows).to_csv(out / "place_latent.csv", index=False)
        (out / "summary.json").write_text(json.dumps(
            {"model": args.model, "lens": lens, "parts": len(parts), "rows": n_rows, "answers": n_ans,
             "n_layers": int(top1.shape[0])}, indent=2))
        lc.log(f"{args.model}/{lens}: {len(parts)} parts, {n_rows} positions, {n_ans} answers -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
