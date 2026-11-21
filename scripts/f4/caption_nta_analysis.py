# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "numpy>=2.0", "rapidfuzz>=3.9"]
# ///
"""F4: do the models make the same captions in every NTA? (D101 to D105)

For each model, and for all models pooled, this script groups the captions and
the hashtags of the caption case by the NTA of the restaurant, then measures:

| Measure | Unit | What a low value says |
|---------|------|-----------------------|
| `word_entropy` | the words of all captions of the NTA | few words carry the captions |
| `hashtag_entropy` | the hashtags of the NTA (1 count per tag per answer) | few tags carry the hashtags |
| `nearest_between` | for each caption of NTA a, the edit distance to the nearest caption of NTA b; mean over the captions of a, then over every other NTA b | a caption of a is close to some caption of every other NTA |
| `allpairs_between` | the mean edit distance over all caption pairs of a and b; then the mean over every other NTA b | the captions of a and b are alike on average |
| `nearest_within`, `allpairs_within` | the same, inside NTA a; a pair of captions of the same answer does not count | the reference for the 2 rows above |

The edit distance is the Levenshtein distance on characters, divided by the
length of the longer caption (0 = the same text, 1 = nothing in common). The
text is lower case with 1 space between words; emojis and punctuation stay.

Entropy is in bits. It grows with the sample, and an NTA holds 1 to 38
restaurants, thus each entropy has 3 forms:

| Form | Column suffix |
|------|---------------|
| plug-in | `_plugin` |
| Miller-Madow | `_mm` |
| rarefied: the mean over `--draws` random draws of `k` tokens without replacement | `_rare` |

`k` is the same for every NTA of 1 model. By default (`--word-k 0`,
`--tag-k 0`) it is the smallest token count among the NTAs with at least
`--min-restaurants` restaurants, thus each such NTA gets a value. Compare a
rarefied value across the NTAs of 1 model, not across models: `k` differs.
An NTA with fewer tokens than `k` gets no rarefied value.

Inputs: `data/f4_traces/captions/f4_captions_final.parquet` (captions),
`data/f4_traces/captions/f4_caption_hashtags.parquet` (hashtags),
`data/f4_traces/f4_restaurants_nta.parquet` (NTAs).

Output (default `data/f4_traces/captions/nta_analysis/`):

| File | Content |
|------|---------|
| `nta_measures.parquet`, `.csv` | 1 row for each (model, NTA): sizes, entropies, distances |
| `nta_pairs.parquet` | 1 row for each (model, NTA a, NTA b): `nearest`, `allpairs` |
| `model_summary.csv` | for each model: the mean of each measure over NTAs with at least `--min-restaurants` restaurants |
| `manifest.json` | the inputs, the settings, the counts |

The street captions (D133): give the street caption extraction, the street
hashtags, and the street views (`f4_street_1k.parquet`, columns `nta2020`,
`nta_name`, `borough`) as `--ntas`. The column `restaurants` then counts
street views.

Use:
    uv run --script scripts/f4/caption_nta_analysis.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

ROOT = "/share/ju/matt/mllmsci"
CAPTIONS = os.path.join(ROOT, "data/f4_traces/captions/f4_captions_final.parquet")
HASHTAGS = os.path.join(ROOT, "data/f4_traces/captions/f4_caption_hashtags.parquet")
NTAS = os.path.join(ROOT, "data/f4_traces/f4_restaurants_nta.parquet")
OUT_DIR = os.path.join(ROOT, "data/f4_traces/captions/nta_analysis")
MODELS = ("gemma-4-12b", "gemma-4-31b", "qwen3.5-9b", "qwen3.8-27b", "llama-3.2-11b", "llama-3.2-90b")
POOLED = "all_models"
# A word is a run of letters, with an inner apostrophe (`it's`, `city's`).
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")

__version__ = "1.4.0"


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def entropy_plugin(counts: np.ndarray) -> float:
    n = counts.sum()
    if n == 0:
        return float("nan")
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())


def entropy_mm(counts: np.ndarray) -> float:
    """The Miller-Madow form: plug-in + (K - 1) / (2 N), in bits."""
    n = counts.sum()
    if n == 0:
        return float("nan")
    return entropy_plugin(counts) + (int((counts > 0).sum()) - 1) / (2 * n * math.log(2))


def entropy_rare(tokens: list, k: int, draws: int, rng: np.random.Generator) -> float:
    """The mean plug-in entropy of `draws` draws of `k` tokens without replacement."""
    if k <= 0 or len(tokens) < k:
        return float("nan")
    codes = pd.factorize(pd.Series(tokens))[0]
    out = []
    for _ in range(draws):
        pick = rng.choice(codes, size=k, replace=False)
        out.append(entropy_plugin(np.bincount(pick)))
    return float(np.mean(out))


def entropies(tokens: list, k: int, draws: int, rng: np.random.Generator) -> dict:
    counts = np.array(list(Counter(tokens).values()), dtype=float)
    return {"plugin": entropy_plugin(counts), "mm": entropy_mm(counts),
            "rare": entropy_rare(tokens, k, draws, rng), "tokens": len(tokens), "types": len(counts)}


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------


def distances(frame: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give the between-NTA pairs and the within-NTA values of 1 group of captions.

    `frame` holds `nta`, `answer`, `text`. The work goes block by block: 1 NTA
    against all captions, thus the memory is 1 block, not the full matrix.
    """
    frame = frame.sort_values("nta", kind="stable").reset_index(drop=True)
    texts = frame.text.tolist()
    ntas = frame.nta.to_numpy()
    answers = frame.answer.to_numpy()
    names, starts = np.unique(ntas, return_index=True)
    bounds = list(starts) + [len(frame)]
    pairs, within = [], []
    for a, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        block = cdist(texts[s:e], texts, scorer=Levenshtein.normalized_distance,
                      dtype=np.float32, workers=workers)
        nearest = np.minimum.reduceat(block, starts, axis=1)  # (captions of a, NTAs)
        total = np.add.reduceat(block, starts, axis=1)
        sizes = np.diff(bounds)
        for b in range(len(names)):
            if b == a:
                continue
            pairs.append((names[a], names[b], float(nearest[:, b].mean()), float(total[:, b].sum() / (block.shape[0] * sizes[b]))))
        # Within a: a pair of captions of the same answer does not count.
        inner = block[:, s:e].astype(np.float64)
        same = answers[s:e][:, None] == answers[s:e][None, :]
        inner[same] = np.nan
        if np.isfinite(inner).any():
            with np.errstate(all="ignore"):
                near = np.nanmin(inner, axis=1)
            within.append((names[a], float(np.nanmean(near)), float(np.nanmean(inner))))
        else:
            within.append((names[a], float("nan"), float("nan")))
    pair_df = pd.DataFrame(pairs, columns=["nta_a", "nta_b", "nearest", "allpairs"])
    within_df = pd.DataFrame(within, columns=["nta", "nearest_within", "allpairs_within"])
    return pair_df, within_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def clean(text: str) -> str:
    return " ".join(str(text).lower().split())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--captions", default=CAPTIONS,
                    help="the caption table; the x2 round (D130) is data/f4_traces/x2/captions/f4_captions_x2_extractions.parquet")
    ap.add_argument("--hashtags", default=HASHTAGS, help="the hashtag table (caption_hashtags.py)")
    ap.add_argument("--ntas", default=NTAS, help="the NTA of each restaurant image (restaurant_nta.py)")
    ap.add_argument("--word-k", type=int, default=0, help="words in 1 rarefied draw; 0 = auto (see the docstring)")
    ap.add_argument("--tag-k", type=int, default=0, help="hashtags in 1 rarefied draw; 0 = auto")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--min-restaurants", type=int, default=3, help="the NTAs that enter model_summary.csv")
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20261025)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    nta = pd.read_parquet(args.ntas)
    # The street views name the NTA columns in lower case (D133).
    nta = nta.rename(columns={"nta2020": "NTA2020", "nta_name": "NTAName", "borough": "BoroName"})
    nta = nta[["image_idx", "NTA2020", "NTAName", "BoroName"]]
    cap = pd.read_parquet(args.captions)
    if "copy_suspect" in cap:
        # x2: a text that got a copy of an example even after the retry was not read.
        cap = cap[~cap.copy_suspect.astype(bool)]
    cap = cap[(cap.extraction_class == "caption") & cap.is_quotable.fillna(False)]
    cap = cap.merge(nta, on="image_idx", how="left", validate="many_to_one")
    cap = pd.DataFrame({"model": cap.model, "nta": cap.NTA2020, "image_idx": cap.image_idx,
                        "answer": cap.doc_id, "text": cap.extraction_text.map(clean)})
    tags = pd.read_parquet(args.hashtags).merge(nta, on="image_idx", how="left", validate="many_to_one")
    # 1 count for each tag in each answer: a looping answer repeats its tags.
    tags = tags.drop_duplicates(["doc_id", "tag"])[["model", "NTA2020", "image_idx", "tag"]].rename(columns={"NTA2020": "nta"})
    if cap.nta.isna().any() or tags.nta.isna().any():
        raise SystemExit("a caption or a hashtag has no NTA")

    groups = {m: (cap[cap.model == m], tags[tags.model == m]) for m in MODELS}
    groups[POOLED] = (cap, tags)
    restaurants = nta.groupby("NTA2020").size().rename("restaurants")

    rows, pair_frames, ks = [], [], {}
    for model, (c, t) in groups.items():
        words = c.groupby("nta").text.apply(lambda s: [w for x in s for w in WORD.findall(x)])
        tag_lists = t.groupby("nta").tag.apply(list)
        big = restaurants[restaurants >= args.min_restaurants].index
        word_k = args.word_k or int(min(len(words.get(n, [])) for n in big))
        tag_k = args.tag_k or int(min(len(tag_lists.get(n, [])) for n in big))
        ks[model] = {"word_k": word_k, "tag_k": tag_k}
        pair_df, within_df = distances(c[["nta", "answer", "text"]], args.workers)
        between = pair_df.groupby("nta_a")[["nearest", "allpairs"]].mean().add_suffix("_between")
        pair_frames.append(pair_df.assign(model=model))
        for n in restaurants.index:
            w = entropies(words.get(n, []), word_k, args.draws, rng)
            h = entropies(tag_lists.get(n, []), tag_k, args.draws, rng)
            row = {"model": model, "nta": n, "restaurants": int(restaurants[n]),
                   "captions": int((c.nta == n).sum()),
                   "word_tokens": w["tokens"], "word_types": w["types"],
                   "word_entropy_plugin": w["plugin"], "word_entropy_mm": w["mm"], "word_entropy_rare": w["rare"],
                   "tag_tokens": h["tokens"], "tag_types": h["types"],
                   "hashtag_entropy_plugin": h["plugin"], "hashtag_entropy_mm": h["mm"], "hashtag_entropy_rare": h["rare"]}
            if n in between.index:
                row.update(between.loc[n].to_dict())
            wr = within_df[within_df.nta == n]
            if len(wr):
                row.update(wr.iloc[0][["nearest_within", "allpairs_within"]].to_dict())
            rows.append(row)
        print(f"[nta] {model}: {len(c):,} captions, {len(t):,} tags, {c.nta.nunique()} NTAs with a caption")

    measures = pd.DataFrame(rows).merge(
        nta.drop_duplicates("NTA2020")[["NTA2020", "NTAName", "BoroName"]].rename(columns={"NTA2020": "nta"}),
        on="nta", how="left")
    os.makedirs(args.out, exist_ok=True)
    measures.to_parquet(os.path.join(args.out, "nta_measures.parquet"), index=False)
    measures.to_csv(os.path.join(args.out, "nta_measures.csv"), index=False)
    pd.concat(pair_frames, ignore_index=True).to_parquet(os.path.join(args.out, "nta_pairs.parquet"), index=False)

    cols = ["word_entropy_plugin", "word_entropy_mm", "word_entropy_rare",
            "hashtag_entropy_plugin", "hashtag_entropy_mm", "hashtag_entropy_rare",
            "nearest_between", "nearest_within", "allpairs_between", "allpairs_within"]
    big = measures[measures.restaurants >= args.min_restaurants]
    summary = big.groupby("model")[cols].mean().round(4)
    summary.insert(0, "ntas", big.groupby("model").size())
    summary.insert(1, "captions", big.groupby("model").captions.sum())
    summary = summary.reindex(list(MODELS) + [POOLED])
    summary.insert(2, "word_k", [ks[m]["word_k"] for m in summary.index])
    summary.insert(3, "tag_k", [ks[m]["tag_k"] for m in summary.index])
    summary.to_csv(os.path.join(args.out, "model_summary.csv"))
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump({"script": os.path.relpath(__file__, ROOT), "version": __version__,
                   "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "inputs": [os.path.relpath(p, ROOT) for p in (args.captions, args.hashtags, args.ntas)],
                   "settings": vars(args), "rarefy_k": ks, "captions": int(len(cap)), "hashtags_deduped": int(len(tags)),
                   "ntas": int(restaurants.size)}, fh, indent=1)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summary.to_string())
    print(f"[nta] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
