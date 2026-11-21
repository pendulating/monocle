#!/usr/bin/env python
"""F10.8: preset word sets on the logits of each cell. CPU only.

The sets are fixed here, before any result. External lists need no split of the images.

| Set | Source |
|-----|--------|
| `concrete`, `abstract` | Brysbaert et al. 2014: rating at least 4.5, or at most 2.0 |
| `positive`, `negative` | NRC VAD: valence at least 0.85, or at most 0.15 |
| `spanish_portuguese` | the NRC VAD translations in Spanish and Portuguese, minus each English word of the 2 lists |
| `aesthetic_*`, `status_*`, `p_*`, `v_*`, `s_*`, `o_*` | the lists below. They are project lists with no external validation |

The hand lists have 4 families: `p_` aesthetic properties, `v_` aesthetic values, `s_`
process stages, `o_` visual objects. Some words come from the hand reads of the paper,
thus these lists are not blind. Use the text-baseline test and the held-out borough test
to judge them, and read `sets_drop.csv`.

Each word has 1 control word, matched without replacement inside 1 set on log frequency (SUBTLEX), the
norm of the unembedding row and the letter count. A word with no frequency (`vibe`, or a
Spanish word) matches on the other 2 features. Reads are uncapped and standardized
over the store words. A score is the set mean minus the control mean.

| Question | Unit | Test |
|----------|------|------|
| Is the score above the same score with no image | NTA mean | one-sample t, df 196 |
| Does the score move with the answer margin | image | r, NTA-clustered t, df 196 |
| Does a pair of sets move with the margin | image | the same, on the difference of the 2 scores |

Holm covers each family: all sets and cells for 1 model and 1 question.

Word sets can overlap the yes and no rows of the unembedding. `sets_cosine.csv` gives
that overlap, raw and whitened. `sets_drop.csv` drops each word pair of the hand lists at
the best cell, to show if 1 word carries a result.

Output: `data/f10/hop_sets/{model}/`, `figures/f10/sets/sets_*.csv` and `sets_{pair}.png`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_sets.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/f10"), str(ROOT / "notebooks/cvpr")]
from analyze_logit import LOGIT_ROOT, MODELS, NAMES, N_RANDOM, SEED, cluster_r, holm, load, plot_grid, shard_reads  # noqa: E402

__version__ = "1.0.0"

LEX = ROOT / "data/lexicons"
OUT = ROOT / "figures/f10/sets"
DATA = ROOT / "data/f10/hop_sets"
PAIRS = {"aesthetic": ("aesthetic_pos", "aesthetic_neg"), "status": ("status_high", "status_low"), "valence": ("positive", "negative"),
         "concreteness": ("concrete", "abstract"), "order": ("p_order", "p_decay"), "age": ("p_new", "p_old"), "density": ("p_busy", "p_open"),
         "setting": ("s_urban", "s_rural")}
CAP = 30.0
HAND = {
    "aesthetic_pos": "beautiful pretty lovely elegant stylish charming attractive gorgeous stunning tasteful chic trendy hip fashionable artistic "
                     "picturesque cozy vibrant quaint graceful exquisite handsome delightful striking",
    "aesthetic_neg": "ugly drab dull shabby bleak dreary gloomy tacky unattractive hideous grim dingy unsightly boring plain monotonous gaudy "
                     "bland cluttered dismal sterile lifeless repulsive grubby",
    "status_high": "rich wealthy affluent luxury luxurious upscale expensive prestigious exclusive elite posh opulent lavish classy privileged "
                   "premium sophisticated high-end fancy prosperous",
    "status_low": "poor impoverished cheap deprived destitute slum rundown neglected squalid derelict dilapidated ghetto needy lowly shoddy "
                  "broke decrepit blighted seedy disadvantaged",
    "p_order": "clean tidy neat orderly pristine spotless immaculate sanitary maintained manicured cleanliness",
    "p_decay": "dirty filthy messy grimy deteriorating crumbling weathered worn rusty abandoned vacant decaying peeling",
    "p_new": "modern contemporary sleek renovated modernized futuristic minimalist polished refurbished",
    "p_old": "historic vintage antique traditional classic heritage older aged nostalgic retro rustic",
    "p_color": "colorful bright vivid pastel gray grey beige muted monochrome",
    "p_busy": "crowded busy bustling lively dense cramped narrow congested packed",
    "p_open": "quiet empty spacious deserted sparse secluded peaceful calm tranquil",
    "p_nature": "green leafy lush trees garden shady scenic verdant waterfront park",
    "p_grit": "gritty grungy edgy raw industrial graffiti urban underground rugged",
    "v_authentic": "authentic authenticity genuine real original local honest unpretentious sincere",
    "v_thrift": "inexpensive affordable budget humble modest frugal thrifty austere austerity",
    "v_utility": "functional practical utilitarian convenient useful efficient accessible ordinary mundane",
    "v_community": "community neighborhood friendly social family welcoming neighborly communal residential",
    "v_trend": "instagram boutique boutiques gentrified artisanal curated bohemian nightclubs bistro",
    "v_safety": "safe secure dangerous unsafe crime risky threatening sketchy",
    "v_tourism": "tourist tourists attractions sightseeing famous iconic landmark destination tourism",
    "v_transgress": "illegal vandalism gangs anarchy rebellious punk tattoo subversive outlaw homeless",
    "s_locate": "location nearby surroundings geographic coordinates latitude longitude map address relocated",
    "s_recognize": "recognize recognizes familiarity identify recognizable familiar unfamiliar recognition",
    "s_standard": "vibe vibes idea opinion concept definition meaning notion",
    "s_evaluate": "judging subjective questionable debatable attractiveness rating evaluate assessment criteria",
    "s_uncertain": "inconclusive cannot idk unsure uncertain unclear depends maybe perhaps",
    "s_urban": "city metropolitan downtown borough cityscape municipal urbanized citywide",
    "s_rural": "rural suburban countryside coastal farm seaside farmland acreage",
    "o_commerce": "storefront shop store restaurant awning signage bodega sign market pharmacy",
    "o_built": "building facade brick window door wall roof stoop concrete glass steel",
    "o_street": "street sidewalk road car traffic pavement crosswalk parking bus",
    "o_people": "people pedestrians crowd residents pedestrian commuters",
}


def lexicon() -> pd.DataFrame:
    """Give the rating lists on 1 table of bare lower-case words."""
    b = pd.read_csv(LEX / "brysbaert.txt", sep="\t", usecols=["Word", "Conc.M", "SUBTLEX"]).dropna(subset=["Word"])
    b["word"] = b["Word"].str.lower()
    v = pd.read_csv(LEX / "NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt", sep="\t", header=0, names=["word", "valence", "arousal", "dominance"])
    return b.drop_duplicates("word").merge(v[["word", "valence"]], on="word", how="outer")


def romance_words(lex: pd.DataFrame) -> set[str]:
    """Give the Spanish and Portuguese translations in NRC VAD that are not English words."""
    t = pd.read_csv(LEX / "NRC-VAD-Lexicon/NRC-VAD-Lexicon-ForVariousLanguages.txt", sep="\t", dtype=str)
    cols = [c for c in t.columns if c.lower().startswith(("spanish", "portuguese"))]
    if len(cols) != 2:
        raise ValueError(f"the lexicon columns changed: {cols}")
    english = set(lex["word"].dropna()) | set(t.iloc[:, 0].dropna().str.lower())
    return {w for c in cols for w in t[c].dropna().str.lower()} - english


def build_sets(words: pd.Series, lex: pd.DataFrame) -> dict[str, list[str]]:
    on = set(words)
    pick = lambda mask: sorted(set(lex.loc[mask, "word"]) & on)
    sets = {"concrete": pick(lex["Conc.M"] >= 4.5), "abstract": pick(lex["Conc.M"] <= 2.0), "positive": pick(lex["valence"] >= 0.85),
            "negative": pick(lex["valence"] <= 0.15)}
    sets["spanish_portuguese"] = sorted(romance_words(lex) & on)
    sets |= {k: sorted(set(v.split()) & on) for k, v in HAND.items()}
    return sets


def match_controls(features: pd.DataFrame, chosen: list[str], banned: set[str]) -> list[str]:
    """Give 1 nearest control word for each set word, without replacement. `features` has 1 row for each word."""
    known = [w for w in chosen if w in features.index]
    pool = features.loc[[w for w in features.index if w not in banned and np.isfinite(features.at[w, "freq"])]]
    a = features.to_numpy(dtype=float)
    mean, sd = np.nanmean(a, 0), np.nanstd(a, 0)
    x, names = (pool.to_numpy(dtype=float) - mean) / sd, pool.index.to_numpy()
    free = np.ones(len(x), bool)
    out = []
    for w in known:
        q = (features.loc[w].to_numpy(dtype=float) - mean) / sd
        d = np.nansum((x - q) ** 2, axis=1)
        d[~free] = np.inf
        i = int(np.argmin(d))
        free[i] = False
        out.append(str(names[i]))
    return out


def scores(d: dict, model: str, sets: dict[str, list[str]], controls: dict[str, list[str]], words: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Give the set-minus-control score of each image, the same score of the text with no image, and the columns of the hand lists."""
    at = {w: i for i, w in enumerate(words)}
    index = {k: (np.array([at[w] for w in v if w in at]), np.array([at[w] for w in controls[k]])) for k, v in sets.items()}
    hand = sorted({at[w] for k in sets if k in HAND for w in sets[k] + controls[k]})
    out, kept = [], []
    for f in shard_reads(d):
        x = np.load(f)
        x = (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)
        out.append(np.stack([x[..., a].mean(-1) - x[..., b].mean(-1) for a, b in index.values()]))
        kept.append(x[..., hand])
        print(f"[f10.8] {model}: {f.name}", flush=True)
    text = np.load(d["store"].path / "text_reads.npy")
    if d["head"]["softcap"]:
        text = CAP * np.arctanh((text.astype(np.float64) / CAP).clip(-1 + 1e-7, 1 - 1e-7))
    text = (text - np.nanmean(text, -1, keepdims=True)) / np.nanstd(text, -1, keepdims=True)
    base = np.stack([text[..., a].mean(-1) - text[..., b].mean(-1) for a, b in index.values()])
    return np.concatenate(out, axis=1), base, np.concatenate(kept), hand


def tables(d: dict, model: str, names: list[str], score: np.ndarray, base: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    im, head = d["images"], d["head"]
    tokens = d["store"].manifest["tail_tokens"]
    y, cl = im["margin"].to_numpy(), im["nta2020"].to_numpy()
    codes = pd.factorize(cl)[0]
    rows, pairs = [], []
    for s, name in enumerate(names):
        for a, pos in enumerate(head["positions"]):
            for b, layer in enumerate(head["layers"]):
                x = score[s, :, a, b]
                gap = np.bincount(codes, x - base[s, a, b]) / np.bincount(codes)
                r, t, p = cluster_r(x, y, cl)
                t0, p0 = (stats.ttest_1samp(gap, 0.0) if np.isfinite(base[s, a, b]) else (np.nan, np.nan))
                rows.append({"model": model, "set": name, "position": pos, "token": tokens[a], "index": a, "layer": layer, "mean": x.mean(), "text": base[s, a, b],
                             "gap": gap.mean(), "d": gap.mean() / gap.std(ddof=1), "t_text": t0, "p_text": p0, "r": r, "t": t, "p": p})
    for name, (hi, lo) in PAIRS.items():
        i, j = names.index(hi), names.index(lo)
        for a, pos in enumerate(head["positions"]):
            for b, layer in enumerate(head["layers"]):
                r, t, p = cluster_r(score[i, :, a, b] - score[j, :, a, b], y, cl)
                pairs.append({"model": model, "pair": name, "position": pos, "token": tokens[a], "index": a, "layer": layer, "r": r, "t": t, "p": p})
    rows, pairs = pd.DataFrame(rows), pd.DataFrame(pairs)
    rows["p_holm"] = holm(rows["p"].to_numpy())
    ok = rows["p_text"].notna()
    rows.loc[ok, "p_text_holm"] = holm(rows.loc[ok, "p_text"].to_numpy())
    pairs["p_holm"] = holm(pairs["p"].to_numpy())
    return rows, pairs


def drop_words(d: dict, model: str, pairs: pd.DataFrame, kept: np.ndarray, hand: list[int], sets: dict, controls: dict, words: list[str]) -> pd.DataFrame:
    """Drop each word and its control at the best cell of a hand pair, and give the range of r."""
    at = {w: i for i, w in enumerate(words)}
    col = {w: i for i, w in enumerate(hand)}
    y, cl = d["images"]["margin"].to_numpy(), d["images"]["nta2020"].to_numpy()
    tokens = d["store"].manifest["tail_tokens"]
    end = tokens.index("?")
    rows = []
    for pair in ("aesthetic", "status"):
        hi, lo = PAIRS[pair]
        for window, part in (("question", pairs[(pairs["pair"] == pair) & (pairs["index"] <= end)]), ("all", pairs[pairs["pair"] == pair])):
            best = part.iloc[part["r"].abs().to_numpy().argmax()]
            b = d["head"]["layers"].index(best["layer"])
            x = kept[:, int(best["index"]), b]
            side = lambda k: x[:, [col[at[w]] for w in sets[k]]] - x[:, [col[at[w]] for w in controls[k]]]
            a, c = side(hi), side(lo)
            got = [cluster_r(np.delete(a, i, 1).mean(1) - c.mean(1), y, cl)[0] for i in range(a.shape[1])]
            got += [cluster_r(a.mean(1) - np.delete(c, i, 1).mean(1), y, cl)[0] for i in range(c.shape[1])]
            names = sets[hi] + sets[lo]
            worst = int(np.argmin(np.abs(got)))
            rows.append({"model": model, "pair": pair, "window": window, "token": best["token"], "layer": best["layer"], "r": best["r"], "p_holm": best["p_holm"],
                         "r_min": min(got), "r_max": max(got), "word_at_weakest": names[worst]})
    return pd.DataFrame(rows)


def cosine_table(model: str, sets: dict, controls: dict, words: list[str]) -> pd.DataFrame:
    """Give the cosine of each set contrast with the yes - no direction, raw and whitened."""
    root = LOGIT_ROOT / model
    at = {w: i for i, w in enumerate(words)}
    u = np.load(root / "unembed_store.npy", mmap_mode="r")
    answer = np.load(root / "answer_rows.npy").astype(np.float64)
    value, vector = np.linalg.eigh(np.load(root / "unembed_cov.npy").astype(np.float64))
    white = (vector / np.sqrt(value.clip(min=value.max() * 1e-8))) @ vector.T
    target = answer[:len(answer) // 2].mean(0) - answer[len(answer) // 2:].mean(0)
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    row = lambda names: np.asarray(u[sorted(at[w] for w in names)], np.float64).mean(0)
    rng = np.random.default_rng(SEED)
    rows = []
    for k in sets:
        v = row(sets[k]) - row(controls[k])
        n = len(sets[k])
        null = []
        for _ in range(N_RANDOM):
            pick = rng.choice(u.shape[0], 2 * n, replace=False)
            null.append(cos(np.asarray(u[np.sort(pick[:n])], np.float64).mean(0) - np.asarray(u[np.sort(pick[n:])], np.float64).mean(0), target))
        rows.append({"model": model, "set": k, "n": n, "raw": cos(v, target), "white": cos(white @ v, white @ target), "null_sd_raw": float(np.std(null))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lex = lexicon()
    freq = lex.dropna(subset=["SUBTLEX"]).set_index("word")["SUBTLEX"]
    all_rows, all_pairs, all_drop, all_cos, info = [], [], [], [], {}
    for model in args.models:
        d = load(model)
        vocab = d["store"].vocab["token"].str.strip()
        sets = build_sets(vocab, lex)
        norm = pd.Series(np.linalg.norm(np.load(LOGIT_ROOT / model / "unembed_store.npy", mmap_mode="r"), axis=1), index=vocab.to_numpy())
        norm = norm[~norm.index.duplicated()]
        banned = {w for v in sets.values() for w in v}
        common = [w for w in norm.index if w in freq.index or w in banned]
        features = pd.DataFrame({"freq": np.log10(freq.reindex(common).astype(float).to_numpy() + 1), "norm": norm.loc[common].to_numpy(),
                                 "length": [len(w) for w in common]}, index=common)
        controls = {}
        for name, chosen in sets.items():
            controls[name] = match_controls(features, chosen, banned)
            sets[name] = [w for w in chosen if w in features.index]
        info[model] = {k: len(v) for k, v in sets.items()}
        print(f"[f10.8] {model}: sizes {info[model]}", flush=True)
        (DATA / model).mkdir(parents=True, exist_ok=True)
        (DATA / model / "sets.json").write_text(json.dumps({"sets": sets, "controls": controls}, indent=0))
        all_cos.append(cosine_table(model, sets, controls, vocab.tolist()))
        score, base, kept, hand = scores(d, model, sets, controls, vocab.tolist())
        np.savez(DATA / model / "scores.npz", score=score, base=base, names=np.array(list(sets)))
        a, b = tables(d, model, list(sets), score, base)
        all_rows.append(a)
        all_pairs.append(b)
        all_drop.append(drop_words(d, model, b, kept, hand, sets, controls, vocab.tolist()))
    rows, pairs = pd.concat(all_rows, ignore_index=True), pd.concat(all_pairs, ignore_index=True)
    rows.to_csv(OUT / "sets_cells.csv", index=False)
    pairs.to_csv(OUT / "sets_pairs.csv", index=False)
    pd.concat(all_drop, ignore_index=True).to_csv(OUT / "sets_drop.csv", index=False)
    pd.concat(all_cos, ignore_index=True).to_csv(OUT / "sets_cosine.csv", index=False)
    for pair in PAIRS:
        plot_grid({NAMES[m]: pairs[(pairs["model"] == m) & (pairs["pair"] == pair)] for m in args.models}, "r", OUT / f"sets_{pair}.png",
                  f"The {pair} pair against the answer margin",
                  f"Pearson r over 591 images between the {pair} set difference at a token and layer and the answer margin of the model. The reads are uncapped "
                  "and standardized, and each set has matched control words.", ["F10.8", "word sets"], -1, 1, "r with the answer margin", True)
    print("[f10.8] finished", flush=True)


if __name__ == "__main__":
    main()
