#!/usr/bin/env python
"""F10.35 step 2: the readout instruments on each image of the F10.6 stores. CPU only.

The definitions are in `research/monocle/instruments.py`. Salient = rank 10 or better
among the store words. This script uses the top-10 form, because the full ranks of 591
images do not fit in memory.

| Stage | Input | Output |
|-------|-------|--------|
| `extract` | `data/f10/hop_all/{model}/reads/` | `data/f10/instruments/{model}/top10.npz`: the top 10 word ids of each image, tail token and read layer, and of the text read with no image |
| `events` | `top10.npz` | `data/f10/instruments/{model}/{instrument}.parquet`: for each cell and word, the count of images with the event, by group |
| `test` | the parquet files | `research/figures/f10/instruments/{model}_tests.csv`, `summary.csv` |

| Instrument | Cell | Event in 1 image |
|------------|------|------------------|
| `salient` | token `p`, layer `l` | The word is salient |
| `layer_step` | token `p`, from read layer `X` to the next read layer `Y` | `kind` loss: salient at `X` and not at `Y`. `kind` gain: the reverse |
| `position_step` | layer `l`, from token `A` to the next token `B` | The same, across tokens |
| `smear` | token `p` | Salient for 3 or more contiguous read layers |
| `layer_loss` | token `p`, from layer `X` to layer `Y` (all ordered pairs) | Salient at `X` and not at `Y`. Not tested |
| `position_loss` | layer `l`, from token `A` to token `B` (all ordered pairs) | Salient at `A` and not at `B`. Not tested |

The tests use the steps, not all the pairs. In the pairs, 1 change repeats: a word that
is salient at L56 only has a loss from L56 to each other layer. A step counts each change
once, at the boundary where it occurs. The tables of all pairs stay for the viewer: a gain
from `X` to `Y` is the row of the loss from `Y` to `X`.

Answer words. Column `answer` is True for a word that is salient at the last token and
the last read layer in 5% or more of the images. This is near the answer of the model.
Many rows of these words pass only because the groups come from the answer margin.

Groups. The NTAs go into 2 halves with a fixed seed: `s` (search) and `t` (test). The
images go into thirds of `yes_no`: `low`, `mid`, `high` (the groups of the viewer). The
column `n_s_high` is the count of search images in the top third with the event.

The test. For each row, a chi-square test (2 x 2, Yates) compares the event share of
the top third with the bottom third. A row is a candidate when the search half gives
p < .001 and each expected count is 5 or more. The test half then tests each
candidate. A candidate passes when p < .05 / (the candidates of the instrument) and the
sign agrees with the search half. In `summary.csv`, `passes_clean` counts the rows that pass, are not
answer words, and do not occur in the text read. Column `text`: 1 when the event also occurs in the
read of the same text with no image, NaN when that read does not exist.

**Warning:** this is a search. A word that passes is a candidate for a step of the
judgment, not a mechanism. The reads are each 4th layer only.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_instruments.py --models qwen3.5-9b --stage all
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_instruments.py --limit 1 --out /tmp/x   # 1 shard, a smoke run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.monocle.instruments import K, M, leaves, runs, steps, top_ids
from research.scripts.f10.hop_store import ALL_ROOT, open_store

__version__ = "1.1.0"

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/f10/instruments"
FIGS = ROOT / "research/figures/f10/instruments"
MODELS = ("gemma-4-31b", "qwen3.5-9b", "qwen3.6-27b")
SEED = 20261123
HALVES, THIRDS = ("s", "t"), ("low", "mid", "high")
GROUPS = [f"n_{h}_{t}" for h in HALVES for t in THIRDS]
TESTED = ("salient", "layer_step", "position_step", "smear")
ANSWER_SHARE = 0.05
P_SEARCH, MIN_EXPECTED, ALPHA = 1e-3, 5, 0.05


def extract(model: str, out: Path, limit: int | None) -> None:
    """Write the top 10 word ids of each image, tail token and read layer, and of the text read."""
    s = open_store(model, ALL_ROOT)
    files = sorted((s.path / "reads").glob("*.npy"))[:limit]
    parts = []
    for f in files:
        x = np.load(f, mmap_mode="r")
        parts.append(np.stack([top_ids(np.asarray(x[:, p]), K) for p in range(x.shape[1])], axis=1))
        print(f"[instruments] {model}: {f.name} {x.shape}", flush=True)
    ids = np.concatenate(parts)
    text = top_ids(np.load(s.path / "text_reads.npy"), K)
    images = s.images.iloc[: len(ids)]
    dest = out / model
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / "top10.npz", ids=ids, text=text, positions=np.array(s.manifest["positions"]),
             tokens=np.array(s.manifest["tail_tokens"]), layers=np.array(s.manifest["layers"]),
             sample_id=images["sample_id"].to_numpy(), words=s.vocab["token"].to_numpy())
    print(f"[instruments] {model}: top10 {ids.shape}, text {text.shape}", flush=True)


def groups(model: str, n: int) -> tuple[np.ndarray, dict]:
    """Give the group code of each image (half x 3 + third) and the size of each group."""
    images = open_store(model, ALL_ROOT).images.iloc[:n]
    ntas = np.array(sorted(images["nta2020"].unique()))
    test = set(np.random.default_rng(SEED).permutation(ntas)[len(ntas) // 2:])
    half = images["nta2020"].isin(test).to_numpy().astype(int)
    lo, hi = images["yes_no"].quantile([1 / 3, 2 / 3])
    third = np.where(images["yes_no"] >= hi, 2, np.where(images["yes_no"] <= lo, 0, 1))
    code = half * 3 + third
    return code, {g: int((code == i).sum()) for i, g in enumerate(GROUPS)}


def count(cols: list[np.ndarray], sizes: list[int], g: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Count the events by cell, word and group. ``cols`` are the cell and word of each event; ``g`` is its group."""
    key = np.zeros(len(g), np.int64)
    for c, size in zip(cols, sizes):
        key = key * size + c
    key = key * len(GROUPS) + g
    uniq, n = np.unique(key, return_counts=True)
    grp = uniq % len(GROUPS)
    rest = uniq // len(GROUPS)
    parts = {}
    for name, size in zip(reversed(names), reversed(sizes)):
        parts[name] = rest % size
        rest //= size
    df = pd.DataFrame({**{k: parts[k] for k in names}, "g": grp, "n": n})
    wide = df.pivot_table(index=names, columns="g", values="n", fill_value=0, aggfunc="sum")
    wide = wide.reindex(columns=range(len(GROUPS)), fill_value=0)
    wide.columns = GROUPS
    return wide.reset_index()


def text_keys(cols: list[np.ndarray], sizes: list[int]) -> set[int]:
    key = np.zeros(len(cols[0]), np.int64)
    for c, size in zip(cols, sizes):
        key = key * size + c
    return set(key.tolist())


def mark_text(df: pd.DataFrame, names: list[str], sizes: list[int], keys: set[int], valid: np.ndarray) -> pd.DataFrame:
    """Add `text`: 1 when the event occurs in the text read, 0 when not, NaN when the read does not exist."""
    key = np.zeros(len(df), np.int64)
    for name, size in zip(names, sizes):
        key = key * size + df[name].to_numpy()
    df["text"] = np.where(valid, np.isin(key, list(keys)).astype(float), np.nan)
    return df


def events(model: str, out: Path) -> None:
    """Write 1 table for each instrument."""
    z = np.load(out / model / "top10.npz", allow_pickle=True)
    ids, text, words = z["ids"], z["text"], z["words"]
    n, P, L, _ = ids.shape
    V = len(words)
    code, sizes = groups(model, n)
    img = np.arange(n)
    tables = {}

    # salient: token p, layer l
    nn, pp, ll, _ = np.indices(ids.shape)
    ok = ids >= 0
    df = count([pp[ok], ll[ok], ids[ok]], [P, L, V], code[nn[ok]], ["p", "l", "w"])
    tp, tl, _ = np.indices(text.shape)
    tok = text >= 0
    tables["salient"] = mark_text(df, ["p", "l", "w"], [P, L, V], text_keys([tp[tok], tl[tok], text[tok]], [P, L, V]),
                                  text[df["p"].to_numpy(), df["l"].to_numpy(), 0] >= 0)

    # layer_loss: token p, layer X to layer Y. leave[n, X, Y, j]
    parts, tkeys = [], set()
    for p in range(P):
        a = ids[:, p]
        leave = leaves(a[:, :, None, :], a[:, None, :, :])
        ni, X, Y, j = np.nonzero(leave)
        parts.append(count([np.full(len(X), p), X, Y, a[ni, X, j]], [P, L, L, V], code[ni], ["p", "X", "Y", "w"]))
        t = text[p]
        tleave = leaves(t[:, None, :], t[None, :, :])
        X, Y, j = np.nonzero(tleave)
        tkeys |= text_keys([np.full(len(X), p), X, Y, t[X, j]], [P, L, L, V])
        print(f"[instruments] {model}: layer loss, token {p + 1}/{P}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    tables["layer_loss"] = mark_text(df, ["p", "X", "Y", "w"], [P, L, L, V], tkeys, text[df["p"].to_numpy(), 0, 0] >= 0)

    # position_loss: layer l, token A to token B. leave[n, A, B, j]
    parts, tkeys = [], set()
    for layer in range(L):
        a = ids[:, :, layer]
        leave = leaves(a[:, :, None, :], a[:, None, :, :])
        ni, A, B, j = np.nonzero(leave)
        parts.append(count([np.full(len(A), layer), A, B, a[ni, A, j]], [L, P, P, V], code[ni], ["l", "A", "B", "w"]))
        t = text[:, layer]
        A, B, j = np.nonzero(leaves(t[:, None, :], t[None, :, :]))
        tkeys |= text_keys([np.full(len(A), layer), A, B, t[A, j]], [L, P, P, V])
        print(f"[instruments] {model}: position loss, layer {layer + 1}/{L}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    A, B, layer = df["A"].to_numpy(), df["B"].to_numpy(), df["l"].to_numpy()
    valid = (text[A, layer, 0] >= 0) & (text[B, layer, 0] >= 0)
    tables["position_loss"] = mark_text(df, ["l", "A", "B", "w"], [L, P, P, V], tkeys, valid)

    # smear: token p, a run of M or more read layers
    ev_p, ev_w, ev_n, ev_first, ev_last = [], [], [], [], []
    for i in img:
        for p in range(P):
            w, run, first, last = runs(ids[i, p])
            hit = run >= M
            ev_p.append(np.full(hit.sum(), p))
            ev_w.append(w[hit])
            ev_n.append(np.full(hit.sum(), i))
            ev_first.append(first[hit])
            ev_last.append(last[hit])
    ev_p, ev_w, ev_n = map(np.concatenate, (ev_p, ev_w, ev_n))
    df = count([ev_p, ev_w], [P, V], code[ev_n], ["p", "w"])
    layer_no = z["layers"]
    spans = pd.DataFrame({"p": ev_p, "w": ev_w, "first": layer_no[np.concatenate(ev_first)], "last": layer_no[np.concatenate(ev_last)]})
    span = spans.groupby(["p", "w"]).agg(first=("first", "median"), last=("last", "median")).reset_index()
    df = df.merge(span, on=["p", "w"], how="left")
    tkeys = set()
    for p in range(P):
        w, run, _, _ = runs(text[p])
        tkeys |= text_keys([np.full((run >= M).sum(), p), w[run >= M]], [P, V])
    tables["smear"] = mark_text(df, ["p", "w"], [P, V], tkeys, text[df["p"].to_numpy(), 0, 0] >= 0)

    # layer_step: token p, from read layer X to X + 1. kind 0 = loss, 1 = gain
    loss, gain = steps(ids)  # [n, P, L - 1, k]
    a, b = np.nonzero(loss), np.nonzero(gain)
    cols = [np.concatenate([a[1], b[1]]), np.concatenate([a[2], b[2]]), np.repeat([0, 1], [len(a[0]), len(b[0])]),
            np.concatenate([ids[a], ids[b[0], b[1], b[2] + 1, b[3]]])]
    df = count(cols, [P, L - 1, 2, V], code[np.concatenate([a[0], b[0]])], ["p", "X", "kind", "w"])
    tl, tg = steps(text)
    a, b = np.nonzero(tl), np.nonzero(tg)
    tkeys = text_keys([np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]), np.repeat([0, 1], [len(a[0]), len(b[0])]),
                       np.concatenate([text[a], text[b[0], b[1] + 1, b[2]]])], [P, L - 1, 2, V])
    tables["layer_step"] = mark_text(df, ["p", "X", "kind", "w"], [P, L - 1, 2, V], tkeys, text[df["p"].to_numpy(), 0, 0] >= 0)

    # position_step: layer l, from token A to A + 1
    by_layer = ids.transpose(0, 2, 1, 3)  # [n, L, P, k]
    loss, gain = steps(by_layer)
    a, b = np.nonzero(loss), np.nonzero(gain)
    cols = [np.concatenate([a[1], b[1]]), np.concatenate([a[2], b[2]]), np.repeat([0, 1], [len(a[0]), len(b[0])]),
            np.concatenate([by_layer[a], by_layer[b[0], b[1], b[2] + 1, b[3]]])]
    df = count(cols, [L, P - 1, 2, V], code[np.concatenate([a[0], b[0]])], ["l", "A", "kind", "w"])
    t = text.transpose(1, 0, 2)  # [L, P, k]
    tl, tg = steps(t)
    a, b = np.nonzero(tl), np.nonzero(tg)
    tkeys = text_keys([np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]), np.repeat([0, 1], [len(a[0]), len(b[0])]),
                       np.concatenate([t[a], t[b[0], b[1] + 1, b[2]]])], [L, P - 1, 2, V])
    A, layer = df["A"].to_numpy(), df["l"].to_numpy()
    tables["position_step"] = mark_text(df, ["l", "A", "kind", "w"], [L, P - 1, 2, V], tkeys,
                                        (text[A, layer, 0] >= 0) & (text[A + 1, layer, 0] >= 0))

    # answer words: salient at the last token and the last read layer in ANSWER_SHARE of the images or more
    last = ids[:, list(z["positions"]).index("final"), -1]
    share = np.bincount(last[last >= 0], minlength=V) / n
    answer = np.flatnonzero(share >= ANSWER_SHARE)
    answer = answer[np.argsort(-share[answer])]

    names = {"layers": z["layers"].tolist(), "positions": z["positions"].tolist(), "tokens": z["tokens"].tolist(),
             "answer_share": ANSWER_SHARE, "answer_words": {str(words[w]): round(float(share[w]), 3) for w in answer}}
    for name, df in tables.items():
        df.insert(df.columns.get_loc("w") + 1, "word", words[df["w"]])
        df["answer"] = np.isin(df["w"].to_numpy(), answer)
        df.to_parquet(out / model / f"{name}.parquet", index=False)
        print(f"[instruments] {model}: {name} {len(df)} rows", flush=True)
    (out / model / "meta.json").write_text(json.dumps({"version": __version__, "k": K, "m": M, "seed": SEED, "images": int(n),
                                                       "group_sizes": sizes, **names}, indent=1))


def chi2(a: np.ndarray, na: np.ndarray, b: np.ndarray, nb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give the p of a 2 x 2 chi-square test with Yates, and the smallest expected count. Events a of na, b of nb."""
    table = np.stack([a, na - a, b, nb - b], -1).astype(float)
    total = na + nb
    col = np.stack([a + b, total - a - b], -1)
    row = np.stack([na, nb], -1)
    exp = row[:, :, None] * col[:, None, :] / total[:, None, None]
    obs = table.reshape(-1, 2, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = (((np.abs(obs - exp) - 0.5).clip(min=0)) ** 2 / exp).sum((1, 2))
    return stats.chi2.sf(np.nan_to_num(stat), 1), exp.min((1, 2))


def test(model: str, out: Path, figs: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Test the events of the top third against the bottom third: search half, then test half."""
    meta = json.loads((out / model / "meta.json").read_text())
    size = meta["group_sizes"]
    rows, summary = [], []
    layers, tokens = np.array(meta["layers"]), np.array(meta["tokens"])
    for name in TESTED:
        df = pd.read_parquet(out / model / f"{name}.parquet")
        n = len(df)
        p_s, e_s = chi2(df["n_s_high"].to_numpy(), np.full(n, size["n_s_high"]), df["n_s_low"].to_numpy(), np.full(n, size["n_s_low"]))
        cand = df[(p_s < P_SEARCH) & (e_s >= MIN_EXPECTED)].copy()
        cand["p_search"] = p_s[(p_s < P_SEARCH) & (e_s >= MIN_EXPECTED)]
        m = len(cand)
        p_t, _ = chi2(cand["n_t_high"].to_numpy(), np.full(m, size["n_t_high"]), cand["n_t_low"].to_numpy(), np.full(m, size["n_t_low"]))
        share = lambda h, t: cand[f"n_{h}_{t}"] / size[f"n_{h}_{t}"]
        cand["share_s_high"], cand["share_s_low"] = share("s", "high"), share("s", "low")
        cand["share_t_high"], cand["share_t_low"] = share("t", "high"), share("t", "low")
        same = np.sign(cand["share_s_high"] - cand["share_s_low"]) == np.sign(cand["share_t_high"] - cand["share_t_low"])
        cand["p_test"] = p_t
        cand["passes"] = (p_t < ALPHA / max(m, 1)) & same
        cand.insert(0, "instrument", name)
        if "X" in cand:
            cand.insert(cand.columns.get_loc("X") + 1, "Y", layers[cand["X"] + 1])
            cand["X"] = layers[cand["X"]]
        if "A" in cand:
            cand["A_token"], cand["B_token"] = tokens[cand["A"]], tokens[cand["A"] + 1]
        if "l" in cand:
            cand["l"] = layers[cand["l"]]
        if "p" in cand:
            cand["p_token"] = tokens[cand["p"]]
        if "kind" in cand:
            cand["kind"] = np.where(cand["kind"] == 0, "loss", "gain")
        rows.append(cand)
        clean = cand["passes"] & ~cand["answer"] & (cand["text"] != 1)
        summary.append({"model": model, "instrument": name, "rows": n, "candidates": m, "passes": int(cand["passes"].sum()),
                        "passes_not_answer": int((cand["passes"] & ~cand["answer"]).sum()), "passes_clean": int(clean.sum())})
        print(f"[instruments] {model}: {name}: {n} rows, {m} candidates, {int(cand['passes'].sum())} pass", flush=True)
    got = pd.concat(rows, ignore_index=True).sort_values(["instrument", "passes", "p_test"], ascending=[True, False, True])
    figs.mkdir(parents=True, exist_ok=True)
    got.to_csv(figs / f"{model}_tests.csv", index=False)
    return got.assign(model=model), summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--stage", choices=["extract", "events", "test", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="the first shards only (a smoke run)")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--figs", default=str(FIGS))
    args = ap.parse_args()
    out, figs = Path(args.out), Path(args.figs)
    summary = []
    for model in args.models:
        if args.stage in ("extract", "all"):
            extract(model, out, args.limit)
        if args.stage in ("events", "all"):
            events(model, out)
        if args.stage in ("test", "all"):
            summary += test(model, out, figs)[1]
    if summary:
        table = pd.DataFrame(summary)
        table.to_csv(figs / "summary.csv", index=False)
        print(table.to_string(index=False), flush=True)
    print("[instruments] finished", flush=True)


if __name__ == "__main__":
    main()
