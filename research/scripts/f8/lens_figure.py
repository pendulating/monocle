#!/usr/bin/env python3
"""F8.4: the lens figure. The per-position rate of the F8.2 list words at top 5.

Reads the replay of research/scripts/f8/lens_replay_f4.py (multirun/f8/lens/<run>/)
one file at a time. The unit is a POSITION: a position that predicts a
completion token in the reasoning or in the response. It is NOT "any position
in an answer", because that rate grows with the answer length.

| Measure | For each (income group, field, list, layer) |
|---|---|
| `rate_top5` | the share of positions where a token of the list is in the top 5 of the logit lens |
| `rate_top1` | the same at top 1 |
| `rate_top5_not_said` | `rate_top5` at the positions whose written token is not a token of the list |
| `said` | the share of positions whose written token (the target) is a token of the list; the same at each layer |

The tokens of a list. Each word has 4 forms: `w`, ` w`, `W`, ` W` (W is the
first letter in capitals). A form that the tokenizer encodes as 1 token gives
that token. A word with no 1-token form is a multi-token word. The `+first`
lists add the first token of its ` w` form, and the token table marks it
`first_token`. The plain lists use the 1-token forms only; the figure uses
them (see the doc for the reason). The controls are the 6 control cities of
the whatami study, as one list `cities` and one list for each city.

Outputs in research/figures/f8/lens/:

| File | Content |
|---|---|
| lens_rates.csv | all the measures above, with the counts |
| lens_bands.csv | the rates in the early, middle and late thirds of the layers below the last, with a 95% bootstrap interval over the answers |
| lens_top1_bands.csv | the 20 tokens most often at top 1 in each band, for each income group (reasoning and response positions) |
| list_tokens.csv | the tokens of each list |
| said_answers.csv | the share of answers that say a list word (whole word, case-insensitive, the F8.2 match), for each group and field |
| lens_rates.png | the figure, plain lists; lens_rates_first.png: the `+first` lists |

Usage:
    python research/scripts/f8/lens_figure.py --run figure
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path("/share/ju/matt/mllmsci")
MODEL_DIR = "/share/ju/matt/zoo/models/Gemma-4-31B-it"
TRACES = REPO / "data/f4_traces/pooled/f4_traces_1k.parquet"
FIELDS = {1: "reasoning", 2: "response"}
GROUPS = ["low", "high"]
SEED = 20261028

#: The F8.2 word lists (facct-runbook.md, F8.2).
LISTS = {
    "upscale": ["upscale", "high-end", "upmarket", "luxury", "luxurious", "elegant", "sophisticated", "refined",
                "chic", "pricey", "expensive", "exclusive", "posh", "fancy", "swanky"],
    "cheap": ["cheap", "affordable", "budget", "inexpensive", "bargain", "no-frills", "low-cost",
              "hole-in-the-wall", "cheap eats"],
    "danger": ["unsafe", "dangerous", "sketchy", "rough", "gritty", "seedy", "crime", "run-down", "rundown",
               "dilapidated", "neglected"],
    "safe": ["safe", "secure", "peaceful", "quiet", "tranquil"],
}
#: The control cities of the whatami study (whatami-lens-study.md).
CITIES = ["Philadelphia", "Boston", "Chicago", "London", "Paris", "Toronto"]
#: Colors: slots 1 to 4 of the reference categorical palette; the controls are neutral and dashed.
COLORS = {"upscale": "#2a78d6", "cheap": "#eb6834", "danger": "#1baf7a", "safe": "#eda100", "cities": "#52514e"}


def bands(n_layers: int) -> dict[str, tuple[int, int]]:
    """Early, middle and late thirds of the layers below the last: [lo, hi) (lens_report.py)."""
    m = n_layers - 1
    a, b = round(m / 3), round(2 * m / 3)
    return {"early": (0, a), "middle": (a, b), "late": (b, m)}


def list_tokens(tok) -> tuple[dict[str, list[int]], pd.DataFrame]:
    """{list name: token ids}, and the table of the tokens."""
    rows = []
    words = {**LISTS, "cities": CITIES}
    for lst, ws in words.items():
        for w in ws:
            forms = [w, " " + w, w[0].upper() + w[1:], " " + w[0].upper() + w[1:]]
            single = []
            for f in forms:
                ids = tok(f, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    single.append((f, ids[0]))
            for f, i in single:
                rows.append({"list": lst, "word": w, "form": f, "token_id": int(i),
                             "token": tok.convert_ids_to_tokens(int(i)), "first_token": False})
            if not single:
                ids = tok(" " + w, add_special_tokens=False).input_ids
                rows.append({"list": lst, "word": w, "form": " " + w, "token_id": int(ids[0]),
                             "token": tok.convert_ids_to_tokens(int(ids[0])), "first_token": True})
    tab = pd.DataFrame(rows).drop_duplicates(["list", "word", "token_id"])
    out: dict[str, list[int]] = {}
    for lst in words:
        sub = tab[tab.list == lst]
        out[lst] = sorted(set(sub[~sub.first_token].token_id))
        if lst in LISTS:
            out[lst + "+first"] = sorted(set(sub.token_id))
    for c in CITIES:
        out["city:" + c] = sorted(set(tab[(tab.list == "cities") & (tab.word == c)].token_id))
    return out, tab


def said_answers(sel: pd.DataFrame) -> pd.DataFrame:
    """Answer-level: the share of answers whose field holds a list word (the F8.2 match)."""
    tr = pd.read_parquet(TRACES, columns=["model", "case", "image_idx", "reasoning", "response"])
    tr = tr[(tr.model == "gemma-4-31b") & (tr.case == "restaurant")]
    df = sel[["image_idx", "group"]].merge(tr, on="image_idx")
    rows = []
    for lst, ws in {**LISTS, "cities": CITIES}.items():
        pat = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in ws) + r")\b", re.I)
        for g in GROUPS:
            d = df[df.group == g]
            for f in FIELDS.values():
                rows.append({"list": lst, "group": g, "field": f, "answers": len(d),
                             "share_saying": float(d[f].str.contains(pat).mean())})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="figure")
    ap.add_argument("--replay-root", default=str(REPO / "multirun/f8/lens"))
    ap.add_argument("--out", default=str(REPO / "research/figures/f8/lens"))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    vocab = len(tok)
    lists, tab = list_tokens(tok)
    names = list(lists)
    out = Path(args.out) / ("" if args.run == "figure" else args.run)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "list_tokens.csv", index=False)

    root = Path(args.replay_root)
    sel = pd.read_parquet(root / "selection.parquet")
    group_of = dict(zip(sel.image_idx, sel.group))
    run_dir = root / args.run
    cfg = json.loads((run_dir / "run_config.json").read_text())
    layers = cfg["layers"]
    L = len(layers)
    parts = sorted(run_dir.glob("part-*.parquet"))
    answers = pd.read_parquet(run_dir / "answers.parquet") if (run_dir / "answers.parquet").exists() else \
        pd.concat([pd.read_parquet(p) for p in sorted(run_dir.glob("answers-*.parquet"))])
    ans_ids = sorted(answers.image_idx.unique())
    a_of = {a: i for i, a in enumerate(ans_ids)}
    nA, nF, nN = len(ans_ids), len(FIELDS), len(names)
    # Counts for each answer: positions [A, F]; hits [A, F, list, layer].
    n_pos = np.zeros((nA, nF), np.int64)
    hit5 = np.zeros((nA, nF, nN, L), np.int64)
    hit1 = np.zeros((nA, nF, nN, L), np.int64)
    hit5_ns = np.zeros((nA, nF, nN, L), np.int64)
    n_ns = np.zeros((nA, nF, nN), np.int64)
    said = np.zeros((nA, nF, nN), np.int64)
    bnd = bands(L) if layers == list(range(L)) else None
    top1 = {(g, b): np.zeros(vocab, np.int64) for g in GROUPS for b in (bnd or {})}

    for p in parts:
        t = pq.read_table(p, columns=["image_idx", "target_id", "field", "topk_ids"])
        img = t.column("image_idx").to_numpy()
        tgt = t.column("target_id").to_numpy()
        fld = t.column("field").to_numpy()
        n = t.num_rows
        ids = t.column("topk_ids").combine_chunks().flatten().flatten().to_numpy().reshape(n, L, -1)
        keep = fld > 0
        img, tgt, fld, ids = img[keep], tgt[keep], fld[keep], ids[keep]
        ai = np.array([a_of[i] for i in img])
        fi = fld.astype(np.int64) - 1
        np.add.at(n_pos, (ai, fi), 1)
        for j, nm in enumerate(names):
            m = np.isin(ids, lists[nm])                       # [n, L, k]
            h5 = m.any(2)
            h1 = m[:, :, 0]
            s = np.isin(tgt, lists[nm])
            np.add.at(hit5[:, :, j], (ai, fi), h5)
            np.add.at(hit1[:, :, j], (ai, fi), h1)
            np.add.at(hit5_ns[:, :, j], (ai[~s], fi[~s]), h5[~s])
            np.add.at(n_ns[:, :, j], (ai[~s], fi[~s]), 1)
            np.add.at(said[:, :, j], (ai, fi), s)
        if bnd:
            grp = np.array([group_of[i] for i in img])
            for g in GROUPS:
                gm = grp == g
                for b, (lo, hi) in bnd.items():
                    top1[(g, b)] += np.bincount(ids[gm, lo:hi, 0].reshape(-1), minlength=vocab)[:vocab]
        print(f"[f8-figure] {p.name}: {int(keep.sum())} positions", flush=True)

    grp_a = np.array([group_of[a] for a in ans_ids])
    rows = []
    for gi, g in enumerate(GROUPS):
        am = grp_a == g
        for f in range(nF):
            P = int(n_pos[am, f].sum())
            for j, nm in enumerate(names):
                S = int(said[am, f, j].sum())
                NS = int(n_ns[am, f, j].sum())
                for li, l in enumerate(layers):
                    h5, h1, hn = (int(x[am, f, j, li].sum()) for x in (hit5, hit1, hit5_ns))
                    rows.append({"group": g, "field": FIELDS[f + 1], "list": nm, "layer": l,
                                 "answers": int(am.sum()), "positions": P, "hits_top5": h5,
                                 "rate_top5": h5 / P, "rate_top1": h1 / P,
                                 "rate_top5_not_said": hn / max(NS, 1), "said": S / P})
    rates = pd.DataFrame(rows)
    rates.to_csv(out / "lens_rates.csv", index=False)

    # Bands with a bootstrap over the answers of the group (the answer is the unit of the resample).
    rng = np.random.default_rng(SEED)
    brow = []
    if bnd:
        for g in GROUPS:
            am = np.where(grp_a == g)[0]
            boots = rng.integers(0, len(am), (args.boot, len(am)))
            for f in range(nF):
                pos_a = n_pos[am, f]
                for j, nm in enumerate(names):
                    for b, (lo, hi) in bnd.items():
                        h_a = hit5[am, f, j, lo:hi].sum(1)
                        P = pos_a.sum() * (hi - lo)
                        bs = h_a[boots].sum(1) / (pos_a[boots].sum(1) * (hi - lo))
                        s_a = said[am, f, j]
                        sb = s_a[boots].sum(1) / pos_a[boots].sum(1)
                        brow.append({"group": g, "field": FIELDS[f + 1], "list": nm, "band": b,
                                     "layers": f"{lo}-{hi - 1}", "rate_top5": h_a.sum() / P,
                                     "ci_lo": float(np.quantile(bs, 0.025)), "ci_hi": float(np.quantile(bs, 0.975)),
                                     "said": s_a.sum() / pos_a.sum(), "said_ci_lo": float(np.quantile(sb, 0.025)),
                                     "said_ci_hi": float(np.quantile(sb, 0.975))})
        pd.DataFrame(brow).to_csv(out / "lens_bands.csv", index=False)
        trow = []
        for (g, b), cnt in top1.items():
            tot = cnt.sum()
            for r, i in enumerate(np.argsort(cnt)[::-1][: args.top]):
                trow.append({"group": g, "band": b, "layers": "%d-%d" % (bnd[b][0], bnd[b][1] - 1), "rank": r + 1,
                             "token_id": int(i), "token": tok.convert_ids_to_tokens(int(i)),
                             "count": int(cnt[i]), "share": float(cnt[i] / tot)})
        pd.DataFrame(trow).to_csv(out / "lens_top1_bands.csv", index=False)
    if len(sel) and TRACES.exists():
        said_answers(sel[sel.image_idx.isin(ans_ids)]).to_csv(out / "said_answers.csv", index=False)

    for suffix, fname in (("", "lens_rates.png"), ("+first", "lens_rates_first.png")):
        plot(rates, suffix, out / fname, n_ans={g: int((grp_a == g).sum()) for g in GROUPS},
             incomes=sel.groupby("group").median_household_income.agg(["min", "max"]))
    print(f"[f8-figure] {nA} answers, {int(n_pos.sum())} positions -> {out}")
    return 0


def plot(rates: pd.DataFrame, suffix: str, path: Path, n_ans: dict, incomes: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    lists = [k + suffix for k in LISTS] + ["cities"]
    lo = 1e-5
    for r, field in enumerate(FIELDS.values()):
        for c, g in enumerate(GROUPS):
            ax = axes[r, c]
            d = rates[(rates.group == g) & (rates.field == field)]
            xmax = d.layer.max()
            for nm in lists:
                base = nm.replace("+first", "")
                s = d[d.list == nm].sort_values("layer")
                y = np.maximum(s.rate_top5.to_numpy(), lo)
                ls = "--" if base == "cities" else "-"
                label = "control cities" if base == "cities" else base
                ax.plot(s.layer, y, ls, color=COLORS[base], lw=1.6, label=label)
                # The said rate: a marker right of the last layer.
                ax.plot([xmax + 2.5], [max(float(s.said.iloc[0]), lo)], "o" if base != "cities" else "s",
                        ms=6, color=COLORS[base], mec="white", mew=1, clip_on=False)
            ax.set_yscale("log")
            ax.set_ylim(lo * 0.8, 1)
            ax.set_xlim(-1, xmax + 4)
            ax.axvline(xmax + 1.2, color="#d6d5d0", lw=0.8)
            ax.grid(axis="y", color="#ecebe7", lw=0.6)
            inc = incomes.loc[g]
            ax.set_title(f"{field}, {g}-income NTAs (n = {n_ans[g]} answers; "
                         f"${inc['min'] / 1e3:.0f}k" + (f"-{inc['max'] / 1e3:.0f}k" if round(inc['max'] / 1e3) != round(inc['min'] / 1e3) else "")
                         + ")", fontsize=9, loc="left", color="#0b0b0b")
            if r == 1:
                ax.set_xlabel("layer (logit lens); dot right of the line: rate said")
            if c == 0:
                ax.set_ylabel("share of positions, list token in top 5")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    title = "gemma-4-31b, F4 restaurant answers: list words in the top 5 of the logit lens, per position"
    if suffix:
        title += " (with the first token of the multi-token words)"
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
