#!/usr/bin/env python
"""F10.30: the gradient tests of the 14 prompts with 3, 10 and 50 images for each NTA. CPU only.

The input is the vLLM answer capture of `answer_query.py` (`data/f10/hop_more/answers/{model}/answers.jsonl`,
10,000 images, 14 prompts). For each image E = P(yes) - P(no) over the 8 answer forms, and P_unsure =
1 - P(yes) - P(no) (F10.28). A form that the top-k list does not hold takes the floor of the list (D164).
An answer is `unable` when the first token is not a yes or a no form.

For each k in 3, 10 and 50 the script keeps the images of rank 1 to k in each NTA (the 591 images of F10.26
are ranks 1 to 3) and gives:

| Table | Content |
|-------|---------|
| `check.csv` | the Spearman rho of E over the 591 images against the F10.28 HF answers, for each cell. The vLLM path against the HF path |
| `ses.csv` | Spearman rho over NTAs of the NTA mean E (and P_unsure for Gemma where 5% or more refuse) with the 4 ACS features; Bonferroni over the cells of each k (F10.28) |
| `partial.csv` | the partial Spearman rho of E with the Black share given income, and with income given the Black share, for `ugly`, `nice`, `cool`; Bonferroni over 18 in each family and k (F10.29 question 1). Since 1.2.0, the same 2 partials for the White share |
| `manhattan.csv` | the Welch t of the NTA mean E of Manhattan against the other boroughs |
| `reliability.csv` | ICC(1) of E and the Spearman-Brown reliability of the k-image mean (F10.29 question 3) |
| `unable.csv` | the share of `unable` answers, overall and by borough |
| `summary.json` | the counts of passes at each k |

Output: `figures/f10/more/`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_more_answers.py [--models gemma-4-31b,...]
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
sys.path[:0] = [str(ROOT / "scripts/f10")]
from hop_prompts import ADJECTIVES, MODELS, NOUNS, answer_store  # noqa: E402
from hop_sample import icc1  # noqa: E402

__version__ = "1.2.0"
ANSWERS = ROOT / "data/f10/hop_more/answers"
IMAGES = ROOT / "data/f10/hop_more/images_all.parquet"
HOP_LOGIT = ROOT / "data/f10/hop_logit"
OUT = ROOT / "figures/f10/more"
YES = ["lp_Yes", "lp_yes", "lp_ Yes", "lp_ yes"]
NO = ["lp_No", "lp_no", "lp_ No", "lp_ no"]
FEATURES = ("pct_white_nh", "pct_black_nh", "pct_bachelors", "median_household_income")
KS = (3, 10, 50)
UNSURE_MODELS = ("gemma-4-31b",)
UNSURE_MIN = 0.05
PARTIAL_ADJS = ("ugly", "nice", "cool")
ALPHA = 0.05


def load(model: str) -> pd.DataFrame:
    """Give 1 row for each image and prompt: E, p_unsure, unable, nta2020, borough, rank."""
    a = pd.read_json(ANSWERS / model / "answers.jsonl", lines=True)
    a = a.drop_duplicates(["image_idx", "noun", "adjective"], keep="last")
    for c in YES + NO:
        a[c] = a[c].fillna(a["floor"])
    py, pn = np.exp(a[YES]).sum(1), np.exp(a[NO]).sum(1)
    a["E"] = py - pn
    a["p_unsure"] = (1.0 - py - pn).clip(lower=0.0)
    a["unable"] = ~a["first"].str.strip().str.lower().isin(["yes", "no"])
    img = pd.read_parquet(IMAGES)[["image_idx", "nta2020", "borough", "rank", "in_591"]]
    return a.merge(img, on="image_idx", validate="m:1")


def check(model: str, a: pd.DataFrame) -> list[dict]:
    """Compare E on the 591 images with the F10.28 HF answers (`data/f10/hop_logit`), matched by sample_id."""
    old = pd.read_parquet(ROOT / "data/f10/hop_depth/images.parquet").set_index("image_idx")["sample_id"]
    new = pd.read_parquet(IMAGES).set_index("image_idx")["sample_id"]
    rows = []
    for noun in NOUNS:
        for adj in ADJECTIVES:
            p = HOP_LOGIT / answer_store(model, adj, noun) / "answers.parquet"
            v = a[(a.noun == noun) & (a.adjective == adj) & a.in_591]
            if not p.exists() or v.empty:
                continue
            h = pd.read_parquet(p)
            if "floor" in h.columns:
                for c in YES + NO:
                    h[c] = h[c].fillna(h["floor"])
            hf = pd.DataFrame({"sample_id": old.reindex(h.image_idx).to_numpy(),
                               "E_hf": (np.exp(h[YES]).sum(1) - np.exp(h[NO]).sum(1)).to_numpy()})
            both = v.assign(sample_id=new.reindex(v.image_idx).to_numpy()).merge(hf, on="sample_id").dropna(subset=["E", "E_hf"])
            if len(both) < 10:
                continue
            r = stats.spearmanr(both.E, both.E_hf)
            rows.append({"model": model, "noun": noun, "adjective": adj, "n": len(both), "rho": float(r[0]),
                         "mean_E_vllm": float(both.E.mean()), "mean_E_hf": float(both.E_hf.mean()),
                         "yes_share_vllm": float((both.E > 0).mean()), "yes_share_hf": float((both.E_hf > 0).mean())})
    return rows


def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> tuple[float, float, int]:
    d = pd.concat([x, y, z], axis=1).dropna().rank()
    rx = d.iloc[:, 0] - np.polyval(np.polyfit(d.iloc[:, 2], d.iloc[:, 0], 1), d.iloc[:, 2])
    ry = d.iloc[:, 1] - np.polyval(np.polyfit(d.iloc[:, 2], d.iloc[:, 1], 1), d.iloc[:, 2])
    r = stats.pearsonr(rx, ry)
    return float(r[0]), float(r[1]), len(d)


def tests(model: str, a: pd.DataFrame, acs: pd.DataFrame) -> dict[str, list[dict]]:
    ses, partial, manhattan, reliability, unable = [], [], [], [], []
    for k in KS:
        sub = a[a["rank"] <= k]
        for noun in NOUNS:
            for adj in ADJECTIVES:
                s = sub[(sub.noun == noun) & (sub.adjective == adj)]
                if s.empty:
                    continue
                cell = {"model": model, "noun": noun, "adjective": adj, "k": k, "n_images": len(s), "n_nta": s.nta2020.nunique()}
                nta = s.groupby("nta2020")[["E", "p_unsure", "unable"]].mean().join(acs, how="left")
                share = float(s.unable.mean())
                measures = ("E", "p_unsure") if (model in UNSURE_MODELS and share >= UNSURE_MIN) else ("E",)
                for measure in measures:
                    for feat in FEATURES:
                        ok = nta[[measure, feat]].dropna()
                        rho, p = stats.spearmanr(ok[measure], ok[feat])
                        ses.append(cell | {"measure": measure, "feature": feat, "rho": float(rho), "p": float(p), "n_nta": len(ok), "share_unable": share})
                if adj in PARTIAL_ADJS:
                    rb = stats.spearmanr(nta.E, nta.pct_black_nh, nan_policy="omit")
                    r, p, n = partial_spearman(nta.E, nta.pct_black_nh, nta.median_household_income)
                    ri, pi, _ = partial_spearman(nta.E, nta.median_household_income, nta.pct_black_nh)
                    rw = stats.spearmanr(nta.E, nta.pct_white_nh, nan_policy="omit")
                    rwg, pwg, _ = partial_spearman(nta.E, nta.pct_white_nh, nta.median_household_income)
                    riw, piw, _ = partial_spearman(nta.E, nta.median_household_income, nta.pct_white_nh)
                    partial.append(cell | {"n_nta": n, "rho_black": float(rb[0]), "p_black": float(rb[1]),
                                           "rho_black_given_income": r, "p_black_given_income": p,
                                           "rho_income_given_black": ri, "p_income_given_black": pi,
                                           "rho_white": float(rw[0]), "p_white": float(rw[1]),
                                           "rho_white_given_income": rwg, "p_white_given_income": pwg,
                                           "rho_income_given_white": riw, "p_income_given_white": piw})
                bor = s.groupby("nta2020")["borough"].first().reindex(nta.index)
                t = stats.ttest_ind(nta.E[bor == "Manhattan"], nta.E[bor != "Manhattan"], equal_var=False)
                manhattan.append(cell | {"mean_manhattan": float(nta.E[bor == "Manhattan"].mean()), "mean_other": float(nta.E[bor != "Manhattan"].mean()),
                                         "t": float(t[0]), "p": float(t[1])})
                i1 = icc1(s.E, s.nta2020)
                kk = float(s.groupby("nta2020").size().mean())
                reliability.append(cell | {"k_mean": kk, "icc1": i1, "icc1_k": kk * i1 / (1 + (kk - 1) * i1)})
                row = cell | {"share_unable": share}
                row |= {b: float(v) for b, v in s.groupby("borough")["unable"].mean().items()}
                unable.append(row)
    return {"ses": ses, "partial": partial, "manhattan": manhattan, "reliability": reliability, "unable": unable}


def bonferroni(df: pd.DataFrame, p: str, by: list[str]) -> pd.Series:
    n = df.groupby(by)[p].transform("size")
    return (df[p] * n).clip(upper=1.0)


TABLE = ROOT / "notebooks/facct/tables/f10_prompt_ses_k50.tex"
BOROUGHS = ("Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island")


def borough_order(models: list[str], k: int = 50) -> pd.DataFrame:
    """Give the borough order of the NTA means of E for each cell at k images, with a Welch t of each borough against the rest.

    Bonferroni over the 42 cells for each borough. Writes `borough_order.csv` and `borough_tests.csv`.
    """
    order, tests = [], []
    for model in models:
        if not (ANSWERS / model / "answers.jsonl").exists():
            continue
        a = load(model)
        a = a[a["rank"] <= k]
        for noun in NOUNS:
            for adj in ADJECTIVES:
                s = a[(a.noun == noun) & (a.adjective == adj)]
                if s.empty:
                    continue
                nta = s.groupby("nta2020")["E"].mean()
                bor = s.groupby("nta2020")["borough"].first().reindex(nta.index)
                means = nta.groupby(bor).mean().reindex(list(BOROUGHS))
                ranked = means.sort_values(ascending=False)
                cell = {"model": model, "noun": noun, "adjective": adj, "k": k, "n_images": len(s)}
                order.append(cell | {"first": ranked.index[0], "last": ranked.index[-1], "order": " > ".join(ranked.index)}
                             | {f"mean_{b.replace(' ', '_')}": float(means[b]) for b in BOROUGHS})
                for b in BOROUGHS:
                    t = stats.ttest_ind(nta[bor == b], nta[bor != b], equal_var=False)
                    tests.append(cell | {"borough": b, "n_nta": int((bor == b).sum()), "mean": float(means[b]),
                                         "mean_other": float(nta[bor != b].mean()), "t": float(t[0]), "p": float(t[1])})
    o, t = pd.DataFrame(order), pd.DataFrame(tests)
    t["p_bonferroni"] = bonferroni(t, "p", ["borough"])
    o.to_csv(OUT / "borough_order.csv", index=False)
    t.to_csv(OUT / "borough_tests.csv", index=False)
    print(f"[f10.30] borough order at k = {k}, {len(o)} cells:")
    print(pd.crosstab(o.adjective, o["first"]).to_string())
    print("last:")
    print(pd.crosstab(o.adjective, o["last"]).to_string())
    print("passes of the borough against the rest, by adjective (of 6 cells):")
    print(t.assign(pas=t.p_bonferroni < ALPHA).pivot_table(index="adjective", columns="borough", values="pas", aggfunc="sum").to_string())
    borough_table(o, t, k)
    return o


BOROUGH_TABLE = ROOT / "notebooks/facct/tables/f10_borough_order.tex"


def borough_table(o: pd.DataFrame, t: pd.DataFrame, k: int) -> None:
    """Write `tab:borough-order`: for each adjective, the borough that ranks first and last in the 6 cells, and the passes of each borough against the rest."""
    def lead(s: pd.Series) -> str:
        c = s.value_counts()
        return ", ".join(f"{b} ({n})" for b, n in c.items())
    pas = t.assign(pas=t.p_bonferroni < ALPHA).pivot_table(index="adjective", columns="borough", values="pas", aggfunc="sum")
    cols = list(BOROUGHS)
    lines = [
        f"% Generated by scripts/f10/hop_more_answers.py --borough (k = {k}, F10.30). Do not edit by hand.",
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{Each standard ranks the boroughs in its own order. For each adjective, over the 6 cells of 3 models and 2 nouns ('place', 'scene'): the borough whose neighborhoods have the highest and the lowest mean expected answer $E$ (count of cells in brackets), and the number of cells in which that borough differs from the rest of the city (Welch $t$ of the neighborhood means of $E$ over {k} images, Bonferroni over the 42 cells of each borough). {k * 197:,} images.}}",
        "\\label{tab:borough-order}", "\\setlength{\\tabcolsep}{4pt}", "\\small",
        "\\begin{tabular}{lll" + "r" * len(cols) + "}", "\\toprule",
        " & & & \\multicolumn{" + str(len(cols)) + "}{c}{cells that differ from the rest (of 6)} \\\\",
        "\\cmidrule(lr){4-" + str(3 + len(cols)) + "}",
        "Adjective & First & Last & " + " & ".join(cols) + " \\\\", "\\midrule",
    ]
    for adj in ADJECTIVES:
        s = o[o.adjective == adj]
        lines.append(f"{adj} & {lead(s['first'])} & {lead(s['last'])} & " + " & ".join(str(int(pas.loc[adj, b])) for b in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    BOROUGH_TABLE.write_text("\n".join(lines))
    print(f"[f10.30] {BOROUGH_TABLE}")


def table(k: int = 50) -> None:
    """Write `tab:prompt-ses` at k images for each NTA, in the paper layout of `hop_hierarchy.prompt_table`, from `ses.csv`."""
    import hop_hierarchy as hh

    d = pd.read_csv(OUT / "ses.csv")
    d = d[d.k == k].copy()
    hh.TABLE = TABLE
    hh.prompt_table(d)
    t = TABLE.read_text()
    t = t.replace("% Generated by scripts/f10/hop_hierarchy.py", f"% Generated by scripts/f10/hop_more_answers.py --table (k = {k}, F10.30) with hop_hierarchy.prompt_table")
    t = t.replace("and four ACS features of the NTA, for three models.}", f"and four ACS features of the NTA, for three models, with {k} images for each NTA ({k * 197:,} images).}}")
    t = t.replace("NTA means over 197 NTAs.", f"NTA means of {k} images over 197 NTAs.")
    TABLE.write_text(t)
    print(f"[f10.30] {TABLE}: k = {k}, {int((d.p_bonferroni < ALPHA).sum())} of {len(d)} tests pass")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--table", action="store_true", help="only write the k = 50 table from ses.csv")
    ap.add_argument("--borough", action="store_true", help="only write the borough order of the NTA means at k = 50")
    args = ap.parse_args()
    if args.table:
        table()
        return
    if args.borough:
        borough_order(args.models.split(","))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    acs = pd.read_parquet(ROOT / "data/f4_traces/nta_acs.parquet").set_index("nta2020")[list(FEATURES)]
    out: dict[str, list[dict]] = {"check": [], "ses": [], "partial": [], "manhattan": [], "reliability": [], "unable": []}
    for model in args.models.split(","):
        if not (ANSWERS / model / "answers.jsonl").exists():
            print(f"[f10.30] {model}: no answers yet")
            continue
        a = load(model)
        done = a.groupby(["noun", "adjective"]).size()
        print(f"[f10.30] {model}: {len(a)} answers, {done.min()} to {done.max()} images for each prompt, {int(a.image_idx.nunique())} images")
        out["check"] += check(model, a)
        for key, rows in tests(model, a, acs).items():
            out[key] += rows
    tables = {key: pd.DataFrame(rows) for key, rows in out.items() if rows}
    if "ses" in tables:
        tables["ses"]["p_bonferroni"] = bonferroni(tables["ses"], "p", ["k"])
    if "partial" in tables:
        for c in ("p_black", "p_black_given_income", "p_income_given_black",
                  "p_white", "p_white_given_income", "p_income_given_white"):
            tables["partial"][c + "_bonferroni"] = bonferroni(tables["partial"], c, ["k"])
    if "manhattan" in tables:
        tables["manhattan"]["p_bonferroni"] = bonferroni(tables["manhattan"], "p", ["k"])
    for key, df in tables.items():
        df.to_csv(OUT / f"{key}.csv", index=False)
    summary = {}
    if "ses" in tables:
        d = tables["ses"]
        summary["ses_pass"] = {str(k): int((g.p_bonferroni < ALPHA).sum()) for k, g in d.groupby("k")}
        summary["ses_n"] = {str(k): int(len(g)) for k, g in d.groupby("k")}
        print("ses passes by k, measure E:")
        print(d[d.measure == "E"].assign(pas=d.p_bonferroni < ALPHA).pivot_table(index=["model", "feature"], columns="k", values="pas", aggfunc="sum").to_string())
    if "partial" in tables:
        d = tables["partial"]
        summary["black_given_income_pass"] = {str(k): int((g.p_black_given_income_bonferroni < ALPHA).sum()) for k, g in d.groupby("k")}
        summary["income_given_black_pass"] = {str(k): int((g.p_income_given_black_bonferroni < ALPHA).sum()) for k, g in d.groupby("k")}
        summary["black_pass"] = {str(k): int((g.p_black_bonferroni < ALPHA).sum()) for k, g in d.groupby("k")}
        summary["white_given_income_pass"] = {str(k): int((g.p_white_given_income_bonferroni < ALPHA).sum()) for k, g in d.groupby("k")}
        summary["income_given_white_pass"] = {str(k): int((g.p_income_given_white_bonferroni < ALPHA).sum()) for k, g in d.groupby("k")}
        summary["white_pass"] = {str(k): int((g.p_white_bonferroni < ALPHA).sum()) for k, g in d.groupby("k")}
        print("partial:", {k: v for k, v in summary.items() if "pass" in k and "ses" not in k})
    if "reliability" in tables:
        d = tables["reliability"]
        summary["icc1_median"] = {str(k): float(g.icc1.median()) for k, g in d.groupby("k")}
        summary["icc1_k_median"] = {str(k): float(g.icc1_k.median()) for k, g in d.groupby("k")}
        print("reliability:", summary["icc1_median"], summary["icc1_k_median"])
    if "check" in tables:
        d = tables["check"]
        summary["check_rho_median"] = float(d.rho.median())
        print("check against HF (rho over the 591):", d.rho.round(2).tolist())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
