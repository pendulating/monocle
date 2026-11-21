"""Compare the J++ patch words with the R-lens runs of the same 8 images (D146).

The tables copy the R-lens tables of `facct-runbook.md` (D141): the share of
blocks whose top token changes, and the question words in the 4 deepest
layers. The same word lists go over both lenses, thus 2 counts compare
directly. Layers pair by relative depth, in the order of `--layers`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/rq1/jpp_compare.py jpp31b
    .venv-mllmsci-vllm025cu129/bin/python scripts/rq1/jpp_compare.py jppq9b 8
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "notebooks" / "cvpr"), str(REPO_ROOT / "notebooks" / "facct")]

import _rq as Q  # noqa: E402

__version__ = "1.0.0"

RLENS = (6, 12, 18, 24, 30, 36, 42, 46)
QUESTION = {"how", "appeal", "appealing", "restaurant", "restaurants", "eat", "block", "blocks", "street", "streets",
            "photography", "photoshoot"}
FILLER = {"this", "is", "a", "photograph"}
NORWOOD = {"illegal", "inhabitants", "homelessness"}
LAYOUTS = ("image_first", "text_first", "question_first", "neutral_first")


def runs(prefix: str, ids: list, pool: int = 4) -> dict:
    tags = {x: (f"{prefix}_{x}" if prefix != "rlens" or x != "image_first" else "rlens") for x in LAYOUTS}
    return {x: Q.load_patch_layers(ids, pool=pool, tag=t) for x, t in tags.items()}


def word_counts(run: dict, layers: list) -> dict:
    w = run["words"][run["words"]["layer"].isin(layers)]
    low = w["word"].str.lower()
    hit = w[low.isin(QUESTION)]["word"].value_counts().to_dict()
    return {"blocks": len(w), "question": int(low.isin(QUESTION).sum()), "words": hit,
            "filler": int(low.isin(FILLER).sum()), "norwood": sorted(w[low.isin(NORWOOD)]["word"])}


def top3(tag: str, names: list, layers: dict, pool: int = 4) -> pd.DataFrame:
    """Give the top 3 tokens of each block, keyed by the matched R-lens layer."""
    out = []
    for n in names:
        d = pd.read_parquet(Q.RQ1 / "patch_words" / tag / f"{n}_p{pool}.jlens.parquet")
        d = d[d["layer"].isin(layers)].assign(layer=lambda f: f["layer"].map(layers), image=n)
        out.append(d.assign(word=d["token"].str.strip().str.lower()))
    return pd.concat(out, ignore_index=True)


def main() -> int:
    prefix, pool = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
    sample = pd.read_parquet(Q.RQ1 / "patch_words_sample.parquet")
    ids = sample["photo_id"].tolist()
    names = [f"{r.photo_id}_{r.lens_case}" for r in sample.itertuples()]
    J, R = runs(prefix, ids, pool), runs("rlens", ids)
    layers = sorted(J["image_first"]["words"]["layer"].unique())
    pairs = {"question: question_first vs neutral_first": ("question_first", "neutral_first"),
             "filler: neutral_first vs image_first": ("neutral_first", "image_first"),
             "text_first vs image_first": ("text_first", "image_first")}
    for name, runs_of in (("J++", J), ("R-lens", R)):
        print(name, "changed blocks (%)")
        print(Q.patch_change_table({k: (runs_of[a], runs_of[b]) for k, (a, b) in pairs.items()}).T.to_string())
    deep = {"J++": layers[-4:], "R-lens": list(RLENS[-4:])}
    for name, runs_of in (("J++", J), ("R-lens", R)):
        for x in LAYOUTS:
            print(name, x, word_counts(runs_of[x], deep[name]))
    print("Norwood words, any depth:", {f"{n} {x}": word_counts(r[x], list(r[x]["words"]["layer"].unique()))["norwood"]
                                         for n, r in (("J++", J), ("R-lens", R)) for x in LAYOUTS})
    keys = ["image", "layer", "patch_row", "patch_col"]
    a = top3(f"{prefix}_image_first", names, dict(zip(layers, RLENS)), pool)
    b = top3("rlens", names, {x: x for x in RLENS})
    one = a[a["rank"] == 0].merge(b[b["rank"] == 0], on=keys, suffixes=("_j", "_r"))
    both = a.merge(b, on=keys + ["word"]).groupby(keys).size().rename("shared").reset_index()
    print("J++ and R-lens agree, image first, by matched R-lens layer (%):")
    print(pd.DataFrame({"top-1": one.assign(same=one["word_j"] == one["word_r"]).groupby("layer")["same"].mean() * 100,
                        "any of top 3": both.groupby("layer").size() / one.groupby("layer").size() * 100}).round(1).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
