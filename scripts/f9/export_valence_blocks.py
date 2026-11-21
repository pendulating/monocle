#!/usr/bin/env python
"""The F9 win blocks: the cues of the subjective answers, in order of net valence (D147).

F9 has no pairs, thus it has no win rate. The block keeps the layout of
`scripts/export_wordblocks.py` and takes this score:

| Item | Rule |
|------|------|
| Text outcome | the net valence of method B, as in H9 |
| Cue | a term inside a span: a content word, or 2 adjacent content words. A span of these schemas is a clause (median 9 to 15 words), thus whole spans almost never repeat |
| Negation | a term in the scope of a negator takes the form `not <term>`, and it is a separate cue (`terms`). Thus "not upscale" does not score `upscale` |
| Score of 1 cue | for each text that holds the term, the net valence of the spans that hold it; then the mean over those texts. 1 vote for each text |
| Shrink | toward the mean net valence of all texts, with a prior of 25 texts (`_extractions.win_rates`) |
| Photographs | the images with the highest and the lowest net valence, pooled over the spans of the 6 models |

A word leaves the list when 1 of its 2-word terms holds most of its texts.

`--sources` picks the spans. `visual_evidence` reads the valence of the F4
schema. `inference` reads the social inferences, with the valence of
`label_inference_valence.py`. These spans are sparse, thus the second block
pools the 6 models for the cues, and it has lower floors for the images
(`SOURCES`): no image holds 30 of these spans.

Warning: do not score a cue by the rest of its text. Removing a `bad` span
raises the net valence of what remains, and removing a `good` span lowers it.
That score ranks `distractions` first and `nice` near the end in the street
case. `rest_valence` in the cue table keeps it for that check, and the block
does not use it.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f9/export_valence_blocks.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (HERE, REPO_ROOT / "notebooks" / "cvpr", REPO_ROOT / "notebooks" / "facct"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

__version__ = "1.6.0"

OUT = REPO_ROOT / "notebooks" / "facct" / "figures" / "f9_blocks"
FIELD = "response"
MODEL = "gemma-4-12b"
MODEL_SHORT = "Gemma-4-12B"
MIN_TEXTS = 25
MAX_CUES = 300
TOKEN = re.compile(r"[a-z][a-z'-]+|[.;:!?\u2014\u2013]")
NEGATORS = frozenset({"not", "no", "never", "none", "nothing", "neither", "nor", "without", "lack", "lacks", "lacking",
                      "less", "unlikely", "hardly", "barely", "cannot", "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't", "didn't",
                      "won't", "wouldn't", "can't", "couldn't", "shouldn't", "ain't"})
# A negator stays in force up to the end of its clause. A comma or a bracket does not end it: "not a fancy, high-end
# spot". An answer also denies a standard after it names it: "If you want a high-end meal, this is not it" (`terms`).
SCOPE_END = frozenset({"but", "though", "although", "however", "yet", "while", "because", "instead", "which"})
# A span that opens with 1 of these continues its sentence, thus a negator before the span can govern it:
# "[nothing] that would suggest a high-end meal". An independent clause does not inherit: "It is not fancy, it looks clean".
FRAGMENT = frozenset({"that", "to", "or", "and", "like"})
# Words of the frame of the question, or of the image itself: they name no thing in the scene.
FRAME_WORDS = frozenset({"image", "images", "photo", "photos", "photograph", "picture", "pictures", "shot", "shots",
                         "look", "looks", "looking", "appears", "appear", "seems", "seem", "visible", "shows", "show",
                         "see", "seen", "like", "feel", "feels", "restaurant", "restaurants", "street", "streets",
                         "having", "it's", "isn't", "provide", "provides", "offer", "offers", "add", "adds", "make",
                         "makes", "create", "creates", "act", "acts", "takes", "draw", "says", "called", "located",
                         "situated", "features", "including", "showing", "shown", "directly", "clearly",
                         "specifically", "currently", "somewhat", "quite", "bit", "lot", "allow", "allows"})
# An inference span often carries the verb of the inference and a hedge: these words say how it is put, not what it is about.
HEDGE_WORDS = frozenset({"suggest", "suggests", "suggesting", "suggested", "indicate", "indicates", "indicating", "imply",
                         "implies", "means", "signal", "signals", "signaling", "convey", "conveys", "gives", "reads",
                         "usually", "typically", "generally", "likely", "probably", "possibly", "potentially",
                         "certainly", "definitely", "necessarily", "particularly", "especially", "exactly", "overly",
                         "does", "that's", "you'd"})
PRIOR = 25.0
MIN_IMAGE_SPANS = 30
PHOTOS = 6
# The inference block of the paper is a side figure at about half of the text width. Thus it shows fewer cues in a
# larger type, and 4 photographs in each row instead of 6, so that a reader can read it at that width.
PAPER = {"max_words": 150, "fontsize": 19.0, "label_fontsize": 14.0, "width_in": 11.0, "unit_photos": 4, "photo_in": 1.8}
SUBJ = REPO_ROOT / "data" / "f4_traces" / "subj"
# A source: its words in the caption, the model of the cues (None pools all), and the floors of an image.
SOURCES = {
    "visual_evidence": {"stem": "valence", "span": "a span of visual evidence", "model": MODEL, "who": MODEL_SHORT,
                        "min_spans": MIN_IMAGE_SPANS, "min_models": None},
    "inference": {"stem": "inference", "span": "a social inference (upkeep, wealth, demographic, class, safety)",
                  "model": None, "who": "the 6 models, pooled", "min_spans": 3, "min_models": 3},
}
CASES = {
    "restaurant": ("cues", "restaurant_subj_x2", SUBJ / "f4_restaurants_10k.parquet", "restaurants",
                   "Does this look like a nice restaurant to eat at?"),
    "street": ("cues_street", "street_subj_x2", SUBJ / "f4_street_10k.parquet", "street photography",
               "Would this street make a good spot for taking a photo?"),
}


def _tokens(text) -> list:
    return TOKEN.findall(str(text).lower().replace("\u2019", "'"))


def _open_negation(before) -> bool:
    """Say if a negator is still in force at the end of the text before a span."""
    open_ = False
    for tok in _tokens(before):
        if tok in NEGATORS:
            open_ = True
        elif not tok[0].isalpha() or tok in SCOPE_END:
            open_ = False
    return open_


def terms(text: str, drop: frozenset = frozenset(), before: str = "") -> list:
    """Give the content words of 1 span and the pairs of adjacent content words, with no repeats.

    `drop` adds stop words. `before` is the text before the span, for a span that opens as a fragment.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    tokens = _tokens(text)
    uni, bi, last = [], [], None
    # The standard of "If you want X, this is not it" stays negated up to the negator, past any scope end.
    held = negated = tokens[:1] == ["if"] and tokens[1:2] in (["you"], ["your"]) and bool(NEGATORS & set(tokens))
    negated = negated or (tokens[:1] != [] and tokens[0] in FRAGMENT and _open_negation(before))
    for i, tok in enumerate(tokens):
        if tok in NEGATORS or (tok == "rather" and tokens[i + 1:i + 2] == ["than"]) or (tok == "opposite" and tokens[i + 1:i + 2] == ["of"]):
            negated, held, last = True, False, None
            continue
        if held:
            pass
        elif not tok[0].isalpha() or (tok in SCOPE_END and tokens[max(i - 1, 0)] != "of"):
            negated, last = False, None
            continue
        if not tok[0].isalpha():
            last = None
            continue
        if len(tok) < 3 or tok in ENGLISH_STOP_WORDS or tok in FRAME_WORDS or tok in drop:
            last = None
            continue
        mark = "not " if negated else ""
        uni.append(mark + tok)
        if last:
            bi.append(f"{mark}{last} {tok}")
        last = tok
    return sorted(set(uni + bi))


def valence_spans(case: str) -> pd.DataFrame:
    """Give the `visual_evidence` spans with a valence of the `response` field: 1 row each, with `terms`."""
    import analyze_subjective as A
    import _f4 as F

    arm, name = CASES[case][:2]
    raw = pd.read_parquet(A.X2 / arm / f"f4_{name}_extractions.parquet", filters=[("field", "==", FIELD)])
    raw = raw[~raw.copy_suspect.astype(bool)].assign(case=case, prompt="f9")
    sp = A.spans_of(raw)
    sp = sp[(sp.extraction_class == "visual_evidence") & sp.valence.isin(F.VALENCES)]
    sp = sp.assign(terms=[terms(t) for t in sp.extraction_text])
    return sp[["doc_id", "model", "image_idx", "valence", "terms"]].reset_index(drop=True)


def inference_spans(case: str) -> pd.DataFrame:
    """Give the labelled social inference spans of the `response` field, with the columns of `valence_spans`."""
    root = SUBJ / "inference_valence"
    sp = pd.read_parquet(root / "input.parquet", columns=["span_id", "doc_id", "case", "model", "field", "image_idx",
                                                          "extraction_text", "before"], filters=[("case", "==", case), ("field", "==", FIELD)])
    sp = sp.merge(pd.read_parquet(root / "full.parquet", columns=["span_id", "valence"]), on="span_id")
    sp = sp.assign(terms=[terms(t, HEDGE_WORDS, b) for t, b in zip(sp.extraction_text, sp.before)])
    return sp[["doc_id", "model", "image_idx", "valence", "terms"]].reset_index(drop=True)


def _drop_covered(t: pd.DataFrame) -> pd.DataFrame:
    """Drop a word when 1 of its 2-word terms holds over 60% of its texts."""
    def split(cue: str) -> tuple:
        mark, words = ("not ", cue[4:]) if cue.startswith("not ") else ("", cue)
        return mark, words.split()

    n = dict(zip(t.cue, t.n))
    best: dict = {}
    for cue, k in n.items():
        mark, words = split(cue)
        if len(words) > 1:
            for w in words:
                best[mark + w] = max(best.get(mark + w, 0), k)
    return t[[not (len(split(c)[1]) == 1 and best.get(c, 0) > 0.6 * k) for c, k in zip(t.cue, t.n)]]


def cue_scores(sp: pd.DataFrame, model: str | None = MODEL) -> pd.DataFrame:
    """Give the score of the `MAX_CUES` commonest terms of 1 model, or of all with None: `cue`, `n`, `net_valence`, `shrunk_rate`, `base_rate`.

    `rest_valence` is the score of the rest of the text (see the warning of the module).
    """
    part = (sp if model is None else sp[sp.model == model]).assign(good=lambda d: d.valence.eq("good").astype(int),
                                        bad=lambda d: d.valence.eq("bad").astype(int))
    text = part.groupby("doc_id").agg(g=("good", "sum"), b=("bad", "sum"), s=("good", "size"))
    base = float(((text.g - text.b) / text.s).mean())
    long = part.explode("terms").dropna(subset=["terms"]).rename(columns={"terms": "cue"})
    cue = long.groupby(["doc_id", "cue"]).agg(gc=("good", "sum"), bc=("bad", "sum"), sc=("good", "size")).reset_index()
    cue = cue.merge(text, left_on="doc_id", right_index=True)
    cue["own"] = (cue.gc - cue.bc) / cue.sc
    cue["rest"] = ((cue.g - cue.gc) - (cue.b - cue.bc)) / (cue.s - cue.sc).where(cue.s > cue.sc)
    t = cue.groupby("cue").agg(n=("own", "size"), total=("own", "sum"), net_valence=("own", "mean"),
                               rest_valence=("rest", "mean")).reset_index()
    t = _drop_covered(t[t.n >= MIN_TEXTS]).sort_values("n", ascending=False).head(MAX_CUES).copy()
    t["shrunk_rate"] = ((t.total + PRIOR * base) / (t.n + PRIOR)).round(4)
    t["base_rate"] = round(base, 4)
    return t.drop(columns="total").sort_values("shrunk_rate", ascending=False).reset_index(drop=True)


def image_table(sp: pd.DataFrame, case: str, min_spans: int = MIN_IMAGE_SPANS, min_models: int | None = None) -> pd.DataFrame:
    """Give the images in order of net valence, with the columns of `unit_win_rates`.

    `min_models` of None asks for all the models.
    """
    ima = pd.read_parquet(CASES[case][2])
    g = sp.assign(good=sp.valence.eq("good"), bad=sp.valence.eq("bad")).groupby("image_idx")
    t = g.agg(good=("good", "sum"), bad=("bad", "sum"), n=("good", "size"), models=("model", "nunique")).reset_index()
    t = t[(t.models >= (min_models or sp.model.nunique())) & (t.n >= min_spans)]
    # On a tie, good - bad puts the image with more spans at the outer end, at both ends of the table.
    t = t.assign(net=(t.good - t.bad) / t.n, margin=t.good - t.bad).merge(ima, on="image_idx")
    # 2 restaurants of 1 building can share 1 capture, thus the rows of photographs keep 1 image of each.
    t = t.sort_values(["net", "margin"], ascending=False).drop_duplicates("recording_id")
    name = t["dba"] if "dba" in t.columns else t["nta_name"]
    # `_photo_row` reads `shrunk_rate` on a scale of 0 to 1, with the middle at 0.5: here that is a net valence of 0.
    t = t.assign(unit=t.image_idx, unit_name=name, win_rate=(t.net + 1) / 2, shrunk_rate=(t.net + 1) / 2,
                 caption=[f"net {v:+.2f}, {n} spans" for v, n in zip(t.net, t.n)])
    return t.sort_values(["net", "margin"], ascending=False).reset_index(drop=True)


def caption(case: str, table: pd.DataFrame, n_texts: int, shown: int, photos: int, source: str = "visual_evidence") -> str:
    what, src = CASES[case][3], SOURCES[source]
    models = f"{src['min_models']} models or more" if src["min_models"] else "all 6 models"
    part = f"{shown} of the {len(table)} cues" if shown < len(table) else f"the {len(table)} cues"
    return (f"The cues of the {what} answers to the subjective prompt (\"{CASES[case][4]}\"), {src['who']}, final "
            f"answers, {n_texts:,} texts. A cue is a word or a pair of adjacent words inside {src['span']}. "
            f"The block shows {part}, the commonest with {MIN_TEXTS} texts or more, in the order of their score, "
            "highest first, and the colour repeats the order. The score of a cue is the net valence of the spans "
            "that hold it, with 1 vote for each text, shrunk toward the mean net valence of all the texts "
            f"({table.base_rate.iloc[0]:+.2f}, the middle of the colour bar). The {photos} photographs above are the "
            f"images with the highest net valence, the {photos} below the lowest, pooled over the 6 models, with "
            f"{src['min_spans']} spans or more from {models}; on a tie, the image with more spans goes to the outer end. The frames of the photographs use a middle of 0, not the middle of "
            "the bar. A term after a negator in its clause forms a separate cue: `not upscale` is not `upscale`. The score "
            "says how the answers frame a cue, not that a cue causes a judgment.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--sources", nargs="*", default=list(SOURCES), choices=list(SOURCES))
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import _extractions as X
    import _figmeta as FM
    import _style as S

    S.apply_house_style()
    OUT.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        src = SOURCES[source]
        for case in args.cases:
            sp = (valence_spans if source == "visual_evidence" else inference_spans)(case)
            table = cue_scores(sp, src["model"])
            units = image_table(sp, case, src["min_spans"], src["min_models"])
            stem = f"{case}_{src['stem']}_block"
            table.to_csv(OUT / f"{stem}_cues.csv", index=False)
            units[["image_idx", "unit_name", "net", "good", "bad", "n", "image_path"]].to_csv(OUT / f"{stem}_images.csv", index=False)
            n_texts = int(sp[sp.model == src["model"]].doc_id.nunique() if src["model"] else sp.doc_id.nunique())
            print(f"[{stem}] {len(sp):,} spans, {n_texts:,} texts, {len(table)} cues, {len(units):,} images, "
                  f"centre {table.base_rate.iloc[0]:+.3f}, range {table.shrunk_rate.min():+.3f} to "
                  f"{table.shrunk_rate.max():+.3f}", flush=True)
            bar = f"net valence of the cue  (all texts {table.base_rate.iloc[0]:+.2f})"
            # The inference block of the paper: no business names under the photographs, and centred lines.
            wide = {"center_words": True, "photo_names": False, **PAPER} if source == "inference" else {"unit_photos": PHOTOS}
            if source == "inference":
                bar = "net valence of the inference"
            fig, shown = X.plot_win_block(pd.DataFrame(), case, units=units, table=table, bar_label=bar, **wide)
            _save(FM, plt, fig, stem, case, caption(case, table, n_texts, len(shown), wide["unit_photos"], source), source)
            fig, shown = X.plot_win_block_column(pd.DataFrame(), case, units=units, table=table, bar_label=bar)
            _save(FM, plt, fig, f"{stem}_column", case, caption(case, table, n_texts, len(shown), 3, source), source)
    return 0


def _save(FM, plt, fig, stem: str, case: str, text: str, source: str) -> None:
    FM.save_figure(fig, OUT / f"{stem}.png", dpi=300, title=f"Cue valence, {CASES[case][3]}", caption=text,
                   label=f"fig:{stem.replace('_', '-')}", tags=["f9", "wordblocks", case, "valence", source],
                   provenance={"script": "scripts/f9/export_valence_blocks.py", "version": __version__,
                               "model": SOURCES[source]["model"] or "pooled", "field": FIELD})
    plt.close(fig)
    print(f"[blocks] {stem}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
