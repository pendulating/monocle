"""Do the 2 poles of the causal map carry different words, and are the words readable?

`monocle.causal_words` reads the lens at the 2 poles of the MEASURED causal
map, and at random patches of the same photograph. This script scores 3
questions over every case and every fitted layer.

| Block | Question | The null |
|---|---|---|
| poles | Do the 2 poles carry different words? | Shuffle the pole label inside a photograph. |
| readable | Are the words words at all, and do they name the subject? | A junk rate, and the target list of the case. |
| caption | Do the words name what the crop holds? | The caption of ANOTHER photograph. |

The pole block is the test that failed with the J-lens at block 18. Run this
against a J run and an R run of the same shape to see whether the lens, and
not the poles, was the limit.

Warning: a permutation must move a whole PATCH, not 1 word row. The 10 words
of 1 patch are not independent, so a row shuffle makes the null too tight.

Usage:
    python scripts/rlens_word_report.py --root outputs/_monocle/causal_words/rlens8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.k1_target_words import TARGET_WORDS  # noqa: E402

JUNK = re.compile(r"^[\s\W_]*$")
ASCII_WORD = re.compile(r"^[a-z][a-z\-']*$")
CAP_ROOT = REPO / "outputs/_monocle/patch_caption_reader"


def clean(w: str) -> str:
    return str(w).strip().lower()


def js(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        ok = a > 0
        return float(np.sum(a[ok] * np.log(a[ok] / b[ok])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def bag(words: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    out = np.zeros(len(vocab), dtype=float)
    for w, n in words.value_counts().items():
        j = vocab.get(w)
        if j is not None:
            out[j] += n
    return out


def bh(p: list[float]) -> list[float]:
    """Benjamini-Hochberg. Returns q in the order of the input."""
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return [float(x) for x in q]


# --- block 1: the 2 poles against each other -------------------------------

def pole_test(df: pd.DataFrame, n_perm: int, seed: int) -> dict:
    """JS between the pos bag and the neg bag, against a within-photograph null.

    The unit of the shuffle is a PATCH. Each photograph holds the same count
    of pos patches and neg patches, so the label swap is exchangeable.

    Each patch becomes 1 row of a count matrix. A permutation is then a sum
    of rows, and not a join, so 2000 draws cost seconds.
    """
    d = df[df.role.isin(["pos", "neg"])]
    if d.empty or d.role.nunique() < 2:
        return {}
    vocab = {w: i for i, w in enumerate(sorted(d.word.unique()))}
    d = d.assign(_w=d.word.map(vocab))

    units = d.groupby(["image_id", "patch"], sort=True)
    keys = list(units.groups)
    M = np.zeros((len(keys), len(vocab)), dtype=np.float32)
    role = np.empty(len(keys), dtype=object)
    img = np.empty(len(keys), dtype=object)
    for i, (k, g) in enumerate(units):
        np.add.at(M[i], g["_w"].to_numpy(), 1.0)
        role[i] = g["role"].iloc[0]
        img[i] = k[0]
    is_pos = role == "pos"
    obs = js(M[is_pos].sum(0), M[~is_pos].sum(0))

    # The photograph blocks, so a shuffle never moves a patch between scenes.
    blocks = {}
    for i, k in enumerate(img):
        blocks.setdefault(k, []).append(i)
    blocks = [np.asarray(v) for v in blocks.values()]

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for t in range(n_perm):
        mask = np.zeros(len(keys), dtype=bool)
        for b in blocks:
            mask[rng.permutation(b)[:int(is_pos[b].sum())]] = True
        null[t] = js(M[mask].sum(0), M[~mask].sum(0))
    return {"js": obs, "null_median": float(np.median(null)),
            "p": float((np.sum(null >= obs) + 1) / (null.size + 1)),
            "n_words": len(vocab)}


# --- block 2: are the words readable ---------------------------------------

def readable(df: pd.DataFrame, case: str) -> dict:
    """A junk rate, and the share of words on the target list of the case."""
    targets = set(TARGET_WORDS.get(case, []))
    out = {}
    for role in ("pos", "neg", "random"):
        w = df[df.role == role].word
        if w.empty:
            continue
        n = len(w)
        junk = float(np.mean([not bool(ASCII_WORD.match(x)) for x in w]))
        hit = float(np.mean([x in targets for x in w]))
        out[role] = {"n": int(n), "junk": junk, "target": hit}
    return out


# --- block 3: do the words name what the crop holds ------------------------

def caption_test(df: pd.DataFrame, case: str, n_perm: int, seed: int) -> dict:
    """Score the lens words of a pole patch against the caption of that crop.

    The null gives the same words the caption of ANOTHER photograph. So the
    lens words must name THIS scene, and not a street scene in general.
    """
    path = CAP_ROOT / f"{case}_causal" / "captions.parquet"
    if not path.exists():
        return {}
    cap = pd.read_parquet(path)
    cap = cap[cap.arm.isin(["pos", "neg"])]
    text = {(str(r.image_id), str(r.arm)): set(
        re.findall(r"[a-z][a-z\-']*", str(r.text).lower()))
        for r in cap.itertuples()}
    if not text:
        return {}

    out = {}
    rng = np.random.default_rng(seed)
    for role in ("pos", "neg"):
        # Keep the words of the pole patch that the caption run also chose.
        cells = {k[0]: None for k in text if k[1] == role}
        d = df[(df.role == role) & (df.image_id.isin(cells))]
        if d.empty:
            continue
        per = {img: [w for w in g.word if ASCII_WORD.match(w)]
               for img, g in d.groupby("image_id")}
        per = {k: v for k, v in per.items() if v}
        if not per:
            continue
        obs = float(np.mean([
            np.mean([w in text[(img, role)] for w in ws])
            for img, ws in per.items()]))
        imgs = list(per)
        null = []
        for _ in range(n_perm):
            other = rng.permutation(imgs)
            vals = []
            for img, alt in zip(imgs, other):
                if alt == img or (alt, role) not in text:
                    continue
                vals.append(np.mean([w in text[(alt, role)] for w in per[img]]))
            if vals:
                null.append(float(np.mean(vals)))
        null = np.asarray(null)
        out[role] = {"match": obs, "null_median": float(np.median(null)),
                     "p": float((np.sum(null >= obs) + 1) / (null.size + 1)),
                     "n_images": len(per)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                    help="a causal_words tag directory")
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--top-rank", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-perm-cap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    root = Path(a.root)
    cases = a.cases or sorted(p.name for p in root.iterdir() if p.is_dir())
    rows: list[dict] = []
    for case in cases:
        f = root / case / "words.parquet"
        if not f.exists():
            print(f"[skip] {case}: no words.parquet")
            continue
        df = pd.read_parquet(f)
        df["word"] = df["word"].map(clean)
        df = df[(df["rank"] < a.top_rank)
                & (~df["word"].map(lambda w: bool(JUNK.match(w))))]
        keep = df["word"].value_counts()
        df = df[df["word"].isin(keep[keep >= a.min_count].index)]
        for layer in sorted(df.layer.unique()):
            g = df[df.layer == layer]
            row = {"case": case, "layer": int(layer)}
            row.update({f"pole_{k}": v
                        for k, v in pole_test(g, a.n_perm, a.seed).items()})
            for role, m in readable(g, case).items():
                row[f"junk_{role}"] = m["junk"]
                row[f"target_{role}"] = m["target"]
            for role, m in caption_test(g, case, a.n_perm_cap,
                                        a.seed).items():
                row[f"cap_{role}"] = m["match"]
                row[f"cap_{role}_null"] = m["null_median"]
                row[f"cap_{role}_p"] = m["p"]
            rows.append(row)
            print(f"  {case:<20} L{layer:<3} done", flush=True)

    out = pd.DataFrame(rows)
    for col, q in (("pole_p", "pole_q"), ("cap_pos_p", "cap_pos_q"),
                   ("cap_neg_p", "cap_neg_q")):
        if col in out.columns:
            ok = out[col].notna()
            out.loc[ok, q] = bh(list(out.loc[ok, col]))

    dest = Path(a.out) if a.out else root / "report.csv"
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    with pd.option_context("display.width", 200,
                           "display.max_columns", 40):
        print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
