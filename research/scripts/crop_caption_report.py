"""What words does the model use for the causal square?

The crop captions come from `monocle.patch_caption`. Each photograph gives 1
crop at the map's highest cell, 1 at its lowest, and 4 arbitrary cells. This
report asks 2 questions.

| Question | Test |
|---|---|
| Is a pole crop described differently from a random crop? | permutation |
| Are the 2 poles described differently from each other? | permutation |

The permutation stays INSIDE a photograph: the arm labels of the 6 crops of
one photograph are shuffled, and the statistic is rebuilt. So the scene, the
camera and the light never move, and only the role of the crop changes.

The statistic is the Jensen-Shannon divergence between the 2 word counts.
The words that drive it come from Monroe's log-odds with an informative
Dirichlet prior.

Usage:
    python research/scripts/crop_caption_report.py \
        --dir outputs/_monocle/patch_caption/restaurants_causal
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_word_report import js, log_odds  # noqa: E402

WORD = re.compile(r"[a-z]{3,}")
STOP = {"the", "and", "can", "see", "you", "this", "that", "with", "for",
        "are", "there", "here", "image", "picture", "following", "list",
        "short", "things", "some", "its", "his", "her", "они", "photo",
        "photograph", "shows", "showing", "appears", "seems", "likely",
        "possibly", "part", "parts", "视", "look", "looks", "into", "from",
        "over", "under", "near", "very", "more", "than", "also", "which",
        "have", "has", "was", "were", "been", "being", "one", "two"}


def words_of(text: str) -> list[str]:
    return [w for w in WORD.findall(str(text).lower()) if w not in STOP]


def bag(df: pd.DataFrame, vocab: list[str]) -> np.ndarray:
    idx = {w: i for i, w in enumerate(vocab)}
    out = np.zeros(len(vocab), dtype=float)
    for t in df.text:
        for w in words_of(t):
            j = idx.get(w)
            if j is not None:
                out[j] += 1
    return out


def vocab_of(df: pd.DataFrame, arms: tuple, min_count: int) -> list[str]:
    """Words seen `min_count` times in the 2 arms under test.

    The vocabulary must come from the arms being compared. A word that only
    a third arm uses has 0 counts on both sides, and Monroe's prior then
    divides by 0.
    """
    sub = df[df.arm.isin(arms)]
    all_words = [w for t in sub.text for w in words_of(t)]
    keep = pd.Series(all_words).value_counts()
    return sorted(keep[keep >= min_count].index)


def stat(df: pd.DataFrame, a: str, b: str, vocab: list[str]) -> float:
    return js(bag(df[df.arm == a], vocab), bag(df[df.arm == b], vocab))


def target_rate(df: pd.DataFrame, words: set) -> pd.Series:
    """For each crop, 1 when the caption names a target word."""
    return df.text.map(lambda t: float(bool(set(words_of(t)) & words)))


def directed(df: pd.DataFrame, words: set, a: str, b: str) -> dict:
    """Does arm `a` name the target object more often than arm `b`?

    The comparison is INSIDE a photograph, so the scene is held fixed, and
    the test is a Wilcoxon over the photographs.
    """
    from scipy.stats import wilcoxon
    d = df.copy()
    d["hit"] = target_rate(d, words)
    rows = []
    for img, g in d.groupby("image_id"):
        ga, gb = g[g.arm == a], g[g.arm == b]
        if ga.empty or gb.empty:
            continue
        rows.append(float(ga.hit.mean()) - float(gb.hit.mean()))
    x = np.asarray(rows, dtype=float)
    out = {"n": int(x.size), "mean": float(x.mean()) if x.size else float("nan"),
           "rate_a": float(d[d.arm == a].hit.mean()),
           "rate_b": float(d[d.arm == b].hit.mean())}
    if x.size > 5 and not np.allclose(x, 0):
        out["p"] = float(wilcoxon(x, alternative="greater").pvalue)
    return out


def permute(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    arm = out.arm.to_numpy(dtype=object).copy()
    for _, idx in out.groupby("image_id").groups.items():
        pos = out.index.get_indexer(idx)
        arm[pos] = rng.permutation(arm[pos])
    out["arm"] = arm
    return out


def report(path: Path, n_perm: int, seed: int, n_show: int,
           min_count: int, targets: set | None = None) -> dict:
    df = pd.read_parquet(path / "captions.parquet").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    out: dict = {"dir": str(path), "n_images": int(df.image_id.nunique()),
                 "n_crops": int(len(df)), "tests": {}, "directed": {}}
    for a, b in (("pos", "random"), ("neg", "random"), ("pos", "neg")):
        vocab = vocab_of(df, (a, b), min_count)
        real = stat(df, a, b, vocab)
        null = np.array([stat(permute(df, rng), a, b, vocab)
                         for _ in range(n_perm)])
        p = float((1 + int((null >= real).sum())) / (1 + len(null)))
        z = log_odds(bag(df[df.arm == a], vocab), bag(df[df.arm == b], vocab))
        order = np.argsort(z)
        out["tests"][f"{a}_vs_{b}"] = {
            "vocab": len(vocab),
            "js": float(real), "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)), "p": p,
            f"top_{a}": [vocab[i] for i in order[::-1][:n_show]],
            f"top_{b}": [vocab[i] for i in order[:n_show]]}
    if targets:
        for a, b in (("pos", "random"), ("neg", "random")):
            out["directed"][f"{a}_vs_{b}"] = directed(df, targets, a, b)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-show", type=int, default=12)
    ap.add_argument("--min-count", type=int, default=4)
    ap.add_argument("--target-words", nargs="*", default=None,
                    help="words that name the object judged")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    r = report(Path(args.dir), args.n_perm, args.seed, args.n_show,
               args.min_count,
               set(args.target_words or []) or None)
    print(f"{r['dir']}: {r['n_images']} photographs, "
          f"{r['n_crops']} crops")
    for name, t in r["tests"].items():
        print(f"  {name:>14}  {t['vocab']:>3} words  JS={t['js']:.4f}  "
              f"null={t['null_mean']:.4f}+-{t['null_sd']:.4f}  "
              f"p={t['p']:.4f}")
        for k, v in t.items():
            if k.startswith("top_"):
                print(f"      {k:>12}: {', '.join(v)}")
    for name, t in r.get("directed", {}).items():
        print(f"  target words, {name}: rate {t['rate_a']:.3f} against "
              f"{t['rate_b']:.3f}, paired {t['mean']:+.3f}, "
              f"p={t.get('p', float('nan')):.4f}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
