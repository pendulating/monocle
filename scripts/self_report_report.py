"""Does the model's own explanation name what the reader points at?

The model answers the case question, then names the 1 thing that most
made it answer that way. We ask whether that thing appears in the
caption of the cell the reader names, more than in the caption of a
random cell of the SAME photograph.

    python scripts/self_report_report.py \
        --dir outputs/_monocle/self_report/restaurants_gxa
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

STOP = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "with",
    "is", "are", "was", "were", "it", "its", "this", "that", "these",
    "those", "there", "here", "for", "from", "by", "as", "be", "been",
    "i", "you", "we", "they", "he", "she", "picture", "image", "photo",
    "photograph", "can", "see", "seen", "one", "thing", "things", "some",
    "any", "not", "no", "very", "more", "most", "much", "look", "looks",
    "appears", "appear", "shows", "show", "shown", "answer", "answered",
    "good", "poor", "safe", "dangerous", "made", "me", "my", "list",
    "short", "give", "name", "named", "which", "what", "cannot", "s",
}


def words(text: str) -> set[str]:
    got = re.findall(r"[a-z]+", str(text).lower())
    return {w for w in got if len(w) > 2 and w not in STOP}


def hit(report: str, caption: str) -> float:
    """1 when a content word of the report is in the caption."""
    a, b = words(report), words(caption)
    return float(bool(a & b))


def share(report: str, caption: str) -> float:
    """The share of the report's words that the caption also holds."""
    a, b = words(report), words(caption)
    return float(len(a & b) / len(a)) if a else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.dir)
    hits = sorted(root.glob("**/reports.parquet"))
    if not hits:
        raise SystemExit(f"no reports.parquet under {root}")
    df = pd.concat([pd.read_parquet(h) for h in hits], ignore_index=True)
    case = str(df["case"].iloc[0])
    n_img = df["image_id"].nunique()
    print(f"{root}: {case}, {n_img} photographs, {len(df)} crops")

    rows = []
    for arm in ("pos", "neg"):
        pair_h, pair_s, keys = [], [], []
        for img, g in df.groupby("image_id"):
            a = g[g["arm"] == arm]
            r = g[g["arm"] == "random"]
            if a.empty or r.empty:
                continue
            rep = str(a["report"].iloc[0])
            pair_h.append((hit(rep, a["text"].iloc[0]),
                           float(np.mean([hit(rep, t)
                                          for t in r["text"]]))))
            pair_s.append((share(rep, a["text"].iloc[0]),
                           float(np.nanmean([share(rep, t)
                                             for t in r["text"]]))))
            keys.append(img)
        H = np.array(pair_h)
        S = np.array(pair_s)
        for name, M in (("any word", H), ("share of words", S)):
            d = M[:, 0] - M[:, 1]
            ok = np.isfinite(d)
            d = d[ok]
            try:
                pv = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                pv = 1.0
            rows.append({"case": case, "arm": arm, "statistic": name,
                         "n": int(d.size),
                         "reader": float(M[ok, 0].mean()),
                         "random": float(M[ok, 1].mean()),
                         "paired": float(d.mean()), "p": pv})

    print(f"{'arm':>5} {'statistic':>16} {'reader':>8} {'random':>8} "
          f"{'paired':>8} {'p':>8}")
    for r in rows:
        print(f"{r['arm']:>5} {r['statistic']:>16} {r['reader']:8.3f} "
              f"{r['random']:8.3f} {r['paired']:+8.3f} {r['p']:8.4f}")

    reps = df.drop_duplicates("image_id")["report"]
    from collections import Counter
    c = Counter(w for t in reps for w in words(t))
    print("\nthe words the model gives as its reason: "
          + ", ".join(w for w, _ in c.most_common(15)))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
