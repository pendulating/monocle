"""Which residual dimensions does the causal patch reader use?

The fitted reader is `sum_d w_d * g_d * h_d`. The anatomy sweep showed
that a top-m weight already carries much of the gain, so the weight is
not spread evenly over the 3840 dimensions. This script asks 2 things.

1. Do the cases pick the SAME dimensions? The full weight vectors have a
   cosine near 0, and yet a weight fitted on 1 case reads another. If
   the top dimensions overlap, the shared part is a small subspace and
   the rest of the vector is noise that the cosine counts.

2. What does a dimension mean? Gemma ties its output embedding, so the
   row of the embedding matrix for a token gives that token's weight on
   every dimension. Column `d` thus ranks the vocabulary by how much
   dimension `d` pushes each word. That is a logit lens on 1 coordinate.

    python research/scripts/k1_reader_dims.py --cases restaurants libraries \\
        subway_safety schools street_photography --layer 20 --top 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k1_fitted_probe as fp  # noqa: E402

MODEL_DIR = "/share/ju/matt/zoo/models/gemma-4-12B-it"


def weight_of(case: str, own: Path, layer: int, alpha: float) -> tuple:
    A = fp.load_acts(own, case, layer, "gh")
    keys = sorted(A)
    w = fp.Gram(A, keys).fit(alpha)
    sq = np.zeros(w.size)
    n = 0
    for k in keys:
        X = fp.prep(A[k][0])
        sq += (X * X).sum(0)
        n += X.shape[0]
    return w, np.sqrt(sq / max(n - 1, 1))


def words_of(dims: list[int], k: int) -> dict:
    """The tokens that each dimension pushes hardest, both signs."""
    from safetensors import safe_open
    from transformers import AutoTokenizer

    # The weights may sit in 1 file or in shards. Find the file that
    # holds the token embedding, whichever layout the folder uses.
    root = Path(MODEL_DIR)
    ix = root / "model.safetensors.index.json"
    if ix.exists():
        wmap = json.load(open(ix))["weight_map"]
        name = next((c for c in wmap if c.endswith("embed_tokens.weight")),
                    None)
        path = root / wmap[name] if name else None
    else:
        path = root / "model.safetensors"
        with safe_open(str(path), framework="pt") as f:
            name = next((k for k in f.keys()
                         if k.endswith("embed_tokens.weight")), None)
    if name is None or path is None:
        raise SystemExit("no token embedding in the model folder")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    out = {}
    with safe_open(str(path), framework="pt") as f:
        E = f.get_slice(name)
        for d in dims:
            v = E[:, d].float().cpu().numpy().ravel()
            hi = np.argsort(-v)[:k]
            lo = np.argsort(v)[:k]
            out[int(d)] = {
                "high": [tok.decode([int(i)]).strip() for i in hi],
                "low": [tok.decode([int(i)]).strip() for i in lo],
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", default="outputs/_monocle/acts60")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1e5)
    ap.add_argument("--alpha-from", nargs="*", default=[])
    ap.add_argument("--top", type=int, default=32)
    ap.add_argument("--words", type=int, default=0,
                    help="read this many tokens for the shared dimensions")
    ap.add_argument("--out", default="outputs/_monocle/k1_reader_dims.json")
    args = ap.parse_args()

    alpha = {c: args.alpha for c in args.cases}
    for f in args.alpha_from:
        for r in json.load(open(f)):
            if (r["feature"] == "gh" and r["layer"] == args.layer
                    and r["split"] == "leave 1 out"):
                alpha[r["case"]] = float(r["alpha"])

    top, W = {}, {}
    for c in args.cases:
        try:
            w, sd = weight_of(c, Path(args.own), args.layer, alpha[c])
        except Exception as e:
            print(f"skip {c}: {e}", flush=True)
            continue
        W[c] = w
        score = np.abs(w * sd)
        top[c] = [int(i) for i in np.argsort(-score)[:args.top]]
        print(f"{c:>18} top {args.top}: {top[c][:8]} ...", flush=True)

    cs = sorted(top)
    print(f"\nthe overlap of the top {args.top} dimensions")
    pairs = []
    for i, a in enumerate(cs):
        for b in cs[i + 1:]:
            sa, sb = set(top[a]), set(top[b])
            j = len(sa & sb) / len(sa | sb)
            pairs.append({"a": a, "b": b, "shared": len(sa & sb),
                          "jaccard": j})
            print(f"  {a:>18} {b:>18} shared {len(sa & sb):>3} "
                  f"of {args.top}  jaccard {j:.3f}")

    # a random pair of sets of the same size gives this by chance
    exp = args.top * args.top / 3840.0
    print(f"  chance gives about {exp:.2f} shared")

    counts = {}
    for c in cs:
        for d in top[c]:
            counts[d] = counts.get(d, 0) + 1
    shared = sorted([d for d, n in counts.items() if n >= max(2, len(cs) // 2)],
                    key=lambda d: -counts[d])
    print(f"\n{len(shared)} dimensions appear in at least "
          f"{max(2, len(cs) // 2)} cases: {shared[:20]}")

    out = {"layer": args.layer, "top": {c: top[c] for c in cs},
           "pairs": pairs, "chance_shared": exp,
           "shared_dims": shared,
           "counts": {str(d): n for d, n in sorted(counts.items(),
                                                   key=lambda x: -x[1])}}
    if args.words and shared:
        out["words"] = words_of(shared[:12], args.words)
        for d, v in out["words"].items():
            print(f"  dim {d:>5} high {', '.join(v['high'][:8])}")
            print(f"  dim {d:>5} low  {', '.join(v['low'][:8])}")
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
