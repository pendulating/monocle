#!/usr/bin/env python
"""Analyze the persona probe: does removing the system persona move abstention?

The 2026-08-11 consolidation dropped the system turn entirely (`system: null`)
from all seven ranking cases. Reviewer-2 arm A2 already showed that removing a
much stronger *domain* persona leaves the judgments at the decoding-noise
bound — but it ran when `allow_not_sure` still defaulted to `false`, so it
could not measure the one channel where a persona plausibly bites: "expert" is
a competence prime, so dropping it may raise the NotSure rate.

Primary readout is therefore the **abstention rate** per arm, paired within
model. Ordinal agreement on the pairs *both* arms answered is the secondary
read (i.e. are the surviving judgments the same, or did the persona re-rank?).

Runs are attributed from each stage dir's own ``.hydra/overrides.yaml`` — never
from directory order, which does not survive a re-run (see the HYDRA_SWEEP_DIR
stage-dir note in the urban-pair-vqa wiki page).

Usage::

    python scripts/pairwise_persona_probe.py multirun/2026-08-11_URBANPAIRVQA/16-40-19
    python scripts/pairwise_persona_probe.py <stage_dir> --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

ORDINAL = ["MuchLess", "Less", "Same", "More", "MuchMore"]
SCORE = {"MuchLess": -2, "Less": -1, "Same": 0, "More": 1, "MuchMore": 2}
NOT_SURE = "NotSure"


# ---------------------------------------------------------------- discovery


def _arm_from_overrides(path: Path) -> Optional[Dict[str, str]]:
    """Read (model, persona arm) out of a stage dir's Hydra overrides."""
    try:
        items = yaml.safe_load(path.read_text()) or []
    except (OSError, yaml.YAMLError):
        return None
    ov = {}
    for raw in items:
        if not isinstance(raw, str) or "=" not in raw:
            continue
        k, _, v = raw.partition("=")
        ov[k.strip().lstrip("+~")] = v.strip()
    if "prompt.system" not in ov:
        return None
    sysval = ov["prompt.system"]
    # `null` (or an empty/quoted-empty value) == the production no-persona arm.
    persona = "no-persona" if sysval.lower() in {"null", "none", "", "''", '""'} else "persona"
    return {
        "persona": persona,
        "system_text": "" if persona == "no-persona" else sysval.strip("'\""),
        "model": ov.get("model", "?"),
        "pipeline": ov.get("pipeline", "?"),
    }


def discover(stage_root: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for job_dir in sorted(p for p in stage_root.iterdir() if p.is_dir() and p.name.isdigit()):
        arm = _arm_from_overrides(job_dir / ".hydra" / "overrides.yaml")
        if arm is None:
            continue
        # The stage writes outputs/pairwise/<case>_<ts>.parquet; pairs.parquet is
        # the *input* pair manifest, not results.
        cands = [
            p for p in (job_dir / "outputs" / "pairwise").glob("*.parquet")
            if p.name != "pairs.parquet"
        ]
        if not cands:
            print(f"  [skip] {job_dir.name}: no result parquet yet ({arm['persona']}, {arm['model']})",
                  file=sys.stderr)
            continue
        arm["path"] = max(cands, key=lambda p: p.stat().st_mtime)
        arm["job"] = job_dir.name
        runs.append(arm)
    return runs


# ------------------------------------------------------------------ metrics


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves at the small counts an abstention rate hits."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z test; returns (z, two-sided p)."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return (z, p)


def _linear_weighted_kappa(a: pd.Series, b: pd.Series) -> float:
    """Linear-weighted Cohen's kappa over the 5-point ordinal scale."""
    cats = ORDINAL
    n = len(a)
    if n == 0:
        return float("nan")
    idx = {c: i for i, c in enumerate(cats)}
    ai = a.map(idx).to_numpy()
    bi = b.map(idx).to_numpy()
    k = len(cats)
    obs = num = den = 0.0
    ca = pd.Series(ai).value_counts().reindex(range(k), fill_value=0).to_numpy() / n
    cb = pd.Series(bi).value_counts().reindex(range(k), fill_value=0).to_numpy() / n
    for i in range(k):
        for j in range(k):
            w = abs(i - j) / (k - 1)
            o = ((ai == i) & (bi == j)).sum() / n
            e = ca[i] * cb[j]
            num += w * o
            den += w * e
    return float("nan") if den == 0 else 1 - num / den


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    lab = df["relative_label"].astype(str)
    n = len(lab)
    k_abs = int((lab == NOT_SURE).sum())
    answered = lab[lab != NOT_SURE]
    lo, hi = _wilson(k_abs, n)
    dist = lab.value_counts(normalize=True).reindex(ORDINAL + [NOT_SURE], fill_value=0.0)
    scores = answered.map(SCORE).dropna()
    return {
        "n": n,
        "abstentions": k_abs,
        "abstention_rate": k_abs / n if n else float("nan"),
        "abstain_ci95": [lo, hi],
        "same_rate_of_answered": float((answered == "Same").mean()) if len(answered) else float("nan"),
        "mean_abs_score": float(scores.abs().mean()) if len(scores) else float("nan"),
        "dist": {k: float(v) for k, v in dist.items()},
    }


# --------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage_dir", type=Path, help="Sweep stage dir containing numbered job subdirs")
    ap.add_argument("--retest", type=Path, default=None,
                    help="Stage dir of a persona_retest run (the no-persona arm re-run). "
                         "Supplies the temperature-0.6 test-retest ceiling that every "
                         "persona-vs-no-persona kappa must be judged against.")
    ap.add_argument("--json", type=Path, default=None, help="Also write results as JSON")
    args = ap.parse_args()

    runs = discover(args.stage_dir)
    if not runs:
        print(f"No completed runs with result parquets under {args.stage_dir}", file=sys.stderr)
        return 1

    for r in runs:
        df = pd.read_parquet(r["path"], columns=["pair_id", "relative_label"])
        r["df"] = df
        r["summary"] = summarize(df)

    # Retest anchor: the same no-persona arm run twice. Its self-agreement is the
    # ceiling — a persona-vs-no-persona kappa can only be called "low" relative
    # to it, never relative to 1.0.
    retest: Dict[str, Dict[str, Any]] = {}
    if args.retest:
        for r in discover(args.retest):
            if r["persona"] != "no-persona":
                continue
            r["df"] = pd.read_parquet(r["path"], columns=["pair_id", "relative_label"])
            r["summary"] = summarize(r["df"])
            retest[r["model"]] = r
        if not retest:
            print(f"  [warn] no no-persona runs found under {args.retest}", file=sys.stderr)

    print(f"\n{'='*88}\nPERSONA PROBE — {runs[0]['pipeline']}\n{'='*88}")
    print(f"{'model':26s} {'arm':12s} {'n':>6s} {'abstain%':>9s} {'95% CI':>16s} "
          f"{'Same%':>7s} {'|score|':>8s}")
    print("-" * 88)
    for r in sorted(runs, key=lambda x: (x["model"], x["persona"])):
        s = r["summary"]
        ci = f"[{s['abstain_ci95'][0]*100:.1f},{s['abstain_ci95'][1]*100:.1f}]"
        print(f"{r['model']:26s} {r['persona']:12s} {s['n']:6d} {s['abstention_rate']*100:8.2f}% "
              f"{ci:>16s} {s['same_rate_of_answered']*100:6.1f}% {s['mean_abs_score']:8.3f}")

    print(f"\n{'='*88}\nPAIRED CONTRAST (within model: no-persona vs persona)\n{'='*88}")
    out: Dict[str, Any] = {"pipeline": runs[0]["pipeline"], "arms": [], "contrasts": []}
    for r in runs:
        out["arms"].append({k: r[k] for k in ("model", "persona", "job")} | {"summary": r["summary"]})

    for model in sorted({r["model"] for r in runs}):
        arms = {r["persona"]: r for r in runs if r["model"] == model}
        if len(arms) != 2:
            print(f"\n{model}: only {list(arms)} present — skipping contrast")
            continue
        npr, pr = arms["no-persona"], arms["persona"]
        sn, sp = npr["summary"], pr["summary"]
        z, pval = _two_prop_z(sn["abstentions"], sn["n"], sp["abstentions"], sp["n"])
        delta = (sn["abstention_rate"] - sp["abstention_rate"]) * 100

        merged = npr["df"].merge(pr["df"], on="pair_id", suffixes=("_np", "_p"))
        both = merged[(merged.relative_label_np != NOT_SURE) & (merged.relative_label_p != NOT_SURE)]
        kappa = _linear_weighted_kappa(both.relative_label_np, both.relative_label_p)
        sc_np = both.relative_label_np.map(SCORE)
        sc_p = both.relative_label_p.map(SCORE)
        flip = float(((sc_np > 0) != (sc_p > 0)).mean()) if len(both) else float("nan")

        print(f"\n{model}")
        print(f"  abstention   no-persona {sn['abstention_rate']*100:6.2f}%  vs  "
              f"persona {sp['abstention_rate']*100:6.2f}%   Δ={delta:+.2f} pp   "
              f"z={z:+.2f}  p={pval:.4f}  {'← SIGNIFICANT' if pval < 0.05 else '(n.s.)'}")
        if sn["abstentions"] + sp["abstentions"] < 20:
            print(f"               ⚠ only {sn['abstentions'] + sp['abstentions']} abstention events "
                  f"total — a significant p here is a tiny-count artifact, not a usable effect")
        print(f"  agreement    linear-weighted κ = {kappa:.3f} on {len(both)} pairs both arms answered")
        print(f"  sign flips   {flip*100:.1f}% of those pairs changed which side won")

        rec: Dict[str, Any] = {
            "model": model, "abstain_delta_pp": delta, "z": z, "p": pval,
            "kappa_both_answered": kappa, "n_both_answered": int(len(both)), "flip_rate": flip,
        }

        anchor = retest.get(model)
        if anchor is not None:
            m2 = npr["df"].merge(anchor["df"], on="pair_id", suffixes=("_a", "_b"))
            both2 = m2[(m2.relative_label_a != NOT_SURE) & (m2.relative_label_b != NOT_SURE)]
            k_re = _linear_weighted_kappa(both2.relative_label_a, both2.relative_label_b)
            s_a = both2.relative_label_a.map(SCORE)
            s_b = both2.relative_label_b.map(SCORE)
            flip_re = float(((s_a > 0) != (s_b > 0)).mean()) if len(both2) else float("nan")
            rec |= {"kappa_retest": k_re, "flip_rate_retest": flip_re,
                    "n_retest": int(len(both2))}
            print(f"  ── anchor    test-retest κ = {k_re:.3f} on {len(both2)} pairs "
                  f"(same prompt twice, temp 0.6) · {flip_re*100:.1f}% sign flips")

            # The abstention rate needs the same treatment as kappa: the retest
            # arm is the SAME prompt, so any gap between it and the probe's
            # no-persona arm is pure sampling noise. A persona delta only counts
            # if it clears that band.
            sr = anchor["summary"]
            noise = abs(sn["abstention_rate"] - sr["abstention_rate"]) * 100
            zr, pr_ = _two_prop_z(sn["abstentions"], sn["n"], sr["abstentions"], sr["n"])
            print(f"               abstention noise band: no-persona {sn['abstention_rate']*100:.2f}% "
                  f"vs its own retest {sr['abstention_rate']*100:.2f}% "
                  f"→ |{noise:.2f}| pp of pure noise (p={pr_:.4f})")
            if abs(delta) <= noise:
                print(f"               ⚠ the persona Δ ({delta:+.2f} pp) does NOT clear the "
                      f"noise band — do not report it as a persona effect")
            rec |= {"abstain_retest_rate": sr["abstention_rate"],
                    "abstain_noise_pp": noise, "abstain_clears_noise": bool(abs(delta) > noise)}
            if not (math.isnan(k_re) or math.isnan(kappa)):
                gap = k_re - kappa
                if k_re <= 0:
                    verdict = "anchor itself is at chance — this model is noise on this case"
                elif gap <= 0.05:
                    verdict = ("persona κ is AT the decoding-noise ceiling → the persona "
                               "changed nothing beyond temperature")
                elif gap <= 0.15:
                    verdict = "persona κ is slightly below the ceiling → weak, probably not meaningful"
                else:
                    verdict = ("persona κ is WELL below the ceiling → the persona really did "
                               "move the judgments")
                print(f"               gap to ceiling = {gap:+.3f} → {verdict}")
                rec["kappa_gap_to_ceiling"] = gap
                rec["verdict"] = verdict
        out["contrasts"].append(rec)

    print(f"\n{'='*88}")
    print("Reading it: a significant positive Δ means removing the persona made the model")
    print("abstain MORE. High κ with a null Δ means the persona was cosmetic — the judgments")
    print("and the confidence to state them both survived its removal.")
    print(f"{'='*88}\n")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
