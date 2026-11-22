#!/usr/bin/env python
"""F9: the new schema against the x2 spans of record, on the same F4 smoke texts (a description, not a gate).

`f9-extraction-design.md`, section 6.2. D142 extracts the F4 answers again
with the F9 schema, thus the F9 test reads 1 schema on both sides. This
script tells how far the new-schema F4 spans are from the x2 spans of
record, on the F4 x2 smoke sets (`x2smoke_restaurant`, `x2smoke_street`).

| Measure | Rule |
|---------|------|
| `ve_ratio` | `visual_evidence` spans, new schema / x2 |
| `overlap` | the share of x2 `visual_evidence` spans that 1 new-schema `visual_evidence` span covers by 50% or more of its characters |
| `valence_agree` | the share of the overlapped spans with the same valence |
| valence mix | the share of `good`, `bad`, `neutral` over all `visual_evidence` spans, for each schema |
| `verdict_texts` | the share of texts with 1 `verdict` span or more (new schema only) |
| class mix | the spans per text of each class, for each schema |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f9/compare_f4_schema.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

__version__ = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "research" / "notebooks" / "facct"))

# case -> (new-schema arm, x2 arm, smoke input case)
ARMS = {"restaurant": ("x2_cues_subj", "x2_cues", "x2smoke_restaurant"),
        "street": ("x2_cues_street_subj", "x2_cues_street", "x2smoke_street")}
CLASSES = ("visual_evidence", "inference", "uncertainty", "image_artifact", "person_reference",
           "photo_advice", "verdict")


def _read(root: Path, arm: str, case: str) -> pd.DataFrame:
    files = glob.glob(str(root / arm / "runs" / f"f4_{case}__*" / "outputs" / "extractions" / "*.parquet"))
    if not files:
        raise FileNotFoundError(f"no runs of {case} under {root / arm}")
    import _f4

    return _f4.spans(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))


def overlap(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Give each x2 `visual_evidence` span its best new-schema cover (share of its characters) and that span's valence."""
    a = old[old.extraction_class == "visual_evidence"][["doc_id", "char_start", "char_end", "valence"]]
    b = new[new.extraction_class == "visual_evidence"][["doc_id", "char_start", "char_end", "valence"]]
    m = a.reset_index().merge(b, on="doc_id", suffixes=("", "_new"))
    inter = (m[["char_end", "char_end_new"]].min(axis=1) - m[["char_start", "char_start_new"]].max(axis=1)).clip(lower=0)
    m["cover"] = inter / (m.char_end - m.char_start).clip(lower=1)
    best = m.sort_values("cover").drop_duplicates("index", keep="last").set_index("index")
    out = a.copy()
    out["cover"] = best.cover.reindex(out.index).fillna(0.0)
    out["valence_new"] = best.valence_new.reindex(out.index)
    return out


def compare(new_root: Path, old_root: Path) -> dict:
    res = {}
    for case, (arm_new, arm_old, smoke) in ARMS.items():
        new, old = _read(new_root, arm_new, smoke), _read(old_root, arm_old, smoke)
        texts = sorted(set(new.doc_id) | set(old.doc_id))
        ov = overlap(new, old)
        hit = ov[ov.cover >= 0.5]
        mix = {}
        for name, d in (("new", new), ("x2", old)):
            ve = d[d.extraction_class == "visual_evidence"]
            mix[name] = {
                "valence": ve.valence.value_counts(normalize=True).reindex(["good", "bad", "neutral"]).fillna(0).round(3).to_dict(),
                "per_text": (d[d.extraction_class.isin(CLASSES)].extraction_class.value_counts() / len(texts)).round(2).to_dict(),
            }
        n_ve_new = int((new.extraction_class == "visual_evidence").sum())
        n_ve_old = int((old.extraction_class == "visual_evidence").sum())
        res[case] = {
            "texts": len(texts), "ve_new": n_ve_new, "ve_x2": n_ve_old,
            "ve_ratio": round(n_ve_new / max(n_ve_old, 1), 3),
            "overlap": round(float((ov.cover >= 0.5).mean()), 3),
            "valence_agree": round(float((hit.valence == hit.valence_new).mean()), 3),
            "verdict_texts": round(float(new[new.extraction_class == "verdict"].doc_id.nunique() / len(texts)), 3),
            "mix": mix,
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--new-root", type=Path, default=REPO_ROOT / "multirun" / "f4x2_subj_smoke_f4")
    ap.add_argument("--x2-root", type=Path, default=REPO_ROOT / "multirun" / "f4x2_smoke")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "research" / "figures" / "f9" / "f4_schema_compare.json")
    args = ap.parse_args()
    res = compare(args.new_root, args.x2_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"script": "research/scripts/f9/compare_f4_schema.py", "version": __version__,
                                    **res}, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
