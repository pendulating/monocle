#!/usr/bin/env python
"""F8.5: build the 4 arms of the decoupled cue (D136).

In F8.2 the rich name is also the whiter NTA in 9 of 10 pairs. Thus F8.2
cannot separate class from race. F8.5 uses names where the 2 go apart. In
New York City only Brooklyn and Queens have such NTAs: Brooklyn has white
NTAs with a low income, and Queens has Black NTAs with a high income. The
Bronx, Manhattan and Staten Island have neither.

Each image gets 4 prompts. Only the first sentence changes:

| Arm | Brooklyn | Queens | Role |
|-----|----------|--------|------|
| `w_poor` | Borough Park | Maspeth | the whiter name, with the lower income |
| `n_match` | East New York | Hollis | a non-white name with the same income and education as `w_poor` |
| `n_rich` | Canarsie | Laurelton | a non-white name with a higher income than `w_poor` |
| `city` | "I am in New York City." | the same | the neutral cue: the same form, no neighbourhood |

The ACS values of the names are in the manifest (`dc_manifest.json`).

An image in Brooklyn gets the Brooklyn names, and an image in Queens the
Queens names. An image in another borough gets 1 of the 2 sets, drawn with
a fixed seed (`own_borough` is False). When the image is in an NTA of its
set, it gets the other set (`own_nta_hit` is True).

Outputs (data/f8/): `dc_{case}_{arm}.parquet`, 1 row for each image, with the
columns of the F8.2 files; `dc_manifest.json`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f8/build_decoupled_sample.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cue_sample import CUE, OUT, PROMPT, canonical_images  # noqa: E402

SEED = 20261029
CITY_CUE = "I am in New York City. "
ARMS = ("w_poor", "n_match", "n_rich", "city")

# borough -> arm -> (name in the prompt, the 2020 NTAs that it names)
SETS = {
    "Brooklyn": {
        "w_poor": ("Borough Park", ["Borough Park"]),
        "n_match": ("East New York", ["East New York (North)", "East New York-New Lots", "East New York-City Line"]),
        "n_rich": ("Canarsie", ["Canarsie"]),
    },
    "Queens": {
        "w_poor": ("Maspeth", ["Maspeth"]),
        "n_match": ("Hollis", ["Hollis"]),
        "n_rich": ("Laurelton", ["Laurelton"]),
    },
}
ACS_COLS = ["pct_white_nh", "pct_black_nh", "pct_asian_nh", "pct_bachelors", "median_household_income"]


def name_acs() -> dict:
    """Give the ACS values of each name: the NTA values, weighted by population (income: by households)."""
    a = pd.read_parquet("data/f4_traces/nta_acs.parquet")
    out = {}
    for boro, arms in SETS.items():
        for arm, (name, ntas) in arms.items():
            x = a[a["ntaname"].isin(ntas)]
            assert len(x) == len(ntas), (name, sorted(set(ntas) - set(x["ntaname"])))
            w = x["population"]
            v = {c: float((x[c] * w).sum() / w.sum()) for c in ACS_COLS[:-1]}
            v["median_household_income"] = float(
                (x["median_household_income"] * x["households"]).sum() / x["households"].sum())
            out[f"{boro}|{arm}"] = {"name": name, "ntas": ntas, **{k: round(val, 1) for k, val in v.items()}}
    return out


def assign_sets(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        own = r.boro in SETS
        boro = r.boro if own else ("Brooklyn", "Queens")[int(rng.integers(2))]
        hit = r.nta_name in {n for _, ntas in SETS[boro].values() for n in ntas}
        if hit:
            boro = "Queens" if boro == "Brooklyn" else "Brooklyn"
        rows.append({"image_idx": r.image_idx, "cue_boro": boro, "own_borough": own and not hit, "own_nta_hit": hit,
                     **{f"{arm}_name": SETS[boro][arm][0] for arm in ARMS[:3]}})
    return df.merge(pd.DataFrame(rows), on="image_idx", validate="1:1")


def main() -> None:
    rng = np.random.default_rng(SEED)
    manifest = {"seed": SEED, "cue": CUE, "city_cue": CITY_CUE, "prompts": PROMPT, "names": name_acs(), "files": {}}
    for case, df in canonical_images().items():
        df = assign_sets(df, rng)
        for arm in ARMS:
            a = df.copy()
            a["arm"] = arm
            if arm == "city":
                a["cue_name"] = "New York City"
                a["prompt"] = CITY_CUE + PROMPT[case]
            else:
                a["cue_name"] = a[f"{arm}_name"]
                a["prompt"] = [CUE.format(name=n, borough=b) + PROMPT[case]
                               for n, b in zip(a["cue_name"], a["cue_boro"])]
            keep = ["image_idx", "sample_id", "image_path", "boro", "nta_name", "cue_boro", "own_borough",
                    "own_nta_hit", "arm", "cue_name", "w_poor_name", "n_match_name", "n_rich_name", "prompt"]
            path = OUT / f"dc_{case}_{arm}.parquet"
            a[keep].to_parquet(path, index=False)
            manifest["files"][path.name] = {
                "rows": len(a), "own_borough": int(a["own_borough"].sum()), "own_nta_hit": int(a["own_nta_hit"].sum()),
                "cue_boro": a["cue_boro"].value_counts().to_dict()}
            print(path, len(a), "example:", a["prompt"].iloc[0])
    (OUT / "dc_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["names"], indent=1))


if __name__ == "__main__":
    main()
