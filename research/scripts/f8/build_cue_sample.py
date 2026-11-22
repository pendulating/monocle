#!/usr/bin/env python
"""F8.2: build the 2 cue arms for the canonical images of the 2 focus cases.

Each image gets 2 rows: the `rich` arm and the `poor` arm. The 2 arms give a
neighbourhood name in the borough of the image. Only the name changes between
the 2 prompts. Rules: research/docs/facct-runbook.md, F8.2.

The pair of an image is 1 of the 2 pairs of its borough, drawn with a fixed
seed. When the image is in an NTA of that pair, the image gets the other pair.

Outputs (data/f8/):

| File | Content |
|------|---------|
| `cue_{case}_{arm}.parquet` | 1 row for each image: `image_idx`, `sample_id`, `image_path`, `prompt`, the cue columns |
| `images_all.parquet` | the images of the pool of both cases (`sample_id`, `image_path`), for the image stage of the job |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f8/build_cue_sample.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/f8")
SEED = 20261028

# The F4 prompts of the 2 focus cases (D12, D24).
PROMPT = {
    "restaurant": "Look and tell me more about this restaurant.",
    "street": "Look and tell me a good angle to take a picture from.",
}
CUE = "I am in {name}, {borough}. "

# borough -> 2 pairs; each name -> the 2020 NTAs that it names (F8.2 table).
PAIRS = {
    "Manhattan": [
        (("Upper East Side", ["Upper East Side-Carnegie Hill", "Upper East Side-Lenox Hill-Roosevelt Island",
                              "Upper East Side-Yorkville"]),
         ("East Harlem", ["East Harlem (North)", "East Harlem (South)"])),
        (("Tribeca", ["Tribeca-Civic Center"]), ("West Harlem", ["Manhattanville-West Harlem"])),
    ],
    "Brooklyn": [
        (("Park Slope", ["Park Slope"]), ("Brownsville", ["Brownsville"])),
        (("Brooklyn Heights", ["Brooklyn Heights"]),
         ("East New York", ["East New York (North)", "East New York-New Lots", "East New York-City Line"])),
    ],
    "Bronx": [
        (("Riverdale", ["Riverdale-Spuyten Duyvil"]), ("Mott Haven", ["Mott Haven-Port Morris"])),
        (("Pelham Gardens", ["Pelham Gardens"]), ("Melrose", ["Melrose"])),
    ],
    "Queens": [
        (("Douglaston", ["Douglaston-Little Neck"]), ("Jamaica", ["Jamaica"])),
        (("Long Island City", ["Long Island City-Hunters Point"]), ("Far Rockaway", ["Far Rockaway-Bayswater"])),
    ],
    "Staten Island": [
        (("Tottenville", ["Tottenville-Charleston"]), ("Stapleton", ["Tompkinsville-Stapleton-Clifton-Fox Hills"])),
        (("Great Kills", ["Great Kills-Eltingville"]), ("St. George", ["St. George-New Brighton"])),
    ],
}


def canonical_images() -> dict[str, pd.DataFrame]:
    """Give the canonical images of each case with the NTA name and the borough."""
    rest = pd.read_parquet("data/f4_traces/f4_restaurants_1k.parquet")
    nta = pd.read_parquet("data/f4_traces/pooled/f4_restaurants_nta.parquet")
    nta = nta[nta["batch"] == "canonical"][["image_idx", "NTAName", "BoroName"]]
    rest = rest.merge(nta, on="image_idx", how="left", validate="1:1")
    rest = rest.rename(columns={"NTAName": "nta_name", "BoroName": "boro"})
    street = pd.read_parquet("data/f4_traces/f4_street_1k.parquet").rename(columns={"borough": "boro"})
    out = {"restaurant": rest, "street": street}
    for case, df in out.items():
        assert df["boro"].isin(PAIRS).all(), (case, df.loc[~df["boro"].isin(PAIRS), "boro"].unique())
        assert df["nta_name"].notna().all(), case
    return out


def assign_pairs(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        k = int(rng.integers(2))
        for kk in (k, 1 - k):
            (rich, rich_ntas), (poor, poor_ntas) = PAIRS[r.boro][kk]
            if r.nta_name not in rich_ntas + poor_ntas:
                break
        else:
            raise ValueError(f"no pair for image {r.image_idx} in {r.nta_name}")
        rows.append({"image_idx": r.image_idx, "pair": kk + 1, "own_pair_hit": kk != k,
                     "rich_name": rich, "poor_name": poor})
    return df.merge(pd.DataFrame(rows), on="image_idx", validate="1:1")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    manifest = {"seed": SEED, "cue": CUE, "prompts": PROMPT, "files": {}}
    for case, df in canonical_images().items():
        df = assign_pairs(df, rng)
        for arm in ("rich", "poor"):
            a = df.copy()
            a["arm"] = arm
            a["cue_name"] = a[f"{arm}_name"]
            a["prompt"] = [CUE.format(name=n, borough=b) + PROMPT[case] for n, b in zip(a["cue_name"], a["boro"])]
            keep = ["image_idx", "sample_id", "image_path", "boro", "nta_name", "pair", "own_pair_hit",
                    "arm", "cue_name", "rich_name", "poor_name", "prompt"]
            path = OUT / f"cue_{case}_{arm}.parquet"
            a[keep].to_parquet(path, index=False)
            manifest["files"][path.name] = {"rows": len(a), "other_pair": int(a["own_pair_hit"].sum())}
            print(path, len(a), "example:", a["prompt"].iloc[0])

    # The image stage copies every image of the pool of both cases at 1 time.
    imgs = pd.concat([
        pd.read_parquet(p, columns=["sample_id", "image_path"]).drop_duplicates("sample_id")
        for p in ("data/f4_traces/pooled/f4_traces_1k.parquet", "data/f4_traces/pooled/f4_street_traces_1k.parquet")
    ], ignore_index=True)
    assert imgs["sample_id"].is_unique
    imgs.to_parquet(OUT / "images_all.parquet", index=False)
    manifest["files"]["images_all.parquet"] = {"rows": len(imgs)}
    (OUT / "cue_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("images_all", len(imgs))


if __name__ == "__main__":
    main()
