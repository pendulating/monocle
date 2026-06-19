---
title: "Guide: Neighborhood (NTA) Aggregation of Pairwise Rankings"
category: guide
created: 2026-06-09
updated: 2026-06-09
tags:
  - guide
  - pairwise
  - ranking
  - trueskill
  - geospatial
  - marimo
---

# Guide: Neighborhood (NTA) Aggregation of Pairwise Rankings

How to turn the 100k-pair [[urban-pair-vqa]] outputs into a **per-neighborhood ranking** over NYC's 2020 Neighborhood Tabulation Areas (NTAs), using the marimo notebook `notebooks/css/neighborhoods.py`.

Built for the non-thinking restaurant ("which would you rather eat at") and school ("which would you rather send your child to") sweeps, but works on any `unit`-mode pairwise run whose `pairs.parquet` carries per-side `unit_uid` + lat/lon.

## Inputs

| Input | Path | Notes |
|---|---|---|
| Pairwise output parquet | `<run>/0/outputs/pairwise/<dataset>_mvp_<ts>.parquet` | Carries `relative_score` (de-swapped, see [[concept-counterbalancing]]). **No geo columns.** |
| Sibling `pairs.parquet` | same directory | Geo + identity: `unit_uid_a/b`, `unit_name_a/b`, `latitude_a/b`, `longitude_a/b`. Joined on `pair_id`. |
| NTA shapefile | `data/geo/nynta2020_26b/nynta2020.shp` | **EPSG:2263** (NY State Plane, feet), 262 NTAs. Key fields: `NTA2020` (code), `NTAName`, `BoroName`. |

The output parquet intentionally drops the metadata columns — **all geography lives in `pairs.parquet`** and must be re-joined on `pair_id`. (If the consolidated output is missing entirely, see [[troubleshooting#Issue 5: Interrupted urbanpairvqa Run — Recompile From Streaming Chunks]].)

## Pipeline

1. **Join** output ⨝ `pairs.parquet` on `pair_id` (inner) → one row per comparison with `relative_score` + both units' ids/coords.
2. **Locate each unit** — one point per `unit_uid` = mean of its recording lat/lon across every row where it appears as side A or B. (Recording location ≈ unit location to within tens of feet — robust for polygon assignment.)
3. **Assign to NTA** — build a `GeoDataFrame` of unit points (EPSG:4326), reproject to **2263**, `gpd.sjoin(..., predicate="within")` against the NTA polygons → `unit_uid → NTA2020/NTAName/BoroName`.
4. **Unit-level TrueSkill** — the canonical recipe ([[concept-trueskill]]): `relative_score > 0` → A wins, `< 0` → B wins, `0` → draw; magnitude discarded. Yields per-unit `mu/sigma/n_comparisons`.
5. **Two NTA-level rankings** (shown side by side):

| Method | How | Character |
|---|---|---|
| **(A) Direct zone TrueSkill** | Relabel each comparison by each side's NTA, drop within-NTA pairs, fit TrueSkill on NTA-vs-NTA matches | Principled aggregation; mirrors the PUMA/tract step in `wealth.ipynb`. Headline. |
| **(B) Mean of unit μ** | Average unit `mu` within each NTA | Simple, interpretable cross-check. |

Both are surfaced equally; their agreement is the validation (see below).

## Controls

The notebook is reactive (marimo): a **dataset** dropdown (`restaurants` / `schools`) and a **min-comparisons** slider (reliability filter). Changing either recomputes the tables, choropleths, agreement scatter, and borough breakdown for the selected dataset.

## Outputs

Persisted to `notebooks/css/results/`:

- `nta_<dataset>_zone_trueskill.parquet` — method (A), one row per NTA
- `nta_<dataset>_mean_of_units.parquet` — method (B), one row per NTA
- `unit_<dataset>_trueskill.parquet` — per-unit ratings + NTA assignment

## Run it

```bash
source /share/pierson/matt/mllmsci/.venv/bin/activate
marimo edit notebooks/css/neighborhoods.py
# headless validation / static export:
MPLBACKEND=Agg marimo export html notebooks/css/neighborhoods.py -o /tmp/neighborhoods.html
```

## Results snapshot (2026-06-09, qwen3.5-9b/instruct, non-thinking, 100k pairs)

| | restaurants | schools |
|---|---|---|
| comparisons | 110,000 | 110,000 |
| units → NTA | 18,487 / 18,488 (100%) | 2,287 / 2,287 (100%) |
| NTAs covered | 219 | 207 |
| method (A) vs (B) agreement | **Spearman ρ = 0.905** | — |

Face-valid: top restaurant NTAs are SoHo-Little Italy, Park Slope, West Village, Williamsburg; bottom are outer-borough / industrial (Oakwood-Richmondtown, Co-op City, Flushing Meadows-Corona Park). The 0.905 rank correlation between the two independent aggregation methods cross-validates the result.

## Gotchas

- **Geo is not in the output parquet** — always join `pairs.parquet`. Forgetting this silently yields a no-geo table.
- **Reproject before `sjoin`** — unit points are WGS84 (4326); the NTA shapefile is 2263. Mixing CRS produces an empty/garbage join.
- **Min-comparisons filter** — NTAs below threshold (and unscored ones) are grayed on the choropleth and excluded from the ranking tables; sparse zones have huge sigma (same logic as the `--min-comparisons` flag on `pairwise_vqa_report.py`).
- **Scales differ between methods** — compare (A) and (B) by **rank correlation**, not absolute μ (the two TrueSkill fits live on different scales).

## Relationship to `pairwise_vqa_report.py --zone-geojson`

The [[concept-trueskill#Zone-geometry aggregation]] utility rates polygons directly for **image-mode** runs (no `unit_uid`). This notebook is the **unit-mode** counterpart: it keeps unit identity, ranks units first, then offers *both* a unit-first (mean) and a zone-first (direct NTA TrueSkill) aggregation in one interactive view.

## See Also

- [[concept-trueskill]] — the rating recipe and its gotchas
- [[concept-counterbalancing]] — why `relative_score` is already de-swapped
- [[urban-pair-vqa]] — the dagspace that produces the match table
- [[troubleshooting#Issue 5: Interrupted urbanpairvqa Run — Recompile From Streaming Chunks]] — recovering a missing consolidated output
