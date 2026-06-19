---
title: "TrueSkill for Pairwise Comparison Aggregation"
category: concept
created: 2026-04-22
updated: 2026-06-09
tags:
  - concept
  - pairwise
  - ranking
  - trueskill
  - aggregation
---

# TrueSkill for Pairwise Comparison Aggregation

How [[urban-pair-vqa]] outputs (a long table of A-vs-B ordinal judgments) get turned into a **ranked list of entities** with calibrated uncertainty.

## Problem

Pairwise VQA produces rows of the form "entity A is `More` / `Less` / `Same` / ... than entity B on attribute X". A single entity can appear in dozens of pairs against different opponents, each with different implicit difficulty. We want a single scalar "how much of X does entity E have?" that:

- Handles **sparse pairings** (not every entity is compared to every other)
- Produces **uncertainty estimates** (some entities have 5 comparisons, others 50)
- Is **rank-invariant** to which opponents an entity was lucky/unlucky enough to draw
- Tolerates **draws** (the `Same` label) as a real outcome, not a missing vote

TrueSkill, Microsoft's Bayesian rating system for Xbox Live, ticks every box. It generalizes Elo to non-2-player games and draws, and it tracks `(mu, sigma)` per entity instead of a point rating — so you get variance for free.

## Mapping ordinal labels to TrueSkill outcomes

[[urban-pair-vqa]] emits a signed integer `relative_score` in `{-2, -1, 0, +1, +2}` corresponding to `{MuchLess, Less, Same, More, MuchMore}`. These are already de-swapped — see [[concept-counterbalancing]] — so the score reflects the **canonical** A-vs-B comparison regardless of presentation order.

The canonical treatment collapses the 5-point scale to a 3-outcome match:

| `relative_score` | outcome | `rate_1vs1` call |
|---|---|---|
| `+2` (MuchMore) | A wins | `env.rate_1vs1(ra, rb, drawn=False)` |
| `+1` (More) | A wins | `env.rate_1vs1(ra, rb, drawn=False)` |
| `0` (Same) | draw | `env.rate_1vs1(ra, rb, drawn=True)` |
| `-1` (Less) | B wins | `env.rate_1vs1(rb, ra, drawn=False)` |
| `-2` (MuchLess) | B wins | `env.rate_1vs1(rb, ra, drawn=False)` |

Magnitude (`MuchMore` vs `More`) is **discarded** in this canonical recipe. Stronger outcomes don't move ratings more, just as a 10-0 Elo match isn't worth more than a 2-1 one. Weighting magnitude is possible (replay `MuchMore` rows multiple times, or swap in [Weng–Lin](https://openskill.me/) with continuous margins) but nothing in the repo currently does this — stick to the default unless you have a concrete reason.

## The canonical recipe (from `notebooks/css/wealth.ipynb`)

```python
import trueskill
from collections import defaultdict

env = trueskill.TrueSkill(draw_probability=0.05)
ratings = defaultdict(env.create_rating)

for row in matches.itertuples(index=False):
    a, b = str(row.unit_uid_a), str(row.unit_uid_b)
    score = int(row.relative_score)
    ra, rb = ratings[a], ratings[b]
    if score > 0:
        ra, rb = env.rate_1vs1(ra, rb, drawn=False)   # A wins
    elif score < 0:
        rb, ra = env.rate_1vs1(rb, ra, drawn=False)   # B wins (winner first!)
    else:
        ra, rb = env.rate_1vs1(ra, rb, drawn=True)    # draw
    ratings[a], ratings[b] = ra, rb
```

Then derive ranking-ready columns:

```python
summary = pd.DataFrame([
    {
        "unit_uid": uid,
        "mu": r.mu,
        "sigma": r.sigma,
        "ts_point_estimate": r.mu,
        "ts_conservative": r.mu - 3.0 * r.sigma,
        "n_comparisons": counts[uid],
    }
    for uid, r in ratings.items()
]).sort_values("ts_conservative", ascending=False)
```

### Why `mu - 3 * sigma`?

`mu - 3 * sigma` is Microsoft's **conservative** display rating — a rating the entity is ~99.7% likely to exceed. It penalizes entities with few comparisons (high sigma) and is what you want for **ranking podiums** where you don't want to promote a lucky underdog. For **scatter plots against covariates** (income, population, ...) prefer the point estimate `mu`.

## Defaults and tuning knobs

| Param | Default | What it controls |
|---|---|---|
| `mu` | 25.0 | Starting rating |
| `sigma` | 25/3 ≈ 8.33 | Starting uncertainty |
| `beta` | 25/6 ≈ 4.17 | Skill-class width (controls how much ratings move per match) |
| `tau` | 25/300 ≈ 0.083 | Dynamics factor (skill drift over time; irrelevant for offline batch ratings) |
| `draw_probability` | 0.1 | Prior on draws |

Knobs that actually matter for this pipeline:

- **`draw_probability`**: set to match the observed `Same` rate. [[urban-pair-vqa]] runs typically see 5–10%. The wealth.ipynb canon uses `0.05`; the libraries MVP had ~9.5% — either works. Numerically the ranking is fairly stable for 0.05–0.10.
- **Everything else**: leave alone. TrueSkill is scale-free — mu/sigma/beta/tau only change absolute numbers, not relative ranks.

## Gotchas

### Row order matters

TrueSkill is an **online** algorithm: `rate_1vs1` updates ratings one match at a time, using only the current posterior. Feeding the same matches in a different order produces **slightly different** final `(mu, sigma)` pairs (top-level ranks are usually stable, but fringe cases shuffle). For reproducibility:

- Do not re-shuffle between runs. The natural row order from `pairs.parquet` is seeded by `pair_sampler.pair_seed` — keep that seed pinned.
- If you need to iterate more carefully, consider running multiple passes or switching to a batch solver (e.g., `choix`, `openskill`).

### `rate_1vs1` is **not** commutative

`env.rate_1vs1(ra, rb, drawn=False)` means "A beat B", not "A and B played". Always pass the **winner first**. The canonical recipe above swaps `ra`/`rb` when B wins.

### What should the "unit" be?

TrueSkill is agnostic — you decide what gets a rating. Options in this project:

| Unit | When to use | Example |
|---|---|---|
| **Image / recording** | Direct visual ranking, or aggregating later | Face-level rating of library exteriors |
| **Facility / permit / library** | Most common for UAIR — pairs sampled at the unit level | `pair_sampler.mode: unit` → one rating per library |
| **Geographic area (PUMA, tract, NTA)** | Aggregate image-level matches by containing geometry | See wealth.ipynb for PUMA and tract ratings; `pairwise_vqa_report.py --zone-geojson` (image-mode); or [[guide-neighborhood-aggregation]] for the unit-mode NTA notebook (unit-first + zone-first side by side) |

For any unit that is **not** already the pair-sampling unit, you have to re-join the match table against a lookup (e.g., image → tract) before iterating. For the geographic case the [[#Utility script]] now does this natively — see [Zone-geometry aggregation](#zone-geometry-aggregation).

### Sparse units

An entity with `n_comparisons < ~10` has a huge sigma; the point estimate is basically the prior. Filter these out of the top-N / bottom-N displays when interpreting results. The `--min-comparisons` flag on [[#Utility script]] enforces this.

## Utility script

`scripts/pairwise_vqa_report.py` wraps the full recipe — label stats, reasoning word cloud, TrueSkill ranking — into a markdown report with embedded plots. See the script's `--help` for options. Run against any `<stage_output>.parquet` that sits alongside a `pairs.parquet` containing `unit_uid_a`/`unit_uid_b`.

For multi-model sweeps (one Hydra job per model), use `scripts/pairwise_vqa_aggregation_report.py <multirun_dir>` instead — it discovers all jobs, checks prompt/pair-set equality, and emits pooled + per-model + normalized TrueSkill plus inter-model Cohen's κ.

### Zone-geometry aggregation

When a run has **no unit identity** (image-mode sampler — e.g. the sterility runs, where `pairs.parquet` has no `unit_uid_*`), `pairwise_vqa_report.py` can rate a **containing polygon** instead of the image. It spatial-joins each side's point to the zone and sets `unit_*` from the zone, so the normal TrueSkill path then rates zones.

| Flag | Purpose |
|---|---|
| `--zone-geojson PATH` | Polygon file (any geopandas-readable). Enables zone mode. |
| `--zone-id-column` | Zone property used as the rated unit id (e.g. `geoid`). Required. |
| `--zone-name-column` | Display name; comma-separated props are joined with ` · ` (e.g. `boroname,ctlabel,ntaname`). |
| `--coords-parquet` + `--coords-id-column`/`--coords-lat-column`/`--coords-lon-column` | Coordinate lookup, used when `pairs.parquet` lat/lon are absent/null. Joined to pairs via `--id-col-a`/`--id-col-b` (default `sample_id_a`/`sample_id_b`). |
| `--point-crs` | CRS of the point coords. Default `EPSG:4326`. |

Gotchas: NYC Open Data GeoJSON exports carry **no embedded CRS** (assumed WGS84 lon/lat); the sterility `pairs.parquet` lat/lon columns are **string-typed and all-null**, so coords must come from the source Cyclomedia parquet via `--coords-parquet` keyed on `sample_id` (use its float `lat`/`lon`, not the string `latitude`/`longitude`). Points outside every polygon (water, boundary) are dropped from TrueSkill with a count warning (~0.6% for Manhattan tracts).

## References

- Herbrich, Minka & Graepel (NIPS 2006). [TrueSkill: A Bayesian skill rating system](https://papers.nips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html).
- `trueskill` Python package: <https://trueskill.org/>. Pinned in `pyproject.toml` as `trueskill>=0.4.5`.
- `notebooks/css/wealth.ipynb` — canonical usage in this project (image-level → PUMA → tract aggregation + regression against ACS covariates).

## See Also

- [[urban-pair-vqa]] — the dagspace that produces the input match table
- [[concept-counterbalancing]] — why `relative_score` is already de-swapped before TrueSkill sees it
