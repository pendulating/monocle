# The parks and plazas proxies, version 1.2.0

`research/notebooks/cvpr/_proxies_parks.py`, `parks/parks_validation.py`,
`plazas/plazas_validation.py`. Exports: `*_proxy_v1.2.0.parquet`,
`*_unit_correlations_v1.2.0.parquet`, `*_correlations_v1.2.0.parquet`,
`*_provenance_v1.2.0.csv`.

## What version 1.2.0 adds

| Case | Proxy | Source | Window | Measures |
|------|-------|--------|--------|----------|
| Parks | `pip_condition_acceptable_rate` | PIP `yg3y-7juh` | 2023-2026 | The structure of the site |
| Parks | `park_acreage` | DPR `enfh-gkve` | pulled 2026-08-31 | The SIZE of the site. No orientation. |
| Plazas | `plaza_bid_partner` | DOT plazas `k5k6-6jex` | pulled 2026-08-31 | A BID maintains the plaza |
| Plazas | `bid_membership` | BID polygons `7jdm-inj8` | as of 2026-08-31 | A district covers the location |
| Plazas | `dob_ecb_violations` | DOB `3h2n-5cm9` + ECB `6bgk-3dad` | 2021-2026, negated | Open orders at the tax lot |

The old rows do not move. Every v1.1.0 value that survives the JIA filter is
identical in v1.2.0, at all 3 layers, for both cases (verified row for row on
2026-08-31).

Cleanliness and condition are different visual constructs: a park can be
clean and broken, or dirty and sound. Version 1.2.0 exports both PIP metrics.
The PIP FEATURE ratings (benches, paths, play equipment) were found
(`5mma-5n3h`, 1.4M rows) and are NOT usable: the source carries no property
id, and its only key points at an Inspections source that serves 0 rows
today. `_proxies_parks.pip_feature_status()` re-tests this and says when the
blocker lifts.

## The join rules and their tolerances

- A BID boundary follows the sidewalk and a DOT plaza sits in the ROADBED,
  thus a strict `within` test finds 5 of 92 DOT plazas. The loader uses a
  100 ft tolerance, which finds 43. The sensitivity table is in
  `_proxies_parks.BID_TOLERANCE_FT`.
- The partner value covers the DOT plazas ONLY, and the code enforces it.
  Without the filter, 28 POPS units take the partner of a plaza beside them.
- 419 of 504 plaza units carry a usable tax lot (390 of 392 POPS). A DOT
  plaza sits in the roadbed, which owns no lot, thus its absence is by
  construction.
- A violation names the LOT, not the plaza surface. Read it as the upkeep
  capacity of the owner.

## The floor decision (plazas)

`_proxy_arms.CASE_SPEC["plazas"]` moved to `min_units=3` on 2026-08-31 and
the evidence says 1 is right. Measured at the community-district layer after
the JIA filter:

| Floor | Broad rows | Violations | Partner |
|-------|-----------|------------|---------|
| 1 | 43 areas | 27 | 36 |
| 3 | 16 areas | 11 | 10 |

`MIN_AREAS` is 20, thus at floor 3 EVERY plaza row leaves the paper. The NTA
layer does not rescue floor 3: 21 areas on the broad rows, 8 on the partner
row. The decision: keep floor 1 at the area layer, and answer the
single-plaza-district cost with the UNIT layer, where such a district cannot
appear. **Change `_proxy_arms.CASE_SPEC["plazas"]` back to `min_units=1`.**
The parks floor stays 1: no park community district holds a single park.

## What the units say (canonical arm, 2026-08-31)

Brackets hold a 95% bootstrap interval over units (1,000 draws, seed
20260831). Every value reconciles exactly with the
`*_unit_correlations_v1.2.0.parquet` export.

| Case | Proxy | gemma-4-12b | qwen3.5-9b |
|------|-------|-------------|------------|
| Plazas | `dob_ecb_violations` (n=395) | **+0.158 [+0.06, +0.26]** | **+0.139 [+0.04, +0.24]** |
| Plazas | `plaza_bid_partner` (n=81) | +0.191 [-0.03, +0.38] | **+0.245 [+0.05, +0.44]** |
| Plazas | `bid_membership` (n=470) | +0.031 [-0.06, +0.12] | +0.087 [-0.01, +0.17] |
| Plazas | `year_completed` (n=342) | +0.015 [-0.09, +0.12] | +0.014 [-0.09, +0.13] |
| Parks | `pip_acceptable_rate` (n=1,054) | -0.023 [-0.08, +0.03] | +0.062 [+0.00, +0.12] |
| Parks | `pip_condition_acceptable_rate` (n=1,054) | -0.068 [-0.13, -0.01] | +0.064 [+0.00, +0.12] |
| Parks | `park_acreage` (n=1,573) | **+0.285 [+0.24, +0.33]** | +0.077 [+0.03, +0.13] |

Three readings:

1. **The violation record is the first plaza proxy that both raters agree
   with at the unit.** The interval excludes 0 on both. A plaza on a cleaner
   lot takes a higher score.
2. **The 2 raters disagree in SIGN on both PIP metrics.** The parks case has
   no unit-level agreement on upkeep, in either metric.
3. **gemma-4-12b reads park SIZE.** Acreage carries no orientation, thus
   +0.285 is a finding about the model, not an agreement: that rater calls a
   larger park better. It is also that rater's strongest parks association,
   which marks size as a confounder for its PIP rows.

## The area layer, for contrast

At the community-district layer (59 areas; plazas 43): parks income tau
+0.47/+0.26, PIP tau +0.15/+0.03 — the area layer flatters the SES row, the
same pattern `unit-validation.md` documents. New plaza rows: bid_membership
tau +0.21/+0.28, violations -0.11/+0.04 (27 areas), partner +0.17/+0.21 (36
areas). The area violation row DISAGREES with its own unit row; the
aggregation makes the difference, thus read the unit row.

## The shared-file changes this version needs

Applied by the owner of `_results_table.py` / `_proxy_arms.py`, not here:

1. `_proxy_arms.CASE_SPEC["plazas"]` → `min_units=1` (see the floor
   decision).
2. `PROXY_LABEL` additions: `pip_condition_acceptable_rate` → "PIP condition
   acceptable rate", `park_acreage` → "Park acreage", `bid_membership` →
   "BID membership", `plaza_bid_partner` → "BID maintenance partner",
   `dob_ecb_violations` → "DOB and ECB violations (negated)".
3. `UNORIENTED_PROXIES` gains `park_acreage`.
4. `_unit_validation` seams: parks can add `pip_condition_acceptable_rate`
   and `park_acreage` through the same property map; plazas becomes a unit
   case with the identity key (`unit_uid` → `unit_uid`) on the 3 capacity
   proxies.
