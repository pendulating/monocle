# The master results table

`notebooks/cvpr/_results_table.py`, `notebooks/cvpr/master/results_table.py`,
and `scripts/export_cvpr_results_table.py`.

## What it is

One camera-ready table over every case of the canonical arm. The case sits
at the first indent, and its proxies indent under it. Each rater gets 4
columns (n, agreement, Pearson r, Kendall tau-b), and a fifth block compares
the 2 raters with each other.

The master notebook holds the canonical arm alone since 2026-09-04. The
prompt ablations sit in `master/ablations.py`. See
[prompt-ablations.md](prompt-ablations.md).

Since 2026-09-06 the paper shows the validation as 1 figure and 2 small
tables, not as this table. `master/proxy_overview.py` reads the exports of
this notebook and writes them. See [proxy-overview.md](proxy-overview.md).
This table stays as the source of the figure and as the appendix table.

## The rules a renderer applies

The long frame (`results_table_<layer>_v*_long.csv`) keeps every row. The
LaTeX and the on-screen table apply 4 rules, in this order, and the footnote
of the table states each rule that fired.

| Rule | Constant | What it does |
|------|----------|--------------|
| Superseded | `SUPERSEDED_PROXIES` | Holds back an old measurement when the same case carries its replacement |
| Cap | `MAX_PROXIES`, `TABLE_PROXIES` | A case prints at most 4 proxies; the list names them |
| Area floor | `MIN_AREAS` (20) | Drops a proxy row where no rater reaches the floor; dashes a rater under it |
| Context | `CONTEXT_PROXIES` | Marks a row that describes the place with a dagger; never hides it |

The cap holds back only a row that reaches the area floor. A row under the
floor stays with the floor, so the footnote names each row once, under the
rule that removed it.

## The cap (2026-09-04)

The user asked for at most 4 proxies for each case. `TABLE_PROXIES` names
the rows by case folder and proxy key. A case that is absent from the list
prints every row. The rule that picked the 4:

1. The socio-economic anchor stays: household income, or the economic need
   index for schools.
2. The headline proxy of `_unit_validation.CASE_SPEC` stays.
3. A context row stays. The reader must see the confounder next to the
   claim it explains.
4. Of 2 measures of the same thing, the one that names the unit's own
   condition stays.

| Case | Prints | Held back |
|------|--------|-----------|
| Subway safety | income, rider entries (context), transit crime, transit felonies | none (4 rows) |
| Libraries | income, building violations, alteration permits, construction year | open building violations |
| Schools | economic need index, attendance (context), building violations, kept clean | alteration permits, construction year, open building violations |
| Road quality | income, DOT rating (length-weighted), crime density, pothole complaints | pothole repairs per mile, years since resurfacing |
| Parks | income, PIP acceptable rate, crime density, acreage (context) | PIP condition rate, acquisition year |
| Plazas | income, DOB and ECB violations, BID membership, crime density | BID maintenance partner |
| Restaurants | inspection score, grade A rate, outdoor dining (context) | none (3 rows) |

To change a row, edit `TABLE_PROXIES` and run the export script again. The
module refuses a list longer than `MAX_PROXIES`.

## Run it

```bash
.venv-mllmsci-vllm025cu129/bin/python scripts/export_cvpr_results_table.py
```

The script refuses when a case export is older than the registry. Pass
`--allow-stale` when the register call changed no canonical cell, as on
2026-08-31 and 2026-09-04. See
[canonical-run-registry.md](canonical-run-registry.md).
