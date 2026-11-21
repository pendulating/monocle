# Tables for the FAccT results section

To change a table, edit `build_tables.py` and run it from the repo root. Do not edit a `.tex` file by hand.

The numbers come from data files. The rows match `vlm-narratives-docs/facct-claim-map.md`.

| File | Label | Place in `sec/4_results.tex` |
|------|-------|------------------------------|
| `inference-kinds-ses.tex` | `tab:inference-kinds-ses` | Claim 1, after L18 |
| `uncertainty-by-case.tex` | `tab:uncertainty-by-case` | Claim 2, the TODO at L37 |
| `uncertainty-reasons.tex` | `tab:uncertainty-reasons` | Claim 2, after L37 |
| `quote-sources.tex` | `tab:quote-sources` | Claim 3, the TODOs at L45 to L51, or the appendix |
| `descriptive-prescriptive.tex` | `tab:descriptive-prescriptive` | Claim 3, the prescriptive text |
| `cue-prices.tex` | `tab:cue-prices` | Claim 4, after L75 |
| `paradigm-cue-order.tex` | `tab:paradigm-cue-order` | Claim 4, at L82 |
| `trace-ses-counts.tex` | `tab:trace-ses-counts` | Claim 5, at L92 |
| `h9-summary.tex` | `tab:h9-summary` | Claim 5, after L100 |

Each table needs `booktabs` and `graphicx`. The paper already uses both in `tables/`.
