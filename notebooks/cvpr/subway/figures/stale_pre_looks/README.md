# Stale figures — the extraction panels of the old prompts

These files come from the extraction corpus of `thinking_public_investment_10k`
(2026-08-13), which read the traces of the PREVIOUS prompts. On 2026-08-14 every
prompt moved to "looks like" wording, thus these panels answer a question the
paper no longer asks.

Keep them for history only. Do not put one in the paper.

To make them again from the canonical runs:

1. Run the extraction stage on the registered trace runs.
2. `python scripts/merge_trace_extractions.py`
3. `python scripts/export_cvpr_trace_figures.py`

`_extractions.registry_mismatch()` says whether the corpus is current.
