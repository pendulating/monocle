# The "looks like" re-run of the pairwise battery

On 2026-08-14 the canonical prompts changed. A prompt now asks what the image
SHOWS, not what is true of the place. "Which station entrance looks safer?"
replaces "which is safer?". The proxy grades the place, thus a weak correlation
now means the model cannot read the property from a facade, which is the claim
the paper tests.

Every earlier run answers a different question. Thus the whole battery runs
again: 7 cases, 2 raters, 2 kinds of run.

## The 4 sweeps

All 4 live in `dagspaces/urbanpairvqa/conf/sweep/`. Each 1 holds the model AND
the 7-case axis, so a launch command carries neither.

| Sweep | Kind | Model | Pairs | Jobs | GPU-hours |
|-------|------|-------|-------|------|-----------|
| `looks_proxy_qwen9b` | validation by proxy | qwen3.5-9b/instruct | 100k | 7 | about 115 |
| `looks_proxy_gemma12b` | validation by proxy | gemma-4-12b/instruct | 100k | 7 | about 54 |
| `looks_thinking_qwen9b` | reasoning traces | qwen3.5-9b/instruct_thinking | 10k | 7 | about 85 |
| `looks_thinking_gemma12b` | reasoning traces | gemma-4-12b/instruct_thinking | 10k | 7 | about 63 |

The 7 cases, in the order the sweeps list them:

1. `pairwise_subway_safety_mvp`
2. `pairwise_libraries_mvp`
3. `pairwise_schools_mvp`
4. `pairwise_road_quality_mvp`
5. `pairwise_parks_plazas_mvp`
6. `pairwise_restaurants_mvp`
7. `pairwise_street_photography_mvp`

The thinking runs draw `max_pairs=10000`, a deterministic seed-777 PREFIX of
the 100,000-pair proxy draw. Thus pairs 0..9,999 match pair for pair, and you
can read a trace beside the greedy label for the same pair.

## Why 2 sweeps for each kind

The 2 raters cannot share 1 sweep:

- gemma-4-12b needs `prompt.image_layout=interleaved_labels`. qwen does not.
- The 2 raters differ in throughput by about 2x, so a shared array holds GPUs
  for the slower half.
- The thinking raters differ in sampling: qwen uses `top_p=0.95` and
  `max_tokens=8192`; gemma adds `top_k=64` and stops at 4,096 tokens.

## New on 2026-08-14: the gemma proxy runs pin the layout

`looks_proxy_gemma12b` sets `prompt.image_layout=interleaved_labels`. The old
`canonical_gemma12b` did not.

Under the default `images_then_text` layout, gemma-4-12b does not bind the
second image. Measured on 16 pairs with traces, 2026-08-13:

| Layout | "only one image" in the trace | NotSure |
|--------|-------------------------------|---------|
| `images_then_text` | 15 of 18 | 83% |
| `interleaved_labels` | 0 of 18 | 5.6% |

The cause is the prompt-replacement path of the encoder-free `gemma4_unified`
arch, not the thought channel. Thus a label-only run fails the same way, and it
cannot show the failure: a blind run looks like an honest abstention rate.

**Cost of the fix:** a gemma-4-12b number from this battery differs from a
gemma-4-12b number of 2026-08-11..14 for 2 reasons, the prompt and the layout.
The qwen half changed for 1 reason.

## How to launch

```bash
bash scripts/launch_looks_battery.sh --smoke     # 16 pairs, schools, each sweep
bash scripts/launch_looks_battery.sh --dry-run   # print the commands
bash scripts/launch_looks_battery.sh             # start all 4 sweeps
```

The script starts each sweep with `nohup`, because the submitit launcher waits
for its jobs. It puts all 4 monitor trees under
`multirun/looks_battery_<stamp>/` and writes a launch log for each sweep.

Each sweep runs 4 stage jobs at a time. klara holds 8 GPUs and 56 CPUs, and a
stage job takes 1 GPU and 8 CPUs, thus 7 stage jobs run together and the rest
wait. All 4 sweeps together need about 2 days of wall clock.

**Warning:** run the smoke test first when the venv, the node, or a model config
changed. A bad launch wastes a night of GPUs.

## Status: complete and canonical (2026-08-17)

All 28 runs finished. They are the registered canonical battery — see
[canonical-run-registry.md](canonical-run-registry.md). The results table and
the word-block figures come from them alone.

## After the runs land

1. DONE 2026-08-17. `CONSOLIDATION_DATE` is now 2026-08-14, and the canonical
   registry, not a date, selects the runs.
2. Check each run before you trust it. `Same` much above 30% with almost no
   `Much*` labels means the model probably never saw the images.
3. The trace runs sit in the registry under `kind=trace`. They stay out of the
   results table, because a thinking run uses different sampling and its labels
   are not comparable with the greedy battery. `_traces.py` reads them for the
   word figures.

## Superseded sweeps

These files keep a warning at the top. Use them only to repeat an old run.

| Old | New |
|-----|-----|
| `canonical_qwen9b` | `looks_proxy_qwen9b` |
| `canonical_gemma12b` | `looks_proxy_gemma12b` |
| `thinking_public_investment_10k` | `looks_thinking_gemma12b` |
| `thinking_remaining_cases_10k` | `looks_thinking_gemma12b` |
| `thinking_street_subway_10k` | `looks_thinking_qwen9b`, `looks_thinking_gemma12b` |

## Open point: 2 prompts still ask a preference

5 of the 7 prompts ask what the image looks like. 2 do not:

| Case | Question | Kind |
|------|----------|------|
| Restaurants | "Which restaurant would you rather eat at?" | preference |
| Street photography | "Which block is a more appealing location ...?" | preference |

A preference mixes the property with the reader. This is the same fault that
retired `pairwise_school_send_child_ordinal` on 2026-08-13. Decide these 2
prompts BEFORE the launch: a later edit costs about 48 GPU-hours of repeat work.
