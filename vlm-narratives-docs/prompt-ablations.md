# The prompt ablations

Two batteries judged the SAME image pairs as the paper battery, with ONE thing
changed each. The registry holds them as **arms** beside the canonical prompt.

| Arm | What changed | Pairs for each case | Rows |
|-----|--------------|---------------------|------|
| `canonical` | nothing — the prompt the paper reports | 1,000,000 | 15,400,000 |
| `noabstain` | the decoder cannot emit `NotSure` | 1,000,000 | 15,400,000 |
| `nocity` | the prompt does not name the city | 100,000 | 1,540,000 |
| `textfirst` | the prompt text comes BEFORE the images | 100,000 | 1,540,000 |
| `matched` | nothing — the canonical prompt, `images_then_labels`, on the 100,000-pair table (gemma-4-12b only) | 100,000 | 770,000 |

The batteries ran 2026-08-21 to 2026-08-27 and are complete: 966 of 966 shards
for `noabstain`, 105 of 105 for `nocity`, and 105 of 105 for `textfirst`. The
`matched` control launched 2026-09-02 (35 shards). See the section below.

## The `textfirst` arm

Each rater makes ONE change against ITS OWN production layout.

| Rater | Production | The arm |
|-------|------------|---------|
| qwen3.5-9b | `images_then_text` — A, B, prompt | `text_first` — prompt, A, B |
| gemma-4-12b | `images_then_labels` — A, "Image A.", B, "Image B.", prompt | `text_first_labels` — prompt, A, "Image A.", B, "Image B." |

**Warning: gemma-4-12b does not take plain `text_first`.** That layout puts
the 2 images next to each other, and this arch reads 2 adjacent image blocks
as 1 image: 55 of 70 traces speak of one image and `NotSure` reaches 81.4%
(measured 2026-08-20). The anchors of `text_first_labels` keep the images
apart, thus the arm measures the position of the prompt and nothing else.

We tested for that failure before we merged the run. The first shard of 3
cases gives `NotSure` at 0.2%, 19.9% and 0.0% for gemma, thus the binding
held.

`text_first_labels` is a layout of the pairwise stage, added 2026-08-26. The
registry gives the arm an exemption from `REQUIRED_LAYOUT`
(`LAYOUT_PROBE_ARMS`), because a layout probe that the layout gate refuses
cannot exist.

**Warning: this arm is a probe, not a default.** Do not copy its layout into a
new sweep. See `feedback_images_then_labels_always`.

## The join key is `pair_id`

**Join on `pair_id`. Do not join on `canonical_pair_id`.** `pair_id` carries
the presentation order and `canonical_pair_id` does not.

We measured the overlap on all 7 cases on 2026-08-26:

| Arm | Shares `canonical_pair_id` | Shares `pair_id` |
|-----|---------------------------|------------------|
| `noabstain` | 100% | 100% of 1,100,000 rows |
| `nocity` | 100% | 91.7% of 110,000 rows |

The draw at seed 777 is **nested**: the 100,000-pair draw is a subset of the
1,000,000-pair draw. An earlier comment in the `nocity` sweep files said the
opposite, and we corrected it.

**Warning:** about 8% of the pairs that `nocity` shares with the canonical
battery came up in the OPPOSITE A/B order. A join on `canonical_pair_id`
compares a pair with its own mirror image, thus it reads the order effect as a
prompt effect.

## What to read it with

| Module | What it gives |
|--------|---------------|
| `research/notebooks/cvpr/_ablations.py` | the join, the statistics, the display table |
| `research/notebooks/cvpr/master/ablations.py` | sections 1 to 4, and the CSV export |
| `research/scripts/export_cvpr_ablations.py` | the same export, from a command |

```python
import _ablations as A
A.all_arms()                       # 1 row for each (case, model, arm)
A.abstention_flow_table()          # where the held-back judgments went
A.to_display(A.all_arms())         # the on-screen table
```

The export button of the ablations notebook, or the script, writes
`ablations_v*.csv` and `ablation_abstention_flow_v*.csv` to
`research/notebooks/cvpr/master/outputs/`. The master table in `results_table.py`
holds the canonical arm alone since 2026-09-04, so the paper presents the
ablations apart from it.

## The statistics

| Column | Meaning | No effect |
|--------|---------|-----------|
| `n_pairs` | Pairs both arms judged, in the same order | -- |
| `match_rate` | Share of the arm's rows the join kept | 1.0 |
| `label_agreement` | Same label of the 6 | the marginals |
| `label_kappa` | The same, corrected for chance | 0 |
| `direction_kappa` | Same side, corrected for chance | 0 |
| `score_delta` | Mean ordinal score, arm minus canonical, PAIRED | 0 |
| `score_delta_se` | Standard error of that paired mean | -- |

The standard error is the **paired** one. Both arms judged the same pair, thus
it uses the standard deviation of the difference. It is much smaller than an
unpaired error, and the test is much sharper.

**Warning:** `label_agreement` is not a quality measure. It answers "how far
did the prompt change move the labels?". A low value is a large effect.

**Warning:** a row under `MIN_BOTH` ANSWERED pairs shows no direction and no
score. A score needs a pair that BOTH arms answered. The qwen3.5-9b schools
`nocity` row leaves 26 such pairs of 100,938.

## What we found

### The abstentions hide no judgment

`lean` is the mean ordinal score of the forced labels, over the pairs the
canonical arm declined. It is about 0 on every case and both models. The 2
models decline in opposite ways:

| Case | gemma-4-12b | qwen3.5-9b |
|------|-------------|------------|
| Schools | 94.8% `Same` | 12.77% `Less` / 12.85% `More` |
| Subway safety | 84.1% `Same` | 41.98% `Less` / 41.94% `More` |
| Parks / Plazas | 58.8% `Same` | 45.16% `Less` / 45.05% `More` |

gemma-4-12b names the tie. qwen3.5-9b splits between the 2 directions, and the
split is symmetric to 3 decimals over as many as 730,000 pairs. Neither model
gives an opinion that it held back.

The aggregate does not move: `score_delta` is 0.000 with a standard error of
about 0.001 on all 14 `noabstain` rows, even where 2 thirds of the pairs were
abstentions.

### The place name moves the labels, not the order

Label agreement runs 0.39 to 0.85 and label kappa runs 0.22 to 0.78, thus the
place name changes which label lands on a pair. But `score_delta` is about 0 on
every readable row, thus the ordering does not move.

Street photography is the least disturbed case, and it is also the only case
with no abstention in either arm. The cases that abstain most are the cases the
place name moves most.

### The layout decision of 2026-09-02, and the re-run

The canonical gemma layout is `interleaved_labels` (see
`gemma-4-best-practices.md`). The first gemma halves of `nocity` and
`textfirst` ran `images_then_labels` and `text_first_labels`, thus they held
the layout as well as the named change. Both re-ran on 2026-09-02:

| Item | `nocity` re-run | `textfirst` re-run |
|------|------|------|
| Sweep | `million_nocity_gemma12b_il` | `million_textfirst_gemma12b_il` |
| Layout | `interleaved_labels` | `text_first_interleaved` (prompt, "Image A:", A, "Image B:", B) |
| Pairs | `multirun/pair_tables_100k_of_1m` | the same |
| Shards | 5 for each case, 35 jobs | 5 for each case, 35 jobs |
| Launch | `bash research/scripts/launch_ablation_rerun_gemma.sh` | the same script |
| Merge target | `multirun/nocity_merged_il` | `multirun/textfirst_merged_il` |
| Arrays | 200662 | 200680 |
| Finished | 2026-09-02 18:43, 35 of 35 | 2026-09-02 18:58, 35 of 35 |
| Registered | 2026-09-02 19:00, `proxy/nocity/<case>__gemma-4-12b` | `proxy/textfirst/<case>__gemma-4-12b` |

Both arms join the canonical gemma results on `pair_id` at 100% on all 7
cases. The abstention rate on the shared pairs:

| Case | canonical | `nocity` new | `nocity` old layout | `textfirst` new | `textfirst` old layout |
|------|------|------|------|------|------|
| Schools | 51.8% | 52.6% | 63.3% | 28.2% | 19.4% |
| Parks / Plazas | 43.6% | 49.6% | 59.9% | 35.2% | 33.6% |
| Restaurants | 28.8% | 27.2% | 44.7% | 15.8% | 15.4% |
| Libraries | 14.3% | 13.0% | 21.1% | 10.0% | 15.2% |
| Subway safety | 6.9% | 6.7% | 11.5% | 0.1% | 0.3% |

With the layout held fixed, the place name moves the abstention rate by 6
points on parks and by 2 points or less on the other cases. The 10 to 16
point rise the first run showed was the layout. The prompt position still
suppresses abstention on every case that abstains, as the first text-first
run found: that effect was real and not the layout. `_ablations.LAYOUT_CONFOUNDED`
is empty since 2026-09-04 (see below).

**The pairs are rows of the 1,000,000-pair table.** A fresh 100,000-pair
draw shares every canonical pair with the 1,000,000-pair draw, but about 8% of
them come up in the opposite A/B order, thus only 91.7% of the `pair_id` rows
join. `scripts/subset_pair_tables.py` cuts the rows of the 1,000,000-pair
table whose canonical pair is in the 100,000-pair draw: about 109,950 rows for
each case, every one a byte-for-byte row of the parent. Verified 2026-09-02 on
all 7 cases: 100% of the rows are in the canonical gemma results, with the
same images in the same order. The canonical pairs are the same 100,000 pairs
the qwen halves judged.

The runner accepts a subset table for a job whose settings match the parent
in everything but `max_pairs` (`_subset_table_matches`), and the prebuild
refuses to draw over a subset sidecar.

**The `noabstain` gemma half re-ran too, and registered 2026-09-04.** It
also ran `images_then_labels`. Sweep `million_noabstain_gemma12b_il`
(`interleaved_labels`, abstention off, the `pair_tables_1m` tables, 46 shards
for each case, 322 jobs) launched 2026-09-02 23:16 as array 217915, base
`multirun/noabstain_battery_20260902_231642`. That array made the `/share/ju`
server collapse and was cancelled; the 200 unfinished shards resumed as array
252078 on the image pack (see `nfs-load-pairvqa-sweeps.md`) and finished
2026-09-04 09:30. Merged to `multirun/noabstain_merged_il`, registered with
all 10 roots, `verify` OK with 77 runs. Every gemma cell of every arm now
sits on the layout of its control, and `_ablations.LAYOUT_CONFOUNDED` is
empty.
The `matched` arm (canonical prompt on `images_then_labels`) is now a layout
probe, not a control, and the registry lists it under `LAYOUT_PROBE_ARMS`.

### The gemma `nocity` rows carry a confound (the first run, before 2026-09-02)

**Warning:** the canonical gemma control ran `interleaved_labels`. Thus a
gemma delta of an arm holds the named change AND a layout change:

| Cell | What the delta holds |
|------|----------------------|
| `nocity` / gemma-4-12b | the place name AND the layout |
| `textfirst` / gemma-4-12b | the prompt position AND the modality order |

`_ablations.LAYOUT_CONFOUNDED` named the cells, and the display marked them,
until the re-runs of 2026-09-02 and 2026-09-04 cleared it.
The qwen cells of both arms read cleanly.

Read the qwen rows as a place-name effect. To read the gemma rows the same way,
read them against the `matched` arm below.

### The `matched` arm is the gemma control

The 1,000,000-pair gemma control did not serve the 2 small arms, for 2 reasons:

| Quantity | `canonical` / gemma-4-12b | `nocity` and `textfirst` / gemma-4-12b |
|----------|---------------------------|----------------------------------------|
| Layout | `interleaved_labels` | `images_then_labels` / `text_first_labels` |
| Pairs | 1,000,000 | 100,000 |
| `pair_id` shared with the arms | 91.7% | 100% between the 2 arms |

We measured the overlap on 2026-09-01 on schools and street photography. The 2
arms share every `pair_id` row with each other, because both read the same
prebuilt table in `multirun/pair_tables_100k`.

The `matched` arm runs the canonical prompt, with abstention on, on
`images_then_labels`, over that SAME table. It changes nothing else. Thus it
joins both arms pair for pair, and a gemma delta against it holds the named
change alone.

| Item | Value |
|------|-------|
| Sweep | `million_matched_gemma12b` |
| Pipelines | `pairwise_<case>_mvp_matched`, which inherit the base case and override nothing |
| Launch | `bash research/scripts/launch_matched_battery.sh` |
| Shards | 5 for each case, 35 jobs, `slurm_gpu_preempt`, 480 min |
| Launched | 2026-09-02 00:06, array 135242, `multirun/matched_battery_20260902_000637` |
| Finished | 2026-09-02 11:45, 35 of 35 shards COMPLETED, none lost |
| Merged | `multirun/matched_merged`, 110,000 rows for each case |
| Registered | 2026-09-02 11:51, `proxy/matched/<case>__gemma-4-12b`, 7 runs, no problems |

The merged rows join `nocity` and `textfirst` on `pair_id` at 100% on all 7
cases. The registry knows that this arm holds 1 rater (`ARM_MODELS`), thus
the grid check does not ask for a qwen half.

**Warning: the layout moves the abstention rate on its own.** The register
step prints the `NotSure` share of each run:

| Case | `canonical` (`interleaved_labels`, 1M) | `matched` (`images_then_labels`, 100k) | `nocity` / gemma |
|------|------|------|------|
| Schools | 51.5% | 62.0% | 63.3% |
| Parks / Plazas | 43.7% | 54.2% | 59.9% |
| Restaurants | 27.9% | 44.9% | 44.7% |
| Libraries | 14.2% | 24.8% | 21.1% |
| Subway safety | 6.8% | 12.9% | 11.5% |
| Road quality | 0.3% | 0.2% | 0.1% |
| Street photography | 0.0% | 0.0% | 0.0% |

The `matched` rate sits next to the `nocity` rate, not next to the `canonical`
rate. Thus most of the abstention change that the gemma `nocity` row showed
against `canonical` was the layout, not the place name. Read the gemma rows
against `matched` before you quote a delta.

The registry reads the `_matched` suffix and files the run as the `matched`
arm (`ARM_SUFFIXES`, `MIN_ROWS`). Without the suffix the run would read as
`canonical`, collide with the 1,000,000-pair cell, and fail the
1,000,000-row floor.

**Warning: `matched` is a control, not an ablation.** Do not put it in a
delta as an arm. Read `nocity` / gemma and `textfirst` / gemma against it.
The qwen halves keep the `canonical` control: that rater ran 1 layout in the
arms and in the control.

`_ablations.py` and `_proxy_arms.py` read every arm against `canonical`.
Since the re-runs of 2026-09-02 and 2026-09-04 that is the correct control
for the gemma cells too, and `LAYOUT_CONFOUNDED` is empty. The `matched`
arm is a layout probe (`LAYOUT_PROBE_ARMS`), not a control.

The gemma re-runs are complete: `verify` is OK with 77 runs since
2026-09-04. The 2 warnings it keeps are the qwen schools cells that answer
NotSure on 99.7% of rows.

## What the arms do to the OUTSIDE measurements

The table above reads an arm against the canonical arm alone. To read an arm
against a proxy — an outside measurement of the same place — see
[proxy-arms.md](proxy-arms.md). It rebuilds the place scores, the geography
join and the 3 metrics for each arm, on the pairs that both arms judged.

The headline: the 3 arms do not drift together. `nocity` costs agreement (19
of 32 cells fall), `noabstain` compresses the scale without moving the order,
and `textfirst` halves the scale and holds the one clean reversal of the whole
table.

## Related

- `proxy-arms.md` — each arm read against the outside measurements
- `quantile-shift.md` — how far an arm moved the score of a place
- `canonical-run-registry.md` — the arm dimension and the registry gates
- `million-pair-battery.md` — the shards, the resume, and the pair tables
- `gemma-4-best-practices.md` — why `images_then_labels` is required
