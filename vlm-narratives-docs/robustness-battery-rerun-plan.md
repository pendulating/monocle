# Robustness battery re-run on the canonical setup

Plan of 2026-09-04. The CVPR appendix table `tab:robustness` and the
"downstream estimand" paragraph come from the July 2026 reviewer-2 battery.
That battery ran a prompt, a decoding setting, a layout, and a pair draw the
paper no longer uses. This plan re-runs it on the canonical setup. Check each
item off when it is done.

## What changes against the July battery

| Item | July battery | This re-run |
|------|--------------|-------------|
| Base prompt | Persona, cue list, "appears safer" | `pairwise_subway_safety_ordinal`, no system turn, "looks safer" |
| Decoding | Temperature 0.6, temp-0 arm as the ceiling | Temperature 0.0 (greedy). A retest arm reads the batch-effect floor. |
| Gemma-4-12b layout | Default, `images_then_text` | `interleaved_labels` |
| Pairs | 1k prefix of the old 100k draw | 1,000 canonical pairs cut from the 1M table, row for row |
| Anchor | A0 retest at 0.6 | The registered canonical 1M run on the same `pair_id` rows, plus a greedy retest |
| Persona arm | Persona removed | Persona added (the canonical prompt has none) |
| Gemma resolution arm | Not run | `max_soft_tokens` 280 to 140 |
| Proxies for the estimand | Old tract income and crime per capita | The v1.2.0 proxies of the case notebook |

The raters stay the same 4: gemma-4-12b, qwen3.5-9b, qwen3.5-2b, gemma-4-e2b.
Section 3 of the paper uses this appendix to justify the choice of the 2 large
raters, thus the small ladder stays in.

## Base configuration for each rater

The base is the config of the registered canonical run. Read it from
`multirun/million_merged/subway_safety__<model>/.hydra/config.yaml`, never from
a sweep file.

| Rater | Layout | Other |
|-------|--------|-------|
| gemma-4-12b | `interleaved_labels` | 1 GPU, 40 GB card or more |
| qwen3.5-9b | default (`images_then_text`) | `max_pixels` 1048576, `limit_mm_per_prompt.image` 2 |
| qwen3.5-2b | default | as the 9b |
| gemma-4-e2b | `interleaved_labels` (decided 2026-09-04 from the 100-pair probe) | Standard `gemma4` arch with a vision encoder, but it does not bind 2 adjacent images either |

Shared: `prompt=pairwise_subway_safety_ordinal`, `allow_not_sure: true`,
`temperature: 0.0`, `max_tokens: 128`, `pair_seed` 777, `repeat_fraction`
0.1, `counterbalance_mode: balanced`, `pair_sampler.mode: unit`.

**Warning:** the working tree holds an uncommitted edit to
`conf/sweep/million_proxy_gemma12b.yaml` that sets `images_then_labels`. That
contradicts the decision of 2026-09-02 and the registry. Do not copy it.

## Phase 0. Pre-flight

- [x] 0.1 Run `python research/scripts/register_canonical_runs.py verify`. It must
      report 0 FAIL (last known: OK, 77 runs, 2026-09-04 11:01).
- [x] 0.2 Revert or correct the layout line in
      `conf/sweep/million_proxy_gemma12b.yaml` so the file says
      `interleaved_labels` again. Do not launch from it in this plan.
- [x] 0.3 (qwen3.5-2b and gemma-4-e2b load and answer on the default venv, smoke of 2026-09-04 20:53) Confirm the 4 model configs load on the default venv
      `.venv-mllmsci-vllm025cu129`: gemma-4-12b, qwen3.5-9b, qwen3.5-2b,
      gemma-4-e2b. The canonical 1M runs prove the first 2. The small 2 last ran
      in July on older venvs.
- [x] 0.4 Confirm `multirun/pair_tables_1m/image_pack` covers the subway rows.
      Pass `runtime.image_pack` to every job, as the noabstain resume did.

## Phase 1. Pair tables

- [x] 1.1 Build a 1,000-pair select table:
      `multirun/pair_tables_select_1k/subway_safety_pairs.parquet`, with the
      column `canonical_pair_id` for the first 1,000 canonical pairs of
      `multirun/pair_tables_100k/subway_safety_pairs.parquet` in draw order.
- [x] 1.2 Check that these 1,000 ids equal the ids of the July A0 probe
      (`multirun/2026-07-10_URBANPAIRVQA/19-21-27/0`). The draw does not depend
      on the prompt or the model, so they should match. If they match, the new
      table and the July table score the same station pairs.
- [x] 1.3 Cut the 1k rows out of the 1M table:
      `python scripts/subset_pair_tables.py --parent multirun/pair_tables_1m
      --select multirun/pair_tables_select_1k --out-dir
      multirun/pair_tables_1k_of_1m --cases subway_safety`.
      Expect about 1,100 rows and a sidecar with `subset.canonical_pairs: 1000`.
- [x] 1.4 Build a 25,000-pair select table the same way
      (`multirun/pair_tables_select_25k`) and cut it to
      `multirun/pair_tables_25k_of_1m`. Expect about 27,500 rows.
- [x] 1.5 Verify each cut against the canonical gemma run: 100% of `pair_id`
      rows join, same images in the same order.

## Phase 2. Prompt variants

New files under `research/conf/pairwisevqa/prompt/`, each a copy of
`pairwise_subway_safety_ordinal.yaml` with ONE change. The
`structured_output` block stays byte-identical so guided decoding does not
change.

- [x] 2.1 `pairwise_subway_safety_paraphrase.yaml`: a full reword of the 3
      lines of the user template. Same unit noun, same question, same 5 labels,
      same interpretation line. No cue list, no persona.
- [x] 2.2 `pairwise_subway_safety_persona.yaml`: the canonical user template,
      plus the July expert system turn
      ("You are an expert urban planner and sociologist...") as `system:`.
      This is the inverse of the July `nopersona` arm.
- [x] 2.3 `pairwise_subway_safety_enumrev.yaml`: the label list in the prompt
      reads MuchMore, More, Same, Less, MuchLess. The schema enum does not
      change.
- [x] 2.4 `pairwise_subway_safety_flipped.yaml`: "Which station entrance looks
      less safe?" and the interpretation line "<label> unsafe". The analysis
      negates its scores (`--flip-arms flipped`).
- [x] 2.5 Check `research/scripts/register_canonical_runs.py:arm_of` does not read any
      of the 4 new names as a registered arm. Only `_nocity` and `_matched`
      are suffix arms today. Phase 6 adds the 2 names it needs.
- [x] 2.6 Wait 2 minutes after the last edit before any launch (NFS stale
      module).

## Phase 3. Sweep and launch tooling

- [x] 3.1 New sweep `conf/sweep/robustness_1k_<rater>.yaml`, 1 file for each of
      the 4 raters: `pair_sampler.max_pairs: 1000`, `shard_count: 1`,
      `max_tokens: 128`, `resume: true`, launcher `slurm_gpu_preempt`,
      `array_parallelism: 1`, the rater's layout from the table above, and
      `pipeline: pairwise_subway_safety_mvp`.
- [x] 3.2 New script `research/scripts/launch_robustness_battery.sh`. For each rater and
      each arm it calls `dagspaces.pairwisevqa.submit_shards --sweep
      robustness_1k_<rater> --base-dir multirun/robustness_<stamp>/<rater>/<arm>
      --pairs-dir multirun/pair_tables_1k_of_1m --set <arm overrides>`. It
      sleeps 20 s between submits (the sbatch ssh shim refuses a burst).
- [x] 3.3 `--dry-run` of every (rater, arm) pair. Read the printed command and
      confirm the `--set` override wins over the sweep value for `prompt`,
      `temperature`, and `image_layout`.
- [x] 3.4 (done for e2b R0/F1/L1/L2 and qwen2b R0; the gemma-4-12b layouts and the 140 budget already ran in production on this venv) Smoke test with a fresh 20-pair draw (`pair_sampler.max_pairs=20`, no
      `pairs_dir`) for the fragile cells: gemma-4-12b L1 and L2, gemma-4-e2b
      F1, L1 and L2. Check for the one-image signature (NotSure near 80%,
      Same near 30% with no Much* labels).

## Phase 4. Tier A: the 1k agreement battery

44 jobs, about 1,100 rows each. A job is about 5 minutes of inference plus the
model load. Run every arm of a rater from the same sweep file.

Arms and their overrides. Every arm is ONE change against the rater's base.

| Id | Arm | Override | Notes |
|----|-----|----------|-------|
| R0 | retest | none | Greedy floor. Expect near 1.0; a gap is the batch effect. |
| W1 | paraphrase | `prompt=pairwise_subway_safety_paraphrase` | |
| W2 | persona | `prompt=pairwise_subway_safety_persona` | Persona ADDED |
| W3 | enumrev | `prompt=pairwise_subway_safety_enumrev` | |
| W4 | flipped | `prompt=pairwise_subway_safety_flipped` | Negate scores |
| W5 | nocity | `prompt=pairwise_subway_safety_nocity` | Optional. Links to the registered 100k arm. |
| F1 | res-half | qwen: `model.engine_kwargs.mm_processor_kwargs.max_pixels=262144`; gemma: `model.engine_kwargs.mm_processor_kwargs.max_soft_tokens=140` | Fills the July dash for gemma |
| F2 | freeform | `prompt.structured_output.enabled=false` | |
| F3 | temp0.6 | `sampling_params_vqa.temperature=0.6` | The July production temperature |
| L1 | alt layout | both gemma: `prompt.image_layout=images_then_labels`; both qwen: `interleaved_labels` | |
| L2 | text-first | both gemma: `prompt.image_layout=text_first_interleaved`; both qwen: `text_first` | |

Checklist, 1 line for each rater:

- [x] 4.1 (DONE 2026-09-04 21:45, base multirun/robustness_a_20260904_210129) gemma-4-12b: R0 W1 W2 W3 W4 W5 F1 F2 F3 L1 L2
- [x] 4.2 (done) qwen3.5-9b: R0 W1 W2 W3 W4 W5 F1 F2 F3 L1 L2
- [x] 4.3 (done) qwen3.5-2b: R0 W1 W2 W3 W4 W5 F1 F2 F3 L1 L2
- [x] 4.4 (done) gemma-4-e2b: R0 W1 W2 W3 W4 W5 F1 F2 F3 L1 L2
- [x] 4.5 (44 of 44 runs, 1,091 rows each, every proof ok) Every run holds about 1,100 rows. The shard submitter writes only the
      4 base overrides to `.hydra/overrides.yaml`, thus the arm is proven by
      `.hydra/config.yaml` (prompt text, layout, temperature, budget). The
      launch script names the run dir after the arm and writes `arms.tsv`.
- [x] 4.6 (no one-image signature on gemma-4-12b; e2b L1 is a known non-binding layout) Landing check for each run: the NotSure rate and the Same rate. Flag
      any gemma cell with the one-image signature.

## Phase 5. Tier A analysis and the table

- [x] 5.1 (plus `first_share`, the side bias, which the kappa cannot see) New module `research/notebooks/cvpr/_robustness.py`, in the style of
      `_reliability.py`. For each (rater, arm) it joins the arm to R0 and to
      the registered canonical run on `pair_id`, and reports: linear-weighted
      kappa, raw agreement, sign-flip rate, NotSure rate, Same rate, Spearman.
      The canonical join exists for gemma-4-12b and qwen3.5-9b only; the small
      raters read against R0 alone.
- [x] 5.2 The flipped arm is negated before the join.
- [x] 5.3 (plus the abstention and side-bias companion tables) The module writes `research/notebooks/cvpr/master/outputs/robustness_v1.0.0.tex`
      and the long CSV. The table: rows are the arms, columns are the 4 raters,
      the first row is the R0 greedy floor. No temp-0 row.
- [x] 5.4 (tables/robustness.tex, robustness-abstention.tex, robustness-side.tex; all 3 are input) Copy the generated table to `papers/.../tables/robustness.tex` and
      `\input` it in place of the hand-written table in
      `sec/B_addtl-results.tex`.
- [x] 5.5 (done 2026-09-04; tier B results and the merge trap added 2026-09-04 23:50) Doc page `vlm-narratives-docs/robustness-battery.md`: the arms, the
      commands, the results, and the traps.

## Phase 6. Tier B: the downstream estimand at 25k pairs

4 runs: paraphrase and persona, on gemma-4-12b and qwen3.5-9b, over the 25k
cut. Gemma needs about 2 h in 1 shard; qwen about 4 h. Use 2 shards each
(`--shards 2`) under the 480-minute wall. The 2b control drops out: it is
not a canonical rater.

- [x] 6.1 (done 2026-09-04; also ARM_CASES and QUESTION_PROBE_ARMS, because a subway-only arm and a reworded question both failed the old gates) Registry: add arms `paraphrase` and `persona` to `ARMS` and
      `MIN_ROWS` (25,000) in `research/scripts/register_canonical_runs.py`, and teach
      `arm_of` to read them from the prompt name in `.hydra/config.yaml`.
      Add the labels to `_ablations.ARM_LABEL`.
- [x] 6.2 (launched 2026-09-04 21:24, base multirun/robustness_b_20260904_212400, 8 shards, all 8 COMPLETED by 23:20) Launch the 4 runs from the same launch script with
      `--pairs-dir multirun/pair_tables_25k_of_1m` and `max_pairs=25000`.
- [x] 6.3 (merged to `multirun/robustness_b_merged/{paraphrase,persona}/subway_safety__<model>`, 27,442 rows each; the merge now copies the shard `.hydra/config.yaml`, see the trap in robustness-battery.md) Merge with `scripts/merge_pairwise_shards.py` to
      `multirun/robustness_25k_merged`.
- [x] 6.4 (done 2026-09-04 23:40: 12 roots, 81 runs, verify OK) Register with ALL roots plus the new one and `--force`
      (`register` rebuilds the manifest; a 1-root call erases the rest).
      Then `verify` must stay at 0 FAIL.
- [x] 6.5 (done 2026-09-04 23:41; `research/notebooks/cvpr/master/outputs/robustness_estimand_v1.0.0.csv`) Read each arm with `_proxy_arms.py` with `match_arm=` set, so the
      canonical control reads the same 25k pairs. Report the v1.2.0 proxies:
      tract median household income, station crime per 100k riders, felonies
      per 100k riders. Report `d_tau` beside `spread_ratio`.
- [x] 6.6 (done; `robustness_estimand_tract_corr_v1.0.0.csv`: 12B 0.955/0.934, 9B 0.750/0.637) Tract-score correlation between each arm and its matched control.
- [x] 6.7 (done from the canonical 1M run: tract income r 0.421/0.423 gemma, 0.458/0.454 qwen on the unswapped/swapped halves; half correlation 0.995 and 0.968) Counterbalanced halves: recompute the per-half income correlation
      from the registered canonical 1M run (the old "freebies" tier). No GPU.

## Phase 7. Paper text

- [x] 7.1 `sec/A_addtl-methods.tex`, Robustness Checks: drop the temperature
      0.6 sentence and the temp-0 ceiling; say the 1,000 pairs are rows of
      the 1M table; the persona arm ADDS a persona; the resolution arm runs
      on both families.
- [x] 7.2 (done 2026-09-04 23:45; the estimand paragraph now reads tier B, the 2B sentence is gone) `sec/B_addtl-results.tex`: replace the table with the generated one;
      rewrite the 4 paragraphs from the new numbers; drop the "2B control"
      sentence from the estimand paragraph; use the current proxy names.
- [x] 7.3 The data comment at the top of `B_addtl-results.tex` names the new
      output directories.
- [x] 7.4 `sec/3_methods.tex`, Model selection: the sentence still holds only
      if the small ladder still fails under the canonical prompt. Rewrite from
      the new table.
- [~] 7.5 (the sign is fixed; the 46%/52.6% source is still open) `sec/4_results.tex`, Robustness paragraph: "$\leq 0.997$" must read
      "$\geq 0.997$". Find the source of the 46% and 52.6% east-versus-west
      values or replace them with values from `camera-face-effect.md`.

## Compute

| Tier | Jobs | Rows for each job | Wall for each job | GPU-hours |
|------|------|-------------------|-------------------|-----------|
| A | 44 | about 1,100 | 10 to 20 min | about 12 |
| B | 8 shards | about 13,750 | 1 to 2 h | about 12 |

## Open decisions

1. **gemma-4-e2b layout.** The plan first took the default layout, as in
   July. The smoke test of 2026-09-04 (20 fresh pairs) says otherwise: the
   default layout and `text_first` give Same on 100% and 91% of the rows,
   and `interleaved_labels` gives a spread (Same 64%). That is the adjacency
   signature of the 12b. A 100-pair probe over 5 layouts
   (`multirun/robustness_smoke_e2b_layouts_20260904_205908`) decides the
   base before the e2b arms launch.
2. **W5 nocity.** Cheap, and it ties the 1k table to the registered 100k arm.
   Keep it unless the table gets too wide.
3. **Tier B registration.** The registry route follows the rule "never point a
   figure at a run directory". The alternative is a direct-path reader in
   `_proxy_arms.py`. The plan takes the registry route.
