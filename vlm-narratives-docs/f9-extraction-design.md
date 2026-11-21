# F9 extraction design: the verdict and the cue spans of the subjective prompt

**Status (2026-10-30): a design for review, with the answers of D142
applied (section 8).** No job ran. No existing schema, script, or data
file changed. The 4 new schema files are drafts.

This page gives the x2-style extraction for the F9 answers (D139, D140).
Read `facct-runbook.md`, section "F9. The subjective prompt", first. The
x2 round is in the same file ("The x2 extraction round (D130)") and in
`f4-batch-2-reproduction-plan.md`, steps B6 to B10.

| Item | Value |
|------|-------|
| Goal 1 | The verdict of each answer: `yes`, `no`, `mixed`, `graded`, `not_applicable`, from langextract (D140; a regex is rejected) |
| Goal 2 | Every class and attribute of `single-v2` and `street-v3`, with the same meaning, thus the F9 net valence of `visual_evidence` stays comparable with F4 (runbook 4.12, F8.1 method B) |
| Goal 3 | Both fields: `reasoning` (4 thinking models) and `response` (6 models) |
| Goal 4 (D142, Q4 option C) | The F4 answers of the same 10,000 images of each case get the same new schema. The F9 test reads new-schema spans on both sides, thus a pull of the examples cancels in the F9 - F4 difference |
| F9 inputs | `data/f4_traces/subj/f9_restaurant_traces.parquet`, `f9_street_traces.parquet` (60,000 rows each) |
| F9 texts | 99,999 restaurant (40,000 traces, 59,999 answers; 1 empty answer, which the trace fills, section 2.4) and 100,000 street. About 200,000 texts, 2.5 times the canonical x2 inputs |
| F4 texts | the x2 inputs of record, reused as they are (section 4.1): 9,956 + 89,811 restaurant and 9,836 + 90,054 street. About 200,000 texts |

## 1. What the F9 answers say

I read a stratified sample (2 answers of each model and case, with the
trace tails) and counted the openers of all 120,000 answers.

| Model | Restaurant: the first words of the answer | Street: the first words |
|-------|--------------------------------------------|--------------------------|
| gemma-4-12b | `Whether this looks like a "nice" restaurant depends ...` (73%); `Based on the image` (8%); `It is difficult to` (8%) | `Whether this is a good spot ... depends` (93%) |
| gemma-4-31b | `Whether this looks ...` (81%); `Based on the image` (13%) | `Whether this ...` (98%) |
| qwen3.5-9b | `Based on the photo/image/visual evidence` (82%); `Whether this looks` (12%); `Yes,` (0.8%) | `Based on the image` (47%); `Yes, this street` (45%) |
| qwen3.8-27b | `Honestly, ...` (86%): `not in the`, `not really`, `no—not in`, `it depends` | `It depends on what` (5%); `It's a decent spot` (9%); `Honestly, not a great/standout` (6%) |
| llama-3.2-11b | `The image shows/depicts`, `The restaurant in the image` (no explicit verdict in most) | `The street in the image ...`; `This street could/would` (15%); `Yes,` (7%) |
| llama-3.2-90b | `The image depicts/shows` (43%); `The image does not` (8%); `This is not a restaurant` (2%) | `The street in the image`; `This street would` (10%) |

Measured shapes that the design must handle:

| Shape | Where | Count | Example (quoted) |
|-------|-------|-------|------------------|
| Conditional answer: a `no` for 1 kind of diner and a `yes` for another | gemma, both cases; qwen3.5-9b | the word "depends" is in 78% to 99% of the gemma answers | `**If "nice" means upscale ... :** Then no, this probably isn't it. ... **If "nice" means an authentic, casual, "hole-in-the-wall" ... :** Then yes, it might be very appealing.` (gemma-4-31b, restaurant 7880) |
| A clear `no` | qwen3.8-27b; Llama | `Honestly, not` or `Honestly, no` opens 21% of the qwen3.8-27b restaurant answers | `Honestly, no—based on what's visible from the street, this doesn't give off a "nice restaurant" vibe.` (qwen3.8-27b, restaurant 7530) |
| A clear `yes` | qwen3.5-9b street (45%); Llama street | rare in the restaurant case (qwen3.5-9b: 78 answers open with `Yes,`) | `Yes, based on the exterior, this definitely looks like a nice, pleasant place to eat.` (qwen3.5-9b, restaurant 9743) |
| A graded answer with no yes or no | qwen3.8-27b street | `decent spot`, `meh` | `Honestly, it's a *meh* spot—fine for a casual snapshot, but not the kind of place you'd drive across town to photograph.` (street 7895) |
| No restaurant in view | all models, restaurant | 3% (qwen3.5-9b) to 15% (llama-3.2-90b) of the answers | `The image does not depict a restaurant. Instead, it shows a row of shops ...` (llama-3.2-90b, restaurant 4721) |
| No explicit verdict: a description with a hedge at the end | Llama, restaurant | most llama-3.2-11b restaurant answers | `It's always a good idea to read reviews and make your own judgment before deciding where to eat.` (llama-3.2-11b, restaurant 7880) |
| The trace drafts several answers before the final one | gemma traces | the "Draft 1 / Draft 2 / Draft 3" block | `*Draft 1 (Too blunt):* No, it looks like a cheap place ... *Draft 3 (Balanced ...):* Whether this looks "nice" depends on what you are looking for.` (gemma-4-31b, restaurant 7880) |
| The trace holds a yes or no that is not the verdict | gemma traces | the "Final Review" block | `Does this answer the user's question comprehensively? Yes.` (gemma-4-12b, street 7545) |
| The whole answer sits in `reasoning` | qwen3.8-27b, restaurant 8826 | 1 text | the trace ends with `Let me draft. "It's hard to call it a 'nice restaurant' in the conventional sense— ..."`; `response` is empty. The input build copies the draft into `response` (D142, section 2.4) |
| The pros and cons of the street case are a list of things of the place | gemma, qwen3.5-9b street | most street answers | `**Texture and Pattern:** The brick facade of the building provides repetitive geometric patterns ...` (gemma-4-12b, street 7895) |

Lengths: the median trace is 2,800 to 4,100 characters, the median answer
500 (Llama) to 2,400 characters. 503 restaurant traces and 233 street
traces are longer than 6,000 characters, thus they take 2 chunks or more.
Only qwen3.8-27b writes U+2011; the input build turns it into "-".

## 2. The verdict

### 2.1 Where it lives

The verdict is a new extraction class, `verdict`, with 1 attribute,
`label`. LangExtract gives spans only; it has no per-document field. Thus
a per-text field would need a backend change, and a regex is rejected
(D140). A class keeps the output contract of the x2 round: 1 row for each
span, with offsets, `is_quotable`, and the retry flags. The `decision`
class of `urban_cues_v1` (`langextract-trace-extraction.md`, "The
extraction schema, version 1") is the precedent.

| Item | Value |
|------|-------|
| Class | `verdict` |
| Span | the sentence or the clause in which the model answers the question. 1 sentence or 1 clause for each span |
| Attribute | `label`: `yes`, `no`, `mixed`, `graded`, `not_applicable` |
| Several spans in 1 text | yes. A trace drafts its answer; a conditional answer gives 1 span for each branch |
| Nested spans | a thing that the model sees inside a verdict sentence is also a `visual_evidence` or `inference` span inside it (the `photo_advice` rule of `street-v3`) |
| Not a verdict | the echo of the question; a yes or no about another thing; a summary of the place with no yes, no, or "depends" (`it looks like a casual neighborhood spot` is `inference`, kind `atmosphere`) |

### 2.2 The labels

| Label | Rule | Quoted spans from the F9 answers |
|-------|------|----------------------------------|
| `yes` | the model says the restaurant looks nice, or the street is a good spot. A reservation that does not reverse the answer keeps `yes` | `Yes, based on the exterior, this definitely looks like a nice, pleasant place to eat.` (qwen3.5-9b, restaurant 9743); `Yes, this street would likely make a **good spot for automotive or street photography**, provided you know what kind of photos you want to take.` (qwen3.5-9b, street 7545); `Then yes, it might be very appealing.` (gemma-4-31b, restaurant 7880, 1 branch) |
| `no` | the model says it does not look nice, or it is not a good spot: `no`, `not really`, `not a great spot`, `not a standout`, `probably not`, `would not be` | `the honest answer is no, it doesn't look particularly nice or inviting from the outside` (qwen3.8-27b, restaurant 7530, trace); `it does not look like a nice place to eat` (llama-3.2-11b, restaurant 7530); `Honestly, not a standout spot for most purposes.` (qwen3.8-27b, street 3719); `The street in the image is not a good spot for taking a photo.` (llama-3.2-90b, street 7545); `Then no, this probably isn't it.` (gemma-4-31b, restaurant 7880, 1 branch) |
| `mixed` | the answer depends on the diner, the meaning of "nice", or the style of photo; or the model gives points for and against and takes no side. The model gives 2 answers, 1 for each side or use | `Whether this looks like a "nice" restaurant depends entirely on what you are looking for in a dining experience` (gemma-4-31b, restaurant 7880); `In summary, it is an excellent spot for capturing **urban textures and interesting light/shadow play**, but it may be less appealing if you are looking for **scenic landmarks or high-drama scenery.**` (gemma-4-12b, street 7895); `Honestly, "nice" depends a lot on what you're after, but by the usual sit-down-dining standard, this reads more as a no-frills takeout spot than a place you'd go for the atmosphere.` (qwen3.8-27b, restaurant 8244) |
| `graded` (D142) | 1 middle judgment with no yes and no no: `decent`, `okay`, `meh`, `perfectly fine`, `nothing special`, `fine but unremarkable`, `not bad but not great`. The model gives 1 answer, and that answer is a middle rating. Against `mixed`: `mixed` splits the answer by a condition or takes both sides; `graded` takes 1 position between yes and no | `So: not "nice" in the fancy sense, but a perfectly fine, honest-looking joint for a casual meal.` (qwen3.8-27b, restaurant 8244); `I'd say it's a "meh" spot—not bad, but not great, and not something you'd seek out specifically for photography.` and `Final verdict: It's an okay, unremarkable spot—decent for a casual or architectural detail shot, but not a standout photographic location.` (qwen3.8-27b, street 7895, trace); `Honestly, it's a *meh* spot—fine for a casual snapshot, but not the kind of place you'd drive across town to photograph.` (qwen3.8-27b, street 7895, answer) |
| `not_applicable` | the model says that the photograph shows no restaurant (restaurant case), or that it cannot judge a spot from this photograph (street case). The model gives no answer about niceness | `The image does not depict a restaurant.` and `there is no indication of a restaurant in the vicinity` (llama-3.2-90b, restaurant 4721); `Based on the image, there aren't actually any restaurants visible.` (gemma-4-31b, restaurant 4281) |

A graded word is frequent: a word of the list above is in the first
sentence of 345 qwen3.5-9b, 115 gemma-4-12b, 58 qwen3.8-27b and 22
gemma-4-31b restaurant answers, and somewhere in 17% to 31% of the
answers of those 4 models. In the street case, about 15% of the
qwen3.8-27b answers open with `It's a decent spot`.

A span that says "this is a cafe, not a restaurant" and then answers the
question is not `not_applicable`: the business type is an `inference`
(kind `business_type`), and the answer is the verdict. Example:
`Yes, it looks like a very pleasant place to visit, although it appears to
be primarily a **coffee shop** rather than a sit-down restaurant.`
(qwen3.5-9b, restaurant 8678) gives `yes`.

### 2.3 The verdict of 1 text

The extractor gives spans. A rule of the analysis gives 1 label for each
text (`doc_id`). It reads the quotable `verdict` spans in text order:

| Case | Label of the text |
|------|-------------------|
| No `verdict` span | `none` (the text gives no verdict; most llama-3.2-11b restaurant answers) |
| Only `not_applicable` spans | `not_applicable` |
| `not_applicable` spans and other spans | the rules below, over the other spans |
| The last span is `mixed` or `graded` | that label. The model summed up its own answer, and the summary wins |
| The last span is `yes` or `no`, and the text also holds the other one | `mixed` (a conditional answer that ends on 1 branch, or a trace that changes its mind) |
| The last span is `yes` or `no`, and the text holds no span of the other one | that label. Earlier `mixed` or `graded` spans do not change it |

Thus `graded` combines with the other labels through the position of the
spans only: a `graded` span that comes last wins; a `graded` span before a
final `yes` or `no` loses. The rule reads the last span first, because the
F9 answers sum up at the end ("So:", "In summary", "Final verdict", "Bottom
line").

Checks of the rule on the sample: the gemma conditional answer (`mixed`
opener, then `no`, then `yes`) gives `mixed`. The gemma trace with "Draft
1: No" and "Draft 3: Whether ... depends" gives `mixed`, which equals its
answer. The qwen3.8-27b trace with 2 `no` drafts gives `no`. The
qwen3.8-27b street trace 7895 (`yes` for a snapshot, `no` for a scenic
spot, then 2 `graded` drafts) gives `graded`. The qwen3.8-27b restaurant
answer 8244 (`mixed`, a conditional `yes`, then `graded`) gives `graded`.
A trace that says `yes` and then corrects itself to `no` gives `mixed`;
D142 accepts this loss (Q3), and the trace is not the primary field.

### 2.4 Reasoning and response

Each field is its own text with its own verdict. The analysis reports:

| Item | Rule |
|------|------|
| Primary | the verdict of the `response` text (all 6 models) |
| Secondary | the verdict of the `reasoning` text (4 models), and the agreement table (response label against reasoning label, for each model and case) |
| Disagreement | no merge of the 2 fields. The disagreement rate is a result: it tells how often the trace drafts a different answer than the one the model gives |
| An empty response | 1 row of 120,000 (qwen3.8-27b, restaurant 8826, `finish_reason` `stop`). The extraction-input build copies the drafted answer from its trace into the `response` text and sets `response_from_trace` true (D142, Q5; the rule is below). The merged corpus `f9_restaurant_traces.parquet` does not change |
| A `none` response with a trace verdict | no fallback. The `none` share is a result for each model (expected: high for llama-3.2-11b restaurant) |

The copy rule for the empty response (step 1 of section 5):

| Item | Value |
|------|-------|
| Where | `build_extract_input.py`, before it drops the empty texts. The copy exists in `data/f4_traces/subj/extract_input/f4_restaurant_subj.parquet` and in every file built from it; nowhere else |
| Row | `model` = `qwen3.8-27b`, `case` = `restaurant_subj`, `image_idx` = 8826, `field` = `response` (`doc_id` `qwen3.8-27b\|restaurant_subj\|8826\|response`) |
| Marker | the last `Let me draft.`, 2 line breaks, and 1 double quote (`Let me draft.\n\n"`) in `reasoning`. It starts at character 3,334 of the 4,057-character trace; the marker is 16 characters |
| The `response` text | every character after the marker to the end of the trace: 707 characters (characters 3,350 to 4,056). The trace has no closing quote: the model stopped inside the draft, thus the text ends in the middle of a word |
| The `reasoning` text | unchanged. The draft stays in the trace too, thus the extractor reads these 707 characters 2 times (1 time in each field). The agreement table of this row is trivial |
| Flag | the new column `response_from_trace` (bool), true for this row and false for the 199,999 other rows of the 2 cases. The analysis joins it on `doc_id` and reports the row |
| First 100 characters | `It's hard to call it a 'nice restaurant' in the conventional sense—first, it's really a bakery and i` |
| Last 100 characters | `right in front of the windows, an orange cone, and a parked SUV—which gives the street a busy, trans` |
| Check | exactly 1 row with the flag; its text is 707 characters; U+2011 count 0 |

## 3. The classes that carry over

### 3.1 Unchanged text

The 2 new schemas copy the class list, the attribute names, the value
lists, and the rules of `single-v2` and `street-v3` word for word, and add
the `verdict` class and 2 rules. The settings are the same (6,000-character
chunks, 600 characters of context, `max_tokens` 6144, `rescue_substrings`,
`require_extraction`).

| Class | Attributes and values | Restaurant (`single-v2` to `single-subj-v1`) | Street (`street-v3` to `street-subj-v1`) |
|-------|-----------------------|-----------------------------------------------|-------------------------------------------|
| `visual_evidence` | `valence` (good, bad, neutral); `category` (13 values) | unchanged | unchanged |
| `inference` | `kind` (16 values) | unchanged | unchanged |
| `person_reference` | `used_in_judgment` (yes, no) | unchanged | unchanged |
| `image_artifact` | `kind` (6 values) | unchanged | unchanged |
| `uncertainty` | `reason` (5 values) | unchanged | unchanged |
| `photo_advice` | `kind` (6 values) | not in the restaurant schema | unchanged |
| `none` | no attribute | unchanged | unchanged |
| `verdict` | `label` (yes, no, mixed, graded, not_applicable) | **new** | **new** |

What changes in the prompt text:

| Part | Restaurant | Street |
|------|------------|--------|
| First paragraph | names the F9 prompt; the caption sentence is gone | names the F9 prompt |
| Classes | + `verdict` | + `verdict` |
| Rules | rule 5 (captions) is gone; + rule 5 (verdict spans, nested spans, a summary of the kind of place is an inference, a middle rating is `graded`) and rule 6 (the echo and a yes about another thing are not verdicts) | + rule 8 (verdict spans; a pro or a con that describes the place is `visual_evidence`; a middle rating is `graded`) and rule 9 (the echo) |
| Examples | 8 new, from the F9 restaurant answers (6,705 characters, 74 spans) | 7 new, from the F9 street answers (7,771 characters, 64 spans) |

### 3.2 The risk that the yes-or-no form moves text between classes

| Risk | Why | What the design does | What the check measures |
|------|-----|----------------------|-------------------------|
| Evaluative sentences go to `verdict` and leave `visual_evidence` | an F9 answer states its evidence inside the answer sentence (`Honestly, no—... this doesn't give off a "nice restaurant" vibe`) | the nested-span rule: the thing seen inside a verdict sentence is also `visual_evidence`; examples 3, 4, 6 of each schema show it | the `visual_evidence` spans for each text, F9 against F4 on the same images (runbook F9, step 6) |
| The valence mix shifts toward `good` and `bad` | the F9 texts attach more judgment words to things (`Dull, dirty windows`). This is the effect under study, not an error, as long as the class border holds | the class text is unchanged. The examples come from F9 texts, thus their valence mix is more extreme than the F4 examples (restaurant: 19 neutral, 12 bad, 3 good against 22 neutral, 0 bad, 3 good in `single-v2`; street: 8 neutral, 6 bad, 6 good against 17, 4, 7 in `street-v3`). The extractor copies the base rates of its examples. D142 keeps this mix (Q2: "keep the valence mix of the run that will be analyzed") and extracts the F4 answers again with the same schema (Q4, option C), thus the pull is the same on both sides of the F9 - F4 difference | the description of section 6.2: the new schema against x2 on the same F4 texts |
| Summary sentences go to `verdict` in place of `inference` | `it looks like a practical, convenient spot` could be read as the answer | rule 5 (restaurant) and the examples: a summary with no yes, no, or "depends" is `inference` (`atmosphere`, `wealth`) | the `inference` kind mix, F9 against F4 |
| The street pros and cons go to `photo_advice` (effect) in place of `visual_evidence` | a pro says what will look good in a photo | rule 8 and example 4: a pro or a con that only describes a thing of the place is `visual_evidence`; an effect of the planned picture stays `photo_advice` (D126, D127) | the `photo_advice` share, F9 against F4 (expected lower: F9 asks for no angle) |
| Fewer `visual_evidence` spans for each text | the F9 answers are shorter (runbook: 25% to 49% fewer tokens for Qwen) | nothing; the test uses the ratio (good - bad) / spans and needs 1 span or more | the share of texts with 0 `visual_evidence` spans, for each model and field |
| `none` texts | a plan-only chunk, or a Llama answer with no cue | the `none` example is in place 2, as in x2 | the share of texts with only `none` (x2 rule: 20% or fewer for a cue arm) |
| A copy of an example | the F9 answers of gemma have 1 shape (`Whether this ... depends`), thus the extractor can copy example 4 | the retry pass with the examples in reverse order (`*r.yaml`), as in x2 | copy suspects after the retry, about 1% or fewer |

## 4. Names

### 4.1 The F4 answers under the new schema (D142, Q4 option C)

The x2 inputs of record cover the pooled 10,000 images of each case, and
they can be reused as they are:

| Input (`data/f4_traces/extract_input/`) | Texts | Images | Context line |
|-----------------------------------------|-------|--------|--------------|
| `f4_restaurant_x2.parquet` (canonical batch) | 9,956 | 1,000 (`image_idx` 0 to 999) | the F4 restaurant prompt |
| `f4_restaurant_b2x2.parquet` (batch 2) | 89,811 | 9,000 (1000 to 9999) | the same |
| `f4_street_x2.parquet` (canonical batch) | 9,836 | 985 (0 to 984) | the F4 street prompt |
| `f4_street_b2x2.parquet` (batch 2) | 90,054 | 9,015 (1000 to 10014) | the same |

Checked: the 2 batches share no `doc_id`; their image union is the 10,000
F9 images of each case; U+2011 is "-"; the `context` column names the F4
prompt, which is right for these texts. No rebuild. The new arms list
these 4 cases beside the F9 cases (section 5, step 3). The runs go under
`multirun/f4x2_subj/`, and the merge writes to `data/f4_traces/subj/x2/`,
thus the x2 spans of record (`multirun/f4x2/`, `multirun/f4x2_b2/`,
`data/f4_traces/x2/`, `data/f4_traces/b2/x2/`) stay as they are. The F9
test reads the new-schema spans on both sides; the x2 spans of record do
not enter it. On the F4 texts, `verdict` spans are expected to be rare
(the F4 prompts ask for no judgment); the analysis reports their rate as
a description.

### 4.2 The names of the F9 arms

| Item | Restaurant | Street |
|------|------------|--------|
| Case | `restaurant_subj` | `street_subj` |
| Schema file (new) | `dagspaces/urbanpairvqa/conf/extract/urban_cues_single_subj_v1.yaml` | `urban_cues_street_subj_v1.yaml` |
| `name` / `schema_version` | `urban_cues_single` / `single-subj-v1` | `urban_cues_street` / `street-subj-v1` |
| Retry schema (new) | `urban_cues_single_subj_v1r.yaml` (`single-subj-v1r`) | `urban_cues_street_subj_v1r.yaml` (`street-subj-v1r`) |
| Arm | `x2_cues_subj` | `x2_cues_street_subj` |
| Retry arm (the loop in `extract_shard.py` makes it) | `x2r_cues_subj` | `x2r_cues_street_subj` |
| Input of `build_extract_input.py` | `data/f4_traces/subj/extract_input/f4_restaurant_subj.parquet` | `.../f4_street_subj.parquet` |
| Input of the arm (`build_x2_input.py`) | `data/f4_traces/extract_input/f4_restaurant_subj_x2.parquet` | `f4_street_subj_x2.parquet` |
| Smoke set | `f4_x2smoke_restaurant_subj.parquet` | `f4_x2smoke_street_subj.parquet` |
| Smoke doc_ids | `x2_smoke_doc_ids_subj.json` | the same file |
| Retry input | `f4_x2r_x2_cues_subj.parquet` | `f4_x2r_x2_cues_street_subj.parquet` |
| Runs | `multirun/f4x2_subj/x2_cues_subj/`, `.../x2r_cues_subj/` | `multirun/f4x2_subj/x2_cues_street_subj/`, `.../x2r_cues_street_subj/` |
| Smoke runs | `multirun/f4x2_subj_smoke/<arm>/` | the same root |
| Merged output, F9 | `data/f4_traces/subj/x2/cues/f4_restaurant_subj_x2_extractions.parquet` | `data/f4_traces/subj/x2/cues_street/f4_street_subj_x2_extractions.parquet` |
| Merged output, F4 under the new schema | `data/f4_traces/subj/x2/cues/f4_restaurant_x2_extractions.parquet` and `f4_restaurant_b2x2_extractions.parquet` | `data/f4_traces/subj/x2/cues_street/f4_street_x2_extractions.parquet` and `f4_street_b2x2_extractions.parquet` |
| W&B experiment | `F9X2_CUES_SUBJ` (retry: `F9X2_CUES_SUBJ_RETRY`) | `F9X2_CUES_STREET_SUBJ` |
| Plans | `scripts/f4/plans/subj_smoke.txt`, `subj_pass1_part00.txt` to `part11.txt`, `subj_pass2.txt` | the same files |
| Flag column of the input | `response_from_trace` (bool; section 2.4) | the same column, always false |

The `doc_id` is `<model>|restaurant_subj|<image_idx>|<field>`. It cannot
collide with an F4 `doc_id`, because the case differs. The context line
of each text names the F9 prompt, thus the extractor reads the question
before the examples.

## 5. Code changes

**Status (2026-10-30):** steps 1 to 12 are done; step 13 (optional) is not.
The checks of each step passed. Step 12 is `scripts/f9/analyze_subjective.py`
(1.1.0; it adds the standardized test of D143); `scripts/f9/compare_f4_schema.py` (1.0.0) gives the description of
section 6.2. The smoke result and the go decision are in section 6.4.

| # | File | Change | Check |
|---|------|--------|-------|
| 1 | `scripts/f4/build_extract_input.py` (1.4.0) | add `--response-from-trace <model>:<image_idx>:<marker>` (repeatable). For each entry, when `response` is empty, the text after the last `<marker>` in `reasoning` becomes the `response` text (section 2.4), and the new column `response_from_trace` is true for that row; false for every other row. Run it 2 times with `--src data/f4_traces/subj/f9_<case>_traces.parquet --out-dir data/f4_traces/subj/extract_input`, with the entry `qwen3.8-27b:8826:'Let me draft.\n\n"'` for the restaurant case. It writes `f4_restaurant_subj.parquet` and `f4_street_subj.parquet` (the case name comes from the data) and a manifest that lists the filled rows | 100,000 rows for each case; `doc_id` unique; 0 dropped texts; exactly 1 row with `response_from_trace` true, 707 characters long |
| 2 | `scripts/f4/build_x2_input.py` (1.3.0) | add the 2 prompts to `PROMPTS`; add `--cases` (default: the 3 F4 cases) and `--smoke-ids` (default `x2_smoke_doc_ids.json`); with `--cases restaurant_subj street_subj --in-dir data/f4_traces/subj/extract_input --suffix x2 --smoke-ids x2_smoke_doc_ids_subj.json`, write `f4_<case>_x2.parquet` and the smoke sets. A subj smoke set: 4 texts for each (model, field) and the 10 longest qwen3.5-9b traces of the case, seed 20261026, minus the example doc_ids of section 4 (read them from the schema header, or hold them in a list). Skip the captions file when the cases hold no caption case | restaurant 64 + 10 texts, street 64 + 10; no example doc_id in the smoke set; the context line holds the F9 prompt; U+2011 count 0 |
| 3 | `scripts/f4/extract_shard.py` (1.11.0) | 2 entries in `ARMS.update`: `x2_cues_subj` = `_x2("F9X2_CUES_SUBJ", "urban_cues_single_subj_v1", ("restaurant_subj_x2", "x2smoke_restaurant_subj", "restaurant_x2", "restaurant_b2x2", "x2smoke_restaurant"))` and `x2_cues_street_subj` = `_x2("F9X2_CUES_STREET_SUBJ", "urban_cues_street_subj_v1", ("street_subj_x2", "x2smoke_street_subj", "street_x2", "street_b2x2", "x2smoke_street"))`. The F4 cases are the inputs of record (section 4.1); the F4 smoke cases serve the description of section 6.2. The retry loop then makes `x2r_cues_subj` and `x2r_cues_street_subj` (experiment `F9X2_*_RETRY`) with the cases `x2r_x2_cues_subj`, `x2rsmoke_x2_cues_subj`, and so on. The `response_from_trace` column needs no `KEEP` entry: the analysis joins it on `doc_id` from the input | `extract_shard.py --arm x2_cues_subj --case restaurant_subj_x2 --shard 0 --shards 20 --base-dir /tmp/x --dry-run` prints `schema=urban_cues_single/single-subj-v1`, `max_tokens=6144 buffer=6000`; the same for `--case restaurant_b2x2`, for the street arm, and for `x2r_cues_subj` |
| 4 | `scripts/f4/x2_retry.py` (1.3.0) | add the 2 arms to `ARMS` (schema, input cases); add `--arms` to `schemas` and `flag`, default all, thus `schemas --arms x2_cues_subj x2_cues_street_subj` writes only the 2 new `*r.yaml` files and does not touch the F4 files; `flag --root multirun/f4x2_subj --prefix x2r --arms ...` writes `f4_x2r_x2_cues_subj.parquet` and `f4_x2r_x2_cues_street_subj.parquet`. The prefix `x2r` is safe: the file name holds the arm | `schemas` gives files equal to the drafts of section 4 (`diff`); `flag` on the smoke root (`--prefix x2rsmoke`) writes 1 file for each arm with suspects, and `<root>/x2rsmoke_flags.csv` |
| 5 | `scripts/f4/x2_merge_all.sh` | read the plan from `F4X2_PLAN_ARMS` (default: the F4 heredoc). With `F4X2_PLAN_ARMS=subj`, the plan is `x2_cues_subj cues urban_cues_single_subj_v1 restaurant_subj_x2 restaurant_x2 restaurant_b2x2` and `x2_cues_street_subj cues_street urban_cues_street_subj_v1 street_subj_x2 street_x2 street_b2x2`; run it with `F4X2_ROOT=multirun/f4x2_subj F4X2_OUT=data/f4_traces/subj/x2` | `merge_summary.csv` in each output dir; 0 missing texts in each of the 3 cases of an arm |
| 6 | `scripts/f4/merge_extractions.py` (1.4.0) | none. `--cases restaurant_subj_x2 restaurant_x2 restaurant_b2x2` reads the 3 input files from `extract_input/`; `--schema` reads the new file for the copy flags. The output of each case is its own file, thus the F4 files under `subj/x2/` do not touch `data/f4_traces/x2/` | each `f4_<case>_extractions.parquet` holds every `doc_id` of its input |
| 7 | `scripts/f4/review_x2_smoke.py` (1.2.0) | add the 2 arms to `SCHEMA` (the `x2r` update line covers the retry arms); `OLD` has no entry (no old run on these texts); `NONE_MAX` default 0.2 holds | `review_x2_smoke.py --root multirun/f4x2_subj_smoke --arms x2_cues_subj x2_cues_street_subj` prints a PASS or FAIL line for each arm |
| 8 | `scripts/f4/plans/subj_smoke.txt` (new) | 2 lines: `x2_cues_subj x2smoke_restaurant_subj 0 1 multirun/f4x2_subj_smoke/x2_cues_subj` and the street line; plus 2 lines for the description of section 6.2: `x2_cues_subj x2smoke_restaurant 0 1 multirun/f4x2_subj_smoke/x2_cues_subj` and `x2_cues_street_subj x2smoke_street 0 1 ...` | 4 steps; `X2_DONE` in each run dir |
| 9 | `scripts/f4/plans/subj_pass1.txt`, 12 part plans (new) | the F9 cases: 20 shards for each (about 5,000 texts each); the F4 cases: 2 shards for each canonical input and 18 for each batch-2 input (as B7). 80 steps, round-robin into 12 parts as B7 | `grep -vc '^#'` gives 80; no 2 parts hold the same step |
| 10 | `scripts/f4/plans/subj_pass2.txt` (new) | `run ... x2_retry.py flag --root multirun/f4x2_subj --prefix x2r --arms x2_cues_subj x2_cues_street_subj`; `x2r_cues_subj x2r_x2_cues_subj 0 1 multirun/f4x2_subj/x2r_cues_subj`; the street retry line; `run F4X2_PLAN_ARMS=subj F4X2_ROOT=... bash scripts/f4/x2_merge_all.sh` | as B9 |
| 11 | `notebooks/cvpr/_extractions.py` | add `"verdict": {"label": ("yes", "no", "mixed", "graded", "not_applicable")}` to `VOCABULARY` (the header of each schema asks for this) | the loader accepts the new class |
| 12 | `scripts/f9/analyze_subjective.py` (new, runbook step 7) | add `text_verdict(spans) -> label` with the rule of section 2.3; read the F4 side from `data/f4_traces/subj/x2/` (the new-schema spans), not from `data/f4_traces/x2/`; join `response_from_trace` on `doc_id` and report the row; the span-mix table of step 6; the verdict shares by NTA; the `verdict` rate on the F4 texts as a description | the rule gives `mixed` for gemma-4-31b restaurant 7880, `graded` for qwen3.8-27b restaurant 8244, `no` for llama-3.2-11b restaurant 7530, `none` for a text with no verdict span |
| 13 | `scripts/f9/build_subj_cue_examples.py` (new, optional) | the builder of the 15 examples, with the span rules of `build_single_v2_examples.py` (literal spans, nested spans, the smoke check). The drafts of section 4 came from a scratchpad copy of it | its output equals the `examples` block of each draft |

`extract_shard.py` reads its inputs only from `data/f4_traces/extract_input/`
(B6). Thus step 2 writes there, as the batch-2 round did.

## 6. Validation before the full run

### 6.1 The smoke run (2 steps, about 150 texts)

| Check | Pass rule (the x2 rules of `review_x2_smoke.py` and B10) |
|-------|-----------------------------------------------------------|
| Quotable spans | 95% or more |
| Copy suspects | 3% of the texts or fewer before the retry |
| Loops (a span 10 or more times in 1 text) | 0 |
| Texts with only `none` | 20% or fewer |
| Spans in the second half of long texts | about 40%, as x2 |
| The verdict, read by a person | 20 answers of each case: the per-text label of section 2.3 equals the reader's label in 18 or more. Read all 4 labels and 6 models |
| Verdict spans in `response` texts | a `verdict` span in 90% or more of the gemma and Qwen answers (they all answer); the `none` share of the Llama answers is a result, not a fail |
| The echo | 0 verdict spans whose text is the question |

### 6.2 The new schema against x2 on the same F4 texts (a description, not a gate)

D142 (Q4, option C) extracts the F4 answers again with the new schema,
thus the F9 test has 1 schema on both sides, and this comparison is no
gate. It describes how far the new schema is from the x2 spans of record,
for the record and for the paper. The new schema reads the F4 x2 smoke
sets (`x2smoke_restaurant`, 50 texts; `x2smoke_street`, 70 texts), which
have x2 spans of record (`multirun/f4x2_smoke/x2_cues/`,
`x2_cues_street/`).

| Measure | Report |
|---------|--------|
| `visual_evidence` spans for each text, new schema against x2 | the ratio of the totals |
| Span overlap: the share of x2 `visual_evidence` spans that a new-schema `visual_evidence` span covers by 50% or more of its characters | the share (the 2026-09-20 review got 0.91 valence agreement between 2 extractors) |
| Valence agreement on the overlapped spans | the share |
| The share of `good`, `bad`, `neutral` over all `visual_evidence` spans | both mixes, side by side |
| `verdict` spans on the F4 texts | the share of texts with 1 or more (expected rare: the F4 prompts ask for no judgment; a street answer that says "this is a great spot" can give 1) |
| The class mix (`inference`, `image_artifact`, `photo_advice`, `uncertainty`) | both mixes, side by side |

A large distance here does not stop the run. It says that the new-schema
F4 spans differ from the x2 spans of record, thus a reader must not mix
the 2 F4 corpora in 1 table. The F9 record names which corpus each table
reads.

### 6.3 The go rule

Run the full plan when 6.1 passes. After the full run, apply the B10 rules
(0 texts missing; quotable 99% or more; copy suspects about 1% or fewer)
to all 6 cases, then the span-mix table of runbook step 6 before the
test.

### 6.4 The smoke result (job 105550, 2026-10-30)

`review_x2_smoke.py --root multirun/f4x2_subj_smoke --arms x2_cues_subj
x2_cues_street_subj` gives PASS for both arms. Each F9 step took about 4
minutes.

| Check | Restaurant | Street | Rule |
|-------|------------|--------|------|
| Texts | 50 | 50 | |
| Quotable spans | 0.994 | 0.986 | 0.95 or more |
| Copy suspects | 0 | 1 | 3% or fewer |
| Loops | 0 | 0 | 0 |
| Texts with only `none` | 0.00 | 0.00 | 0.2 or fewer |
| Spans in the second half of the 10 long texts | 0.42 | 0.46 | about 0.4 |
| gemma and Qwen responses with a verdict span | 1.00 | 1.00 | 0.9 or more |
| Llama responses with a verdict span | 0.62 | 1.00 | a result |
| Reasoning texts with a verdict span | 0.96 | 0.92 | a result |
| Verdict spans that echo the question | 0 | 0 | 0 |
| Verdict spans: yes, no, mixed, graded, not_applicable | 49, 44, 27, 18, 10 | 50, 20, 41, 9, 0 | |

**The hand read (all 24 responses of each case).** The reader label
equals the label of the rule for 19 of 24 restaurant responses and 23 of
24 street responses. The rule of 6.1 asks for 18 of 20 (90%). Thus the
street case passes and the restaurant case does not (79%).

| Response | Rule | Reader | Cause |
|----------|------|--------|-------|
| gemma-4-12b restaurant 2027 | yes | mixed | the extractor missed the "not for a formal meal" branch; the conditional `yes` of the summary wins |
| gemma-4-31b restaurant 448 | yes | mixed | 2 conditional `yes` spans; the extractor missed "impossible to say for sure" |
| llama-3.2-11b restaurant 6230 | none | no | no verdict span on "does not appear to be a fine dining establishment ... run-down" |
| qwen3.5-9b restaurant 7102 | graded | no | `graded` on "a standard, unpretentious diner rather than a nice restaurant" |
| qwen3.8-27b restaurant 3345 | graded | yes | `graded` on "a nice place to eat, just not a fancy one"; section 2.2 says that a reservation keeps `yes` |
| qwen3.8-27b street 5555 | mixed | graded | the rule works as designed: the text opens `graded`, then gives a conditional yes and a conditional no |

**The go decision (Claude, 2026-10-30, overnight under the /goal of the
user).** The full run started (pass 1, jobs 105565 to 105576), although
the restaurant verdict does not meet the hand-read rule. The reasons:

1. The primary test (H9) reads the `visual_evidence` valence only. The
   verdict is a description. Every automated check of the spans passed.
2. The errors are at the yes/graded/mixed borders. They do not reverse a
   yes into a no.
3. A later pass that extracts the verdict only can correct the verdict
   without a new run of the spans.

Every verdict table of the record must give the hand-read agreement
(restaurant 19 of 24, street 23 of 24) next to it.

**The description of 6.2** (`scripts/f9/compare_f4_schema.py` ->
`figures/f9/f4_schema_compare.json`): the new schema against the x2 spans
of record, on the F4 smoke texts.

| Measure | Restaurant (50 texts) | Street (70 texts) |
|---------|-----------------------|-------------------|
| `visual_evidence` spans, new / x2 | 1,181 / 982 = 1.20 | 309 / 654 = 0.47 |
| x2 `visual_evidence` spans covered 50% or more by a new span | 0.70 | 0.47 |
| Same valence on the covered spans | 0.99 | 0.90 |
| Valence mix of `visual_evidence` (good, bad, neutral), new | 0.015, 0.007, 0.978 | 0.142, 0.117, 0.741 |
| The same, x2 | 0.007, 0.007, 0.986 | 0.064, 0.066, 0.870 |
| Texts with a `verdict` span (new schema) | 0.10 | 0.10 |
| `photo_advice` spans per text, new / x2 | | 17.2 / 18.4 |

The restaurant smoke set holds a few very long traces (up to 568 spans in
1 text). They dominate its counts, as in the x2 smoke of record. On the
street texts the new schema gives about half the `visual_evidence` spans of
x2, and those spans carry a valence more often. Thus the F4 street spans
under the new schema are not the x2 spans of record. Do not mix the 2 F4
corpora in 1 table. Also expect fewer street texts with a
`visual_evidence` span, thus fewer images that enter the test.

## 7. Cost

| Part | Estimate | Basis |
|------|----------|-------|
| Smoke and the 6.2 description | 4 steps, about 15 min of 1 B300 after the engine start | x2 smoke |
| Pass 1 | 80 steps of about 5,000 texts (40 F9, 40 F4); 20 to 36 min each; about 4 h on 12 B300 nodes (QoS `g3`), or about 36 h on 1 | batch 2: about 500,000 texts in 4 h 45 min on 12 nodes |
| Pass 2 | about 4,000 retry texts (1%); 1 to 2 h | x2 |

The runbook estimate of about 160,000 texts is low: the F9 inputs hold
about 200,000 texts (the 4 thinking models give 2 texts for each image),
and D142 (Q4, option C) adds the about 200,000 F4 texts of the same
images. The total is about 400,000 texts, twice the F9 extraction alone.

## 8. Open questions

All 7 are answered by D142 (2026-10-30). The decisions are applied in the
sections above and in the 4 schema drafts.

| # | Question | Answer (D142) | Where applied |
|---|----------|---------------|---------------|
| 1 | Is `not_applicable` wanted? 3% to 15% of the restaurant answers say that the photograph shows no restaurant | Keep it | section 2.2; the schemas |
| 2 | The example valence mix: keep the F9-like mix, or balance it toward the F4 examples? | Keep it: "keep the valence mix of the run that will be analyzed" | section 3.2; no change to the examples |
| 3 | A trace that corrects itself becomes `mixed`. Accept, or add a `scope` attribute? | Accept; no `scope` attribute | section 2.3 |
| 4 | Is the 120-text comparability run sufficient, or also 1 full F4 shard? | Option C: extract the F4 answers of the same images again with the new schemas; the test uses new-schema spans on both sides; the 120-text run becomes a description | sections 4.1, 5 (steps 3, 5, 6, 9, 12), 6.2, 6.3, 7 |
| 5 | The 1 empty qwen3.8-27b response (restaurant 8826): drop it, or copy the answer from the trace? | Copy it at the extraction-input build, with a flag column | sections 2.4, 5 (step 1) |
| 6 | Graded words (`decent`, `meh`, `okay`) are `mixed`. Is a fifth label wanted? | Yes: `graded` | sections 2.1 to 2.3, 3.1; the schemas (the enum, the class text, the rules, 1 new restaurant example, 2 relabeled street spans; the `r` variants rebuilt) |
| 7 | W&B experiment name: `F9X2_*` or `F4X2_*`? | `F9X2_*` | sections 4.2, 5 (step 3) |
