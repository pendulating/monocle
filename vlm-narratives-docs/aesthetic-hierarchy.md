# The aesthetic hierarchy across the cases

A `<case>_traces.py` notebook draws 1 word cloud for each prompt. Its
`distinctive` score answers "which words does THIS case use and the others do
not". The 7 clouds thus give 7 vocabularies, and none of them can say whether
the model carries 1 aesthetic standard from prompt to prompt.

This report asks the opposite question over all 7 cases at once.

| Piece | Where |
|-------|-------|
| Module | `research/notebooks/cvpr/_words_across.py` |
| Notebook | `research/notebooks/cvpr/master/aesthetic_hierarchy.py` |
| Figures and tables | `research/notebooks/cvpr/master/figures/aesthetic_hierarchy_*` |
| Source | The 14 registered `trace` runs (`_canonical.py`) |

```bash
.venv-mllmsci-vllm025cu129/bin/marimo edit \
    research/notebooks/cvpr/master/aesthetic_hierarchy.py
```

## Read this after `cue-hierarchy.md`

**This report answers the narrower question.** A word count cannot separate
what a case is ABOUT from what the model VALUES, thus the top of its list is
`library`, `entrance`, and `roadway`, and its cross-case agreement sits near
0.37.

`vlm-narratives-docs/cue-hierarchy.md` answers the hierarchy question with the
typed cues. A cue names its image, so the test turns directional, and the
cross-case agreement of the cue PRICE is 0.90 over all 7 cases. Use the cue
report for a claim about the hierarchy. Use this one for a claim about the
vocabulary.

**A word can also carry the price.** The `cue_norm` unit of the cue report
keys on the word the model wrote, and it reaches 0.85 over the 5 facility
cases. That report is the one to cite for a claim that joins the vocabulary to
the hierarchy. This report counts words and never prices them.

## The question

Does the model write the same words, at the same rate, in the same order, for
a school, a park, a library, and a subway entrance?

## The scatter

1 point is 1 word. The word set is the union of the top words of each case, by
rate.

| Axis | Meaning |
|------|---------|
| x | the mean rate of the word, per 10,000 trace words |
| y | the evenness of that rate over the 7 cases, from 0 to 1 |

| Where a word sits | What it is |
|-------------------|------------|
| top right | common and even — a rung that every case keeps |
| bottom right | common and uneven — a case noun, such as `library` |
| top left | rare and even — a word of the shared background |
| bottom left | rare and uneven — a detail of 1 case |

Evenness is the Shannon entropy of the rate shares, divided by the entropy of a
flat share. 1.0 says the model writes the word at the same rate in every case.
A value near 0 says 1 case holds the word alone.

Evenness uses the RATE, never the count. A count share rewards the case with
the longest traces, and the corpora differ by more than 3 times.

## Words, or phrases

The **Token** control reads 1 word, 2 words, or 3. A phrase says what a word
cannot: `well lit`, `less safe`, `not maintained`, and `well maintained` are
the judgment, and a word count sees none of them. `well`, `less`, and `not` are
stopwords, and `maintained` alone cannot say which way the model leans.

`ngram=1` counts through `_traces.counts_for_run`, thus a word figure matches
the per-case clouds exactly. `ngram=2` and `ngram=3` count through
`_words_across.count_phrases`.

### Warning: `count_phrases` is not `_traces.count_words(ngram=2)`

`_traces.count_words` removes the stopwords BEFORE it joins the tokens.

| Text | `_traces.count_words` | `count_phrases` |
|------|----------------------|-----------------|
| `the building is clean` | `building clean` | nothing |
| `well maintained` | nothing | `well maintained` |

The first row is a phrase that nobody wrote. The second is a phrase that the
model wrote thousands of times. Do not read a `_traces` bigram cloud as a
phrase report.

### The 4 rules of a phrase

`count_phrases` joins ADJACENT tokens of the raw text, then it tests each one.

| Rule | It drops |
|------|----------|
| The phrase stays inside 1 sentence | `entrance the entrance` |
| No token is barred | `roadway looks better`, `more seems appropriate` |
| The first and the last token carry meaning | `look the`, `the entrance` |
| 1 token or more is a real content word | `very well`, `much more` |

**A plain stopword glues a phrase, thus it is barred at the edges alone.**
`cars are parked` and `near the entrance` are phrases. `cars parked` and `near
entrance` are not.

**The prompt echo, the reasoning voice, and the city name are barred
everywhere.** `looks`, `image`, `prompt`, and `york` say nothing wherever they
sit, so a phrase that holds one does not describe the photograph.
`barred_words()` builds that set from `_traces.BOILERPLATE`, `SCAFFOLD`, and
`CITY_NAME`. Without the rule a trigram set fills with `more seems
appropriate` and `roadway looks better`.

`MODIFIER` holds the degree and the polarity words: `very`, `well`, `more`,
`most`, `less`, `least`, `much`, `not`, `never`, and the rest. Every stopword
list drops them, and for 1 word that is right — `very` alone says nothing, and
`less` is a label word. In a phrase they carry the meaning, so a modifier is
exempt from the bar, it passes the third rule, and it fails the fourth.
`less safe` stays and `much more` goes.

### The city name: `new` comes back at 2 words, `york` never does

The prompt names New York in every case, thus every trace repeats it. 2
constants split that problem.

| Constant | Holds | Where it acts |
|----------|-------|---------------|
| `CITY_NAME` | `york`, `nyc`, and the 5 borough words | barred from every phrase |
| `CITY_ECHO` | `CITY_NAME` and `new` | a stopword at 1 word only |

`new` is the difficult word. At 1 word nothing separates `New York` from `new
building`, thus `new` is a stopword there. A phrase separates them: `new york`
dies on `york`, and `new building` lives.

**Warning: do not let the city into a phrase.** Measured on the registered
runs, `new york` (108 per 10,000) and `york city` (87) were the 2 most EVEN
phrases of the whole corpus. They took the top of the shared band, where they
say only that the prompt names a city.

### Warning: a trigram set still holds some reasoning voice

The 4 rules clean a bigram set well. A trigram set keeps 3 phrases of the
answer statement and the abstention instruction at the top of its shared band:
`stick with more`, `stick with less`, and `true uncertainty not`. `stick` and
`uncertainty` sit in no list, because neither is a scaffold word on its own.

Put `stick uncertainty information` in the notebook's **More stopwords** box to
drop them. Read a trigram figure as the noisier one, and quote a bigram.

### The count floor follows the token length

A bigram corpus holds about 20 times more distinct tokens than a word corpus,
thus each one is rarer. 1 fixed floor either buries the phrases or floods the
words. `min_count_for(ngram)` gives 500, 200, and 60. The notebook slider
resets to that value when you change the length.

### The export names the length

A phrase set writes `aesthetic_hierarchy_2gram_*`, so it never replaces the
word set.

## The 4 statistics

| Name | Question |
|------|----------|
| `mean_rate` | How high does the word sit in the hierarchy? |
| `floor_rate` | What does the case that writes it LEAST still write? |
| `evenness` | Does every case write it at the same rate? |
| `breadth` | How many cases rank it in their own top list? |

`floor_rate` is the strict test. A word takes a high floor only when no case
drops it.

## The 2 support figures

- `_case_pairs.png` plots each case against the mean of the other 6, and it
  gives the Spearman rho of that panel. A tight diagonal says the case keeps
  the shared order. A vertical stripe says the case writes its own words.
- `_agreement.png` gives rho for each of the 21 case pairs.

Read rho over the whole word set to ask "do the vocabularies match". Read it
over the shared words alone to ask "where the vocabulary IS shared, is the
order shared". The notebook has a slider for the second reading.

## Warning: the reasoning scaffold fills the even band

`_traces.BOILERPLATE` drops the prompt echo alone. That is right for a per-case
cloud, because the `distinctive` score divides out anything the 7 cases share.
This figure asks the opposite question, so the shared scaffold does not divide
out.

Measured on the registered runs, `looks`, `like`, `maybe`, `actually`, and
`prompt` all sit above 0.95 evenness, and they push every aesthetic word off
the top of the picture. `_words_across.SCAFFOLD` drops 126 such words:
the perception verbs, the hedges, and the meta-reasoning about the task.
`CITY_ECHO` drops 8 more at 1 word.

The set is a switch, not a rule. The notebook has a checkbox that puts the
scaffold back, and the scaffold band is a real fact about the traces.

## The commitment lift

A shared vocabulary is not yet a hierarchy. A hierarchy also RANKS, thus the
report carries a second statistic: does the model write the word when it picks
a winner, or when it cannot separate the 2 photographs?

The lift is the log-odds ratio with an informative Dirichlet prior, the same
statistic as the trace clouds. The target is the traces that answered
`MuchLess`, `Less`, `More`, or `MuchMore`. The background is the traces that
answered `Same` or `NotSure`.

**The lift is computed INSIDE each case, then averaged.** A pooled lift follows
the abstention rate instead: the street photography runs abstain on about 1% of
the pairs, and the schools runs on more than half.

**Warning: this is an association, not a cause.** The model narrates while it
decides, thus a word can follow the judgment rather than drive it. The
directional test lives in `_extractions.plot_win_block`, which knows the image
that each cue sits on.

## Warning: the rate divides by the whole kept mass

`_traces.counts_for_run` keeps the 20,000 commonest words of a run, and the
phrase cache keeps 60,000. A word run holds under 20,000 distinct words, thus
no word is lost. A phrase run holds far more, and the tail IS cut.

`run_counts` returns the whole kept mass beside the counts, and `rates_frame`
divides by that number. Without the rule the top 30,000 trigrams — under half
of the mass — would move every rate.

## Warning: the 2 raters do not write the same amount

qwen3.5-9b writes about 3 times the words of gemma-4-12b. A rate divides that
out, but a pooled group weights qwen more, because the pool sums counts and
then divides by the pooled corpus. Pick 1 rater in the notebook to test whether
a result belongs to 1 model.

## The gate

The notebook calls `_canonical.verify_or_raise()` before it reads a run, like
every other CVPR figure. It queries no network. See
`vlm-narratives-docs/canonical-run-registry.md`.
