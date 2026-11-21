# F10.34: sums and differences of the prompt directions (D190)

**Status (2026-11-03):** done. Script `scripts/f10/hop_sum_directions.py` 1.1.0, tables and the figure in
`figures/f10/sum/`. The figure is in the paper as `fig:pair-cosines` (Appendix B, `app:pair-cosines`), with
1 sentence in section 4.2.

## The question

The user asked what the word readout gives when we add the `urban` direction to the `nice`, `cool` and
`ordinary` directions element-wise, and then asked for all pair sums and all pair differences of the 7
adjectives, with 2 heatmaps for the appendix.

## The objects

| Item | Rule |
|------|------|
| Prompt direction | At the noun token `place`, for 1 model, 1 prompt and 1 saved layer: the unit vector over the lens words of the r, over the 197 NTAs, of the word's NTA mean readout with the prompt's NTA answer margin. This is `g[p]` of `hop_shared.py` (variant `main`; at `place` there is no removal) |
| Shared direction (F10.27) | The mean of the `cool`, `nice`, `fancy` and `pretty` directions |
| Sum, difference | `unit(g[a] + g[b])`, `unit(g[a] - g[b])` at each layer, for the 21 pairs of `cool`, `nice`, `pretty`, `fancy`, `ugly`, `ordinary`, `urban` (a before b in this order) |
| Cosine | Of the sum or difference with each single direction, at each layer; the tables and the figure give the mean over the saved layers |
| Words | The F10.28 rule: z-score the direction at each layer, mean over the layers, top words at each end |
| Images | The 591 images of F10.26 (`data/f10/prompts/`); `--k 10` runs the same on the 10-image summaries |

## Results

**The single directions** (layer mean of the cosine, Gemma-4-31B, Qwen3.5-9B, Qwen3.6-27B):

| Pair | Cosine |
|------|--------|
| `urban`, `nice` | -.51, -.46, -.64 |
| `urban`, `ordinary` | -.49, -.65, -.65 |
| `urban`, `cool` | .15, .07, .65 |
| `urban`, shared direction | -.17, -.21, -.15 |
| `nice`, `cool` | .66, .59, .09 |

`urban` is a density axis, not a valence axis: its top words are `downtown`, `nightlife`, `cafes`,
`theaters`, `commerce`, and its bottom words are `farms`, `pasture`, `forestry`, `plantation`.

**Sums and differences that land on a third adjective** (`pairs.csv`; the figure gives all cells):

| Combination | Nearest single direction | Cosine |
|-------------|--------------------------|--------|
| `nice` + `urban` | `cool` | .83, .64, .87 |
| `pretty` + `urban` | `cool` (Gemma, Qwen3.6-27B) or `fancy` (Qwen3.5-9B) | .87, .81, .83 |
| `fancy` + `urban` | `cool` | .51, .65, .93 |
| `ugly` + `urban` | minus `pretty` | -.75, -.88, -.93 |
| `cool` - `nice` | `urban` | .82, .59, .96 |
| `cool` - `pretty` | `urban` | .87, .77, .96 |
| `nice` - `urban` | `pretty` | .90, .90, .94 |
| `nice` - `urban` | `ordinary` | .46, .73, .67 |
| `nice` - `fancy` | `ordinary` | .85, .87, .84 |
| `pretty` - `fancy` | minus `urban` | -.81, -.82, -.85 |

A least-squares fit of `cool` on `nice` and `urban` (`fit_cool.csv`) gives weights 1.02 and .70 (Gemma),
.79 and .43 (Qwen3.5-9B), .88 and 1.21 (Qwen3.6-27B), and the fit has cosine .90, .72, .94 with `cool`.
`cool` is near `nice` + `urban` at every saved layer (.77 to .88 in Gemma; .61 to .71 in Qwen3.5-9B; .64
at L0 and .76 to .94 from L4 in Qwen3.6-27B).

**The words change with the sum** (`words.csv`, `pair_words.csv`, top of the list):

| Direction | Up | Down |
|-----------|----|------|
| `nice` (Gemma) | flower, floral, hydrangea, gardens, orchid | contaminate, smelting, contempt, contaminants |
| `nice` + `urban` (Gemma) | celebrate, heritage, birthday, medallion, admiration, bouquet | pointless, contaminate, wastes, landfill, swamp, deteriorating |
| `nice` + `urban` (Qwen3.5-9B) | renowned, stylish, celebrities, famed, prestigious, iconic | abuses, waste, contamination, asbestos, pesticides, murderers |
| `cool` + `urban` (Qwen3.5-9B) | renowned, iconic, marble, famous, pedestal, dazzling, jazz | farms, farmer, agriculture, barn, ranch, motel |
| `ordinary` + `urban` (Gemma) | grocery, mercado, wholesale, gasolina, barato | telescopes, heavenly, waterfalls, inspirational, scientists |
| `cool` - `urban` (Gemma) | honeymoon, hydrangea, gardens, beautiful, romantic | industrial, contaminate, warehouses, refinery |
| `nice` - `urban` (Qwen3.6-27B) | agricultural, serene, mindful, harmony, peacefully | prostitutes, dumpsters, nightclub, graffiti, dubious |
| all 4 (`nice` + `cool` + `ordinary` + `urban`) | about `nice` + `cool` (cosine .64, .78, .60 with the shared direction); the top of Qwen3.6-27B is code tokens | |

Reading: with `urban` added, the "nice" words move from gardens and flowers to celebration, heritage and
renown, and the "not nice" words move from industry and labor to waste, contamination and landfill.
With `urban` removed, the top of `cool` is rural and romantic, and its bottom is the gritty city. The
2 axes that describe much of the 7 directions are valence (`nice`, `pretty` against `ugly`) and density
(`urban` against the rural and the ordinary); `cool` and `fancy` sit between them.

**Limits.** The directions come from the 591 images (3 for each NTA; F10.30 gives the 10-image summaries
for a check with `--k 10`). A cosine between 2 directions in a space of 43,563 to 53,999 words is a
statement about the word readout at the noun token, not about the residual. The layer mean hides the
band structure; `pairs.csv` keeps the layers.

## Outputs

| File | Content |
|------|---------|
| `figures/f10/sum/pairs.csv` | cosine of each pair sum and difference with each single direction, by model and layer |
| `figures/f10/sum/pair_words.csv` | 10 words at each end of each sum and difference |
| `figures/f10/sum/pair_cosines.{pdf,png,json}` | the 2 heatmaps (`fig:pair-cosines`) |
| `figures/f10/sum/words.csv`, `overlap.csv`, `cosine.csv` | the 4 named sums against the single directions and the shared direction |
| `figures/f10/sum/fit_cool.csv` | the least-squares fit of `cool` and `ordinary` on `nice` and `urban`, by layer |

## Related

- [f10-shared-direction.md](f10-shared-direction.md): the F10.27 fit that defines the prompt directions
- [f10-30-more-images.md](f10-30-more-images.md): the 10-image summaries for a check
