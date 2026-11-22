# The layer at which the judgment becomes decodable, at 1-layer resolution

Date: 2026-09-01. Review item P2 (`jacobian-lens-review.md`). Case
subway_safety, run kind proxy, 300 pairs, the 47-layer mm lens
(`outputs/_monocle/jlens/mm_all47/gemma4_12b_lens.pt`), conditions prod,
neutral, and axis. Output: `outputs/_monocle/safety_workspace_all47/` (the
last-token readout over the late layers).

## The main result

**The judgment does not become decodable at L42. It becomes decodable at
L40-41 and strengthens through L45.** The old 8-layer grid sampled a ramp at
stride 6 and reported its sample point. The dense readout shows:

| Layer | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 |
|---|---|---|---|---|---|---|---|---|---|---|
| p(prod label), prod | .0002 | .0001 | .0000 | .005 | .013 | .017 | .024 | **.069** | .014 | .765 |
| 4-way agreement, prod | .33 | .34 | .47 | .61 | .67 | .66 | .63 | **.70** | .67 | .89 |
| layer type | sliding | sliding | sliding | **global** | post-global | sliding | sliding | sliding | sliding | global |

Three facts replace the old statement:

1. **The judgment first becomes decodable at L41, a global attention layer.**
   p(prod label) rises about 300-fold from L40 to L41, and the agreement takes
   its 2 largest steps at L40 and L41.
2. **The strongest translated readout sits at L45, a sliding-window attention
   layer.** Thus the ramp is not a post-global artifact (review flaw F4):
   decodability does not track the layer type inside the ramp. The confound
   of the grid with the layer type no longer holds.
3. **L46 is a local dip** (0.014 against 0.069 at L45), which the old grid
   already reported as "the degenerate L46".

## The neutral condition carries a constant label

Paired statistics (`research/scripts/answer_depth_stats.py`, 300 pairs, 20000-resample
bootstrap, BH correction over layers): the prod-minus-neutral difference
remains significant after BH correction at every layer from L38 up. At L45 it
is +0.051 [+0.045, +0.058] in p and +0.42 [+0.36, +0.49] in agreement.

The neutral condition is not at the minimum (p about 0.018 at L45), but its
argmax is `Less` for 75 to 94 percent of the pairs at every ramp layer, while
the prod argmax tracks the pair. Thus the correct statement is: **under the
neutral prompt the last-token readout carries a near-constant label, not the
pair's judgment.** The statement "never under neutral" was too strong.

## Two more facts at 1-layer resolution

- **The safety vocabulary mass is 0.000 at all 47 layers.** The readout
  carries the label token and never the safety words, now with no gap between
  sample layers.
- Under prod, the mid-ramp argmax is most often `Same` (140-180 of 300 at
  L42-L45) and the final layer shifts to `More`. The translated readout is
  more conservative than the model's own answer.

## Limitations

- The registered subway run predates the 2026-08-20 layout decision
  (`prompt_cfg` warns). Both conditions read the same recorded prompt, so the
  prod-neutral difference is internally consistent, but a re-registered run
  must repeat this readout.
- One case, one lens. The 47-layer lens exists only for the mm estimand.

## How to repeat

```bash
sbatch --time=03:00:00 research/monocle/monocle.sub monocle.safety_workspace \
    --case subway_safety --kind proxy --n-pairs 300 --n-map-pairs 2 \
    --lenses outputs/_monocle/jlens/mm_all47/gemma4_12b_lens.pt \
    --layers $(seq -s' ' 0 46) --map-layers 30 36 42 \
    --out-dir outputs/_monocle/safety_workspace_all47

python research/scripts/answer_depth_stats.py \
    --parquet outputs/_monocle/safety_workspace_all47/answer_depth.parquet \
    --out outputs/_monocle/safety_workspace_all47/answer_depth_stats.json
```
