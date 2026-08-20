"""Word blocks over the IC ingredient spans.

`_traces` counts the words of a WHOLE trace and asks what a case talks about.
This module counts the words of a located SPAN and asks a narrower question:
what does each PART of the reasoning sound like?

    dimension       the cues the model names
    perspective     the readings it develops
    weighing        the text where it relates 2 things
    dismissal       the text where it sets 1 aside
    hedge           the text where it doubts
    reconsideration the text where it turns back
    verdict         the text where it decides

The 3 groupings
---------------
| `group_by` | 1 block for each | The question it answers |
|------------|------------------|-------------------------|
| `type` | ingredient type | What does a weighing sound like, against a dismissal? |
| `case` | case | Which words does this case put in its spans? |
| `case_type` | case, inside 1 type | Which cues does this case name? |

The weights come from the same `distinctive` score as the trace clouds: a word
is large when this block uses it and the others do not. Thus a block is read
against its siblings, and the words every block shares fall away.

**Only a located span counts.** A quote that no search finds in the trace is an
invention of the extractor, and its words are not the model's.

Warning: a span is short. A block therefore holds far fewer words than a trace
cloud, and `min_count` matters more: raise it when a block shows noise.

See `vlm-narratives-docs/ic-ingredient-extraction.md`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

import _ic
import _style as S
import _traces as T

__version__ = "1.0.0"

# The order the paper uses: the 3 that carry the codes first.
TYPE_ORDER = ("dimension", "perspective", "weighing", "dismissal",
              "reconsideration", "hedge", "verdict")

# The type whose words a per-case block draws. A dimension is the cue the model
# names, thus it is the part that differs most between cases.
DEFAULT_CASE_TYPE = "dimension"

GROUP_BY = ("type", "case", "case_type")


def _texts(df: pd.DataFrame) -> pd.Series:
    """The located spans, as text."""
    real = df[df["ingredient_type"].notna() & df["quote_found"].astype(bool)]
    return real["quote"].astype(str)


def counts_by_group(
    df: pd.DataFrame,
    stopwords: frozenset,
    group_by: str = "type",
    ngram: int = 1,
    ingredient_type: str = DEFAULT_CASE_TYPE,
) -> Tuple[Dict[str, Counter], pd.DataFrame]:
    """Count the words of each block.

    Returns:
        (counts, sizes). `sizes` says how many spans and how many words each
        block holds, which a reader needs before a picture.
    """
    if group_by not in GROUP_BY:
        raise ValueError(f"group_by must be one of {GROUP_BY}")
    real = df[df["ingredient_type"].notna() & df["quote_found"].astype(bool)].copy()
    if group_by == "type":
        real["block"] = real["ingredient_type"]
    elif group_by == "case":
        real["block"] = real["case"].map(_ic.CASE_LABEL).fillna(real["case"])
    else:
        real = real[real["ingredient_type"] == ingredient_type]
        real["block"] = real["case"].map(_ic.CASE_LABEL).fillna(real["case"])

    counts: Dict[str, Counter] = {}
    rows: List[Dict[str, object]] = []
    for block, part in real.groupby("block"):
        c = T.count_words(part["quote"].astype(str).tolist(), stopwords, ngram=ngram)
        counts[str(block)] = c
        rows.append({"block": str(block), "spans": int(len(part)),
                     "words": int(sum(c.values())), "distinct": int(len(c))})
    sizes = pd.DataFrame(rows).sort_values("spans", ascending=False).reset_index(drop=True)
    return counts, sizes


def ordered_blocks(counts: Dict[str, Counter], group_by: str = "type") -> List[str]:
    """The blocks in the order the paper uses."""
    if group_by == "type":
        known = [t for t in TYPE_ORDER if t in counts]
        return known + sorted(set(counts) - set(known))
    known = [_ic.CASE_LABEL.get(c, c) for c in _ic.CASE_ORDER]
    have = [b for b in known if b in counts]
    return have + sorted(set(counts) - set(have))


def block_weights(counts: Dict[str, Counter], block: str,
                  mode: str = "exclusive", max_words: int = 120,
                  min_count: int = 10,
                  blocks: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """The word-to-weight dict of 1 block.

    Modes:

    | mode | What it draws |
    |------|---------------|
    | `frequency` | the commonest words of the block |
    | `distinctive` | the words this block uses and the POOLED rest does not |
    | `exclusive` | as `distinctive`, but a word goes to its BEST block only |

    Warning: `distinctive` does NOT separate the blocks. Its background is every
    other block pooled, and `dimension` holds 1.16 million spans against 26,000
    for `weighing`. Thus any word that the small blocks share, and the huge one
    does not, scores high in ALL of them at once: measured on the corpus of
    2026-08-18, 65 of the 198 top-40 words sat in 2 blocks or more. "usually"
    was top in both `perspective` and `weighing`, and "library" in `dismissal`,
    `hedge`, and `reconsideration`.
    `exclusive` fixes that: it computes every block's score and gives each word
    to the single block that scores it highest, so 2 blocks never share a word
    and the picture answers "what is THIS part, and not that one".
    """
    if mode != "exclusive":
        return T.cloud_weights(counts, block, mode=mode, max_words=max_words,
                               min_count=min_count)
    names = list(blocks or counts)
    scored = {b: T.cloud_weights(counts, b, mode="distinctive",
                                 max_words=max_words * 4, min_count=min_count)
              for b in names}
    best: Dict[str, Tuple[str, float]] = {}
    for b, weights in scored.items():
        for word, value in weights.items():
            if word not in best or value > best[word][1]:
                best[word] = (b, value)
    mine = {w: v for w, (b, v) in best.items() if b == block}
    top = sorted(mine.items(), key=lambda kv: kv[1], reverse=True)[:max_words]
    return dict(top)


def word_table(counts: Dict[str, Counter], blocks: Sequence[str],
               mode: str = "exclusive", max_words: int = 120,
               min_count: int = 10) -> pd.DataFrame:
    """The numbers behind the blocks, so a word can be quoted with its weight.

    `also_in` names every OTHER block that ranks the same word in its own top
    list. A reader can then see at once whether a word belongs to this part of
    the reasoning or to several.
    """
    frames = []
    per_block = {b: T.cloud_weights(counts, b, mode="distinctive",
                                    max_words=max_words, min_count=min_count)
                 for b in blocks}
    for block in blocks:
        weights = block_weights(counts, block, mode=mode, max_words=max_words,
                                min_count=min_count, blocks=blocks)
        if not weights:
            continue
        also = [";".join(sorted(b for b in blocks
                                if b != block and w in per_block.get(b, {})))
                for w in weights]
        frames.append(pd.DataFrame({
            "block": block,
            "word": list(weights),
            "weight": [round(float(v), 5) for v in weights.values()],
            "count": [int(counts[block].get(w, 0)) for w in weights],
            "rank": range(1, len(weights) + 1),
            "also_in": also,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_block(counts: Dict[str, Counter], block: str, mode: str = "exclusive",
               max_words: int = 120, min_count: int = 10,
               blocks: Optional[Sequence[str]] = None):
    """Draw 1 block. The palette and the seed come from `_style`, as elsewhere."""
    import matplotlib.pyplot as plt

    weights = block_weights(counts, block, mode=mode, max_words=max_words,
                            min_count=min_count, blocks=blocks)
    cloud = T.make_cloud(weights)
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    return fig


def export(df: pd.DataFrame, out_dir: Path, mode: str = "exclusive",
           max_words: int = 120, min_count: int = 10, ngram: int = 1,
           groupings: Sequence[str] = ("type", "case_type"),
           ingredient_type: str = DEFAULT_CASE_TYPE) -> List[str]:
    """Write 1 PNG for each block, plus its word table and the block sizes.

    The default writes 2 sets: 1 block for each ingredient type over the whole
    battery, and 1 block for each case inside the dimension type.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stopwords = T.default_stopwords()
    written: List[str] = []

    for group_by in groupings:
        counts, sizes = counts_by_group(df, stopwords, group_by=group_by,
                                        ngram=ngram, ingredient_type=ingredient_type)
        if not counts:
            continue
        tag = group_by if group_by != "case_type" else f"{ingredient_type}_by_case"
        blocks = ordered_blocks(counts, group_by)

        sizes.to_csv(out_dir / f"ic_words_{tag}_sizes.csv", index=False)
        written.append(f"ic_words_{tag}_sizes.csv")
        table = word_table(counts, blocks, mode=mode, max_words=max_words,
                           min_count=min_count)
        if not table.empty:
            table.to_csv(out_dir / f"ic_words_{tag}.csv", index=False)
            written.append(f"ic_words_{tag}.csv")

        for block in blocks:
            try:
                fig = plot_block(counts, block, mode=mode, max_words=max_words,
                                 min_count=min_count, blocks=blocks)
            except ValueError:
                # An empty block is a real answer: no word passed `min_count`.
                continue
            name = f"ic_words_{tag}_{T.safe_name(block)}.png"
            fig.savefig(out_dir / name, dpi=200)
            plt.close(fig)
            written.append(name)
    return written
