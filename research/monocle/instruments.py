"""Readout instruments on J-lens reads: salience, layer loss, position loss, smear (F10.35).

The input is a read of 1 run: the lens logits ``x[positions, layers, vocab]``. The
functions use the rank of each word, not the logit, because each layer has its own
lens and thus its own scale.

| Term | Definition |
|------|------------|
| Rank ``r[p, l, w]`` | The rank of word ``w`` at token ``p`` and layer ``l``. 1 = top. 0 = no read (a NaN row) |
| Salient | ``1 <= r <= k``. The default ``k`` is 10 |
| Loss | Salient at the start point and not salient at the end point |
| Gain | Not salient at the start point and salient at the end point |

| Function | Instrument | Output |
|----------|------------|--------|
| ``layer_loss(r, words, X, Y, p=p)`` | Layer loss, 1 token | word, start rank, end rank |
| ``layer_loss(r, words, X, Y)`` | Layer loss, all tokens | word, loss, gain |
| ``position_loss(r, words, A, B, layer=l)`` | Position loss, 1 layer | word, start rank, end rank |
| ``position_loss(r, words, A, B)`` | Position loss, all layers | word, loss, gain |
| ``salience_count(r, words, X, over="tokens")`` | Salience count at layer ``X`` | word, salient, not salient |
| ``salience_count(r, words, A, over="layers")`` | Salience count at token ``A`` | word, salient, not salient |
| ``smear(r, words, p)`` | Smear at token ``p`` | word, run, first, last |

``X``, ``Y`` and ``l`` are indices on the layer axis. ``p``, ``A`` and ``B`` are indices on
the token axis. To limit the layers (the slider of the viewer), give a slice of ``r``.

Rows with no read (rank 0) do not count. An "all" table lists only the words that are
salient at 1 or more of its points, thus it does not list each store word.

Ties: 2 equal logits take the order of the words in the store (a stable sort).

The top-k form. Salience needs only the top ``k`` words at each point. For many images,
the full ranks are too large (about 80 MB for each image of Gemma-4-31B). Thus 3
functions work on the word ids of the top ``k``:

| Function | Output |
|----------|--------|
| ``top_ids(x)`` | ``[..., k]`` word ids, the best first. A row with a NaN gives -1 |
| ``leaves(start, end)`` | ``[..., k]``: True where the start word is not in the end list. A gain from ``X`` to ``Y`` is a loss from ``Y`` to ``X`` |
| ``runs(ids)`` | For ``[layers, k]``: each word, its longest run, the first and the last index |
| ``steps(ids)`` | The losses and gains between neighbor layers or tokens: each change counts once |

The tests check that the 2 forms agree.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__version__ = "1.1.0"

K = 10
M = 3


def ranks(x: np.ndarray) -> np.ndarray:
    """Give the rank of each word over the last axis, 1 = the largest logit.

    A row with a NaN gets rank 0 for each word: it has no read.
    """
    x = np.asarray(x)
    order = np.argsort(-x, axis=-1, kind="stable")
    r = np.empty(order.shape, np.int32)
    steps = np.broadcast_to(np.arange(1, x.shape[-1] + 1, dtype=np.int32), order.shape)
    np.put_along_axis(r, order, steps, axis=-1)
    r[np.isnan(x).any(-1)] = 0
    return r


def salient(r: np.ndarray, k: int = K) -> np.ndarray:
    """Give True where a word is salient: 1 <= rank <= k."""
    return (r >= 1) & (r <= k)


def _valid(r: np.ndarray) -> np.ndarray:
    """Give True for each row with a read. ``r`` is ``[..., vocab]``."""
    return r[..., 0] > 0


def _one(start: np.ndarray, end: np.ndarray, words: np.ndarray, k: int) -> pd.DataFrame:
    """Give the words that lose salience from 1 point to 1 point. ``start``, ``end``: ``[vocab]``."""
    cols = ["word", "start_rank", "end_rank"]
    if start[0] == 0 or end[0] == 0:
        return pd.DataFrame(columns=cols)
    lost = np.flatnonzero(salient(start, k) & ~salient(end, k))
    out = pd.DataFrame({"word": words[lost], "start_rank": start[lost], "end_rank": end[lost]})
    return out.sort_values(["end_rank", "start_rank"], ascending=[False, True], kind="stable").reset_index(drop=True)


def _counts(start: np.ndarray, end: np.ndarray, words: np.ndarray, k: int) -> pd.DataFrame:
    """Give the loss and gain counts over N pairs of points. ``start``, ``end``: ``[N, vocab]``."""
    keep = _valid(start) & _valid(end)
    s, e = salient(start[keep], k), salient(end[keep], k)
    loss, gain = (s & ~e).sum(0), (~s & e).sum(0)
    seen = np.flatnonzero((s | e).any(0))
    out = pd.DataFrame({"word": words[seen], "loss": loss[seen], "gain": gain[seen]})
    return out.sort_values(["loss", "gain"], ascending=[False, True], kind="stable").reset_index(drop=True)


def layer_loss(r: np.ndarray, words: Sequence[str], X: int, Y: int, k: int = K, p: int | None = None) -> pd.DataFrame:
    """Give the layer loss from layer index ``X`` to ``Y``. ``r`` is ``[positions, layers, vocab]``.

    With ``p``: the words that lose salience at token ``p``, with the start and end rank,
    the largest end rank first. With no ``p``: the loss and gain counts over the tokens.
    """
    words = np.asarray(words)
    if p is not None:
        return _one(r[p, X], r[p, Y], words, k)
    return _counts(r[:, X], r[:, Y], words, k)


def position_loss(r: np.ndarray, words: Sequence[str], A: int, B: int, k: int = K, layer: int | None = None) -> pd.DataFrame:
    """Give the position loss from token index ``A`` to ``B``. ``r`` is ``[positions, layers, vocab]``.

    With ``layer``: the words that lose salience at that layer index. With no ``layer``:
    the loss and gain counts over the layers of ``r``.
    """
    words = np.asarray(words)
    if layer is not None:
        return _one(r[A, layer], r[B, layer], words, k)
    return _counts(r[A], r[B], words, k)


def salience_count(r: np.ndarray, words: Sequence[str], at: int, over: str = "tokens", k: int = K) -> pd.DataFrame:
    """Give, for each word, the count of points where it is salient and where it is not.

    ``over="tokens"``: ``at`` is a layer index, and the points are the tokens with a read.
    ``over="layers"``: ``at`` is a token index, and the points are the layers with a read.
    The table lists the words that are salient at 1 or more of these points.
    """
    if over not in ("tokens", "layers"):
        raise ValueError(f"over must be 'tokens' or 'layers', not {over!r}")
    words = np.asarray(words)
    rows = r[:, at] if over == "tokens" else r[at]
    rows = rows[_valid(rows)]
    s = salient(rows, k)
    n = s.sum(0)
    seen = np.flatnonzero(n > 0)
    out = pd.DataFrame({"word": words[seen], "salient": n[seen], "not_salient": len(rows) - n[seen]})
    return out.sort_values("salient", ascending=False, kind="stable").reset_index(drop=True)


def smear(r: np.ndarray, words: Sequence[str], p: int, k: int = K, m: int = M,
          layers: Sequence[int] | None = None) -> pd.DataFrame:
    """Give the words that stay salient for ``m`` or more contiguous layers at token ``p``.

    ``run`` is the longest run. ``first`` and ``last`` give its ends, as layer numbers
    from ``layers`` (or as indices when ``layers`` is None). A layer with no read
    breaks a run. The longest run comes first; for 2 equal runs, the first run counts.
    """
    words = np.asarray(words)
    s = salient(r[p], k)  # [layers, vocab]
    names = np.asarray(layers if layers is not None else range(s.shape[0]))
    cur = np.zeros(s.shape[1], np.int32)
    best = np.zeros_like(cur)
    last = np.full_like(cur, -1)
    for i in range(s.shape[0]):
        cur = np.where(s[i], cur + 1, 0)
        longer = cur > best
        best = np.where(longer, cur, best)
        last = np.where(longer, i, last)
    hit = np.flatnonzero(best >= m)
    out = pd.DataFrame({"word": words[hit], "run": best[hit],
                        "first": names[last[hit] - best[hit] + 1], "last": names[last[hit]]})
    return out.sort_values(["run", "first"], ascending=[False, True], kind="stable").reset_index(drop=True)


def top_ids(x: np.ndarray, k: int = K) -> np.ndarray:
    """Give the ids of the top ``k`` words over the last axis, the best first.

    The order is the order of ``ranks``. A row with a NaN gives -1 for each id. An
    exact tie at rank ``k`` can choose a different word than ``ranks``; float32 lens
    logits have almost no exact ties.
    """
    x = np.asarray(x)
    idx = np.argpartition(-x, k - 1, axis=-1)[..., :k]
    vals = np.take_along_axis(x, idx, -1)
    order = np.lexsort((idx, -vals), axis=-1)
    ids = np.take_along_axis(idx, order, -1).astype(np.int32)
    ids[np.isnan(x).any(-1)] = -1
    return ids


def leaves(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Give True where a word of the start list is not in the end list. Both are ``[..., k]``.

    A row with no read (-1) on either side gives False.
    """
    out = (start[..., :, None] != end[..., None, :]).all(-1)
    return out & (start[..., :1] >= 0) & (end[..., :1] >= 0)


def runs(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Give each word of ``ids[layers, k]``, its longest salient run, and the first and last index.

    A layer with no read breaks a run. For 2 equal runs, the first run counts.
    """
    words = np.unique(ids[ids >= 0])
    member = (ids[:, :, None] == words).any(1)  # [layers, words]
    cur = np.zeros(len(words), np.int32)
    best = np.zeros_like(cur)
    last = np.full_like(cur, -1)
    for i in range(member.shape[0]):
        cur = np.where(member[i], cur + 1, 0)
        longer = cur > best
        best = np.where(longer, cur, best)
        last = np.where(longer, i, last)
    return words, best, last - best + 1, last


def steps(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give the losses and gains between neighbor points of ``ids[..., n, k]`` (n layers or n tokens).

    ``loss[..., i, j]``: the word ``ids[..., i, j]`` is salient at point i and not at i + 1.
    ``gain[..., i, j]``: the word ``ids[..., i + 1, j]`` is salient at point i + 1 and not at i.
    Thus each change counts once, at the boundary where it occurs.
    """
    return leaves(ids[..., :-1, :], ids[..., 1:, :]), leaves(ids[..., 1:, :], ids[..., :-1, :])
