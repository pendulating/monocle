"""Tests for the export of the depth reads to the Neuronpedia viewer."""

import numpy as np
import pandas as pd

from research.scripts.f10.hop_viewer import CONTRAST, IMAGE_MARK, RAW, group_masks, run, run_link, sequence, share, top_words  # noqa: E402

MANIFEST = {"prefix_tokens": ["<s>"], "prefix_ids": [1], "tail_tokens": ["</v>", "Is", " place"], "tail_ids": [3, 4, 6],
            "positions": ["after_image", "first_text", "place"]}


def test_sequence_puts_the_image_between_the_prefix_and_the_tail():
    tokens, at = sequence(MANIFEST)
    assert [t for t, _ in tokens] == ["<s>", IMAGE_MARK, "</v>", "Is", " place"]
    assert at == {"p00": 0, "after_image": 2, "first_text": 3, "place": 4}


def test_top_words_sorts_and_keeps_an_empty_row_for_nan():
    x = np.array([[0.0, 3.0, 1.0], [np.nan, 0.0, 0.0]])
    names, probs = top_words(x, np.array(["a", "b", "c"]), n=2)
    assert names == [["b", "c"], []]
    assert probs[0][0] > probs[0][1] and probs[1] == []


def test_run_puts_the_reads_at_the_right_tokens():
    tokens, at = sequence(MANIFEST)
    positions = list(at)
    raw = np.zeros((4, 2, 40), np.float32)
    raw[positions.index("place"), 1, 7] = 5
    words = np.array([f" w{i}" for i in range(40)])
    r = run("m", tokens, at, positions, [0, 4], raw, raw, words)
    assert r["layerRange"] == [0, 4] and r["meta"]["layers_by_type"] == {RAW: [0, 4], CONTRAST: [0, 4]}
    assert r["tokens"][at["place"]]["results"][0]["top_tokens"][1][0] == " w7"
    assert r["tokens"][1]["results"][0]["top_tokens"] == [[], []]


def test_group_masks_keep_the_thirds_apart():
    d = pd.DataFrame({"borough": ["Bronx", "Staten Island"] * 3, "yes_no": [1.0, 2, 3, 4, 5, 6]})
    m = group_masks(d)
    assert set(m) == {"all", "bronx", "staten-island", "high-yes-no", "low-yes-no"}
    assert not (m["high-yes-no"] & m["low-yes-no"]).any()
    assert m["high-yes-no"].sum() == m["low-yes-no"].sum() == 2


def test_the_link_of_a_run_opens_the_row_of_that_run():
    row = share("qwen3.5-9b", "W0E8UOWG", "text", 9, {"place": 1, "final": 2})
    assert run_link("qwen3.5-9b", "W0E8UOWG") == f"/qwen3.5-9b/jlens?shareId={row['id']}"


def test_the_rank_block_gives_the_core_ranks_and_skips_a_token_with_no_read():
    from research.monocle.instruments import ranks
    from research.scripts.f10.hop_viewer import rank_block

    tokens, at = sequence(MANIFEST)
    positions = list(at)
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(len(positions), 3, 40)).astype(np.float32)
    contrast = raw.copy()
    contrast[0] = np.nan  # the prefix token has no contrast
    words = np.array([f" w{i}" for i in range(40)])
    block = rank_block(at, positions, [0, 4, 8], raw, contrast, words)
    assert block["unit"] == "rank" and block["k"] == 10 and block["vocab"] == 40
    col = block["columns"][RAW]
    r = ranks(raw)
    place = positions.index("place")
    entry = col["byPosition"][str(at["place"])]
    for w, row in zip(entry["w"], entry["r"]):
        word = int(col["words"][w].strip()[1:])
        assert row == r[place, :, word].tolist()
        assert min(row) <= 10
    salient_any = {int(i) for i in np.flatnonzero((r[place] <= 10).any(0))}
    assert {int(col["words"][w].strip()[1:]) for w in entry["w"]} == salient_any
    layer = col["byLayer"]["4"]
    assert layer["positions"] == [at[p] for p in positions]
    first = layer["w"][0]
    assert [rk[0] for rk in layer["r"]][0] == r[0, 1, int(col["words"][first].strip()[1:])]
    con = block["columns"][CONTRAST]
    assert str(at["p00"]) not in con["byPosition"]
    assert at["p00"] not in con["byLayer"]["4"]["positions"]
