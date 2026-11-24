"""Tests for the readout instruments (F10.35): salience, layer loss, position loss, smear."""

import numpy as np
import pytest

from research.monocle.instruments import leaves, layer_loss, position_loss, ranks, runs, salience_count, salient, smear, steps, top_ids

V = 600
WORDS = np.array([f"w{i}" for i in range(V)])


def base(positions: int = 3, layers: int = 4) -> np.ndarray:
    """Give logits where word i has rank i + 1 at each point."""
    return np.broadcast_to(-np.arange(V, dtype=np.float32), (positions, layers, V)).copy()


def put(x: np.ndarray, p: int, l: int, word: int, rank: int) -> None:
    """Move 1 word to a rank at 1 point. The other words keep their order."""
    order = [w for w in np.argsort(-x[p, l], kind="stable") if w != word]
    order.insert(rank - 1, word)
    x[p, l, order] = -np.arange(V, dtype=np.float32)


def test_ranks_give_1_for_the_top_and_0_for_a_nan_row():
    x = np.array([[0.5, 2.0, 1.0], [np.nan, np.nan, np.nan]])
    assert ranks(x).tolist() == [[3, 1, 2], [0, 0, 0]]


def test_ranks_break_ties_by_the_order_of_the_words():
    assert ranks(np.array([1.0, 1.0, 2.0])).tolist() == [2, 3, 1]


def test_salient_is_the_top_k_and_not_a_missing_read():
    assert salient(np.array([0, 1, 10, 11])).tolist() == [False, True, True, False]


def test_layer_loss_one_token_finds_a_drop_from_rank_1_to_500():
    x = base()
    put(x, 1, 2, 0, 500)
    r = ranks(x)
    got = layer_loss(r, WORDS, X=1, Y=2, p=1)
    assert got.iloc[0].tolist() == ["w0", 1, 500]


def test_a_word_that_stays_in_the_top_10_is_not_a_loss():
    x = base()
    put(x, 0, 1, 2, 9)  # rank 3 -> rank 9
    got = layer_loss(ranks(x), WORDS, X=0, Y=1, p=0)
    assert "w2" not in set(got["word"])


def test_a_word_not_salient_at_the_start_is_not_a_loss():
    x = base()
    put(x, 0, 1, 50, 400)
    got = layer_loss(ranks(x), WORDS, X=0, Y=1, p=0)
    assert "w50" not in set(got["word"])


def test_layer_loss_all_tokens_counts_loss_and_gain():
    x = base(positions=8)
    for p in range(5):
        put(x, p, 1, 0, 300)  # w0 loses salience at 5 tokens
    for p in (5, 6):
        put(x, p, 1, 20, 1)  # w20 gains salience at 2 tokens
    got = layer_loss(ranks(x), WORDS, X=0, Y=1).set_index("word")
    assert got.loc["w0"].tolist() == [5, 0]
    assert got.loc["w20"].tolist() == [0, 2]
    assert got.loc["w1"].tolist() == [0, 0]
    assert got.index[0] == "w0"


def test_salience_count_over_tokens_gives_salient_and_not_salient():
    x = base(positions=20)
    for p in range(6, 20):
        put(x, p, 2, 3, 100)  # w3 is salient at layer 2 at 6 of 20 tokens
    got = salience_count(ranks(x), WORDS, 2, over="tokens").set_index("word")
    assert got.loc["w3"].tolist() == [6, 14]
    assert "w50" not in got.index  # never salient: not in the list


def test_salience_count_over_layers_ignores_the_end_point():
    x = base(positions=2, layers=5)
    put(x, 0, 4, 0, 99)
    got = salience_count(ranks(x), WORDS, 0, over="layers").set_index("word")
    assert got.loc["w0"].tolist() == [4, 1]


def test_position_loss_one_layer_and_all_layers():
    x = base(positions=3, layers=4)
    for layer in (0, 1, 3):
        put(x, 2, layer, 4, 1000 % V)
    r = ranks(x)
    one = position_loss(r, WORDS, A=0, B=2, layer=1)
    assert one.iloc[0]["word"] == "w4"
    every = position_loss(r, WORDS, A=0, B=2).set_index("word")
    assert every.loc["w4"].tolist() == [3, 0]


def test_one_layer_and_all_layers_agree_on_1_read_layer():
    x = base(positions=3, layers=1)
    put(x, 1, 0, 0, 40)
    r = ranks(x)
    one = set(position_loss(r, WORDS, A=0, B=1, layer=0)["word"])
    every = position_loss(r, WORDS, A=0, B=1)
    assert one == set(every.loc[every["loss"] > 0, "word"])


def test_smear_takes_the_longest_contiguous_run():
    x = base(positions=1, layers=7)
    put(x, 0, 4, 0, 200)  # w0: salient at layers 0-3, then a gap, then 5-6
    got = smear(ranks(x), WORDS, 0, layers=[0, 4, 8, 12, 16, 20, 24]).set_index("word")
    assert got.loc["w0"].tolist() == [4, 0, 12]
    assert got.loc["w1"].tolist() == [7, 0, 24]


def test_smear_drops_runs_shorter_than_m():
    x = base(positions=1, layers=4)
    for layer in (0, 2):
        put(x, 0, layer, 5, 300)
    got = smear(ranks(x), WORDS, 0, m=3)
    assert "w5" not in set(got["word"])


def test_a_row_with_no_read_gives_no_output_and_no_error():
    x = base(positions=3)
    x[0] = np.nan
    r = ranks(x)
    assert layer_loss(r, WORDS, X=0, Y=1, p=0).empty
    assert position_loss(r, WORDS, A=0, B=1, layer=0).empty
    assert smear(r, WORDS, 0).empty
    every = layer_loss(r, WORDS, X=0, Y=1)
    assert (every["loss"] == 0).all()
    count = salience_count(r, WORDS, 0, over="tokens").set_index("word")
    assert count.loc["w0"].tolist() == [2, 0]


def test_salience_count_refuses_a_wrong_axis():
    with pytest.raises(ValueError):
        salience_count(ranks(base()), WORDS, 0, over="images")


def test_the_top_k_form_agrees_with_the_rank_form():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 6, 80)).astype(np.float32)
    x[1, 2] = np.nan
    r, ids = ranks(x), top_ids(x)
    for p in range(4):
        for layer in range(6):
            got = ids[p, layer]
            if r[p, layer, 0] == 0:
                assert (got == -1).all()
                continue
            assert r[p, layer, got].tolist() == list(range(1, 11))
    for p in range(4):
        one = layer_loss(r, np.arange(80).astype(str), X=1, Y=4, p=p)
        lost = ids[p, 1][leaves(ids[p, 1], ids[p, 4])]
        assert sorted(one["word"].astype(int)) == sorted(lost.tolist())
        words, run, first, last = runs(ids[p])
        rank_form = smear(r, np.arange(80), p, m=1).set_index("word")
        keep = run >= 1
        assert rank_form.loc[words[keep], "run"].tolist() == run[keep].tolist()
        assert rank_form.loc[words[keep], "first"].tolist() == first[keep].tolist()


def test_leaves_gives_a_gain_as_the_reverse_loss():
    a, b = np.array([1, 2, 3]), np.array([3, 4, 5])
    assert a[leaves(a, b)].tolist() == [1, 2]
    assert b[leaves(b, a)].tolist() == [4, 5]
    assert not leaves(a, np.array([-1, -1, -1])).any()


def test_steps_count_each_change_once_at_its_boundary():
    # 4 layers. w7 is salient at layers 0-1, leaves at 1 -> 2, and comes back at 2 -> 3.
    ids = np.array([[7, 1], [7, 1], [2, 1], [7, 1]])
    loss, gain = steps(ids)
    assert np.argwhere(loss).tolist() == [[1, 0], [2, 0]]  # w7 leaves at 1 -> 2, w2 leaves at 2 -> 3
    assert ids[1:][gain].tolist() == [2, 7]  # w2 enters at 1 -> 2, w7 enters at 2 -> 3
    assert np.argwhere(gain)[:, 0].tolist() == [1, 2]
