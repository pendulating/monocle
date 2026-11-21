"""Tests for the causal-patch word reader."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from monocle import causal_words as cw


class FakeTok:
    def decode(self, ids):
        return f"w{int(ids[0])}"


def test_the_poles_are_the_extremes():
    t = np.array([-5.0, 0.0, 1.0, 2.0, 9.0])
    g = cw.pick_patches(t, n_pole=1, n_rand=2, rng=np.random.default_rng(0))
    assert g["pos"] == [4]
    assert g["neg"] == [0]


def test_the_poles_keep_their_order():
    t = np.array([-5.0, -4.0, 1.0, 2.0, 9.0])
    g = cw.pick_patches(t, n_pole=2, n_rand=1, rng=np.random.default_rng(0))
    assert g["pos"] == [4, 3]
    assert g["neg"] == [0, 1]


def test_a_random_patch_never_carries_a_second_role():
    t = np.arange(20.0)
    for seed in range(8):
        g = cw.pick_patches(t, n_pole=3, n_rand=6,
                            rng=np.random.default_rng(seed))
        assert not (set(g["random"]) & (set(g["pos"]) | set(g["neg"])))


def test_the_random_draw_shrinks_when_the_pool_is_small():
    t = np.arange(8.0)
    g = cw.pick_patches(t, n_pole=3, n_rand=99,
                        rng=np.random.default_rng(1))
    assert len(g["random"]) == 2


def test_words_at_reads_the_top_k_in_order():
    logits = torch.tensor([[0.0, 5.0, 1.0], [7.0, 0.0, 2.0]])
    got = cw.words_at(logits, [0, 1], FakeTok(), top_k=2)
    assert got[0] == (0, 0, "w1", 5.0)
    assert got[1] == (0, 1, "w2", 1.0)
    assert got[2] == (1, 0, "w0", 7.0)


def test_words_at_on_an_empty_patch_list():
    logits = torch.zeros((3, 4))
    assert cw.words_at(logits, [], FakeTok(), top_k=2) == []


def test_words_at_keeps_the_asked_patch_index():
    logits = torch.eye(5) * 3.0
    got = cw.words_at(logits, [3], FakeTok(), top_k=1)
    assert got[0][0] == 3
    assert got[0][2] == "w3"


def test_a_flat_map_still_gives_2_distinct_poles():
    t = np.zeros(6)
    g = cw.pick_patches(t, n_pole=1, n_rand=1, rng=np.random.default_rng(2))
    assert g["pos"] != g["neg"]
