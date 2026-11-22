"""CPU tests for the answer-position attention routing."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import attn_route as ar  # noqa: E402


class TestGroups:
    def blocks(self):
        return [torch.arange(8, 264), torch.arange(269, 525)]

    def test_names_the_aggregator_patch_of_each_block(self):
        g = ar.group_positions(self.blocks(), 706)
        assert g["A_last"] == [263] and g["B_last"] == [524]
        assert g["A_first"] == [8] and g["B_first"] == [269]

    def test_rest_excludes_both_reported_endpoints(self):
        """A_first must not also appear in A_rest, or the summary double counts."""
        g = ar.group_positions(self.blocks(), 706)
        assert len(g["A_rest"]) == 254
        assert 263 not in g["A_rest"] and 8 not in g["A_rest"]

    def test_text_is_everything_outside_both_blocks(self):
        g = ar.group_positions(self.blocks(), 706)
        assert len(g["text"]) == 706 - 512
        assert 264 in g["text"] and 263 not in g["text"]

    def test_groups_cover_the_sequence_once(self):
        g = ar.group_positions(self.blocks(), 706)
        cover = (g["A_last"] + g["B_last"] + g["A_first"] + g["B_first"]
                 + g["A_rest"] + g["B_rest"] + g["text"])
        assert sorted(cover) == list(range(706))


class TestRows:
    def test_mass_sums_per_head_and_group(self):
        att = {47: torch.tensor([[0.5, 0.25, 0.25], [1.0, 0.0, 0.0]])}
        groups = {"x": [0], "y": [1, 2]}
        rows = ar.rows_for(att, groups, {"case": "t", "pair_id": "p"})
        got = {(r["head"], r["group"]): r["mass"] for r in rows}
        assert got[(0, "x")] == pytest.approx(0.5)
        assert got[(0, "y")] == pytest.approx(0.5)
        assert got[(1, "y")] == pytest.approx(0.0)

    def test_skips_an_empty_group(self):
        att = {47: torch.ones(2, 3)}
        rows = ar.rows_for(att, {"empty": [], "x": [0]}, {})
        assert all(r["group"] == "x" for r in rows)


class TestSummarize:
    def frame(self):
        rows = []
        for pair in ("p0", "p1"):
            for h in range(4):
                for grp, m in (("A_last", 0.02), ("A_rest", 0.10)):
                    rows.append({"case": "t", "pair_id": pair,
                                 "query": "answer", "block": 47,
                                 "head": h, "group": grp, "n_pos": 1,
                                 "mass": m})
        return pd.DataFrame(rows)

    def test_returns_a_per_head_average_not_a_head_sum(self):
        """Regression: `df.head` is the DataFrame METHOD, not the column."""
        s = ar.summarize(self.frame()).xs("answer", level="query")
        assert s.loc[47, "A_last"] == pytest.approx(0.02)
        assert s.loc[47, "A_rest"] == pytest.approx(0.10)

    def test_indexes_by_block(self):
        s = ar.summarize(self.frame())
        assert list(s.xs("answer", level="query").index) == [47]

    def test_keeps_each_query_apart(self):
        """A_last and the answer must not average together."""
        f = self.frame()
        other = f.copy()
        other["query"] = "A_last"
        other["mass"] = 0.50
        s = ar.summarize(pd.concat([f, other], ignore_index=True))
        assert s.xs("answer", level="query").loc[47, "A_last"] == pytest.approx(0.02)
        assert s.xs("A_last", level="query").loc[47, "A_last"] == pytest.approx(0.50)


class TestDelimiters:
    """`boi` and `eoi` wrap each block and are NOT vision tokens.

    They attend causally over the island behind them, so they have the shape
    of a pooling token. An earlier version hid them inside `text`.
    """

    def ids(self, seq_len=706):
        t = torch.zeros(seq_len, dtype=torch.long)
        t[7] = ar.BOI_TOKEN_ID
        t[264] = ar.EOI_TOKEN_ID
        t[268] = ar.BOI_TOKEN_ID
        t[525] = ar.EOI_TOKEN_ID
        return t

    def blocks(self):
        return [torch.arange(8, 264), torch.arange(269, 525)]

    def test_delimiters_leave_the_text_group(self):
        g = ar.group_positions(self.blocks(), 706, self.ids())
        assert g["boi"] == [7, 268]
        assert g["eoi"] == [264, 525]
        for pos in g["boi"] + g["eoi"]:
            assert pos not in g["text"]

    def test_still_partitions_the_sequence(self):
        g = ar.group_positions(self.blocks(), 706, self.ids())
        ar.check_partition(g, 706)

    def test_partition_check_catches_a_gap(self):
        g = ar.group_positions(self.blocks(), 706, self.ids())
        g["text"] = g["text"][:-1]
        with pytest.raises(RuntimeError):
            ar.check_partition(g, 706)

    def test_partition_check_catches_an_overlap(self):
        g = ar.group_positions(self.blocks(), 706, self.ids())
        g["text"] = g["text"] + [g["boi"][0]]
        with pytest.raises(RuntimeError):
            ar.check_partition(g, 706)

    def test_without_ids_the_delimiters_stay_in_text(self):
        """Back-compatible: no input_ids means no delimiter groups."""
        g = ar.group_positions(self.blocks(), 706)
        assert "boi" not in g
        assert 7 in g["text"]


class TestCli:
    def test_defaults_cover_every_global_block(self):
        a = ar.build_parser().parse_args([])
        assert set(ar.GLOBAL_BLOCKS) <= set(a.blocks)
        assert 47 in a.blocks

    def test_rejects_an_unregistered_case(self):
        with pytest.raises(SystemExit):
            ar.build_parser().parse_args(["--cases", "nope"])


class TestSummaryJson:
    """Regression: the (query, block) index cannot be a JSON key."""

    def frame(self):
        rows = []
        for q in ("answer", "A_last"):
            for h in range(2):
                for grp in ("A_last", "text"):
                    rows.append({"case": "t", "pair_id": "p0", "query": q,
                                 "block": 47, "head": h, "group": grp,
                                 "n_pos": 1, "mass": 0.25})
        return ar.summarize(pd.DataFrame(rows))

    def test_nests_query_then_block(self):
        j = ar.summary_json(self.frame())
        assert set(j) == {"answer", "A_last"}
        assert j["answer"]["47"]["A_last"] == pytest.approx(0.25)

    def test_result_is_json_serializable(self):
        import json
        json.dumps(ar.summary_json(self.frame()))


class TestControlQuery:
    def test_run_case_records_an_ordinary_patch_as_control(self):
        """A_last's behaviour means nothing without an ordinary patch."""
        src = (ar.__file__ or "")
        text = open(src).read()
        assert '"A_mid"' in text and '"B_mid"' in text
