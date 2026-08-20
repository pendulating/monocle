"""Tests for the Integrative Complexity ingredient schema.

These tests need no GPU and no model. They cover the 3 places where this path
can lose data without a sign:

- a JSON answer that the token cap cut,
- a quote that no search finds,
- a sub-quote list that holds the word "none" in place of an empty list.
"""

from __future__ import annotations

import json

import pytest

from dagspaces.common import ic_schema as S


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------


class TestJsonSchema:
    def test_every_array_is_present(self):
        props = S.json_schema()["properties"]
        assert set(props) == set(S.ARRAY_KEYS.values())

    def test_the_schema_order_follows_the_ingredient_order(self):
        """Guided decoding fills the arrays in this order, and the cap eats the
        last one. v1 declared `weighings` last and lost it in 93% of the cut
        traces."""
        props = list(S.json_schema()["properties"])
        assert props == [S.ARRAY_KEYS[k] for k in S.INGREDIENT_TYPES]

    def test_the_rare_classes_come_before_the_frequent_ones(self):
        order = list(S.INGREDIENT_TYPES)
        assert order.index("verdict") < order.index("hedge")
        assert order.index("weighing") < order.index("reconsideration")

    def test_only_dimensions_and_verdicts_have_a_lower_bound(self):
        props = S.json_schema()["properties"]
        bounded = {k for k, v in props.items() if v.get("minItems")}
        assert bounded == set(S.BOUNDED_ARRAYS)

    def test_a_closed_field_is_a_real_enum(self):
        """The whole reason to write the schema by hand."""
        props = S.json_schema()["properties"]
        verdict = props["verdicts"]["items"]["properties"]
        assert verdict["label"]["enum"] == list(S.VERDICT_LABELS)
        dimension = props["dimensions"]["items"]["properties"]
        assert dimension["type"]["enum"] == list(S.DIMENSION_TYPES)
        assert dimension["valence"]["enum"] == list(S.VALENCES)

    def test_a_name_stays_open(self):
        """The codebook grows out of the names, thus they must not be an enum."""
        props = S.json_schema()["properties"]
        for key in ("dimensions", "perspectives"):
            name = props[key]["items"]["properties"]["name"]
            assert name == {"type": "string"}

    def test_a_sub_quote_field_is_a_list_and_never_nullable(self):
        props = S.json_schema()["properties"]
        hedge = props["hedges"]["items"]["properties"]
        assert hedge["justification_quotes"]["type"] == "array"
        assert "null" not in json.dumps(hedge["justification_quotes"])

    def test_every_property_is_required(self):
        props = S.json_schema()["properties"]
        for key, spec in props.items():
            item = spec["items"]
            assert set(item["required"]) == set(item["properties"]), key


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_a_brace_in_the_trace_survives(self):
        """A trace holds JSON. `str.format` would read its braces as fields."""
        trace = 'I answer {"label": "Same"} because {x} looks equal.'
        out = S.build_prompt(trace)
        assert trace in out

    def test_the_placeholder_is_gone(self):
        out = S.build_prompt("hello")
        assert "{trace}" not in out
        assert "{examples}" not in out

    def test_an_empty_trace_does_not_raise(self):
        assert "<trace>" in S.build_prompt("")

    def test_a_placeholder_inside_the_trace_is_not_filled(self):
        """The trace is data. One pass stops it from acting as a template."""
        trace = "The model wrote {examples} and then {trace} in its notes."
        out = S.build_prompt(trace, examples=[])
        assert trace in out

    def test_zero_shot_holds_no_example_block(self):
        out = S.build_prompt("hello")
        assert "### Example 1" not in out
        assert "worked examples" not in out


class TestExamples:
    @staticmethod
    def _examples():
        try:
            return S.load_examples()
        except FileNotFoundError:
            pytest.skip("run scripts/build_ic_examples.py first")

    def test_every_span_of_every_example_is_exact(self):
        """An example span that no search finds teaches a shape we cannot use."""
        for example in self._examples():
            rows = S.ingredient_rows(example["trace"], example["answer"])
            for row in rows:
                assert row["quote_method"] == "exact", (
                    example["pair_id"], row["ingredient_type"], row["quote"]
                )
                assert row["n_sub_quotes"] == row["n_sub_quotes_found"]

    def test_the_examples_cover_every_ingredient_type(self):
        examples = self._examples()
        seen = set()
        for example in examples:
            for kind in S.INGREDIENT_TYPES:
                if example["answer"].get(S.ARRAY_KEYS[kind]):
                    seen.add(kind)
        assert seen == set(S.INGREDIENT_TYPES)

    def test_an_example_shows_an_empty_list(self):
        """The model must learn that an empty list is a correct answer."""
        examples = self._examples()
        empty = [
            kind
            for example in examples
            for kind in S.INGREDIENT_TYPES
            if kind not in S.BOUNDED_ARRAYS
            and example["answer"].get(S.ARRAY_KEYS[kind]) == []
        ]
        assert empty, "no example holds an empty list"

    def test_an_example_shows_a_weighing_with_no_reason(self):
        """Your rule: the word "outweigh" alone is not a real weighing."""
        found = False
        for example in self._examples():
            for item in example["answer"].get("weighings", []):
                if not item["justification_quotes"]:
                    found = True
        assert found

    def test_the_prompt_warns_that_examples_are_not_a_sample(self):
        """A few-shot prompt leaks the base rate of its own examples."""
        out = S.build_prompt("hello", examples=self._examples())
        assert "not a typical sample" in out
        assert "### Example 3" in out

    def test_a_version_mismatch_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema_version": "v99", "examples": []}))
        with pytest.raises(ValueError, match="never pool"):
            S.load_examples(str(path))


# ---------------------------------------------------------------------------
# The repair of a cut answer
# ---------------------------------------------------------------------------


class TestCloseTruncatedJson:
    def test_it_closes_a_nested_cut(self):
        text = '{"dimensions": [{"name": "traffic", "quote": "cars"}, {"name": "gre'
        out = S.close_truncated_json(text)
        assert out is not None
        obj = json.loads(out)
        assert len(obj["dimensions"]) == 1
        assert obj["dimensions"][0]["name"] == "traffic"

    def test_an_array_that_never_closed_disappears(self):
        """This is why `parse_answer` reports the repair.

        The cut point sits after the last container that closed. An array that
        the cap stopped in its first item is gone, thus a lost `verdicts` key
        looks the same as a trace that stated no verdict.
        """
        text = (
            '{"dimensions": [{"name": "a", "quote": "x"}], '
            '"verdicts": [{"label": "Same", "quote": "same", "is_fin'
        )
        obj = json.loads(S.close_truncated_json(text))
        assert obj["dimensions"][0]["name"] == "a"
        assert "verdicts" not in obj

    def test_an_item_that_closed_survives_the_cut(self):
        """The useful case: the cap stops the third item, and two remain."""
        text = (
            '{"verdicts": [{"label": "More", "quote": "a"}, '
            '{"label": "Same", "quote": "b"}, {"label": "Le'
        )
        obj = json.loads(S.close_truncated_json(text))
        assert [v["label"] for v in obj["verdicts"]] == ["More", "Same"]

    def test_a_brace_inside_a_string_does_not_count(self):
        text = '{"dimensions": [{"name": "a", "quote": "he said {done]"}], "verd'
        out = S.close_truncated_json(text)
        obj = json.loads(out)
        assert obj["dimensions"][0]["quote"] == "he said {done]"

    def test_it_gives_none_when_nothing_closed(self):
        assert S.close_truncated_json('{"dimensions": [{"name": "tra') is None

    def test_an_escaped_quote_does_not_end_the_string(self):
        text = '{"dimensions": [{"quote": "she said \\"safe\\" here"}], "verdi'
        obj = json.loads(S.close_truncated_json(text))
        assert obj["dimensions"][0]["quote"] == 'she said "safe" here'


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


class TestParseAnswer:
    def test_it_reads_bare_json(self):
        obj, err, repaired = S.parse_answer('{"dimensions": []}')
        assert err == ""
        assert obj == {"dimensions": []}
        assert repaired is False

    def test_it_reads_json_inside_a_fence(self):
        obj, err, _ = S.parse_answer('```json\n{"dimensions": []}\n```')
        assert err == ""
        assert obj == {"dimensions": []}

    def test_it_repairs_a_cut_answer_and_says_so(self):
        obj, err, repaired = S.parse_answer('{"dimensions": [{"name": "traffic"}, {"na')
        assert err == ""
        assert obj["dimensions"] == [{"name": "traffic"}]
        assert repaired is True

    def test_an_empty_answer_reports_why(self):
        obj, err, _ = S.parse_answer("   ")
        assert obj is None
        assert "empty" in err

    def test_a_bare_list_holds_no_object(self):
        obj, err, _ = S.parse_answer("[1, 2]")
        assert obj is None
        assert "no JSON object" in err

    def test_prose_with_no_json_reports_why(self):
        obj, err, _ = S.parse_answer("I cannot do this task.")
        assert obj is None
        assert "no JSON object" in err


# ---------------------------------------------------------------------------
# The grounds
# ---------------------------------------------------------------------------

TRACE = (
    "The user wants me to compare two subway entrances.\n\n"
    "**Image A analysis:**\n"
    "- There is a lot of traffic (cars, trucks).\n"
    "- Visibility seems good.\n\n"
    "**Comparison:**\n"
    "- However, both are standard NYC subway entrances.\n"
    "- Therefore, Image A is \"Less\" safe than Image B.\n"
)


class TestLocateQuote:
    def test_an_exact_quote(self):
        start, end, method = S.locate_quote(TRACE, "Visibility seems good")
        assert method == "exact"
        assert TRACE[start:end] == "Visibility seems good"

    def test_a_quote_whose_line_break_moved(self):
        """A model joins two lines into one space. The span is still real."""
        quote = "**Comparison:** - However, both are standard NYC subway entrances."
        start, end, method = S.locate_quote(TRACE, quote)
        assert method == "whitespace"
        assert "However" in TRACE[start:end]
        assert TRACE[start:end].startswith("**Comparison:**")

    def test_a_composed_span_is_not_found(self):
        start, end, method = S.locate_quote(TRACE, "Image A is safer overall")
        assert method == "none"
        assert start is None and end is None

    def test_an_empty_quote_is_not_found(self):
        assert S.locate_quote(TRACE, "")[2] == "none"

    def test_offsets_give_the_order_of_two_spans(self):
        """`verdict_revised` and `pseudo_differentiation` rest on this."""
        first = S.locate_quote(TRACE, "a lot of traffic")[0]
        second = S.locate_quote(TRACE, "both are standard")[0]
        assert first < second


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------

ANSWER = {
    "dimensions": [
        {
            "name": "traffic",
            "type": "descriptive",
            "image": "A",
            "valence": "bad",
            "quote": "a lot of traffic (cars, trucks)",
        }
    ],
    "perspectives": [
        {
            "name": "pedestrian",
            "quote": "Visibility seems good",
            "favors": "A",
            "supporting_quotes": ["a lot of traffic"],
            "is_winner": True,
        }
    ],
    "dismissals": [],
    "reconsiderations": [],
    "verdicts": [
        {
            "label": "Less",
            "quote": 'Image A is "Less" safe than Image B',
            "is_final": True,
        }
    ],
    "hedges": [
        {
            "quote": "However, both are standard NYC subway entrances",
            "marker_type": "contrastive",
            "justification_quotes": [],
            "affects_conclusion": False,
        }
    ],
    "weighings": [],
}


class TestIngredientRows:
    def test_one_row_for_each_ingredient(self):
        rows = S.ingredient_rows(TRACE, ANSWER)
        kinds = [r["ingredient_type"] for r in rows]
        assert kinds == ["dimension", "perspective", "verdict", "hedge"]

    def test_every_quote_of_a_good_answer_is_found(self):
        rows = S.ingredient_rows(TRACE, ANSWER)
        assert all(r["quote_found"] for r in rows)

    def test_a_sub_quote_is_checked_too(self):
        """A code rests on the sub-quote, thus an invented one must show."""
        answer = json.loads(json.dumps(ANSWER))
        answer["hedges"][0]["justification_quotes"] = [
            "both are standard",          # real
            "the weather was poor",       # invented
        ]
        rows = S.ingredient_rows(TRACE, answer)
        hedge = [r for r in rows if r["ingredient_type"] == "hedge"][0]
        assert hedge["n_sub_quotes"] == 2
        assert hedge["n_sub_quotes_found"] == 1
        checked = json.loads(hedge["attrs_json"])["justification_quotes"]
        assert [c["method"] for c in checked] == ["exact", "none"]

    def test_the_word_none_is_not_a_span(self):
        answer = json.loads(json.dumps(ANSWER))
        answer["weighings"] = [
            {
                "quote": "However, both are standard",
                "mechanism": "balancing",
                "justification_quotes": ["None"],
                "condition_quotes": ["n/a", ""],
            }
        ]
        rows = S.ingredient_rows(TRACE, answer)
        weigh = [r for r in rows if r["ingredient_type"] == "weighing"][0]
        attrs = json.loads(weigh["attrs_json"])
        assert attrs["justification_quotes"] == []
        assert attrs["condition_quotes"] == []
        assert weigh["n_sub_quotes"] == 0

    def test_a_trace_with_no_ingredient_still_writes_a_row(self):
        rows = S.ingredient_rows(TRACE, None, parse_error="invalid JSON")
        assert len(rows) == 1
        assert rows[0]["ingredient_type"] is None
        assert rows[0]["parse_error"] == "invalid JSON"

    def test_an_empty_object_writes_the_null_row(self):
        rows = S.ingredient_rows(TRACE, {})
        assert len(rows) == 1
        assert rows[0]["ingredient_type"] is None

    def test_the_base_columns_reach_every_row(self):
        rows = S.ingredient_rows(TRACE, ANSWER, base={"case": "subway_safety"})
        assert all(r["case"] == "subway_safety" for r in rows)
        assert all(r["schema_version"] == S.SCHEMA_VERSION for r in rows)

    def test_a_non_dict_item_is_skipped(self):
        answer = json.loads(json.dumps(ANSWER))
        answer["dismissals"] = ["a bare string"]
        rows = S.ingredient_rows(TRACE, answer)
        assert not any(r["ingredient_type"] == "dismissal" for r in rows)

    def test_the_index_follows_the_order_in_the_list(self):
        answer = json.loads(json.dumps(ANSWER))
        answer["dimensions"].append(
            {
                "name": "light",
                "type": "evaluative",
                "image": "A",
                "valence": "good",
                "quote": "Visibility seems good",
            }
        )
        rows = S.ingredient_rows(TRACE, answer)
        dims = [r for r in rows if r["ingredient_type"] == "dimension"]
        assert [d["ingredient_index"] for d in dims] == [0, 1]


class TestGrammar:
    """Test that the backend ENFORCES the schema, and does not only accept it.

    xgrammar drops a keyword it cannot support, and it gives no warning. Both
    keywords below carry a design decision, thus an upgrade must not remove
    them in silence.
    """

    @staticmethod
    def _rules():
        xgr = pytest.importorskip("xgrammar")
        grammar = xgr.Grammar.from_json_schema(json.dumps(S.json_schema()))
        rules = {}
        for line in str(grammar).splitlines():
            if "::=" in line:
                key, value = line.split("::=", 1)
                rules[key.strip()] = value.strip()
        return rules

    def test_a_bounded_array_cannot_be_empty(self):
        """This is the fix for the v1 silent-empty failure."""
        rules = self._rules()
        keys = [S.ARRAY_KEYS[k] for k in S.INGREDIENT_TYPES]
        for name in S.BOUNDED_ARRAYS:
            rule = rules[f"root_prop_{keys.index(name)}"]
            assert '"[" [ \\n\\t]* "]"' not in rule, name

    def test_an_unbounded_array_may_be_empty(self):
        """A trace with no weighing must not be forced to invent one."""
        rules = self._rules()
        keys = [S.ARRAY_KEYS[k] for k in S.INGREDIENT_TYPES]
        for name in keys:
            if name in S.BOUNDED_ARRAYS:
                continue
            rule = rules[f"root_prop_{keys.index(name)}"]
            assert '"[" [ \\n\\t]* "]"' in rule, name

    def test_an_enum_becomes_a_closed_choice(self):
        """A binary column must never hold `yes`, `Yes`, and `true` at once."""
        rules = self._rules()
        # The grammar text escapes a quote, thus a literal reads `\"good\"`.
        alternatives = {
            rule
            for rule in rules.values()
            if '\\"good\\"' in rule and '\\"bad\\"' in rule
        }
        assert alternatives, "no rule holds the valence choice"
        for rule in alternatives:
            assert '\\"neutral\\"' in rule
            # A closed choice, not a general string.
            assert "[^" not in rule


class TestAnswerStats:
    def test_it_counts_each_type(self):
        stats = S.answer_stats(S.ingredient_rows(TRACE, ANSWER))
        assert stats["ingredients"] == 4
        assert stats["type/dimension"] == 1
        assert stats["quote_found_rate"] == 1.0

    def test_a_null_row_counts_as_nothing(self):
        stats = S.answer_stats(S.ingredient_rows(TRACE, None))
        assert stats["ingredients"] == 0

    def test_an_invented_quote_lowers_the_rate(self):
        answer = json.loads(json.dumps(ANSWER))
        answer["dimensions"][0]["quote"] = "a quote that is not there"
        stats = S.answer_stats(S.ingredient_rows(TRACE, answer))
        assert stats["quote_found_rate"] == 0.75


# --------------------------------------------------------- the ingredient order

# The prompt of schema v2, as it stood before the prompt became 3 pieces. The
# refactor must not move a byte: the corpus of 2026-08-18 was extracted with
# this exact text, and a change would make a later run incomparable with it.
V2_PROMPT_SHA256 = "36d771b54b4ed6c474590515175b54a341151c2ff03cceeb7d62784ec0aab991"


def test_the_v2_prompt_is_byte_for_byte_unchanged():
    import hashlib

    from dagspaces.common import ic_schema as S

    digest = hashlib.sha256(S.build_prompt("TRACE_HERE", None).encode()).hexdigest()
    assert digest == V2_PROMPT_SHA256


def test_v3_moves_dismissal_and_nothing_else():
    from dagspaces.common import ic_schema as S

    v2, v3 = S.order_for("v2"), S.order_for("v3")
    assert v2.index("dismissal") == 4 and v3.index("dismissal") == 2
    # Every other class keeps its relative order.
    assert [k for k in v2 if k != "dismissal"] == [k for k in v3 if k != "dismissal"]
    # The prompt numbers follow the order, thus the text and the grammar agree.
    text = S.build_prompt("t", None, order=v3)
    assert "3. dismissals" in text and "5. weighings" in text
    assert list(S.json_schema(v3)["properties"])[2] == "dismissals"


def test_the_row_takes_its_version_from_the_order():
    """A row that says v2 while the run asked in v3 would pool 2 measurements."""
    from dagspaces.common import ic_schema as S

    answer = {"dimensions": [{"name": "x", "quote": "a clean platform",
                              "type": "descriptive", "image": "A",
                              "valence": "good"}],
              "verdicts": [{"label": "More", "quote": "A is safer",
                            "is_final": True}]}
    trace = "a clean platform. A is safer"
    v2_rows = S.ingredient_rows(trace, answer, order=S.order_for("v2"))
    v3_rows = S.ingredient_rows(trace, answer, order=S.order_for("v3"))
    assert {r["schema_version"] for r in v2_rows} == {"v2"}
    assert {r["schema_version"] for r in v3_rows} == {"v3"}


def test_an_unknown_order_is_not_a_version():
    import pytest as _pytest

    from dagspaces.common import ic_schema as S

    with _pytest.raises(ValueError, match="no schema version"):
        S.version_for(("hedge", "dimension"))
