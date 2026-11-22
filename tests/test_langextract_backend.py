"""Unit tests for the LangExtract bridge.

These tests use a stub engine. They never load a model and never touch a GPU,
thus they run on the login node.
"""

import json

import pytest
from omegaconf import OmegaConf

from dagspaces.common.langextract_backend import (
    ExtractionSpec,
    VLLMLanguageModel,
    annotated_to_rows,
    examples_from_config,
    extract_documents,
    validate_examples,
    spec_from_config,
)

# A trace in the shape that gemma-4-12b writes. The spans below appear in it
# word for word, thus the aligner can ground them.
TRACE = (
    "The user wants me to compare the safety of two subway entrances.\n\n"
    "**Image A analysis:**\n"
    "- There are people walking around, some on skateboards.\n"
    "- There's a trash can, a sign, and traffic lights.\n\n"
    "**Image B analysis:**\n"
    "- Shows a subway entrance behind a construction fence.\n\n"
    "**Comparison of safety:**\n"
    "- Image A has more clear visibility of the surroundings.\n"
)

CONFIG = OmegaConf.create(
    {
        "name": "test_cues",
        "schema_version": "test-1",
        "prompt_description": "Extract the cues that the model names.",
        "max_char_buffer": 12000,
        "batch_length": 4,
        "examples": [
            {
                "text": "Image A shows a trash can on the corner.",
                "extractions": [
                    {
                        "extraction_class": "visual_evidence",
                        "extraction_text": "trash can",
                        "attributes": {"image": "A", "valence": "bad"},
                    }
                ],
            },
            {
                # The negative example. A trace repeats the prompt, and that
                # echo is not evidence.
                "text": "The user wants me to compare two images.",
                "extractions": [],
            },
        ],
    }
)


def _answer(*items) -> str:
    """Build the JSON answer that LangExtract expects from a model."""
    extractions = []
    for cls, text, attrs in items:
        entry = {cls: text}
        if attrs:
            entry[f"{cls}_attributes"] = attrs
        extractions.append(entry)
    return json.dumps({"extractions": extractions})


class TestConfigLoading:
    def test_examples_round_trip(self):
        examples = examples_from_config(CONFIG.examples)
        assert len(examples) == 2
        assert examples[0].extractions[0].extraction_class == "visual_evidence"
        assert examples[0].extractions[0].attributes == {"image": "A", "valence": "bad"}

    def test_negative_example_survives(self):
        """An example with no extraction must stay. It teaches the prompt echo."""
        examples = examples_from_config(CONFIG.examples)
        assert examples[1].extractions == []

    def test_spec_from_config(self):
        spec = spec_from_config(CONFIG)
        assert spec.name == "test_cues"
        assert spec.schema_version == "test-1"
        assert spec.max_char_buffer == 12000
        assert spec.temperature == 0.0
        assert spec.classes == ["visual_evidence"]

    def test_guided_json_names_the_classes(self):
        spec = spec_from_config(CONFIG)
        schema = spec.guided_json()
        item = schema["properties"]["extractions"]["items"]
        assert "visual_evidence" in item["properties"]
        assert "visual_evidence_attributes" in item["properties"]

    def test_guided_json_is_optional(self):
        spec = spec_from_config(OmegaConf.merge(CONFIG, {"use_guided_json": False}))
        assert spec.guided_json() is None


class TestExampleValidation:
    def test_a_clean_set_reports_nothing(self):
        assert validate_examples(spec_from_config(CONFIG)) == []

    def test_a_negative_example_is_legal(self):
        """The built-in check raises on this. Ours must not."""
        spec = spec_from_config(CONFIG)
        assert any(not ex.extractions for ex in spec.examples)
        assert validate_examples(spec) == []

    def test_a_paraphrase_is_caught(self):
        """A span that the example text does not hold can never align."""
        bad = OmegaConf.merge(
            CONFIG,
            {
                "examples": [
                    {
                        "text": "Image A shows a trash can on the corner.",
                        "extractions": [
                            {
                                "extraction_class": "visual_evidence",
                                "extraction_text": "a bin for rubbish",
                                "attributes": {"image": "A"},
                            }
                        ],
                    }
                ]
            },
        )
        assert validate_examples(spec_from_config(bad))

    def test_no_span_at_all_is_caught(self):
        empty = OmegaConf.merge(
            CONFIG, {"examples": [{"text": "Nothing to see.", "extractions": []}]}
        )
        assert validate_examples(spec_from_config(empty))


class TestRepair:
    """The token cap cuts the JSON, and the parser then drops every extraction."""

    def _cut(self, n_complete: int) -> str:
        whole = _answer(*[("visual_evidence", f"thing {i}", {"image": "A"})
                          for i in range(n_complete + 1)])
        # Cut inside the last extraction, after the ones before it closed.
        marker = f'"thing {n_complete}"'
        return whole[: whole.index(marker) + 6]

    def test_repair_keeps_the_complete_extractions(self):
        from dagspaces.common.langextract_backend import repair_truncated_json

        repaired = repair_truncated_json(self._cut(3))
        assert repaired is not None
        parsed = json.loads(repaired)
        assert len(parsed["extractions"]) == 3

    def test_repair_gives_none_when_nothing_closed(self):
        from dagspaces.common.langextract_backend import repair_truncated_json

        assert repair_truncated_json('{"extractions": [{"visual_evidence": "tra') is None

    def test_repair_survives_a_brace_inside_a_string(self):
        """A brace inside a value must not move the depth counter."""
        from dagspaces.common.langextract_backend import repair_truncated_json

        text = _answer(("visual_evidence", 'a sign that says "{closed}"', None),
                       ("visual_evidence", "trash can", None))
        cut = text[: text.index("trash can")]
        parsed = json.loads(repair_truncated_json(cut))
        assert len(parsed["extractions"]) == 1
        assert parsed["extractions"][0]["visual_evidence"] == 'a sign that says "{closed}"'

    def test_whole_answer_needs_no_repair(self):
        from dagspaces.common.langextract_backend import repair_truncated_json

        whole = _answer(("visual_evidence", "trash can", None))
        assert json.loads(repair_truncated_json(whole)) == json.loads(whole)


class TestLanguageModelAdapter:
    def test_one_call_for_the_whole_batch(self):
        seen = []

        def fake_generate(prompts):
            seen.append(len(prompts))
            return [_answer(("visual_evidence", "trash can", None))] * len(prompts)

        model = VLLMLanguageModel(fake_generate)
        results = list(model.infer(["a", "b", "c"]))
        assert seen == [3]
        assert len(results) == 3
        assert results[0][0].output.startswith("{")

    def test_short_answer_raises(self):
        """A silent length mismatch would put an answer on the wrong trace."""
        model = VLLMLanguageModel(lambda prompts: ["only one"])
        with pytest.raises(RuntimeError):
            list(model.infer(["a", "b"]))

    def test_empty_batch_makes_no_call(self):
        called = []
        model = VLLMLanguageModel(lambda prompts: called.append(1) or [])
        assert list(model.infer([])) == []
        assert called == []


class TestExtraction:
    def _run(self, answer_text, texts=(TRACE,), doc_ids=("pair-1",)):
        spec = spec_from_config(CONFIG)
        model = VLLMLanguageModel(lambda prompts: [answer_text] * len(prompts))
        return spec, extract_documents(list(texts), list(doc_ids), spec, model)

    def test_spans_align_to_the_source(self):
        spec, docs = self._run(
            _answer(("visual_evidence", "trash can", {"image": "A", "valence": "bad"}))
        )
        assert len(docs) == 1
        extraction = docs[0].extractions[0]
        assert extraction.alignment_status is not None
        start = extraction.char_interval.start_pos
        end = extraction.char_interval.end_pos
        assert TRACE[start:end] == "trash can"

    def test_document_id_survives(self):
        """Read the id, never the position. The order is not a contract."""
        _, docs = self._run(
            _answer(("visual_evidence", "trash can", None)),
            texts=(TRACE, TRACE),
            doc_ids=("pair-1", "pair-2"),
        )
        assert {d.document_id for d in docs} == {"pair-1", "pair-2"}

    def test_one_trace_is_one_chunk(self):
        """The buffer holds the whole trace, thus the engine sees 1 prompt."""
        spec = spec_from_config(CONFIG)
        batches = []

        def fake_generate(prompts):
            batches.append(len(prompts))
            return [_answer(("visual_evidence", "trash can", None))] * len(prompts)

        model = VLLMLanguageModel(fake_generate)
        extract_documents([TRACE], ["pair-1"], spec, model)
        assert sum(batches) == 1

    def test_a_small_buffer_splits_the_trace(self):
        """The guard for the default. A 1000-character buffer breaks a trace."""
        spec = dataclass_replace(spec_from_config(CONFIG), max_char_buffer=200)
        batches = []

        def fake_generate(prompts):
            batches.append(len(prompts))
            return [_answer(("visual_evidence", "trash can", None))] * len(prompts)

        extract_documents([TRACE], ["pair-1"], spec, VLLMLanguageModel(fake_generate))
        assert sum(batches) > 1


class TestRows:
    def test_rows_carry_the_provenance(self):
        spec, docs = TestExtraction()._run(
            _answer(("visual_evidence", "trash can", {"image": "A", "valence": "bad"}))
        )
        rows = annotated_to_rows(
            docs[0], spec, "gemma-4-12b", base={"pair_id": "pair-1", "case": "subway_safety"}
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["pair_id"] == "pair-1"
        assert row["case"] == "subway_safety"
        assert row["extractor_model"] == "gemma-4-12b"
        assert row["schema_version"] == "test-1"
        assert row["extraction_class"] == "visual_evidence"
        assert json.loads(row["attributes_json"]) == {"image": "A", "valence": "bad"}
        assert row["char_start"] is not None
        assert row["alignment_status"] == "match_exact"
        assert row["is_quotable"] is True

    def test_a_composed_sentence_is_not_quotable(self):
        """A sentence the model built from a fragment plus its own words.

        The trace holds the opening. The rest is the model's own summary, thus
        the aligner can place only the opening, and the offsets point at a
        fragment. No reader may quote such a row.
        """
        composed = (
            "They both look pretty standard for NYC and neither entrance shows "
            "any real hazard to a commuter"
        )
        spec, docs = TestExtraction()._run(_answer(("decision", composed, None)))
        rows = annotated_to_rows(docs[0], spec, "gemma-4-12b")
        assert rows[0]["alignment_status"] != "match_exact"
        assert rows[0]["is_quotable"] is False

    def test_a_silent_trace_still_writes_a_row(self):
        spec, docs = TestExtraction()._run(json.dumps({"extractions": []}))
        rows = annotated_to_rows(docs[0], spec, "gemma-4-12b", base={"pair_id": "pair-1"})
        assert len(rows) == 1
        assert rows[0]["extraction_class"] is None
        assert rows[0]["pair_id"] == "pair-1"


def dataclass_replace(spec: ExtractionSpec, **changes) -> ExtractionSpec:
    import dataclasses

    return dataclasses.replace(spec, **changes)


# ---------------------------------------------------------------------------
# Version 2: enums on the attribute values, and the verbatim test
# ---------------------------------------------------------------------------

V2_CONFIG = OmegaConf.create(
    {
        "name": "test_cues",
        "schema_version": "test-2",
        "prompt_description": "Extract the cues that the model names.",
        "max_char_buffer": 12000,
        "batch_length": 4,
        "attribute_values": {
            "visual_evidence": {
                "image": ["A", "B", "both", "unclear"],
                "valence": ["good", "bad", "neutral"],
            },
            "comparison": {
                "direction": ["A", "B", "neither"],
                # `dimension` stays free: it is absent here.
                "hedged": ["yes", "no"],
            },
        },
        "examples": [
            {
                "text": "Image A shows a trash can on the corner. A beats B on light.",
                "extractions": [
                    {
                        "extraction_class": "visual_evidence",
                        "extraction_text": "trash can",
                        "attributes": {"image": "A", "valence": "bad"},
                    },
                    {
                        "extraction_class": "comparison",
                        "extraction_text": "A beats B on light",
                        "attributes": {"direction": "A", "dimension": "light", "hedged": "no"},
                    },
                ],
            },
        ],
    }
)


class TestAttributeEnums:
    def test_the_enum_reaches_the_schema(self):
        schema = spec_from_config(V2_CONFIG).guided_json()
        props = schema["properties"]["extractions"]["items"]["properties"]
        valence = props["visual_evidence_attributes"]["properties"]["valence"]
        assert valence["enum"] == ["good", "bad", "neutral"]
        assert valence["type"] == "string"

    def test_a_declared_attribute_is_required(self):
        schema = spec_from_config(V2_CONFIG).guided_json()
        props = schema["properties"]["extractions"]["items"]["properties"]
        assert set(props["comparison_attributes"]["required"]) == {"direction", "hedged"}

    def test_a_free_field_stays_free(self):
        schema = spec_from_config(V2_CONFIG).guided_json()
        props = schema["properties"]["extractions"]["items"]["properties"]
        dimension = props["comparison_attributes"]["properties"]["dimension"]
        assert "enum" not in dimension

    def test_a_class_without_an_enum_stays_as_it_was(self):
        """Version 1 has no `attribute_values`, and its schema must not change."""
        schema = spec_from_config(CONFIG).guided_json()
        props = schema["properties"]["extractions"]["items"]["properties"]
        assert "enum" not in props["visual_evidence_attributes"]["properties"]["image"]
        assert "required" not in props["visual_evidence_attributes"]

    def test_the_v2_examples_pass(self):
        assert validate_examples(spec_from_config(V2_CONFIG)) == []

    def test_a_value_outside_the_enum_is_caught(self):
        bad = OmegaConf.create(OmegaConf.to_container(V2_CONFIG, resolve=True))
        bad.examples[0].extractions[0].attributes.valence = "ugly"
        issues = validate_examples(spec_from_config(bad))
        assert any("valence='ugly'" in i for i in issues)

    def test_a_missing_required_attribute_is_caught(self):
        bad = OmegaConf.create(OmegaConf.to_container(V2_CONFIG, resolve=True))
        del bad.examples[0].extractions[1].attributes["hedged"]
        issues = validate_examples(spec_from_config(bad))
        assert any("lacks the required attribute 'hedged'" in i for i in issues)


class TestTheShippedSchemas:
    """The config groups of the paper corpus (v1 to v3) and of F4 (single-v1)."""

    CONF = "dagspaces/pairwisevqa/conf/extract"

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_the_examples_align(self, version):
        spec = spec_from_config(OmegaConf.load(f"{self.CONF}/urban_cues_{version}.yaml"))
        assert validate_examples(spec) == []
        assert len(spec.classes) == 7

    def test_v2_holds_the_longest_qwen_trace(self):
        """The qwen3.5-9b runs of 2026-08-14 reach 35,598 characters."""
        spec = spec_from_config(OmegaConf.load(f"{self.CONF}/urban_cues_v2.yaml"))
        assert spec.max_char_buffer > 35598

    @pytest.mark.parametrize("name,version", [
        ("urban_cues_v2", "v2"),
        ("urban_cues_v3", "v3"),
        ("urban_cues_single_v1", "single-v1"),
    ])
    def test_enums_sit_inside_the_analysis_vocabulary(self, name, version):
        """`_extractions.VOCABULARY` is the union of the values of every
        version, thus each value a version declares must be in it. A value
        that is absent there becomes `other` in the analysis."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "notebooks" / "cvpr"))
        import _extractions as X  # noqa: E402

        spec = spec_from_config(OmegaConf.load(f"{self.CONF}/{name}.yaml"))
        assert spec.schema_version == version
        assert version in X.SCHEMA_VERSIONS
        for klass, attrs in spec.attribute_values.items():
            assert klass in X.VOCABULARY, klass
            for attr, values in attrs.items():
                have = {str(v) for v in X.VOCABULARY[klass][attr]}
                missing = [v for v in values if v not in have]
                assert not missing, (klass, attr, missing)
        assert set(spec.classes) == set(X.CLASSES_BY_VERSION[version])

    def test_single_v1_examples_align_and_hold_the_longest_f4_text(self):
        """The longest F4 text is a qwen3.5-9b trace of 55,231 characters."""
        spec = spec_from_config(OmegaConf.load(f"{self.CONF}/urban_cues_single_v1.yaml"))
        assert validate_examples(spec) == []
        assert len(spec.classes) == 5
        assert spec.max_char_buffer > 55231

    def test_v3_examples_align_and_teach_every_class(self):
        spec = spec_from_config(OmegaConf.load(f"{self.CONF}/urban_cues_v3.yaml"))
        assert validate_examples(spec) == []
        assert len(spec.classes) == 12


class TestVerbatim:
    def test_the_source_text_at_the_offsets_is_the_span(self):
        from langextract.core import data as core_data

        from dagspaces.common.langextract_backend import is_verbatim

        text = "Image A shows a trash can on the corner."
        exact = core_data.CharInterval(start_pos=16, end_pos=25)
        assert is_verbatim(text, exact, "trash can")
        assert not is_verbatim(text, exact, "trash cans")
        assert not is_verbatim(text, None, "trash can")

    def test_the_row_carries_the_flag(self):
        from langextract import data as lx_data
        from langextract.core import data as core_data

        spec = spec_from_config(CONFIG)
        text = "Image A shows a trash can on the corner."
        good = lx_data.Extraction(
            extraction_class="visual_evidence", extraction_text="trash can",
            char_interval=core_data.CharInterval(start_pos=16, end_pos=25),
            alignment_status=core_data.AlignmentStatus.MATCH_EXACT,
        )
        # A fuzzy row whose text differs from the source. Quotable, not verbatim.
        near = lx_data.Extraction(
            extraction_class="visual_evidence", extraction_text="a trash bin",
            char_interval=core_data.CharInterval(start_pos=14, end_pos=25),
            alignment_status=core_data.AlignmentStatus.MATCH_FUZZY,
        )
        doc = lx_data.AnnotatedDocument(document_id="d", text=text, extractions=[good, near])
        rows = annotated_to_rows(doc, spec, "stub")
        assert [r["is_verbatim"] for r in rows] == [True, False]
        assert [r["is_quotable"] for r in rows] == [True, True]


class TestDuplicates:
    def test_a_later_copy_is_a_duplicate(self):
        from langextract import data as lx_data
        from langextract.core import data as core_data

        spec = spec_from_config(CONFIG)
        text = "a trash can. a trash can. a sign."
        def ext(txt, start, cls="visual_evidence"):
            return lx_data.Extraction(
                extraction_class=cls, extraction_text=txt,
                char_interval=core_data.CharInterval(start_pos=start, end_pos=start + len(txt)),
                alignment_status=core_data.AlignmentStatus.MATCH_EXACT,
            )
        doc = lx_data.AnnotatedDocument(
            document_id="d", text=text,
            extractions=[ext("trash can", 2), ext("trash can", 15), ext("sign", 28),
                         ext("trash can", 2, cls="inference")],
        )
        rows = annotated_to_rows(doc, spec, "stub")
        # The same text under another class is not a duplicate.
        assert [r["is_duplicate"] for r in rows] == [False, True, False, False]
