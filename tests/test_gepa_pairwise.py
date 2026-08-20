"""Stub-level tests for the pairwise GEPA harness (no GPU / vLLM / gepa).

The load-bearing properties here are the LEAK GUARANTEES: in the
prompt-recovery experiments, no harness-authored text shown to the reflection
model may mention the task. Everything else is prompt-construction plumbing
that a stub engine can exercise.
"""
from __future__ import annotations

import json

import pytest

from dagspaces.urbanpairvqa.prompt_opt import gepa_pairwise as gp
from dagspaces.urbanpairvqa.prompt_opt import task_inference_probe as probe

# Task words that must never appear in harness-authored reflector-visible
# text. (The true-axis phrases in the probe's TASKS registry are exempt: the
# axis-ceiling evals are reference numbers, not part of a recovery run.)
TASK_LEAK_TERMS = [
    "subway", "safe", "safety", "restaurant", "eat", "school", "child",
    "library", "libraries", "maintain", "maintained", "photography",
    "photoshoot", "appealing", "wealth", "livability",
]


def assert_no_leak(text: str, *, where: str) -> None:
    low = text.lower()
    for term in TASK_LEAK_TERMS:
        assert term not in low, f"task term {term!r} leaked into {where}: {text[:120]}"


class StubEngine:
    """Duck-typed PairwiseTaskEngine: records calls, returns fixed answers."""

    def __init__(self, answers=None):
        self.classify_calls = []
        self.describe_calls = []
        self.n_task_calls = 0
        self._answers = answers

    def classify(self, rows, system, user_text):
        self.classify_calls.append({"system": system, "user_text": user_text,
                                    "n": len(rows)})
        self.n_task_calls += len(rows)
        answers = self._answers or ["More"] * len(rows)
        return list(answers)[: len(rows)]

    def describe(self, rows, style="independent"):
        self.describe_calls.append({"style": style, "n": len(rows)})
        return [f"caption-{i}" for i in range(len(rows))]


def _rows(n=3, expected="More"):
    return [{"sample_id": f"s{i}", "expected_answer": expected,
             "presented_left_path": f"/img/{i}_a.jpg",
             "presented_right_path": f"/img/{i}_b.jpg"} for i in range(n)]


# ---------------------------------------------------------------------------
# leak guarantees
# ---------------------------------------------------------------------------

def test_caption_prompts_are_task_neutral():
    for name, text in [("DESCRIBE_SYSTEM", gp.DESCRIBE_SYSTEM),
                       ("DESCRIBE_PROMPT", gp.DESCRIBE_PROMPT),
                       ("DESCRIBE_CONTRASTIVE_PROMPT",
                        gp.DESCRIBE_CONTRASTIVE_PROMPT)]:
        assert_no_leak(text, where=name)


def test_axis_scaffold_and_notes_are_task_neutral():
    for name, text in [("AXIS_SYSTEM", gp.AXIS_SYSTEM),
                       ("AXIS_USER_TEMPLATE", gp.AXIS_USER_TEMPLATE),
                       ("AXIS_COMPONENT_NOTE", gp.AXIS_COMPONENT_NOTE),
                       ("AXIS_REFLECTION_TEMPLATE", gp.AXIS_REFLECTION_TEMPLATE),
                       ("REFLECTION_INPUT_NOTE", gp.REFLECTION_INPUT_NOTE)]:
        assert_no_leak(text, where=name)


def test_axis_reflection_template_satisfies_gepa_contract():
    # gepa validates these placeholders and extracts the proposal from the
    # first/last ``` block of the reflector output.
    assert "<curr_param>" in gp.AXIS_REFLECTION_TEMPLATE
    assert "<side_info>" in gp.AXIS_REFLECTION_TEMPLATE
    assert "``` blocks" in gp.AXIS_REFLECTION_TEMPLATE


def test_sanitize_axis_skips_markdown_headings():
    proposal = "### Core Logic: The Rule\n\nsunlit and open"
    assert gp._sanitize_axis(proposal) == "sunlit and open"


def test_reflective_feedback_is_task_neutral():
    adapter = gp.PairwiseGEPAAdapter(StubEngine(answers=["Less", "More"]))
    batch = adapter.evaluate(_rows(2), {"system_prompt": "s", "user_prompt": "u"})
    data = adapter.make_reflective_dataset(
        {"user_prompt": "u"}, batch, ["user_prompt"])
    for rec in data["user_prompt"]:
        assert_no_leak(rec["Feedback"], where="Feedback")


def test_contrastive_prompt_allows_comparison_forbids_verdict():
    text = gp.DESCRIBE_CONTRASTIVE_PROMPT.lower()
    assert "do not compare" not in text
    assert "differences" in text
    assert "prefer" in text  # the verdict ban


# ---------------------------------------------------------------------------
# axis sanitization + rendering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("inviting and well kept", "inviting and well kept"),
    ('"orderly"', "orderly"),
    ("```\nlively at street level\n```", "lively at street level"),
    ("Axis: calm and orderly.", "calm and orderly"),
    ("", gp.DEFAULT_SEED_AXIS),
    ("   \n\n  ", gp.DEFAULT_SEED_AXIS),
])
def test_sanitize_axis(raw, expected):
    assert gp._sanitize_axis(raw) == expected


def test_sanitize_axis_truncates_long_proposals():
    long = " ".join(f"w{i}" for i in range(40))
    out = gp._sanitize_axis(long)
    assert len(out.split()) == 16


def test_render_candidate_axis_mode():
    adapter = gp.PairwiseGEPAAdapter(StubEngine(), candidate_mode="axis")
    system, user = adapter.render_candidate({"axis": "orderly"})
    assert system == gp.AXIS_SYSTEM
    assert user.count("orderly") == 2  # question + interpretation line
    assert user.endswith(gp.FIXED_SUFFIX)


def test_render_candidate_prompt_mode_unchanged():
    adapter = gp.PairwiseGEPAAdapter(StubEngine())
    system, user = adapter.render_candidate(
        {"system_prompt": "sys", "user_prompt": "usr"})
    assert (system, user) == ("sys", f"usr\n\n{gp.FIXED_SUFFIX}")


def test_evaluate_axis_mode_sends_scaffold_and_scores():
    engine = StubEngine(answers=["More", "Same", "MuchLess"])
    adapter = gp.PairwiseGEPAAdapter(engine, candidate_mode="axis")
    batch = adapter.evaluate(_rows(3, expected="More"), {"axis": "orderly"})
    assert engine.classify_calls[0]["system"] == gp.AXIS_SYSTEM
    assert "orderly" in engine.classify_calls[0]["user_text"]
    assert batch.scores == [1.0, 0.75, 0.25]


# ---------------------------------------------------------------------------
# reflective dataset shapes
# ---------------------------------------------------------------------------

def _reflective_records(**adapter_kwargs):
    engine = StubEngine(answers=["Less", "More"])
    adapter = gp.PairwiseGEPAAdapter(engine, **adapter_kwargs)
    component = "axis" if adapter_kwargs.get("candidate_mode") == "axis" else "user_prompt"
    candidate = ({"axis": "orderly"} if component == "axis"
                 else {"system_prompt": "s", "user_prompt": "u"})
    batch = adapter.evaluate(_rows(2), candidate)
    data = adapter.make_reflective_dataset(candidate, batch, [component])
    return engine, data[component]


def test_reflective_records_legacy_prompt_mode():
    _, records = _reflective_records()
    assert all(set(r["Inputs"]) == {"user_prompt"} for r in records)


def test_reflective_records_caption_mode_has_note_and_captions():
    engine, records = _reflective_records(reflection_descriptions=True,
                                          caption_style="contrastive")
    assert engine.describe_calls[0]["style"] == "contrastive"
    for rec in records:
        assert set(rec["Inputs"]) == {"note", "image_descriptions"}
        assert rec["Inputs"]["note"] == gp.REFLECTION_INPUT_NOTE


def test_reflective_records_axis_mode_carries_component_role():
    _, records = _reflective_records(candidate_mode="axis",
                                     reflection_descriptions=True)
    for rec in records:
        assert set(rec["Inputs"]) == {"component_role", "note",
                                      "image_descriptions"}
        assert rec["Inputs"]["component_role"] == gp.AXIS_COMPONENT_NOTE


def test_description_cache_dedups_by_sample_id(tmp_path):
    engine = StubEngine()
    cache = tmp_path / "descriptions.jsonl"
    adapter = gp.PairwiseGEPAAdapter(engine, reflection_descriptions=True,
                                     description_cache_path=cache)
    candidate = {"system_prompt": "s", "user_prompt": "u"}
    batch = adapter.evaluate(_rows(2), candidate)
    adapter.make_reflective_dataset(candidate, batch, ["user_prompt"])
    adapter.make_reflective_dataset(candidate, batch, ["user_prompt"])
    assert sum(c["n"] for c in engine.describe_calls) == 2  # no re-captioning
    lines = cache.read_text().strip().splitlines()
    assert len(lines) == 2
    assert {json.loads(l)["sample_id"] for l in lines} == {"s0", "s1"}


# ---------------------------------------------------------------------------
# probe helpers
# ---------------------------------------------------------------------------

def test_guess_prompt_is_task_neutral_and_well_formed():
    records = [{"caption": f"cap {i}", "label": "More"} for i in range(3)]
    text = probe._build_guess_prompt(records)
    assert_no_leak(text, where="guess prompt")
    assert "QUESTION:" in text
    assert text.count("Description:") == 3


def test_constant_baseline_math():
    valset = [{"expected_answer": a} for a in
              ["More", "More", "Same", "MuchLess"]]
    out = probe._constant_baseline(valset, "More")
    assert out["exact"] == pytest.approx(0.5)
    assert out["ordinal"] == pytest.approx((1.0 + 1.0 + 0.75 + 0.25) / 4)


# ---------------------------------------------------------------------------
# multimodal reflection (rung 5)
# ---------------------------------------------------------------------------

def test_multimodal_reflection_strings_are_task_neutral():
    for name, text in [("MULTIMODAL_AXIS_REFLECTION_TEMPLATE",
                        gp.MULTIMODAL_AXIS_REFLECTION_TEMPLATE),
                       ("MULTIMODAL_INPUT_NOTE", gp.MULTIMODAL_INPUT_NOTE)]:
        assert_no_leak(text, where=name)


def test_multimodal_axis_template_satisfies_gepa_contract():
    t = gp.MULTIMODAL_AXIS_REFLECTION_TEMPLATE
    assert "<curr_param>" in t and "<side_info>" in t and "``` blocks" in t


def _png_data_uri(color=(10, 20, 30), size=(4, 4)):
    import base64, io
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_parse_multimodal_reflection_roundtrip():
    uri = _png_data_uri(size=(5, 4))
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "look at [IMAGE-1] and [IMAGE-2]"},
        {"type": "image_url", "image_url": {"url": uri}},
        {"type": "image_url", "image_url": {"url": uri}},
    ]}]
    text, imgs = gp._parse_multimodal_reflection(messages)
    assert text == "look at [IMAGE-1] and [IMAGE-2]"
    assert len(imgs) == 2 and imgs[0].size == (5, 4)


def test_parse_multimodal_reflection_text_only():
    text, imgs = gp._parse_multimodal_reflection(
        [{"role": "user", "content": [{"type": "text", "text": "x"}]}])
    assert text == "x" and imgs == []


def test_reflection_images_and_descriptions_mutually_exclusive():
    with pytest.raises(ValueError):
        gp.PairwiseGEPAAdapter(StubEngine(), reflection_images=True,
                               reflection_descriptions=True)


def test_reflective_records_image_mode_attaches_pair(monkeypatch):
    import sys, types
    fake = types.ModuleType("gepa")

    class FakeImage:
        def __init__(self, path=None, **kw):
            self.path = path
    fake.Image = FakeImage
    monkeypatch.setitem(sys.modules, "gepa", fake)

    engine = StubEngine(answers=["Less", "More"])
    adapter = gp.PairwiseGEPAAdapter(engine, candidate_mode="axis",
                                     reflection_images=True)
    batch = adapter.evaluate(_rows(2), {"axis": "orderly"})
    data = adapter.make_reflective_dataset({"axis": "orderly"}, batch, ["axis"])
    for rec in data["axis"]:
        ins = rec["Inputs"]
        assert ins["component_role"] == gp.AXIS_COMPONENT_NOTE
        assert ins["note"] == gp.MULTIMODAL_INPUT_NOTE
        assert isinstance(ins["Image A"], FakeImage)
        assert isinstance(ins["Image B"], FakeImage)
        assert ins["Image A"].path.endswith("_a.jpg")
        assert ins["Image B"].path.endswith("_b.jpg")
        # no caption channel when images are on
        assert "image_descriptions" not in ins
