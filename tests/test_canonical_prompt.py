"""The canonical prompt path must match the battery, token for token.

`monocle/canonical.py` builds the forward pass the Jacobian lens reads. If that
prompt drifts from the CVPR battery, every lens number describes a run nobody
published. These tests are the gate.

The three mismatches this file locks down (each one broke the pre-2026-08-11
`safety_workspace.py` build):

  1. image_layout must be `interleaved_labels`, so gemma-4-12b binds image B.
  2. `system: null` must send NO system turn.
  3. The abstention label must reach both the answer enum and the user text.

No GPU is needed. The tokenizer tests skip when the model directory is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import canonical  # noqa: E402

MODEL_DIR = Path("/share/pierson/matt/zoo/models/gemma-4-12B-it")

ROW = {
    "pair_id": "unit_00000042",
    "presented_left_path": "/tmp/a.jpg",
    "presented_right_path": "/tmp/b.jpg",
}


def _registered(case: str = "subway_safety", kind: str = "proxy") -> bool:
    try:
        canonical.registry_dir(case, kind)
    except (FileNotFoundError, ValueError):
        return False
    return True


needs_registry = pytest.mark.skipif(
    not _registered(), reason="canonical registry not present")
needs_model = pytest.mark.skipif(
    not MODEL_DIR.is_dir(), reason="gemma-4-12B-it not on this host")


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------
def test_rejects_unlensable_model():
    with pytest.raises(ValueError, match="not lensable"):
        canonical.registry_dir("subway_safety", "proxy", model="qwen3.5-9b")


def test_rejects_unknown_case_and_kind():
    with pytest.raises(ValueError, match="unknown case"):
        canonical.registry_dir("not_a_case")
    with pytest.raises(ValueError, match="unknown kind"):
        canonical.registry_dir("subway_safety", "not_a_kind")


@needs_registry
@pytest.mark.parametrize("case", canonical.CASES)
def test_every_case_records_interleaved_labels(case):
    """Mismatch 1. The registry gate demands this layout for gemma-4-12b."""
    cfg = canonical.prompt_cfg(case, "proxy")
    assert cfg.prompt.image_layout == "interleaved_labels"


@needs_registry
@pytest.mark.parametrize("case", canonical.CASES)
def test_every_case_sends_no_system_turn(case):
    """Mismatch 2. The minimal prompt contract omits the system turn."""
    cfg = canonical.prompt_cfg(case, "proxy")
    assert canonical.system_text(cfg) is None


@needs_registry
@pytest.mark.parametrize("case", canonical.CASES)
def test_every_case_enables_abstention(case):
    """Mismatch 3, part one. Abstention is on for the whole battery."""
    cfg = canonical.prompt_cfg(case, "proxy")
    assert canonical.not_sure_enabled(cfg)
    assert canonical.not_sure_label(cfg) == "NotSure"


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------
@needs_registry
def test_messages_carry_no_system_role():
    """Mismatch 2. An empty system turn is not the same as no system turn."""
    cfg = canonical.prompt_cfg("subway_safety", "proxy")
    messages = canonical.build_messages(ROW, cfg)
    assert [m["role"] for m in messages] == ["user"]


@needs_registry
def test_messages_interleave_the_text_anchors():
    """Mismatch 1, at the block level: anchor, image, anchor, image, prompt."""
    cfg = canonical.prompt_cfg("subway_safety", "proxy")
    blocks = canonical.build_messages(ROW, cfg)[0]["content"]
    kinds = [b["type"] for b in blocks]
    assert kinds == ["text", "image", "text", "image", "text"]
    assert blocks[0]["text"] == "Image A:"
    assert blocks[2]["text"] == "Image B:"


@needs_registry
def test_image_blocks_are_rewritten_for_the_hf_processor():
    """gemma4_unified's chat template matches on type == 'image'."""
    cfg = canonical.prompt_cfg("subway_safety", "proxy")
    hf = canonical.build_messages(ROW, cfg, for_hf=True)[0]["content"]
    raw = canonical.build_messages(ROW, cfg, for_hf=False)[0]["content"]
    assert [b["type"] for b in hf if b["type"] != "text"] == ["image", "image"]
    assert [b["type"] for b in raw if b["type"] != "text"] == [
        "image_url", "image_url"]


@needs_registry
def test_user_text_carries_the_abstention_guidance():
    """Mismatch 3, part two. The guidance line is part of the token sequence."""
    cfg = canonical.prompt_cfg("subway_safety", "proxy")
    text = canonical.user_text("unit_00000042", cfg)
    assert '"NotSure"' in text
    assert "true uncertainty" in text
    assert "Pair ID: unit_00000042" in text
    assert text.endswith("Interpret labels as Image A relative to Image B.")


@needs_registry
def test_user_text_matches_the_recorded_question():
    """The rendered text must hold the question the registry recorded."""
    import json
    manifest = json.loads(
        (canonical.REGISTRY / "manifest.json").read_text())
    questions = {
        r["case"]: r["question"] for r in manifest["runs"]
        if r["model"] == canonical.MODEL and r["kind"] == "proxy"
    }
    for case, question in questions.items():
        text = canonical.user_text("p0", canonical.prompt_cfg(case, "proxy"))
        assert question in text, f"{case}: recorded question absent"


# ---------------------------------------------------------------------------
# The identity test: our text == the production text
# ---------------------------------------------------------------------------
@needs_registry
@needs_model
@pytest.mark.parametrize("case", canonical.CASES)
def test_chat_text_is_identical_to_the_production_path(case):
    """The whole point of this file.

    Builds the chat text twice: once through `monocle.canonical`, once through
    the production chain that `run_vllm_inference` runs for gemma4_unified.
    The strings must be equal.
    """
    from transformers import AutoTokenizer

    from dagspaces.common.vllm_inference import (
        _gemma4_unified_chat_template, _to_image_type_blocks)
    from monocle.canonical import _load_pairwise_vqa

    is_g4u, tmpl = _gemma4_unified_chat_template(str(MODEL_DIR))
    assert is_g4u and tmpl, "gemma4_unified template not found"
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    cfg = canonical.prompt_cfg(case, "proxy")

    # Production chain, as the stage runs it.
    row = dict(ROW)
    row["prompt"] = _load_pairwise_vqa()._render_pair_prompt(row, cfg)
    prod_messages = _to_image_type_blocks(
        _load_pairwise_vqa()._make_pairwise_preprocess(cfg)(row)["messages"])
    prod_text = tok.apply_chat_template(
        prod_messages, tokenize=False, add_generation_prompt=True,
        chat_template=tmpl)

    ours = canonical.render_chat_text(
        tok, tmpl, canonical.build_messages(ROW, cfg), cfg, force_prefix="")

    assert ours == prod_text


@needs_registry
@needs_model
def test_forced_prefix_is_appended_last():
    """The lens reads the final position, so the prefix must end the text."""
    from transformers import AutoTokenizer

    from dagspaces.common.vllm_inference import _gemma4_unified_chat_template

    _, tmpl = _gemma4_unified_chat_template(str(MODEL_DIR))
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    cfg = canonical.prompt_cfg("subway_safety", "proxy")
    text = canonical.render_chat_text(
        tok, tmpl, canonical.build_messages(ROW, cfg), cfg)
    assert text.endswith(canonical.FORCE_PREFIX)


# ---------------------------------------------------------------------------
# Label first tokens, with abstention
# ---------------------------------------------------------------------------
@needs_registry
@needs_model
def test_label_classes_include_abstention_and_collapse_much():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    cfg = canonical.prompt_cfg("subway_safety", "proxy")
    label_first, class_ids, collapse = canonical.label_classes(tok, cfg)

    assert set(label_first) == {*canonical.ORDINAL, "NotSure"}
    # MuchLess and MuchMore share their first token; nothing else does.
    assert label_first["MuchLess"] == label_first["MuchMore"]
    assert collapse["MuchLess"] == collapse["MuchMore"] == "Much*"
    assert set(class_ids) == {"Much*", "Less", "Same", "More", "NotSure"}
    assert len(set(class_ids.values())) == len(class_ids)


# ---------------------------------------------------------------------------
# Presented order
# ---------------------------------------------------------------------------
def test_presented_images_uses_presented_order():
    left, right = canonical.presented_images(ROW)
    assert (left, right) == ("/tmp/a.jpg", "/tmp/b.jpg")


def test_presented_images_rejects_a_row_without_presented_paths():
    with pytest.raises(KeyError, match="presented order"):
        canonical.presented_images({"image_path_a": "x", "image_path_b": "y"})


# ---------------------------------------------------------------------------
# Contrast conditions
# ---------------------------------------------------------------------------
@needs_registry
def test_conditions_vary_only_the_user_template():
    """The whole point of the rebuilt contrast.

    prod / neutral / axis must differ in `prompt.user_template` and in nothing
    else. The pre-2026-08-11 build varied the persona at the same time.
    """
    from omegaconf import OmegaConf

    conds = canonical.build_conditions("subway_safety")
    assert set(conds) == {"prod", "neutral", "axis"}

    stripped = {}
    for name, cfg in conds.items():
        d = OmegaConf.to_container(cfg, resolve=True)
        d["prompt"].pop("user_template")
        stripped[name] = d
    assert stripped["prod"] == stripped["neutral"] == stripped["axis"]

    templates = {n: str(c.prompt.user_template) for n, c in conds.items()}
    assert len(set(templates.values())) == 3


@needs_registry
def test_every_condition_keeps_the_canonical_frame():
    """No system turn, interleaved layout, abstention guidance — in all arms."""
    for name, cfg in canonical.build_conditions("subway_safety").items():
        assert canonical.system_text(cfg) is None, name
        assert cfg.prompt.image_layout == "interleaved_labels", name
        blocks = canonical.build_messages(ROW, cfg)[0]["content"]
        assert [b["type"] for b in blocks] == [
            "text", "image", "text", "image", "text"], name
        text = canonical.user_text("p0", cfg)
        assert '"NotSure"' in text, name
        assert text.endswith(
            "Interpret labels as Image A relative to Image B."), name


@needs_registry
def test_neutral_never_names_the_attribute_or_the_unit():
    """The neutral arm must not leak the judgment it is the control for."""
    cfg = canonical.build_conditions("subway_safety")["neutral"]
    text = canonical.user_text("p0", cfg).lower()
    for leak in ("safe", "safer", "safety", "subway", "station", "new york"):
        assert leak not in text, f"neutral arm leaked {leak!r}"


@needs_registry
def test_a_case_without_a_recovered_axis_has_no_axis_arm():
    conds = canonical.build_conditions("road_quality")
    assert set(conds) == {"prod", "neutral"}
    with pytest.raises(ValueError, match="no recovered axis"):
        canonical.build_conditions("road_quality", conds=["axis"])


@needs_registry
@pytest.mark.parametrize("case", canonical.CASES)
def test_prod_condition_is_the_registered_prompt(case):
    """The prod arm must stay byte-identical to what the run recorded."""
    prod = canonical.build_conditions(case, conds=["prod"])["prod"]
    assert (str(prod.prompt.user_template)
            == str(canonical.prompt_cfg(case).prompt.user_template))


# ---------------------------------------------------------------------------
# The teacher-forced prefix
# ---------------------------------------------------------------------------
def test_default_prefix_matches_how_the_model_writes_the_object():
    """Measured: the compact prefix reads a flat zero on most cases.

    Jobs 199648/199649, 150 pairs per case, mean p(prod label) at L47 under
    the production prompt: road_quality 0.001 compact vs 0.769 natural;
    restaurants 0.001 vs 0.771; subway_safety 0.778 vs 0.774. The default must
    therefore be the natural form.
    """
    assert canonical.FORCE_PREFIX == '{\n  "answer": "'
    assert canonical.FORCE_PREFIX_COMPACT == '{"answer": "'


@needs_registry
def test_answer_tokens_defaults_to_the_natural_prefix():
    from monocle import answer_tokens as at

    assert at.build_parser().parse_args([]).force_prefix == "natural"


@needs_registry
def test_registered_answers_use_the_natural_formatting():
    """The claim above, checked against the registry rather than trusted."""
    df = canonical.load_results(
        "subway_safety", "proxy", n=200, columns=["answer"])
    starts = df["answer"].astype(str).str.startswith('{\n  "answer": "')
    assert starts.mean() > 0.9, (
        f"only {starts.mean():.1%} of answers use the natural prefix")
