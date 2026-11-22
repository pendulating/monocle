"""The reasoning-on read. See research/monocle/thinking_read.py."""
import pytest

from research.monocle import single as sg
from research.monocle import thinking_read as tr


class TestSplitThought:
    def test_it_keeps_the_closing_marker(self):
        """The read prompt must hold a CLOSED channel, or the label position
        sits inside a half-written thought."""
        block, closed = tr.split_thought("thinking hard<channel|>{\"answer\"")
        assert closed and block.endswith(tr.CHANNEL_CLOSE)

    def test_it_drops_what_comes_after(self):
        block, _ = tr.split_thought("abc<channel|>THE ANSWER")
        assert "THE ANSWER" not in block

    def test_an_unclosed_trace_is_reported_not_guessed(self):
        block, closed = tr.split_thought("the model rambled on and on")
        assert closed is False
        assert block == "the model rambled on and on"

    def test_only_the_first_close_counts(self):
        block, _ = tr.split_thought("a<channel|>b<channel|>c")
        assert block == "a<channel|>"


class TestReasoningText:
    def test_the_markers_are_stripped(self):
        got = tr.reasoning_text("<|channel>thought\nit looks bright<channel|>")
        assert got == "it looks bright"

    def test_it_survives_a_missing_opener(self):
        """In `on` mode the model writes the opener, so it may not be there."""
        assert tr.reasoning_text("just words<channel|>") == "just words"

    def test_it_survives_a_missing_closer(self):
        assert tr.reasoning_text("<|channel>thought\nhalf a t") == "half a t"

    def test_an_empty_thought_is_empty(self):
        assert tr.reasoning_text("<|channel>thought\n<channel|>") == ""


class TestPromptForms:
    """The 2 forms must really differ, and in the documented way."""

    class FakeTok:
        def apply_chat_template(self, messages, **kw):
            return ("<|turn>model\n" if kw.get("enable_thinking")
                    else "<|turn>model\n<|channel>thought\n<channel|>")

    class FakeProc:
        def __init__(self):
            self.tokenizer = TestPromptForms.FakeTok()

    def test_thinking_off_closes_the_channel_at_once(self):
        got = sg.render_single_text(self.FakeProc(), "t", "subway_safety",
                                    force_prefix="")
        assert got.endswith(tr.CHANNEL_CLOSE)

    def test_thinking_on_leaves_the_channel_open(self):
        got = sg.render_single_text(self.FakeProc(), "t", "subway_safety",
                                    force_prefix="", enable_thinking=True)
        assert not got.endswith(tr.CHANNEL_CLOSE)

    def test_the_two_forms_are_not_the_same_prompt(self):
        p = self.FakeProc()
        off = sg.render_single_text(p, "t", "subway_safety", force_prefix="")
        on = sg.render_single_text(p, "t", "subway_safety", force_prefix="",
                                   enable_thinking=True)
        assert off != on

    def test_the_forced_prefix_still_goes_on_the_end_when_asked(self):
        got = sg.render_single_text(self.FakeProc(), "t", "subway_safety")
        assert got.endswith(sg.FORCE_PREFIX)

    def test_a_thinking_prompt_must_not_carry_the_prefix(self):
        """It would land inside the open thought, not at the answer."""
        got = sg.render_single_text(self.FakeProc(), "t", "subway_safety",
                                    force_prefix="", enable_thinking=True)
        assert sg.FORCE_PREFIX not in got


class TestBuildSingleInputs:
    class FakeTok:
        def apply_chat_template(self, messages, **kw):
            return "<PROMPT>"

    class FakeProc:
        def __init__(self):
            self.tokenizer = TestBuildSingleInputs.FakeTok()
            self.seen = None

        def __call__(self, text=None, images=None, **kw):
            self.seen = text[0]
            return self

        def to(self, device):
            return self

    def test_a_caller_can_pass_its_own_text(self):
        """This is how a thinking read re-feeds its own trace."""
        proc = self.FakeProc()
        sg.build_single_inputs(proc, "t", object(), "subway_safety",
                               text="MY OWN PROMPT", device="cpu")
        assert proc.seen == "MY OWN PROMPT"

    def test_passing_text_overrides_the_prefix(self):
        proc = self.FakeProc()
        sg.build_single_inputs(proc, "t", object(), "subway_safety",
                               text="RAW", device="cpu")
        assert sg.FORCE_PREFIX not in proc.seen


class TestContract:
    def test_a_masked_read_demands_the_image_token_id(self):
        """Without it the 2 passes could mask different things."""
        with pytest.raises(ValueError, match="image_token_id"):
            tr.thinking_read(None, None, None, "t", None, "subway_safety",
                             1, 2, drop_patches=[0])

    def test_the_trace_cap_is_finite(self):
        assert 0 < tr.MAX_NEW <= 1024
