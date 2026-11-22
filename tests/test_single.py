"""The single-image lens path. See research/monocle/single.py."""
import pytest

from research.monocle import canonical
from research.monocle import single as sg


class StubTokenizer:
    TABLE = {"Safe": [1], "Neutral": [2], "Dangerous": [3],
             "Good": [4], "Poor": [5], "Bad": [6, 7]}

    def encode(self, text, add_special_tokens=False):
        return list(self.TABLE[text])


class TestCoverage:
    def test_every_canonical_case_has_a_rating(self):
        assert set(sg.CASE_RATING) == set(canonical.CASES)

    def test_rejects_an_unknown_case(self):
        with pytest.raises(ValueError, match="unknown case"):
            sg.rating("nope")

    def test_rating_is_a_copy(self):
        r = sg.rating("subway_safety")
        r["positive"] = "MUTATED"
        assert sg.CASE_RATING["subway_safety"]["positive"] == "Safe"


class TestPrompt:
    def test_names_the_subject_and_asks_the_question(self):
        t = sg.prompt_text("subway_safety")
        assert "subway station entrances" in t
        assert "How safe does this station entrance look?" in t

    def test_lists_both_labels(self):
        t = sg.prompt_text("subway_safety")
        for lab in ("Safe", "Dangerous"):
            assert f"- {lab}" in t

    def test_offers_no_middle_label(self):
        """A Neutral midpoint took p(Neutral) to 1.000 on every image."""
        for case in sg.CASE_RATING:
            t = sg.prompt_text(case)
            assert "Neutral" not in t
            assert "middle" not in sg.CASE_RATING[case]

    def test_forces_a_choice(self):
        assert "must choose" in sg.prompt_text("subway_safety")

    def test_no_interpolation_survives(self):
        """A leaked ${...} would reach the model verbatim."""
        for case in canonical.CASES:
            assert "${" not in sg.prompt_text(case)

    def test_carries_no_pairwise_language(self):
        for case in canonical.CASES:
            t = sg.prompt_text(case)
            for banned in ("Image A", "Image B", "Pair ID", "MuchMore"):
                assert banned not in t

    def test_overrides_apply(self):
        t = sg.prompt_text("road_quality", {"positive": "Smooth"})
        assert "- Smooth" in t and "- Good" not in t


class TestMessages:
    def test_image_comes_before_the_text(self):
        content = sg.build_messages("subway_safety")[0]["content"]
        assert content[0]["type"] == "image"
        assert content[1]["type"] == "text"

    def test_is_a_single_user_turn(self):
        m = sg.build_messages("libraries")
        assert len(m) == 1 and m[0]["role"] == "user"

    def test_holds_exactly_one_image(self):
        content = sg.build_messages("schools")[0]["content"]
        assert sum(c["type"] == "image" for c in content) == 1


class TestContrast:
    def test_returns_the_two_poles(self):
        assert sg.contrast_ids(StubTokenizer(), "subway_safety") == (1, 3)

    def test_every_case_has_exactly_two_labels(self):
        for case, r in sg.CASE_RATING.items():
            assert {"subject", "question", "positive", "negative"} == set(r)

    def test_quality_cases_share_a_direction(self):
        tk = StubTokenizer()
        assert sg.contrast_ids(tk, "libraries") == sg.contrast_ids(
            tk, "road_quality") == (4, 5)

    def test_rejects_a_multi_token_label(self):
        with pytest.raises(ValueError, match="not 1 token"):
            sg.contrast_ids(StubTokenizer(), "libraries",
                            {"negative": "Bad"})

    def test_uses_the_same_forced_prefix_as_the_pairwise_path(self):
        assert sg.FORCE_PREFIX == canonical.FORCE_PREFIX


class TestForcedPrefix:
    """The prompt must end where the model emits the LABEL.

    Without the forced JSON prefix the last position is the end of the
    generation prompt, where the natural next token is `{`. A contrast read
    there is a disposition, not the answer. `force_prefix` was accepted and
    then dropped on the floor for a while; these tests keep that from
    happening again in silence.
    """

    class FakeTok:
        def apply_chat_template(self, messages, **kw):
            return "<PROMPT>"

    class FakeProc:
        def __init__(self):
            self.tokenizer = TestForcedPrefix.FakeTok()
            self.seen = None

        def __call__(self, text=None, images=None, **kw):
            self.seen = text[0]
            return self

        def to(self, device):
            return self

    def test_the_prompt_ends_with_the_forced_prefix(self):
        proc = self.FakeProc()
        sg.build_single_inputs(proc, "tmpl", object(), "subway_safety",
                               device="cpu")
        assert proc.seen.endswith(sg.FORCE_PREFIX)

    def test_the_prefix_is_the_canonical_one(self):
        assert sg.FORCE_PREFIX == canonical.FORCE_PREFIX
        assert sg.FORCE_PREFIX.endswith('"answer": "')

    def test_it_can_be_switched_off_on_purpose(self):
        proc = self.FakeProc()
        sg.build_single_inputs(proc, "tmpl", object(), "subway_safety",
                               force_prefix="", device="cpu")
        assert proc.seen == "<PROMPT>"

    def test_the_prefix_is_appended_not_inserted(self):
        proc = self.FakeProc()
        sg.build_single_inputs(proc, "tmpl", object(), "subway_safety",
                               device="cpu")
        assert proc.seen == "<PROMPT>" + sg.FORCE_PREFIX


class TestGeometry:
    """The single path must build exactly ONE bidirectional island."""

    def frame(self, runs):
        import torch
        ids = []
        for n in runs:
            ids += [0, 0] + [99] * n
        return {"input_ids": torch.tensor([ids + [0]])}

    def test_accepts_one_block(self):
        from research.monocle import safety_workspace as sw
        got = sw.image_token_blocks(self.frame([256]), 99, expect=1)
        assert len(got) == 1 and got[0].numel() == 256

    def test_still_defaults_to_the_pairwise_contract(self):
        """The pairwise path must not change when the default is used."""
        from research.monocle import safety_workspace as sw
        with pytest.raises(RuntimeError, match="expected exactly 2"):
            sw.image_token_blocks(self.frame([256]), 99)
        assert len(sw.image_token_blocks(self.frame([256, 256]), 99)) == 2

    def test_rejects_two_blocks_when_one_is_expected(self):
        from research.monocle import safety_workspace as sw
        with pytest.raises(RuntimeError, match="expected exactly 1"):
            sw.image_token_blocks(self.frame([256, 256]), 99, expect=1)

    def test_rejects_ragged_blocks(self):
        from research.monocle import safety_workspace as sw
        with pytest.raises(RuntimeError, match="unequal length"):
            sw.image_token_blocks(self.frame([256, 200]), 99)


class TestSingleRowIds:
    """The id must name the PHOTOGRAPH. `_A`/`_B` reads as a pair."""

    def rows(self, monkeypatch, pairs):
        import research.monocle.safety_workspace as sw
        monkeypatch.setattr(sw, "load_pairs",
                            lambda n, c, k, seed=777: pairs)
        monkeypatch.setattr(
            canonical, "presented_images",
            lambda r: (r["presented_left_path"], r["presented_right_path"]))
        return sg.load_single_rows(10, "subway_safety")

    def test_a_cyclomedia_face_carries_its_unit(self):
        """A bare stem is only `B`, and 4 units all write `B.png`."""
        assert sg.photograph_id(
            "/share/ju/cyclomedia/raw/bronx_2025_1k/W0E96/W0E966FV/faces/B.jpg"
        ) == "W0E966FV_B"

    def test_two_faces_of_two_units_do_not_collide(self):
        a = sg.photograph_id("/r/W0E9/W0E966FV/faces/B.jpg")
        b = sg.photograph_id("/r/W0EF/W0EF5WAT/faces/B.jpg")
        assert a != b

    def test_any_other_layout_keeps_the_stem(self):
        assert sg.photograph_id("/x/img_aaa.jpg") == "img_aaa"

    def test_the_rows_use_it(self, monkeypatch):
        rows = self.rows(monkeypatch, [
            {"pair_id": "p1",
             "presented_left_path": "/r/W0E9/W0E966FV/faces/B.jpg",
             "presented_right_path": "/r/W0EF/W0EF5WAT/faces/B.jpg"}])
        assert [r["image_id"] for r in rows] == ["W0E966FV_B", "W0EF5WAT_B"]

    def test_id_is_the_image_not_the_slot(self, monkeypatch):
        rows = self.rows(monkeypatch, [
            {"pair_id": "p1", "presented_left_path": "/x/img_aaa.jpg",
             "presented_right_path": "/x/img_bbb.jpg"}])
        assert [r["image_id"] for r in rows] == ["img_aaa", "img_bbb"]

    def test_no_id_implies_a_pairing(self, monkeypatch):
        rows = self.rows(monkeypatch, [
            {"pair_id": "p1", "presented_left_path": "/x/img_aaa.jpg",
             "presented_right_path": "/x/img_bbb.jpg"}])
        for r in rows:
            assert not r["image_id"].endswith(("_A", "_B"))

    def test_keeps_provenance_for_tracing(self, monkeypatch):
        rows = self.rows(monkeypatch, [
            {"pair_id": "p1", "presented_left_path": "/x/img_aaa.jpg",
             "presented_right_path": "/x/img_bbb.jpg"}])
        assert rows[0]["pair_id"] == "p1" and rows[0]["slot"] == "A"

    def test_drops_a_repeated_photograph(self, monkeypatch):
        """A donor that is its own target would read as an empty map."""
        rows = self.rows(monkeypatch, [
            {"pair_id": "p1", "presented_left_path": "/x/img_aaa.jpg",
             "presented_right_path": "/x/img_bbb.jpg"},
            {"pair_id": "p2", "presented_left_path": "/x/img_aaa.jpg",
             "presented_right_path": "/x/img_ccc.jpg"}])
        ids = [r["image_id"] for r in rows]
        assert ids == ["img_aaa", "img_bbb", "img_ccc"]
        assert len(ids) == len(set(ids))

    def test_stops_at_the_quota(self, monkeypatch):
        import research.monocle.safety_workspace as sw
        pairs = [{"pair_id": f"p{i}",
                  "presented_left_path": f"/x/l{i}.jpg",
                  "presented_right_path": f"/x/r{i}.jpg"} for i in range(20)]
        monkeypatch.setattr(sw, "load_pairs", lambda n, c, k, seed=777: pairs)
        monkeypatch.setattr(
            canonical, "presented_images",
            lambda r: (r["presented_left_path"], r["presented_right_path"]))
        assert len(sg.load_single_rows(7, "subway_safety")) == 7
