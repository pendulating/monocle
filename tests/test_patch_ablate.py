"""Input-level patch ablation. See monocle/patch_ablate.py."""
import numpy as np
import pytest
import torch

from monocle import patch_ablate as ab

IMG = 258880


def inputs(n_patches=4, n_slots=6, n_text=3):
    """A tiny stand-in: text, then the image tokens, then text."""
    ids = torch.tensor([[7] + [IMG] * n_patches + [8] * (n_text - 1)])
    length = ids.shape[1]
    pos = torch.full((1, n_slots, 2), -1)
    for i in range(n_patches):
        pos[0, i] = torch.tensor([i, 0])
    return {
        "input_ids": ids,
        "attention_mask": torch.ones(1, length, dtype=torch.long),
        "mm_token_type_ids": torch.tensor(
            [[0] + [1] * n_patches + [0] * (n_text - 1)]),
        "pixel_values": torch.arange(
            float(n_slots * 5)).reshape(1, n_slots, 5),
        "image_position_ids": pos,
    }


def slots(n_patches=4):
    return list(range(1, 1 + n_patches))


class TestSplitKeys:
    def test_sorts_the_two_families(self):
        seq, slot, other = ab.split_keys(inputs())
        assert set(seq) == {"input_ids", "attention_mask", "mm_token_type_ids"}
        assert set(slot) == {"pixel_values", "image_position_ids"}
        assert other == []

    def test_rejects_an_unknown_tensor(self):
        """A new processor key must not pass through unedited."""
        d = inputs(); d["surprise"] = torch.zeros(1, 4)
        with pytest.raises(ValueError, match="unknown tensor"):
            ab.split_keys(d)

    def test_keeps_a_non_tensor(self):
        d = inputs(); d["note"] = "hello"
        assert ab.split_keys(d)[2] == ["note"]

    def test_demands_the_two_it_edits(self):
        d = inputs(); del d["image_position_ids"]
        with pytest.raises(ValueError, match="image_position_ids"):
            ab.split_keys(d)


class TestAblateRow:
    def test_removes_one_image_token(self):
        out = ab.ablate_row(inputs(), slots(), [1])
        assert int((out["input_ids"] == IMG).sum()) == 3

    def test_removes_the_right_one(self):
        """Patch 1 sits at sequence position 2, so token 2 must go."""
        out = ab.ablate_row(inputs(), slots(), [1])
        assert out["input_ids"][0].tolist() == [7, IMG, IMG, IMG, 8, 8]

    def test_marks_the_slot_as_padding(self):
        out = ab.ablate_row(inputs(), slots(), [1])
        assert out["image_position_ids"][0, 1].tolist() == [-1, -1]

    def test_leaves_the_other_slots_alone(self):
        out = ab.ablate_row(inputs(), slots(), [1])
        assert out["image_position_ids"][0, 2].tolist() == [2, 0]

    def test_edits_every_sequence_tensor(self):
        out = ab.ablate_row(inputs(), slots(), [1])
        n = out["input_ids"].shape[1]
        assert out["attention_mask"].shape[1] == n
        assert out["mm_token_type_ids"].shape[1] == n

    def test_does_not_mutate_the_caller(self):
        d = inputs()
        ab.ablate_row(d, slots(), [0, 1])
        assert int((d["input_ids"] == IMG).sum()) == 4
        assert d["image_position_ids"][0, 0].tolist() == [0, 0]

    def test_drops_a_whole_group(self):
        out = ab.ablate_row(inputs(), slots(), [0, 2])
        assert int((out["input_ids"] == IMG).sum()) == 2
        assert out["image_position_ids"][0, 0].tolist() == [-1, -1]
        assert out["image_position_ids"][0, 2].tolist() == [-1, -1]

    def test_an_empty_ablation_changes_nothing(self):
        out = ab.ablate_row(inputs(), slots(), [])
        assert torch.equal(out["input_ids"], inputs()["input_ids"])
        assert torch.equal(out["image_position_ids"],
                           inputs()["image_position_ids"])

    def test_rejects_a_patch_out_of_range(self):
        with pytest.raises(ValueError, match="outside 0..3"):
            ab.ablate_row(inputs(), slots(), [9])

    def test_rejects_a_batch(self):
        d = inputs()
        d = {k: v.repeat(2, *([1] * (v.ndim - 1))) for k, v in d.items()}
        with pytest.raises(ValueError, match="needs 1 row"):
            ab.ablate_row(d, slots(), [0])

    def test_a_repeated_patch_counts_once(self):
        out = ab.ablate_row(inputs(), slots(), [1, 1])
        assert int((out["input_ids"] == IMG).sum()) == 3


class TestAlignment:
    def test_a_clean_ablation_agrees(self):
        out = ab.ablate_row(inputs(), slots(), [1])
        assert ab.check_alignment(out, IMG) == 3

    def test_catches_a_half_done_edit(self):
        """Mark the slot but keep the token: the model would raise later."""
        d = inputs()
        d["image_position_ids"][0, 1] = -1
        with pytest.raises(ValueError, match="must always agree"):
            ab.check_alignment(d, IMG)


class TestAblationBatch:
    def test_one_row_for_each_group(self):
        out = ab.ablation_batch(inputs(), slots(), [[0], [1], [2]], IMG)
        assert out["input_ids"].shape[0] == 3

    def test_each_row_loses_its_own_patch(self):
        out = ab.ablation_batch(inputs(), slots(), [[0], [3]], IMG)
        assert out["image_position_ids"][0, 0].tolist() == [-1, -1]
        assert out["image_position_ids"][0, 3].tolist() == [3, 0]
        assert out["image_position_ids"][1, 3].tolist() == [-1, -1]
        assert out["image_position_ids"][1, 0].tolist() == [0, 0]

    def test_the_image_tensors_come_along(self):
        out = ab.ablation_batch(inputs(), slots(), [[0], [1]], IMG)
        assert out["pixel_values"].shape[0] == 2

    def test_rejects_mixed_group_sizes(self):
        """Rows of 2 lengths would need a pad inside the image block."""
        with pytest.raises(ValueError, match="1 group size"):
            ab.ablation_batch(inputs(), slots(), [[0], [1, 2]], IMG)

    def test_rejects_an_empty_batch(self):
        with pytest.raises(ValueError, match="no groups"):
            ab.ablation_batch(inputs(), slots(), [], IMG)

    def test_the_null_arm_is_a_real_row(self):
        out = ab.ablation_batch(inputs(), slots(), [[]], IMG)
        assert out["input_ids"].shape == inputs()["input_ids"].shape

    def test_the_all_arm_leaves_no_patch(self):
        out = ab.ablation_batch(inputs(), slots(), [list(range(4))], IMG)
        assert int((out["input_ids"] == IMG).sum()) == 0
        assert int((out["image_position_ids"] != -1).all(-1).sum()) == 0


class TestSubstitute:
    def donor(self, n_slots=6, width=5):
        """Offset by 1000 so a donor row can never be mistaken for a target
        row: the target's own pixels are an arange from 0."""
        return 1000.0 + torch.arange(float(n_slots * width)).reshape(
            n_slots, width)

    def test_the_chosen_slot_takes_the_donor_pixels(self):
        out = ab.substitute_row(inputs(), self.donor(), [1])
        assert out["pixel_values"][0, 1].tolist() == [1005.0, 1006, 1007,
                                                      1008, 1009]

    def test_the_other_slots_are_untouched(self):
        out = ab.substitute_row(inputs(), self.donor(), [1])
        assert out["pixel_values"][0, 0].tolist() == [0.0, 1, 2, 3, 4]

    def test_no_token_is_removed(self):
        """A substitution keeps the slot, so the sequence never shortens."""
        d = inputs()
        out = ab.substitute_row(d, self.donor(), [0, 1, 2])
        assert out["input_ids"].shape == d["input_ids"].shape
        assert int((out["input_ids"] == IMG).sum()) == 4

    def test_the_position_ids_stay_real(self):
        """The square keeps its coordinates; only the content changes."""
        out = ab.substitute_row(inputs(), self.donor(), [1])
        assert out["image_position_ids"][0, 1].tolist() == [1, 0]

    def test_it_does_not_mutate_the_caller(self):
        d = inputs()
        before = d["pixel_values"].clone()
        ab.substitute_row(d, self.donor(), [0, 1])
        assert torch.equal(d["pixel_values"], before)

    def test_an_empty_substitution_changes_nothing(self):
        d = inputs()
        out = ab.substitute_row(d, self.donor(), [])
        assert torch.equal(out["pixel_values"], d["pixel_values"])

    def test_rejects_a_slot_out_of_range(self):
        with pytest.raises(ValueError, match="outside 0..5"):
            ab.substitute_row(inputs(), self.donor(), [9])

    def test_rejects_a_donor_of_the_wrong_width(self):
        with pytest.raises(ValueError, match="donor rows are"):
            ab.substitute_row(inputs(), torch.zeros(6, 4), [1])

    def test_a_batch_may_mix_set_sizes(self):
        """Unlike an ablation: no row changes length, so no pad is needed."""
        out = ab.substitution_batch(inputs(), self.donor(), [[0], [1, 2]], IMG)
        assert out["pixel_values"].shape[0] == 2

    def test_each_row_gets_its_own_set(self):
        out = ab.substitution_batch(inputs(), self.donor(), [[0], [1]], IMG)
        assert out["pixel_values"][0, 0].tolist()[0] == 1000.0
        assert out["pixel_values"][1, 0].tolist()[0] == 0.0
        assert out["pixel_values"][1, 1].tolist()[0] == 1005.0

    def test_rejects_an_empty_batch(self):
        with pytest.raises(ValueError, match="no groups"):
            ab.substitution_batch(inputs(), self.donor(), [], IMG)


class TestLocalBaseline:
    """Every chunk carries a no-op row, because batch size moves `c`.

    Measured on gemma-4-12b: the same input with no intervention gives a `c`
    that moves by up to 0.67 logits when the batch size changes, while 16
    identical rows inside one batch agree exactly. So a map must subtract the
    zero of its OWN batch, not a `c0` measured alone.
    """

    def test_the_substitution_map_uses_the_same_discipline(self):
        import inspect
        src = inspect.getsource(ab.substitution_map)
        assert "while len(chunk) < batch" in src and "[[]] * batch" in src

    def test_an_empty_ablation_really_is_a_no_op(self):
        """The baseline row must not perturb anything itself."""
        d = inputs()
        out = ab.ablate_row(d, slots(), [])
        assert torch.equal(out["input_ids"], d["input_ids"])
        assert torch.equal(out["image_position_ids"], d["image_position_ids"])

    def test_an_empty_substitution_really_is_a_no_op(self):
        d = inputs()
        out = ab.substitute_row(d, torch.ones(6, 5) * 99, [])
        assert torch.equal(out["pixel_values"], d["pixel_values"])

    def test_a_no_op_cannot_share_a_batch_with_an_ablation(self):
        """This is why the zero is measured in its own batch, not inline.

        An ablation shortens the sequence, so the 2 rows cannot be stacked.
        The first attempt at the fix put `[[]] + chunk` in one batch, and this
        rule caught it before any job ran.
        """
        with pytest.raises(ValueError, match="1 group size"):
            ab.ablation_batch(inputs(), slots(), [[], [1]], IMG)

    def test_the_map_pads_every_chunk_to_one_size(self):
        import inspect
        src = inspect.getsource(ab.ablation_map)
        assert "while len(chunk) < batch" in src
        assert "[[]] * batch" in src


class TestControlBatch:
    """The whole must sit on the SAME zero as the parts.

    `ablation_map` reads every cell against a no-op row in a batch of
    `batch`. In bf16 the same row gives a different `c` at a different batch
    size, so a whole measured alone sits on another reference. The sum of 256
    cells then carries 256 copies of that offset.
    """

    OFFSET = 0.5   # what a batch over 1 row adds, in this stand-in
    WHOLE = 3.0    # what the picture is really worth

    def fake_contrast(self, monkeypatch):
        """`c` that depends on the batch size AND on what is removed."""

        full = len(slots())   # the patch count BEFORE any ablation

        def contrast(model, lens, rows, pos_id, neg_id):
            n = int(rows["input_ids"].shape[0])
            kept = (rows["image_position_ids"] != -1).all(-1).sum(-1)
            out = []
            for i in range(n):
                gone = 1.0 - float(kept[i]) / full
                out.append(10.0 - self.WHOLE * gone
                           + (self.OFFSET if n > 1 else 0.0))
            return np.array(out, dtype=np.float32)

        monkeypatch.setattr(ab, "contrast", contrast)

    def run(self, monkeypatch, batch):
        self.fake_contrast(monkeypatch)
        return ab.controls(None, None, inputs(), slots(), 10.0, 1, 2,
                           image_token_id=IMG, batch=batch)

    def test_the_whole_shares_the_maps_zero(self, monkeypatch):
        """Not `c0 - c_all`, which would carry the batch offset."""
        ctl = self.run(monkeypatch, batch=8)
        assert ctl["image_worth"] == pytest.approx(self.WHOLE)
        assert ctl["image_worth"] != pytest.approx(self.WHOLE - self.OFFSET)

    def test_batch_one_is_unchanged(self, monkeypatch):
        """The other callers keep the behaviour they had."""
        ctl = self.run(monkeypatch, batch=1)
        assert ctl["image_worth"] == pytest.approx(self.WHOLE)
        assert ctl["batch_offset"] == pytest.approx(0.0)

    def test_the_offset_is_recorded(self, monkeypatch):
        ctl = self.run(monkeypatch, batch=8)
        assert ctl["batch_offset"] == pytest.approx(self.OFFSET)

    def test_the_null_arm_stays_at_batch_one(self, monkeypatch):
        """It is compared against `c0`, which the caller reads alone."""
        ctl = self.run(monkeypatch, batch=8)
        assert ctl["null_error"] == pytest.approx(0.0)
        assert ctl["null_ok"]

    def test_the_worth_gate_reads_the_corrected_whole(self, monkeypatch):
        self.fake_contrast(monkeypatch)
        monkeypatch.setattr(ab, "WORTH_FLOOR", self.WHOLE - 0.25)
        assert ab.controls(None, None, inputs(), slots(), 10.0, 1, 2,
                           image_token_id=IMG, batch=8)["worth_ok"]

    def test_the_zero_is_the_one_the_map_uses(self, monkeypatch):
        """`controls` and `ablation_map` must build the same no-op batch."""
        import inspect
        assert "[[]] * batch" in inspect.getsource(ab.controls)
        assert "[[]] * batch" in inspect.getsource(ab.ablation_map)


class TestGateRebuild:
    def base(self, **kw):
        d = {"c0": 5.0, "c_none": 5.0, "c_all": 1.0, "null_error": 0.0,
             "image_worth": 4.0, "null_ok": True, "worth_ok": True}
        d.update(kw)
        return d

    def test_passes_a_clean_rebuild(self):
        ab.gate_rebuild(self.base())

    def test_stops_when_the_rebuild_moves_c(self):
        with pytest.raises(SystemExit, match="null arm"):
            ab.gate_rebuild(self.base(null_ok=False, null_error=0.02))

    def test_a_weak_image_does_not_stop_the_run(self):
        """A photograph the model barely reads is data, not a fault."""
        ab.gate_rebuild(self.base(worth_ok=False, image_worth=0.9))

    def test_the_null_tolerance_is_tighter_than_the_bf16_grid(self):
        """fp32_contrast exists because bf16 quantizes to about 0.125."""
        assert ab.NULL_TOL < 0.125


class TestGateBattery:
    def test_passes_when_most_images_carry_the_judgment(self):
        ab.gate_battery([10.0, 8.0, 0.4, 12.0])

    def test_one_weak_image_is_allowed(self):
        ab.gate_battery([10.0, 0.1])

    def test_stops_when_almost_every_image_is_worth_nothing(self):
        with pytest.raises(SystemExit, match="does not reach the model"):
            ab.gate_battery([0.1, 0.2, 0.0, 9.0])

    def test_a_negative_worth_still_counts(self):
        """An image that pushes toward Dangerous is a strong image."""
        ab.gate_battery([-9.0, -8.0, 0.1])

    def test_stops_on_an_empty_battery(self):
        with pytest.raises(SystemExit, match="no image was measured"):
            ab.gate_battery([])


class TestControlsShareTheMapBatch:
    """The whole and the parts must sit on ONE reference.

    `ablation_map` reads every cell against a no-op row in a batch of
    `batch`. In bf16 the same row gives a different `c` at a different batch
    size, so a whole measured alone carries an offset the parts do not. The
    sum of 256 cells then holds 256 copies of it, and `parts_over_whole`
    compares 2 quantities that were never on one scale.
    """

    OFFSET = 0.5      # what a batch over 1 row adds, in this stand-in
    WORTH = 3.0       # what the picture is really worth

    def fake_contrast(self, monkeypatch):
        """`c` depends on the batch size AND on how much was removed."""
        seen = []

        def contrast(model, lens_model, rows, pos_id, neg_id):
            n_rows = int(rows["input_ids"].shape[0])
            kept = int((rows["image_position_ids"][0] != -1).all(-1).sum())
            seen.append((n_rows, kept))
            gone = 1.0 - kept / 4.0
            c = 10.0 - self.WORTH * gone
            if n_rows > 1:
                c += self.OFFSET
            return np.full(n_rows, c, dtype=np.float32)

        monkeypatch.setattr(ab, "contrast", contrast)
        return seen

    def run(self, monkeypatch, batch):
        seen = self.fake_contrast(monkeypatch)
        ctl = ab.controls(None, None, inputs(), slots(), 10.0, 1, 2,
                          image_token_id=IMG, batch=batch)
        return ctl, seen

    def test_the_whole_is_free_of_the_batch_offset(self, monkeypatch):
        ctl, _ = self.run(monkeypatch, batch=8)
        assert ctl["image_worth"] == pytest.approx(self.WORTH)

    def test_the_old_reference_would_have_been_wrong(self, monkeypatch):
        """`c0 - c_all` mixes a batch-1 zero with a batch-N whole."""
        ctl, _ = self.run(monkeypatch, batch=8)
        stale = ctl["c0"] - ctl["c_all"]
        assert stale == pytest.approx(self.WORTH - self.OFFSET)
        assert ctl["image_worth"] != pytest.approx(stale)

    def test_the_zero_and_the_whole_are_read_at_the_same_size(
            self, monkeypatch):
        _, seen = self.run(monkeypatch, batch=8)
        sizes = {n for n, kept in seen if n > 1}
        assert sizes == {8}

    def test_the_rebuild_check_stays_at_batch_one(self, monkeypatch):
        """`c_none` is compared against `c0`, which the caller reads alone."""
        ctl, seen = self.run(monkeypatch, batch=8)
        assert seen[0] == (1, 4)
        assert ctl["null_error"] == pytest.approx(0.0)
        assert ctl["null_ok"]

    def test_it_records_the_offset_it_removed(self, monkeypatch):
        ctl, _ = self.run(monkeypatch, batch=8)
        assert ctl["batch_offset"] == pytest.approx(self.OFFSET)

    def test_batch_one_leaves_every_arm_on_one_reference(self, monkeypatch):
        ctl, _ = self.run(monkeypatch, batch=1)
        assert ctl["batch_offset"] == pytest.approx(0.0)
        assert ctl["image_worth"] == pytest.approx(self.WORTH)

    def test_the_weak_image_gate_reads_the_corrected_worth(self, monkeypatch):
        """A gate on the stale worth would judge a different number."""
        ctl, _ = self.run(monkeypatch, batch=8)
        assert ctl["worth_ok"] == (abs(ctl["image_worth"]) >= ab.WORTH_FLOOR)

    def test_the_map_and_the_controls_agree_on_the_zero(self, monkeypatch):
        """One end-to-end check: a 1-cell map must equal the whole.

        A `group` that covers the picture makes the map's only cell the
        whole-image ablation. It must reproduce `image_worth` exactly.
        """
        self.fake_contrast(monkeypatch)
        d, sl = inputs(), slots()
        ctl = ab.controls(None, None, d, sl, 10.0, 1, 2,
                          image_token_id=IMG, batch=8)
        cell = -ab.ablation_map(None, None, d, sl, 10.0, 1, 2,
                                batch=8, group=2, image_token_id=IMG)
        assert cell.size == 1
        assert float(cell[0]) == pytest.approx(ctl["image_worth"])


class TestStackRows:
    """Stacking one edit of MANY photographs, which is what batches a run."""

    def rows(self, n=3, keep=2):
        out = []
        for i in range(n):
            d = inputs()
            d["pixel_values"] = d["pixel_values"] + float(i)
            out.append(ab.ablate_row(d, slots(), list(range(keep))))
        return out

    def test_it_stacks_into_one_batch(self):
        got = ab.stack_rows(self.rows(3))
        assert int(got["input_ids"].shape[0]) == 3

    def test_each_photograph_keeps_its_own_pixels(self):
        got = ab.stack_rows(self.rows(3))
        assert not torch.equal(got["pixel_values"][0], got["pixel_values"][1])

    def test_the_sequence_length_is_unchanged(self):
        one = self.rows(1)[0]
        got = ab.stack_rows(self.rows(3))
        assert got["input_ids"].shape[1] == one["input_ids"].shape[1]

    def test_it_refuses_rows_of_different_lengths(self):
        """A pad inside a bidirectional image block is never safe."""
        mixed = [ab.ablate_row(inputs(), slots(), [0]),
                 ab.ablate_row(inputs(), slots(), [0, 1])]
        with pytest.raises(ValueError, match="Stacking would need a pad"):
            ab.stack_rows(mixed)

    def test_it_refuses_an_empty_list(self):
        with pytest.raises(ValueError, match="no rows"):
            ab.stack_rows([])

    def test_it_refuses_an_already_batched_row(self):
        batched = ab.ablation_batch(inputs(), slots(), [[0], [1]], IMG)
        with pytest.raises(ValueError, match="single-row"):
            ab.stack_rows([batched])

    def test_rows_that_keep_the_same_count_always_agree(self):
        """This is the rule the caller groups by."""
        a = ab.ablate_row(inputs(), slots(), [0, 1])
        b = ab.ablate_row(inputs(), slots(), [2, 3])
        ab.stack_rows([a, b])
