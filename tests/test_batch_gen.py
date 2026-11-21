"""Batched generation over many rows. See monocle/batch_gen.py."""
import pytest
import torch

from monocle import batch_gen as bg
from monocle import patch_ablate as ab

IMG = 258880


def inputs(n_patches=8, n_slots=10, n_text=3):
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


def slots(n_patches=8):
    return list(range(1, 1 + n_patches))


class TestGroupByLength:
    def test_rows_that_drop_the_same_count_share_a_group(self):
        jobs = [{"n_drop": 4}, {"n_drop": 4}, {"n_drop": 0}]
        got = bg.group_by_length(jobs)
        assert got == {4: [0, 1], 0: [2]}

    def test_the_whole_image_arm_stands_alone(self):
        """It drops nothing, so it can never share a length with an arm."""
        jobs = [{"n_drop": 48}] * 3 + [{"n_drop": 0}]
        assert bg.group_by_length(jobs)[0] == [3]

    def test_the_index_order_is_kept(self):
        jobs = [{"n_drop": 1}, {"n_drop": 2}, {"n_drop": 1}]
        assert bg.group_by_length(jobs)[1] == [0, 2]

    def test_an_empty_plan_gives_no_groups(self):
        assert bg.group_by_length([]) == {}

    def test_it_can_group_on_another_key(self):
        jobs = [{"len": 3}, {"len": 3}]
        assert bg.group_by_length(jobs, key="len") == {3: [0, 1]}


class FakeTok:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=False):
        return "|".join(str(int(i)) for i in ids)


class FakeProc:
    def __init__(self):
        self.tokenizer = FakeTok()


class FakeModel:
    """Echoes the row's own first pixel, so a mix-up is visible."""

    def __init__(self):
        self.calls = []

    def generate(self, input_ids=None, pixel_values=None, max_new_tokens=2,
                 **kw):
        self.calls.append(int(input_ids.shape[0]))
        tag = pixel_values[:, 0, 0].long().reshape(-1, 1)
        new = tag.repeat(1, max_new_tokens)
        return torch.cat([input_ids, new], dim=1)


class TestGenerateMany:
    def test_one_string_for_each_row(self):
        rows = [ab.ablate_row(inputs(), slots(), [0]) for _ in range(3)]
        got = bg.generate_many(FakeModel(), FakeProc(),
                               ab.stack_rows(rows), 2)
        assert len(got) == 3

    def test_only_the_new_tokens_are_decoded(self):
        rows = [ab.ablate_row(inputs(), slots(), [0])]
        got = bg.generate_many(FakeModel(), FakeProc(),
                               ab.stack_rows(rows), 3)
        assert len(got[0].split("|")) == 3

    def test_each_row_gets_its_own_generation(self):
        """A batch must not hand row 0's text to row 1."""
        rows = []
        for i in range(3):
            d = inputs()
            d["pixel_values"] = d["pixel_values"] + 100.0 * i
            rows.append(ab.ablate_row(d, slots(), [0]))
        got = bg.generate_many(FakeModel(), FakeProc(),
                               ab.stack_rows(rows), 1)
        assert len(set(got)) == 3


class TestRunBatched:
    def jobs(self, n_arm=5, n_whole=2):
        return ([{"n_drop": 4, "tag": i} for i in range(n_arm)]
                + [{"n_drop": 0, "tag": 100 + i} for i in range(n_whole)])

    def build(self, job):
        d = inputs()
        d["pixel_values"] = d["pixel_values"] + float(job["tag"])
        return ab.ablate_row(d, slots(), list(range(job["n_drop"])))

    def test_every_job_gets_a_result(self):
        jobs = self.jobs()
        got = bg.run_batched(jobs, self.build, FakeModel(), FakeProc(),
                             max_new=1, batch=2)
        assert len(got) == len(jobs) and all(g is not None for g in got)

    def test_the_order_matches_the_jobs(self):
        jobs = self.jobs(n_arm=3, n_whole=0)
        got = bg.run_batched(jobs, self.build, FakeModel(), FakeProc(),
                             max_new=1, batch=8)
        assert got == [str(j["tag"]) for j in jobs]

    def test_lengths_never_share_a_batch(self):
        model = FakeModel()
        bg.run_batched(self.jobs(n_arm=4, n_whole=2), self.build, model,
                       FakeProc(), max_new=1, batch=8)
        # 4 arm rows in one call, 2 whole rows in another. Never 6.
        assert sorted(model.calls) == [2, 4]

    def test_the_batch_cap_is_honoured(self):
        model = FakeModel()
        bg.run_batched(self.jobs(n_arm=5, n_whole=0), self.build, model,
                       FakeProc(), max_new=1, batch=2)
        assert sorted(model.calls) == [1, 2, 2]

    def test_one_row_still_works(self):
        got = bg.run_batched([{"n_drop": 4, "tag": 7}], self.build,
                             FakeModel(), FakeProc(), max_new=1, batch=16)
        assert got == ["7"]

    def test_an_empty_plan_does_nothing(self):
        model = FakeModel()
        assert bg.run_batched([], self.build, model, FakeProc(),
                              max_new=1) == []
        assert model.calls == []


class TestEquivalence:
    def rows(self, n=3):
        out = []
        for i in range(n):
            d = inputs()
            d["pixel_values"] = d["pixel_values"] + float(i)
            out.append(ab.ablate_row(d, slots(), [0]))
        return out

    def test_a_deterministic_model_matches_exactly(self):
        got = bg.equivalence(FakeModel(), FakeProc(), self.rows(), 1)
        assert got["rate"] == 1.0 and got["exact"] == 3

    def test_it_reports_where_a_pair_diverges(self):
        class Drifter(FakeModel):
            def generate(self, input_ids=None, pixel_values=None,
                         max_new_tokens=2, **kw):
                out = FakeModel.generate(self, input_ids=input_ids,
                                         pixel_values=pixel_values,
                                         max_new_tokens=max_new_tokens, **kw)
                if int(input_ids.shape[0]) > 1:      # only inside a batch
                    out[0, -1] += 5
                return out

        got = bg.equivalence(Drifter(), FakeProc(), self.rows(), 1)
        assert got["rate"] < 1.0
        assert got["first_diff"][0] >= 0

    def test_a_match_reports_no_divergence_point(self):
        got = bg.equivalence(FakeModel(), FakeProc(), self.rows(), 1)
        assert got["first_diff"] == [-1, -1, -1]

    def test_it_keeps_both_texts_for_reading(self):
        got = bg.equivalence(FakeModel(), FakeProc(), self.rows(2), 1)
        assert len(got["alone"]) == 2 and len(got["together"]) == 2

    def test_it_times_both_paths(self):
        """The gate must answer "worth it?" as well as "the same?"."""
        got = bg.equivalence(FakeModel(), FakeProc(), self.rows(), 1)
        assert got["t_alone"] > 0.0 and got["t_together"] > 0.0
        assert got["speedup"] == pytest.approx(
            got["t_alone"] / got["t_together"])


class TestNormalise:
    """A batch pads a row that stopped early. That is not a divergence."""

    def test_a_trailing_pad_run_goes(self):
        assert bg.trim_pad("Safe<turn|><eos><eos><eos>") == "Safe<turn|>"

    def test_a_clean_text_is_untouched(self):
        assert bg.trim_pad("Safe<turn|>") == "Safe<turn|>"

    def test_a_pad_in_the_middle_stays(self):
        """Only the tail is an artifact. A pad inside is real output."""
        assert bg.trim_pad("a<eos>b") == "a<eos>b"

    def test_it_cuts_at_the_channel_close(self):
        got = bg.split_at_close("think<channel|>Safe<turn|><eos>")
        assert got == "think<channel|>"

    def test_an_unclosed_trace_is_kept_whole(self):
        """There is no answer position, so the caller must see all of it."""
        assert bg.split_at_close("still thinking") == "still thinking"

    def test_the_two_together_undo_the_batch_artifact(self):
        alone = "reasons.<channel|>Safe<turn|>"
        batch = "reasons.<channel|>Safe<turn|><eos><eos><eos>"
        f = lambda t: bg.trim_pad(bg.split_at_close(t))
        assert f(alone) == f(batch)

    def test_a_default_run_still_trims_the_pad(self):
        class Pad(FakeModel):
            def generate(self, input_ids=None, pixel_values=None,
                         max_new_tokens=2, **kw):
                n = int(input_ids.shape[0])
                new = torch.zeros(n, max_new_tokens, dtype=input_ids.dtype)
                return torch.cat([input_ids, new], dim=1)
        rows = [ab.ablate_row(inputs(), slots(), [0]) for _ in range(2)]
        got = bg.equivalence(Pad(), FakeProc(), rows, 2)
        assert got["rate"] == 1.0


class TestScore:
    """The gate is the reported number, not the words."""

    def rows(self, n=3):
        return [ab.ablate_row(inputs(), slots(), [0]) for _ in range(n)]

    def test_no_score_means_no_dc(self):
        got = bg.equivalence(FakeModel(), FakeProc(), self.rows(), 1)
        assert "dc" not in got

    def test_dc_is_the_batch_minus_the_lone_read(self):
        got = bg.equivalence(FakeModel(), FakeProc(), self.rows(), 1,
                             score=lambda i, t: float(len(t)))
        assert got["dc"] == [b - a for a, b in
                             zip(got["c_alone"], got["c_together"])]

    def test_an_agreeing_pair_gives_zero_dc(self):
        got = bg.equivalence(FakeModel(), FakeProc(), self.rows(), 1,
                             score=lambda i, t: 1.0)
        assert got["dc"] == [0.0, 0.0, 0.0]

    def test_the_score_sees_the_row_index(self):
        seen = []
        bg.equivalence(FakeModel(), FakeProc(), self.rows(2), 1,
                       score=lambda i, t: seen.append(i) or 0.0)
        assert seen == [0, 1, 0, 1]

    def test_it_scores_the_normalised_text(self):
        """Or `dc` would count a pad the read never sees."""
        got = bg.equivalence(FakeModel(), FakeProc(), self.rows(1), 1,
                             normalize=lambda t: "fixed",
                             score=lambda i, t: float(len(t)))
        assert got["c_alone"] == [5.0] and got["c_together"] == [5.0]


class TestContract:
    def test_the_default_batch_is_documented_as_fitting(self):
        assert 1 <= bg.DEFAULT_BATCH <= 64

    def test_stacking_is_what_refuses_a_pad(self):
        """`run_batched` relies on `stack_rows` for that rule."""
        mixed = [ab.ablate_row(inputs(), slots(), [0]),
                 ab.ablate_row(inputs(), slots(), [0, 1])]
        with pytest.raises(ValueError, match="pad"):
            ab.stack_rows(mixed)


class TestRankingGuard:
    """A ranking from the wrong run draws a figure that means nothing."""

    def write(self, root, **meta):
        import json
        import pandas as pd
        base = {"case": "subway_safety", "group": 2, "seed": 20260821,
                "reference": "absent"}
        base.update(meta)
        (root / "summary.json").write_text(json.dumps(base))
        pd.DataFrame([{"image_id": "A", "values": [0.1, 0.2, 0.3, 0.4]}]
                     ).to_parquet(root / "maps.parquet")
        return root

    def load(self, root, **kw):
        from monocle import caption_loop as cl
        args = {"case": "subway_safety", "group": 2, "seed": 20260821}
        args.update(kw)
        return cl.load_ranking(root, args["case"], args["group"],
                               args["seed"])

    def test_a_matching_run_loads(self, tmp_path):
        got = self.load(self.write(tmp_path))
        assert list(got) == ["A"] and len(got["A"]) == 4

    def test_a_donor_run_is_refused(self, tmp_path):
        """The retired method must never rank a figure."""
        with pytest.raises(SystemExit, match="reference is 'donor'"):
            self.load(self.write(tmp_path, reference="donor"))

    def test_another_grid_is_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="group is 4"):
            self.load(self.write(tmp_path, group=4))

    def test_another_seed_is_refused(self, tmp_path):
        """A different seed means different photographs."""
        with pytest.raises(SystemExit, match="seed is 1"):
            self.load(self.write(tmp_path, seed=1))

    def test_another_case_is_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="case is 'schools'"):
            self.load(self.write(tmp_path, case="schools"))

    def test_a_missing_run_says_so(self, tmp_path):
        with pytest.raises(SystemExit, match="--rank-from needs a finished"):
            self.load(tmp_path)

    def test_every_mismatch_is_named_at_once(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            self.load(self.write(tmp_path, group=8, seed=5))
        assert "group" in str(e.value) and "seed" in str(e.value)
