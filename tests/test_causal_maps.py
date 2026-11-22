"""The causal map figure path. See research/monocle/causal_maps.py."""
import numpy as np
import pytest
from PIL import Image

from research.monocle import causal_maps as cm


class TestGrid:
    def test_reads_a_square_grid(self):
        assert cm.grid_of(256) == (16, 16)

    def test_rejects_a_non_square_count(self):
        with pytest.raises(ValueError, match="not square"):
            cm.grid_of(266)


class TestOverlay:
    def img(self):
        return Image.new("RGB", (64, 64), (128, 128, 128))

    def test_returns_an_upscaled_picture(self):
        out = cm.overlay(self.img(), np.zeros(256), 1.0, upscale=2)
        assert out.size == (128, 128)

    def test_uses_the_shared_scale_not_a_local_one(self):
        """Two maps of different strength must not render identically."""
        weak = cm.overlay(self.img(), np.full(256, 0.1), 1.0, grid_lines=False)
        strong = cm.overlay(self.img(), np.full(256, 1.0), 1.0, grid_lines=False)
        assert np.any(np.array(weak) != np.array(strong))

    def test_clips_beyond_the_scale(self):
        a = cm.overlay(self.img(), np.full(256, 2.0), 1.0, grid_lines=False)
        b = cm.overlay(self.img(), np.full(256, 5.0), 1.0, grid_lines=False)
        assert np.array_equal(np.array(a), np.array(b))

    def test_rejects_a_non_square_map(self):
        with pytest.raises(ValueError, match="not square"):
            cm.overlay(self.img(), np.zeros(266), 1.0)


class TestColourDirection:
    """The picture paints the NEGATIVE label coral. See CORAL_SIGN."""

    def img(self):
        return Image.new("RGB", (16, 16), (128, 128, 128))

    def test_the_positive_label_draws_teal(self):
        """A patch that holds up `Safe` must not draw as a danger."""
        out = np.array(cm.overlay(self.img(), np.full(256, 1.0), 1.0,
                                  grid_lines=False, alpha=1.0))
        assert out[..., 2].mean() > out[..., 0].mean()

    def test_the_negative_label_draws_coral(self):
        out = np.array(cm.overlay(self.img(), np.full(256, -1.0), 1.0,
                                  grid_lines=False, alpha=1.0))
        assert out[..., 0].mean() > out[..., 2].mean()

    def test_the_flip_can_be_turned_off(self):
        a = np.array(cm.overlay(self.img(), np.full(256, 1.0), 1.0,
                                grid_lines=False, coral_sign=+1.0))
        b = np.array(cm.overlay(self.img(), np.full(256, -1.0), 1.0,
                                grid_lines=False, coral_sign=cm.CORAL_SIGN))
        assert np.array_equal(a, b)

    def test_the_convention_is_one_constant(self):
        assert cm.CORAL_SIGN == -1.0


class TestColourBar:
    def test_the_labels_reach_the_drawing(self):
        """A bar that says only `change` is read backwards. It happened."""
        a = cm.colour_bar(300, 34, 1.0, teal_label="Safe",
                          coral_label="Dangerous")
        b = cm.colour_bar(300, 34, 1.0, teal_label="Good",
                          coral_label="Poor")
        assert not np.array_equal(np.array(a), np.array(b))

    def test_the_ramp_runs_teal_to_coral(self):
        bar = np.array(cm.colour_bar(300, 34, 1.0, teal_label="Safe",
                                     coral_label="Dangerous"))
        left, right = bar[:8, :20], bar[:8, -20:]
        assert left[..., 2].mean() > left[..., 0].mean()
        assert right[..., 0].mean() > right[..., 2].mean()

    def test_demands_the_labels(self):
        with pytest.raises(TypeError):
            cm.colour_bar(200, 34, 1.0)


class TestColourRamp:
    def test_positive_goes_coral_and_negative_goes_teal(self):
        coral = cm.pa._div_color(1.0)
        teal = cm.pa._div_color(-1.0)
        assert coral[0] > coral[2]      # red channel leads
        assert teal[2] > teal[0]        # blue channel leads

    def test_zero_is_white(self):
        assert cm.pa._div_color(0.0) == (255, 255, 255)


class TestContactSheet:
    def panels(self, n=4):
        return [(f"img{i}", Image.new("RGB", (64, 64))) for i in range(n)]

    def test_lays_panels_out_in_rows(self):
        sheet = cm.contact_sheet(self.panels(6), 1.0, cols=5,
                                 teal_label="Safe",
                                 coral_label="Dangerous")
        assert sheet.size[0] == 5 * 260

    def test_rejects_an_empty_set(self):
        with pytest.raises(ValueError, match="no panels"):
            cm.contact_sheet([], 1.0, teal_label="Safe", coral_label="Dangerous")

    def test_leaves_room_for_the_colour_bar(self):
        one_row = cm.contact_sheet(self.panels(3), 1.0, cols=5,
                                   teal_label="Safe",
                                   coral_label="Dangerous")
        assert one_row.size[1] > 260


class TestGroupedRuns:
    def test_a_fine_run_keeps_the_plain_folder(self, tmp_path):
        assert cm.case_dir_for(tmp_path, "subway_safety", 1).name == \
            "subway_safety"

    def test_a_grouped_run_gets_its_own_folder(self, tmp_path):
        """It must not overwrite the fine maps we compare it against."""
        assert cm.case_dir_for(tmp_path, "subway_safety", 2).name == \
            "subway_safety_g2"

    def test_64_cells_draw_as_an_8x8_grid(self):
        assert cm.grid_of(64) == (8, 8)


class TestReference:
    def test_the_two_references_never_share_a_folder(self, tmp_path):
        a = cm.case_dir_for(tmp_path, "sub", 1, "donor")
        b = cm.case_dir_for(tmp_path, "sub", 1, "absent")
        assert a != b and b.name == "sub_absent"

    def test_group_and_reference_both_show(self, tmp_path):
        assert cm.case_dir_for(tmp_path, "sub", 2, "absent").name == \
            "sub_g2_absent"

    def test_rejects_an_unknown_reference(self, tmp_path):
        with pytest.raises(ValueError, match="reference must be"):
            cm.case_dir_for(tmp_path, "sub", 1, "grey")

    def test_an_absence_run_has_no_layer_to_be_wrong_about(self):
        """The layer gate must not stop a run that edits the input."""
        args = cm.build_parser().parse_args(["--reference", "absent",
                                             "--layer", "43"])
        assert args.reference == "absent"

    def test_the_layer_gate_still_holds_for_a_donor_run(self):
        with pytest.raises(SystemExit, match="above the causal window"):
            cm.main(["--reference", "donor", "--layer", "43"])

    def test_the_default_reads_only_the_target_photograph(self):
        """A bare run must NOT swap in another photograph.

        `donor` was the default while it was the only method. It is retired,
        and a reader who omits the flag must get the input removal, which
        opens no second image.
        """
        assert cm.build_parser().parse_args([]).reference == "absent"

    def test_every_reference_has_a_method_note(self):
        assert set(cm.METHOD_NOTE) == set(cm.REFERENCES)

    def test_the_note_says_the_absence_run_reads_no_other_photograph(self):
        note = cm.METHOD_NOTE["absent"]
        assert "no other photograph" in note
        assert "donor" not in note

    def test_the_note_marks_the_donor_run_retired(self):
        assert cm.METHOD_NOTE["donor"].startswith("RETIRED")

    def summary_for(self, tmp_path, reference):
        import json
        import numpy as np
        from PIL import Image
        items = [("img", Image.new("RGB", (48, 48)),
                  np.array([1.0]), 2.0)]
        cm.draw_figures("subway_safety", tmp_path, items,
                        {"reference": reference})
        return json.loads((tmp_path / "summary.json").read_text())

    def test_the_summary_names_the_edit_that_ran(self, tmp_path):
        """A stale label made every absence map claim a donor swap."""
        got = self.summary_for(tmp_path, "absent")
        assert got["method"] == cm.METHOD_NOTE["absent"]
        assert "donor" not in got["method"]

    def test_a_substitution_summary_says_so(self, tmp_path):
        got = self.summary_for(tmp_path, "substitute")
        assert got["method"] == cm.METHOD_NOTE["substitute"]


class TestAdditivity:
    def write(self, root, name, values):
        import pandas as pd
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"image_id": "im1", "layer": 20, "values": v,
                       "spread": [0.0] * len(v)}
                      for v in [values]]).to_parquet(d / "maps.parquet")

    def test_a_perfectly_additive_pair_reads_1(self, tmp_path):
        rng = np.random.default_rng(0)
        fine = rng.normal(size=256)
        blocks = cm.pmc.patch_groups(256, 2)
        self.write(tmp_path, "c", list(fine))
        self.write(tmp_path, "c_g2", [float(fine[b].sum()) for b in blocks])
        df = cm.additivity("c", tmp_path, 2)
        assert df.reference.iloc[0] == "donor"
        assert df.ratio.iloc[0] == pytest.approx(1.0)
        assert df.pearson.iloc[0] == pytest.approx(1.0)

    def test_a_redundant_pair_reads_below_1(self, tmp_path):
        rng = np.random.default_rng(1)
        fine = rng.normal(size=256)
        blocks = cm.pmc.patch_groups(256, 2)
        self.write(tmp_path, "c", list(fine))
        self.write(tmp_path, "c_g2",
                   [float(fine[b].sum()) * 0.25 for b in blocks])
        df = cm.additivity("c", tmp_path, 2)
        assert df.ratio.iloc[0] == pytest.approx(0.25)

    def test_refuses_a_fine_run(self, tmp_path):
        with pytest.raises(SystemExit, match="group 2 or more"):
            cm.additivity("c", tmp_path, 1)

    def test_says_so_when_the_runs_share_no_image(self, tmp_path):
        import pandas as pd
        for name in ("c", "c_g2"):
            d = tmp_path / name; d.mkdir()
            pd.DataFrame([{"image_id": name, "layer": 20,
                           "values": [0.0] * 256, "spread": [0.0] * 256}]
                         ).to_parquet(d / "maps.parquet")
        with pytest.raises(SystemExit, match="same seed"):
            cm.additivity("c", tmp_path, 2)


class TestGroupSweep:
    def write(self, root, case, g, cells, worth, sum_parts):
        """A fake absence run: `cells` values that add up to `sum_parts`."""
        import pandas as pd
        d = cm.case_dir_for(root, case, g, "absent")
        d.mkdir(parents=True, exist_ok=True)
        v = [sum_parts / cells] * cells
        pd.DataFrame([{"image_id": "im1", "layer": None, "values": v,
                       "spread": [0.0] * cells}]).to_parquet(
                           d / "maps.parquet")
        pd.DataFrame([{"case": case, "image_id": "im1", "group": g,
                       "reference": "absent", "c": 18.0,
                       "image_worth": worth}]).to_parquet(
                           d / "map_stats.parquet")

    def test_the_curve_carries_every_group_on_disk(self, tmp_path):
        self.write(tmp_path, "c", 1, 256, 10.0, -5.0)
        self.write(tmp_path, "c", 16, 1, 10.0, 10.0)
        df = cm.group_sweep("c", tmp_path, "absent", [1, 16])
        assert sorted(df.group.unique()) == [1, 16]

    def test_a_missing_group_is_skipped_not_fatal(self, tmp_path):
        self.write(tmp_path, "c", 1, 256, 10.0, -5.0)
        df = cm.group_sweep("c", tmp_path, "absent", [1, 2, 4])
        assert list(df.group.unique()) == [1]

    def test_parts_over_whole_is_the_ratio(self, tmp_path):
        self.write(tmp_path, "c", 1, 256, 10.0, -5.0)
        df = cm.group_sweep("c", tmp_path, "absent", [1])
        assert df.parts_over_whole.iloc[0] == pytest.approx(-0.5)

    def test_the_end_point_passes_when_one_cell_is_the_whole(self, tmp_path):
        self.write(tmp_path, "c", 1, 256, 10.0, -5.0)
        self.write(tmp_path, "c", 16, 1, 10.0, 10.0)
        df = cm.group_sweep("c", tmp_path, "absent", [1, 16])
        assert df[df.group == 16].parts_over_whole.iloc[0] == \
            pytest.approx(1.0)

    def test_the_end_point_stops_a_broken_sweep(self, tmp_path):
        """1 cell that is not the whole-image ablation means a real bug."""
        self.write(tmp_path, "c", 1, 256, 10.0, -5.0)
        self.write(tmp_path, "c", 16, 1, 10.0, 7.0)
        with pytest.raises(SystemExit, match="does not reproduce"):
            cm.group_sweep("c", tmp_path, "absent", [1, 16])

    def test_refuses_the_donor_reference(self, tmp_path):
        with pytest.raises(SystemExit, match="needs --reference absent"):
            cm.group_sweep("c", tmp_path, "donor", [1, 2])

    def test_says_so_when_nothing_is_on_disk(self, tmp_path):
        with pytest.raises(SystemExit, match="no absence run"):
            cm.group_sweep("c", tmp_path, "absent", [1, 2])

    def test_the_cli_takes_a_list_of_groups(self):
        a = cm.build_parser().parse_args(["--group", "1", "2", "4"])
        assert a.group == [1, 2, 4]

    def test_one_group_still_works(self):
        assert cm.build_parser().parse_args(["--group", "2"]).group == [2]


class TestContract:
    def test_default_layer_is_inside_the_causal_window(self):
        assert cm.DEFAULT_LAYER < 30

    def test_refuses_a_layer_above_the_window(self):
        with pytest.raises(SystemExit, match="above the causal window"):
            cm.main(["--reference", "donor", "--layer", "43"])
