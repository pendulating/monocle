"""Tests for manifest construction."""

from __future__ import annotations

import pandas as pd

from trashbot_annotator.manifest import (
    MANIFEST_COLUMNS,
    build_manifest,
    derive_fields,
    make_sample_id,
)


def make_tree(root) -> None:  # type: ignore[no-untyped-def]
    """Create a small fake data tree."""
    files = [
        # dataset / date-park folder / capture folder / video
        "Han_2025_Trashbots/2025-07-01-tappen-park/onboard_camera/a.mp4",
        "Han_2025_Trashbots/2025-07-01-tappen-park/Interview/b.MP4",
        "25_little-italy/oct_1/Recycle/c.mov",
        "25_starlight-park/20250915/Interview/d.mp4",
        "Bu_TrashbotWoz/2022-10-12-astor-place/clips/e.MOV",
        "trashbot_summer_2026/joy_t2t/20260626/20260626_trashbot_cam/f.mp4",
        # both Albee Square spellings and the pilot folder
        "Bu_TrashbotWoz/2023-7-14-albee_square/landfill/g.mp4",
        "Bu_TrashbotWoz/albee_square_pilot/h.mp4",
        # non-video files must stay out of the manifest
        "Han_2025_Trashbots/2025-07-01-tappen-park/onboard_camera/skip.jpg",
        # Insta360 dual-fisheye files stay out of the manifest
        "25_starlight-park/20250915/Interview/skip_me.insv",
        "Han_2025_Trashbots/2025-07-01-tappen-park/onboard_camera/skip.LRV",
        # excluded folders must stay out of the walk
        "trashbot_summer_2026/joy_t2t/20260626/consent/skip.mp4",
        "Bu_TrashbotWoz/Trashbot_backup_img/skip2.mp4",
    ]
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"0")


def test_manifest_columns_and_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The manifest holds the exact columns and one row per video."""
    make_tree(tmp_path)
    df = build_manifest(tmp_path)

    assert list(df.columns) == list(MANIFEST_COLUMNS)
    # 8 videos. The .jpg, the .LRV, and the .insv stay out.
    assert len(df) == 8
    # Sorted by video_path.
    assert df["video_path"].tolist() == sorted(df["video_path"].tolist())
    # Absolute paths.
    assert all(str(p).startswith(str(tmp_path)) for p in df["video_path"])


def test_insv_files_are_excluded(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Insta360 dual-fisheye files stay out of the manifest."""
    make_tree(tmp_path)
    df = build_manifest(tmp_path)

    assert not any(str(p).endswith(".insv") for p in df["video_path"])
    assert "25_starlight-park_20250915_Interview_skip_me" not in set(df["sample_id"])


def test_sample_id_join_convention(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The sample id joins relative segments with '_' and drops the suffix."""
    make_tree(tmp_path)
    df = build_manifest(tmp_path).set_index("sample_id")

    sample = "Han_2025_Trashbots_2025-07-01-tappen-park_onboard_camera_a"
    assert sample in df.index
    # The id round-trips through the shared helper.
    assert make_sample_id(tmp_path, df.loc[sample, "video_path"]) == sample


def test_place_parsing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Date-park folders yield a park slug. Other styles yield null."""
    make_tree(tmp_path)
    df = build_manifest(tmp_path).set_index("sample_id")

    assert df.loc[
        "Han_2025_Trashbots_2025-07-01-tappen-park_onboard_camera_a", "place"
    ] == "tappen-park"
    assert pd.isna(df.loc["25_little-italy_oct_1_Recycle_c", "place"])
    assert pd.isna(df.loc["25_starlight-park_20250915_Interview_d", "place"])
    assert pd.isna(
        df.loc["trashbot_summer_2026_joy_t2t_20260626_20260626_trashbot_cam_f", "place"]
    )


def test_place_normalization_albee(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Both Albee Square spellings and the pilot folder yield one slug."""
    make_tree(tmp_path)
    df = build_manifest(tmp_path).set_index("sample_id")

    assert df.loc[
        "Bu_TrashbotWoz_2023-7-14-albee_square_landfill_g", "place"
    ] == "albee-square"
    assert df.loc["Bu_TrashbotWoz_albee_square_pilot_h", "place"] == "albee-square"


def test_control_mode_parsing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The summer 2026 folders yield a control mode. Others yield null."""
    make_tree(tmp_path)
    df = build_manifest(tmp_path).set_index("sample_id")

    assert df.loc[
        "trashbot_summer_2026_joy_t2t_20260626_20260626_trashbot_cam_f",
        "control_mode",
    ] == "joy_t2t"
    assert pd.isna(
        df.loc["Han_2025_Trashbots_2025-07-01-tappen-park_onboard_camera_a", "control_mode"]
    )


def test_excluded_dirs_stay_out(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Consent and backup folders drop out of the walk."""
    make_tree(tmp_path)
    df = build_manifest(tmp_path)

    joined = " ".join(df["video_path"].tolist())
    assert "consent" not in joined
    assert "Trashbot_backup_img" not in joined


def test_multi_root_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Two roots combine into one manifest; duplicate ids raise."""
    from trashbot_annotator.manifest import build_manifest_multi

    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    for root, rel in ((root_a, "ds_a/2025-07-01-tappen-park/x.mp4"), (root_b, "ds_b/y.mp4")):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"0")

    df = build_manifest_multi([root_a, root_b])
    assert len(df) == 2
    assert set(df["dataset"]) == {"ds_a", "ds_b"}

    try:
        build_manifest_multi([root_a, root_a])
        raise AssertionError("duplicate ids must raise")
    except ValueError:
        pass


def test_capture_type_parsing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Capture folders map to tokens. Unknown folders yield 'other'."""
    make_tree(tmp_path)
    df = build_manifest(tmp_path).set_index("sample_id")

    assert df.loc[
        "Han_2025_Trashbots_2025-07-01-tappen-park_onboard_camera_a",
        "capture_type",
    ] == "onboard_camera"
    assert df.loc[
        "Han_2025_Trashbots_2025-07-01-tappen-park_Interview_b", "capture_type"
    ] == "interview"
    assert df.loc["25_little-italy_oct_1_Recycle_c", "capture_type"] == "recycle"
    assert df.loc["Bu_TrashbotWoz_2022-10-12-astor-place_clips_e", "capture_type"] == "other"
    assert df.loc[
        "trashbot_summer_2026_joy_t2t_20260626_20260626_trashbot_cam_f",
        "capture_type",
    ] == "trashbot_cam"


def test_derive_fields_from_absolute_video_path() -> None:
    """Derive fields from a path with a 'data' ancestor."""
    video = "/share/ju/robot_norms/data/Han_2025_Trashbots/2025-07-03-tappen-park/onboard_camera/x.mp4"
    dataset, place, capture_type = derive_fields(video)
    assert dataset == "Han_2025_Trashbots"
    assert place == "tappen-park"
    assert capture_type == "onboard_camera"

    dataset2, place2, capture2 = derive_fields("/some/other/root/20250915/Interview/y.mp4")
    assert dataset2 is None
    assert place2 is None
    assert capture2 == "interview"


def test_write_manifest_parquet(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """write_manifest writes a parquet that reads back equal."""
    from trashbot_annotator.manifest import write_manifest

    make_tree(tmp_path)
    out = tmp_path / "out" / "manifest.parquet"
    df = write_manifest(tmp_path, out)

    back = pd.read_parquet(out)
    pd.testing.assert_frame_equal(df, back)
