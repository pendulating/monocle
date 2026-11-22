"""A row whose images do not load must not go to the model (single-GPU path)."""
from types import SimpleNamespace

import pytest

from dagspaces.common import vllm_inference as vi


def _msgs(*images):
    content = [{"type": "image", "image": p} for p in images] + [{"type": "text", "text": "Q"}]
    return [{"role": "user", "content": content}]


def test_count_image_refs():
    msgs = _msgs("/a.jpg", "/b.jpg")
    msgs[0]["content"].append({"type": "image_url", "image_url": {"url": "file:///c.jpg"}})
    assert vi._count_image_refs(msgs) == 3
    assert vi._count_image_refs([{"role": "user", "content": "text only"}]) == 0
    assert vi._count_image_refs(None) == 0


def test_missing_file_gives_an_error(example_images):
    good = str(sorted(example_images.glob("*.jpg"))[0])
    msgs = _msgs(good, "/no/such/file.jpg")
    hydrated = vi._hydrate_messages_for_chat(msgs)
    err = vi._image_load_error(msgs, vi._count_image_refs(hydrated))
    assert err is not None and "1 of 2 images" in err
    assert vi._image_load_error(_msgs(good), 1) is None


def test_remote_url_gives_an_error():
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}]}]
    assert vi._count_image_refs(vi._hydrate_messages_for_chat(msgs)) == 0
    assert vi._image_load_error(msgs, 0).startswith("1 of 1 images")


def test_with_skipped_keeps_the_order():
    outs = ["o0", "o2"]
    merged = vi._with_skipped(outs, {1: "bad"}, 3)
    assert merged[0] == "o0" and merged[2] == "o2"
    assert merged[1]["image_load_error"] == "bad" and merged[1]["raw_text"] == ""
    assert vi._with_skipped(outs, {}, 2) is outs
    with pytest.raises(RuntimeError):
        vi._with_skipped(["a", "b", "c"], {0: "bad"}, 3)


def test_records_keep_the_skip_marker():
    out = SimpleNamespace(outputs=[SimpleNamespace(text="yes", token_ids=[1, 2])],
                          prompt_token_ids=[1, 2, 3])
    recs = vi._records_from_outputs([out, vi._skipped_record("bad")])
    assert recs[0] == {"raw_text": "yes", "prompt_tokens": 3, "completion_tokens": 2}
    assert recs[1]["image_load_error"] == "bad"


def test_strict_setting():
    assert vi._strict_images_setting(SimpleNamespace(runtime=SimpleNamespace(strict_images=True)))
    assert not vi._strict_images_setting(SimpleNamespace(runtime=SimpleNamespace()))
    assert not vi._strict_images_setting(SimpleNamespace())
    vi._check_image_errors({0: "x"}, "s", False)
    with pytest.raises(RuntimeError, match="strict_images"):
        vi._check_image_errors({0: "x"}, "s", True)


def test_postprocess_keeps_the_error_columns():
    pp = vi._keep_error_columns(lambda row: {"answer": row.get("generated_text", "")})
    out = pp({"generated_text": "", "__image_load_error__": "bad", "__preprocess_error__": "bad"})
    assert out == {"answer": "", "__image_load_error__": "bad", "__preprocess_error__": "bad"}
    assert pp({"generated_text": "ok"}) == {"answer": "ok"}


def test_wrapper_logs_and_strict(monkeypatch, caplog):
    import pandas as pd

    def fake(df, cfg, pre, post, stage_name):
        return pd.DataFrame([post({"generated_text": "a"}),
                             post({"generated_text": "", "__image_load_error__": "gone"})])

    monkeypatch.setattr(vi, "_run_vllm_inference", fake)
    loose = SimpleNamespace(runtime=SimpleNamespace(strict_images=False))
    with caplog.at_level("WARNING"):
        out = vi.run_vllm_inference(pd.DataFrame(), loose, None, lambda r: {"a": r["generated_text"]})
    assert out["__image_load_error__"].tolist()[1] == "gone"
    assert "1 rows have images that did not load" in caplog.text
    strict = SimpleNamespace(runtime=SimpleNamespace(strict_images=True))
    with pytest.raises(RuntimeError, match="strict_images"):
        vi.run_vllm_inference(pd.DataFrame(), strict, None, lambda r: {"a": r["generated_text"]})


def test_vqa_names_an_image():
    from dagspaces.vqa.stages.vqa import _names_an_image

    assert _names_an_image({"image_path": "/no/such.jpg"})
    assert _names_an_image({"image_path": "https://example.com/a.jpg"})
    assert not _names_an_image({"image_path": None})
    assert not _names_an_image({"image_path": " "})
