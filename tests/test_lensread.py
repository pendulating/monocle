"""Tests of the J-lens readout core (research/monocle/readout.py) and the lensread dagspace.

A fake tokenizer, model and lens stand in for the real ones, thus no test needs a GPU.
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from research.monocle import readout as R

ROOT = Path(__file__).resolve().parent.parent
CONF = ROOT / "research" / "dagspaces" / "lensread" / "conf"

# ---------------------------------------------------------------------------
# A fake tokenizer: greedy longest match over a small vocabulary.
# ---------------------------------------------------------------------------
VOCAB = ["<bos>", "<img>", "<gen>", "\n", "Is", " this", " place", " cool", "?", " Answer",
         "Yes", "yes", " Yes", " yes", "No", "no", " No", " no", " red", " big", " street",
         " the", "ab", " x"]
ID = {t: i for i, t in enumerate(VOCAB)}
IMG = ID["<img>"]


def encode(text: str) -> list[int]:
    out, i = [], 0
    while i < len(text):
        best = max((t for t in VOCAB if text.startswith(t, i)), key=len, default=None)
        if best is None:
            raise ValueError(f"no token at {text[i:]!r}")
        out.append(ID[best])
        i += len(best)
    return out


def decode(ids: list[int]) -> str:
    return "".join(VOCAB[i] for i in ids)


PATCHES = 4
D = 6
LAYERS = [0, 2, 4]


def fake_tokenize(images, prompt):
    seq = [ID["<bos>"]] + [IMG] * (PATCHES * len(images)) + encode(prompt) + [ID["<gen>"], ID["\n"]]
    return R.Tokenized(inputs={"seq": seq}, seq=seq, n_gen=2)


def fake_forward(inputs, jump_at=None):
    seq = inputs["seq"]
    g = torch.Generator().manual_seed(len(seq))
    base = torch.randn(len(seq), D, generator=g)
    acts = {}
    for k, l in enumerate(LAYERS):
        h = base + 0.01 * k
        if jump_at is not None and k == len(LAYERS) - 1:
            h = h.clone()
            h[jump_at] = h[jump_at] + 50.0
        acts[l] = h
    logits = torch.zeros(len(VOCAB))
    logits[ID["Yes"]] = 2.0
    logits[ID["No"]] = 1.0
    return acts, logits


_W = {l: torch.randn(D, len(VOCAB), generator=torch.Generator().manual_seed(100 + l)) for l in LAYERS}


def fake_read(h, layer):
    return h.float() @ _W[layer]


def make_reader(**kw):
    args = dict(tokenize=fake_tokenize, forward=fake_forward, read=fake_read, decode=decode,
                encode=encode, layers=LAYERS, image_token=IMG, selectors=[{"kind": "last"}], top_k=3)
    args.update(kw)
    return R.Reader(**args)


PROMPT = "Is this place cool? Answer"

# ---------------------------------------------------------------------------
# Segments and selectors
# ---------------------------------------------------------------------------


def test_segments_one_image():
    seq = fake_tokenize(["a"], PROMPT).seq
    s = R.segments(seq, IMG, 2)
    assert s["prefix"] == [0]
    assert s["image"] == [1, 2, 3, 4]
    assert s["text"] == list(range(5, len(seq) - 2))
    assert s["gen"] == [len(seq) - 2, len(seq) - 1]
    assert s["all"] == list(range(len(seq)))


def test_segments_two_images_and_text_between():
    seq = [ID["<bos>"], IMG, IMG, ID[" x"], IMG, IMG] + encode(" red") + [ID["<gen>"]]
    s = R.segments(seq, IMG, 1)
    assert s["image"] == [1, 2, 4, 5]
    assert s["text"] == [3, 6]
    assert s["prefix"] == [0]


def test_segments_no_image_and_bad_gen():
    seq = encode(PROMPT) + [ID["<gen>"]]
    s = R.segments(seq, IMG, 1)
    assert s["image"] == [] and s["text"] == [] and s["prefix"] == list(range(len(seq) - 1))
    with pytest.raises(ValueError):
        R.segments(seq, IMG, len(seq) + 1)


@pytest.mark.parametrize("spec, msg", [
    ({"kind": "nope"}, "unknown selector"),
    ({"kind": "match"}, "give text"),
    ({"kind": "tail", "n": 0}, "n must be"),
    ({"kind": "head", "segment": "body"}, "unknown segment"),
    ({"kind": "top_delta", "metric": "kl"}, "read_js or residual"),
    ({"kind": "index"}, "give at"),
])
def test_parse_selectors_rejects(spec, msg):
    with pytest.raises(ValueError, match=msg):
        R.parse_selectors([spec])


def test_parse_selectors_labels_are_unique():
    with pytest.raises(ValueError, match="not unique"):
        R.parse_selectors([{"kind": "last"}, {"kind": "last"}])
    specs = R.parse_selectors([{"kind": "last"}, {"kind": "last", "label": "final"}])
    assert [s["label"] for s in specs] == ["last", "final"]


def _select(specs, seq=None):
    seq = seq or fake_tokenize(["a"], PROMPT).seq
    segs = R.segments(seq, IMG, 2)
    return seq, R.select_positions(R.parse_selectors(specs), seq, segs, encode)


def test_select_each_kind():
    seq, got = _select([
        {"kind": "last"},
        {"kind": "tail", "n": 2, "segment": "text"},
        {"kind": "head", "n": 1},
        {"kind": "match", "text": " place cool"},
        {"kind": "match", "text": " place cool", "which": "all", "label": "both"},
        {"kind": "index", "at": [0, -2]},
        {"kind": "segment", "name": "image"},
    ])
    n = len(seq)
    place = seq.index(ID[" place"])
    assert got[n - 1] == ["last"]
    assert "tail2_text" in got[n - 3] and "tail2_text" in got[n - 4]
    assert "head1_text" in got[5]
    assert got[place + 1] == ["match: place cool", "both"]
    assert got[place] == ["both"]
    assert "index:0,-2" in got[0] and "index:0,-2" in got[n - 2]
    assert all(got[p] == ["segment:image"] for p in (1, 2, 3, 4))


def test_select_match_occurrence():
    seq = encode(" red x red x red")  # " x" separates the 3 reds
    seq = [ID["<bos>"]] + seq + [ID["<gen>"], ID["\n"]]
    reds = [i for i, t in enumerate(seq) if t == ID[" red"]]
    for occ, want in (("first", reds[:1]), ("last", reds[-1:]), ("all", reds)):
        segs = R.segments(seq, IMG, 2)
        got = R.select_positions(R.parse_selectors([{"kind": "match", "text": " red", "occurrence": occ}]),
                                 seq, segs, encode)
        assert sorted(got) == want


def test_select_empty_raises_unless_optional():
    with pytest.raises(ValueError, match="selects no position"):
        _select([{"kind": "match", "text": " street"}])
    _, got = _select([{"kind": "match", "text": " street", "optional": True}, {"kind": "last"}])
    assert len(got) == 1


def test_select_index_out_of_range():
    with pytest.raises(ValueError, match="outside"):
        _select([{"kind": "index", "at": [999]}])


# ---------------------------------------------------------------------------
# Delta metrics
# ---------------------------------------------------------------------------


def test_js_divergence():
    a = torch.randn(3, 10)
    assert torch.allclose(R.js_divergence(a, a), torch.zeros(3), atol=1e-6)
    b = torch.randn(3, 10)
    assert torch.allclose(R.js_divergence(a, b), R.js_divergence(b, a), atol=1e-6)
    assert (R.js_divergence(a, b) <= math.log(2) + 1e-6).all()


def test_residual_delta():
    h = torch.ones(2, 3, 4)
    h[1, 2] = 3.0
    d = R.residual_delta(h)
    assert d.shape == (2, 2)
    assert torch.allclose(d[0], torch.zeros(2))
    assert d[1, 1] == pytest.approx(2.0)


def test_top_k_positions_ties_take_the_earlier():
    assert R.top_k_positions({5: 1.0, 2: 1.0, 9: 3.0}, 2) == [9, 2]


@pytest.mark.parametrize("metric", ["residual", "read_js"])
def test_top_delta_finds_the_jump(metric):
    seq = fake_tokenize(["a"], PROMPT).seq
    jump = seq.index(ID[" cool"])
    reader = make_reader(forward=lambda inp: fake_forward(inp, jump_at=jump),
                         selectors=[{"kind": "top_delta", "k": 1, "metric": metric}])
    out = reader.read_row("s", ["a"], PROMPT)
    positions = {r["position"] for r in out.results}
    assert positions == {jump}
    assert all(r["selectors"] == f"top_delta1_{metric}" for r in out.results)
    assert all(r["delta"] > 0 for r in out.results)


def test_top_delta_layer_range_needs_2_layers():
    reader = make_reader(selectors=[{"kind": "top_delta", "k": 1, "layers": [2, 3]}])
    with pytest.raises(ValueError, match="fewer than 2"):
        reader.read_row("s", ["a"], PROMPT)


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------


def test_read_row_shapes_and_values():
    reader = make_reader(selectors=[{"kind": "last"}, {"kind": "head", "n": 2}], top_k=3)
    out = reader.read_row("s1", ["a"], PROMPT)
    assert len(out.results) == 3 * len(LAYERS) * 3
    df = pd.DataFrame(out.results)
    seq = fake_tokenize(["a"], PROMPT).seq
    acts, _ = fake_forward({"seq": seq})
    row = df[(df.position == len(seq) - 1) & (df.layer == 2)].sort_values("rank")
    want = fake_read(acts[2][[len(seq) - 1]], 2)[0]
    assert row["token_id"].tolist() == want.topk(3).indices.tolist()
    assert np.allclose(row["logit"], want.topk(3).values.numpy(), atol=1e-5)
    lp = torch.log_softmax(want, -1)
    assert np.allclose(row["logprob"], lp[want.topk(3).indices].numpy(), atol=1e-5)
    assert set(df.segment) == {"text", "gen"}
    assert out.reads is None and out.residuals is None


def test_read_row_keep_ids_and_store():
    keep = [ID[" red"], ID[" big"], ID[" street"], ID[" the"]]
    reader = make_reader(selectors=[{"kind": "last"}, {"kind": "tail", "n": 2}], top_k=2,
                         keep_ids=keep, store=True)
    out = reader.read_row("s1", ["a"], PROMPT)
    assert set(pd.DataFrame(out.results).token_id) <= set(keep)
    assert out.reads.shape == (2, len(LAYERS), len(keep)) and out.reads.dtype == np.float16
    assert out.residuals.shape == (2, len(LAYERS), D)
    assert [i["position"] for i in out.index] == sorted(i["position"] for i in out.index)


def test_answer_margin():
    out = make_reader().read_row("s", ["a"], PROMPT)
    a = out.answer
    _, logits = fake_forward({"seq": fake_tokenize(["a"], PROMPT).seq})
    lp = torch.log_softmax(logits, -1)
    assert a["top"] == "Yes"
    assert a["yes_no"] == pytest.approx(float(lp[ID["Yes"]] - lp[ID["No"]]))
    forms = torch.tensor([encode(f)[0] for f in R.ANSWERS])
    assert a["margin"] == pytest.approx(float(R.pooled_margin(lp[forms])))
    assert a["n_patches"] == PATCHES
    no_forms = make_reader(answer_forms=None).read_row("s", ["a"], PROMPT).answer
    assert "margin" not in no_forms


def test_non_finite_read_raises():
    reader = make_reader(read=lambda h, l: torch.full((h.shape[0], len(VOCAB)), float("nan")))
    with pytest.raises(RuntimeError, match="not finite"):
        reader.read_row("s", ["a"], PROMPT)


def test_hop_all_gives_the_same_helpers():
    sys.path.insert(0, str(ROOT / "research" / "scripts" / "f10"))
    from research.scripts.f10 import hop_all

    assert hop_all.pooled_margin is R.pooled_margin
    assert hop_all.parts is R.parts
    assert hop_all.ANSWERS is R.ANSWERS


# ---------------------------------------------------------------------------
# The runner and the dagspace
# ---------------------------------------------------------------------------
def _cfg(tmp_path, n_rows=5, chunk=2, store=False, prompt=None):
    table = tmp_path / "rows.parquet"
    pd.DataFrame({"sample_id": [f"r{i}" for i in range(n_rows)],
                  "image_path": [f"img{i}.jpg" for i in range(n_rows)],
                  "prompt": [PROMPT] * n_rows}).to_parquet(table)
    return OmegaConf.create({
        "data": {"parquet_path": str(table), "image_root": "/imgs", "columns": None},
        "runtime": {"debug": False, "sample_n": None},
        "readout": {"chunk_rows": chunk, "store": store, "prompt": prompt},
    })


def _context(cfg, out):
    node = SimpleNamespace(key="readout", outputs={
        "results": SimpleNamespace(optional=False), "answers": SimpleNamespace(optional=False),
        "store": SimpleNamespace(optional=True)})
    paths = {"results": str(out / "results.parquet"), "answers": str(out / "answers.parquet"),
             "store": str(out / "store")}
    return SimpleNamespace(cfg=cfg, inputs={}, output_paths=paths, node=node, logger=None, output_dir=str(out))


class _Factory:
    def __init__(self, store=False):
        self.calls, self.rows, self.store = 0, [], store

    def __call__(self, cfg):
        self.calls += 1
        reader = make_reader(selectors=[{"kind": "last"}], store=self.store,
                             keep_ids=[ID[" red"], ID[" big"]])
        real = reader.read_row

        def read_row(sample_id, images, prompt):
            self.rows.append((sample_id, images, prompt))
            return real(sample_id, images, prompt)

        reader.read_row = read_row
        tok = SimpleNamespace(decode=decode)
        return reader, {"manifest": {"model": "fake", "vocab_size": 2}, "vocab": [ID[" red"], ID[" big"]],
                        "tokenizer": tok}


def test_runner_end_to_end_and_resume(tmp_path):
    from research.dagspaces.lensread.runner import ReadoutRunner

    cfg, out = _cfg(tmp_path), tmp_path / "outputs" / "readout"
    out.mkdir(parents=True)
    fac = _Factory()
    ReadoutRunner(make_reader=fac, load_images=lambda paths: paths).run(_context(cfg, out))
    res, ans = pd.read_parquet(out / "results.parquet"), pd.read_parquet(out / "answers.parquet")
    assert ans["sample_id"].tolist() == [f"r{i}" for i in range(5)]
    assert len(res) == 5 * len(LAYERS) * 2
    assert fac.rows[0][1] == ["/imgs/img0.jpg"]
    assert sorted(os.listdir(out / "chunks")) == [f"{k}-{c:05d}.parquet" for k in ("answers", "results") for c in range(3)]
    assert (out / "manifest.json").exists()

    # A rerun after a lost chunk reads only that chunk.
    os.remove(out / "chunks" / "answers-00001.parquet")
    fac2 = _Factory()
    ReadoutRunner(make_reader=fac2, load_images=lambda paths: paths).run(_context(cfg, out))
    assert [r[0] for r in fac2.rows] == ["r2", "r3"]
    assert pd.read_parquet(out / "answers.parquet")["sample_id"].tolist() == [f"r{i}" for i in range(5)]

    # A rerun with all chunks done does not load the model.
    fac3 = _Factory()
    ReadoutRunner(make_reader=fac3, load_images=lambda paths: paths).run(_context(cfg, out))
    assert fac3.calls == 0


def test_runner_refuses_other_rows(tmp_path):
    from research.dagspaces.lensread.runner import ReadoutRunner

    cfg, out = _cfg(tmp_path), tmp_path / "o"
    out.mkdir()
    ReadoutRunner(make_reader=_Factory(), load_images=lambda p: p).run(_context(cfg, out))
    pd.DataFrame({"sample_id": ["zz", "r1"]}).to_parquet(out / "chunks" / "answers-00000.parquet")
    with pytest.raises(RuntimeError, match="other rows"):
        ReadoutRunner(make_reader=_Factory(), load_images=lambda p: p).run(_context(cfg, out))


def test_runner_store_and_fixed_prompt(tmp_path):
    from research.dagspaces.lensread.runner import ReadoutRunner

    cfg, out = _cfg(tmp_path, n_rows=3, chunk=2, store=True, prompt="Is this street cool?"), tmp_path / "o"
    out.mkdir()
    fac = _Factory(store=True)
    ReadoutRunner(make_reader=fac, load_images=lambda p: p).run(_context(cfg, out))
    assert {r[2] for r in fac.rows} == {"Is this street cool?"}
    reads = np.load(out / "store" / "reads-00000.npy")
    assert reads.shape == (2, len(LAYERS), 2)
    assert np.load(out / "store" / "residuals-00001.npy").shape == (1, len(LAYERS), D)
    idx = pd.read_parquet(out / "store" / "index-00000.parquet")
    assert idx["sample_id"].tolist() == ["r0", "r1"]
    vocab = pd.read_parquet(out / "store" / "vocab.parquet")
    assert vocab["token"].tolist() == [" red", " big"]


def test_image_lists(tmp_path):
    from research.dagspaces.lensread.runner import image_lists

    cfg = OmegaConf.create({"data": {"image_root": "/r"}})
    df = pd.DataFrame({"image_paths": [["a.jpg", "/abs/b.jpg"]]})
    assert image_lists(df, cfg) == [["/r/a.jpg", "/abs/b.jpg"]]
    with pytest.raises(ValueError):
        image_lists(pd.DataFrame({"x": [1]}), cfg)


@pytest.mark.parametrize("overrides", [
    ["pipeline=readout_example"],
    ["pipeline=readout", "model=jpp_gemma-4-31b"],
    ["pipeline=readout", "model=jpp_qwen3.6-27b", "readout=f10", "readout.noun=street"],
])
def test_configs_compose(overrides):
    from hydra import compose, initialize_config_dir

    from dagspaces.common.config_schema import load_pipeline_graph

    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        cfg = compose("config", overrides=overrides)
    readout = OmegaConf.to_container(cfg.readout, resolve=True)
    R.parse_selectors(readout["positions"])
    assert cfg.runtime.stage_launcher == cfg.model.launcher
    graph = load_pipeline_graph(cfg)
    from research.dagspaces.lensread import orchestrator

    registry = orchestrator.get_stage_registry()
    assert all(node.stage in registry for node in graph.nodes.values())


def test_execute_stage_job_pickles_by_reference():
    from research.dagspaces.lensread import orchestrator

    assert pickle.loads(pickle.dumps(orchestrator.execute_stage_job)) is orchestrator.execute_stage_job


@pytest.mark.gpu
@pytest.mark.data
def test_f10_parity_qwen35_9b():
    """The f10 preset gives the reads and the margin of the F10.30 store (qwen3.5-9b, 2 images)."""
    from hydra import compose, initialize_config_dir

    from research.dagspaces.lensread.runner import _load_images

    store = ROOT / "data" / "f10" / "hop_more" / "lean" / "qwen3.5-9b"
    if not store.exists():
        pytest.skip(f"no F10.30 store at {store}")
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        cfg = compose("config", overrides=["model=jpp_qwen3.5-9b", "readout=f10", "readout.store=true"])
    model = OmegaConf.to_container(cfg.model, resolve=True)
    model["device_map"] = "cuda:0"
    reader, extras = R.load_reader(model, OmegaConf.to_container(cfg.readout, resolve=True))
    ref_vocab = pd.read_parquet(store / "vocab.parquet")["token_id"].tolist()
    assert list(extras["vocab"]) == ref_vocab
    sample = pd.read_parquet(ROOT / "data" / "f10" / "hop_more" / "images_k10.parquet").head(2)
    ref = np.load(store / "reads" / "reads-00000.npy").astype(np.float32)
    ref_ans = pd.read_parquet(store / "answers" / "answers-00000.parquet")
    for k, row in enumerate(sample.itertuples()):
        out = reader.read_row(row.sample_id, _load_images([row.image_path]), cfg.readout.prompt)
        labels = [i["selectors"].split(",") for i in out.index]
        for j, name in enumerate(("place", "after_place", "final")):
            (p,) = [n for n, ls in enumerate(labels) if name in ls]
            got, want = out.reads[p].astype(np.float32), ref[k, j]
            assert np.abs(got - want).max() <= 1e-2 * np.abs(want).max()
            assert (got.argmax(-1) == want.argmax(-1)).all()
        assert out.answer["margin"] == pytest.approx(ref_ans["margin"].iloc[k], abs=0.1)
