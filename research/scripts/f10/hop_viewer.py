#!/usr/bin/env python
"""F10.6: write the saved reads of every token as runs for the Neuronpedia lens viewer.

The viewer of Neuronpedia opens a saved run from a JSON file and needs no model server.
Thus this script needs no GPU: it reads `data/f10/hop_all/{model}/` only.

Each run has 2 columns, at each read position and each layer:

| Column in the viewer | Label after the local patch | Content |
|----------------------|-----------------------------|---------|
| `JPP_LENS` | J++ Lens | the words with the highest lens logits |
| `LOGIT_LENS` | Contrast | the words with the highest contrast (see below) |

| Run | Contrast |
|-----|----------|
| 1 image | the read minus the mean of all images |
| all images | the mean minus the read of the same text with no image |
| 1 borough | the mean of the borough minus the mean of the other images |
| high `yes_no`, low `yes_no` | the mean of the top third minus the mean of the bottom third, and the reverse |

The tokens of the template before the image cannot see the image, thus they have 1 read
and no contrast. The image shows as 1 token with no read: the patch reads are in the store.
The words are the store words (` [a-z]{3,}`). The viewer needs a probability for each
word, thus the script gives a softmax over the store words. Read the rank, not the
number: the store does not hold the full vocabulary, and a contrast is not a logit.

Output in `data/f10/hop_all/viewer/`:

| File | Content |
|------|---------|
| `{model}/runs/{id}.json` | 1 run, in the `JlensExport` format of the viewer |
| `{model}/shares.json` | 1 row of the table `JlensShare` for each run |
| `img/{sample_id}.jpg` | a small copy of each street image |
| `index.html` | the list of runs, with the images and the answers |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_viewer.py [--limit 50]
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hop_store import ALL_ROOT, Store, open_store  # noqa: E402

__version__ = "1.0.0"

MODELS = {
    "gemma-4-31b": "/share/ju/matt/zoo/models/Gemma-4-31B-it",
    "qwen3.5-9b": "/share/ju/matt/zoo/models/Qwen3.5-9B",
    "qwen3.6-27b": "/share/ju/matt/zoo/models/Qwen3.6-27B",
}
NAMES = {"gemma-4-31b": "Gemma-4-31B", "qwen3.5-9b": "Qwen3.5-9B", "qwen3.6-27b": "Qwen3.6-27B"}
TOP_N = 30
RAW, CONTRAST = "JPP_LENS", "LOGIT_LENS"
SHOW = ("place", "final")
IMAGE_MARK = "[image]"


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(-1, keepdims=True)


def top_words(x: np.ndarray, words: np.ndarray, n: int = TOP_N) -> tuple[list, list]:
    """Give the top words and their softmax for [layers, vocab]. A NaN row gives empty lists."""
    names, probs = [], []
    for row in x:
        if np.isnan(row).any():
            names.append([])
            probs.append([])
            continue
        p = softmax(row)
        ix = np.argpartition(-row, n - 1)[:n]
        ix = ix[np.argsort(-row[ix])]
        names.append(words[ix].tolist())
        probs.append([round(float(v), 6) for v in p[ix]])
    return names, probs


def sequence(manifest: dict) -> tuple[list[tuple[str, int]], dict[str, int]]:
    """Give the tokens of the prompt with the image as 1 token, and the index of each token with a read."""
    prefix = list(zip(manifest["prefix_tokens"], manifest["prefix_ids"]))
    tail = list(zip(manifest["tail_tokens"], manifest["tail_ids"]))
    tokens = prefix + [(IMAGE_MARK, -1)] + tail
    at = {f"p{i:02d}": i for i in range(len(prefix))}
    at |= {name: len(prefix) + 1 + i for i, name in enumerate(manifest["positions"])}
    return tokens, at


def run(model: str, tokens: list, at: dict, positions: list[str], layers: list[int], raw: np.ndarray, contrast: np.ndarray,
        words: np.ndarray) -> dict:
    """Build 1 run in the `JlensExport` format. `raw` and `contrast` are [positions, layers, vocab]."""
    read = {at[p]: i for i, p in enumerate(positions)}
    blank = [[] for _ in layers]
    rows = []
    for k, (word, token_id) in enumerate(tokens):
        slices = {RAW: (blank, blank), CONTRAST: (blank, blank)}
        if k in read:
            slices = {RAW: top_words(raw[read[k]], words), CONTRAST: top_words(contrast[read[k]], words)}
        rows.append({"kind": "token", "position": k, "token": word, "id": token_id, "is_generated": False, "message_index": None,
                     "role": None, "channel": None, "section": None,
                     "results": [{"type": t, "top_tokens": a, "top_probs": b} for t, (a, b) in slices.items()]})
    meta = {"kind": "meta", "model": model, "types": [RAW, CONTRAST], "layers_by_type": {RAW: layers, CONTRAST: layers}, "top_n": TOP_N,
            "prompt_len": len(tokens), "num_completion_tokens": 0, "temperature": 0, "prepend_bos": False, "reuse_len": 0}
    return {"version": 1, "kind": "completion", "modelId": model, "exportedAt": datetime.now(timezone.utc).isoformat(), "meta": meta,
            "prompt": "".join(w for w, _ in tokens), "tokens": rows, "layerRange": [layers[0], layers[-1]]}


def share_id(model: str, name: str) -> str:
    return f"f10-{model.replace('.', '-')}-{name}".lower()


def run_link(model: str, name: str) -> str:
    return f"/{model}/jlens?shareId={share_id(model, name)}"


def share(model: str, name: str, description: str, n_tokens: int, at: dict) -> dict:
    return {"id": share_id(model, name), "kind": "completion", "modelId": model,
            "url": f"/f10/{model}/runs/{name}.json", "description": description, "selectedPositions": [at[p] for p in SHOW],
            "lensColumns": [RAW, CONTRAST], "activeLensModeTab": RAW, "topN": 10, "hideNonWordTokens": False, "temperature": 0.0,
            "numCompletionTokens": 0, "numPromptTokens": n_tokens}


def shards(s: Store, limit: int | None) -> list[Path]:
    files = sorted((s.path / "reads").glob("*.npy"))
    return files[: int(np.ceil(limit / s.manifest["chunk"]))] if limit else files


def group_masks(images: pd.DataFrame) -> dict[str, np.ndarray]:
    """Give the groups of images. The top and bottom thirds use `yes_no` on the images that the run reads."""
    masks = {"all": np.ones(len(images), bool)}
    for b, _ in images.groupby("borough"):
        masks[b.lower().replace(" ", "-")] = (images["borough"] == b).to_numpy()
    lo, hi = images["yes_no"].quantile([1 / 3, 2 / 3])
    masks["high-yes-no"], masks["low-yes-no"] = (images["yes_no"] >= hi).to_numpy(), (images["yes_no"] <= lo).to_numpy()
    return masks


def write_model(model: str, out: Path, limit: int | None) -> pd.DataFrame:
    s = open_store(model, ALL_ROOT)
    files = shards(s, limit)
    n = sum(np.load(f, mmap_mode="r").shape[0] for f in files)
    images = s.images.iloc[:n].reset_index(drop=True)
    tokens, at = sequence(s.manifest)
    print(f"[viewer] {model}: {n} images, {len(tokens)} tokens, {len(at)} with a read", flush=True)
    positions, layers = list(at), s.manifest["layers"]
    prefix = np.load(s.path / "prefix_reads.npy")
    no_contrast = np.full_like(prefix, np.nan)
    words = s.vocab["token"].to_numpy()
    dest = out / model / "runs"
    dest.mkdir(parents=True, exist_ok=True)
    masks = group_masks(images)
    sums, count, seen = {}, {k: int(v.sum()) for k, v in masks.items()}, 0
    for f in files:
        x = np.load(f)
        for k, m in masks.items():
            part = x[m[seen: seen + len(x)]].sum(0, dtype=np.float64)
            sums[k] = sums.get(k, 0) + part
        seen += len(x)
        print(f"[viewer] {model}: sums {seen}/{n}", flush=True)
    mean = {k: (v / count[k]).astype(np.float32) for k, v in sums.items()}
    pool = mean["all"]
    text = np.load(s.path / "text_reads.npy")
    rows = []

    def add(name: str, description: str, raw: np.ndarray, contrast: np.ndarray) -> None:
        full = run(model, tokens, at, positions, layers, np.concatenate([prefix, raw]), np.concatenate([no_contrast, contrast]), words)
        (dest / f"{name}.json").write_text(json.dumps(full, ensure_ascii=False))
        rows.append(share(model, name, description, len(tokens), at))

    add("all", f"All {n} images. Contrast: the mean minus the same text with no image.", pool, pool - text)
    for k in masks:
        if k in ("all", "high-yes-no", "low-yes-no"):
            continue
        rest = (sums["all"] - sums[k]) / (n - count[k])
        add(k, f"{k.title()}, {count[k]} images. Contrast: the borough minus the other images.", mean[k], mean[k] - rest.astype(np.float32))
    gap = mean["high-yes-no"] - mean["low-yes-no"]
    add("high-yes-no", f"Top third of yes_no, {count['high-yes-no']} images. Contrast: top third minus bottom third.", mean["high-yes-no"], gap)
    add("low-yes-no", f"Bottom third of yes_no, {count['low-yes-no']} images. Contrast: bottom third minus top third.", mean["low-yes-no"], -gap)
    seen = 0
    for f in files:
        x = np.load(f)
        for i, r in enumerate(images.iloc[seen: seen + len(x)].itertuples()):
            add(r.sample_id, f"{r.nta_name}, {r.borough}. Answer {r.top!r}, yes_no {r.yes_no:+.2f}. Contrast: this image minus the mean of all images.",
                x[i], x[i] - pool)
        seen += len(x)
        print(f"[viewer] {model}: runs {seen}/{n}", flush=True)
    (out / model / "shares.json").write_text(json.dumps(rows, indent=0, ensure_ascii=False))
    return images.assign(model=model)


def thumbs(images: pd.DataFrame, out: Path) -> None:
    from PIL import Image

    dest = out / "img"
    dest.mkdir(parents=True, exist_ok=True)
    for r in images.drop_duplicates("sample_id").itertuples():
        path = dest / f"{r.sample_id}.jpg"
        if not path.exists():
            with Image.open(r.image_path) as im:
                im.convert("RGB").resize((320, int(320 * im.height / im.width))).save(path, quality=80)


def index(images: pd.DataFrame, out: Path) -> None:
    """Write the list of runs. The page has 1 row for each image and 1 link for each model."""
    link = run_link
    groups = ["all", "bronx", "brooklyn", "manhattan", "queens", "staten-island", "high-yes-no", "low-yes-no"]
    head = "".join(f"<tr><th>{NAMES[m]}</th>" + "".join(f'<td><a href="{link(m, g)}">{g}</a></td>' for g in groups) + "</tr>" for m in MODELS)
    body = []
    for (sid, nta, boro), g in images.groupby(["sample_id", "nta_name", "borough"]):
        cells = "".join(f'<td><a href="{link(r.model, sid)}">{NAMES[r.model]}</a><br>{html.escape(str(r.top).strip())} {r.yes_no:+.2f}</td>'
                        for r in g.itertuples())
        nta, boro = html.escape(nta), html.escape(boro)
        body.append(f'<tr data-k="{nta.lower()} {boro.lower()}"><td><img loading="lazy" src="img/{sid}.jpg" alt=""></td><td>{nta}<br>{boro}</td>{cells}</tr>')
    page = f"""<!doctype html><meta charset="utf-8"><title>F10.6 token reads</title>
<style>body{{font:15px system-ui;margin:16px;max-width:900px}}table{{border-collapse:collapse}}td,th{{padding:4px 10px;text-align:left;vertical-align:top}}
img{{width:200px;display:block}}#l tr{{border-top:1px solid #ccc}}input{{font:inherit;padding:4px;width:260px}}</style>
<h1>F10.6 token reads</h1><p>Prompt: the image, then "Is this place cool? Answer with one word: yes or no."</p>
<p>A read at a token predicts the next token. Reads exist at each token, but the viewer shows the image as 1 token with no read. Column "Contrast": see the text of each run.</p>
<h2>Groups</h2><table>{head}</table><h2>Images</h2>
<p><input id="q" placeholder="Filter by NTA or borough"></p><table id="l">{"".join(body)}</table>
<script>q.oninput=()=>{{const v=q.value.toLowerCase();for(const r of l.rows)r.hidden=!r.dataset.k.includes(v)}}</script>"""
    (out / "index.html").write_text(page)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="the first images only, rounded up to a whole shard")
    ap.add_argument("--out", default=str(ALL_ROOT / "viewer"))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    out = Path(args.out)
    images = pd.concat([write_model(m, out, args.limit) for m in args.models], ignore_index=True)
    thumbs(images, out)
    index(images, out)
    print("[viewer] finished", flush=True)


if __name__ == "__main__":
    main()
