#!/usr/bin/env python
"""F4: write static HTML pages to read each answer of the corpus.

1 card for each row of `data/f4_traces/f4_traces_1k.parquet` (1 model, 1 case,
1 image). A card holds the photograph, the metadata, the raw trace
(`reasoning`), and the raw final answer (`response`). The extracted spans
(`data/f4_traces/extractions/`) show as highlights, with a colour for each
class. Hover over a highlight to see its class and attributes.

The images are embedded, thus a page needs no file beside it and you can
copy it to a laptop. 1 image is embedded 1 time on a page, as a CSS class, and
every card of that image uses the class. 1 page holds the cards of
`--per-page` images (default 50: about 600 cards and 10 MB).

Output (default `data/f4_traces/viewer/`):

| File | Content |
|------|---------|
| `index.html` | a table of the pages, with the restaurants of each page |
| `page_NNN.html` | the cards of 1 page, with filters for model, case, and class |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/build_trace_viewer.py
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/build_trace_viewer.py --models qwen3.8-27b --per-page 100
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
from datetime import datetime, timezone

import pandas as pd
from PIL import Image

ROOT = "/share/ju/matt/mllmsci"
TRACES = os.path.join(ROOT, "data/f4_traces/f4_traces_1k.parquet")
EXTRACTIONS = os.path.join(ROOT, "data/f4_traces/extractions")
OUT_DIR = os.path.join(ROOT, "data/f4_traces/viewer")

__version__ = "1.0.1"

CASES = ("restaurant", "caption")
MODELS = (
    "gemma-4-12b", "gemma-4-31b", "qwen3.5-9b", "qwen3.8-27b",
    "llama-3.2-11b", "llama-3.2-90b",
)
FIELDS = ("reasoning", "response")
# The classes of `single-v1`, in the priority of a highlight: when 2 spans
# overlap, the segment takes the colour of the first class in this list.
CLASSES = ("inference", "person_reference", "uncertainty", "image_artifact", "visual_evidence")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_spans(directory: str) -> dict:
    """Give the quotable spans of each text, keyed by `doc_id`.

    A span with no offsets cannot be placed, thus it drops out. A later copy of
    the same class and text inside 1 text drops out too (an extractor loop).
    """
    frames = []
    for case in CASES:
        path = os.path.join(directory, f"f4_{case}_extractions.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path, columns=[
                "doc_id", "extraction_class", "extraction_text", "attributes_json",
                "char_start", "char_end", "is_quotable", "is_verbatim", "is_duplicate"]))
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df = df[df.extraction_class.notna() & df.is_quotable & df.char_start.notna()
            & ~df.is_duplicate.fillna(False).astype(bool)]
    out: dict = {}
    for row in df.itertuples(index=False):
        out.setdefault(row.doc_id, []).append({
            "cls": row.extraction_class, "text": row.extraction_text,
            "attrs": json.loads(row.attributes_json or "{}"),
            "start": int(row.char_start), "end": int(row.char_end),
            "verbatim": bool(row.is_verbatim),
        })
    return out


def image_data_uri(path: str, px: int, quality: int) -> tuple[str, float]:
    """Give a JPEG data URI of the image at `px` wide, and its width/height ratio."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        ratio = im.width / im.height
        if im.width > px:
            im = im.resize((px, round(px / ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), ratio


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def highlight(text: str, spans: list) -> str:
    """Give the text as HTML, with 1 `<mark>` for each segment that a span covers.

    Spans can overlap. The text is cut at every span edge, and each segment
    lists all spans that cover it in its tooltip.
    """
    if not text:
        return '<span class="empty">(empty)</span>'
    spans = [s for s in spans if 0 <= s["start"] < s["end"] <= len(text)]
    if not spans:
        return html.escape(text)
    edges = sorted({0, len(text), *(s["start"] for s in spans), *(s["end"] for s in spans)})
    rank = {c: i for i, c in enumerate(CLASSES)}
    parts = []
    for a, b in zip(edges, edges[1:]):
        piece = html.escape(text[a:b])
        cover = [s for s in spans if s["start"] <= a and b <= s["end"]]
        if not cover:
            parts.append(piece)
            continue
        cover.sort(key=lambda s: rank.get(s["cls"], len(rank)))
        tip = "\n".join(
            s["cls"] + "".join(f" {k}={v}" for k, v in s["attrs"].items())
            + ("" if s["verbatim"] else " (fuzzy match)")
            for s in cover
        )
        classes = " ".join(sorted({f"c-{s['cls']}" for s in cover}))
        parts.append(
            f'<mark class="{classes} top-{cover[0]["cls"]}" title="{html.escape(tip)}">{piece}</mark>'
        )
    return "".join(parts)


def span_table(spans: list) -> str:
    if not spans:
        return ""
    rows = "".join(
        f'<tr><td><span class="dot top-{s["cls"]}"></span>{s["cls"]}</td>'
        f'<td>{html.escape(", ".join(f"{k}={v}" for k, v in s["attrs"].items()))}</td>'
        f'<td>{html.escape(s["text"])}</td><td class="num">{s["start"]}–{s["end"]}</td></tr>'
        for s in sorted(spans, key=lambda s: s["start"])
    )
    return (f'<details class="spans"><summary>{len(spans)} spans</summary>'
            f'<table><tr><th>class</th><th>attributes</th><th>text</th><th>offsets</th></tr>{rows}</table></details>')


def card(row, spans: dict) -> str:
    """Give the HTML of 1 card: 1 row of the corpus."""
    meta = [
        ("model", row.model), ("case", row.case), ("image_idx", row.image_idx),
        ("camis", row.camis), ("borough", row.borough),
        ("finish", row.finish_reason), ("tokens", f"{row.completion_tokens:,} / {row.max_tokens:,}"),
        ("run", row.source_run),
    ]
    flags = '<span class="flag">cut</span>' if row.cut else ""
    blocks = []
    for field in FIELDS:
        text = getattr(row, field) or ""
        doc = f"{row.model}|{row.case}|{row.image_idx}|{field}"
        s = spans.get(doc, [])
        if field == "reasoning" and not text:
            blocks.append('<section class="text"><h4>reasoning</h4><span class="empty">(no trace: this model has no thinking mode)</span></section>')
            continue
        blocks.append(
            f'<section class="text"><h4>{field} <span class="sub">{len(text):,} chars · {len(s)} spans</span></h4>'
            f'<div class="body">{highlight(text, s)}</div>{span_table(s)}</section>'
        )
    dl = "".join(f"<dt>{k}</dt><dd>{html.escape(str(v))}</dd>" for k, v in meta)
    return (
        f'<article class="card" data-model="{row.model}" data-case="{row.case}" '
        f'data-dba="{html.escape(str(row.dba).lower())}" data-cut="{int(row.cut)}" id="r-{row.model}-{row.case}-{row.image_idx}">'
        f'<div class="side"><div class="img im{row.image_idx}" data-im="{row.image_idx}" title="click to enlarge"></div>'
        f'<h3>{html.escape(str(row.dba))} {flags}</h3>'
        f'<p class="prompt">{html.escape(row.prompt)}</p><dl>{dl}</dl></div>'
        f'<div class="texts">{"".join(blocks)}</div></article>'
    )


# Warning: `.card` sets `display`, and an author `display` rule wins over the
# browser rule for the `hidden` attribute. The `[hidden]` rule below keeps the
# filters working; without it a filtered card stays on the page.
CSS = """
:root{--bg:#f6f5f2;--card:#fff;--ink:#1a1a1a;--mute:#6b665f;--line:#e2ddd5;
--inference:#f3b14f99;--person_reference:#e5726499;--uncertainty:#a6c0f4cc;--image_artifact:#a7b3d2cc;--visual_evidence:#96cebf88}
@media (prefers-color-scheme:dark){:root{--bg:#16171a;--card:#202226;--ink:#e8e6e1;--mute:#a19b92;--line:#34363b;
--inference:#f3b14f66;--person_reference:#e5726477;--uncertainty:#5674b399;--image_artifact:#7a6f6399;--visual_evidence:#49857388}}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}
header.bar{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center}
header.bar fieldset{border:0;margin:0;padding:0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
header.bar legend{float:left;color:var(--mute);margin-right:4px;font-size:12px}
header.bar label{white-space:nowrap;font-size:13px}header.bar input[type=search]{padding:3px 6px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--ink)}
header.bar .count{color:var(--mute);font-size:12px}nav.pages{display:flex;gap:10px;font-size:13px}a{color:inherit}
main{max-width:1500px;margin:0 auto;padding:12px 16px 80px}
h2.image{margin:28px 0 8px;font-size:15px;color:var(--mute);font-weight:600}
.card{display:grid;grid-template-columns:260px 1fr;gap:16px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin:0 0 12px}
.side h3{margin:8px 0 2px;font-size:15px}.prompt{margin:0 0 6px;color:var(--mute);font-style:italic;font-size:12px}
.img{width:100%;aspect-ratio:1;background-size:cover;background-position:center;border-radius:6px;cursor:zoom-in}
dl{display:grid;grid-template-columns:auto 1fr;gap:1px 8px;margin:0;font-size:12px}dt{color:var(--mute)}dd{margin:0;word-break:break-all}
.flag{background:#e57264;color:#fff;border-radius:3px;padding:0 5px;font-size:11px;vertical-align:middle}
.texts{display:grid;grid-template-columns:1fr 1fr;gap:14px;min-width:0}.text{min-width:0}
.text h4{margin:0 0 4px;font-size:13px}.sub{color:var(--mute);font-weight:400}
.body{white-space:pre-wrap;word-wrap:break-word;font:12.5px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:520px;overflow:auto;border:1px solid var(--line);border-radius:6px;padding:8px;background:var(--bg)}
.empty{color:var(--mute);font-style:italic}
mark{background:transparent;color:inherit;border-radius:2px}
""" + "".join(f"body.on-{c} mark.top-{c}{{background:var(--{c})}}.dot.top-{c},.key.top-{c}{{background:var(--{c})}}" for c in CLASSES) + """
.dot,.key{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
details.spans{margin-top:6px;font-size:12px}details.spans table{border-collapse:collapse;width:100%}
details.spans td,details.spans th{border-top:1px solid var(--line);padding:2px 6px;text-align:left;vertical-align:top}.num{white-space:nowrap;color:var(--mute)}
#lightbox{display:none;position:fixed;inset:0;background:#000c;z-index:10;cursor:zoom-out}#lightbox.open{display:block}
#lightbox div{position:absolute;inset:3vh 3vw;background-size:contain;background-repeat:no-repeat;background-position:center}
@media (max-width:900px){.card{grid-template-columns:1fr}.texts{grid-template-columns:1fr}.img{max-width:320px}}
"""

JS = """
const q=s=>[...document.querySelectorAll(s)];
function apply(){const m=new Set(q('input[name=model]:checked').map(i=>i.value)),c=new Set(q('input[name=case]:checked').map(i=>i.value));
const t=document.getElementById('dba').value.trim().toLowerCase(),cut=document.getElementById('cut').checked;let n=0;
q('.card').forEach(e=>{const ok=m.has(e.dataset.model)&&c.has(e.dataset.case)&&(!t||e.dataset.dba.includes(t))&&(!cut||e.dataset.cut==='1');e.hidden=!ok;n+=ok});
q('h2.image').forEach(h=>{let s=h.nextElementSibling,any=false;while(s&&s.tagName==='ARTICLE'){any=any||!s.hidden;s=s.nextElementSibling}h.hidden=!any});
document.getElementById('n').textContent=n+' cards shown';}
// Listen to `change` too: an older Safari sends no `input` event from a checkbox.
q('header.bar input').forEach(i=>['input','change'].forEach(ev=>i.addEventListener(ev,()=>{if(i.name==='cls')document.body.classList.toggle('on-'+i.value,i.checked);else apply()})));
const lb=document.getElementById('lightbox');q('.img').forEach(d=>d.addEventListener('click',()=>{lb.firstElementChild.className='im'+d.dataset.im;lb.classList.add('open')}));
lb.addEventListener('click',()=>lb.classList.remove('open'));document.addEventListener('keydown',e=>{if(e.key==='Escape')lb.classList.remove('open')});
apply();
"""


def filter_bar(page: int, pages: int, models: list, cases: list) -> str:
    def boxes(name, values):
        return "".join(f'<label><input type="checkbox" name="{name}" value="{v}" checked> {v}</label>' for v in values)
    cls = "".join(
        f'<label><input type="checkbox" name="cls" value="{c}" checked> <span class="key top-{c}"></span>{c}</label>'
        for c in CLASSES)
    prev = f'<a href="page_{page - 1:03d}.html">← prev</a>' if page > 1 else ""
    nxt = f'<a href="page_{page + 1:03d}.html">next →</a>' if page < pages else ""
    return (
        f'<header class="bar"><nav class="pages"><a href="index.html">index</a>{prev}<b>page {page}/{pages}</b>{nxt}</nav>'
        f'<fieldset><legend>model</legend>{boxes("model", models)}</fieldset>'
        f'<fieldset><legend>case</legend>{boxes("case", cases)}</fieldset>'
        f'<fieldset><legend>highlight</legend>{cls}</fieldset>'
        f'<fieldset><input type="search" id="dba" placeholder="restaurant name"> '
        f'<label><input type="checkbox" id="cut"> cut only</label> <span class="count" id="n"></span></fieldset></header>'
    )


def page_html(title: str, body: str, image_css: str = "", script: str = "") -> str:
    classes = " ".join(f"on-{c}" for c in CLASSES)
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>'
        f'<style>{CSS}{image_css}</style></head><body class="{classes}">{body}'
        f'{"<script>" + script + "</script>" if script else ""}</body></html>'
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default=TRACES)
    ap.add_argument("--extractions", default=EXTRACTIONS, help="the merged spans; pass '' to skip the highlights")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--images", type=int, nargs="*", help="keep only these image_idx values")
    ap.add_argument("--per-page", type=int, default=50, help="images on 1 page")
    ap.add_argument("--image-px", type=int, default=512, help="width of the embedded image")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality of the embedded image")
    args = ap.parse_args()

    t = pd.read_parquet(args.traces)
    t = t[t.model.isin(args.models) & t.case.isin(args.cases)]
    if args.images:
        t = t[t.image_idx.isin(args.images)]
    if t.empty:
        raise SystemExit("no row passes the filters")
    for field in FIELDS:
        t[field] = t[field].fillna("")
    t = t.assign(
        _case=t.case.map({c: i for i, c in enumerate(CASES)}),
        _model=t.model.map({m: i for i, m in enumerate(MODELS)}),
    ).sort_values(["image_idx", "_case", "_model"])
    spans = load_spans(args.extractions) if args.extractions else {}
    print(f"[viewer] {len(t):,} rows, {t.image_idx.nunique():,} images, {sum(map(len, spans.values())):,} spans")

    models = [m for m in MODELS if m in set(t.model)]
    cases = [c for c in CASES if c in set(t.case)]
    images = list(dict.fromkeys(t.image_idx))
    chunks = [images[i:i + args.per_page] for i in range(0, len(images), args.per_page)]
    os.makedirs(args.out, exist_ok=True)
    index_rows = []
    for n, chunk in enumerate(chunks, start=1):
        part = t[t.image_idx.isin(chunk)]
        css, body = [], []
        for idx, g in part.groupby("image_idx", sort=False):
            first = g.iloc[0]
            uri, _ = image_data_uri(first.image_path, args.image_px, args.quality)
            css.append(f".im{idx}{{background-image:url({uri})}}")
            body.append(f'<h2 class="image" id="i-{idx}">image {idx} · {html.escape(str(first.dba))} · '
                        f'{html.escape(str(first.borough))} · {html.escape(os.path.basename(first.image_path))}</h2>')
            body.extend(card(r, spans) for r in g.itertuples(index=False))
        name = f"page_{n:03d}.html"
        main_html = (filter_bar(n, len(chunks), models, cases) + "<main>" + "".join(body) + "</main>"
                     + '<div id="lightbox"><div></div></div>')
        with open(os.path.join(args.out, name), "w") as fh:
            fh.write(page_html(f"F4 traces {n}/{len(chunks)}", main_html, "".join(css), JS))
        names = part.drop_duplicates("image_idx")
        index_rows.append(
            f'<tr><td><a href="{name}">page {n}</a></td><td class="num">{chunk[0]}–{chunk[-1]}</td><td>{len(part):,}</td>'
            f'<td>{html.escape(", ".join(f"{i}: {d}" for i, d in zip(names.image_idx, names.dba)))}</td></tr>'
        )
        print(f"[viewer] {name}: {len(part):,} cards, {os.path.getsize(os.path.join(args.out, name)) / 1e6:.1f} MB")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    index = (
        '<main><h1>F4 traces</h1>'
        f'<p>{len(t):,} rows: {len(images):,} images × {", ".join(cases)} × {len(models)} models. '
        f'Source <code>{html.escape(os.path.relpath(args.traces, ROOT))}</code>; spans '
        f'<code>{html.escape(os.path.relpath(args.extractions, ROOT)) if args.extractions else "none"}</code>. '
        f'Written {stamp} by <code>research/scripts/f4/build_trace_viewer.py</code> {__version__}.</p>'
        '<table class="index"><tr><th>page</th><th>image_idx</th><th>cards</th><th>restaurants</th></tr>'
        + "".join(index_rows) + "</table></main>"
    )
    index_css = ("main h1{font-size:20px}table.index{border-collapse:collapse;font-size:13px}"
                 "table.index td,table.index th{border-top:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}")
    with open(os.path.join(args.out, "index.html"), "w") as fh:
        fh.write(page_html("F4 traces", index, index_css))
    print(f"[viewer] wrote {len(chunks)} pages and index.html to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
