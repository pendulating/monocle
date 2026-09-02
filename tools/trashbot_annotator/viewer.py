"""TrashBot footage viewer.

A marimo app. It lists annotation windows from JSONL files, plays each
window as a spliced clip, and shows the annotations beside the clip.
When an overlays/ folder sits next to the data dir, it also plays the
annotated mp4 of a sample.

Serve it with scripts/serve_viewer.sub. The service binds port 8925.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import html
    import json
    import os
    import re
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    import marimo as mo

    from trashbot_annotator.taxonomy import (
        NORM_DIMENSIONS,
        PROXEMIC_EVENTS,
        TALK_CATEGORIES,
    )

    return (
        NORM_DIMENSIONS,
        PROXEMIC_EVENTS,
        Path,
        TALK_CATEGORIES,
        hashlib,
        html,
        json,
        mo,
        os,
        re,
        shutil,
        subprocess,
        tempfile,
    )


@app.cell
def _(Path, hashlib, mo, os, shutil, subprocess, tempfile):
    # Configuration and helpers. One topic: setup for load and splice.
    DEFAULT_DIR = Path(__file__).resolve().parent / "outputs"
    JSONL_DIR = Path(os.environ.get("TRASHBOT_VIEWER_DIR") or DEFAULT_DIR)

    ZOO_FFMPEG = Path("/share/pierson/matt/zoo/ffmpeg-libs/n7.1/bin/ffmpeg")
    CLIP_CACHE_DIR = Path(tempfile.gettempdir()) / "trashbot_viewer_clips"
    MAX_CLIP_BYTES = 6_000_000
    CAPTURE_TOKENS = (
        "interview",
        "onboard_camera",
        "overview_camera",
        "landfill",
        "recycle",
    )

    def resolve_ffmpeg():
        """Return the first usable ffmpeg binary.

        Order: TRASHBOT_FFMPEG env override, zoo build, system build.
        """
        candidates = []
        env_bin = os.environ.get("TRASHBOT_FFMPEG")
        if env_bin:
            candidates.append(Path(env_bin))
        candidates.append(ZOO_FFMPEG)
        found = shutil.which("ffmpeg")
        if found:
            candidates.append(Path(found))
        for cand in candidates:
            if cand.is_file() and os.access(cand, os.X_OK):
                return cand
        return None

    FFMPEG_BIN = resolve_ffmpeg()

    def splice_env():
        """Subprocess environment with the shared libraries of the binary."""
        env = dict(os.environ)
        if FFMPEG_BIN is None:
            return env
        lib_dir = FFMPEG_BIN.parent.parent / "lib"
        current = env.get("LD_LIBRARY_PATH", "")
        if lib_dir.is_dir() and str(lib_dir) not in current.split(os.pathsep):
            env["LD_LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{current}" if current else str(lib_dir)
        return env

    SPLICE_ENV = splice_env()

    def encoder_args(binary):
        """Return H.264 codec flags that this binary supports.

        The zoo build lacks libx264. It ships libopenh264, which reads a
        bitrate instead of a CRF. The probe needs the library path, or the
        binary fails to start and the check lies.
        """
        x264 = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30"]
        if binary is None:
            return x264
        try:
            proc = subprocess.run(
                [str(binary), "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=30,
                env=SPLICE_ENV,
            )
        except Exception:  # noqa: BLE001
            return x264
        listing = proc.stdout
        if " libx264" in listing:
            return x264
        if " libopenh264" in listing:
            return ["-c:v", "libopenh264", "-b:v", "800k"]
        return x264

    VIDEO_ARGS = encoder_args(FFMPEG_BIN)

    def splice_clip(video_path, start_s, end_s, pad=2.0, height=480):
        """Cut [start-pad, end+pad] into a small seekable MP4. Return bytes.

        The cache lives on disk. Key: path hash, start, length, height.
        A cached file above the size cap counts as a miss.
        """
        s = max(0.0, float(start_s) - float(pad))
        dur = max(0.5, float(end_s) - float(start_s) + 2.0 * float(pad))
        vid_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:10]
        stem = Path(video_path).stem or "clip"
        out = CLIP_CACHE_DIR / f"{stem}_{vid_hash}_{s:.2f}_{dur:.2f}_{height}.mp4"
        CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if (not out.exists()) or out.stat().st_size > MAX_CLIP_BYTES:
            cmd = [
                str(FFMPEG_BIN),
                "-y",
                "-ss", f"{s:.2f}",
                "-i", str(video_path),
                "-t", f"{dur:.2f}",
                "-vf", f"scale=-2:{int(height)}",
                *VIDEO_ARGS,
                "-maxrate", "800k",
                "-bufsize", "1600k",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "96k",
                "-movflags", "+faststart",
                str(out),
            ]
            proc = subprocess.run(cmd, capture_output=True, env=SPLICE_ENV)
            if proc.returncode != 0:
                tail = proc.stderr.decode(errors="replace").strip()[-400:]
                raise RuntimeError(tail or f"ffmpeg exit code {proc.returncode}")
        return out.read_bytes()

    def dataset_of(video_path):
        """Name of the corpus folder after the data segment of the path."""
        parts = Path(str(video_path)).parts
        if "data" in parts:
            i = parts.index("data")
            if i + 1 < len(parts):
                return parts[i + 1]
        return "unknown"

    def capture_type_of(video_path):
        """Capture token from the path. Unknown paths map to other."""
        for part in Path(str(video_path)).parts:
            low = part.lower()
            for token in CAPTURE_TOKENS:
                if low == token or low.startswith(token):
                    return token
        return "other"

    def make_record(raw, source_path):
        """Flatten one JSONL line into a plain record dict."""
        ann = raw.get("annotation")
        ann = ann if isinstance(ann, dict) else None
        err = raw.get("error")
        interactions = list(ann.get("interactions") or []) if ann else []
        dimensions = list(ann.get("norm_dimensions") or []) if ann else []
        events = list(ann.get("proxemic_events") or []) if ann else []
        summary = str(ann.get("summary") or "") if ann else ""
        video_path = str(raw.get("video_path") or "")
        cats = sorted({
            str(ix.get("category"))
            for ix in interactions
            if isinstance(ix, dict) and ix.get("category")
        })
        dim_names = sorted({
            str(d.get("dimension"))
            for d in dimensions
            if isinstance(d, dict) and d.get("dimension")
        })
        event_keys = sorted({
            str(e.get("event"))
            for e in events
            if isinstance(e, dict) and e.get("event")
        })
        idx = raw.get("window_idx")
        return {
            "ok": bool(err is None and ann is not None),
            "error": "" if err is None else str(err),
            "video_path": video_path,
            "sample_id": str(raw.get("sample_id") or "(no id)"),
            "window_idx": int(idx) if isinstance(idx, (int, float)) else 0,
            "t_start_s": float(raw.get("t_start_s") or 0.0),
            "t_end_s": float(raw.get("t_end_s") or 0.0),
            "model": str(raw.get("model") or ""),
            "timestamps_s": list(raw.get("timestamps_s") or []),
            "response_raw": raw.get("response_raw"),
            "interactions": [ix for ix in interactions if isinstance(ix, dict)],
            "dimensions": [d for d in dimensions if isinstance(d, dict)],
            "events": [e for e in events if isinstance(e, dict)],
            "summary": summary,
            "categories": cats,
            "dim_names": dim_names,
            "event_keys": event_keys,
            "dataset": dataset_of(video_path),
            "capture_type": capture_type_of(video_path),
            "source_file": Path(source_path).name,
        }

    def load_records(jsonl_dir):
        """Read every JSONL file under the dir. Return records, skip count.

        Bad lines drop out with a count. Every good line becomes one
        record. Duplicate keys stay visible, because each file carries
        its own variant of a window.
        """
        records = []
        skipped = 0
        files = sorted(jsonl_dir.rglob("*.jsonl")) if jsonl_dir.is_dir() else []
        for path in files:
            try:
                handle = path.open(encoding="utf-8")
            except OSError:
                skipped += 1
                continue
            with handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except ValueError:
                        skipped += 1
                        continue
                    if not isinstance(raw, dict):
                        skipped += 1
                        continue
                    records.append(make_record(raw, path))
        ordered = sorted(
            records,
            key=lambda r: (r["dataset"], r["sample_id"], r["window_idx"], r["source_file"]),
        )
        return ordered, skipped

    def fmt_clock(t):
        """Format seconds as minutes:seconds with one decimal."""
        m, s = divmod(max(0.0, float(t)), 60.0)
        return f"{int(m):02d}:{s:04.1f}"

    def esc(text):
        """Escape text for safe placement inside markdown HTML."""
        return html.escape(str(text))

    def section_label(text):
        """Small caps label. It sits above a detail block."""
        return mo.md(
            "<div style='letter-spacing:.08em;text-transform:uppercase;"
            "font-size:.72rem;font-weight:600;color:#6b7280'>" + text + "</div>"
        )

    def truncate(text, limit=90):
        """Cut long text to the limit. Add an ellipsis when cut."""
        text = str(text or "").strip()
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"

    return (
        FFMPEG_BIN,
        JSONL_DIR,
        esc,
        fmt_clock,
        load_records,
        section_label,
        splice_clip,
        truncate,
    )


@app.cell
def _(JSONL_DIR, Path, re):
    # One topic: the overlay index. It maps each sample id to its newest mp4.
    MAX_OVERLAY_BYTES = 64 * 1024 * 1024

    OVERLAY_NAME = re.compile(
        r"^(?P<sid>.+)_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{3})$"
    )

    def overlay_sample_id(path):
        """Sample id from an overlay file name. None when the name does not match."""
        hit = OVERLAY_NAME.match(Path(path).stem)
        return hit.group("sid") if hit else None

    def scan_overlays(base_dir):
        """Scan base_dir/overlays once. Map each sample id to its newest file."""
        found = {}
        root = Path(base_dir) / "overlays"
        if not root.is_dir():
            return found
        try:
            paths = sorted(root.glob("*.mp4"))
        except OSError:
            return found
        for p in paths:
            sid = overlay_sample_id(p)
            if not sid:
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            old = found.get(sid)
            if old is None or mtime >= old[0]:
                found[sid] = (mtime, p)
        return {k: v[1] for k, v in found.items()}

    OVERLAYS = scan_overlays(JSONL_DIR)

    return MAX_OVERLAY_BYTES, OVERLAYS, overlay_sample_id


@app.cell
def _(JSONL_DIR, load_records):
    # One topic: read the window records once per session.
    RECORDS, SKIPPED_LINES = load_records(JSONL_DIR)

    intro = None
    if not JSONL_DIR.is_dir():
        intro = f"The data dir does not exist: `{JSONL_DIR}`"
    elif not RECORDS:
        intro = f"No JSONL window records sit under `{JSONL_DIR}`."

    return RECORDS, SKIPPED_LINES, intro


@app.cell
def _(SKIPPED_LINES, esc, mo, intro):
    _title = mo.md(
        "# TrashBot footage viewer\n\n"
        "<span style='color:#6b7280'>Browse the deployment windows. "
        "Select a row to play its clip and read its annotations.</span>"
    )
    _note = (
        mo.callout(mo.md(esc(intro)), kind="warn")
        if intro
        else mo.md(
            f"<span style='color:#9ca3af;font-size:.8rem'>"
            f"{SKIPPED_LINES} unreadable lines skipped.</span>"
        )
    )
    mo.vstack([_title, _note], gap=1.2)
    return


@app.cell
def _(NORM_DIMENSIONS, PROXEMIC_EVENTS, RECORDS, TALK_CATEGORIES, mo):
    # One topic: filter controls. Empty multiselects mean no constraint.
    datasets = sorted({r["dataset"] for r in RECORDS})
    captures = sorted({r["capture_type"] for r in RECORDS})
    cat_keys = sorted({c for r in RECORDS for c in r["categories"]})
    cat_opts = {k: TALK_CATEGORIES.get(k, k) for k in cat_keys}
    dims_seen = {d for r in RECORDS for d in r["dim_names"]}
    dim_opts = [d for d in NORM_DIMENSIONS if d in dims_seen]
    dim_opts += sorted(dims_seen.difference(NORM_DIMENSIONS))
    event_keys_seen = sorted({e for r in RECORDS for e in r["event_keys"]})
    event_opts = {k: PROXEMIC_EVENTS.get(k, k) for k in event_keys_seen}

    dataset_dd = mo.ui.dropdown(options=["(all)", *datasets], value="(all)", label="Dataset")
    capture_dd = mo.ui.dropdown(options=["(all)", *captures], value="(all)", label="Capture type")
    cat_ms = mo.ui.multiselect(options=cat_opts, label="Talk category in window")
    dim_ms = mo.ui.multiselect(options=dim_opts, label="Norm dimension in window")
    event_ms = mo.ui.multiselect(options=event_opts, label="Proxemic event in window")
    hide_err = mo.ui.checkbox(value=False, label="Hide error windows")

    mo.vstack(
        [
            mo.hstack([dataset_dd, capture_dd, hide_err], widths=[1, 1, 1], align="end"),
            mo.hstack([cat_ms, dim_ms, event_ms], widths=[1, 1, 1], align="start"),
        ],
        gap=0.8,
    )
    return capture_dd, cat_ms, dataset_dd, dim_ms, event_ms, hide_err


@app.cell
def _(
    RECORDS,
    capture_dd,
    cat_ms,
    dataset_dd,
    dim_ms,
    event_ms,
    fmt_clock,
    hide_err,
    mo,
    truncate,
):
    # One topic: the window table. Rows carry an index into MATCHED.
    def _keep(r):
        """Test one record against every filter."""
        if hide_err.value and not r["ok"]:
            return False
        if dataset_dd.value != "(all)" and r["dataset"] != dataset_dd.value:
            return False
        if capture_dd.value != "(all)" and r["capture_type"] != capture_dd.value:
            return False
        if cat_ms.value and not set(cat_ms.value).intersection(r["categories"]):
            return False
        if dim_ms.value and not set(dim_ms.value).intersection(r["dim_names"]):
            return False
        if event_ms.value and not set(event_ms.value).intersection(r["event_keys"]):
            return False
        return True

    MATCHED = [r for r in RECORDS if _keep(r)]

    TABLE_DATA = [
        {
            "sample_id": r["sample_id"],
            "window": r["window_idx"],
            "time": f"{fmt_clock(r['t_start_s'])}\u2013{fmt_clock(r['t_end_s'])}",
            "talk": len(r["interactions"]),
            "events": len(r["events"]),
            "summary": truncate(r["summary"]),
            "source": r["source_file"],
            "status": "ok" if r["ok"] else "error",
            "_ok": r["ok"],
        }
        for r in MATCHED
    ]

    def _grey_rows(row_id, _col, _value):
        """Fade whole rows that carry an error."""
        try:
            i = int(row_id)
        except (TypeError, ValueError):
            return {}
        if 0 <= i < len(TABLE_DATA) and not TABLE_DATA[i]["_ok"]:
            return {"opacity": "0.45"}
        return {}

    table = mo.ui.table(
        data=TABLE_DATA,
        selection="single",
        initial_selection=[0] if TABLE_DATA else None,
        pagination=len(TABLE_DATA) > 15,
        page_size=15,
        max_height=420,
        wrapped_columns=["summary"],
        hidden_columns=["_ok"],
        show_column_summaries=False,
        show_download=False,
        style_cell=_grey_rows,
        label=f"{len(MATCHED)} of {len(RECORDS)} windows match the filters.",
    )
    table
    return MATCHED, table


@app.cell
def _(
    MAX_OVERLAY_BYTES,
    MATCHED,
    NORM_DIMENSIONS,
    OVERLAYS,
    PROXEMIC_EVENTS,
    Path,
    TALK_CATEGORIES,
    esc,
    fmt_clock,
    json,
    mo,
    section_label,
    table,
):
    # One topic: pick one window. Build the header, the clip source control,
    # and the annotation column.
    def _pick(value):
        """Map the table selection back to one record. Return None if empty."""
        if value is None:
            return None
        rows = value
        if hasattr(rows, "to_dict"):
            rows = rows.to_dict("records")
        rows = list(rows or [])
        if not rows:
            return None
        row = rows[0]
        want = (
            str(row.get("sample_id")),
            int(row.get("window", -1)),
            str(row.get("source")),
        )
        for r in MATCHED:
            have = (r["sample_id"], r["window_idx"], r["source_file"])
            if have == want:
                return r
        return None

    SEL = _pick(table.value)

    MODE_RADIO = None
    OVERLAY_NOTE = None
    overlay_path = None
    ann_col = None
    tail = []
    if SEL is not None:
        cand = OVERLAYS.get(SEL["sample_id"])
        if cand is not None:
            try:
                osize = cand.stat().st_size
            except OSError:
                osize = 0
            if osize > MAX_OVERLAY_BYTES:
                mb = osize / (1024 * 1024)
                OVERLAY_NOTE = mo.callout(
                    mo.md(
                        f"`{esc(str(cand))}`<br>"
                        "<span style='color:#9ca3af;font-size:.75rem'>"
                        f"{mb:.1f} MB \u00b7 copy or transcode the file "
                        "to watch it here.</span>"
                    ),
                    kind="warn",
                )
            elif osize > 0:
                overlay_path = cand
                MODE_RADIO = mo.ui.radio(
                    options=["Annotated clip", "Raw clip"],
                    value="Annotated clip",
                    inline=True,
                    label="Clip source",
                )

    if SEL is None:
        top = mo.callout(
            mo.md("Select a row in the table. The clip and its annotations appear here."),
            kind="info",
        )
    else:
        a, b = SEL["t_start_s"], SEL["t_end_s"]
        head = mo.md(
            f"## {esc(SEL['sample_id'])} \u00b7 window {SEL['window_idx']}\n\n"
            f"<span style='color:#6b7280;font-size:.85rem'>"
            f"{fmt_clock(a)}\u2013{fmt_clock(b)} \u00b7 "
            f"{len(SEL['timestamps_s'])} frames \u00b7 "
            f"{esc(SEL['model'])} \u00b7 "
            f"{esc(SEL['dataset'])} / {esc(SEL['capture_type'])} \u00b7 "
            f"source: {esc(SEL['source_file'])}</span>"
        )
        top_blocks = [head]
        if MODE_RADIO is not None:
            top_blocks.append(MODE_RADIO)
        elif OVERLAY_NOTE is not None:
            top_blocks.append(OVERLAY_NOTE)
        top = mo.vstack(top_blocks, gap=0.8)

        # Annotation column. Error records show the error instead.
        if not SEL["ok"]:
            ann_col = mo.vstack(
                [
                    section_label("Annotation"),
                    mo.callout(
                        mo.md("**This window carries an error.** " + esc(SEL["error"])),
                        kind="warn",
                    ),
                ],
                gap=0.5,
            )
        else:
            summary_md = mo.vstack(
                [section_label("Summary"), mo.callout(mo.md(esc(SEL["summary"])), kind="neutral")],
                gap=0.4,
            )

            ix_chunks = []
            for n, ix in enumerate(SEL["interactions"], 1):
                cat = str(ix.get("category") or "?")
                lab = TALK_CATEGORIES.get(cat, cat)
                who = str(ix.get("who") or "").strip()
                ev = str(ix.get("evidence") or "").strip()
                address = str(ix.get("address") or "").strip()
                seg = f"{n}. **{esc(lab)}**"
                if address:
                    seg += (
                        f" <span style='color:#6b7280;font-size:.85em'>"
                        f"[{esc(address)}]</span>"
                    )
                if who:
                    seg += f"<br><span style='color:#6b7280'>who: {esc(who)}</span>"
                if ev:
                    seg += f"<br>{esc(ev)}"
                ix_chunks.append(seg)
            ix_body = "\n".join(ix_chunks) if ix_chunks else "*None recorded.*"
            ix_md = mo.vstack([section_label("Interactions"), mo.md(ix_body)], gap=0.4)

            px_chunks = []
            for n, pe in enumerate(SEL["events"], 1):
                key = str(pe.get("event") or "?")
                lab = PROXEMIC_EVENTS.get(key, key)
                who = str(pe.get("who") or "").strip()
                ev = str(pe.get("evidence") or "").strip()
                zone = str(pe.get("zone") or "").strip()
                dist = pe.get("distance_m")
                dist_txt = ""
                if isinstance(dist, (int, float)) and not isinstance(dist, bool):
                    dist_txt = f"{float(dist):.1f} m"
                tag = " · ".join(t for t in (dist_txt, zone) if t)
                seg = f"{n}. **{esc(lab)}**"
                if tag:
                    seg += (
                        f" <span style='color:#6b7280;font-size:.85em'>"
                        f"[{esc(tag)}]</span>"
                    )
                if who:
                    seg += f"<br><span style='color:#6b7280'>who: {esc(who)}</span>"
                if ev:
                    seg += f"<br>{esc(ev)}"
                px_chunks.append(seg)
            px_body = "\n".join(px_chunks) if px_chunks else "*None recorded.*"
            px_md = mo.vstack([section_label("Proxemic events"), mo.md(px_body)], gap=0.4)

            order = {name: i for i, name in enumerate(NORM_DIMENSIONS)}
            dim_rows = []
            for d in SEL["dimensions"]:
                name = str(d.get("dimension") or "?")
                try:
                    rel = max(0, min(3, int(d.get("relevance") or 0)))
                except (TypeError, ValueError):
                    rel = 0
                ev = str(d.get("evidence") or "").strip()
                dim_rows.append((rel, order.get(name, 99), name, ev))
            dim_rows.sort(key=lambda t: (-t[0], t[1], t[2]))
            dim_segs = []
            for rel, _, name, ev in dim_rows:
                dots = "\u25cf" * rel + "\u25cb" * (3 - rel)
                ink = "#111827" if rel else "#9ca3af"
                seg = (
                    f"<strong style='color:{ink};letter-spacing:.12em'>{dots}</strong> "
                    f"<span style='color:{ink}'>{esc(name)}</span>"
                )
                if ev:
                    seg += (
                        "<br><span style='color:#9ca3af;font-size:.85em'>"
                        + esc(ev)
                        + "</span>"
                    )
                dim_segs.append(f"<div style='margin-bottom:.45em'>{seg}</div>")
            legend = (
                "<span style='color:#9ca3af;font-size:.75rem'>"
                "Relevance scale 0 to 3. Three filled dots mark strong relevance.</span>"
            )
            dim_md = mo.vstack(
                [section_label("Norm dimensions"), mo.md("".join(dim_segs)), mo.md(legend)],
                gap=0.4,
            )

            ann_col = mo.vstack([summary_md, ix_md, px_md, dim_md], gap=1.2)

        tail = []
        raw = SEL["response_raw"]
        if raw:
            try:
                pretty = json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
                body = f"```json\n{pretty}\n```"
            except (TypeError, ValueError):
                body = f"```\n{raw}\n```"
            tail.append(mo.accordion({"Raw model response": mo.md(body)}))

    top
    return MODE_RADIO, SEL, ann_col, overlay_path, tail


@app.cell
def _(MODE_RADIO, Path, SEL, ann_col, esc, mo, overlay_path, section_label, splice_clip, tail):
    # One topic: the media column. The toggle picks the annotated or the raw clip.
    media = None
    if SEL is not None:
        mode = MODE_RADIO.value if MODE_RADIO is not None else "Raw clip"
        vp = SEL["video_path"]
        _a, _b = SEL["t_start_s"], SEL["t_end_s"]

        if not vp:
            clip_col = mo.callout(mo.md("The record carries no video path."), kind="warn")
        elif mode == "Annotated clip" and overlay_path is not None:
            try:
                oblob = overlay_path.read_bytes()
                clip_col = mo.vstack(
                    [
                        section_label("Clip \u00b7 annotated"),
                        mo.video(src=oblob, controls=True, width=500),
                        mo.md(
                            "<span style='color:#9ca3af;font-size:.75rem'>"
                            f"{len(oblob) // 1024} KB \u00b7 full overlay</span>"
                        ),
                    ],
                    gap=0.5,
                )
            except OSError as exc:
                clip_col = mo.callout(
                    mo.md(f"**Overlay clip failed to load.** `{esc(str(exc)[:400])}`"),
                    kind="warn",
                )
        elif not Path(vp).exists():
            clip_col = mo.callout(
                mo.md(f"Source video is not on disk: `{esc(vp)}`"), kind="warn"
            )
        else:
            try:
                blob = splice_clip(vp, _a, _b)
                clip_col = mo.vstack(
                    [
                        section_label("Clip"),
                        mo.video(src=blob, controls=True, width=500),
                        mo.md(
                            "<span style='color:#9ca3af;font-size:.75rem'>"
                            f"{len(blob) // 1024} KB \u00b7 pad 2 s</span>"
                        ),
                    ],
                    gap=0.5,
                )
            except Exception as exc:  # noqa: BLE001
                clip_col = mo.callout(
                    mo.md(f"**Clip splice failed.** `{esc(str(exc)[:400])}`"),
                    kind="warn",
                )

        media = mo.vstack(
            [
                mo.hstack([clip_col, ann_col], widths=[2, 3], gap=2.2, align="start"),
                *tail,
            ],
            gap=1.4,
        )
    media
    return


@app.cell
def _(FFMPEG_BIN, JSONL_DIR, OVERLAYS, RECORDS, SKIPPED_LINES, mo):
    # One topic: quiet footer with run facts.
    bin_txt = str(FFMPEG_BIN) if FFMPEG_BIN else "not found"
    ov_txt = f" \u00b7 {len(OVERLAYS)} overlay clips" if OVERLAYS else ""
    mo.md(
        "<span style='color:#9ca3af;font-size:.75rem'>"
        f"data dir: <code>{JSONL_DIR}</code> \u00b7 "
        f"{len(RECORDS)} windows \u00b7 "
        f"{SKIPPED_LINES} unreadable lines skipped{ov_txt} \u00b7 "
        f"ffmpeg: <code>{bin_txt}</code></span>"
    )
    return


if __name__ == "__main__":
    app.run()

