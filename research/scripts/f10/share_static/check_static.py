#!/usr/bin/env python3
"""Check the static share site in a headless browser.

Serves the output folder over HTTP, opens each share page in Chromium, and checks that:

- the page loads with no console error and no failed request;
- the description of the run is on the page;
- every prompt token of the run file is on the page;
- the Contrast column label is present;
- a click on a token opens its readout, and the top word at each layer, for the J++ column and
  the Contrast column, equals the first entry of the run file at that layer (0 mismatches);
- the index links every share, and each model's landing page links its runs.

    /scratch/$USER/neuronpedia-static/pwvenv/bin/python research/scripts/f10/share_static/check_static.py
    ... --explore   # print the text and the clickable elements of the first page, for the selectors
    ... --base-path /artificial-aesthetic          # a build for a site under a path
    ... --base-url https://<host>/artificial-aesthetic   # the live site; the run files come from --out
    ... --base-path /w/x/site/anon --sandbox     # as Anonymous GitHub serves it: a CSP sandbox, no CORS

Writes a screenshot for each model into --shots. Exit code 1 when a check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import http.server
import shutil
import socket
import sys
import tempfile
import threading
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

__version__ = "1.3.0"
USER = os.environ.get("USER", "user")
# The page prints a leading space as an open box and a newline as a return arrow.
GLYPHS = {"␣": " ", "↵": "\n"}
PAGE_TYPES = {"J++ READOUT": "JPP_LENS", "CONTRAST": "LOGIT_LENS"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# The header that Anonymous GitHub sends with a page under /w/: an opaque origin, so every fetch
# is cross-origin, and its files carry no CORS header.
SANDBOX_CSP = "sandbox allow-scripts allow-popups allow-forms allow-modals"
# What fails by design under the sandbox: the translation table (its failure is caught), the RSC
# payloads of a link prefetch (the router falls back to a full page load), the fonts (a font load
# is a CORS request; the page falls back to a system font), and localStorage (the webapp catches
# the SecurityError and logs it).
SANDBOX_ALLOWED = ("chinese-translations", "index.txt", "/_next/static/media/", "CORS policy", "net::ERR_FAILED", "localStorage")


class Handler(http.server.SimpleHTTPRequestHandler):
    sandbox = False
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".woff2": "font/woff2", ".json": "application/json"}

    def end_headers(self):
        if self.sandbox:  # Anonymous GitHub sets it on every response under /w/
            self.send_header("Content-Security-Policy", SANDBOX_CSP)
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass  # the browser drops connections it no longer needs; that is not an error of the site


def serve(root: Path, port: int, sandbox: bool = False) -> http.server.ThreadingHTTPServer:
    handler = partial(type("H", (Handler,), {"sandbox": sandbox}), directory=str(root))
    server = Server(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def explore(page) -> None:
    print("=== text")
    print(page.inner_text("body")[:3000])
    print("=== clickable")
    for el in page.query_selector_all("button, [role=button], [id]")[:120]:
        tag = el.evaluate("e => e.tagName")
        attrs = el.evaluate("e => Array.from(e.attributes).map(a => a.name + '=' + a.value.slice(0, 40)).join(' ')")
        text = (el.inner_text() or "")[:40].replace("\n", " ")
        print(f"{tag} | {attrs[:160]} | {text}")


def unglyph(word: str) -> str:
    for g, c in GLYPHS.items():
        word = word.replace(g, c)
    return word


def readout_from_text(text: str) -> dict[str, dict[int, str]]:
    """The top word by layer for each column, from the text of the clicked readout.

    The readout prints, for each column: a header line ("J++ READOUT" or "CONTRAST"), then
    "COUNT BY LAYER", then rows of 4 lines: the layer, the word, STEER, SWAP.
    """
    lines = text.split("\n")
    found: dict[str, dict[int, str]] = {}
    i = 0
    while i < len(lines):
        if lines[i] in PAGE_TYPES and i + 1 < len(lines) and lines[i + 1] == "COUNT BY LAYER":
            col: dict[int, str] = {}
            j = i + 2
            while j + 1 < len(lines) and lines[j].isdigit():
                col[int(lines[j])] = unglyph(lines[j + 1])
                j += 2
                while j < len(lines) and lines[j] in ("STEER", "SWAP"):
                    j += 1
            found[PAGE_TYPES[lines[i]]] = col
            i = j
        else:
            i += 1
    return found


def expected_readout(run: dict, token: dict) -> dict[str, dict[int, str]]:
    layers = run["meta"]["layers_by_type"]
    out: dict[str, dict[int, str]] = {}
    for r in token["results"]:
        out[r["type"]] = {layer: words[0] for layer, words in zip(layers[r["type"]], r["top_tokens"]) if words}
    return out


def allowed(text: str, sandbox: bool) -> bool:
    return sandbox and any(a in text for a in SANDBOX_ALLOWED)


def check_page(page, origin: str, row: dict, run: dict, shot: Path | None, sandbox: bool = False) -> tuple[list[str], int]:
    errors: list[str] = []
    console: list[str] = []
    failed: list[str] = []
    page.on("console", lambda m: console.append(m.text) if m.type == "error" and not allowed(m.text, sandbox) else None)
    page.on("requestfailed", lambda r: failed.append(r.url) if not allowed(r.url, sandbox) else None)
    page.on("response", lambda r: failed.append(f"{r.status} {r.url}") if r.status >= 400 and not allowed(r.url, sandbox) else None)
    page.goto(f"{origin}/{row['modelId']}/jlens/{row['id']}/", wait_until="networkidle")
    # The share view renders on the client. Wait for the description, up to 20 s.
    try:
        page.get_by_text(row["description"], exact=False).first.wait_for(timeout=20000)
    except Exception:
        pass
    page.wait_for_timeout(500)
    body = page.inner_text("body")
    if "Could not load this shared run" in body:
        errors.append("the run did not load")
    if row["description"] not in body:
        errors.append("the description is missing")
    if "CONTRAST" not in body:
        errors.append("the Contrast column label is missing")
    tokens = [t for t in run["tokens"] if t["kind"] == "token"]
    words = [t["token"].strip() for t in tokens if t["token"].strip()]
    missing = [w for w in words if w not in body]
    if missing:
        errors.append(f"prompt tokens missing from the page: {missing[:5]}")
    # Click the token at the first selected position, then compare the readout by layer.
    compared = 0
    pos = (row.get("selectedPositions") or [len(tokens) - 1])[0]
    target = next((t for t in tokens if t["position"] == pos), None)
    if target is None:
        errors.append(f"no token at position {pos}")
    else:
        label = target["token"].strip()
        candidates = page.get_by_text(label, exact=True)
        if candidates.count() == 0:
            errors.append(f"no element for the token {label!r} at position {pos}")
        else:
            candidates.first.click()
            page.wait_for_timeout(500)
            after = page.inner_text("body")
            if f"Position {pos}" not in after:
                errors.append(f"the readout of position {pos} did not open")
            got = readout_from_text(after)
            want = expected_readout(run, target)
            for typ, by_layer in want.items():
                if typ not in got:
                    errors.append(f"no {typ} readout on the page")
                    continue
                bad = [(layer, w, got[typ].get(layer)) for layer, w in by_layer.items() if got[typ].get(layer) != w]
                compared += len(by_layer)
                if bad:
                    errors.append(f"{typ}: {len(bad)} of {len(by_layer)} layers differ, e.g. {bad[:2]}")
    if console:
        errors.append(f"console errors: {console[:3]}")
    if failed:
        errors.append(f"failed requests: {failed[:3]}")
    if shot is not None:
        shot.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(shot), full_page=False)
    return errors, compared


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"/scratch/{USER}/neuronpedia-static/app/out")
    ap.add_argument("--shots", default=f"/scratch/{USER}/neuronpedia-static/shots")
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shares", default=None, help="the shares.json of the build; default: <out>/../data/shares.json")
    ap.add_argument("--base-path", default="", help="the path of the site under its host, as in the build")
    ap.add_argument("--base-url", default=None, help="check a live site at this URL; no local server")
    ap.add_argument("--sandbox", action="store_true", help="serve the pages with the CSP sandbox of Anonymous GitHub")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    shares = Path(args.shares) if args.shares else out.parent / "data" / "shares.json"
    rows = json.loads(shares.read_text())[: args.limit]
    models = sorted({r["modelId"] for r in rows})
    server = None
    tmp = None
    if args.base_url:
        base = args.base_url.rstrip("/")
        base_path = urlparse(base).path.rstrip("/")
    else:
        base_path = ("/" + args.base_path.strip("/")) if args.base_path.strip("/") else ""
        root = out
        if base_path:
            # Serve a parent folder that holds the output at the base path.
            tmp = Path(tempfile.mkdtemp(prefix="check_static_"))
            link = tmp / base_path.strip("/")
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(out)
            root = tmp
        port = free_port()
        server = serve(root, port, args.sandbox)
        base = f"http://127.0.0.1:{port}{base_path}"
    print(f"[check] site {base}{' (sandbox)' if args.sandbox else ''}", flush=True)
    bad = 0
    compared = 0
    shot_done: set[str] = set()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            ctx = browser.new_context(viewport={"width": 1440, "height": 900})
            if args.explore:
                page = ctx.new_page()
                row = rows[0]
                page.goto(f"{base}/{row['modelId']}/jlens/{row['id']}/", wait_until="networkidle")
                page.wait_for_timeout(800)
                explore(page)
                return 0
            page = ctx.new_page()
            page.goto(f"{base}/", wait_until="networkidle")
            links = page.locator("a[href*='/jlens/']").count()
            print(f"[check] index: {links} share links for {len(rows)} shares", flush=True)
            bad += links != len(rows)
            for m in models:
                page.goto(f"{base}/{m}/jlens/", wait_until="networkidle")
                n = page.locator(f"a[href^='{base_path}/{m}/jlens/']").count()
                want = sum(r["modelId"] == m for r in rows)
                print(f"[check] landing {m}: {n} links for {want} runs", flush=True)
                bad += n != want
            page.close()
            for row in rows:
                file = row.get("file")
                if file is None:  # an older shares.json: the file is the url without the base path
                    file = row["url"]
                    if base_path and file.startswith(base_path + "/"):
                        file = file[len(base_path) :]
                run = json.loads((out / file.lstrip("/")).read_text())
                page = ctx.new_page()
                shot = None
                if row["modelId"] not in shot_done:
                    shot = Path(args.shots) / f"{row['modelId']}.png"
                    shot_done.add(row["modelId"])
                errors, n = check_page(page, base, row, run, shot, args.sandbox)
                compared += n
                page.close()
                status = "ok" if not errors else "FAIL"
                note = "; ".join(errors) or f"loads, tokens, click, {n} layer readouts match"
                print(f"[check] {status} {row['modelId']} {file.split('/')[-1]}: {note}", flush=True)
                bad += bool(errors)
            browser.close()
    finally:
        if server is not None:
            server.shutdown()
        if tmp is not None:
            shutil.rmtree(tmp)
    print(f"[check] {len(rows) - bad} of {len(rows)} pages pass; {compared} layer readouts compared", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
