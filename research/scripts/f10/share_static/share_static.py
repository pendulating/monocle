#!/usr/bin/env python3
"""Build a static copy of the Neuronpedia share view for the F10.6 runs.

The static site keeps the Neuronpedia interface and drops the server: no database, no session,
no inference host. One page for each share row, at /{model}/jlens/{share id}/, and an index at /.
The run files sit under /runs/{model}/. The browser fetches them, as the webapp does.

The build assembles a Next.js project in BASE/app from the Neuronpedia checkout (BASE/src, pinned)
with symlinks to its node_modules, components, lib, types and prisma folders, the templates of this
folder, and the data. Then it runs `next build` with `output: 'export'`, writes MANIFEST.json and
README.txt into the output, and zips it.

    python research/scripts/f10/share_static/share_static.py            # the 24 group runs
    python research/scripts/f10/share_static/share_static.py --all      # every run in the viewer folder
    python research/scripts/f10/share_static/share_static.py --skip-build   # data and manifest only
    python research/scripts/f10/share_static/share_static.py \
        --base-path /artificial-aesthetic --dest artificial-aesthetic/site/jlens-share --no-zip
                                                               # the copy that GitHub Pages serves
    python research/scripts/f10/share_static/share_static.py --base-path /w/<id>/site/anon \
        --dest artificial-aesthetic/site/anon --inline-runs --no-zip
                        # the copy that Anonymous GitHub serves at anonymous.4open.science/w/<id>/site/anon/
                        # (a sandboxed page with no CORS, so the run goes into the page as a data: URL)

Requires node (the webapp uses Next 15) and git. See research/docs/neuronpedia-public-release.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

__version__ = "1.4.0"

ROOT = Path("/share/ju/matt/mllmsci")
HERE = Path(__file__).resolve().parent
PIN = "e2573d3"
REPO = "https://github.com/hijohnnylin/neuronpedia.git"
GROUPS = ("all", "bronx", "brooklyn", "manhattan", "queens", "staten-island", "high-yes-no", "low-yes-no")
# The rows that neuronpedia_load.py inserts into the Model table, as the global provider wants them.
MODELS = {
    "gemma-4-31b": ("Gemma-4-31B", 59, 5376, "Google"),
    "qwen3.5-9b": ("Qwen3.5-9B", 31, 4096, "Qwen"),
    "qwen3.6-27b": ("Qwen3.6-27B", 63, 5120, "Qwen"),
}
# The 2 local patches of neuronpedia_up.sh: the second column holds a contrast, not a logit lens.
PATCHES = (
    ("[LensType.LOGIT_LENS]: 'Logit Lens'", "[LensType.LOGIT_LENS]: 'Contrast'"),
    ("[LensColumn.LOGIT_LENS]: 'Logit'", "[LensColumn.LOGIT_LENS]: 'Contrast'"),
)
SITE = "f10-jlens-share"
# The text of the model tabs. The webapp hard-codes 3 tabs of neuronpedia.org; our 3 models replace them.
TAB_LABELS = {"gemma-4-31b": "Gemma 4 31B", "qwen3.5-9b": "Qwen 3.5 9B", "qwen3.6-27b": "Qwen 3.6 27B"}
# The share view fetches this table at an absolute path. A site under a base path needs the prefix.
TRANSLATIONS = "const TRANSLATIONS_URL = '"


def sh(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    print("[share_static] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkout(base: Path, base_path: str = "") -> Path:
    src = base / "src"
    if not src.exists():
        sh(["git", "clone", "--filter=blob:none", "-q", REPO, str(src)], cwd=base)
    sh(["git", "-C", str(src), "checkout", "-q", PIN], cwd=base)
    web = src / "apps" / "webapp"
    # The build writes a patched copy of the selector. Start from the clean file each time.
    sh(["git", "-C", str(src), "checkout", "-q", "--", "apps/webapp/app/[modelId]/jlens/jlens-model-selector.tsx"], cwd=base)
    if not (web / "node_modules").exists():
        sh(["npm", "ci", "--no-audit", "--no-fund"], cwd=web, env={**os.environ, "NEXT_TELEMETRY_DISABLED": "1"})
    lens = web / "lib" / "utils" / "lens.ts"
    text = lens.read_text()
    for old, new in PATCHES:
        text = text.replace(old, new)
    lens.write_text(text)
    tr = web / "lib" / "utils" / "chinese-translations.ts"
    lines = tr.read_text().split("\n")
    for i, line in enumerate(lines):
        if line.startswith(TRANSLATIONS):
            lines[i] = f"{TRANSLATIONS}{base_path}/chinese-translations-qwen.json';"
    tr.write_text("\n".join(lines))
    return web


def link(target: Path, name: Path) -> None:
    if name.is_symlink() or name.exists():
        if name.is_dir() and not name.is_symlink():
            shutil.rmtree(name)
        else:
            name.unlink()
    name.symlink_to(target)


def data_url(path: Path) -> str:
    """The run file as a data: URL, in plain JSON (readable, and the anonymizer sees the same words).

    A browser fetches a data: URL from any origin, also from a sandboxed page with no CORS.
    Only % and # need an escape; the JSON is ASCII (ensure_ascii) and has no whitespace.
    """
    text = json.dumps(json.loads(path.read_text()), separators=(",", ":"), ensure_ascii=True)
    return "data:application/json," + text.replace("%", "%25").replace("#", "%23")


def select_rows(runs: Path, models: list[str], all_runs: bool, base_path: str = "", inline: bool = False) -> list[dict]:
    """The share rows. `file` is the run file under the output; `url` is what the page fetches."""
    rows = []
    for model in models:
        for row in json.loads((runs / model / "shares.json").read_text()):
            name = row["url"].rsplit("/", 1)[-1]
            if all_runs or name[: -len(".json")] in GROUPS:
                file = f"runs/{model}/{name}"
                url = data_url(runs / model / "runs" / name) if inline else f"{base_path}/{file}"
                rows.append({**row, "file": file, "url": url})
    return rows


def patch_tabs(selector: str) -> str:
    """One tab for each of our models, in place of the 3 hard-coded tabs and the dropdown.

    The webapp's tabs name 3 models of neuronpedia.org that this site does not hold, and its
    dropdown lists the inference-enabled models, so it is empty here.
    """
    start = selector.index("const TAB_MODEL_IDS: readonly string[] = [")
    end = selector.index("\n", start)
    ids = ", ".join(f"'{m}'" for m in MODELS)
    selector = selector[:start] + f"const TAB_MODEL_IDS: readonly string[] = [{ids}];" + selector[end:]
    open_tag = '<div className="z-10 inline-flex h-7">\n'
    start = selector.index(open_tag) + len(open_tag)
    close_tag = "</DropdownMenu.Root>\n"
    end = selector.index(close_tag, start) + len(close_tag)
    buttons = []
    last = len(MODELS) - 1
    for i, m in enumerate(MODELS):
        rounding = "rounded-l-md px-2.5 sm:rounded-l-lg" if i == 0 else "rounded-r-md px-2.5 sm:rounded-r-lg" if i == last else "px-2.5"
        margin = "" if i == 0 else "-ml-px "
        buttons.append(
            "          <button\n"
            '            type="button"\n'
            f"            onClick={{() => onModelChange('{m}')}}\n"
            f"            className={{`{margin}${{tabClass(modelId === '{m}', '{rounding}')}}`}}\n"
            "          >\n"
            f"            {TAB_LABELS[m]}\n"
            "          </button>\n"
        )
    return selector[:start] + "".join(buttons) + selector[end:]


NAV_HELPER = """
// share_static: a static site has no server round trip, so every navigation is a full page load.
// A client-side push fails inside the sandbox of Anonymous GitHub (its fetch has no CORS) and the
// router then lands on the RSC payload file. The base path is inlined at build time.
const SHARE_BASE_PATH = process.env.NEXT_PUBLIC_SHARE_BASE_PATH ?? '';
const hardNavigate = (href: string) => {
  const withSlash = href.includes('?') || href.endsWith('/') ? href : `${href}/`;
  window.location.assign(withSlash.startsWith('/') ? `${SHARE_BASE_PATH}${withSlash}` : withSlash);
};
"""


def patch_navigation(client: str) -> str:
    """Every `router.push` of the share client becomes a full page load (see NAV_HELPER)."""
    anchor = "from './jlens-urls';\n"
    assert client.count(anchor) == 1
    client = client.replace(anchor, anchor + NAV_HELPER)
    n = client.count("router.push(")
    assert n == 3, n
    return client.replace("router.push(", "hardNavigate(")


def assemble(web: Path, app: Path, runs: Path, rows: list[dict]) -> None:
    app.mkdir(parents=True, exist_ok=True)
    for name in ("node_modules", "components", "lib", "types", "prisma"):
        link(web / name, app / name)
    for name in ("tailwind.config.js", "postcss.config.js", "tsconfig.json"):
        shutil.copy(web / name, app / name)
    shutil.copy(HERE / "templates" / "next.config.js", app / "next.config.js")
    a = app / "app"
    a.mkdir(exist_ok=True)
    shutil.copy(web / "app" / "globals.css", a / "globals.css")
    (a / "fonts").mkdir(exist_ok=True)
    shutil.copy(web / "app" / "fonts" / "index.ts", a / "fonts" / "index.ts")
    shutil.copy(HERE / "templates" / "layout.tsx", a / "layout.tsx")
    shutil.copy(HERE / "templates" / "index-page.tsx", a / "page.tsx")
    jl = a / "[modelId]" / "jlens"
    (jl / "[shareId]").mkdir(parents=True, exist_ok=True)
    # The client files of the route, but not its server page (it reads the database), and not
    # the 2 files that get a patched copy below.
    for f in (web / "app" / "[modelId]" / "jlens").iterdir():
        if f.name not in ("page.tsx", "jlens-model-selector.tsx", "jlens-page-client.tsx"):
            link(f, jl / f.name)
    client = patch_navigation((web / "app" / "[modelId]" / "jlens" / "jlens-page-client.tsx").read_text())
    dst = jl / "jlens-page-client.tsx"
    if dst.is_symlink():
        dst.unlink()
    dst.write_text(client)
    # The model tabs: our 3 models, no demos. The demos of neuronpedia.org do not exist here.
    selector = (web / "app" / "[modelId]" / "jlens" / "jlens-model-selector.tsx").read_text()
    start = selector.index("export const JLENS_MODEL_TOGGLE_OPTIONS = [")
    end = selector.index("\n] as const", start)
    options = ",\n".join(
        f"  {{ modelId: '{m}', displayName: '{name}', featured: true, demos: [] }}" for m, (name, _, _, _) in MODELS.items()
    )
    selector = selector[:start] + "export const JLENS_MODEL_TOGGLE_OPTIONS = [\n" + options + "," + selector[end:]
    selector = patch_tabs(selector)
    sel = jl / "jlens-model-selector.tsx"
    if sel.is_symlink():
        sel.unlink()  # never write through a link into the checkout
    sel.write_text(selector)
    shutil.copy(HERE / "templates" / "share-page.tsx", jl / "[shareId]" / "page.tsx")
    shutil.copy(HERE / "templates" / "model-page.tsx", jl / "page.tsx")
    # The tour navigates to the demos of neuronpedia.org, so hide its button.
    with open(a / "globals.css", "a") as css:
        css.write("\n/* share_static: no tour, the demos do not exist here */\n")
        css.write("#jlens-guide-button, #jlens-model-selector button.border-emerald-600 { display: none !important; }\n")
    data = app / "data"
    data.mkdir(exist_ok=True)
    (data / "shares.json").write_text(json.dumps(rows, indent=1))
    models = {
        m: {
            "id": m,
            "displayName": name,
            "displayNameShort": name,
            "layers": layers,
            "neuronsPerLayer": width,
            "owner": owner,
            "visibility": "PUBLIC",
            "instruct": True,
            "inferenceEnabled": False,
            "sourceSets": [],
        }
        for m, (name, layers, width, owner) in MODELS.items()
    }
    (data / "models.json").write_text(json.dumps(models, indent=1))
    pub = app / "public"
    if pub.exists():
        shutil.rmtree(pub)
    pub.mkdir()
    # The public files that the share view fetches or links: the translation tables and the icons.
    for f in (web / "public").iterdir():
        if f.is_file() and (f.name.startswith("chinese-translations") or f.name.startswith("favicon") or f.name == "apple-touch-icon.png"):
            shutil.copy(f, pub / f.name)
    # GitHub Pages runs Jekyll on a branch deploy, and Jekyll drops the _next folder. This file stops it.
    (pub / ".nojekyll").write_text("")
    for row in rows:
        model, name = row["file"].split("/")[-2:]
        dst = pub / row["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(runs / model / "runs" / name, dst)


def build(app: Path, base_path: str = "") -> Path:
    env = {
        **os.environ,
        "NEXT_TELEMETRY_DISABLED": "1",
        "SHARE_BASE_PATH": base_path,
        "NEXT_PUBLIC_SHARE_BASE_PATH": base_path,
        "NEXT_PUBLIC_JPP_LENS_MODEL_IDS": ",".join(MODELS),
        "NEXT_PUBLIC_URL": "http://localhost:8000",
    }
    for stale in (".next", "out"):
        if (app / stale).exists():
            shutil.rmtree(app / stale)
    sh(["npx", "--no-install", "next", "build"], cwd=app, env=env)
    return app / "out"


def manifest(out: Path, runs: Path, rows: list[dict], web: Path, base_path: str = "") -> dict:
    models = {}
    for model in MODELS:
        m = json.loads((ROOT / "data" / "f10" / "hop_all" / model / "manifest.json").read_text())
        models[model] = {k: m.get(k) for k in ("model", "layers", "vocab", "n_images", "lens", "positions")}
    files = {}
    for row in rows:
        rel = row["file"]
        run = json.loads((out / rel).read_text())
        files[rel] = {
            "sha256": sha256(out / rel),
            "share_id": row["id"],
            "description": row["description"],
            "prompt": run.get("prompt"),
            "layer_range": run.get("layerRange"),
            "n_tokens": len(run.get("tokens", [])),
        }
    return {
        "site": SITE,
        "writer": f"research/scripts/f10/share_static/share_static.py {__version__}",
        "neuronpedia_commit": PIN,
        "neuronpedia_patches": [f"{old} -> {new}" for old, new in PATCHES],
        "base_path": base_path,
        "runs_inline": rows[0]["url"].startswith("data:") if rows else False,
        "exporter": "research/scripts/f10/hop_viewer.py",
        "models": models,
        "files": files,
    }


README = """J++ Lens readouts: a static copy of the Neuronpedia share view.

Serve this folder over HTTP, then open the index. A double-click on index.html does not work,
because the pages fetch the run files.

    cd <this folder>
    python3 -m http.server 8000
    open http://localhost:8000/

Each page shows one run. Click a token to read it. Inference is off: Steer, Swap and Send do nothing.
MANIFEST.json lists the sha256 of each run file, the model, the lens, the layers and the prompt.
"""

README_BASE = """J++ Lens readouts: a static copy of the Neuronpedia share view.

This copy is built for a site under the path {base_path}/. To run the site on your own computer,
go to the repository folder (the one with serve.py) and run `python3 serve.py`; it serves the copy
in site/jlens-share. A double-click on index.html does not work: the pages fetch their files, and
a browser does not fetch from file:// URLs. Any static server works if it serves a copy at its path.

Each page shows one run. Click a token to read it. Inference is off: Steer, Swap and Send do nothing.
MANIFEST.json lists the sha256 of each run file, the model, the lens, the layers and the prompt.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=f"/scratch/{os.environ.get('USER', 'user')}/neuronpedia-static")
    ap.add_argument("--runs", default=str(ROOT / "data" / "f10" / "hop_all" / "viewer"))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--all", action="store_true", help="every run, not only the group runs")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--zip", default=str(ROOT / "data" / "f10" / "hop_all" / f"{SITE}.zip"))
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--base-path", default="", help="the path of the site under its host, e.g. /artificial-aesthetic")
    ap.add_argument("--dest", default=None, help="copy the output folder here (the old copy is removed)")
    ap.add_argument("--inline-runs", action="store_true", help="put each run into its page as a data: URL (no fetch)")
    args = ap.parse_args()
    base_path = args.base_path.rstrip("/")
    if base_path and not base_path.startswith("/"):
        base_path = "/" + base_path
    base, runs, app = Path(args.base), Path(args.runs), Path(args.base) / "app"
    base.mkdir(parents=True, exist_ok=True)
    web = checkout(base, base_path)
    rows = select_rows(runs, args.models, args.all, base_path, args.inline_runs)
    print(f"[share_static] {len(rows)} shares, base path {base_path!r}, inline {args.inline_runs}", flush=True)
    assemble(web, app, runs, rows)
    if args.skip_build:
        return
    out = build(app, base_path)
    (out / "MANIFEST.json").write_text(json.dumps(manifest(out, runs, rows, web, base_path), indent=1))
    readme = README_BASE.format(base_path=base_path, name=base_path.strip("/")) if base_path else README
    (out / "README.txt").write_text(readme)
    if args.dest:
        dest = Path(args.dest).resolve()
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out, dest)
        print(f"[share_static] copied the output to {dest}", flush=True)
    if args.no_zip:
        return
    zip_path = Path(args.zip)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f"{SITE}/{f.relative_to(out)}")
    n = sum(1 for f in out.rglob("*") if f.is_file())
    print(f"[share_static] {n} files, zip {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
