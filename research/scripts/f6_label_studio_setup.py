#!/usr/bin/env python
"""Create the F6 project in Label Studio and import the 100 image tasks.

The server must run (`sbatch research/scripts/label_studio_f6.sub`). This script reads
the login from `data/label_studio/credentials.env`, creates 1 project with
the interface of `data/f6_restaurant_depiction/label_config.xml`, registers
the images folder as local storage, and imports
`data/f6_restaurant_depiction/tasks.json`. A second call finds the project by
its title and imports nothing twice.

Build the sample first: `research/scripts/f6_build_restaurant_depiction_sample.py`.

Use:
    python research/scripts/f6_label_studio_setup.py --port 8927

See `vlm-narratives-docs/facct-runbook.md`, section F6.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_studio_setup import _authenticated_session, read_creds  # noqa: E402

ROOT = "/share/ju/matt/mllmsci"
SET_DIR = os.path.join(ROOT, "data", "f6_restaurant_depiction")

__version__ = "1.0.0"

# The criterion in the words of the decision of 2026-10-23 (D28).
INSTRUCTIONS = """<h3>F6: restaurant depiction check</h3>
<p>Each task is 1 street-level image. A geometric filter says the image faces the
restaurant named above it.</p>
<p><b>Pass</b>: the image truly depicts a restaurant, in high definition, from close
proximity. <b>Fail</b>: any of the 3 does not hold.</p>
<p>Use the notes box for the reason of a Fail, or for anything else worth recording.</p>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--title", default="F6 restaurant depiction check (100 images)")
    ap.add_argument("--port", type=int, default=8927)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--set-dir", default=SET_DIR)
    ap.add_argument("--images-dir", default=None,
                    help="the local-files folder of the images; default <set-dir>/images")
    ap.add_argument("--description", default="100 restaurant images, 1 for each restaurant, from the "
                    "top 10% of attribution_confidence. Pass or Fail, and notes.")
    args = ap.parse_args()

    args.set_dir = os.path.abspath(args.set_dir)
    if args.images_dir:
        args.images_dir = os.path.abspath(args.images_dir)
    base = f"http://{args.host}:{args.port}"
    for _ in range(60):
        try:
            if requests.get(f"{base}/health", timeout=5).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(3)
    else:
        raise SystemExit(f"no Label Studio at {base} after 3 minutes")

    api = _authenticated_session(base, read_creds())

    r = api.get(f"{base}/api/projects", timeout=30)
    r.raise_for_status()
    projects = r.json()
    projects = projects.get("results", projects) if isinstance(projects, dict) else projects
    existing = [p for p in projects if p.get("title") == args.title]
    if existing:
        pid = existing[0]["id"]
        print(f"[f6] project exists: id {pid}, {existing[0].get('task_number', '?')} tasks")
        print(f"[f6] open {base}/projects/{pid}")
        return 0

    with open(os.path.join(args.set_dir, "label_config.xml")) as fh:
        label_config = fh.read()
    payload = {
        "title": args.title,
        "description": args.description,
        "label_config": label_config,
        "expert_instruction": INSTRUCTIONS,
        "show_instruction": True,
        "show_skip_button": False,
        "enable_empty_annotation": False,
        "show_collab_predictions": False,
        "sampling": "Sequential sampling",
        "maximum_annotations": 1,
    }
    r = api.post(f"{base}/api/projects", json=payload, timeout=60)
    if not r.ok:
        raise SystemExit(f"project creation failed: {r.status_code} {r.text[:500]}")
    pid = r.json()["id"]
    print(f"[f6] created project id {pid}: '{args.title}'")

    # Label Studio serves a local file only under a Local Files storage that a
    # project registered. No sync: the tasks come from the JSON.
    storage = {
        "project": pid, "title": "f6 images", "path": args.images_dir or os.path.join(args.set_dir, "images"),
        "use_blob_urls": False, "regex_filter": None,
        "description": "the 100 images of the F6 sample",
    }
    r = api.post(f"{base}/api/storages/localfiles", json=storage, timeout=60)
    if not r.ok:
        raise SystemExit(f"local storage registration failed: {r.status_code} {r.text[:300]}")

    with open(os.path.join(args.set_dir, "tasks.json")) as fh:
        tasks = json.load(fh)
    r = api.post(f"{base}/api/projects/{pid}/import", json=tasks, timeout=300)
    if not r.ok:
        raise SystemExit(f"import failed: {r.status_code} {r.text[:500]}")
    print(f"[f6] imported {r.json().get('task_count', len(tasks))} tasks")
    print(f"[f6] open {base}/projects/{pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
