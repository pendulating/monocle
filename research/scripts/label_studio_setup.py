#!/usr/bin/env python
"""Create the gold-set project in Label Studio and import the blind tasks.

The server must run (`bash research/scripts/label_studio_gold.sh`). This script reads
the API token from `data/label_studio/credentials.env`, creates 1 project
with the labelling interface of `data/gold_set/label_config.xml`, and
imports `data/gold_set/tasks_blind.json`. A second call finds the project by
its title and imports nothing twice.

Use:
    python research/scripts/label_studio_setup.py
    python research/scripts/label_studio_setup.py --title "Urban cues gold set v2" --port 8925

The predictions of the extractors are NOT imported. The annotators label
blind, and `research/scripts/score_gold_set.py` compares the export with each
`data/gold_set/predictions_<arm>.json` after.

See `research/docs/langextract-trace-extraction.md`, "The gold set".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD = os.path.join(ROOT, "data", "gold_set")
CREDS = os.path.join(ROOT, "data", "label_studio", "credentials.env")

__version__ = "1.2.0"

INSTRUCTIONS = """<h3>The urban-cues gold set, schema v3 (IC classes)</h3>
<p>Each task is the private reasoning a vision-language model wrote while it compared two
street photographs of New York City, Image A and Image B, shown above the text as the model
saw them. Mark what the model <b>says</b>, not what you think of the images.</p>

<h4>How to mark a span</h4>
<ol>
<li>Select the text and pick its class. Mark the <b>whole clause</b>, first word to end
punctuation. Leave out a list marker ("- ") and a bold header ("**Image A:**"). Keep the
words "Image A" or "Image B" when they open the clause.</li>
<li>Spans <b>nest</b>. An inference that rests on things the model names as visible holds
those things: mark the inference clause, and mark the dimension inside it too. A perspective
word sits inside the inference or dimension that applies it. One span may carry 2 classes
(a sentence that states a standard and also sets A against B is a perspective and a
comparison).</li>
<li>Set the attributes in the panel below the text. The free-text names (dimension,
inference, perspective; the target of a dismissal) are short phrases: reuse a name when
the same attribute comes back.</li>
<li>The opening sentence restates the question. Mark nothing in it. A later sentence in
which the model reads the prompt again is an uncertainty (marker: reconsideration).</li>
<li>A repeated thought is one span: mark the first occurrence only.</li>
<li>The <b>Notes</b> box is free text: depiction (does an image show the item the prompt
names?), or anything else worth recording about this trace. Leave it empty otherwise.</li>
</ol>

<h4>The classes, in the order the reasoning uses them</h4>
<p><b>dimension</b> (IC category 1, descriptive): a single instance that describes the
scene, graspable by sight: an object, a material, a colour, a count, legible text, a
lighting condition ("well-lit"), the layout of the street. Attributes: image, valence, name
(architecture, lighting, greenery, vehicles, signage, construction, ...).</p>
<p><b>inference</b> (IC category 1, evaluative): an evaluative observation, or a claim that
reads knowledge of the world into what is seen: modern, historic, high-end, neglected,
well-maintained, standard, a private school, affluent, safe, gritty, feels city-like. To
call a building modern takes knowledge of what modern is. Attributes: image, kind, name
(maintenance, modernity, prestige, school type, hazard, aesthetic sentiment, ...).</p>
<p><b>perspective</b> (IC category 2): the evaluative standard by which the model informs its
judgment. It <b>follows from a dimension</b> and explains how or why that dimension counts
for or against the scene: a deductive mode of valuation, not an observation. Three shapes:
a general rule ("Affluence often correlates with perceived safety"), a standard applied to
this pair ("They both look pretty standard for NYC"), or <b>the mode word itself inside the
inference</b>: classic, high-end, prestigious, modern, historic, utilitarian, functional,
well-funded, well-maintained, clean. Mark the word or short phrase as its own span inside
the clause that holds it. The test: "well-lit" that names the lighting is a dimension; the
safety the model reads out of the lighting is the perspective. Ask of each evaluative
attribute: is this the standard on which the model grounds its eventual decision?
Attributes: name (the mode), favors (the image it counts FOR), wins (the decision follows
it), source (world knowledge, task convention).</p>
<p><b>weighing</b> (IC category 3): two standards or two dimensions set against each other,
and which counts more ("construction hazards usually outweigh a guard"). Attributes:
mechanism, favors.</p>
<p><b>justification</b>: the span that says WHY, for a weighing, a perspective, a doubt, or a
dismissal. Only when the text gives a reason; "overall" is not a reason. A condition under
which the weight would change takes role = condition. Attributes: of, role.</p>
<p><b>dismissal</b>: a cue or a standard set aside as not relevant. When the model doubts that
the image shows the thing the prompt names ("I don't see a library in Image A"), about =
premise. Attributes: target, about.</p>
<p><b>verdict</b>: an answer the model later restates or changes ("Therefore, the school in
Image A is Less good"). <b>decision</b>: the one sentence that states the FINAL answer, even
when it hedges ("Same seems most appropriate."). A trace that decides three times gives two
verdicts and one decision. Attribute: label.</p>
<p><b>comparison</b>: an explicit A-against-B statement on a quality. Attributes: direction
(the image with MORE of the quality), dimension, hedged.</p>
<p><b>uncertainty</b>: a hedge, a doubt, or a reconsideration ("Wait", "Actually", "Let me
look again", a re-reading of the prompt, weighing two labels). Attributes: reason, marker
(qualifier, contrastive, explicit doubt, reconsideration).</p>
<p><b>person_reference</b>: a statement about the people in a photograph. <b>image_artifact</b>:
a statement about the photograph itself (angle, blur, obstruction, time of day, weather).</p>"""


def read_creds() -> dict:
    out = {}
    with open(CREDS) as fh:
        for line in fh:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                out[k] = v
    return out


def _authenticated_session(base: str, creds: dict) -> requests.Session:
    """Give a session that the API accepts: by token, or by the login form."""
    s = requests.Session()
    s.headers["Authorization"] = f"Token {creds['LABEL_STUDIO_USER_TOKEN']}"
    r = s.get(f"{base}/api/projects", timeout=30)
    if r.status_code != 401:
        return s
    s = requests.Session()
    login = s.get(f"{base}/user/login/", timeout=30)
    login.raise_for_status()
    csrf = s.cookies.get("csrftoken", "")
    r = s.post(
        f"{base}/user/login/",
        data={"email": creds["LABEL_STUDIO_USERNAME"], "password": creds["LABEL_STUDIO_PASSWORD"],
              "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{base}/user/login/"}, timeout=30, allow_redirects=True,
    )
    if "sessionid" not in s.cookies:
        raise SystemExit(f"login failed for {creds['LABEL_STUDIO_USERNAME']}: {r.status_code}")
    s.headers["X-CSRFToken"] = s.cookies.get("csrftoken", "")
    s.headers["Referer"] = base
    print("[setup] the legacy API token is off; the session login is in use")
    return s


def main() -> int:
    global GOLD
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--title", default="Urban cues gold set v3 (confident pairs)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--update-config", action="store_true",
                    help="when the project exists: replace its labelling interface with the set's "
                         "label_config.xml and its instructions with the current text")
    ap.add_argument("--gold-dir", default=GOLD,
                    help="the set to import: label_config.xml and tasks_blind.json live here; "
                         "must sit under data/gold_set, the local-files root")
    args = ap.parse_args()
    GOLD = os.path.abspath(args.gold_dir)

    creds = read_creds()
    port = args.port or int(creds.get("LABEL_STUDIO_PORT", 8925))
    base = f"http://{args.host}:{port}"

    for i in range(60):
        try:
            r = requests.get(f"{base}/health", timeout=5)
            if r.ok:
                break
        except requests.RequestException:
            pass
        time.sleep(3)
    else:
        raise SystemExit(f"no Label Studio at {base} after 3 minutes")

    # Label Studio 1.16 and later ship with the legacy API token OFF for a new
    # organization, thus `Authorization: Token ...` answers 401. The web
    # login still works for a session, and the API accepts the session
    # cookie with the CSRF header. Try the token first, then log in.
    api = _authenticated_session(base, creds)

    r = api.get(f"{base}/api/projects", timeout=30)
    r.raise_for_status()
    projects = r.json()
    projects = projects.get("results", projects) if isinstance(projects, dict) else projects
    existing = [p for p in projects if p.get("title") == args.title]
    if existing:
        project = existing[0]
        print(f"[setup] project exists: id {project['id']} '{args.title}', "
              f"{project.get('task_number', '?')} tasks")
        if args.update_config:
            with open(os.path.join(GOLD, "label_config.xml")) as fh:
                label_config = fh.read()
            r = api.patch(f"{base}/api/projects/{project['id']}",
                          json={"label_config": label_config, "expert_instruction": INSTRUCTIONS,
                                "show_instruction": True}, timeout=60)
            if not r.ok:
                raise SystemExit(f"config update failed: {r.status_code} {r.text[:600]}")
            print(f"[setup] updated the labelling interface and the instructions of project {project['id']}")
        print(f"[setup] open {base}/projects/{project['id']}")
        return 0

    with open(os.path.join(GOLD, "label_config.xml")) as fh:
        label_config = fh.read()
    payload = {
        "title": args.title,
        "description": "30 reasoning traces, 15 of each judge, with the 2 photographs of each pair, "
                       "labelled blind by 2 annotators. The extractors are scored against this set with "
                       "research/scripts/score_gold_set.py and research/notebooks/cvpr/master/gold_alignment.py.",
        "label_config": label_config,
        "expert_instruction": INSTRUCTIONS,
        "show_instruction": True,
        "show_skip_button": False,
        "enable_empty_annotation": False,
        "show_annotation_history": True,
        # No model, thus no prediction to show: the labelling is blind.
        "show_collab_predictions": False,
        "sampling": "Sequential sampling",
        "maximum_annotations": 2,
    }
    r = api.post(f"{base}/api/projects", json=payload, timeout=60)
    if not r.ok:
        raise SystemExit(f"project creation failed: {r.status_code} {r.text[:500]}")
    project = r.json()
    pid = project["id"]
    print(f"[setup] created project id {pid}: '{args.title}'")

    # Label Studio serves a local file only when it sits under a Local Files
    # storage that a project registered. Thus the images folder is registered
    # here, with no sync: the tasks come from the JSON, not from the folder.
    storage = {
        "project": pid, "title": "gold-set images",
        "path": os.path.join(GOLD, "images"), "use_blob_urls": False,
        "regex_filter": None, "description": "the 2 photographs of each task, served to the Image tags",
    }
    r = api.post(f"{base}/api/storages/localfiles", json=storage, timeout=60)
    if r.ok:
        print(f"[setup] registered the images folder as local storage {r.json().get('id')}")
    else:
        print(f"[setup] !! local storage registration failed: {r.status_code} {r.text[:300]}", file=sys.stderr)

    with open(os.path.join(GOLD, "tasks_blind.json")) as fh:
        tasks = json.load(fh)
    r = api.post(f"{base}/api/projects/{pid}/import", json=tasks, timeout=300)
    if not r.ok:
        raise SystemExit(f"import failed: {r.status_code} {r.text[:500]}")
    info = r.json()
    print(f"[setup] imported {info.get('task_count', len(tasks))} tasks "
          f"({info.get('annotation_count', 0)} annotations, {info.get('prediction_count', 0)} predictions)")
    print(f"[setup] open {base}/projects/{pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
