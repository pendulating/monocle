#!/usr/bin/env python
"""F4: print the examples block of `aesthetic_descriptors_<case>_v6.yaml` (D130).

Version 6 is the x2 round of the descriptor schema. It applies the advisor
review of 2026-10-26: short examples (1 for each input shape), a `none` class
for a text with no descriptor, "extract and tag" in place of most "skip"
rules (the new aspects `evaluation`, `size`, `composition`), and a `target`
with 3 values that match the 3 subjects of the street cue schema
(`place`, `current_photo`, `planned_photo`).

Each span is a slice of its text, found in order after the span before it,
thus it is an exact copy. A span is a literal string (`L`), the slice from a
start anchor to an end anchor with the end anchor included (`R`), or a
literal that comes after an anchor (`F`; use it when the same word occurs
earlier in another sense). Each example text is a cut of a real F4 text
from `f4_<case>_x2.parquet`; a cut keeps the text length small and copies a
chunk shape of the chunked run (`max_char_buffer: 6000`).

No example text may come from an x2 smoke set
(`data/f4_traces/extract_input/x2_smoke_doc_ids.json`); the script stops if
one does.

Change a span here, not by hand in the YAML. Then paste the output under
`prompt_description` of both v6 files and raise the version.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/build_descriptor_v6_examples.py --case restaurant > ex.yaml
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/build_descriptor_v6_examples.py --case street --ids
"""
import argparse
import json
import sys

import pandas as pd
import yaml

ROOT = "/share/ju/matt/mllmsci"
INPUT = ROOT + "/data/f4_traces/extract_input/f4_{}_x2.parquet"
SMOKE = ROOT + "/data/f4_traces/extract_input/x2_smoke_doc_ids.json"

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["restaurant", "street"], required=True)
ap.add_argument("--ids", action="store_true", help="print the example doc_ids, not the YAML")
ARGS = ap.parse_args()
TEXTS = pd.read_parquet(INPUT.format(ARGS.case)).set_index("doc_id").text
SMOKE_IDS = set(sum(json.load(open(SMOKE)).values(), []))

PL, CP, PP = "place", "current_photo", "planned_photo"


def L(s):
    return ("L", s, None)


def R(start, end):
    return ("R", start, end)


def F(s, after):
    return ("F", s, after)


def cut(doc_id, start=None, end_before=None):
    """Give the part of a text from `start` (included) to `end_before` (left out)."""
    if doc_id in SMOKE_IDS:
        sys.exit(f"{doc_id} is in an x2 smoke set")
    t = TEXTS[doc_id]
    i = t.index(start) if start else 0
    j = t.index(end_before, i) if end_before else len(t)
    return t[i:j].strip()


def build(doc_id, text, specs):
    """specs: (span, head, vibe, slot, aspect, negated, target) tuples in text order, or ("none", span)."""
    out, cursor = [], 0
    for spec in specs:
        if spec[0] == "none":
            if not text.startswith(spec[1]):
                sys.exit(f"{doc_id}: the none span must open the text")
            out.append({"extraction_class": "none", "extraction_text": spec[1]})
            continue
        span, head, vibe, slot, aspect, negated, target = spec
        kind, a, b = span
        if kind == "F":
            cursor = text.index(b, cursor) + len(b)
        i = text.index(a, cursor)
        j = i + len(a) if kind in ("L", "F") else text.index(b, i) + len(b)
        cursor = j
        out.append({"extraction_class": "descriptor", "extraction_text": text[i:j],
                    "attributes": {"head": head, "vibe": vibe, "slot": slot, "aspect": aspect,
                                   "negated": negated, "target": target}})
    return {"doc_id": doc_id, "text": text, "extractions": out}


def d(span, head, aspect, target=PL, vibe="no", slot="none", negated="no"):
    """Shorthand: most spans have vibe no, slot none, negated no, target place."""
    return (span, head, vibe, slot, aspect, negated, target)


ex = []

if ARGS.case == "restaurant":
    # 1. gemma-4-12b trace, image 538, the start. "User's Question" and
    #    "Problem" echo the request: nothing. The "Image:" line is the
    #    model's own reading: "low-resolution" is a quality of the photo it
    #    got (current_photo), "small" is the size of the place. Business
    #    types from the signs. "blurry" is true only of the frame.
    doc = "gemma-4-12b|restaurant|538|reasoning"
    t = cut(doc, end_before="3.  **Identify")
    ex.append(build(doc, t, [
        d(L("low-resolution"), "low-resolution", "composition", CP),
        d(L("small"), "small", "size"),
        d(L("pizza place"), "pizza place", "business_type"),
        d(L("food joint"), "food joint", "business_type"),
        d(L("deli/grocery/cafe"), "deli/grocery/cafe", "business_type"),
        d(L("local eatery"), "local eatery", "business_type"),
        d(L("pub"), "pub", "business_type"),
        d(L("blurry"), "blurry", "composition", CP),
        d(L("market"), "market", "business_type"),
    ]))

    # 2. qwen3.5-9b trace, image 656, a middle cut. Class and standing from
    #    the building ("working-class", "immigrant"). In "an immigrant
    #    neighborhood" only the descriptor. Skip: "brown brick", "flat roof
    #    line", "air conditioning units" (a color, a material, parts of a
    #    building), "typical of New York City", and the plain land use
    #    "residential/commercial mixed".
    doc = "qwen3.5-9b|restaurant|656|reasoning"
    t = cut(doc, "**3. Contextualize", "**4. Synthesize")
    ex.append(build(doc, t, [
        d(L("older"), "older", "appearance"),
        d(L("functional"), "functional", "appearance"),
        d(L("working-class"), "working-class", "social_register"),
        d(L("immigrant"), "immigrant", "social_register"),
        d(L("established"), "established", "social_register"),
        d(L("a bit dated"), "dated", "appearance"),
        d(L("weathered"), "weathered", "appearance"),
    ]))

    # 3. Negative: the same gemma-4-12b trace (image 538), the plan of the
    #    answer. A plan, the user, and the task hold no descriptor: `none`.
    doc = "gemma-4-12b|restaurant|538|reasoning"
    t = cut(doc, "4.  **Formulate", "    *   *Note limitations")
    ex.append(build(doc, t, [("none", "4.  **Formulate a Strategy")]))

    # 4. gemma-4-12b answer, image 195, the windows and the curb appeal.
    #    Qualities of parts of the place count ("distinctive", "manicured");
    #    their size and light do not ("large front windows", "brightly lit",
    #    "large, uniform planters"). The words before "atmosphere" share its
    #    frame noun.
    doc = "gemma-4-12b|restaurant|195|response"
    t = cut(doc, "*   **Windows:**", "### **The Vibe**")
    ex.append(build(doc, t, [
        d(L("distinctive"), "distinctive", "appearance"),
        d(L("artistic"), "artistic", "appearance"),
        d(L("theatrical"), "theatrical", "atmosphere", vibe="yes", slot="atmosphere"),
        d(L("high-end"), "high-end", "social_register", vibe="yes", slot="atmosphere"),
        d(L("mysterious atmosphere"), "mysterious", "atmosphere", vibe="yes", slot="atmosphere"),
        d(L("manicured"), "manicured", "appearance"),
        d(L("polished"), "polished", "appearance", slot="look"),
        d(L("sophisticated look"), "sophisticated", "appearance", slot="look"),
    ]))

    # 5. qwen3.8-27b answer, image 233, the end. The worn canopy is still worn
    #    in the street (place); the photo that is "dim" and "low-res" is the
    #    frame (current_photo). "small" is the size of the place: it counts.
    #    Skip the plain "stone building" and the menu items.
    doc = "qwen3.8-27b|restaurant|233|response"
    t = cut(doc, "The overall vibe is")
    ex.append(build(doc, t, [
        d(L("working-class"), "working-class", "social_register", vibe="yes", slot="vibe"),
        d(L("shuttered"), "shuttered", "appearance"),
        d(L("graffiti-tagged"), "graffiti-tagged", "appearance"),
        d(L("worn"), "worn", "appearance"),
        d(L("tattered"), "tattered", "appearance"),
        d(L("classic"), "classic", "appearance"),
        d(L("small"), "small", "size"),
        d(L("family-run"), "family-run", "social_register"),
        d(L("Chinese spot"), "chinese spot", "business_type"),
        d(L("sit-down"), "sit-down", "business_type", negated="yes"),
        d(L("dim"), "dim", "composition", CP),
        d(L("low-res"), "low-res", "composition", CP),
    ]))

    # 6. llama-3.2-11b answer, image 92, whole. A short Llama answer of
    #    praise and warmth. Generic praise is a descriptor too ("visual
    #    interest"). Skip "pop of color" (a color) and the second
    #    "inviting" (rule 7).
    doc = "llama-3.2-11b|restaurant|92|response"
    t = cut(doc)
    ex.append(build(doc, t, [
        d(L("charming spot"), "charming spot", "atmosphere", vibe="yes"),
        d(L("warm"), "warm", "atmosphere", vibe="yes", slot="atmosphere"),
        d(L("inviting atmosphere"), "inviting", "atmosphere", vibe="yes", slot="atmosphere"),
        d(L("cozy"), "cozy", "atmosphere", vibe="yes", slot="setting"),
        d(L("intimate setting"), "intimate", "atmosphere", vibe="yes", slot="setting"),
        d(L("visual interest"), "visual interest", "evaluation"),
        d(L("sense of warmth"), "warmth", "atmosphere", vibe="yes", slot="sense"),
        d(L("hospitality"), "hospitality", "atmosphere", vibe="yes", slot="sense"),
        d(L("welcoming"), "welcoming", "atmosphere", vibe="yes"),
    ]))

else:
    # 1. gemma-4-12b trace, image 476, the start (analysis notes). Qualities
    #    of the place and of the things in it are place. "relatively flat"
    #    is the frame: current_photo. Skip "large" (the size of a tree),
    #    "deep green" (a color), and "Bright daylight", "dappled" light (the
    #    light).
    doc = "gemma-4-12b|street|476|reasoning"
    t = cut(doc, end_before="        *   The contrast")
    ex.append(build(doc, t, [
        d(L("historic-looking"), "historic-looking", "appearance"),
        d(L("ornate"), "ornate", "appearance"),
        d(L("lush"), "lush", "appearance"),
        d(L("flat"), "flat", "composition", CP),
        d(L("sleek"), "sleek", "appearance"),
        d(L("modern"), "modern", "appearance"),
        d(F("historic", "iron gate"), "historic", "appearance"),
        d(L("textured"), "textured", "appearance"),
        d(L("organic"), "organic", "appearance"),
    ]))

    # 2. qwen3.8-27b trace, image 86, a middle cut (the options). The plaza
    #    is "empty and clean" in the street (place); "unintentional" is the
    #    tilt of the frame (current_photo); "dramatic", "bold", "moody" are
    #    what the planned picture will feel (planned_photo, D126). Skip
    #    "leading line", "foreground anchor", "wide lens" (technique) and
    #    "classic" (praise of a technique).
    doc = "qwen3.8-27b|street|86|reasoning"
    t = cut(doc, "2. The hydrant as", "4. Use the window column")
    ex.append(build(doc, t, [
        d(L("empty"), "empty", "appearance"),
        d(L("clean"), "clean", "appearance"),
        d(L("dramatic"), "dramatic", "atmosphere", PP, vibe="yes"),
        d(L("tall"), "tall", "size"),
        d(L("unintentional"), "unintentional", "composition", CP),
        d(L("bold"), "bold", "appearance", PP),
        d(L("moody"), "moody", "atmosphere", PP, vibe="yes"),
        d(L("abstract"), "abstract", "appearance", PP),
    ]))

    # 3. Negative: the same qwen3.8-27b trace (image 86), the plan of the
    #    answer. Words about the answer ("practical", "short bold labels") are
    #    not descriptors: `none`.
    doc = "qwen3.8-27b|street|86|reasoning"
    t = cut(doc, "Keep it practical", "Let me make sure")
    ex.append(build(doc, t, [("none", "Keep it practical and not")]))

    # 4. gemma-4-12b answer, image 943, option 3 of 3. The label of the
    #    option names the look of the planned picture (planned_photo). "a bit
    #    flat" is the frame (current_photo). "imposing and grand" is what the
    #    low angle makes (D126). Skip "Crouch down low", "upward", and
    #    "leading lines" (camera instructions and technique).
    doc = "gemma-4-12b|street|943|response"
    t = cut(doc, '### 3. The "Low-Angle', "### Quick")
    ex.append(build(doc, t, [
        d(L("cinematic"), "cinematic", "atmosphere", PP, vibe="yes", slot="look"),
        d(L("dynamic"), "dynamic", "atmosphere", PP, vibe="yes", slot="look"),
        d(L("flat"), "flat", "composition", CP),
        d(L("interesting"), "interesting", "evaluation", PP),
        d(L("imposing"), "imposing", "appearance", PP),
        d(L("grand"), "grand", "appearance", PP),
    ]))

    # 5. qwen3.5-9b answer, image 74, the first 2 options. All 3 targets in
    #    contrast: "messy" power lines stay messy in the street (place, D127);
    #    "cluttered" and "distracting in the original photo" are the frame
    #    (current_photo); "intimate and secret" is the planned picture. Skip
    #    "large" (the size of a tree and of a chimney) and the technique
    #    labels ("Natural Framing", "Leading Line").
    doc = "qwen3.5-9b|street|74|response"
    t = cut(doc, end_before="You can crop out")
    ex.append(build(doc, t, [
        d(L("striking"), "striking", "evaluation", PP),
        d(L("nice"), "nice", "evaluation", CP),
        d(L("cluttered"), "cluttered", "composition", CP),
        d(L("leafy"), "leafy", "appearance"),
        d(L("messy"), "messy", "appearance"),
        d(L("distracting"), "distracting", "composition", CP),
        d(L("intimate"), "intimate", "atmosphere", PP, vibe="yes"),
        d(L("secret"), "secret", "atmosphere", PP, vibe="yes"),
        d(L("sense of depth"), "depth", "composition", PP, slot="sense"),
    ]))

    # 6. llama-3.2-11b answer, image 2, whole. A short Llama answer of
    #    generic praise of the planned picture: extract it, tagged
    #    `evaluation`. Skip "texture" and "architectural details" (plain).
    doc = "llama-3.2-11b|street|2|response"
    t = cut(doc)
    ex.append(build(doc, t, [
        d(L("great spot"), "great spot", "evaluation", PP),
        d(L("visually appealing composition"), "visually appealing", "evaluation", PP, slot="composition"),
        d(L("sense of depth"), "depth", "composition", PP, slot="sense"),
        d(L("sense of mystery"), "mystery", "atmosphere", PP, vibe="yes", slot="sense"),
        d(L("intrigue"), "intrigue", "atmosphere", PP, vibe="yes", slot="sense"),
        d(L("unique perspective"), "unique", "evaluation", PP, slot="perspective"),
        d(L("striking"), "striking", "evaluation", PP),
        d(L("memorable"), "memorable", "evaluation", PP),
    ]))

if ARGS.ids:
    for e in ex:
        print(e["doc_id"], len(e["text"]), len(e["extractions"]))
    sys.exit(0)


class Lit(str):
    pass


yaml.add_representer(Lit, lambda dmp, s: dmp.represent_scalar("tag:yaml.org,2002:str", s, style="|"))
for e in ex:
    del e["doc_id"]
    e["text"] = Lit(e["text"] + "\n")
sys.stdout.write(yaml.dump({"examples": ex}, allow_unicode=True, sort_keys=False, width=1000))
