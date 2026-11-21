#!/usr/bin/env python
"""F4: print the examples block of `urban_cues_single_v2.yaml` from real F4 texts (D130).

Each span is a slice of its text. The script finds a span after the span
before it, thus it is an exact copy. A span can lie inside the span before it
(`nest=True`): the search then starts at the start of that span. A span is a
literal string (`L`) or the slice from a start anchor to an end anchor, with
the end anchor included (`R`).

The texts come from the x2 input (`f4_<case>_x2.parquet`, U+2011 is "-").
The script stops when an example text is in a smoke set
(`x2_smoke_doc_ids.json`). Each example is 1 input shape:

| # | Shape | Text |
|---|-------|------|
| 1 | restaurant trace, start (echo) | gemma-4-31b restaurant 937, cut |
| 2 | a chunk with nothing to extract: `none` | gemma-4-31b restaurant 14, the plan at the end |
| 3 | restaurant trace, a middle chunk (a deliberation) | qwen3.5-9b restaurant 669, a window |
| 4 | caption trace | qwen3.8-27b caption 163, cut |
| 5 | restaurant answer (markdown) | gemma-4-12b restaurant 331 |
| 6 | caption answer (a list of captions) | qwen3.5-9b caption 310 |
| 7 | short Llama restaurant answer | llama-3.2-11b restaurant 6 |

Change a span here, not by hand in the YAML, then paste the output under
`prompt_description` and raise the version.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/build_single_v2_examples.py > examples.yaml
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/build_single_v2_examples.py --ids
"""
import argparse
import json
import sys

import pandas as pd
import yaml

ROOT = "/share/ju/matt/mllmsci"
INPUT = ROOT + "/data/f4_traces/extract_input/f4_{}_x2.parquet"
SMOKE = ROOT + "/data/f4_traces/extract_input/x2_smoke_doc_ids.json"
TEXTS = pd.concat([pd.read_parquet(INPUT.format(c), columns=["doc_id", "text"])
                   for c in ("restaurant", "caption")]).set_index("doc_id").text

ap = argparse.ArgumentParser()
ap.add_argument("--ids", action="store_true", help="print the example doc_ids, not the YAML")
IDS = ap.parse_args().ids

VE, INF, PR, IA, UNC, NONE = ("visual_evidence", "inference", "person_reference", "image_artifact",
                              "uncertainty", "none")
KEY = {INF: "kind", PR: "used_in_judgment", IA: "kind", UNC: "reason"}


def L(s, nest=False):
    return ("L", s, None, nest)


def R(start, end, nest=False):
    return ("R", start, end, nest)


def window(doc_id, start=None, cut_before=None):
    """Give the text of `doc_id` from the anchor `start` up to the anchor `cut_before`."""
    text = TEXTS[doc_id]
    i = text.index(start) if start else 0
    j = text.index(cut_before, i) if cut_before else len(text)
    return text[i:j].strip("\n").rstrip()


def build(doc_id, text, specs):
    """specs: (span, class[, value[, valence]]) tuples, in text order.

    `value` fills the 1 attribute of the class; `visual_evidence` takes
    `category` and `valence` ("neutral" when a tuple leaves it out); `none`
    takes no attribute.
    """
    out, cursor, last_start = [], 0, 0
    for spec in specs:
        (kind, a, b, nest), cls = spec[0], spec[1]
        i = text.index(a, last_start if nest else cursor)
        j = i + len(a) if kind == "L" else text.index(b, i) + len(b)
        if cls == NONE:
            attrs = None
        elif cls == VE:
            attrs = {"valence": spec[3] if len(spec) > 3 else "neutral", "category": spec[2]}
        else:
            attrs = {KEY[cls]: spec[2]}
        item = {"extraction_class": cls, "extraction_text": text[i:j]}
        if attrs:
            item["attributes"] = attrs
        out.append(item)
        last_start, cursor = i, j
    return {"doc_id": doc_id, "text": text, "extractions": out}


ex = []

# 1. Restaurant trace, start. Nothing from the echo ("Analyze the Request")
#    or from the search plan of step 3. "partially visible" is an artifact
#    inside the evidence span: a nested span. The model's feel of the place
#    and the kind of area are inferences.
d = "gemma-4-31b|restaurant|937|reasoning"
ex.append(build(d, window(d, cut_before="\n    *   *Cross-referencing:*"), [
    (L('The prominent sign says "Morris Park Inn"'), VE, "signage"),
    (L('There\'s a number "1024" visible on the sign'), VE, "signage"),
    (L('A blue sign to the left says "Alis..."'), VE, "signage"),
    (L('street signs are partially visible: "Morris Park" and "Padding..."'), VE, "signage"),
    (L("partially visible", nest=True), IA, "occlusion"),
    (L("It looks like a casual neighborhood spot/pub/inn"), INF, "atmosphere"),
    (L("Red awning, outdoor seating (tables and chairs)"), VE, "commerce"),
    (L("Urban area"), INF, "neighborhood_character"),
    (L("brick building"), VE, "architecture"),
    (L("overhead power lines"), VE, "other"),
    (L('"Morris Park" is a known neighborhood in the Bronx, NYC'), INF, "location"),
]))

# 2. A chunk that holds only the plan of the answer: `none`, with the first
#    words of the chunk as its span. Nothing else.
d = "gemma-4-31b|restaurant|14|reasoning"
ex.append(build(d, window(d, start="6.  **Refine the Language", cut_before="\n\n**(Self-Correction"), [
    (L("6.  **Refine the Language"), NONE),
]))

# 3. Restaurant trace, a middle chunk: it starts inside a step. The model
#    weighs the name and the place. Each claim is 1 inference span, not the
#    whole paragraph; each doubt is 1 uncertainty span.
d = "qwen3.5-9b|restaurant|669|reasoning"
ex.append(build(d, window(d, start='    *   Wait, let\'s look at the "LUNAR"', cut_before="\n\n4.  **Drafting"), [
    (L('The sign says "LUNAR PANDA"'), VE, "signage"),
    (L('The "LUNAR" is small and above/to the left'), VE, "signage"),
    (L('the "Lunar" part is a bit faint'), UNC, "illegible"),
    (L("it's a known chain"), INF, "brand_knowledge"),
    (L("there is text that is hard to read"), UNC, "illegible"),
    (L('It might just be "PANDA"'), UNC, "identity"),
    (L('"Lunar Panda" is a real chain'), INF, "brand_knowledge"),
    (L('"PANDA CHINESE RESTAURANT"'), VE, "signage"),
    (L("Yellow facade"), VE, "color"),
    (L('Next to an "Olivia Market" (red sign)'), VE, "commerce"),
    (L("Olivia Markets are famously found in the Rockaway Peninsula and parts of Queens in NYC"), INF, "location"),
    (L('The address "73-85" strongly points to a specific block in Queens'), INF, "location"),
    (L("likely Howard Beach or Rockaway Park area"), UNC, "knowledge"),
    (L("The sign looks well-maintained but older style"), INF, "upkeep"),
]))

# 4. Caption trace. Nothing from the echo of the first 2 sentences. People:
#    person_reference, with the blur nested inside. The kind of store and the
#    city are inferences; "No, looks like" is a doubt about what a thing is.
d = "qwen3.8-27b|caption|163|reasoning"
ex.append(build(d, window(d, cut_before="\n\nThe user said"), [
    (L("a clear, bright day"), IA, "weather"),
    (L('a black storefront labeled "SOLELUNA,"'), VE, "signage"),
    (L('a "SUNNYSIDE PIZZA" banner with red lettering'), VE, "signage"),
    (L('a neon "PIZZA" sign'), VE, "signage"),
    (L("looks like a bodega/convenience store"), INF, "business_type"),
    (L('a brick building with old painted signage "FINGER SEED" (faded)'), VE, "signage"),
    (L("Two pedestrians walking on the sidewalk (faces blurred)"), PR, "no"),
    (L("faces blurred", nest=True), IA, "blur"),
    (L("No, looks like a red mailbox/pillar"), UNC, "identity"),
    (L("A green bike lane painted on the street in the foreground with a bicycle symbol"), VE, "traffic"),
    (L("urban, sunny, corner of a neighborhood"), INF, "neighborhood_character"),
    (L("looks like NYC"), INF, "location"),
    (L("possibly Long Island City / Queens / Brooklyn"), INF, "location"),
    (L("late afternoon, long shadows"), IA, "time_of_day"),
]))

# 5. Restaurant answer in markdown. Facts from outside the image (the
#    address, the building, the interior) are inferences; "upscale" is a
#    claim of wealth.
d = "gemma-4-12b|restaurant|331|response"
ex.append(build(d, window(d), [
    (L("It is located at **225 East 52nd Street** in Manhattan, New York City"), INF, "location"),
    (L("it occupies the ground floor of the historic **American Woolen Building**"), INF, "institution_knowledge"),
    (L("It is a high-end **French restaurant**"), INF, "business_type"),
    (L("its elegant and upscale atmosphere"), INF, "wealth"),
    (L("The large windows, high ceilings, and the iconic red awning with gold lettering"), VE, "architecture", "good"),
    (L("suggest a refined dining experience"), INF, "quality"),
    (L("The outdoor seating area with wicker chairs"), VE, "commerce"),
    (L('a popular spot for "people watching" and alfresco dining in the city'), INF, "popularity"),
    (L("the interior is known for being very spacious and ornate"), INF, "brand_knowledge"),
    (L("a popular choice for special occasions and upscale dinners"), INF, "popularity"),
    (L("partially visible in your photo"), IA, "occlusion"),
    (L("another well-known high-end restaurant in the same building"), INF, "brand_knowledge"),
]))

# 6. Caption answer: a list of captions. Extract what a caption says about
#    this place: a name on a sign, a thing in view, the kind of business or
#    area, the light. Skip the option labels and the hashtags.
d = "qwen3.5-9b|caption|310|response"
ex.append(build(d, window(d), [
    (L("Main street living at its finest"), INF, "neighborhood_character"),
    (L("these classic brick storefronts"), VE, "architecture", "good"),
    (L("the eclectic mix of local businesses"), INF, "neighborhood_character"),
    (L("From legal offices to bridal salons"), VE, "commerce"),
    (L('That "Little Guyana Bake Shop" sign'), VE, "signage"),
    (L("ordering up those empanadas"), INF, "business_type"),
    (L("Golden hour hitting the avenue"), IA, "time_of_day"),
    (L("There is a real charm to these older brick buildings and the colorful awnings"), VE, "architecture", "good"),
    (L("Just another sunny day on the block"), IA, "weather"),
]))

# 7. Short Llama restaurant answer: almost every sentence is a claim from
#    outside the image.
d = "llama-3.2-11b|restaurant|6|response"
ex.append(build(d, window(d), [
    (L("a Spanish tapas restaurant"), INF, "business_type"),
    (L("located in the Greenpoint neighborhood of Brooklyn"), INF, "location"),
    (L("It is situated on Manhattan Avenue"), INF, "location"),
    (L("a popular street lined with shops and restaurants"), INF, "neighborhood_character"),
    (L("The restaurant has a cozy and intimate atmosphere"), INF, "atmosphere"),
    (L("a warm and inviting interior that reflects the traditional Spanish cuisine it serves"), INF, "brand_knowledge"),
    (L("a yellow awning"), VE, "commerce"),
    (L("a small patio area where customers can dine al fresco"), VE, "commerce"),
    (L("a great spot for those looking to experience authentic Spanish flavors"), INF, "quality"),
]))

smoke = set(sum(json.load(open(SMOKE)).values(), []))
clash = [e["doc_id"] for e in ex if e["doc_id"] in smoke]
if clash:
    sys.exit(f"example texts in a smoke set: {clash}")

if IDS:
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
