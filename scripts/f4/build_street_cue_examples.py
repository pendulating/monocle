#!/usr/bin/env python
"""F4: print the examples block of `urban_cues_street_v<N>.yaml` from real street texts.

Each span is a slice of its text, found in order after the span before it,
thus it is an exact copy, and 2 spans never overlap. A span is a literal
string (`L`) or the slice from a start anchor to an end anchor, with the end
anchor included (`R`). Use `R` for a span with a character that is hard to
type, such as an en dash.

The texts come from the descriptor input (`f4_street_desc.parquet`, U+2011
is "-"), the same input as the street cue run. No example text is in the
smoke input (`f4_streetsmoke_desc.parquet`); the script stops if one is.

`--check` prints, for each example text, where the descriptor spans of the
`descriptors-v5` run fall: inside which cue class. A descriptor with target
`place` should fall in `visual_evidence` or `inference`; a descriptor with
target `photo` should fall outside them (D125).

`--version` sets the examples:

| Version | Examples |
|---------|----------|
| v1 | 1 2 3 4 5 (D125) |
| v2 | 1 2 3 4 6 5: example 6, a gemma-4-12b trace with notes that describe the place, a photo critique that describes the place (D127), and descriptions of the place inside advice (D128) |
| v3 | 7 new short examples, 1 for each input shape, with nested spans and a `none` example (D130) |

Version 3 (the x2 round, D130) applies the advisor review of 2026-10-26:
short cut texts, whole-sentence `photo_advice` spans with the descriptions
of the place NESTED inside them (`N(...)` below), more `inference` spans,
and 1 `none` example. A nested span is found inside its parent span, after
the nested span before it. The example texts must not be in
`x2_smoke_doc_ids.json`.

Change a span here, not by hand in the YAML, then paste the output under
`prompt_description` and raise the version.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/build_street_cue_examples.py > examples.yaml
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/build_street_cue_examples.py --check
"""
import argparse
import json
import sys

import pandas as pd
import yaml

ROOT = "/share/ju/matt/mllmsci"
INPUT = f"{ROOT}/data/f4_traces/extract_input/f4_street_desc.parquet"
SMOKE = f"{ROOT}/data/f4_traces/extract_input/f4_streetsmoke_desc.parquet"
DESC = f"{ROOT}/data/f4_traces/descriptors_street/f4_street_desc_final.parquet"
TEXTS = pd.read_parquet(INPUT).set_index("doc_id").text

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true", help="print the descriptor check, not the YAML")
ap.add_argument("--version", choices=["v1", "v2", "v3"], default="v3")
ARGS = ap.parse_args()
CHECK, VERSION = ARGS.check, ARGS.version
ORDER = {"v1": [0, 1, 2, 3, 4], "v2": [0, 1, 2, 3, 5, 4]}

VE, INF, PR, IA, UNC, PA = ("visual_evidence", "inference", "person_reference", "image_artifact",
                            "uncertainty", "photo_advice")
NONE = "none"
KEY = {VE: "category", INF: "kind", PR: "used_in_judgment", IA: "kind", UNC: "reason", PA: "kind"}
X2_SMOKE = f"{ROOT}/data/f4_traces/extract_input/x2_smoke_doc_ids.json"


def L(s):
    return ("L", s, None)


def R(start, end):
    return ("R", start, end)


def build(doc_id, specs, cut_before=None):
    """specs: (span, class, value[, valence]) tuples, in text order.

    `value` fills the 1 attribute of the class (KEY); `valence` is for
    `visual_evidence` only and is "neutral" when a tuple leaves it out.
    """
    raw = TEXTS[doc_id]
    # The offsets of the descriptor run count from the raw text, which can
    # open with newlines that the example drops.
    lead = len(raw) - len(raw.lstrip("\n"))
    text = raw.strip("\n")
    if cut_before:
        text = text[: text.index(cut_before)].rstrip()
    out, cursor = [], 0
    for spec in specs:
        span, cls, value = spec[:3]
        kind, a, b = span
        i = text.index(a, cursor)
        j = i + len(a) if kind == "L" else text.index(b, i) + len(b)
        attrs = {KEY[cls]: value}
        if cls == VE:
            attrs = {"valence": spec[3] if len(spec) > 3 else "neutral", "category": value}
        out.append({"extraction_class": cls, "extraction_text": text[i:j], "attributes": attrs,
                    "_span": (lead + i, lead + j)})
        cursor = j
    return {"doc_id": doc_id, "text": text, "extractions": out}


ex = []

# 1. qwen3.8-27b reasoning, image 0, cut before the plan of the answer. The
#    first sentence echoes the request: nothing from it. Things it sees,
#    inferences of the place (NYC, the 7 line) with hedges, and a critique of
#    the photograph it got (image_artifact). "the gritty urban texture" is new
#    (a quality of the scene), thus it is visual_evidence; the rest of that
#    list repeats earlier things.
ex.append(build("qwen3.8-27b|street|0|reasoning", [
    (L("An elevated train viaduct"), VE, "other"),
    (L("looks like NYC"), INF, "location"),
    (L('the "Resorts World Casino & Hotel" sign suggests Atlantic City or a NYC sign for it'), INF, "location"),
    (L("the elevated structure with the blue steel girders and the pedestrian walkway on top looks like an NYC elevated line"), INF, "location"),
    (L("possibly the 7 line or similar"), UNC, "other"),
    (L("the sign is an ad for Resorts World Casino"), VE, "signage"),
    (L("a clear blue sky taking up the top half"), IA, "weather"),
    (L("a sidewalk corner with a traffic signal pole (yellow housing, pedestrian button)"), VE, "traffic"),
    (L('a "ONE WAY" sign, a "DO NOT ENTER" sign'), VE, "signage"),
    (L("an orange-and-white construction barrel, orange traffic cones"), VE, "construction"),
    (L("a chain-link fence with graffiti on a stone retaining wall"), VE, "upkeep", "bad"),
    (L("some weeds growing through the sidewalk cracks"), VE, "upkeep", "bad"),
    (L("The road runs diagonally"), VE, "traffic"),
    (L("shadows from the viaduct structure creating a striped pattern on the asphalt"), VE, "light"),
    (L("The current framing is a bit flat"), IA, "other"),
    (L("the wide angle captures a lot of empty sky"), IA, "angle"),
    (L("the scene feels scattered"), IA, "other"),
    (L("the gritty urban texture (graffiti, weeds, barrels)"), VE, "upkeep"),
], cut_before="\n\nSo a \"good angle\" recommendation."))

# 2. gemma-4-31b reasoning, image 213. The critique of the photograph it got
#    is image_artifact. The brainstorm and the options are photo_advice; what
#    the planned picture will feel ("secluded and mysterious", "more
#    imposing") is photo_advice `effect`, never inference. Skip: the
#    headings and option labels ("Artistic & Moody"), the goals, the
#    questions the model asks itself, an option that repeats an earlier
#    instruction, and step 5 (notes to itself about its plan). "the larger
#    tree on the left" and "the people sitting by the fence" are new inside
#    advice, thus they get their own span (rules 7 and 8).
ex.append(build("gemma-4-31b|street|213|reasoning", [
    (L("A residential-style building"), VE, "architecture"),
    (L("possibly a house or a small institution"), UNC, "other"),
    (L("partially hidden by trees"), IA, "occlusion"),
    (L("parked bicycles/scooters in the foreground"), VE, "traffic"),
    (L("The line of blue scooters/bikes creates a distracting visual barrier"), IA, "occlusion"),
    (L("The building is largely hidden by a medium-sized tree right in the center"), IA, "occlusion"),
    (L("The shot is taken straight-on from across the street"), IA, "angle"),
    (L('making it feel static and a bit "accidental" (like a street-view capture)'), IA, "other"),
    (L("Overcast sky"), IA, "weather"),
    (L("muted colors"), IA, "other"),
    (L("the arched porch/alcove on the right"), VE, "architecture"),
    (L("the lush greenery"), VE, "greenery", "good"),
    (L("The people and the bikes are incidental elements"), PR, "no"),
    (L("Move closer (cross the street if safe)"), PA, "position"),
    (L("Change the angle (shoot from a diagonal)"), PA, "position"),
    (L("Change the height (shoot from higher up, or lower down)"), PA, "position"),
    (L("Move to the right to capture the arched entryway more prominently"), PA, "position"),
    (L("Move to the left to see the facade"), PA, "position"),
    (L("Find a gap in the trees"), PA, "position"),
    (L('Use the trees to "frame" the building rather than block it'), PA, "framing"),
    (L("Use the sidewalk or the fence to lead the viewer's eye into the image"), PA, "framing"),
    (L("Create distinct foreground, middle ground, and background layers"), PA, "framing"),
    (L("A three-quarter view focusing on that arched entryway"), PA, "framing"),
    (L("Removes the central tree, removes the bikes, emphasizes the most interesting architectural feature"), PA, "effect"),
    (L("Stay across the street but move slightly left"), PA, "position"),
    (L("Shoot *through* the branches of"), PA, "framing"),
    (L("the larger tree on the left"), VE, "greenery"),
    (L('Creates a "peek-a-boo" effect, making the building feel secluded and mysterious'), PA, "effect"),
    (L("Get low to the ground (crouch) near the fence"), PA, "position"),
    (L("Shoot slightly upwards toward the building and the treetops"), PA, "position"),
    (L("Makes the architecture feel more imposing"), PA, "effect"),
    (L("Focus on"), PA, "framing"),
    (L("the people sitting by the fence"), PR, "no"),
    (L("A closer, medium shot of the people, using the building as a blurred background (shallow depth of field)"), PA, "framing"),
    (L("Turns it from a building photo into a street photography piece"), PA, "effect"),
]))

# 3. qwen3.8-27b answer, image 278. Advice and scene in the same sentences.
#    A thing of the scene that the advice names for the first time with a
#    color, a state, or a place in the frame ("the maroon SUV", "the white
#    car at the right edge") is visual_evidence, and the photo_advice span
#    stops before it (rule 7). "the row of parked cars" and "the tree canopy"
#    have no such description and stay inside the advice. Skip the option
#    labels ("The quiet residential shot", "The candid", "The color pop").
ex.append(build("qwen3.8-27b|street|278|response", [
    (L("drop down to street level and line yourself up parallel to the curb, shooting *down* the street"), PA, "position"),
    (L("you're a bit too far back and perched above the scene"), IA, "angle"),
    (L("that rusty green rail in the foreground"), VE, "upkeep", "bad"),
    (L("the photo reads as a flat snapshot"), IA, "other"),
    (L("step to the sidewalk edge and point your camera along the row of parked cars"), PA, "position"),
    (L("the maroon SUV"), VE, "traffic"),
    (L("becomes a strong foreground anchor"), PA, "effect"),
    (L("the little black gate"), VE, "other"),
    (L("create a leading line that pulls the eye into the distance"), PA, "effect"),
    (L("the wall of trees"), VE, "greenery"),
    (L("compresses into a nice atmospheric backdrop"), PA, "effect"),
    (L("The walking pedestrian"), PR, "no"),
    (L("lands somewhere near the rule-of-thirds intersection instead of being lost in the middle"), PA, "effect"),
    (L("stand close to"), PA, "position"),
    (L("the gray house with the red roof on the left"), VE, "architecture"),
    (L("frame it tight against the trees on the right, cropping out nearly all of that blank overcast sky"), PA, "framing"),
    (L("The soft, shadowless light"), IA, "weather"),
    (L("is actually perfect for bringing out the leaf texture"), PA, "effect"),
    (L("wait for the person to be mid-stride with the tree canopy as a clean background"), PA, "timing"),
    (L("shoot from across the street at *their* eye level, not from above"), PA, "position"),
    (R("A longer lens", "isolate them nicely"), PA, "equipment"),
    (L("get low beside the maroon SUV so its body fills the lower third and the green wall of trees sits directly behind it"), PA, "position"),
    (L("red-on-green is a satisfying contrast under this flat light"), PA, "effect"),
    (L("keep the sidewalk line on the lower third"), PA, "framing"),
    (L("kill most of the sky"), PA, "framing"),
    (L("overcast skies are just gray dead space here"), IA, "weather"),
    (L("get the rusty rail out of frame"), PA, "framing"),
    (L("moving a couple of feet or shifting your height"), PA, "position"),
    (L("don't slice"), PA, "framing"),
    (L("the white car at the right edge"), VE, "traffic"),
    (L("either frame it in fully as a counterpoint to the SUV or leave it out"), PA, "framing"),
    (L("get lower, get closer, and shoot *along* the street instead of at it"), PA, "position"),
]))

# 4. qwen3.5-9b answer, image 48 (also example 8 of `descriptors-v5`). The
#    "Why it works" lines mix the scene ("a striking, jagged silhouette",
#    "the orderly city street") with the effect of the planned picture ("a
#    dramatic, almost industrial feel"). "the vehicle depot" is a claim past
#    the pixels: inference. Skip the option labels and short joins such as
#    "You can include" and "It highlights the contrast between".
ex.append(build("qwen3.5-9b|street|48|response", [
    (L("Step onto the sidewalk across the street from where the photo was taken"), PA, "position"),
    (L("Align your camera so that"), PA, "framing"),
    (L("the white stripes of the crosswalk"), VE, "traffic"),
    (L("lead your eye straight down the center of the image"), PA, "effect"),
    (L('the base of the billboard and the street sign ("Tilletsan Av")'), VE, "signage"),
    (L("The parallel lines of the crosswalk create a strong sense of depth and draw attention immediately to the intersection and the tall structure"), PA, "effect"),
    (L("Stand low to the ground, perhaps near"), PA, "position"),
    (L("the orange striped cone"), VE, "construction"),
    (L("the utility pole on the left corner"), VE, "other"),
    (L("Point your camera upward"), PA, "position"),
    (L("The billboard frame is a striking, jagged silhouette against the bright blue sky"), VE, "signage", "good"),
    (L("Shooting from below emphasizes its height and gives the image a dramatic, almost industrial feel"), PA, "effect"),
    (L("Move to the right side of the intersection, closer to"), PA, "position"),
    (L("the line of parked buses and trucks"), VE, "traffic"),
    (L("Zoom in slightly to focus on"), PA, "framing"),
    (L("the cluster of vehicles (the black SUV, the white car, and the trailers/buses)"), VE, "traffic"),
    (L("the overgrown weeds in the foreground"), VE, "upkeep", "bad"),
    (L("the orderly city street in the foreground"), VE, "traffic", "good"),
    (L("the somewhat chaotic, stored-away nature of the vehicle depot behind it"), INF, "other"),
    (L('If you can wait until "Golden Hour" (shortly after sunrise or before sunset)'), PA, "timing"),
    (L("the long shadows will stretch across the crosswalk, making the geometric pattern even more striking"), PA, "effect"),
]))

# 5. llama-3.2-90b answer, image 46 (also example 9 of `descriptors-v5`). A
#    short answer that is mostly advice. "the windows, doors, and ornate trim"
#    and "the building's unique features and architecture" describe the
#    place inside the advice: visual_evidence (rule 7). "warmth and depth"
#    is the effect of the planned picture: photo_advice.
ex.append(build("llama-3.2-90b|street|46|response", [
    (L("The best angle to take a picture from would be directly in front of the building, at a slight height to capture the entire facade"), PA, "position"),
    (L("This would allow for a clear view of the building's architecture and details"), PA, "effect"),
    (L("the windows, doors, and ornate trim"), VE, "architecture", "good"),
    (L("taking the photo from a slight distance would help to capture the building's scale and proportions"), PA, "position"),
    (L("taking the photo during the golden hour, when the sun is low in the sky, to add warmth and depth to the image"), PA, "timing"),
    (L("A wide-angle lens would be ideal for capturing the entire building"), PA, "equipment"),
    (L("a tripod would help to ensure a sharp image"), PA, "equipment"),
    (L("find a vantage point that showcases"), PA, "position"),
    (L("the building's unique features and architecture"), VE, "architecture", "good"),
]))

# 6. gemma-4-12b reasoning, image 154. Version 2 (D127, D128). The notes of
#    step 1 describe the place ("Subject:", "Context:", "Details to
#    notice:"): extract them, they are not notes about the plan. "It feels a
#    bit dull" is true only of the frame: image_artifact. 'The "messy"
#    foreground (fence, weeds, trash)' stays true in the street:
#    visual_evidence (D127). Inside advice, "the urban, lived-in feel" and
#    "a piece of urban decay" belong to the place, not to the planned
#    picture: inference. "the old fence" gives the fence a new quality:
#    visual_evidence. Skip: the goals of step 2, the style labels, and steps
#    5 and 6 (the plan of the answer).
ex.append(build("gemma-4-12b|street|154|reasoning", [
    (L("A three-story tan/brown brick apartment building"), VE, "architecture"),
    (L("Urban/residential setting"), INF, "other"),
    (L("some messy vegetation"), VE, "greenery", "bad"),
    (L('Very flat, straight-on "documentary" style'), IA, "angle"),
    (L("It lacks depth, dynamic angles, or clear focal points"), IA, "other"),
    (L("It feels a bit dull"), IA, "other"),
    (L("Overcast, flat light"), IA, "weather"),
    (L("No strong shadows to define texture"), IA, "weather"),
    (L("Vertical alignment of three windows (one with AC)"), VE, "architecture"),
    (L('The "messy" foreground (fence, weeds, trash)'), VE, "upkeep", "bad"),
    (L("The neighboring house on the left"), VE, "architecture"),
    (L("The trees framing the top left/right"), VE, "greenery"),
    (L("The utility lines across the top"), VE, "other"),
    (L("Get away from the messy foreground"), PA, "position"),
    (L("A low angle looking up to make it look taller and more imposing"), PA, "position"),
    (L("A corner shot to see the depth of the building"), PA, "framing"),
    (L("Lean into"), PA, "framing"),
    (L("the urban, lived-in feel"), INF, "other"),
    (L("A wide-angle shot from across the street to include the context of the neighborhood"), PA, "position"),
    (L("Close-ups on textures"), PA, "framing"),
    (L("the old fence"), VE, "upkeep"),
    (L("Use the surrounding elements to frame the building"), PA, "framing"),
    (L("Shoot through the trees or through the fence"), PA, "framing"),
    (L('Go for a "Dutch tilt" (slightly slanted) for a more uneasy or dynamic feeling'), PA, "framing"),
    (L("This adds height and removes the messy sidewalk/fence from the bottom of the frame"), PA, "effect"),
    (L("Most buildings look better from an angle than perfectly straight-on"), PA, "effect"),
    (L('It shows the "depth" of the structure'), PA, "effect"),
    (L("Use the trees on the right or left to create a natural frame"), PA, "framing"),
    (L("Move in closer"), PA, "position"),
    (L("Don't try to get the whole building"), PA, "framing"),
    (L("Focus on one window or"), PA, "framing"),
    (L("a piece of urban decay"), INF, "upkeep"),
]))

def N(spec):
    """Mark a spec as nested inside the last top-level span (version 3)."""
    return ("N", spec)


def build3(doc_id, specs, cut_after=None, cut_before=None):
    """Build 1 version 3 example. specs: (span, class, value[, valence]) or N(spec).

    `cut_after` starts the text at its anchor (a middle chunk); `cut_before`
    ends it before its anchor. The class `none` has no attribute.
    """
    raw = TEXTS[doc_id]
    lead = len(raw) - len(raw.lstrip("\n"))
    text = raw.strip("\n")
    off = text.index(cut_after) if cut_after else 0
    text = text[off:]
    if cut_before:
        text = text[: text.index(cut_before)].rstrip()
    out, cursor, parent, ncur = [], 0, None, 0
    for spec in specs:
        nested = spec[0] == "N"
        if nested:
            spec = spec[1]
        span, cls, value = spec[:3]
        kind, a, b = span
        if nested:
            i = text.index(a, ncur)
            j = i + len(a) if kind == "L" else text.index(b, i) + len(b)
            if j > parent[1]:
                raise ValueError(f"{doc_id}: nested span {text[i:j]!r} leaves its parent")
            ncur = j
        else:
            i = text.index(a, cursor)
            j = i + len(a) if kind == "L" else text.index(b, i) + len(b)
            cursor, parent, ncur = j, (i, j), i
        item = {"extraction_class": cls, "extraction_text": text[i:j]}
        if cls == VE:
            item["attributes"] = {"valence": spec[3] if len(spec) > 3 else "neutral", "category": value}
        elif cls != NONE:
            item["attributes"] = {KEY[cls]: value}
        item["_span"] = (lead + off + i, lead + off + j)
        out.append(item)
    return {"doc_id": doc_id, "text": text, "extractions": out,
            "_range": (lead + off, lead + off + len(text))}


def examples_v3():
    """The 7 examples of version 3, 1 for each input shape, in prompt order."""
    v3 = []
    # 1. gemma-4-12b trace, image 154, the start. The echo line gives nothing.
    #    The notes of step 1 ("Subject:", "Details to notice:") describe the
    #    place: extract them. "It feels a bit dull" is true only of the frame
    #    (image_artifact); the "messy" foreground stays true in the street
    #    (visual_evidence, D127).
    v3.append(build3("gemma-4-12b|street|154|reasoning", [
        (L("A three-story tan/brown brick apartment building"), VE, "architecture"),
        (L("Urban/residential setting"), INF, "neighborhood_character"),
        (L("some messy vegetation"), VE, "greenery", "bad"),
        (L('Very flat, straight-on "documentary" style'), IA, "angle"),
        (L("It lacks depth, dynamic angles, or clear focal points"), IA, "other"),
        (L("It feels a bit dull"), IA, "other"),
        (L("Overcast, flat light"), IA, "weather"),
        (L("No strong shadows to define texture"), IA, "weather"),
        (L("Vertical alignment of three windows (one with AC)"), VE, "architecture"),
        (L('The "messy" foreground (fence, weeds, trash)'), VE, "upkeep", "bad"),
        (L("The neighboring house on the left"), VE, "architecture"),
        (L("The trees framing the top left/right"), VE, "greenery"),
        (L("The utility lines across the top"), VE, "other"),
    ], cut_before="\n\n2.  **Identify Goals"))
    # 2. qwen3.8-27b trace, image 844, the start. Claims past the pixels
    #    ("residential", "a working-class neighborhood") are inference. The
    #    critique of the frame is image_artifact.
    v3.append(build3("qwen3.8-27b|street|844|reasoning", [
        (L("A large spreading tree dominates the center and upper portion"), VE, "greenery"),
        (L("brick houses"), VE, "architecture"),
        (L("residential"), INF, "neighborhood_character"),
        (L("looks like a working-class neighborhood"), INF, "class"),
        (L("a black SUV (Audi) parked in a driveway on the right"), VE, "traffic"),
        (L("three trash/recycling bins (dark, dark, blue) in front of the tree base"), VE, "other"),
        (L("the glossy black roof and rear window of another car, very close to the camera"), IA, "occlusion"),
        (L("Power lines cross the upper left"), VE, "other"),
        (L("There's dappled sunlight"), IA, "weather"),
        (L("Stairs and porches on the houses"), VE, "architecture"),
        (L("The current framing is a bit cluttered"), IA, "other"),
        (L("the foreground car roof cuts in awkwardly"), IA, "occlusion"),
        (L("the bins are mid-frame and a bit distracting"), IA, "other"),
        (L("the tree is the star but it's competing with the cars"), IA, "other"),
    ], cut_before="The question:"))
    # 3. The same trace, the next part: the model plans its answer. A goal
    #    ("capture the tree/neighborhood vibe") and the plan give nothing,
    #    thus the chunk gets 1 `none` span.
    v3.append(build3("qwen3.8-27b|street|844|reasoning", [
        (L('The question: "tell me'), NONE, None),
    ], cut_after="The question:", cut_before="\n\nAngle options:"))
    # 4. gemma-4-12b trace, image 154, a middle part (a brainstorm). Each
    #    idea is 1 photo_advice span. Inside it, a claim about the place
    #    ("the urban, lived-in feel") and a new quality of a thing ("the old
    #    fence") are nested spans. Headings and labels give nothing.
    v3.append(build3("gemma-4-12b|street|154|reasoning", [
        (L("Get away from the messy foreground."), PA, "position"),
        (L("A low angle looking up to make it look taller and more imposing."), PA, "position"),
        (L("A corner shot to see the depth of the building."), PA, "framing"),
        (L("Lean into the urban, lived-in feel."), PA, "framing"),
        N((L("the urban, lived-in feel"), INF, "neighborhood_character")),
        (L("A wide-angle shot from across the street to include the context of the neighborhood."), PA, "position"),
        (L("Close-ups on textures (the brick, the window AC, the old fence)."), PA, "framing"),
        N((L("the old fence"), VE, "upkeep")),
        (L("Use the surrounding elements to frame the building."), PA, "framing"),
        (L("Shoot through the trees or through the fence."), PA, "framing"),
        (L('Go for a "Dutch tilt" (slightly slanted) for a more uneasy or dynamic feeling.'), PA, "effect"),
    ], cut_after="3.  **Brainstorming", cut_before="\n\n4.  **Refining"))
    # 5. qwen3.5-9b answer, image 48, options 2 and 3. "Why it works" lines
    #    mix the place ("a striking, jagged silhouette") with the effect of
    #    the planned picture ("a dramatic, almost industrial feel", D126).
    v3.append(build3("qwen3.5-9b|street|48|response", [
        (L("Stand low to the ground, perhaps near the orange striped cone or the bottom of the utility pole on the left corner."), PA, "position"),
        N((L("the orange striped cone"), VE, "construction")),
        (L("Point your camera upward."), PA, "position"),
        (L("The billboard frame is a striking, jagged silhouette against the bright blue sky."), VE, "signage", "good"),
        (L("Shooting from below emphasizes its height and gives the image a dramatic, almost industrial feel."), PA, "effect"),
        (L("Move to the right side of the intersection, closer to the line of parked buses and trucks."), PA, "position"),
        (L("You can include the overgrown weeds in the foreground for contrast."), PA, "framing"),
        N((L("the overgrown weeds in the foreground"), VE, "upkeep", "bad")),
        (L("It highlights the contrast between the orderly city street in the foreground and the somewhat chaotic, stored-away nature of the vehicle depot behind it."), PA, "effect"),
        N((L("the orderly city street in the foreground"), VE, "traffic", "good")),
        N((L("the somewhat chaotic, stored-away nature of the vehicle depot behind it"), INF, "neighborhood_character")),
    ], cut_after='**2. The "Skeletal', cut_before="**Pro Tip:**"))
    # 6. qwen3.8-27b answer, image 278, the first paragraph. Whole-sentence
    #    advice with the described things of the place nested inside; the
    #    rusty rail stays true in the street (D127); the pedestrian is a
    #    person_reference inside advice.
    v3.append(build3("qwen3.8-27b|street|278|response", [
        (R("The single best move here is to", "rather than across it."), PA, "position"),
        (L("Right now you're a bit too far back and perched above the scene"), IA, "angle"),
        (L("that rusty green rail in the foreground"), VE, "upkeep", "bad"),
        (L("the photo reads as a flat snapshot"), IA, "other"),
        (R("If you step to the sidewalk edge", "a nice atmospheric backdrop."), PA, "position"),
        N((L("the maroon SUV"), VE, "traffic")),
        N((L("the little black gate"), VE, "other")),
        N((L("the wall of trees"), VE, "greenery")),
        (L("The walking pedestrian then lands somewhere near the rule-of-thirds intersection instead of being lost in the middle."), PA, "effect"),
        N((L("The walking pedestrian"), PR, "no")),
    ], cut_before="\n\nA few alternates"))
    # 7. llama-3.2-90b answer, image 884. A short answer that is all advice
    #    and praise. The praise of the planned picture ("visually appealing
    #    and engaging") is photo_advice effect; the described things of the
    #    place and the claim about the neighborhood are nested inside.
    v3.append(build3("llama-3.2-90b|street|884|response", [
        (L("I would recommend taking a photo from the sidewalk in front of the house."), PA, "position"),
        (L("This vantage point offers a clear view of the well-manicured front yard, complete with neatly trimmed bushes and trees, as well as the charming brick walkway leading up to the house."), PA, "effect"),
        N((L("the well-manicured front yard, complete with neatly trimmed bushes and trees"), VE, "upkeep", "good")),
        N((L("the charming brick walkway leading up to the house"), VE, "architecture", "good")),
        (L("The angle also allows for a glimpse of the neighboring houses on either side, adding context to the scene."), PA, "effect"),
        N((L("the neighboring houses on either side"), VE, "architecture")),
        (R("By positioning yourself on the sidewalk", "the lush greenery of the yard."), PA, "position"),
        N((L("the warm tones of the brick walkway"), VE, "color", "good")),
        N((L("the lush greenery of the yard"), VE, "greenery", "good")),
        (R("Additionally, this angle provides", "a sense of invitation."), PA, "effect"),
        (L("capturing the scene from this angle will result in a visually appealing and engaging photograph that showcases the beauty of the front yard and the charm of the surrounding neighborhood"), PA, "effect"),
        N((L("the beauty of the front yard"), VE, "greenery", "good")),
        N((L("the charm of the surrounding neighborhood"), INF, "neighborhood_character")),
    ]))
    return v3


ex = [ex[i] for i in ORDER[VERSION]] if VERSION != "v3" else examples_v3()

smoke = set(pd.read_parquet(SMOKE).doc_id)
if VERSION == "v3":
    smoke |= set(json.load(open(X2_SMOKE))["street"])
clash = [e["doc_id"] for e in ex if e["doc_id"] in smoke]
if clash:
    sys.exit(f"example texts in the smoke input: {clash}")

if CHECK:
    d = pd.read_parquet(DESC)
    d = d[d.doc_id.isin([e["doc_id"] for e in ex]) & d.is_quotable.fillna(False)]
    rows = []
    for e in ex:
        cues = [(x["_span"], x["extraction_class"]) for x in e["extractions"]]
        lo, hi = e.get("_range", (0, 10 ** 9))
        for r in d[d.doc_id == e["doc_id"]].itertuples():
            if not (lo <= r.char_start and r.char_end <= hi):
                continue
            tgt = json.loads(r.attributes_json).get("target")
            # The innermost span decides (version 3 nests spans).
            inside = sorted(((b - a), c) for (a, b), c in cues if a <= r.char_start and r.char_end <= b)
            cls = inside[0][1] if inside else "none"
            ok = (cls in (VE, INF)) == (tgt == "place")
            rows.append({"doc_id": e["doc_id"], "descriptor": r.extraction_text, "target": tgt,
                         "cue_class": cls, "agree": ok})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.to_string())
    print(out.groupby("target").agree.agg(["sum", "count"]))
    sys.exit(0)


class Lit(str):
    pass


yaml.add_representer(Lit, lambda dmp, s: dmp.represent_scalar("tag:yaml.org,2002:str", s, style="|"))
for e in ex:
    del e["doc_id"]
    e.pop("_range", None)
    e["text"] = Lit(e["text"] + "\n")
    for x in e["extractions"]:
        del x["_span"]
sys.stdout.write(yaml.dump({"examples": ex}, allow_unicode=True, sort_keys=False, width=1000))
