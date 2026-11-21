#!/usr/bin/env python
"""F4: print the examples block of `aesthetic_descriptors_v1.yaml` from real F4 texts.

Each span is a slice of its text, found in order after the span before it,
thus it is an exact copy and the aligner can match it. A span is either a
literal string (`L`) or the slice from a start anchor to an end anchor, with
the end anchor included (`R`). Use `R` when the text holds a character that is
hard to type, such as the non-breaking hyphen (U+2011) of the qwen3.8-27b texts.

Change a span here, not by hand in the YAML, then paste the output under
`prompt_description` and raise the version.

`--version` sets the example order and the text form:

| Version | Order | Text |
|---------|-------|------|
| v1 | 1 2 3 4 5 6 7 (the llama-3.2-90b answer last) | as written |
| v2 | 1 2 7 4 5 6 3 (the qwen answer with few descriptors last) | U+2011 becomes "-" |
| v3 | as v2 | as v2; a descriptor inside a phrase about a building, a street or a neighborhood is only the descriptor (D113) |
| v3r | 1 3 6 5 4 7 2 (the positive examples of v3 in reverse order) | as v3; the retry pass (D121) |
| v4 | 1 2 7 8 4 5 6 3 (example 8, a street angle answer, after the llama-3.2-90b answer) | as v3; every span has `target` (place, photo) (D122) |
| v5 | 1 2 7 8 9 4 5 6 3 (example 9, a short Llama angle answer, after example 8) | as v4 (D123) |
| v5r | 1 3 6 5 4 9 8 7 2 (the positive examples of v5 in reverse order) | as v5; the retry pass (D124) |

Version 2 reads the same text form as its input
(`scripts/f4/build_descriptor_input.py`).

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/build_descriptor_examples.py --version v2 > examples.yaml
"""
import argparse
import sys

import pandas as pd
import yaml

INPUT = "/share/ju/matt/mllmsci/data/f4_traces/extract_input/f4_{}.parquet"
TEXTS = pd.concat([pd.read_parquet(INPUT.format(c)) for c in ("restaurant", "caption", "street")]).set_index("doc_id").text


ap = argparse.ArgumentParser()
ap.add_argument("--version", choices=["v1", "v2", "v3", "v3r", "v4", "v5", "v5r"], default="v5")
VERSION = ap.parse_args().version
ORDER = {"v1": [0, 1, 2, 3, 4, 5, 6], "v2": [0, 1, 6, 3, 4, 5, 2], "v3": [0, 1, 6, 3, 4, 5, 2],
         "v3r": [0, 2, 5, 4, 3, 6, 1],
         "v4": [0, 1, 6, 7, 3, 4, 5, 2],
         "v5": [0, 1, 6, 7, 8, 3, 4, 5, 2],
         "v5r": [0, 2, 5, 4, 3, 8, 7, 6, 1]}
# Version 3 and later: only the descriptor of a phrase about a physical place.
ONLY_DESCRIPTOR = VERSION not in ("v1", "v2")
# Version 4 and later: the attribute `target` (place, photo), D122.
TARGET = VERSION in ("v4", "v5", "v5r")
# Version 5 and later: a short Llama angle answer as example 9 (D123).
LLAMA_STREET = VERSION in ("v5", "v5r")


def text_of(doc_id: str) -> str:
    t = TEXTS[doc_id].strip("\n")
    # Version 2: the non-breaking hyphen becomes "-" (the input does the same).
    return t.replace("\u2011", "-") if VERSION != "v1" else t


def L(s):
    return ("L", s, None)


def R(start, end):
    return ("R", start, end)


def build(text, specs):
    """specs: (span, head, vibe, slot, aspect, negated[, target]) tuples, in text order.

    `target` is "place" when a tuple leaves it out. It reaches the output only
    in version 4.
    """
    out, cursor = [], 0
    for spec in specs:
        span, head, vibe, slot, aspect, negated = spec[:6]
        target = spec[6] if len(spec) > 6 else "place"
        kind, a, b = span
        i = text.index(a, cursor)
        j = i + len(a) if kind == "L" else text.index(b, i) + len(b)
        s = text[i:j]
        cursor = j
        attrs = {"head": head, "vibe": vibe, "slot": slot, "aspect": aspect, "negated": negated}
        if TARGET:
            attrs["target"] = target
        out.append({"extraction_class": "descriptor", "extraction_text": s, "attributes": attrs})
    return {"text": text, "extractions": out}


ex = []

# 1. Negative: the echo of the request and plain physical facts (gemma-4-12b,
#    restaurant trace, image 4; the first lines only).
t0 = text_of("gemma-4-12b|restaurant|4|reasoning")
t0 = t0[: t0.index(" Looks like a cafe")]
ex.append({"text": t0, "extractions": []})

# 2. gemma-4-31b restaurant trace, image 651. A "Vibe:" and an "Atmosphere:"
#    heading frame the words under them. Business types from world
#    knowledge. Skip: the place names, "Blue storefront", the plan of the
#    answer ("Describe the vibe/setting"), and "best" or "worst" (a note of
#    the model to itself). The last step repeats earlier words; only "local
#    vibe" is new.
t1 = text_of("gemma-4-31b|restaurant|651|reasoning")
ex.append(build(t1, [
    (L("classic New York City-style"), "classic new york city-style", "no", "none", "appearance", "no"),
    (L("neighborhood spot"), "neighborhood spot", "yes", "vibe", "social_register", "no"),
    (L("neighborhood bar and restaurant"), "neighborhood bar and restaurant", "no", "none", "business_type", "no"),
    (L("gastropub"), "gastropub", "no", "none", "business_type", "no"),
    (L("Modern American"), "modern american", "no", "none", "business_type", "no"),
    (L("creative cocktails"), "creative cocktail", "no", "none", "business_type", "no"),
    (L("Cozy"), "cozy", "yes", "atmosphere", "atmosphere", "no"),
    (L("neighborhood-centric"), "neighborhood-centric", "yes", "atmosphere", "social_register", "no"),
    (L("bar feel"), "bar", "yes", "feel", "atmosphere", "no"),
    (L("local vibe"), "local", "yes", "vibe", "social_register", "no"),
]))

# 3. qwen3.8-27b restaurant answer, image 577. Mostly plain facts: a sign, a
#    phone number, brick columns, a tree, parked cars. Only 2 descriptors.
t2 = text_of("qwen3.8-27b|restaurant|577|response")
ex.append(build(t2, [
    (L("barbershop"), "barbershop", "no", "none", "business_type", "no"),
    (R("strip", "style"), "strip-mall-style", "no", "none", "appearance", "no") if ONLY_DESCRIPTOR else
    (R("strip", "commercial block"), "strip-mall-style commercial block", "no", "none", "appearance", "no"),
]))

# 4. gemma-4-12b caption answer, image 3. Mood headings, figurative captions,
#    hashtags. Skip: "Short & Sweet", "Playful & Punny" and "The Explorer"
#    (they name the form of a caption, not the place), the "vibe" that the
#    user wants, puns, "favorite corner" (the feeling of the user), and
#    #streetphotography (the photograph). #neighborhoodgem repeats
#    "Neighborhood gems".
t3 = text_of("gemma-4-12b|caption|3|response")
ex.append(build(t3, [
    (L("Cozy"), "cozy", "yes", "none", "atmosphere", "no"),
    (L("Aesthetic"), "aesthetic", "yes", "none", "appearance", "no"),
    (L('"Main Character" vibe'), "main character", "yes", "vibe", "figurative", "no"),
    (L("Slow mornings"), "slow morning", "yes", "none", "atmosphere", "no"),
    (L("hidden gem"), "hidden gem", "no", "none", "figurative", "no"),
    (L("Urban oasis"), "urban oasis", "no", "none", "figurative", "no"),
    (L("Neighborhood gems"), "neighborhood gem", "no", "none", "figurative", "no"),
    (L("Sidewalk cafe vibes"), "sidewalk cafe", "yes", "vibe", "atmosphere", "no"),
    (L("city hustle"), "city hustle", "yes", "none", "atmosphere", "no"),
    (L("little slice of heaven"), "slice of heaven", "no", "none", "figurative", "no"),
    (L("#citylife"), "city life", "yes", "none", "atmosphere", "no"),
    (L("#coffeeshop"), "coffee shop", "no", "none", "business_type", "no"),
    (L("#cafelife"), "cafe life", "yes", "none", "atmosphere", "no"),
    (L("#urbanvibe"), "urban", "yes", "vibe", "atmosphere", "no"),
    (L("#brunchvibes"), "brunch", "yes", "vibe", "atmosphere", "no"),
    (L("#hiddenbeauty"), "hidden beauty", "no", "none", "figurative", "no"),
]))

# 5. qwen3.8-27b caption answer, image 685. Lyrical lines. Skip: "Shutters
#    down" and "lanterns glowing" (plain facts), "cooler tone" (the tone of
#    the caption), the street names, #Brownstone and #FireEscape (building
#    parts), #StreetPhotography, #NYCgram, and #ShutteredUp (a plain fact).
t4 = text_of("qwen3.8-27b|caption|685|response")
ex.append(build(t4, [
    (L("the only thing holding the story together"), "the only thing holding the story together", "no", "none", "figurative", "no"),
    (L("feel alive"), "alive", "yes", "feel", "atmosphere", "no"),
    (L("old brick doing the night shift"), "old brick doing the night shift", "no", "none", "figurative", "no"),
    (L("#UrbanTexture"), "urban texture", "no", "none", "appearance", "no"),
    (L("#CityThatNeverSleeps"), "city that never sleeps", "yes", "none", "figurative", "no"),
    (L("#GritAndGrace"), "grit and grace", "yes", "none", "atmosphere", "no"),
]))

# 6. gemma-4-12b restaurant answer, image 514. Negated descriptors ("isn't a
#    flashy, large-scale restaurant"), hedged business types, and visual
#    details that the text gives as evidence ("warm interior lighting",
#    "black-framed windows"): skip the evidence, extract the quality it
#    supports. Skip "muted tones" and "sharp color contrast" (colors).
t5 = text_of("gemma-4-12b|restaurant|514|response")
ex.append(build(t5, [
    (L("slightly rustic"), "rustic", "no", "none", "appearance", "no"),
    *([(L("classic"), "classic", "no", "none", "appearance", "no"),
       (L("urban"), "urban", "no", "none", "appearance", "no")] if ONLY_DESCRIPTOR else
      [(L("classic urban neighborhood"), "classic urban neighborhood", "no", "none", "appearance", "no")]),
    (L("cozy"), "cozy", "yes", "atmosphere", "atmosphere", "no"),
    (L("intimate"), "intimate", "yes", "atmosphere", "atmosphere", "no"),
    (L('"hidden gem" atmosphere'), "hidden gem", "yes", "atmosphere", "figurative", "no"),
    (L("neighborhood tavern"), "neighborhood tavern", "no", "none", "business_type", "no"),
    (L("casual bistro"), "casual bistro", "no", "none", "business_type", "no"),
    (L("weathered"), "weathered", "no", "none", "appearance", "no"),
    (L("low-key"), "low-key", "yes", "none", "atmosphere", "no"),
    (L("stylish"), "stylish", "no", "none", "appearance", "no"),
    (L("neighborhood spot"), "neighborhood spot", "no", "none", "social_register", "no"),
    (L("flashy"), "flashy", "no", "none", "appearance", "yes"),
    (L("large-scale restaurant"), "large-scale restaurant", "no", "none", "business_type", "yes"),
    (L("relaxed dining or drinking experience"), "relaxed", "yes", "dining or drinking experience", "atmosphere", "no"),
    (L("historic"), "historic", "no", "none", "appearance", "no") if ONLY_DESCRIPTOR else
    (L("historic city setting"), "historic", "no", "city setting", "appearance", "no"),
]))

# 7. llama-3.2-90b restaurant answer, image 408, the last example. Hedged
#    vibes ("may have a playful or tongue-in-cheek vibe"), a "sense of"
#    frame, and a quality of the food. Skip: "small" (a size), the menu
#    items, and "cozy setting" (it repeats "cozy").
t6 = text_of("llama-3.2-90b|restaurant|408|response")
ex.append(build(t6, [
    (L("casual eatery"), "casual eatery", "no", "none", "business_type", "no"),
    (L("cozy atmosphere"), "cozy", "yes", "atmosphere", "atmosphere", "no"),
    (L("comfort food"), "comfort food", "no", "none", "business_type", "no"),
    (L("playful"), "playful", "yes", "vibe", "atmosphere", "no"),
    (L("tongue-in-cheek vibe"), "tongue-in-cheek", "yes", "vibe", "atmosphere", "no"),
    (L("secrecy"), "secrecy", "yes", "sense", "social_register", "no"),
    (L("exclusivity"), "exclusivity", "yes", "sense", "social_register", "no"),
    (L("unique dining experience"), "unique", "no", "dining experience", "other", "no"),
    (L("charming"), "charming", "yes", "none", "atmosphere", "no"),
    (L("intimate"), "intimate", "yes", "none", "atmosphere", "no"),
    (L("delicious food"), "delicious food", "no", "none", "other", "no"),
]))


# 8. qwen3.5-9b street answer (the angle prompt), image 48. Version 4 only.
#    The mood and look of the planned photo get target photo; the place words
#    get target place. Skip camera technique: "Leading Lines", "strong sense
#    of depth", "Point your camera upward", "Zoom in", and "Golden Hour" as
#    advice on when to shoot. Skip "great shot" (praise of a photo), the
#    labels "Best for Geometry", "Best for Atmosphere", "Best for
#    Storytelling", "jagged" and "tall" (shape and size), and the second
#    "striking" (a repeat).
if TARGET:
    t7 = text_of("qwen3.5-9b|street|48|response")
    ex.append(build(t7, [
        (L("Skeletal Skyline"), "skeletal skyline", "no", "none", "figurative", "no", "photo"),
        (L("striking"), "striking", "no", "none", "appearance", "no", "place"),
        (L("dramatic"), "dramatic", "yes", "feel", "atmosphere", "no", "photo"),
        (L("industrial feel"), "industrial", "yes", "feel", "atmosphere", "no", "photo"),
        (L("Depot Life"), "depot life", "yes", "none", "atmosphere", "no", "photo"),
        (L("overgrown"), "overgrown", "no", "none", "appearance", "no", "place"),
        (L("orderly"), "orderly", "no", "none", "appearance", "no", "place"),
        (L("chaotic"), "chaotic", "yes", "none", "atmosphere", "no", "place"),
        (L("stored-away nature"), "stored-away", "no", "nature", "appearance", "no", "place"),
    ]))

# 9. llama-3.2-90b street answer (the angle prompt), image 46. Version 5
#    only. A short answer that is mostly advice, and still holds 3
#    descriptors. Skip: "golden hour" (when to shoot), "depth", "wide-angle
#    lens", "tripod", "sharp image", "clear view", "scale and proportions".
if LLAMA_STREET:
    t8 = text_of("llama-3.2-90b|street|46|response")
    ex.append(build(t8, [
        (L("ornate"), "ornate", "no", "none", "appearance", "no", "place"),
        (L("warmth"), "warmth", "yes", "none", "atmosphere", "no", "photo"),
        (L("unique"), "unique", "no", "none", "appearance", "no", "place"),
    ]))

ex = [ex[i] for i in ORDER[VERSION]]


class Lit(str):
    pass


yaml.add_representer(Lit, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|"))
for e in ex:
    e["text"] = Lit(e["text"] + "\n")
sys.stdout.write(yaml.dump({"examples": ex}, allow_unicode=True, sort_keys=False, width=1000))
