#!/usr/bin/env python
"""F4: print the examples block of `caption_options_v3.yaml` from real F4 answers.

Each span is a slice of its answer between 2 anchors, thus it is an exact copy
and the aligner can match it. Change a span here, not by hand in the YAML,
then paste the output under `prompt_description` and raise the version.
`caption-v1` has the first 4 examples and the negative example last.
`caption-v2` adds llama-3.2-90b image 3. `caption-v3` adds llama-3.2-90b image
153 and moves the negative example to the first place.
`caption-v4` (`--version v4`, D130) changes 3 things:

1. The negative example of v3 (llama-3.2-90b, image 130) offers a caption
   ('A possible Instagram caption for this image could be: "Park life..."'),
   but it had no extraction. It taught the model to skip an offered caption.
   v4 drops it.
2. A plain description (llama-3.2-90b, image 37) is the new negative example,
   with a grounded `none` item (its first words), the x2 pattern.
3. A Llama answer that is 1 quoted text (llama-3.2-11b, image 189) is a
   caption (advisor review, 21 such answers had no caption).

v4 changes U+2011 into "-" in every example text, as the x2 input does.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/build_caption_examples.py > examples.yaml
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/build_caption_examples.py --order retry > examples.yaml
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/build_caption_examples.py --version v4 > examples.yaml
"""
import sys, yaml, pandas as pd
t = pd.read_parquet("/share/ju/matt/mllmsci/data/f4_traces/f4_traces_1k.parquet")
c = t[t.case == "caption"]

def answer(model, idx):
    return c[(c.model == model) & (c.image_idx == idx)].response.iloc[0].strip("\n")

def span(text, start, stop, after=0):
    """The slice from `start` up to (not with) `stop`, searched after `after`."""
    i = text.index(start, after)
    j = text.index(stop, i + len(start)) if stop else len(text)
    return text[i:j].rstrip(), i

def build(text, specs):
    out, cursor = [], 0
    for cls, start, stop, attrs in specs:
        s, i = span(text, start, stop, cursor)
        assert s in text and i >= cursor, (cls, start)
        cursor = i + len(s)
        out.append({"extraction_class": cls, "extraction_text": s, "attributes": attrs})
    return {"text": text, "extractions": out}

def cap(style):
    return {"style": style}
CAP, GEN = {"scope": "caption"}, {"scope": "general"}

ex = []
t1 = answer("gemma-4-12b", 646)
ex.append(build(t1, [
    ("caption", "Keeping it fresh", " #FreshKitchen", cap('The "Foodie" Vibe')),
    ("hashtags", "#FreshKitchen", "\n", CAP),
    ("caption", "Found my new", "\n", cap('The "Foodie" Vibe')),
    ("caption", "Looking for a fresh", "\n", cap('The "Foodie" Vibe')),
    ("caption", "Lunch plans", "\n", cap('The "Foodie" Vibe')),
    ("caption", "Just another day", "\n", cap('The "City Explorer" Vibe')),
    ("caption", "Urban adventures", "\n", cap('The "City Explorer" Vibe')),
    ("caption", "City sights", "\n", cap('The "City Explorer" Vibe')),
    ("caption", "Finding the best", "\n", cap('The "City Explorer" Vibe')),
    ("caption", "A very \"fresh\"", "\n", cap('The "Punny/Witty" Vibe')),
    ("caption", "Feeling very", " (Referring", cap('The "Punny/Witty" Vibe')),
    ("caption", "Stay fresh", "\n", cap('The "Punny/Witty" Vibe')),
    ("caption", "City vibes.", "\n", cap("Short & Aesthetic")),
    ("caption", "Today", "\n", cap("Short & Aesthetic")),
    ("caption", "Fresh. ", "\n", cap("Short & Aesthetic")),
    ("caption", "Street scene.", "\n", cap("Short & Aesthetic")),
    ("hashtags", "#NYC", None, GEN),
]))
t2 = answer("qwen3.5-9b", 463)
ex.append(build(t2, [
    ("caption", "Morning runs", " #DunkinLife", cap("The Coffee Lover")),
    ("hashtags", "#DunkinLife", "\n", CAP),
    ("caption", "Love these little", " #CityWalks", cap("Neighborhood Vibe")),
    ("hashtags", "#CityWalks", "\n", CAP),
    ("caption", "Caffeine secured", "\n", cap("Short & Punchy")),
    ("caption", "Me pretending", " #MondayMood", cap("Funny/Relatable")),
    ("hashtags", "#MondayMood", "\n", CAP),
    ("caption", "Spotting a familiar", " #NYCLife", cap("Documentation Style")),
    ("hashtags", "#NYCLife", None, CAP),
]))
t3 = answer("qwen3.8-27b", 685)
ex.append(build(t3, [
    ("caption", "Shutters down", "\n", cap("none")),
    ("caption", "114 Eldridge", "\n", cap("Alt (cooler tone)")),
    ("hashtags", "#EldridgeSt", None, GEN),
]))
t4 = answer("llama-3.2-11b", 3)
ex.append(build(t4, [
    ("caption", "Sunday vibes", " #coffeebreak", cap("none")),
    ("hashtags", "#coffeebreak", "\"", CAP),
]))
# caption-v2: a description, then a caption that the model offers and that
# reads like a description. caption-v1 had no such example, and the smoke run
# (job 94158) missed 2 such captions and copied an example into a 3rd answer.
t5 = answer("llama-3.2-90b", 3)
ex.append(build(t5, [
    ("caption", "Discovering hidden gems", " #citylife", cap("none")),
    ("hashtags", "#citylife", "\"", CAP),
]))
# caption-v3: a short description, then "The caption could be:" and a caption
# that reads like a description, with no hashtag. Under caption-v2 the smoke
# run (job 94187) still gave no extraction for 2 answers of this shape.
t6 = answer("llama-3.2-90b", 153)
ex.append(build(t6, [
    ("caption", "A quiet street scene", "\"", cap("none")),
]))
# caption-v3: the negative example goes FIRST. In caption-v2 it was the last
# example before the input, and the model then repeated its empty answer on
# short Llama answers (a guess from the smoke runs, not a measured cause).
ex.insert(0, {"text": answer("llama-3.2-90b", 130), "extractions": []})
# caption-v3r (the retry pass, D105): the same examples, the positive ones in
# reverse order. The negative example stays first.
if "--order" in sys.argv and sys.argv[sys.argv.index("--order") + 1] == "retry":
    ex = ex[:1] + ex[1:][::-1]
if "--version" in sys.argv and sys.argv[sys.argv.index("--version") + 1] == "v4":
    t7 = answer("llama-3.2-11b", 189)
    quoted = build(t7, [("caption", "Blueberry Mexican Food is", "\"", cap("none"))])
    t0 = answer("llama-3.2-90b", 37)
    # The `none` item: the first words of a text that offers no caption.
    negative = {"text": t0, "extractions": [{"extraction_class": "none", "extraction_text": "A blue truck is parked"}]}
    ex = [negative] + ex[1:-1] + [quoted, ex[-1]]
    for e in ex:
        e["text"] = e["text"].replace("\u2011", "-")
        for x in e["extractions"]:
            x["extraction_text"] = x["extraction_text"].replace("\u2011", "-")

class Lit(str): pass
yaml.add_representer(Lit, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|"))
for e in ex:
    e["text"] = Lit(e["text"] + "\n")
sys.stdout.write(yaml.dump({"examples": ex}, allow_unicode=True, sort_keys=False, width=1000))
