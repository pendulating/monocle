#!/usr/bin/env python
"""F10.8: write the 2 extraction schemas of the traces of the read prompt.

| Schema | Source |
|--------|--------|
| `urban_cues_cool_v1` | `urban_cues_street_subj_v1`, with the question and the verdict labels of the read prompt |
| `cool_criteria_v1` | new: the criteria of a judgment, their value, stance, holder and visual evidence, and the notes on how the model decides |

The script takes the example texts from the traces and stops if a span is not verbatim.
The examples leave the analysis of the criteria: `example_ids()` names them.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/build_cool_schemas.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "dagspaces/urbanpairvqa/conf/extract"
RUNS = ROOT / "data/f4_traces/runs-cool"
VALUES = ["beauty", "order", "heritage", "modernity", "edge", "character", "vitality", "community", "status", "utility", "nature", "comfort", "other"]
NOTES = ["subjective", "no_taste", "forced_choice", "safe_default", "population_norm", "other"]
__version__ = "1.0.0"

DESCRIPTION = """You read text that a vision-language model wrote about one street-level
photograph in New York City. The user asked the model "Is this place cool?". A line before
the text names the model, and tells if the text is its private reasoning or its final
answer. The text can be a part of a longer text. Extract how the model decides. Do not
judge the image yourself, and do not add anything the text does not say.

Classes:

- criterion: 1 standard, quality or reason that the text uses to say why the place is or
  is not cool. The span is the shortest clause that holds the standard.
  attributes:
  value: the kind of standard. beauty (pretty, pleasing to the eye, attractive); order
  (clean, neat, polished, well kept, or the lack of it); heritage (historic, classic,
  old, traditional); modernity (new, modern, sleek); edge (gritty, industrial, raw,
  underground, urban grit); character (unique, quirky, charm, soul, authentic, hidden
  gem, or generic and lacking character); vitality (life, activity, things to do, or
  boring and nothing going on); community (neighborhood life, residents, local feel);
  status (upscale, trendy, hip, fashionable, wealth); utility (functional, ordinary,
  plain, utilitarian); nature (green, trees, sky, open air); comfort (safe, peaceful,
  quiet, pleasant to be in); other.
  stance: for (it makes the place cool), against (it makes the place not cool),
  conditional (cool for some people or under a condition).
  holder: model (the text says it as its own view), general (people in general, usually,
  typical views), persona (a kind of person: "if you like", "to an engineer").
  evidence: the thing in the photograph that the standard rests on, copied from the text,
  in at most 8 words. Write "" when the text names no such thing near the span.
- decision_note: a statement on how to reach the answer, and not about the place.
  attributes: kind. subjective ("cool" is subjective); no_taste (the model has no personal
  taste or feelings); forced_choice (the model must pick 1 answer); safe_default (the
  model takes an answer because it is safer, usual or expected); population_norm (the
  model appeals to what most people or AI guidelines say); other.
- none: the text holds nothing of the classes above. The span is the first words of the
  text.

Rules:

1. Copy each span from the text, character for character. Never paraphrase, and never join
   two parts of the text.
2. Extract in text order. A repeated thought is 1 extraction.
3. A plain description of the image with no standard ("There's a parked silver car.") is
   not a criterion. A word that judges the thing ("a bland concrete wall") is a criterion.
4. Extract nothing from the echo of the request, from headings, or from notes about the
   plan of the answer.
5. The temperature meaning of "cool" is not a criterion: use value other only when the
   text uses temperature to answer the question.
6. A text with nothing to extract gets 1 none span. Never answer with an empty list.
"""

EXAMPLES = [
    ("gemma-4-31b", "cool_open", 1268, 2, "response", [
        ("criterion", "a \"gritty\" city vibe", dict(value="edge", stance="conditional", holder="persona", evidence="chain-link fences, weathered concrete")),
        ("criterion", "can feel very edgy and modern", dict(value="edge", stance="conditional", holder="persona", evidence="large, plain warehouse buildings")),
        ("criterion", "**Hidden Gems:**", dict(value="character", stance="conditional", holder="persona", evidence="The \"POSE\" sign on the building")),
        ("criterion", "lush greenery", dict(value="nature", stance="conditional", holder="persona", evidence="")),
        ("criterion", "ornate architecture", dict(value="heritage", stance="conditional", holder="persona", evidence="")),
        ("criterion", "polished, clean environments", dict(value="order", stance="conditional", holder="persona", evidence="")),
        ("criterion", "some might find boring or purely functional", dict(value="utility", stance="against", holder="general", evidence="a parking lot and a loading zone")),
        ("decision_note", "\"cool\" is subjective", dict(kind="subjective")),
    ]),
    ("gemma-4-31b", "cool_word", 8171, 2, "reasoning", [
        ("criterion", "It's an industrial, utilitarian structure", dict(value="utility", stance="against", holder="model", evidence="A large green steel beam")),
        ("criterion", "To an engineer, the structural design might be cool", dict(value="other", stance="conditional", holder="persona", evidence="structural design")),
        ("criterion", "To most people, it's just a boring piece of infrastructure", dict(value="vitality", stance="against", holder="general", evidence="")),
        ("decision_note", "\"Cool\" is highly subjective", dict(kind="subjective")),
        ("decision_note", "the user is forcing a binary choice", dict(kind="forced_choice")),
        ("decision_note", "Avoid taking stances on subjective matters if possible", dict(kind="population_norm")),
        ("criterion", "It's bland", dict(value="beauty", stance="against", holder="model", evidence="a concrete wall and a green beam")),
    ]),
    ("qwen3.5-9b", "cool_word", 3721, 0, "reasoning", [
        ("criterion", "somewhat industrial or raw, but potentially promising for new construction", dict(value="edge", stance="conditional", holder="model", evidence="urban development")),
        ("criterion", "the vast open space under a blue sky appealing (peaceful, potential)", dict(value="nature", stance="for", holder="general", evidence="vast open space under a blue sky")),
        ("criterion", "Others might find it ugly (dirt piles, construction zone)", dict(value="beauty", stance="against", holder="general", evidence="dirt piles, construction zone")),
        ("decision_note", "\"Cool\" is subjective", dict(kind="subjective")),
        ("decision_note", "\"yes\" is the safer bet", dict(kind="safe_default")),
        ("decision_note", "\"yes\" is the standard affirmative for things that aren't obviously terrible or dangerous", dict(kind="safe_default")),
    ]),
    ("qwen3.5-9b", "cool_word", 5752, 2, "reasoning", [
        ("criterion", "Some might find the gritty urban texture cool", dict(value="edge", stance="conditional", holder="general", evidence="cracks in sidewalk")),
        ("criterion", "Others might just see a boring street", dict(value="vitality", stance="against", holder="general", evidence="")),
        ("criterion", "This scene has a moody, slightly urban vibe", dict(value="edge", stance="for", holder="model", evidence="")),
        ("decision_note", "I need to make a binary choice", dict(kind="forced_choice")),
        ("criterion", "The large tree framing the shot", dict(value="nature", stance="for", holder="model", evidence="The large tree")),
        ("criterion", "the texture of the brick", dict(value="heritage", stance="for", holder="model", evidence="the brick")),
    ]),
]


def example_ids() -> list[str]:
    return [f"{m}|{c}|{i}.{r}|{f}" for m, c, i, r, f, _ in EXAMPLES]


def source(model: str, case: str, idx: int, repeat: int, field: str) -> str:
    for line in (RUNS / model / case / "responses.jsonl").read_text().splitlines():
        row = json.loads(line)
        if int(row["image_idx"]) == idx and int(row["repeat"]) == repeat:
            return row[field].strip().replace("‑", "-")
    raise SystemExit(f"no trace {model} {case} {idx} {repeat}")


def main() -> None:
    examples = []
    for model, case, idx, repeat, field, spans in EXAMPLES:
        text = source(model, case, idx, repeat, field)
        order = []
        for name, span, attrs in spans:
            if span not in text:
                raise SystemExit(f"{idx}: not verbatim: {span}")
            for k, v in attrs.items():
                if k == "evidence" and v and v not in text:
                    raise SystemExit(f"{idx}: the evidence is not verbatim: {v}")
            order.append((text.index(span), dict(extraction_class=name, extraction_text=span, attributes=attrs)))
        examples.append(dict(text=text, extractions=[e for _, e in sorted(order, key=lambda x: x[0])]))
    head = dict(name="cool_criteria", schema_version="cool-criteria-v1", max_char_buffer=6000, context_window_chars=600, max_model_len=32768, batch_length=512,
                extraction_passes=1, temperature=0.0, max_tokens=6144, repetition_penalty=1.0, use_guided_json=True, require_extraction=True, rescue_substrings=True,
                attribute_values=dict(criterion=dict(value=VALUES, stance=["for", "against", "conditional"], holder=["model", "general", "persona"]), decision_note=dict(kind=NOTES)),
                prompt_description=DESCRIPTION, examples=examples)
    (CONF / "cool_criteria_v1.yaml").write_text("# @package extract\n# Written by scripts/f10/build_cool_schemas.py. Do not edit by hand.\n# Example texts:\n" + "".join(f"#   {i}\n" for i in example_ids())
                                                + yaml.safe_dump(head, sort_keys=False, allow_unicode=True, width=120))
    s = (CONF / "urban_cues_street_subj_v1.yaml").read_text()
    pairs = [('"Would\n  this street make a good spot for taking a photo?"', '"Is this\n  place cool?"'), ("schema_version: street-subj-v1", "schema_version: cool-cues-v1"),
             ("label (yes: a good spot; no: not a good spot; mixed: the", "label (yes: cool; no: not cool; mixed: the"), ("answer depends on the style of photo, or the model", "answer depends on the kind of person, or the model"),
             ("graded: one middle judgment with no yes and\n    no no:", "graded: one middle judgment with no yes and\n    no no:"), ("the model says it cannot judge a spot from this", "the model says it cannot judge the place from this"),
             ("A middle rating of the spot", "A middle rating of the place"), ("an answer\n     that depends on the style of photo is mixed", "an answer\n     that depends on the kind of person is mixed")]
    for a, b in pairs:
        if a not in s:
            raise SystemExit(f"the source schema changed: {a}")
        s = s.replace(a, b)
    (CONF / "urban_cues_cool_v1.yaml").write_text("# Copy of urban_cues_street_subj_v1 for the read prompt, by scripts/f10/build_cool_schemas.py.\n" + s)
    print("schemas written; examples", len(examples))


if __name__ == "__main__":
    main()
