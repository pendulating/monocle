"""The Integrative Complexity ingredient schema, and the prompt that fills it.

This module reads 1 reasoning trace and returns the raw material of an
Integrative Complexity (IC) code. It does NOT return a code and it does not
return a score. A later step computes those from the table, which lets us tune a
threshold without a second run on the GPU.

Why no few-shot example
-----------------------
`langextract_backend` builds its prompt from examples, and it derives its JSON
schema from the same examples. That path types every attribute as a bare
string, thus a binary column can hold `yes`, `Yes`, `true`, and `partially` at
one time.

This module writes the schema by hand. A closed field is a real `enum`, and the
grammar rejects everything else. It also drops about 4,500 tokens of examples
from the prefill.

The rule that makes it work without an example
----------------------------------------------
The IC codebook asks for judgments: is the uncertainty adequately justified? Is
the perspective substantively developed? A model answers a naked yes/no badly,
and no reader can audit the answer.

Thus we never ask for the judgment. We ask for the SPAN that justifies it, and
the code comes from the span.

| The codebook asks | The model returns | We derive |
|-------------------|-------------------|-----------|
| Uncertainty justified | `justification_quotes` | the list is not empty |
| Perspective developed | `supporting_quotes` | the list holds 2 or more |
| Weighing justified | `justification_quotes` | the list is not empty |
| Context sensitivity | `condition_quotes` | the list is not empty |

Grounds, without an aligner
---------------------------
LangExtract supplies a fuzzy aligner. This path has none, thus `locate_quote`
does the work: it finds the quote in the source text and returns the character
offsets. That gives 2 things back.

- A guard against invention. A quote that no search finds is a defect.
- The order of the spans, which `ic_codes` needs for `verdict_revised` and for
  `pseudo_differentiation`.

Warning: raise `SCHEMA_VERSION` when you change a field, an enum, or the prompt.
Never edit a version in place. The version reaches every output row, and 2
versions must never pool.

See `vlm-narratives-docs/ic-ingredient-extraction.md`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

__version__ = "1.0.0"

SCHEMA_NAME = "ic_ingredients"
SCHEMA_VERSION = "v2"

# The 7 kinds of ingredient. This order reaches the prompt AND the schema, and
# it is not free to choose.
#
# Warning: guided decoding fills the arrays in this order, thus the token cap
# eats the LAST ones. Measured on 2026-08-15 over the 30 cut traces of the v1
# pilot, survival fell in exactly the declared order:
#
# | dimension | perspective | dismissal | reconsideration | verdict | hedge | weighing |
# |-----------|-------------|-----------|-----------------|---------|-------|----------|
# | 100%      | 50%         | 53%       | 37%             | 30%     | 27%   | 7%       |
#
# `weighings` was last and it died. A cut trace is a LONG trace, thus the loss
# would fall on the most complex reasoning and push every IC score down.
#
# v2 puts the rare and necessary classes early and the voluminous one late.
# `dimensions` stays first because a perspective grows out of a dimension, and
# because it is the order a reader follows. `hedges` and `reconsiderations` are
# last: they are frequent, thus a partial list still shows that they are there.
INGREDIENT_TYPES = (
    "dimension",
    "perspective",
    "verdict",
    "weighing",
    "dismissal",
    "reconsideration",
    "hedge",
)

# The orders this schema can run in. The KEY is the schema version, thus 2
# orders can never pool: an ingredient rate belongs to the position it was
# asked in, and v2 measured that a move of 2 places cost `dismissal` 21 points.
#
# v3 exists to test that measurement, and it moves `dismissal` alone: back to
# 3rd, where v1 had it, with every other class in its v2 relative order.
ORDERS: Dict[str, Tuple[str, ...]] = {
    "v2": (
        "dimension", "perspective", "verdict", "weighing",
        "dismissal", "reconsideration", "hedge",
    ),
    "v3": (
        "dimension", "perspective", "dismissal", "verdict",
        "weighing", "reconsideration", "hedge",
    ),
}


def version_for(order: Sequence[str]) -> str:
    """Name the schema version of an order. An unknown order is not a version."""
    for name, known in ORDERS.items():
        if tuple(order) == known:
            return name
    raise ValueError(f"no schema version holds this order: {tuple(order)}")


def order_for(version: str) -> Tuple[str, ...]:
    """The ingredient order of 1 schema version."""
    if version not in ORDERS:
        raise ValueError(f"unknown schema version {version!r}; "
                         f"known: {sorted(ORDERS)}")
    return ORDERS[version]


# The array key of each ingredient type inside the answer object.
#
# Warning: this dict follows `INGREDIENT_TYPES`, and the schema reads the order
# from it. Two orders in two places would put the grammar and the reader out of
# step, and nothing would raise.
ARRAY_KEYS = {kind: kind + "s" for kind in INGREDIENT_TYPES}

# The fields of an ingredient that hold more spans. `locate_quote` tests every
# one of them, because a code derives from whether the list is empty. An
# invented justification would raise `uncertainty_justified` without cause.
SUB_QUOTE_FIELDS = ("supporting_quotes", "justification_quotes", "condition_quotes")

# A trace always names something it sees, and it always states a decision. A
# lower bound of 1 makes the grammar unable to write an empty array.
#
# Warning: do NOT bound the other 5. A trace with no weighing is the common
# case, and a bound would force the model to invent one.
BOUNDED_ARRAYS = ("dimensions", "verdicts")

VERDICT_LABELS = ("MuchLess", "Less", "Same", "More", "MuchMore", "NotSure")
DIMENSION_TYPES = ("descriptive", "evaluative")
IMAGES = ("A", "B", "both", "unclear")
VALENCES = ("good", "bad", "neutral")
FAVORS = ("A", "B", "neither")
HEDGE_MARKERS = ("qualifier", "contrastive", "explicit_doubt")
WEIGH_MECHANISMS = ("weighing", "balancing", "prioritization", "trade_off", "conditional")

# The token budget of 1 answer.
#
# Measured on 2026-08-15 over the 300-trace v1 pilot: 1 ingredient costs 54
# tokens, and a median answer holds 32 ingredients, thus 1,728 tokens.
#
# Warning: 4,096 was too small. It covers about 76 ingredients, but 10% of the
# subway traces wanted more, and the cap then cut the answer. A cut answer
# loses whole ARRAYS, not the tail of a list, thus the loss is silent. 8,192
# covers about 152 ingredients, against a measured p99 of 72.
#
# The budget still fits: 6,250 prompt + 7,650 for the longest trace + 8,192 =
# 22,092, inside `max_model_len: 24576`.
DEFAULT_MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# The JSON schema
# ---------------------------------------------------------------------------


def _array(item: Dict[str, Any], *, min_items: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": "array", "items": item}
    if min_items:
        out["minItems"] = min_items
    return out


def _obj(properties: Dict[str, Any]) -> Dict[str, Any]:
    """Build an object schema, and make every property necessary.

    A missing key and a key with an empty value are different facts. `required`
    removes the first one, thus the parser never guesses.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
    }


def _str_array() -> Dict[str, Any]:
    """A list of spans, possibly empty.

    Warning: use a list, never a nullable string. Support for `null` differs
    between structured-output backends, and an empty string cannot be told apart
    from the words "none" and "N/A" that a model writes instead.
    """
    return {"type": "array", "items": {"type": "string"}}


def json_schema(order: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Give the guided-decoding schema of 1 answer, in 1 ingredient order."""
    dimension = _obj(
        {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": list(DIMENSION_TYPES)},
            "image": {"type": "string", "enum": list(IMAGES)},
            "valence": {"type": "string", "enum": list(VALENCES)},
            "quote": {"type": "string"},
        }
    )
    perspective = _obj(
        {
            "name": {"type": "string"},
            "quote": {"type": "string"},
            "favors": {"type": "string", "enum": list(FAVORS)},
            "supporting_quotes": _str_array(),
            "is_winner": {"type": "boolean"},
        }
    )
    dismissal = _obj({"quote": {"type": "string"}, "target": {"type": "string"}})
    reconsideration = _obj({"quote": {"type": "string"}})
    verdict = _obj(
        {
            "label": {"type": "string", "enum": list(VERDICT_LABELS)},
            "quote": {"type": "string"},
            "is_final": {"type": "boolean"},
        }
    )
    hedge = _obj(
        {
            "quote": {"type": "string"},
            "marker_type": {"type": "string", "enum": list(HEDGE_MARKERS)},
            "justification_quotes": _str_array(),
            "affects_conclusion": {"type": "boolean"},
        }
    )
    weighing = _obj(
        {
            "quote": {"type": "string"},
            "mechanism": {"type": "string", "enum": list(WEIGH_MECHANISMS)},
            "justification_quotes": _str_array(),
            "condition_quotes": _str_array(),
        }
    )

    # Warning: this order is the generation order. See `INGREDIENT_TYPES`.
    items = {
        "dimension": _array(dimension, min_items=1),
        "perspective": _array(perspective),
        "verdict": _array(verdict, min_items=1),
        "weighing": _array(weighing),
        "dismissal": _array(dismissal),
        "reconsideration": _array(reconsideration),
        "hedge": _array(hedge),
    }
    return _obj({ARRAY_KEYS[kind]: items[kind]
                 for kind in (order or INGREDIENT_TYPES)})


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

# ─── The prompt, in 3 pieces ────────────────────────────────────────────────
#
# The prompt is NOT 1 string. It is a header, 1 block for each ingredient, and
# a footer, and `build_prompt` numbers the blocks in the order of the schema.
#
# Why: the position of a class in the list changes how much of it the model
# reports. Measured 2026-08-15, `dismissal` fell from 85.2% of traces to 64.2%
# when v2 moved it from 3rd to 5th, with the SAME instruction text and with no
# truncation involved. An order probe therefore has to move the prompt and the
# schema together, and a hard-coded "5." in the text would silently disagree
# with the array order.
PROMPT_HEADER = """\
You read the private reasoning that a vision-language model wrote while it \
compared two street-level photographs of New York City. Report the STRUCTURE of \
that reasoning.

Do not judge the photographs yourself. Do not add anything the text does not \
say. Your task is to describe how the model argued, not whether it was right.

Rules for every quote:

1. Copy the quote from the trace, character for character. It must be a span \
you can point to in the text.
2. Never paraphrase. Never join two separate parts of the text.
3. Keep a quote short. One clause is usually enough.
4. The first sentence of a trace restates the question the model was asked. \
Report nothing from it.
5. Report the items of each list in the order they appear in the trace.
6. Never report the same span two times in the same list.

Report these seven kinds of thing.
"""

# 1 block for each ingredient, WITHOUT its number. `build_prompt` numbers them.
TYPE_BLOCKS: Dict[str, str] = {
    "dimension": """\
dimensions — an attribute of a scene that the trace treats as relevant to \
its judgment. "Multi-story brick buildings" is a dimension of architecture. \
"Wide sidewalk with trees" is a dimension of infrastructure.
   - name: your own short label for the KIND of attribute, in lower case with \
underscores (for example: architecture, infrastructure, traffic, greenery, \
upkeep, cleanliness, light, people, commerce, signage). Use the same name every \
time you see the same kind of attribute, in both photographs.
   - type: "descriptive" when the trace only says what it sees. "evaluative" \
when the trace says the attribute is good or bad. "A well-maintained street" is \
evaluative.
   - image: which photograph the attribute belongs to.
   - valence: whether the trace treats the attribute as good, bad, or neutral.
   - quote: the span.
""",
    "perspective": """\
perspectives — an evaluative standard. A standard says WHY a dimension \
counts for or against a scene. It is a general rule, not a single observation.
   "The street is dense." is a dimension.
   "Dense streets and urban architecture offer more opportunity for street \
photography." is a perspective.
   - name: your own short label for the standard, in lower case with \
underscores (for example: urban, suburban, pedestrian, safety, prestige, \
liveliness, photogenic).
   - quote: the span that states the standard.
   - favors: which photograph the standard counts for.
   - supporting_quotes: other spans where the trace develops this standard. \
Leave the list empty when the trace only names the standard and moves on.
   - is_winner: true only for the standard that decides the final answer.
""",
    "verdict": """\
verdicts — any statement of a decision, INCLUDING one that the trace later \
changes. A trace that decides three times gives three verdicts.
   - label: the decision.
   - quote: the span.
   - is_final: true only for the last decision the trace settles on.
""",
    "weighing": """\
weighings — the trace sets two standards, or two dimensions, against each \
other, and says which one counts more.
   - quote: the span.
   - mechanism: which move the trace makes.
   - justification_quotes: the spans that explain WHY one side counts more. \
Leave the list empty when the trace only says "overall", "despite", or \
"on balance" and explains nothing.
   - condition_quotes: spans that name a condition under which the weight \
would change. Leave the list empty when there is none.
""",
    "dismissal": """\
dismissals — the trace names a cue or a standard and then sets it aside as \
not relevant, or not worth more thought.
   - quote: the span.
   - target: a short label for what the trace sets aside.
""",
    "reconsideration": """\
reconsiderations — the trace signals that it reopens a judgment it already \
made. Examples: "Wait", "Let me re-evaluate", "Actually", "Let's look again".
   - quote: the span.
""",
    "hedge": """\
hedges — language that qualifies a claim, or that admits doubt.
   - quote: the span.
   - marker_type: "qualifier" for a word such as probably, usually, seems, \
maybe, slightly. "contrastive" for a word such as but, however, while, \
although, nevertheless. "explicit_doubt" for a statement such as "hard to \
tell" or "I cannot see".
   - justification_quotes: the spans that give a REASON for the doubt. Leave \
the list empty when the trace only uses the word and gives no reason.
   - affects_conclusion: true when the doubt changes the comparison or the \
final answer.
""",
}

PROMPT_FOOTER = """\
An empty list is the correct answer for dismissals, reconsiderations, hedges, \
and weighings. Many traces hold none of them. Do not invent one.
{examples}
Report the reasoning of this trace:

<trace>
{trace}
</trace>
"""

# The header above the worked examples.
#
# Warning: the 3 examples are chosen to show every kind of thing at least one
# time, thus they are NOT a sample. Measured over the 11,000 subway traces of
# 2026-08-13, a weighing word appears in 0.6% of them, but 3 of 3 examples hold
# a weighing. A few-shot prompt leaks the base rate of its examples into the
# answers, thus the note below tells the model what the examples are for.
EXAMPLES_HEADER = """
Here are worked examples. They are chosen to show every kind of thing at least \
one time. They are not a typical sample: most traces hold fewer things, and \
many hold no dismissal, no weighing, and no hedge at all. Report only what the \
trace in front of you holds.
"""


def render_examples(examples: Sequence[Dict[str, Any]]) -> str:
    """Turn the examples into the block that sits above the trace.

    Warning: put this block BEFORE the trace and never after it. Every request
    shares the same block, thus a prefix cache serves it one time and then
    charges nothing for it.
    """
    if not examples:
        return "\n"
    parts = [EXAMPLES_HEADER]
    for number, example in enumerate(examples, start=1):
        answer = json.dumps(example["answer"], indent=1, ensure_ascii=False)
        parts.append(
            f"\n### Example {number}\n\n"
            f"<trace>\n{example['trace']}\n</trace>\n\n"
            f"Answer:\n{answer}\n"
        )
    return "".join(parts) + "\n"


def load_examples(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read the examples that `scripts/build_ic_examples.py` wrote.

    Warning: the file is JSON, not YAML. A YAML block scalar folds a long line
    and strips trailing space, thus a span that looks correct in the editor
    would not be in the trace any more.
    """
    import os

    if path is None:
        path = os.path.join(os.path.dirname(__file__), f"ic_examples_{SCHEMA_VERSION}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no example file at {path}. Run scripts/build_ic_examples.py"
        )
    with open(path) as handle:
        payload = json.load(handle)
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"the examples are {version} and the schema is {SCHEMA_VERSION}. "
            "Two versions must never pool."
        )
    return list(payload.get("examples") or [])


def build_prompt(
    trace: str, examples: Optional[Sequence[Dict[str, Any]]] = None,
    order: Optional[Sequence[str]] = None,
) -> str:
    """Put 1 trace into the prompt, under the examples.

    Pass `examples=None` for the zero-shot prompt. The pilot runs both over the
    same traces, which measures what the examples buy and what they leak.

    Warning: `PROMPT` holds 2 placeholders, and this fills both in 1 pass.
    `str.format` would read a brace inside the trace itself, and a trace holds
    JSON. Two calls to `str.replace` would be no better: the first call writes
    the trace into the prompt, and the second call would then read a
    placeholder that the TRACE holds.
    """
    prompt = assemble_prompt(order)
    filling = {
        "{examples}": render_examples(examples or []),
        "{trace}": trace or "",
    }
    return re.sub(
        r"\{examples\}|\{trace\}", lambda m: filling[m.group(0)], prompt, count=2
    )


def assemble_prompt(order: Optional[Sequence[str]] = None) -> str:
    """Build the prompt text for 1 ingredient order.

    The blocks are numbered in the order given, thus the text and the schema
    always agree about which class comes first.
    """
    order = tuple(order or INGREDIENT_TYPES)
    missing = [k for k in order if k not in TYPE_BLOCKS]
    if missing or len(set(order)) != len(TYPE_BLOCKS):
        raise ValueError(f"the order must name every ingredient once: {order}")
    # Every piece is stripped of its own blank lines, thus the spacing of the
    # prompt lives here and in 1 place only. A stray newline in a literal then
    # cannot change the prompt, and the v2 text stays byte for byte the same.
    body = "\n\n".join(f"{i}. {TYPE_BLOCKS[name].strip(chr(10))}"
                       for i, name in enumerate(order, 1))
    return (PROMPT_HEADER.strip("\n") + "\n\n" + body + "\n\n"
            + PROMPT_FOOTER.strip("\n") + "\n")


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def close_truncated_json(text: str) -> Optional[str]:
    """Close a JSON answer that the token cap cut in the middle.

    Guided decoding writes valid JSON up to the cap. The cap then cuts it, and
    `json.loads` drops the WHOLE answer, not the tail of it. Thus a cut answer
    looks the same as a silent one.

    This walks to the last point where a container closed, cuts there, and
    closes every container that is still open.

    Returns:
        The repaired text, or None when no container closed.

    Warning: the repair loses the last item, and no reader can know what else
    the model was about to write. Count the repairs, and put the number in the
    stage metadata.
    """
    stack: List[str] = []
    in_string = False
    escape = False
    # The index after the last close, and the stack as it was at that moment.
    safe_end: Optional[int] = None
    safe_stack: List[str] = []

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
            # A cut here leaves valid JSON once we close what is still open.
            safe_end = i + 1
            safe_stack = list(stack)

    if safe_end is None:
        return None
    return text[:safe_end] + "".join(reversed(safe_stack))


def parse_answer(text: str) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    """Read the answer object out of a model answer.

    Warning: a repair can lose a WHOLE array. The cut point sits after the last
    container that closed, thus an array that never closed disappears from the
    object. A lost `verdicts` key then looks the same as a trace that stated no
    verdict, which is the silent failure this project met in v1.

    Thus the third return value exists. Mark every row of a repaired answer, and
    drop those rows from a count of absence.

    Returns:
        (object, error, repaired). The object is None when nothing parses, and
        `error` then says why. `error` is an empty string on success.
    """
    raw = (text or "").strip()
    if not raw:
        return None, "empty answer", False

    # Guided decoding writes bare JSON. A model that ignores the constraint can
    # still wrap it in a fence or add a sentence, thus we cut to the braces.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1:
        return None, "no JSON object in the answer", False
    candidate = raw[start : end + 1] if end > start else raw[start:]

    was_repaired = False
    try:
        obj: Any = json.loads(candidate)
    except json.JSONDecodeError as first_error:
        repaired = close_truncated_json(candidate)
        if repaired is None:
            return None, f"invalid JSON: {first_error}", False
        try:
            obj = json.loads(repaired)
        except json.JSONDecodeError as second_error:
            return None, f"invalid JSON after repair: {second_error}", False
        was_repaired = True

    if not isinstance(obj, dict):
        return None, f"the answer is a {type(obj).__name__}, not an object", was_repaired
    return obj, "", was_repaired


# ---------------------------------------------------------------------------
# The grounds
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _normalise(text: str) -> Tuple[str, List[int]]:
    """Collapse each run of whitespace to 1 space, and keep the index map.

    The map gives the index in the SOURCE text of each character of the
    normalised text. Thus a match in the normalised text maps back to a real
    offset.
    """
    out: List[str] = []
    index: List[int] = []
    previous_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if previous_space:
                continue
            out.append(" ")
            index.append(i)
            previous_space = True
        else:
            out.append(ch)
            index.append(i)
            previous_space = False
    return "".join(out), index


def locate_quote(trace: str, quote: str) -> Tuple[Optional[int], Optional[int], str]:
    """Find a quote in its trace.

    This replaces the LangExtract aligner. It takes 2 steps, and it reports
    which step found the quote.

    | Method | What it means |
    |--------|---------------|
    | `exact` | The quote is the source text, character for character |
    | `whitespace` | The quote is the source text, but a line break moved |
    | `none` | No search found it; the model composed the span |

    Warning: a `none` quote is a defect, not data. Keep the row, and drop it
    from every count. A derived code must never rest on a quote we cannot find.

    Returns:
        (start, end, method). Start and end are None when the method is `none`.
    """
    if not quote or not trace:
        return None, None, "none"

    at = trace.find(quote)
    if at != -1:
        return at, at + len(quote), "exact"

    flat_trace, index = _normalise(trace)
    flat_quote = _WS.sub(" ", quote).strip()
    if not flat_quote:
        return None, None, "none"
    at = flat_trace.find(flat_quote)
    if at == -1:
        return None, None, "none"
    start = index[at]
    end = index[at + len(flat_quote) - 1] + 1
    return start, end, "whitespace"


# ---------------------------------------------------------------------------
# The output rows
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _clean_sub_quotes(value: Any) -> List[str]:
    """Keep the real spans of a sub-quote list.

    A model that has no span to give sometimes writes the word instead of an
    empty list. Those words are not spans, and a code must not count them.
    """
    out: List[str] = []
    for item in _as_list(value):
        text = str(item).strip()
        if not text or text.lower() in {"none", "null", "n/a", "na", "-"}:
            continue
        out.append(text)
    return out


def ingredient_rows(
    trace: str,
    obj: Optional[Dict[str, Any]],
    *,
    base: Optional[Dict[str, Any]] = None,
    parse_error: str = "",
    order: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Turn 1 answer into the long output rows.

    1 row holds 1 ingredient. A trace that gives no ingredient still writes 1
    row, with a null `ingredient_type`. Without that row a reader cannot
    separate "the model reported nothing" from "the stage did not run".
    """
    order = tuple(order or INGREDIENT_TYPES)
    base = dict(base or {})
    # The version follows the ORDER, never a constant. A row that says v2 while
    # the run asked in the v3 order would pool 2 incomparable measurements.
    base.update({"schema_name": SCHEMA_NAME, "schema_version": version_for(order)})

    rows: List[Dict[str, Any]] = []
    if obj is not None:
        for kind in order:
            items = _as_list(obj.get(ARRAY_KEYS[kind]))
            for position, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                rows.append(_row(trace, kind, position, item, base, parse_error))

    if not rows:
        empty = dict(base)
        empty.update(
            {
                "ingredient_type": None,
                "ingredient_index": None,
                "name": None,
                "quote": None,
                "char_start": None,
                "char_end": None,
                "quote_method": None,
                "quote_found": False,
                "attrs_json": None,
                "n_sub_quotes": 0,
                "n_sub_quotes_found": 0,
                "parse_error": parse_error or "no ingredient in the answer",
            }
        )
        return [empty]
    return rows


def _row(
    trace: str,
    kind: str,
    position: int,
    item: Dict[str, Any],
    base: Dict[str, Any],
    parse_error: str,
) -> Dict[str, Any]:
    """Build 1 output row, and test every span it holds."""
    quote = str(item.get("quote", "") or "")
    start, end, method = locate_quote(trace, quote)

    # A sub-quote decides a code on its own, thus it needs the same test as the
    # main quote. `justified` must never rest on a span we cannot find.
    attrs: Dict[str, Any] = {}
    n_sub = 0
    n_sub_found = 0
    for key, value in item.items():
        if key in ("quote", "name"):
            continue
        if key in SUB_QUOTE_FIELDS:
            spans = _clean_sub_quotes(value)
            checked: List[Dict[str, Any]] = []
            for span in spans:
                s, e, m = locate_quote(trace, span)
                found = m != "none"
                n_sub += 1
                n_sub_found += int(found)
                checked.append(
                    {"text": span, "char_start": s, "char_end": e, "method": m}
                )
            attrs[key] = checked
        else:
            attrs[key] = value

    row = dict(base)
    row.update(
        {
            "ingredient_type": kind,
            "ingredient_index": position,
            "name": (str(item["name"]).strip() if item.get("name") is not None else None),
            "quote": quote,
            "char_start": start,
            "char_end": end,
            "quote_method": method,
            "quote_found": method != "none",
            "attrs_json": json.dumps(attrs, sort_keys=True, default=str),
            "n_sub_quotes": n_sub,
            "n_sub_quotes_found": n_sub_found,
            "parse_error": parse_error,
        }
    )
    return row


def answer_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Report how well a set of rows is grounded."""
    real = [r for r in rows if r.get("ingredient_type")]
    if not real:
        return {"ingredients": 0, "quote_found_rate": 0.0, "exact_rate": 0.0}
    found = sum(1 for r in real if r["quote_found"])
    exact = sum(1 for r in real if r["quote_method"] == "exact")
    subs = sum(int(r["n_sub_quotes"]) for r in real)
    subs_found = sum(int(r["n_sub_quotes_found"]) for r in real)
    out: Dict[str, Any] = {
        "ingredients": len(real),
        "quote_found_rate": round(found / len(real), 4),
        "exact_rate": round(exact / len(real), 4),
        "sub_quotes": subs,
        "sub_quote_found_rate": round(subs_found / subs, 4) if subs else 0.0,
    }
    for kind in INGREDIENT_TYPES:
        count = sum(1 for r in real if r["ingredient_type"] == kind)
        if count:
            out[f"type/{kind}"] = count
    return out
