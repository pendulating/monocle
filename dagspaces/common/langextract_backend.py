"""Bridge between LangExtract and the project's vLLM engine.

LangExtract turns a free-text reasoning trace into typed, grounded extractions.
It gives 4 things we do not want to write again:

- a few-shot prompt builder,
- a tolerant JSON parser,
- a fuzzy aligner that maps each extraction back to a character span of the
  source text,
- an HTML view of those spans.

This module supplies the 1 thing LangExtract does not have: our inference
engine.

Who drives the GPU
------------------
LangExtract must NOT drive the GPU. `lx.extract` hands a whole batch of prompts
to `BaseLanguageModel.infer`, and `VLLMLanguageModel.infer` sends that batch to
`LLM.generate` in 1 call. The engine schedules the batch. Thus the stage layer
keeps the batch, the shard, and the SLURM job, and LangExtract keeps the prompt,
the parse, and the alignment.

Warning: ignore `max_workers`. It exists for an HTTP provider that gains from
threads. A local engine schedules its own batch, and threads only add
contention. `lx.extract` warns when `batch_length < max_workers`, thus a caller
must pass `max_workers=1`.

Warning: install LangExtract with `--no-deps`.

    uv pip install --no-deps langextract absl-py ml-collections more-itertools

A plain install pulls `google-genai` and `google-cloud-storage`, which we never
call. It also moves `websockets` down from 17.0.1 to 16.1.1.

See `research/docs/langextract-trace-extraction.md`.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

from langextract import data as lx_data
from langextract.core import base_model as lx_base_model
from langextract.core import types as lx_types

__version__ = "1.3.0"

# The default token budget of 1 extraction. A trace that names 20 cues writes
# about 1,200 tokens of JSON, thus 2,048 gives room without a long tail of
# wasted decode steps.
DEFAULT_MAX_TOKENS = 2048

# 1 trace must become 1 chunk. The subway thinking run of 2026-08-13 has a p99
# of 8,973 characters and a maximum of 19,252.
#
# Warning: the LangExtract default is 1000. At that value a median trace splits
# into 4 chunks, no chunk holds both images, and every A-against-B comparison is
# lost. Never accept the default.
DEFAULT_MAX_CHAR_BUFFER = 12000

# The number of prompts that reach `generate()` at one time.
DEFAULT_BATCH_LENGTH = 512

# `VLLMEngine` and `repair_truncated_json` moved to `vllm_structured` on
# 2026-08-15, because `ic_extract` uses the engine and must NOT need
# LangExtract. They are re-exported here, thus every existing import of
# `langextract_backend` keeps working.
from dagspaces.common.vllm_structured import (  # noqa: E402,F401
    VLLMEngine,
    repair_truncated_json,
)


# ---------------------------------------------------------------------------
# The specification of 1 extraction task
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExtractionSpec:
    """Everything that defines an extraction task, except the input text.

    The stage builds this from a `conf/extract/*.yaml` config group. Write the
    `schema_version` into every output row: a schema change makes a new version,
    never an edit in place.
    """

    name: str
    schema_version: str
    prompt_description: str
    examples: Sequence[lx_data.ExampleData]
    max_char_buffer: int = DEFAULT_MAX_CHAR_BUFFER
    batch_length: int = DEFAULT_BATCH_LENGTH
    extraction_passes: int = 1
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    use_guided_json: bool = True
    require_extraction: bool = True
    # vLLM's repetition penalty; 1.0 is off. See `VLLMStructuredEngine.generate`.
    repetition_penalty: float = 1.0
    # LangExtract's `context_window_chars`: a chunk after the first gets the
    # tail of the chunk before it as "[Previous text]". None is off. Use it
    # when `max_char_buffer` cuts a text into chunks.
    context_window_chars: Optional[int] = None
    # Find the span in the text again when the aligner did not place it
    # exactly (`rescue_offsets`). Measured on 2026-10-26: 2,054 of 7,576
    # `match_lesser` rows of the F4 cue run were verbatim in the text.
    rescue_substrings: bool = False
    # class -> attribute -> the values the model may write. An attribute that
    # is absent here stays a free string, such as `comparison.dimension`.
    # Version 2 of `urban_cues` fills this. Version 1 left it empty, and the
    # model then wrote values that the prompt never listed.
    attribute_values: Mapping[str, Mapping[str, Sequence[str]]] = dataclasses.field(
        default_factory=dict
    )

    @property
    def classes(self) -> List[str]:
        """Name every extraction class that the examples teach."""
        seen: List[str] = []
        for example in self.examples:
            for extraction in example.extractions:
                if extraction.extraction_class not in seen:
                    seen.append(extraction.extraction_class)
        return seen

    def guided_json(self) -> Optional[Dict[str, Any]]:
        """Give the JSON schema for guided decoding, or None.

        `GeminiSchema.from_examples` reads the classes and the attributes out of
        the examples and returns a plain JSON schema. vLLM accepts that dict as
        a structured-output constraint, thus almost no parse can fail.

        Warning: the schema holds only the classes that an example teaches. A
        class with no example cannot be produced under guided decoding.
        """
        if not self.use_guided_json:
            return None
        from langextract.providers.schemas.gemini import GeminiSchema

        schema = GeminiSchema.from_examples(list(self.examples)).schema_dict
        if self.require_extraction:
            # An empty list is the main silent failure. Measured on 2026-08-13
            # over 24 long subway traces: 15 came back as `{"extractions": []}`
            # in about 11 tokens. The grammar cannot write an empty array once
            # the array has a lower bound.
            schema["properties"][lx_data.EXTRACTIONS_KEY]["minItems"] = 1
        if self.attribute_values:
            _apply_attribute_enums(schema, self.attribute_values)
        return schema


def _apply_attribute_enums(
    schema: Dict[str, Any], attribute_values: Mapping[str, Mapping[str, Sequence[str]]]
) -> None:
    """Fix the attribute values of each class in the guided schema.

    `GeminiSchema.from_examples` names an attribute and types it as a string.
    This puts an `enum` on each declared attribute, and it makes the attribute
    `required` inside its attributes object. Thus the grammar cannot write a
    value outside the list, and it cannot leave a declared attribute out.

    A declared attribute that no example uses is added here, because the
    schema must hold every attribute the prompt names.

    Warning: this reads the LangExtract shape, `<class>` for the span and
    `<class>_attributes` for its attributes, both under `items.properties`.
    """
    items = schema["properties"][lx_data.EXTRACTIONS_KEY]["items"]
    props = items.setdefault("properties", {})
    for klass, attrs in attribute_values.items():
        if not attrs:
            continue
        props.setdefault(klass, {"type": "string"})
        holder = props.setdefault(
            f"{klass}{lx_data.ATTRIBUTE_SUFFIX}", {"type": "object", "properties": {}}
        )
        holder_props = holder.setdefault("properties", {})
        required = list(holder.get("required", []))
        for attribute, values in attrs.items():
            entry = holder_props.setdefault(attribute, {"type": "string"})
            entry["type"] = "string"
            entry["enum"] = [str(v) for v in values]
            if attribute not in required:
                required.append(attribute)
        holder["required"] = required


def examples_from_config(raw_examples: Sequence[Any]) -> List[lx_data.ExampleData]:
    """Turn the `examples` block of a config group into `ExampleData`.

    The config shape is:

        examples:
          - text: "..."
            extractions:
              - extraction_class: visual_evidence
                extraction_text: "trash can"
                attributes: {image: A, valence: bad}

    An example with an empty `extractions` list is a NEGATIVE example. Keep at
    least 1. A trace repeats the prompt ("The user wants me to compare..."), and
    without a negative example the model reads that echo as evidence.

    Warning: `extraction_text` must appear in `text` word for word. The aligner
    matches the span against the source, and a paraphrase cannot align.
    """
    out: List[lx_data.ExampleData] = []
    for item in raw_examples or []:
        item = _to_plain(item)
        extractions = []
        for ex in item.get("extractions") or []:
            ex = _to_plain(ex)
            attributes = ex.get("attributes") or None
            if attributes is not None:
                attributes = {str(k): _attr_value(v) for k, v in _to_plain(attributes).items()}
            extractions.append(
                lx_data.Extraction(
                    extraction_class=str(ex["extraction_class"]),
                    extraction_text=str(ex["extraction_text"]),
                    attributes=attributes,
                )
            )
        out.append(lx_data.ExampleData(text=str(item["text"]), extractions=extractions))
    return out


def spec_from_config(cfg_extract: Any) -> ExtractionSpec:
    """Build an `ExtractionSpec` from the `extract` config group."""
    cfg = _to_plain(cfg_extract)
    return ExtractionSpec(
        name=str(cfg.get("name", "unnamed")),
        schema_version=str(cfg.get("schema_version", "0")),
        prompt_description=str(cfg.get("prompt_description", "")),
        examples=examples_from_config(cfg.get("examples") or []),
        max_char_buffer=int(cfg.get("max_char_buffer", DEFAULT_MAX_CHAR_BUFFER)),
        batch_length=int(cfg.get("batch_length", DEFAULT_BATCH_LENGTH)),
        extraction_passes=int(cfg.get("extraction_passes", 1)),
        temperature=float(cfg.get("temperature", 0.0)),
        max_tokens=int(cfg.get("max_tokens", DEFAULT_MAX_TOKENS)),
        use_guided_json=bool(cfg.get("use_guided_json", True)),
        require_extraction=bool(cfg.get("require_extraction", True)),
        repetition_penalty=float(cfg.get("repetition_penalty", 1.0)),
        context_window_chars=(int(cfg["context_window_chars"]) if cfg.get("context_window_chars") else None),
        rescue_substrings=bool(cfg.get("rescue_substrings", False)),
        attribute_values=_attribute_values_from_config(cfg.get("attribute_values") or {}),
    )


def _attribute_values_from_config(
    raw: Any,
) -> Dict[str, Dict[str, List[str]]]:
    """Turn the `attribute_values` block of a config group into plain lists.

    The config shape is:

        attribute_values:
          visual_evidence:
            image: [A, B, both, unclear]
            valence: [good, bad, neutral]

    Every value becomes a string, because the model writes a string and the
    enum must compare equal to it.
    """
    raw = _to_plain(raw) or {}
    out: Dict[str, Dict[str, List[str]]] = {}
    for klass, attrs in raw.items():
        attrs = _to_plain(attrs) or {}
        out[str(klass)] = {
            str(attribute): [str(v) for v in (_to_plain(values) or [])]
            for attribute, values in attrs.items()
        }
    return out


def validate_examples(spec: ExtractionSpec) -> List[str]:
    """Test that every example span appears in its own example text.

    This catches the most common authoring bug: a paraphrase. The aligner works
    on the source text, thus a span that the example text does not hold word for
    word can never align, and the model learns a shape it cannot reproduce.

    Warning: LangExtract runs this same test inside `lx.extract`, but its
    version raises `ValueError` on an example with no extraction. A NEGATIVE
    example has no extraction on purpose. Thus we run the test here over the
    examples that have a span, and `extract_documents` turns the built-in test
    off.

    Returns:
        A list of messages, 1 for each problem. An empty list means every span
        aligns exactly.
    """
    from langextract import prompt_validation as pv

    scored = [ex for ex in spec.examples if ex.extractions]
    if not scored:
        return ["no example holds an extraction; the model has nothing to copy"]
    report = pv.validate_prompt_alignment(scored)
    # A nested span aligns by the fuzzy pass: the exact pass moves a cursor
    # through the text, and a span inside an earlier span sits behind it. The
    # slice the aligner names still equals the span word for word, thus the
    # model can copy it. Only a slice that differs is a problem. Measured
    # 2026-09-28 on the version 3 examples: 6 nested spans, all verbatim.
    issues = []
    for issue in report.issues:
        if _fuzzy_but_verbatim(issue, scored):
            continue
        issues.append(issue.short_msg())
    issues += _example_values_outside_enum(spec)
    return issues


def _fuzzy_but_verbatim(issue: Any, examples: Sequence[Any]) -> bool:
    """True when the issue is a fuzzy match whose aligned slice equals the
    span. `langextract.prompt_validation.ValidationIssue` names the example
    by `example_index`, the slice by `char_interval` (a tuple or a
    CharInterval), and the span by a preview that may be cut."""
    status = getattr(issue, "alignment_status", None)
    name = getattr(status, "name", str(status))
    if not name.endswith("MATCH_FUZZY"):
        return False
    idx = getattr(issue, "example_index", None)
    span = getattr(issue, "char_interval", None)
    if idx is None or span is None or idx >= len(examples):
        return False
    start = getattr(span, "start_pos", None)
    end = getattr(span, "end_pos", None)
    if start is None:
        start, end = span[0], span[1]
    example = examples[idx]
    text = example.text
    piece = text[start:end]
    # The slice must be one of the example's own spans of that class, word
    # for word. The preview alone may be cut, thus it is not compared.
    return any(
        e.extraction_class == getattr(issue, "extraction_class", None) and e.extraction_text == piece
        for e in example.extractions
    )


def _example_values_outside_enum(spec: ExtractionSpec) -> List[str]:
    """Name each example attribute value that the declared enum does not hold.

    Under guided decoding the grammar cannot write such a value. An example
    that holds one teaches a shape the model cannot copy, in the same way as a
    paraphrase does. The test also catches a declared attribute that an example
    leaves out, because the schema makes it required.
    """
    if not spec.attribute_values:
        return []
    out: List[str] = []
    for index, example in enumerate(spec.examples):
        for extraction in example.extractions:
            declared = spec.attribute_values.get(extraction.extraction_class)
            if not declared:
                continue
            attributes = extraction.attributes or {}
            for attribute, values in declared.items():
                if attribute not in attributes:
                    out.append(
                        f"example {index}: {extraction.extraction_class} "
                        f"{extraction.extraction_text!r} lacks the required "
                        f"attribute {attribute!r}"
                    )
                    continue
                value = attributes[attribute]
                if str(value) not in [str(v) for v in values]:
                    out.append(
                        f"example {index}: {extraction.extraction_class}."
                        f"{attribute}={value!r} is not in {list(values)}"
                    )
    return out


def _attr_value(value: Any) -> Any:
    """Keep an attribute value in a shape that LangExtract accepts."""
    value = _to_plain(value)
    if isinstance(value, list):
        return [str(v) for v in value]
    return str(value)


def _to_plain(value: Any) -> Any:
    """Turn an OmegaConf node into a plain dict, list, or scalar."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


# ---------------------------------------------------------------------------
# The language model adapter
# ---------------------------------------------------------------------------


class VLLMLanguageModel(lx_base_model.BaseLanguageModel):
    """Run a LangExtract batch on a local engine.

    The adapter holds no engine of its own. It takes `generate_fn`, which maps a
    sequence of prompts to a sequence of answers, 1 answer for each prompt, in
    the same order. `VLLMEngine.generate` is the production function, and a test
    passes a stub.
    """

    def __init__(
        self,
        generate_fn: Callable[[Sequence[str]], Sequence[str]],
        *,
        stage_name: str = "langextract",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._generate_fn = generate_fn
        self._stage_name = stage_name
        self._extra_kwargs = kwargs
        self.calls = 0
        self.prompts_seen = 0

    def infer(
        self, batch_prompts: Sequence[str], **kwargs: Any
    ) -> Iterator[Sequence[lx_types.ScoredOutput]]:
        """Answer every prompt of the batch in 1 engine call.

        Warning: `kwargs` can hold `max_workers`. Ignore it. The engine
        schedules the batch, and a thread pool around a local engine only adds
        contention.
        """
        prompts = list(batch_prompts)
        if not prompts:
            return
        t0 = time.time()
        outputs = list(self._generate_fn(prompts))
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"[{self._stage_name}] engine returned {len(outputs)} answers "
                f"for {len(prompts)} prompts"
            )
        self.calls += 1
        self.prompts_seen += len(prompts)
        print(
            f"[{self._stage_name}] batch {self.calls}: {len(prompts)} prompts "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )
        for text in outputs:
            yield [lx_types.ScoredOutput(score=1.0, output=text or "")]


# ---------------------------------------------------------------------------
# The extraction call
# ---------------------------------------------------------------------------


def extract_documents(
    texts: Sequence[str],
    doc_ids: Sequence[str],
    spec: ExtractionSpec,
    model: lx_base_model.BaseLanguageModel,
    *,
    contexts: Optional[Sequence[Optional[str]]] = None,
    show_progress: bool = False,
) -> List[lx_data.AnnotatedDocument]:
    """Extract from a set of traces. Give back 1 document for each input.

    `contexts` gives each document 1 line of LangExtract's
    `additional_context`, which the prompt puts after the description and
    before the examples.

    Warning: `lx.extract` drops nothing, but it can return the documents in a
    different order. Read `document_id`, never the position.
    """
    if len(texts) != len(doc_ids):
        raise ValueError("texts and doc_ids must have the same length")
    import langextract as lx
    from langextract import prompt_validation as pv

    if contexts is not None and len(contexts) != len(texts):
        raise ValueError("contexts and texts must have the same length")
    ctx = list(contexts) if contexts is not None else [None] * len(texts)
    documents = [
        lx_data.Document(text=text or "", document_id=str(doc_id),
                         additional_context=(str(c) if c else None))
        for text, doc_id, c in zip(texts, doc_ids, ctx)
    ]
    result = lx.extract(
        text_or_documents=documents,
        prompt_description=spec.prompt_description,
        examples=list(spec.examples),
        model=model,
        max_char_buffer=spec.max_char_buffer,
        batch_length=spec.batch_length,
        extraction_passes=spec.extraction_passes,
        context_window_chars=spec.context_window_chars,
        # The engine applies the JSON constraint, thus LangExtract must not try
        # to apply a provider schema of its own.
        use_schema_constraints=False,
        fence_output=False,
        # 1 local engine. A thread pool around it only adds contention.
        max_workers=1,
        # `validate_examples` does this test, and it accepts a negative example.
        # The built-in test raises on an example with no extraction.
        prompt_validation_level=pv.PromptValidationLevel.OFF,
        show_progress=show_progress,
    )
    return list(result) if isinstance(result, list) else [result]


def rescue_offsets(
    text: str, span: Any, prefer_from: Optional[int] = None, hint: Optional[int] = None
) -> Optional[tuple]:
    """Find a span in its text again. Give (start, end, mode) or None.

    The exact phase of LangExtract's aligner (1.6.0) runs 1 difflib match over
    all spans of a document. A block that starts inside a span is skipped, and
    the span then gets a 1-token interval at a wrong place
    (`match_lesser`). The span is often still verbatim in the text.

    The search takes, in this order: an exact occurrence at or after
    `prefer_from` (the start of the span before it), the exact occurrence
    nearest to `hint` (the aligner's start), then the same 2 steps without
    case, then the same 2 steps without case, Markdown marks, and extra
    white space (`_loose`). The extractor drops the bold marks inside a span
    ("reads "FLAT"" for "reads **"FLAT"**"). `mode` is "exact", "casefold",
    or "markup"; a "markup" span holds the marks of the text.
    """
    needle = str(span or "").strip()
    if not text or len(needle) < 2:
        return None
    loose_text, where = _loose(text)
    loose_needle, _ = _loose(needle)
    for mode, hay, ndl in (("exact", text, needle), ("casefold", text.lower(), needle.lower()),
                           ("markup", loose_text, loose_needle)):
        if len(ndl) < 2:
            continue
        hits = []
        i = hay.find(ndl)
        while i >= 0:
            hits.append(i)
            i = hay.find(ndl, i + 1)
        if not hits:
            continue
        after = [h for h in hits if prefer_from is not None and h >= prefer_from]
        if after:
            pick = after[0]
        elif hint is not None:
            pick = min(hits, key=lambda h: abs(h - hint))
        else:
            pick = hits[0]
        if mode == "markup":
            return where[pick], where[pick + len(ndl) - 1] + 1, mode
        return pick, pick + len(ndl), mode
    return None


_MARKUP = set("*_`#")


def _loose(text: str) -> tuple:
    """Give `text` in lower case with no Markdown marks and 1 space for each run of white space.

    The second value maps each character of the result to its index in `text`.
    """
    out, where = [], []
    space = False
    for i, ch in enumerate(text):
        if ch in _MARKUP:
            continue
        if ch.isspace():
            if space or not out:
                continue
            ch, space = " ", True
        else:
            space = False
        out.append(ch.lower())
        where.append(i)
    return "".join(out), where


def _is_quotable(status: Any) -> bool:
    """Say whether the span of an extraction can be quoted.

    Measured on 2026-08-13 over 200 subway traces:

    | Status | Share | What it means |
    |--------|-------|---------------|
    | `match_exact` | 64% | The span is the source text |
    | `match_fuzzy` | 28% | The span is still the source text; the aligner took the tolerant path because the words repeat |
    | `match_lesser` | 8% | The model COMPOSED a sentence, and the aligner matched only its first word |
    | none | 0.7% | Nothing aligned |

    A `match_lesser` row is the trap. "Same is often the best fit for very
    similar urban environments" aligns to the 4 characters of "Same". The row
    is a real statement of the model, but its offsets point at a fragment, thus
    no reader may quote it and no count of grounded evidence may hold it.
    """
    from langextract.core import data as core_data

    return status in (
        core_data.AlignmentStatus.MATCH_EXACT,
        core_data.AlignmentStatus.MATCH_FUZZY,
    )


def annotated_to_rows(
    annotated: lx_data.AnnotatedDocument,
    spec: ExtractionSpec,
    extractor_model: str,
    base: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Turn 1 annotated document into the long output rows.

    1 row holds 1 extraction. A document with no extraction still writes 1 row,
    with a null class. Without that row a reader cannot separate "the model said
    nothing" from "the stage did not run".
    """
    base = dict(base or {})
    base.update(
        {
            "doc_id": annotated.document_id,
            "extractor_model": extractor_model,
            "schema_name": spec.name,
            "schema_version": spec.schema_version,
        }
    )

    extractions = annotated.extractions or []
    if not extractions:
        empty = dict(base)
        empty.update(
            {
                "extraction_index": None,
                "extraction_class": None,
                "extraction_text": None,
                "attributes_json": None,
                "char_start": None,
                "char_end": None,
                "alignment_status": None,
                "is_quotable": False,
                "is_verbatim": False,
                "alignment_rescue": None,
                "is_duplicate": False,
            }
        )
        return [empty]

    source_text = annotated.text or ""
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    last_start: Optional[int] = None
    for index, extraction in enumerate(extractions):
        interval = extraction.char_interval
        status = extraction.alignment_status
        rescue = None
        verbatim = is_verbatim(source_text, interval, extraction.extraction_text)
        if spec.rescue_substrings and not verbatim:
            hint = interval.start_pos if interval is not None else None
            rescue = rescue_offsets(source_text, extraction.extraction_text, last_start, hint)
        # The prompt says "do not extract the same span two times". A model
        # that reaches its token cap often does so inside a loop: the qwen
        # extractor wrote "Let's look at the hot again" 388 times for 1 trace
        # on 2026-09-20. The first copy stays; every later copy of the same
        # class and text is a duplicate, and a count must drop it.
        key = (extraction.extraction_class, str(extraction.extraction_text or "").strip())
        duplicate = key in seen
        seen.add(key)
        if rescue is not None:
            start, end = rescue[0], rescue[1]
        else:
            start = interval.start_pos if interval else None
            end = interval.end_pos if interval else None
        if start is not None:
            last_start = int(start)
        row = dict(base)
        row.update(
            {
                "extraction_index": (
                    extraction.extraction_index
                    if extraction.extraction_index is not None
                    else index
                ),
                "extraction_class": extraction.extraction_class,
                "extraction_text": extraction.extraction_text,
                "attributes_json": (
                    json.dumps(extraction.attributes, sort_keys=True)
                    if extraction.attributes
                    else None
                ),
                "char_start": start,
                "char_end": end,
                # An unaligned extraction is a defect, not data. Keep the row so
                # the rate stays visible, and drop it from a count.
                "alignment_status": status.value if status is not None else None,
                # A rescued span is in the text, thus a count may hold it.
                "is_quotable": _is_quotable(status) or rescue is not None,
                "is_verbatim": verbatim or (rescue is not None and rescue[2] == "exact"),
                # "exact", "casefold" or "markup" when `rescue_offsets` set the offsets.
                "alignment_rescue": rescue[2] if rescue is not None else None,
                "is_duplicate": duplicate,
            }
        )
        rows.append(row)
    return rows


def is_verbatim(source_text: str, interval: Any, extraction_text: Any) -> bool:
    """Say whether the source text at the offsets IS the extraction text.

    `is_quotable` reads the alignment status. That is not enough for a quote.
    The fuzzy aligner accepts a span when 75% of its tokens match, thus a
    `match_fuzzy` row can differ from the source: measured on 2026-09-20, 12%
    of the fuzzy rows of the gemma extractor and 20% of the qwen extractor did,
    and some changed the meaning ("the construction in A" for "the cars in A").

    A quote in the paper needs this column, not `is_quotable`. A count may
    still use `is_quotable`, because a near-verbatim span is still a statement
    the model made.

    The comparison strips the white space at the 2 ends and nothing else.
    """
    if interval is None or extraction_text is None:
        return False
    start = getattr(interval, "start_pos", None)
    end = getattr(interval, "end_pos", None)
    if start is None or end is None:
        return False
    try:
        return source_text[int(start):int(end)].strip() == str(extraction_text).strip()
    except (TypeError, ValueError):
        return False
