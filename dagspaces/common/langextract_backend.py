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

See `vlm-narratives-docs/langextract-trace-extraction.md`.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from langextract import data as lx_data
from langextract.core import base_model as lx_base_model
from langextract.core import types as lx_types

__version__ = "1.0.0"

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
        return schema


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
    )


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
    return [issue.short_msg() for issue in report.issues]


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
    show_progress: bool = False,
) -> List[lx_data.AnnotatedDocument]:
    """Extract from a set of traces. Give back 1 document for each input.

    Warning: `lx.extract` drops nothing, but it can return the documents in a
    different order. Read `document_id`, never the position.
    """
    if len(texts) != len(doc_ids):
        raise ValueError("texts and doc_ids must have the same length")
    import langextract as lx
    from langextract import prompt_validation as pv

    documents = [
        lx_data.Document(text=text or "", document_id=str(doc_id))
        for text, doc_id in zip(texts, doc_ids)
    ]
    result = lx.extract(
        text_or_documents=documents,
        prompt_description=spec.prompt_description,
        examples=list(spec.examples),
        model=model,
        max_char_buffer=spec.max_char_buffer,
        batch_length=spec.batch_length,
        extraction_passes=spec.extraction_passes,
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
            }
        )
        return [empty]

    rows: List[Dict[str, Any]] = []
    for index, extraction in enumerate(extractions):
        interval = extraction.char_interval
        status = extraction.alignment_status
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
                "char_start": interval.start_pos if interval else None,
                "char_end": interval.end_pos if interval else None,
                # An unaligned extraction is a defect, not data. Keep the row so
                # the rate stays visible, and drop it from a count.
                "alignment_status": status.value if status is not None else None,
                "is_quotable": _is_quotable(status),
            }
        )
        rows.append(row)
    return rows
