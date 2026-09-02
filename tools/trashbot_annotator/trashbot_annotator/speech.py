"""Classify transcript utterances: speech to the robot and about it.

The corpus rows come from the robot_norms transcript corpus. Each row
holds one timestamped utterance chunk of one clip. The classifier sends
each utterance with its neighbor context to the served LLM. The output
marks the address of the speech:

* to_robot — speech directed at the robot.
* about_robot — overheard talk about the robot ("eavesdropping").
* unrelated — speech that does not concern the robot.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .runner import DEFAULT_MODEL, normalize_server_url
from .taxonomy import SPEECH_KINDS, is_valid_address

# The default utterance corpus of the robot_norms project.
DEFAULT_CORPUS: str = "/share/ju/robot_norms/embeddings/chunk_corpus.parquet"

# Number of neighbor utterances on each side in the context block.
CONTEXT_RADIUS: int = 2

# Sampling settings for the classification calls.
MAX_TOKENS: int = 256
TEMPERATURE: float = 0.0
SEED: int = 42

# Column order of the speech annotations parquet.
SPEECH_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "dataset",
    "place",
    "capture_type",
    "sample_id",
    "chunk_idx",
    "start_s",
    "end_s",
    "text",
    "address",
    "address_raw",
    "kind",
    "kind_raw",
    "operator",
    "evidence",
    "eavesdrop",
    "model",
    "error",
    "created_at",
)


def default_speech_prompt() -> Path:
    """Return the default speech prompt template path."""
    tool_root = Path(__file__).resolve().parents[1]
    return tool_root / "config" / "prompt_speech.txt"


def load_utterance_corpus(path: str | Path) -> pd.DataFrame:
    """Load an utterance corpus parquet. Normalize the text column.

    The chunk corpus names the column "text". The caption corpus names
    it "transcript". Both load. Rows without text drop out.
    """
    df = pd.read_parquet(path)
    if "text" not in df.columns and "transcript" in df.columns:
        df = df.rename(columns={"transcript": "text"})
    required = {"doc_id", "sample_id", "start_s", "end_s", "text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"corpus misses columns: {sorted(missing)}")
    df = df[df["text"].astype(str).str.strip() != ""].copy()
    if "chunk_idx" not in df.columns:
        df["chunk_idx"] = df.groupby("sample_id").cumcount()
    return df.sort_values(["sample_id", "chunk_idx"], kind="stable").reset_index(drop=True)


def build_context(texts: list[str], position: int, radius: int = CONTEXT_RADIUS) -> str:
    """Return the neighbor utterances around one position as a block.

    The block marks the target line with ">>". Neighbors come from the
    same clip in time order.
    """
    lines: list[str] = []
    start = max(0, position - radius)
    end = min(len(texts), position + radius + 1)
    for i in range(start, end):
        marker = ">>" if i == position else "  "
        lines.append(f"{marker} {texts[i]}")
    return "\n".join(lines)


def render_speech_prompt(template: str, text: str, context: str) -> str:
    """Fill the speech prompt template with the utterance and context."""
    return template.replace("{TEXT}", text).replace("{CONTEXT}", context)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object inside a response text."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clean_labels(
    parsed: dict[str, Any] | None, warnings: Counter[str]
) -> dict[str, Any]:
    """Validate the model labels. Invalid values land in raw columns."""
    out: dict[str, Any] = {
        "address": None,
        "address_raw": None,
        "kind": None,
        "kind_raw": None,
        "operator": None,
        "evidence": None,
    }
    if parsed is None:
        warnings["unparsable_responses"] += 1
        return out
    address = parsed.get("address")
    if is_valid_address(address):
        out["address"] = address
    else:
        out["address_raw"] = address
        warnings["invalid_address"] += 1
    kind = parsed.get("kind")
    if isinstance(kind, str) and kind in SPEECH_KINDS:
        out["kind"] = kind
    else:
        out["kind_raw"] = kind
        warnings["invalid_kind"] += 1
    operator = parsed.get("operator")
    out["operator"] = operator if isinstance(operator, bool) else None
    evidence = parsed.get("evidence")
    out["evidence"] = evidence if isinstance(evidence, str) else None
    return out


def classify_corpus(
    corpus: pd.DataFrame,
    server_url: str,
    out_jsonl: str | Path,
    model: str = DEFAULT_MODEL,
    prompt_template: str | Path | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> Counter[str]:
    """Classify each utterance against the served LLM. Stream JSONL.

    The output file gets one line per utterance. With resume on, doc
    ids that sit in the file already skip the call. Returns a counter
    with call and warning counts.
    """
    from openai import OpenAI

    template_path = Path(prompt_template) if prompt_template else default_speech_prompt()
    template = template_path.read_text(encoding="utf-8")
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if resume and out_path.is_file():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("doc_id") is not None:
                    done.add(str(record["doc_id"]))

    client = OpenAI(base_url=normalize_server_url(server_url), api_key="EMPTY")
    warnings: Counter[str] = Counter()
    created_at = datetime.now(timezone.utc).isoformat()

    texts_by_sample: dict[str, list[str]] = {
        str(sid): group["text"].astype(str).tolist()
        for sid, group in corpus.groupby("sample_id", sort=False)
    }
    positions_by_sample: Counter[str] = Counter()

    n_calls = 0
    with out_path.open("a", encoding="utf-8") as sink:
        for row in corpus.to_dict("records"):
            sample_id = str(row["sample_id"])
            position = positions_by_sample[sample_id]
            positions_by_sample[sample_id] += 1

            doc_id = str(row["doc_id"])
            if doc_id in done:
                warnings["skipped_resume"] += 1
                continue
            if limit is not None and n_calls >= limit:
                break

            text = str(row["text"]).strip()
            context = build_context(texts_by_sample[sample_id], position)
            prompt = render_speech_prompt(template, text, context)

            error: str | None = None
            response_text = ""
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    seed=SEED,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - record and continue
                error = f"request_error: {exc}"
                warnings["request_errors"] += 1
            n_calls += 1

            labels = _clean_labels(
                extract_json_object(response_text) if error is None else None,
                warnings,
            )
            record = {
                "doc_id": doc_id,
                "dataset": row.get("dataset"),
                "place": row.get("place"),
                "capture_type": row.get("capture_type"),
                "sample_id": sample_id,
                "chunk_idx": row.get("chunk_idx"),
                "start_s": row.get("start_s"),
                "end_s": row.get("end_s"),
                "text": text,
                **labels,
                "model": model,
                "error": error,
                "created_at": created_at,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()

    warnings["calls"] = n_calls
    return warnings


def consolidate_speech(jsonl_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Convert the speech JSONL stream into one parquet table.

    The eavesdrop flag marks bystander talk about the robot: address
    "about_robot" and operator false or unknown.
    """
    records: list[dict[str, Any]] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)

    df = pd.DataFrame(records)
    df["eavesdrop"] = (df.get("address") == "about_robot") & (
        df.get("operator") != True  # noqa: E712 - None must count as False here
    )
    for column in SPEECH_COLUMNS:
        if column not in df.columns:
            df[column] = None
    df = df[list(SPEECH_COLUMNS)]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df
