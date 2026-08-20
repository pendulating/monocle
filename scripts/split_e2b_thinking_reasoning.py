#!/usr/bin/env python
"""Post-hoc reasoning/answer split for Gemma-4-E2B thinking outputs.

Stock Gemma-4-E2B emits its thought as bare ``thought\\n...`` text with no
``<thought>``/``</thought>`` delimiters, so vLLM's ``gemma4`` reasoning parser
cannot separate it: the whole trace lands in ``model_response`` (and the raw
``answer`` column) while ``model_reasoning`` stays empty. The trailing
``{"answer": ...}`` is still emitted verbatim, so the split is deterministic:
everything before the final JSON object is the reasoning trace, the JSON is the
answer. Labels (``relative_label`` etc.) are already correct — this only
repopulates ``model_reasoning`` / cleans ``model_response`` / ``answer`` so the
E2B parquets match the qwen3.5-9b / gemma-4-12b thinking schema.

100% reliable on the 16-pair smoke (all rows start with 'thought', end with a
clean answer-JSON, split cleanly). Idempotent: rows already carrying a
non-empty ``model_reasoning`` are left untouched.

Usage:
    python scripts/split_e2b_thinking_reasoning.py <parquet> [<parquet> ...]
    python scripts/split_e2b_thinking_reasoning.py --dry-run <parquet>

Writes in place after saving a ``<name>.presplit.parquet`` backup (unless
--no-backup). Pass --dry-run to only report what would change.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import pandas as pd

_BRACE = re.compile(r"\{")


def _split(text: str):
    """Return (reasoning, answer_json_str, answer_value) or (None, None, None)."""
    if not isinstance(text, str) or not text.strip():
        return None, None, None
    # Find the trailing JSON object: try each '{' from last to first.
    for m in reversed(list(_BRACE.finditer(text))):
        i = m.start()
        tail = text[i:].strip()
        try:
            obj = json.loads(tail)
        except Exception:
            continue
        if isinstance(obj, dict) and "answer" in obj:
            reasoning = text[:i]
            # strip the leading bare 'thought' marker E2B emits
            reasoning = re.sub(r"^\s*thought\s*\n?", "", reasoning, count=1).strip()
            return reasoning, json.dumps(obj), obj.get("answer")
    return None, None, None


def process(path: str, dry_run: bool = False, backup: bool = True) -> dict:
    df = pd.read_parquet(path)
    if "model_response" not in df.columns:
        return {"path": path, "skipped": "no model_response column"}
    reason = df.get("model_reasoning", pd.Series([""] * len(df))).fillna("").astype(str)
    already = (reason.str.strip() != "")
    n_split = 0
    new_reasoning = df.get("model_reasoning", pd.Series([""] * len(df))).astype("object").copy()
    new_response = df["model_response"].astype("object").copy()
    new_answer = df.get("answer", pd.Series([""] * len(df))).astype("object").copy()
    for idx in df.index[~already]:
        r, j, a = _split(str(df.at[idx, "model_response"]))
        if r is None:
            continue
        new_reasoning.at[idx] = r
        new_response.at[idx] = j
        if "answer" in df.columns and a is not None:
            new_answer.at[idx] = a
        n_split += 1
    stats = {
        "path": path,
        "rows": len(df),
        "already_had_reasoning": int(already.sum()),
        "split": n_split,
        "unsplittable": int((~already).sum() - n_split),
    }
    if not dry_run and n_split:
        if backup:
            df.to_parquet(path.replace(".parquet", ".presplit.parquet"), index=False)
        df["model_reasoning"] = new_reasoning
        df["model_response"] = new_response
        if "answer" in df.columns:
            df["answer"] = new_answer
        df.to_parquet(path, index=False)
        stats["written"] = True
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args(argv)
    for p in args.parquets:
        s = process(p, dry_run=args.dry_run, backup=not args.no_backup)
        print(json.dumps(s))


if __name__ == "__main__":
    main()
