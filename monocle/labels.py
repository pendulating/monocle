"""The label token contract for a single-position lens read.

The lens reads ONE sequence position: the position that emits the label. Thus
a label must be 1 token, and 2 labels must not share their first token. A
label that breaks the rule fails SILENTLY — the read returns the logit of a
shared prefix, which carries no judgment.

Measured on the gemma-4-12B tokenizer:

| Label | Tokens | Usable |
|---|---|---|
| `Safe` | `Safe` | yes |
| `Neutral` | `Neutral` | yes |
| `Dangerous` | `Dangerous` | yes |
| `Unsafe` | `Un` + `safe` | NO |
| `Very unsafe` | `Very` + `_unsafe` | NO |
| `Very safe` | `Very` + `_safe` | NO, and it collides with `Very unsafe` |

The pairwise battery meets the rule by accident: `MuchLess` and `MuchMore`
both start with `Much`, so the contrast uses `More` and `Less` instead.

Do not use DIGITS as labels on this checkpoint. Measured over 40 pairs of
subway_safety with the dense lens, the share of image patches whose top-1
readout is a bare digit is 0.0% at L30-L38 and then:

| L40 | L41 | L42 | L43 | L46 |
|---|---|---|---|---|
| 8.3% | 11.3% | 10.6% | 6.7% | 17.1% |

That is the attribution window. A digit contrast would share a direction with
about 9% of the patches it tries to score. `Safe`, `Neutral` and `Dangerous`
read 0.00% over the same patches.
"""
from __future__ import annotations

from typing import Any


def label_token_ids(tokenizer: Any, labels: list[str]) -> dict[str, list[int]]:
    """Encode each label the way the model meets it.

    The forced prefix ends with a quote, so the label starts a string and
    carries NO leading space. Encode it that way, or the check tests a token
    the model never emits.
    """
    return {
        lab: list(tokenizer.encode(lab, add_special_tokens=False))
        for lab in labels
    }


def check_label_tokens(tokenizer: Any, labels: list[str]) -> dict[str, int]:
    """Raise unless every label is 1 token and all of them differ.

    Returns the label to token-id map when the contract holds.
    """
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate labels: {labels}")
    ids = label_token_ids(tokenizer, labels)
    multi = {k: v for k, v in ids.items() if len(v) != 1}
    if multi:
        raise ValueError(
            "these labels are not 1 token, so a single-position lens cannot "
            f"read them: { {k: len(v) for k, v in multi.items()} }. "
            "Pick another word.")
    first = {k: v[0] for k, v in ids.items()}
    seen: dict[int, str] = {}
    for lab, tid in first.items():
        if tid in seen:
            raise ValueError(
                f"{lab!r} and {seen[tid]!r} share token id {tid}. The lens "
                "cannot tell them apart.")
        seen[tid] = lab
    return first


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--model-dir",
                    default="/share/ju/matt/zoo/models/gemma-4-12B-it")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model_dir)
    ids = label_token_ids(tk, a.labels)
    for lab, t in ids.items():
        mark = "ok " if len(t) == 1 else "BAD"
        print(f"  {mark} {lab:<16} {t} {tk.convert_ids_to_tokens(t)}")
    check_label_tokens(tk, a.labels)
    print("the label contract holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
