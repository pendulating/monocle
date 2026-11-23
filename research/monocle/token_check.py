"""Which surface form of a concept is ONE token for Gemma-4-12B?

The single-image read and the lens steering demand a single-token polar word
(`single_steer.polar_token_ids`): the answer position holds one next-token
distribution, thus a multi-token word shows its first piece alone. Before a
new concept enters a case, this script tells which form of it survives that
rule, under the same call the scorer makes: `tokenizer.encode(word,
add_special_tokens=False)` on the bare word. The space-prefixed form counts
for the contract metric (`polar_variant_ids`), thus it prints too.

    .venv-vllm025cu129/bin/python research/monocle/token_check.py             # the catalogue
    .venv-vllm025cu129/bin/python research/monocle/token_check.py homeless gang  # your words

Each word prints in 4 forms: bare, Capitalized, space-prefixed, and
space-prefixed Capitalized. `ok` marks a form that is exactly 1 token.
"""
from __future__ import annotations

import argparse
import sys

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402

#: The concepts to frame, grouped. Each group holds the forms worth a try.
CATALOGUE: dict[str, list[str]] = {
    "housing": ["homeless", "homelessness", "unhoused", "vagrant", "vagrants",
                "housed", "shelter", "encampment", "tent", "tents"],
    "gang": ["gang", "gangs", "gangster", "criminal", "criminals", "crime",
             "thug", "thugs", "violent", "violence", "drug", "drugs", "dealer"],
    "race": ["black", "white", "asian", "hispanic", "latino", "latina",
             "african", "arab", "jewish", "muslim", "chinese", "mexican"],
    "migration": ["immigrant", "immigrants", "migrant", "migrants", "foreign",
                  "foreigner", "foreigners", "native", "citizen", "illegal",
                  "undocumented", "refugee", "refugees"],
    "class": ["poor", "rich", "wealthy", "poverty", "affluent", "upscale",
              "ghetto", "slum", "projects", "luxury", "working", "elite",
              "gentrified", "gentrification"],
    "disorder": ["dirty", "clean", "trash", "graffiti", "litter", "vandalism",
                 "decay", "blight", "neglected", "abandoned", "sketchy",
                 "dangerous", "safe", "unsafe"],
    "current poles": ["safe", "dangerous", "maintained", "worn", "good", "bad",
                      "smooth", "damaged", "delicious", "gross", "beautiful",
                      "ugly"],
}


def forms(word: str) -> list[tuple[str, str]]:
    return [("bare", word), ("Cap", word.capitalize()),
            ("space", " " + word), ("space Cap", " " + word.capitalize())]


def check(tokenizer, word: str) -> list[dict]:
    rows = []
    for name, form in forms(word):
        ids = tokenizer.encode(form, add_special_tokens=False)
        pieces = tokenizer.convert_ids_to_tokens(ids)
        rows.append({"word": word, "form": name, "n": len(ids), "ids": ids,
                     "pieces": pieces, "ok": len(ids) == 1})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*", help="words to check; none = the catalogue")
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    groups = {"words": args.words} if args.words else CATALOGUE
    for group, words in groups.items():
        print(f"\n== {group}")
        print(f"{'word':<16}{'bare':<26}{'Cap':<26}{'space':<26}{'space Cap':<26}")
        for w in words:
            cells = []
            for r in check(tok, w):
                mark = "ok " if r["ok"] else f"{r['n']}x "
                cells.append(f"{mark}{'|'.join(r['pieces'])}"[:25])
            print(f"{w:<16}" + "".join(f"{c:<26}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
