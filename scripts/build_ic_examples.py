#!/usr/bin/env python
"""Build the few-shot examples of the Integrative Complexity schema.

The trace text comes from the parquet, character for character. Only the
annotation is written by hand. Thus a quote can never drift from its source.

Warning: do NOT put an example in a YAML file. A YAML block scalar folds a long
line and strips trailing space, and `locate_quote` then fails on a span that
looks correct in the editor. JSON keeps every character.

Run:
    python scripts/build_ic_examples.py            # write and verify
    python scripts/build_ic_examples.py --check    # verify only

See `vlm-narratives-docs/ic-ingredient-extraction.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dagspaces.common import ic_schema as S  # noqa: E402

SOURCE_PARQUET = (
    "/share/pierson/matt/mllmsci/multirun/2026-08-13_URBANPAIRVQA/01-37-12/0/"
    "outputs/pairwise/subway_safety_mvp_20260813_013722.parquet"
)

OUT_PATH = REPO / "dagspaces" / "common" / f"ic_examples_{S.SCHEMA_VERSION}.json"


# ---------------------------------------------------------------------------
# The annotations
# ---------------------------------------------------------------------------
#
# The 3 traces cover the complexity range on purpose.
#
# | pair_id        | What it teaches                                          |
# |----------------|----------------------------------------------------------|
# | unit_00004632  | 1 dominant cue, a dismissal, and a weighing with NO reason|
# | unit_00002504  | 1 perspective that the trace justifies; prioritization    |
# | unit_00001205  | 3 perspectives, a conditional, a verdict that changes     |

ANNOTATIONS = {
    # -----------------------------------------------------------------------
    # Low complexity. The trace reconsiders 4 times and never moves.
    # -----------------------------------------------------------------------
    "unit_00004632": {
        "dimensions": [
            {"name": "construction", "type": "descriptive", "image": "A", "valence": "bad",
             "quote": "subway entrance under construction or renovation"},
            {"name": "construction", "type": "descriptive", "image": "A", "valence": "bad",
             "quote": "orange construction barriers and fences"},
            {"name": "security_presence", "type": "descriptive", "image": "A", "valence": "good",
             "quote": "A mobile ticket booth/guard station is visible"},
            {"name": "signage", "type": "descriptive", "image": "A", "valence": "bad",
             "quote": "There is a \"DANGER\" sign"},
            {"name": "hazard", "type": "evaluative", "image": "A", "valence": "bad",
             "quote": "It looks less public and potentially more hazardous due to active construction"},
            {"name": "upkeep", "type": "evaluative", "image": "B", "valence": "good",
             "quote": "a typical, well-maintained subway entrance (MTA booth and stairs)"},
            {"name": "people", "type": "descriptive", "image": "B", "valence": "neutral",
             "quote": "It's on a busy street with pedestrians and cars"},
            {"name": "hazard", "type": "descriptive", "image": "B", "valence": "good",
             "quote": "No visible construction hazards"},
        ],
        "perspectives": [
            {"name": "construction_hazard", "favors": "B", "is_winner": True,
             "quote": "Construction sites are generally less safe for the general public due to tripping hazards, falling objects, etc.",
             "supporting_quotes": [
                 "Usually, construction zones are considered less safe for pedestrians",
                 "Construction zones are inherently less safe for the general public",
             ]},
        ],
        # The trace names the guard booth and then puts it aside.
        "dismissals": [
            {"quote": "Probably not, construction hazards usually outweigh a guard",
             "target": "guard booth"},
        ],
        "reconsiderations": [
            {"quote": "Let's re-evaluate"},
            {"quote": "Wait, let me check if there's any other factor"},
            {"quote": "Let me double check"},
            {"quote": "Is there anything else?"},
        ],
        "verdicts": [
            {"label": "Less", "is_final": False,
             "quote": "the station entrance in Image A is *less* safe than the one in Image B due to the construction activity"},
            {"label": "Less", "is_final": False,
             "quote": "So Image A is \"Less\" safe than Image B"},
            {"label": "Less", "is_final": True, "quote": "Image A is Less safe"},
        ],
        "hedges": [
            {"quote": "Construction debris or materials might be around",
             "marker_type": "qualifier", "justification_quotes": [],
             "affects_conclusion": False},
            {"quote": "Image A might be perceived as safer because of the guard booth",
             "marker_type": "qualifier",
             "justification_quotes": ["Probably not, construction hazards usually outweigh a guard"],
             "affects_conclusion": False},
        ],
        # THE teaching case. It says "outweigh" and explains nothing, thus the
        # justification list stays empty and no code may call it a real weighing.
        "weighings": [
            {"quote": "construction hazards usually outweigh a guard",
             "mechanism": "trade_off", "justification_quotes": [],
             "condition_quotes": []},
        ],
    },
    # -----------------------------------------------------------------------
    # Middle. 1 standard, stated and justified, that decides the answer.
    # -----------------------------------------------------------------------
    "unit_00002504": {
        "dimensions": [
            {"name": "infrastructure", "type": "descriptive", "image": "A", "valence": "neutral",
             "quote": "subway entrance located under an elevated train structure"},
            {"name": "light", "type": "evaluative", "image": "A", "valence": "bad",
             "quote": "It's dark and shaded"},
            {"name": "people", "type": "descriptive", "image": "A", "valence": "neutral",
             "quote": "There are cars and some people visible"},
            {"name": "light", "type": "descriptive", "image": "B", "valence": "good",
             "quote": "on a sunny day"},
            {"name": "infrastructure", "type": "descriptive", "image": "B", "valence": "neutral",
             "quote": "There's a fence/railing"},
            {"name": "light", "type": "evaluative", "image": "B", "valence": "good",
             "quote": "It looks like a standard, well-lit outdoor entrance"},
            {"name": "light", "type": "evaluative", "image": "B", "valence": "good",
             "quote": "well-lit, in the sun, and clearly marked"},
            {"name": "visibility", "type": "evaluative", "image": "A", "valence": "bad",
             "quote": "the entrance is actually quite far back or obscured by the structures"},
            {"name": "visibility", "type": "evaluative", "image": "B", "valence": "good",
             "quote": "the entrance is very clear and in an open area"},
        ],
        "perspectives": [
            {"name": "visibility", "favors": "B", "is_winner": True,
             "quote": "Generally, well-lit and clearly visible areas are considered safer than dark, shaded areas under heavy infrastructure",
             "supporting_quotes": [
                 "Usually, bright and open areas are safer",
                 "Most safety assessments for urban environments prioritize visibility",
             ]},
            {"name": "crime_risk", "favors": "B", "is_winner": False,
             "quote": "In many urban safety contexts, shadows and underpasses are higher-risk areas",
             "supporting_quotes": []},
        ],
        "dismissals": [
            {"quote": "Maybe more people around? It's hard to tell", "target": "people present"},
        ],
        "reconsiderations": [
            {"quote": "Let's re-examine:"},
            {"quote": "Wait, let's look closer at Image A"},
            {"quote": "Wait, \"MuchLess\"?"},
            {"quote": "Let's re-read the options."},
        ],
        "verdicts": [
            {"label": "Less", "is_final": False,
             "quote": "So Image A is \"Less\" safe than Image B"},
            {"label": "Less", "is_final": False, "quote": "I'll go with \"Less\""},
            {"label": "Less", "is_final": True, "quote": "I will choose \"Less\""},
        ],
        "hedges": [
            {"quote": "The entrance itself is not clearly visible",
             "marker_type": "explicit_doubt", "justification_quotes": [],
             "affects_conclusion": False},
            {"quote": "However, the prompt asks",
             "marker_type": "contrastive", "justification_quotes": [],
             "affects_conclusion": False},
            {"quote": "Maybe more people around? It's hard to tell",
             "marker_type": "explicit_doubt", "justification_quotes": [],
             "affects_conclusion": False},
        ],
        "weighings": [
            {"quote": "Most safety assessments for urban environments prioritize visibility",
             "mechanism": "prioritization",
             "justification_quotes": ["Image B has much better visibility"],
             "condition_quotes": []},
        ],
    },
    # -----------------------------------------------------------------------
    # Highest. 3 standards, a condition, and a verdict that moves.
    # `dismissals` stays EMPTY here, which shows that an empty list is correct.
    # -----------------------------------------------------------------------
    "unit_00001205": {
        "dimensions": [
            {"name": "barrier", "type": "descriptive", "image": "A", "valence": "neutral",
             "quote": "a high iron fence around it"},
            {"name": "people", "type": "descriptive", "image": "A", "valence": "good",
             "quote": "There are people walking nearby"},
            {"name": "light", "type": "evaluative", "image": "A", "valence": "good",
             "quote": "It looks like a relatively open, well-lit area during the day"},
            {"name": "infrastructure", "type": "descriptive", "image": "B", "valence": "neutral",
             "quote": "It's a smaller kiosk/entrance"},
            {"name": "traffic", "type": "descriptive", "image": "B", "valence": "bad",
             "quote": "There's a car parked right in front with its trunk open"},
            {"name": "visibility", "type": "evaluative", "image": "A", "valence": "good",
             "quote": "a lot of space and visibility"},
            {"name": "barrier", "type": "descriptive", "image": "B", "valence": "neutral",
             "quote": "Image B has no fence"},
            {"name": "people", "type": "evaluative", "image": "A", "valence": "good",
             "quote": "Image A has a lot of \"eyes on the street\""},
        ],
        "perspectives": [
            {"name": "controlled_access", "favors": "A", "is_winner": False,
             "quote": "a fenced-off area (A) might be considered \"safer\" from some perspectives (controlled access)",
             "supporting_quotes": [
                 "Image A has a prominent fence, which might be seen as a security feature",
                 "Image A's fence might be seen as a protective measure",
             ]},
            {"name": "visibility", "favors": "neither", "is_winner": False,
             "quote": "Usually, \"safer\" in these VQA tasks might refer to visibility or physical barriers",
             "supporting_quotes": []},
            {"name": "no_clear_hazard", "favors": "neither", "is_winner": True,
             "quote": "Often, \"Same\" is the default if there's no clear hazard",
             "supporting_quotes": [
                 "Let's look for any signs of danger. Nothing obvious in either",
                 "Neither has glaring safety issues",
             ]},
        ],
        "dismissals": [],
        "reconsiderations": [
            {"quote": "Actually, Image A's fence might be to keep people *out*"},
            {"quote": "Let's re-examine."},
            {"quote": "Actually, Image A has a lot of \"eyes on the street\""},
            {"quote": "Let's re-examine Image A"},
        ],
        "verdicts": [
            {"label": "More", "is_final": False,
             "quote": "Image A's fence might be seen as a protective measure, making it \"More\" safe in some contexts"},
            {"label": "Same", "is_final": False,
             "quote": "I'll go with \"Same\" as they are both standard city subway entrances in broad daylight"},
            {"label": "Same", "is_final": True, "quote": "I will choose \"Same\""},
        ],
        "hedges": [
            {"quote": "which could be a security concern, or just someone loading/unloading",
             "marker_type": "qualifier", "justification_quotes": [],
             "affects_conclusion": False},
            {"quote": "It's hard to say one is definitively \"safer\" without more context",
             "marker_type": "explicit_doubt",
             "justification_quotes": [
                 "They look like standard NYC subway entrances in different neighborhoods"
             ],
             "affects_conclusion": True},
            {"quote": "However, sometimes fences imply more security",
             "marker_type": "contrastive", "justification_quotes": [],
             "affects_conclusion": False},
        ],
        "weighings": [
            {"quote": "Or \"More\" if I prioritize the fence", "mechanism": "conditional",
             "justification_quotes": [], "condition_quotes": ["if I prioritize the fence"]},
        ],
    },
}

# The order the examples appear in the prompt. Low complexity comes first, so
# the model does not read a rich answer as the normal one.
ORDER = ["unit_00004632", "unit_00002504", "unit_00001205"]


def load_traces() -> dict:
    import pandas as pd

    df = pd.read_parquet(SOURCE_PARQUET, columns=["pair_id", "model_reasoning"])
    df = df[df.pair_id.isin(ANNOTATIONS)]
    return {str(r.pair_id): str(r.model_reasoning) for r in df.itertuples()}


def verify(examples: list) -> list:
    """Test that every span of every example is in its own trace."""
    issues = []
    for example in examples:
        trace = example["trace"]
        name = example["pair_id"]
        rows = S.ingredient_rows(trace, example["answer"])
        for row in rows:
            if row["ingredient_type"] is None:
                issues.append(f"{name}: the answer holds no ingredient")
                continue
            if not row["quote_found"]:
                issues.append(
                    f"{name}: {row['ingredient_type']}[{row['ingredient_index']}] "
                    f"quote not in the trace: {row['quote']!r}"
                )
            elif row["quote_method"] != "exact":
                issues.append(
                    f"{name}: {row['ingredient_type']}[{row['ingredient_index']}] "
                    f"matched only after we changed the spaces: {row['quote']!r}"
                )
            if row["n_sub_quotes"] != row["n_sub_quotes_found"]:
                attrs = json.loads(row["attrs_json"])
                for field in S.SUB_QUOTE_FIELDS:
                    for item in attrs.get(field, []):
                        if item["method"] == "none":
                            issues.append(
                                f"{name}: {row['ingredient_type']}"
                                f"[{row['ingredient_index']}].{field} "
                                f"not in the trace: {item['text']!r}"
                            )
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only, do not write")
    args = ap.parse_args()

    traces = load_traces()
    missing = [p for p in ORDER if p not in traces]
    if missing:
        print(f"[examples] the source parquet has no {missing}")
        return 1

    examples = [
        {"pair_id": pair_id, "trace": traces[pair_id], "answer": ANNOTATIONS[pair_id]}
        for pair_id in ORDER
    ]

    issues = verify(examples)
    for issue in issues:
        print(f"  [bad] {issue}")
    if issues:
        print(f"[examples] {len(issues)} problems; nothing written")
        return 1

    total = sum(len(e["trace"]) for e in examples)
    counts = {
        kind: sum(len(e["answer"].get(S.ARRAY_KEYS[kind], [])) for e in examples)
        for kind in S.INGREDIENT_TYPES
    }
    print(f"[examples] {len(examples)} examples, {total:,} trace characters")
    print(f"[examples] ingredients: {counts}")

    if args.check:
        print("[examples] every span is exact")
        return 0

    payload = {
        "schema_name": S.SCHEMA_NAME,
        "schema_version": S.SCHEMA_VERSION,
        "source_parquet": SOURCE_PARQUET,
        "examples": examples,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"[examples] wrote {OUT_PATH.relative_to(REPO)} ({OUT_PATH.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
