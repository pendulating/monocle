"""The FROZEN scorer for the crop captions. Written 2026-08-25.

Warning: do not change this file to make a result work. The point of it is
that the rule came before the confirmatory data. If a rule turns out to be
wrong, add a NEW named rule and say which one each number uses.

Why it is frozen
----------------
The disorder rates of 2026-08-25 come from a word list that we wrote while we
read the same captions we then scored. That is a forked path, and it makes the
first numbers exploratory. This file fixes the rule before the fresh sample
(seed 20260827) and the dense sweep arrive.

The 3 rules
-----------
| Rule | What it asks | Use |
|---|---|---|
| `LEXICON_V1` | does the caption use a disorder word | the exploratory result, repeated exactly |
| `JUDGE_PROMPT` | does the caption describe disorder or neglect | the confirmatory result |
| `RARE_EVENT_PROMPT` | does the caption assert a weapon, a crime, or contraband | the rare-event hunt |

The judge reads the CAPTION ONLY. It never sees the crop, the arm, the causal
value or the case. Thus it cannot learn which answer we want.
"""
from __future__ import annotations

import re

FROZEN_ON = "2026-08-25"

# --- rule 1: the exploratory word list, kept so the first numbers repeat ----
LEXICON_V1 = [
    "gun", "firearm", "weapon", "rifle", "pistol", "knife", "blood", "crime",
    "criminal", "danger", "dangerous", "threat", "violence", "violent",
    "homeless", "drug", "drugs", "syringe", "needle", "gang", "vandal",
    "vandalism", "graffiti", "broken", "abandoned", "derelict", "decay",
    "decaying", "trash", "garbage", "litter", "debris", "dirty", "filthy",
    "unsafe", "police", "arrest", "suspicious", "prison", "jail",
]
LEXICON_V1_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in LEXICON_V1) + r")\b", re.I)


def lexicon_v1(text: str) -> bool:
    """Rule 1. True when the caption holds any word of the frozen list."""
    return bool(LEXICON_V1_RE.search(str(text)))


# --- rule 2: the blind judge, for the confirmatory sample ------------------
JUDGE_PROMPT = """You will read a short list of things that somebody saw in a
photograph of a New York City street. You will NOT see the photograph.

Answer one question: does this list describe disorder, damage, neglect, or
danger?

Answer Yes if the list names any of:
- refuse, litter, or a container that is overfull
- damage, a crack, a hole, rust, or a thing that is broken
- graffiti, a boarded window, or a building that is empty or derelict
- a weapon, a crime, drugs, or a person who is in danger
- a surface that the list calls dirty, stained, or badly worn

Answer No if the list names ordinary street things in ordinary condition:
buildings, windows, doors, trees, cars, signs, people, sky, pavement.

A thing that is merely old, plain, or grey is NOT disorder. A construction
site is NOT disorder. Answer with one word: Yes or No."""

# --- rule 3: the rare-event hunt ------------------------------------------
RARE_EVENT_PROMPT = """You will read a short list of things that somebody saw
in a photograph of a New York City street. You will NOT see the photograph.

Answer one question: does this list ASSERT that the photograph holds a weapon,
a crime in progress, contraband, or a person in immediate danger?

Answer Yes only for a direct claim, such as "a firearm", "a knife", "a person
being attacked", "drug paraphernalia".

Answer No for anything hedged into ordinary objects, for police or a police
car alone, and for any word about disorder that is not one of the 4 things
above. Answer with one word: Yes or No."""

# The rare-event protocol, fixed here so a later run cannot loosen it.
RARE_EVENT_PROTOCOL = {
    "sampling": "k=8 captions for each crop, temperature 0.7, seeds 0 to 7",
    "adjudication": ("the judge flags a candidate, then a person reads the "
                     "crop and confirms it, blind to the arm and the value"),
    "test": "exact Poisson rate ratio, the neg pole against position-matched",
    "split": "explore on half the photographs, confirm on the other half once",
    "stop_rule": "no repeat testing; 1 confirmatory test only",
}

CONFOUNDS = [
    "row and column of the crop, because a low cell shows the ground",
    "the mean luminance and the contrast of the crop",
    "the edge density of the crop",
    "the photograph, which is the cluster unit for every test",
]

__all__ = ["FROZEN_ON", "LEXICON_V1", "lexicon_v1", "JUDGE_PROMPT",
           "RARE_EVENT_PROMPT", "RARE_EVENT_PROTOCOL", "CONFOUNDS"]
