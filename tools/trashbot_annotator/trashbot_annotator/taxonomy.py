"""Taxonomy constants for TrashBot footage annotation.

The VLM prompt and the consolidation step share these strings.
Keep the strings exact. Do not reword a label here.
"""

from __future__ import annotations

# Talk category key -> display label shown in reports.
TALK_CATEGORIES: dict[str, str] = {
    "instruction": "Instruction — commanding the robot",
    "gossip": "Gossip — talking about it with others",
    "self_talk": "Self-talk — muttering to oneself",
}

# Address of one utterance or one interaction.
# to_robot: the person speaks to the robot.
# about_robot: the person speaks about the robot, not to it
#   ("eavesdropping" evidence).
# unrelated: the speech does not concern the robot.
SPEECH_ADDRESS: dict[str, str] = {
    "to_robot": "To the robot — speech directed at the robot",
    "about_robot": "About the robot — overheard talk about it",
    "unrelated": "Unrelated — speech that does not concern the robot",
}

# Kind of one utterance in the speech channel.
SPEECH_KINDS: frozenset[str] = frozenset(
    {
        "command",
        "greeting",
        "question",
        "praise",
        "insult",
        "warning",
        "explanation",
        "joke",
        "other",
    }
)

# Proxemic event key -> display label. Each event carries a distance
# estimate. The reports use these events for the distance questions.
PROXEMIC_EVENTS: dict[str, str] = {
    "trash_deposit": "Trash deposit — a person puts trash into the robot",
    "trash_toss": "Trash toss — a person throws trash at the robot from a distance",
    "gaze_at_robot": "Gaze — a person looks at the robot or tracks it",
    "talk_to_robot": "Talk to robot — a person speaks to the robot",
    "annoyance": "Annoyance — a person shows irritation or discomfort at the robot",
    "avoidance": "Avoidance — a person changes path or steps away from the robot",
}

# Proxemic distance zones after Hall. Bounds are metres.
# (zone key, lower bound inclusive, upper bound exclusive)
DISTANCE_ZONES: tuple[tuple[str, float, float], ...] = (
    ("intimate", 0.0, 0.45),
    ("personal", 0.45, 1.2),
    ("social", 1.2, 3.7),
    ("public", 3.7, float("inf")),
)

# Zone keys plus the fallback for events without a readable distance.
ZONE_KEYS: tuple[str, ...] = tuple(z[0] for z in DISTANCE_ZONES) + ("unknown",)

# Norm dimensions, in the canonical order.
NORM_DIMENSIONS: list[str] = [
    "Anthropomorphism & positive affect",
    "Conditional acceptance & adoption",
    "Discomfort, fear & distrust",
    "Functional expectations & competence",
    "Interaction norms & legibility",
    "Labor, jobs & value",
    "Legitimacy, safety & accountability",
    "Privacy & surveillance",
    "Shared-sidewalk & right-of-way",
]

# Meaning of the 0..3 relevance score for each norm dimension.
RELEVANCE_SCALE: str = (
    "Relevance scale 0..3: 0 = not relevant, "
    "1 = marginally relevant, 2 = relevant, 3 = highly relevant"
)

# Fixed annotator name written into consolidated rows.
ANNOTATOR_NAME: str = "trashbot_gemma4"

# Set view for fast membership checks.
_CATEGORY_KEYS: frozenset[str] = frozenset(TALK_CATEGORIES)
_DIMENSION_NAMES: frozenset[str] = frozenset(NORM_DIMENSIONS)
_EVENT_KEYS: frozenset[str] = frozenset(PROXEMIC_EVENTS)
_ADDRESS_KEYS: frozenset[str] = frozenset(SPEECH_ADDRESS)
_ZONE_KEYS: frozenset[str] = frozenset(ZONE_KEYS)


def is_valid_category(key: object) -> bool:
    """Return True when the key names a talk category."""
    return isinstance(key, str) and key in _CATEGORY_KEYS


def is_valid_dimension(name: object) -> bool:
    """Return True when the name matches a norm dimension string."""
    return isinstance(name, str) and name in _DIMENSION_NAMES


def is_valid_event(key: object) -> bool:
    """Return True when the key names a proxemic event."""
    return isinstance(key, str) and key in _EVENT_KEYS


def is_valid_address(key: object) -> bool:
    """Return True when the key names a speech address."""
    return isinstance(key, str) and key in _ADDRESS_KEYS


def is_valid_zone(key: object) -> bool:
    """Return True when the key names a distance zone."""
    return isinstance(key, str) and key in _ZONE_KEYS


def zone_for_distance(distance_m: object) -> str:
    """Map a distance in metres to its zone key.

    A value that is not a finite non-negative number maps to "unknown".
    """
    if isinstance(distance_m, bool) or not isinstance(distance_m, (int, float)):
        return "unknown"
    value = float(distance_m)
    if value != value or value < 0.0:
        return "unknown"
    for key, low, high in DISTANCE_ZONES:
        if low <= value < high:
            return key
    return "unknown"
