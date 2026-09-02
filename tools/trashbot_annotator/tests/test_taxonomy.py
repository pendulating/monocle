"""Tests for taxonomy constants."""

from __future__ import annotations

from trashbot_annotator.taxonomy import (
    ANNOTATOR_NAME,
    NORM_DIMENSIONS,
    PROXEMIC_EVENTS,
    RELEVANCE_SCALE,
    SPEECH_ADDRESS,
    TALK_CATEGORIES,
    ZONE_KEYS,
    is_valid_address,
    is_valid_category,
    is_valid_dimension,
    is_valid_event,
    is_valid_zone,
    zone_for_distance,
)


def test_talk_categories_exact() -> None:
    """The category map holds the exact keys and display labels."""
    assert TALK_CATEGORIES == {
        "instruction": "Instruction — commanding the robot",
        "gossip": "Gossip — talking about it with others",
        "self_talk": "Self-talk — muttering to oneself",
    }


def test_norm_dimensions_exact() -> None:
    """The dimension list holds the exact strings in order."""
    assert NORM_DIMENSIONS == [
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


def test_relevance_scale_describes_0_to_3() -> None:
    """The scale text names both ends of the range."""
    assert "0" in RELEVANCE_SCALE
    assert "3" in RELEVANCE_SCALE


def test_annotator_name() -> None:
    """The annotator name is fixed."""
    assert ANNOTATOR_NAME == "trashbot_gemma4"


def test_validators() -> None:
    """Validators accept taxonomy members and reject other values."""
    key = next(iter(TALK_CATEGORIES))
    assert is_valid_category(key)
    assert not is_valid_category("banter")
    assert is_valid_dimension(NORM_DIMENSIONS[0])
    assert not is_valid_dimension("Privacy vibes")
    assert is_valid_event("trash_deposit")
    assert not is_valid_event("hug")
    assert is_valid_address("to_robot")
    assert not is_valid_address("shouting")
    assert is_valid_zone("intimate")
    assert not is_valid_zone("near")


def test_proxemic_event_keys() -> None:
    """The event map holds the six deployment questions."""
    assert set(PROXEMIC_EVENTS) == {
        "trash_deposit",
        "trash_toss",
        "gaze_at_robot",
        "talk_to_robot",
        "annoyance",
        "avoidance",
    }


def test_speech_address_keys() -> None:
    """The address map holds the three directions of speech."""
    assert set(SPEECH_ADDRESS) == {"to_robot", "about_robot", "unrelated"}


def test_zone_for_distance() -> None:
    """Distances map to the Hall zones; bad values map to unknown."""
    assert zone_for_distance(0.2) == "intimate"
    assert zone_for_distance(0.8) == "personal"
    assert zone_for_distance(2.0) == "social"
    assert zone_for_distance(10.0) == "public"
    assert zone_for_distance(None) == "unknown"
    assert zone_for_distance(-1.0) == "unknown"
    assert zone_for_distance(True) == "unknown"
    assert "unknown" in ZONE_KEYS
