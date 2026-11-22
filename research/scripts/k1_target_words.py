"""The word list for each case, taken from the question and nothing else.

The content test asks whether a crop names the thing the question is
about. The list must come from the question, and it must be fixed before
anyone counts. These lists are written here so a later run uses the same
words as an earlier one.

The restaurants list is the one the first content test used, and it is
copied from `vlm-narratives-docs/causal-lens-retrospective.md`. The other
6 lists are built the same way, from the question of the case.
"""
from __future__ import annotations

TARGET_WORDS: dict[str, list[str]] = {
    "restaurants": [
        "restaurant", "storefront", "sign", "awning", "menu", "window",
        "door", "entrance", "table", "chairs", "seating", "cafe", "bar",
        "counter", "food",
    ],
    "libraries": [
        "library", "building", "facade", "wall", "window", "door",
        "entrance", "brick", "stone", "step", "steps", "column", "roof",
        "sign", "sidewalk",
    ],
    "schools": [
        "school", "building", "facade", "wall", "window", "door",
        "entrance", "brick", "fence", "yard", "playground", "sign",
        "roof", "steps", "sidewalk",
    ],
    "parks_plazas": [
        "park", "plaza", "tree", "trees", "bush", "bushes", "grass",
        "planter", "bench", "benches", "path", "lawn", "garden",
        "shrub", "flower",
    ],
    "road_quality": [
        "road", "roadway", "surface", "asphalt", "pavement", "paved",
        "crack", "cracks", "pothole", "patch", "lane", "curb",
        "marking", "markings", "street",
    ],
    "street_photography": [
        "street", "sidewalk", "people", "person", "pedestrian",
        "storefront", "sign", "light", "shadow", "window", "facade",
        "doorway", "awning", "bicycle", "crosswalk",
    ],
    "subway_safety": [
        "subway", "entrance", "stair", "stairs", "staircase", "railing",
        "handrail", "station", "sign", "gate", "platform", "light",
        "tunnel", "kiosk", "sidewalk",
    ],
}
