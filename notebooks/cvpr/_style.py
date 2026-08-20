"""The paper's house palette and figure style.

The swatches come from `UAIR/notebooks/colm-camera-ready/
corpus_descriptives_two_corpora.py`, so a figure here and a figure there read as
one system. Nothing below is invented.

Use `apply_house_style()` for a matplotlib figure and `WORDCLOUD_CMAP` for a
word cloud.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

__version__ = "1.0.0"

# The sampled palette. Keep the names: the other notebooks use them.
PAL: Dict[str, str] = {
    "teal": "#498573",
    "mint": "#96cebf",
    "cream": "#e5d2bb",
    "amber": "#f3b14f",
    "coral": "#e57264",
    "blue": "#5674b3",
    "periwinkle": "#a6c0f4",
    "tan": "#a08a6f",
    "green": "#83bd7b",
    "slate": "#a7b3d2",
    "warmgrey": "#7a6f63",
}

EDGE = "#1a1a1a"       # patch outline
EDGE_LW = 0.5

# The ground of every figure, and the value the sequential ramp starts from.
#
# White since 2026-08-14. It was the parchment `#f7f3ec` before that. A figure
# now carries NO colour of its own: it shows white on screen, and it saves with
# a transparent background (`savefig.transparent`, set in
# `apply_house_style`). Thus a figure sits on the page of the paper, or on a
# slide of any colour, and it brings no box with it.
PAPER = "#ffffff"

# The 2 continuous ramps, built from the same swatches.
CMAP = LinearSegmentedColormap.from_list(
    "uair", [PAL["blue"], PAL["periwinkle"], PAL["cream"], PAL["amber"], PAL["coral"]]
)
# The diverging ramp, for a value with a meaningful middle: a win rate against
# its base rate, for example. Coral is the losing end and teal the winning end.
#
# Warning: this is NOT red-to-green on purpose. About 8% of men cannot separate
# red from green, and a figure that carries its whole meaning in that one
# contrast is unreadable to them. Coral and teal differ in hue AND in
# luminance, so the ramp survives the loss, and both swatches are already in
# the house palette.
# Warning: the middle is a DARK grey, not a near-white. A diverging ramp
# usually goes pale in the middle, and that is right for a filled area sitting
# under black text. This ramp colours the TEXT ITSELF in a word cloud, so a
# pale middle disappears against the white ground: `#efe7dd` measures 0.807
# against the 0.45 that `_check_ink` demands. Here every stop is about as dark
# as the others (coral 0.296, grey 0.301, teal 0.194), so hue carries the
# meaning and luminance carries none. `_check_ramp` proves it.
CMAP_DIV = LinearSegmentedColormap.from_list(
    "uair_div", [PAL["coral"], "#9c948a", PAL["teal"]]
)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "uair_seq", [PAPER, PAL["cream"], PAL["mint"], PAL["teal"]]
)

# The swatches a word can be drawn in.
#
# Warning: this is NOT the whole palette. `mint`, `cream`, `periwinkle`, and
# `slate` are fills, made to sit under black text. As text on a light ground
# they are unreadable, and a word cloud is nothing but text. The list below
# holds only the swatches dark enough to read, and `_check_ink()` proves it.
# `amber` is absent for the same reason, and it is the surprising one: at 0.511
# it is LIGHTER than `green` (0.427), so it fails the same test the fills fail.
# Amber stays a fill colour. Do not add it back by eye.
INK: List[str] = [
    PAL["teal"], PAL["blue"], PAL["coral"],
    PAL["tan"], PAL["green"], PAL["warmgrey"],
]

# A word cloud picks a colour for each word from this map. It is a LISTED map,
# not a continuous one: a continuous ramp between 2 hues returns the muddy
# mixtures in between, and none of those are in the palette. A listed map can
# only ever return a swatch.
WORDCLOUD_CMAP = ListedColormap(INK, name="uair_ink")

# A word cloud draws in a random order, thus it needs a seed to be repeatable.
# Without it the same counts give a different picture every run, and no reader
# can tell a real change from a reshuffle.
WORDCLOUD_SEED = 20260813


def luminance(color: str) -> float:
    """Relative luminance, by the WCAG definition."""
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = mcolors.to_rgb(color)[:3]
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def text_on(fill: str) -> str:
    """Black or white label text, from the fill's luminance.

    A fixed rule breaks when the palette changes. Luminance does not care which
    palette is in use.
    """
    return EDGE if luminance(fill) > 0.42 else "white"


def _check_ink(max_luminance: float = 0.45) -> List[str]:
    """Name any ink too light to read on `PAPER`. Empty means the list is good."""
    return [c for c in INK if luminance(c) > max_luminance]


def _check_ramp(cmap=None, max_luminance: float = 0.45, steps: int = 21) -> List[float]:
    """Name any point of a ramp too light to read as TEXT on `PAPER`.

    Call this for a ramp that colours words, never for one that fills an area.
    A fill sits under black text and may be pale; a word IS the ink.

    Returns the positions that fail. Empty means the whole ramp reads.
    """
    ramp = cmap if cmap is not None else CMAP_DIV
    bad = []
    for i in range(steps):
        t = i / (steps - 1)
        if luminance(mcolors.to_hex(ramp(t))) > max_luminance:
            bad.append(round(t, 3))
    return bad


def apply_house_style() -> None:
    """Set the camera-ready rcParams. Call it one time in a notebook."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # A saved figure carries no background. It shows white on screen and
        # writes transparent, thus it sits on the page rather than on a box of
        # its own colour.
        "savefig.transparent": True,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "patch.edgecolor": EDGE,
        "patch.linewidth": EDGE_LW,
        "patch.force_edgecolor": True,
        "hatch.linewidth": 0.5,
    })
