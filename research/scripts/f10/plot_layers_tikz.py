"""F10.20b: write the TikZ panel of the layers of the judgment for the paper.

The panel reads `research/figures/f10/mechanism/phase_layers.csv` (F10.20). It draws the layer schematic of the Jacobian lens paper (sensory, workspace, motor),
and 3 rows of bands (image,
standard, verdict) for 2 models: dark bars for `TOP`, on the top layer scale, and light bars for
`BOTTOM`, on the bottom scale. A verdict band that no layer reaches (`reached` false) is drawn
at its peak layer, as the others.

Output: `research/figures/f10/mechanism/layers_panel.tex`, a `tikzpicture` that the paper inputs.

Use:
    python research/scripts/f10/plot_layers_tikz.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "research/figures/f10/mechanism"
OUT = FIG / "layers_panel.tex"
__version__ = "1.1.0"

TOP, BOTTOM = "gemma-4-31b", "qwen3.6-27b"
NAMES = {"gemma-4-31b": "Gemma-4-31B", "qwen3.5-9b": "Qwen3.5-9B", "qwen3.6-27b": "Qwen3.6-27B"}
X0, X1 = 2.750, 13.750  # the layer scale, from relative depth 0 to 1
PATCH_LAYERS = (8, 20, 32, 44, 56)  # the layers at which the panel lists the patch words
N_WORDS = 2
ROWS = (("image", "cscene", r"\textbf{\color{cscene!80!black}image}, mean patch"),
        ("standard", "cstandard", r"\textbf{\color{cstandard!80!black}a standard}, \texttt{?}"),
        ("verdict", "cverdict", r"\textbf{\color{cverdict!80!black}verdict}, last token"))

# The schematic, after the layer figure of the Jacobian lens paper. Coordinates are fixed.
SCHEMATIC = r"""
  \node[font=\tiny\sffamily, align=center, anchor=south, inner sep=1pt] at (4.537,-5.090) {\textbf{sensory}\\[-1pt]{\color{black!55}low-level parsing}};
  \draw[black!35, line width=.4pt, rounded corners=1.5pt] (2.780,-5.220) -- (2.780,-5.140) -- (6.295,-5.140) -- (6.295,-5.220);
  \node[font=\tiny\sffamily, align=center, anchor=south, inner sep=1pt] at (9.185,-5.090) {\textbf{workspace layers}\\[-1pt]{\color{black!55}J-space carries reportable, reusable content}};
  \draw[black!35, line width=.4pt, rounded corners=1.5pt] (6.465,-5.220) -- (6.465,-5.140) -- (11.905,-5.140) -- (11.905,-5.220);
  \node[font=\tiny\sffamily, align=center, anchor=south, inner sep=1pt] at (12.898,-5.090) {\textbf{motor}\\[-1pt]{\color{black!55}next-token odds}};
  \draw[black!35, line width=.4pt, rounded corners=1.5pt] (12.075,-5.220) -- (12.075,-5.140) -- (13.720,-5.140) -- (13.720,-5.220);
  \draw[black!15, fill=black!2, rounded corners=4pt] (2.61,-5.270) rectangle (13.89,-5.890);
  \fill[black!12, rounded corners=2pt] (2.75,-5.390) rectangle (13.75,-5.530);
  \draw[cworkspace, fill=cworkspace!12, densely dashed, line width=.5pt, rounded corners=2pt] (6.435,-5.380) rectangle (11.935,-5.540);
  \draw[black!35, <->, >=stealth, line width=.4pt] (2.75,-5.720) -- (7.810,-5.720);
  \draw[black!35, <->, >=stealth, line width=.4pt] (8.690,-5.720) -- (13.75,-5.720);
  \node[font=\tiny\sffamily\itshape, text=black!55, inner sep=0pt] at (8.250,-5.720) {J-space};
  \node[draw=black!15, fill=black!2, rounded corners=2pt, font=\tiny\sffamily\itshape, text=black!60, minimum width=1.0cm, minimum height=.46cm, inner sep=1pt] at (2.01,-5.580) {Input};
  \node[draw=black!15, fill=black!2, rounded corners=2pt, font=\tiny\sffamily\itshape, text=black!60, minimum width=1.0cm, minimum height=.46cm, inner sep=1pt] at (14.49,-5.580) {Output};
  \draw[black!30, ->, >=stealth, line width=.4pt] (2.51,-5.580) -- (2.61,-5.580);
  \draw[black!30, ->, >=stealth, line width=.4pt] (13.89,-5.580) -- (13.99,-5.580);
"""


def x_of(layer: float, n_layers: int) -> float:
    """Give the x of a layer on the scale of a model with `n_layers` layers."""
    return X0 + (X1 - X0) * layer / (n_layers - 1)


def loops() -> str:
    out = []
    for i in range(6):
        cx = 3.667 + i * 1.8333
        out.append(rf"  \node[fill=cworkspace!9, draw=cworkspace!18, rounded corners=1.5pt, font=\scriptsize, minimum width=.72cm, minimum height=.32cm, inner sep=0pt] (l{i}) at ({cx:.3f},-6.180) {{$\ell_{i + 1}$}};")
        out.append(rf"  \draw[black!30, line width=.4pt] ({cx - 0.56:.3f},-5.890) .. controls ({cx - 0.56:.3f},-6.160) .. (l{i}.west);")
        out.append(rf"  \draw[black!30, ->, >=stealth, line width=.4pt] (l{i}.east) .. controls ({cx + 0.56:.3f},-6.160) .. ({cx + 0.56:.3f},-5.890);")
    return "\n".join(out)


def scale(model: str, n_layers: int, y_tick: float, y_num: float, label_y: float, saved_last: int) -> str:
    """Give the ticks of a layer scale every 8 layers, plus the last saved layer."""
    ticks = sorted(set(list(range(0, saved_last + 1, 8)) + [saved_last]))
    out = [rf"  \node[anchor=east, font=\tiny\sffamily, inner sep=1pt] at (2.67,{label_y:.3f}) {{{NAMES[model]} layer}};"]
    for layer in ticks:
        x = x_of(layer, n_layers)
        out.append(rf"  \draw[black!40, line width=.3pt] ({x:.3f},{y_tick:.3f}) -- ({x:.3f},{y_tick - 0.06:.3f});")
        out.append(rf"  \node[font=\tiny\sffamily, inner sep=0pt] at ({x:.3f},{y_num:.3f}) {{{layer}}};")
    return "\n".join(out)


def words(csv: pd.DataFrame, layer: int, end: str) -> str:
    top = csv[(csv["layer"] == layer) & (csv["end"] == end)]["top"].iloc[0]
    return ", ".join(re.sub(r"\s*\(\d+\)", "", w).strip() for w in top.split(",")[:N_WORDS])


def patch_strip(pw: pd.DataFrame, n_layers: int, y: float) -> str:
    """Give the words that the patches of `TOP` read, at 5 layers of its scale."""
    out = [rf"  \node[anchor=east, font=\tiny\sffamily, inner sep=1pt, align=right] at (2.67,{y - 0.16:.3f}) {{\textbf{{patches read}}, {NAMES[TOP]}\\[-1pt]{{\color{{cread!80!black}}up end}} / {{\color{{csense!80!black}}down end}}}};"]
    for layer in PATCH_LAYERS:
        x = x_of(layer, n_layers)
        out.append(rf"  \node[anchor=north, font=\tiny\sffamily, align=center, inner sep=1pt, text width=1.9cm] at ({x:.3f},{y:.3f}) {{{{\color{{cread!80!black}}{words(pw, layer, 'up')}}}\\[-1pt]{{\color{{csense!80!black}}{words(pw, layer, 'down')}}}}};")
        out.append(rf"  \draw[black!25, line width=.3pt] ({x:.3f},{y + 0.02:.3f}) -- ({x:.3f},{y + 0.10:.3f});")
    return "\n".join(out)


def bar(row: pd.Series, color: str, y_top: float, y_bot: float, light: bool) -> str:
    fill = f"{color}!45" if light else color
    n = int(row["n_layers"])
    x_a = x_of(row["first"], n)
    at_end = row["phase"] == "verdict"
    x_b = X1 if at_end else x_of(row["last"], n)
    label = f"L{int(row['first'])}" if row["first"] == row["last"] else f"L{int(row['first'])}--{int(row['last'])}"
    shape = rf"  \fill[{fill}] ({x_a:.3f},{y_top:.3f}) rectangle ({x_b:.3f},{y_bot:.3f});"
    ym = (y_top + y_bot) / 2
    text = (rf"  \node[anchor=east, font=\tiny\sffamily, text=black!70, inner sep=1.5pt] at ({x_a:.3f},{ym:.3f}) {{{label}}};" if at_end
            else rf"  \node[anchor=west, font=\tiny\sffamily, text=black!70, inner sep=1.5pt] at ({x_b:.3f},{ym:.3f}) {{{label}}};")
    return shape + "\n" + text


def main() -> None:
    ph = pd.read_csv(FIG / "phase_layers.csv")
    top = ph[ph["model"] == TOP].set_index("phase")
    bot = ph[ph["model"] == BOTTOM].set_index("phase")
    n_top, n_bot = int(top["n_layers"].iloc[0]), int(bot["n_layers"].iloc[0])
    saved_top, saved_bot = int(top.loc["verdict", "last"]), int(bot.loc["verdict", "last"])
    lines = [f"% Generated by research/scripts/f10/plot_layers_tikz.py {__version__} from phase_layers.csv (F10.20). Do not edit by hand.",
             r"\definecolor{cscene}{HTML}{5674B3}\definecolor{cread}{HTML}{498573}\definecolor{csense}{HTML}{E57264}\definecolor{cstandard}{HTML}{F3B14F}\definecolor{cverdict}{HTML}{7C8FBF}\definecolor{cworkspace}{HTML}{B8863B}",
             r"\begin{tikzpicture}", r"  \begin{scope}[font=\sffamily]", SCHEMATIC.strip("\n"), loops()]
    # the top scale, then the patch strip, then the bands
    y = -6.580
    lines.append(f"  % Dark bars are {NAMES[TOP]}, on the top scale. Light bars are {NAMES[BOTTOM]}, on the bottom scale.")
    lines.append(scale(TOP, n_top, y - 0.07, y + 0.01, y, saved_top))
    y = y - 0.30
    for phase, color, label in ROWS:
        lines.append(rf"  \node[anchor=east, font=\tiny\sffamily, inner sep=1pt, align=right] at (2.67,{y - 0.14:.3f}) {{{label}}};")
        lines.append(bar(ph[(ph["model"] == TOP) & (ph["phase"] == phase)].iloc[0], color, y, y - 0.14, light=False))
        lines.append(bar(ph[(ph["model"] == BOTTOM) & (ph["phase"] == phase)].iloc[0], color, y - 0.165, y - 0.305, light=True))
        y -= 0.385
    lines.append(scale(BOTTOM, n_bot, y - 0.01, y - 0.15, y - 0.15, saved_bot))
    lines += [r"  \end{scope}", r"\end{tikzpicture}"]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
