"""The shared parts of the paper-ready LaTeX tables.

The v3 analysis modules (`_gold_alignment`, `_trace_channels`, `_ic_v3`,
`_ic_metadata`) write their tables with these helpers. The output follows
the house convention of `_proxy_overview.py` and `_results_table.py`:

- A header comment that names the module and its version.
- `table`, `\\centering`, a `\\caption` with a bold question, a `\\label`.
- `\\setlength{\\tabcolsep}{4pt}`, `\\small`, and `\\resizebox` when wide.
- booktabs rules, and numbers without a leading zero (`.93`, `-.01`).
- A dash for an empty cell.

`check_table` is a parse test. It finds unbalanced braces, a row whose
cell count is not the column count of the tabular, and an unescaped `_`.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

import numpy as np

__version__ = "1.0.0"

DASH = "--"

# The printed name of each model. A judge and an extractor use the same map.
MODEL_TEX = {
    "gemma-4-12b": "Gemma-4-12B",
    "gemma-4-31b": "Gemma-4-31B",
    "qwen3.5-9b": "Qwen3.5-9B",
    "qwen3.8-27b": "Qwen3.8-27B",
}

_SPECIAL = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "#": "\\#",
            "_": "\\_", "$": "\\$", "{": "\\{", "}": "\\}",
            "~": "\\textasciitilde{}", "^": "\\textasciicircum{}",
            # The OT1 text font prints these 3 as other glyphs.
            "|": "\\textbar{}", "<": "\\textless{}", ">": "\\textgreater{}"}


def _void(v) -> bool:
    if v is None:
        return True
    try:
        return not np.isfinite(float(v))
    except (TypeError, ValueError):
        return True


def fmt(v, digits: int = 2, signed: bool = False) -> str:
    """Write a number without a leading zero: `.93`, `-.01`, `1.20`.

    A missing or a non-finite value gives a dash. A value that rounds to 0
    carries no sign. `signed` adds a `+` to a positive value.
    """
    if _void(v):
        return DASH
    x = float(v)
    if abs(x) < 0.5 * 10 ** -digits:
        x = 0.0
    s = f"{x:+.{digits}f}" if signed and x != 0 else f"{x:.{digits}f}"
    for lead, short in (("0.", "."), ("-0.", "-."), ("+0.", "+.")):
        if s.startswith(lead):
            s = short + s[len(lead):]
            break
    return s


def fmt_pct(v, digits: int = 0) -> str:
    """Write a share in [0, 1] as a percent number without the sign: `42`."""
    if _void(v):
        return DASH
    return f"{100 * float(v):.{digits}f}"


def fmt_int(v) -> str:
    """Write a count with a LaTeX thousands comma: `12{,}345`."""
    if _void(v):
        return DASH
    return f"{int(round(float(v))):,}".replace(",", "{,}")


def fmt_ci(lo, hi, digits: int = 2) -> str:
    """Write an interval as `[-.04, .02]`, or a dash when a bound is missing."""
    if _void(lo) or _void(hi):
        return DASH
    return f"[{fmt(lo, digits)}, {fmt(hi, digits)}]"


def escape(text) -> str:
    """Escape the LaTeX special characters of a free-text name."""
    if text is None or (isinstance(text, float) and not np.isfinite(text)):
        return DASH
    return "".join(_SPECIAL.get(c, c) for c in str(text))


def model_label(name: str) -> str:
    """`gemma-4-12b/instruct_thinking` -> `Gemma-4-12B`."""
    key = (name or "").split("/")[0]
    return MODEL_TEX.get(key, escape(key))


def arm_label(arm: str) -> str:
    """`gemma_by_qwen27_v3` -> `Gemma-4-12B / Qwen3.8-27B` (judge / extractor)."""
    import _extractions as X
    spec = X.ARMS.get(arm)
    if spec is None:
        return escape(arm)
    return f"{model_label(spec['judge'])} / {model_label(spec['extractor'])}"


def case_label(case: str) -> str:
    """The printed name of a case, from `_words_across.case_label`."""
    import _words_across as WA
    return escape(WA.case_label(case))


def row(cells: Iterable) -> str:
    """Join the cells of 1 row and end it with `\\\\`."""
    return " & ".join(str(c) for c in cells) + " \\\\"


def group_header(groups: Sequence[tuple], lead: int = 1) -> List[str]:
    """A `\\multicolumn` header and its `\\cmidrule` line.

    Args:
        groups: `(title, width)` for each group, left to right.
        lead: The number of columns before the first group.
    """
    cells = [""] * lead
    rules = []
    col = lead + 1
    for title, width in groups:
        cells.append(f"\\multicolumn{{{width}}}{{c}}{{{title}}}")
        rules.append(f"\\cmidrule(lr){{{col}-{col + width - 1}}}")
        col += width
    return [row(cells), "".join(rules)]


def table(module: str, version: str, caption: str, label: str, colspec: str,
          head: Sequence[str], body: Sequence[str], resize: bool = False,
          notes: Optional[str] = None, env: str = "table",
          tabcolsep: str = "4pt") -> str:
    """Wrap a tabular in the house float.

    Args:
        module: The module name without `.py`, e.g. `_ic_v3`.
        version: The `__version__` of the module.
        caption: The full caption, with the `\\textbf{...}` title.
        label: The `\\label` key, e.g. `tab:ic-sign-table`.
        colspec: The tabular column spec, e.g. `lrrr`.
        head: The header rows, without `\\toprule` and `\\midrule`.
        body: The body rows. A row `\\midrule` is permitted.
        resize: Scale the tabular to the line width.
        notes: A footnote line under the tabular.
    """
    needs = "% Needs \\usepackage{booktabs}" + (
        " and \\usepackage{graphicx}." if resize else ".")
    lines = [
        f"% Generated by {module}.py — do not edit by hand.",
        f"% Written by notebooks/cvpr/{module}.py v{version}.",
        needs,
        f"\\begin{{{env}}}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
        "\\small",
    ]
    if resize:
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *head, "\\midrule",
              *body, "\\bottomrule", "\\end{tabular}" + ("%" if resize else "")]
    if resize:
        lines.append("}")
    if notes:
        lines += ["\\par\\smallskip", "{\\footnotesize " + notes + "}"]
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# The parse test
# --------------------------------------------------------------------------

def _brace_arg(text: str, pos: int) -> tuple:
    """Read the `{...}` group that starts at `pos`. Return it and the end."""
    while text[pos].isspace():
        pos += 1
    depth, start = 0, pos
    while pos < len(text):
        c = text[pos]
        if c == "\\":
            pos += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:pos], pos + 1
        pos += 1
    return text[start + 1:], pos


def _n_columns(colspec: str) -> int:
    spec = colspec
    while re.search(r"@\{", spec):
        i = spec.find("@{")
        _, end = _brace_arg(spec, i + 1)
        spec = spec[:i] + spec[end:]
    spec = re.sub(r"[pmb]\{[^}]*\}", "l", spec)
    return sum(1 for c in spec if c in "lrcX")


def _row_width(cells_text: str) -> int:
    # Count the `&` at brace depth 0 that a backslash does not escape.
    width, depth, i = 1, 0, 0
    spans = []
    while i < len(cells_text):
        c = cells_text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "&" and depth == 0:
            width += 1
        i += 1
    for m in re.finditer(r"\\multicolumn\{(\d+)\}", cells_text):
        spans.append(int(m.group(1)) - 1)
    return width + sum(spans)


def check_table(tex: str) -> List[str]:
    """Return the problems of 1 table. An empty list means it passes."""
    problems: List[str] = []
    body = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))
    depth = 0
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(c, 0)
        if depth < 0:
            problems.append("a closing brace has no opening brace")
            depth = 0
        i += 1
    if depth != 0:
        problems.append(f"{depth} brace(s) stay open")
    for m in re.finditer(r"(?<!\\)_", body):
        before = body[:m.start()]
        # A `_` inside math mode is a subscript and is correct.
        if before.count("$") % 2 == 0:
            problems.append(f"an unescaped _ near {body[max(0, m.start() - 20):m.start() + 20]!r}")
    m = re.search(r"\\begin\{(tabular\*?)\}", body)
    if not m:
        problems.append("no tabular")
        return problems
    pos = m.end()
    args = []
    for _ in range(2 if m.group(1) == "tabular*" else 1):
        arg, pos = _brace_arg(body, pos)
        args.append(arg)
    n_cols = _n_columns(args[-1])
    inner = body[pos:body.find("\\end{tabular")]
    for line in inner.split("\\\\"):
        text = line.strip()
        text = re.sub(r"\\(toprule|midrule|bottomrule)", "", text)
        text = re.sub(r"\\cmidrule\(lr\)\{[^}]*\}", "", text).strip()
        if not text:
            continue
        w = _row_width(text)
        if w != n_cols:
            problems.append(f"a row has {w} cells, the tabular has {n_cols}: {text[:60]!r}")
    for m in re.finditer(r"\\cmidrule\(lr\)\{(\d+)-(\d+)\}", inner):
        if int(m.group(2)) > n_cols:
            problems.append(f"a cmidrule passes column {n_cols}")
    return problems
