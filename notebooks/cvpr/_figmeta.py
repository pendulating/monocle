"""The caption of a figure, as a JSON file beside the image.

A figure carries no caption prose on its face. The caption goes into a
co-located `.json` file with the same stem as the image:

| Key | Holds |
|-----|-------|
| `plot-title` | a short title |
| `plot-caption` | the full caption (plain text, LaTeX for math) |
| `plot-latex-label` | the `\\label` of the figure, `fig:<kebab-case>` |
| `plot-tags` | a list of short strings: notebook, cases, arms, `v3`, measure |
| `provenance` | the module, its version, the arm, the corpus root, the date |

A plot function builds this content with `figure_meta` and attaches it to the
figure with `attach`. Then the notebook shows the same text under the figure
(`caption_md`), and `save_figure` writes it beside the PNG or the PDF.

See `/home/mwf62/style/proper-plotting.md` ("Captions Are Not Optional").
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

__version__ = "1.0.0"

#: The 4 keys every sidecar holds, with these exact names.
REQUIRED_KEYS = ("plot-title", "plot-caption", "plot-latex-label", "plot-tags")

#: The attribute of a matplotlib figure that holds its caption content.
ATTR = "_uair_figmeta"

_LABEL = re.compile(r"^fig:[a-z0-9]+(-[a-z0-9]+)*$")


def kebab(text: str) -> str:
    """`gemma_by_qwen27_v3` gives `gemma-by-qwen27-v3`."""
    out = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return re.sub(r"-{2,}", "-", out)


def fig_label(*parts: Optional[str]) -> str:
    """Join the parts into a figure label: `fig:` and kebab case.

    A part that is `None` or empty is left out, thus an optional arm suffix
    needs no test at the call site.
    """
    body = "-".join(kebab(p) for p in parts if p)
    return f"fig:{body}"


def _tags(tags: Iterable[Any]) -> list:
    """Give the tags as a list of unique, non-empty strings, in order."""
    seen: list = []
    for t in tags:
        t = str(t).strip() if t is not None else ""
        if t and t not in seen:
            seen.append(t)
    return seen


def figure_meta(title: str, caption: str, label: str, tags: Iterable[Any],
                **provenance: Any) -> Dict[str, Any]:
    """Build the caption content of 1 figure.

    Args:
        title: A short title, a name and not a sentence.
        caption: The full caption: how to read the figure, the marks and
            the colours, the n, the arm and the corpus, the caveats.
        label: The LaTeX label, `fig:<kebab-case>`.
        tags: Short strings. Put the notebook, the cases, the arms, the
            schema version, and the measure.
        **provenance: Free keys (module, version, arm, root). They go under
            `provenance`, never beside the 4 keys.

    Raises:
        ValueError: The title or the caption is empty, or the label is not
            `fig:` in kebab case.
    """
    if not str(title).strip():
        raise ValueError("a figure needs a title")
    if not str(caption).strip():
        raise ValueError("a figure needs a caption")
    if not _LABEL.match(str(label)):
        raise ValueError(f"the label {label!r} is not fig:<kebab-case>")
    meta: Dict[str, Any] = {
        "plot-title": str(title).strip(),
        "plot-caption": " ".join(str(caption).split()),
        "plot-latex-label": str(label),
        "plot-tags": _tags(tags),
    }
    prov = {k: v for k, v in provenance.items() if v is not None}
    if prov:
        meta["provenance"] = prov
    return meta


def attach(fig, meta: Mapping[str, Any]):
    """Put the caption content on the figure, and give the figure back."""
    setattr(fig, ATTR, dict(meta))
    return fig


def meta_of(fig) -> Optional[Dict[str, Any]]:
    """Give the caption content of a figure, or `None`."""
    meta = getattr(fig, ATTR, None)
    return dict(meta) if meta else None


def caption_md(fig_or_meta) -> str:
    """The caption as markdown, for `mo.md` under the figure in a notebook."""
    meta = (fig_or_meta if isinstance(fig_or_meta, Mapping)
            else meta_of(fig_or_meta))
    if not meta:
        return ""
    return (f"**{meta['plot-title']}.** {meta['plot-caption']} "
            f"(`{meta['plot-latex-label']}`)")


def sidecar_path(path) -> Path:
    """The `.json` path beside an image: `a/b.png` gives `a/b.json`."""
    return Path(path).with_suffix(".json")


def save_figure(fig, path, *, title: Optional[str] = None,
                caption: Optional[str] = None, label: Optional[str] = None,
                tags: Optional[Iterable[Any]] = None, dpi: Optional[float] = None,
                provenance: Optional[Mapping[str, Any]] = None,
                **savefig_kwargs: Any) -> Path:
    """Save the figure, and write its caption as `<path stem>.json`.

    A keyword that is `None` takes the value that the plot function attached
    to the figure (`attach`). A keyword that you give replaces it.

    Args:
        fig: The matplotlib figure.
        path: The image path (`.png` or `.pdf`).
        dpi: Given to `savefig` when it is not `None`.
        provenance: More keys for the `provenance` block.
        **savefig_kwargs: Given to `fig.savefig`.

    Returns:
        The path of the sidecar.

    Raises:
        ValueError: The figure has no caption and the call gives none.
    """
    base = meta_of(fig) or {}
    meta = figure_meta(
        title if title is not None else base.get("plot-title", ""),
        caption if caption is not None else base.get("plot-caption", ""),
        label if label is not None else base.get("plot-latex-label", ""),
        tags if tags is not None else base.get("plot-tags", []),
    )
    prov = dict(base.get("provenance", {}))
    prov.update({k: v for k, v in (provenance or {}).items()
                 if v is not None})
    path = Path(path)
    prov.setdefault("file", path.name)
    prov.setdefault("written", _dt.datetime.now().isoformat(timespec="seconds"))
    meta["provenance"] = prov

    path.parent.mkdir(parents=True, exist_ok=True)
    if dpi is not None:
        savefig_kwargs["dpi"] = dpi
    fig.savefig(path, **savefig_kwargs)
    side = sidecar_path(path)
    side.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return side


def read_sidecar(path) -> Dict[str, Any]:
    """Read the sidecar of an image (or the sidecar path itself)."""
    return json.loads(sidecar_path(path).read_text(encoding="utf-8"))


def captioned(fig):
    """The figure with its caption under it, for a marimo notebook.

    The caption is markdown (`caption_md`), thus the reader of the notebook
    sees it, and the figure itself stays free of caption text.
    """
    import marimo as mo

    text = caption_md(fig)
    return mo.vstack([fig, mo.md(text)]) if text else fig
