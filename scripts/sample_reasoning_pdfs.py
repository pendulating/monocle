#!/usr/bin/env python3
"""Render PDF samples of pairwise-VQA reasoning traces with the paired images.

For a pairwise VQA stage output parquet (the kind produced by
``dagspaces.urbanpairvqa``), draw a reproducible random sample of pairs and
write one PDF per pair containing:

  * a **prompt page** — the exact ``system`` + ``user_template`` prompt the
    model was given, verbatim from the prompt config (rendered once, at the
    top of each file; once at the very top of the ``--combined`` report);
  * a cover page — the two images **in presented order** (left = "Image A",
    right = "Image B", matching how the reasoning text refers to them),
    plus the metadata and the model's final verdict;
  * following pages — the full ``model_reasoning`` thinking trace, wrapped
    and paginated.

Only rows with a non-empty reasoning trace are eligible by default (a thinking
model occasionally spends its whole token budget mid-trace and emits no final
JSON; pass ``--include-empty`` to sample from all rows regardless).

The exact prompt is auto-resolved from the parquet filename
(restaurants / schools / libraries -> the matching
``dagspaces/urbanpairvqa/conf/prompt/*.yaml``); override with ``--prompt-file``.

Examples
--------
Restaurants run (12 pairs)::

    python scripts/sample_reasoning_pdfs.py \\
        multirun/2026-06-04_URBANPAIRVQA/17-28-28/0/outputs/pairwise/restaurants_mvp_20260604_172839.parquet \\
        -n 12 --title "Restaurants — Qwen3.5-9B thinking" \\
        --out reports/pairwise/restaurants_reasoning_pdfs

Schools run, only decisive (non-"Same") verdicts, plus a combined PDF::

    python scripts/sample_reasoning_pdfs.py \\
        multirun/2026-06-04_URBANPAIRVQA/17-28-42/0/outputs/pairwise/schools_mvp_20260604_172848.parquet \\
        -n 12 --decisive-only --combined \\
        --title "Schools — Qwen3.5-9B thinking" \\
        --out reports/pairwise/schools_reasoning_pdfs
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# Verdict labels, ordered from "strongly prefer B" to "strongly prefer A".
LABEL_ORDER = ["MuchLess", "Less", "Same", "More", "MuchMore"]

# Prompt-config dir and the keyword -> prompt-stem map used to auto-resolve the
# exact prompt from a parquet filename.
PROMPT_DIR = Path(__file__).resolve().parents[1] / "dagspaces/urbanpairvqa/conf/prompt"
PROMPT_BY_KEYWORD = {
    "restaurant": "pairwise_restaurant_eat_at_ordinal",
    "school": "pairwise_school_send_child_ordinal",
    "librar": "pairwise_library_maintained_ordinal",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]) if __doc__ else None,
    )
    p.add_argument("output_parquet", type=Path, help="Pairwise VQA stage output parquet.")
    p.add_argument("-n", "--n", type=int, default=12, help="Number of pairs to sample. Default: 12.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for the per-pair PDFs. "
        "Default: <parquet_dir>/<stem>.reasoning_pdfs/",
    )
    p.add_argument("--seed", type=int, default=1234, help="RNG seed for the sample. Default: 1234.")
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Header label printed on every PDF (e.g. dataset + model). "
        "Default: the parquet stem.",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Prompt YAML to embed verbatim (system + user_template). "
        "Default: auto-resolved from the parquet filename. "
        "Pass 'none' to skip the prompt page.",
    )
    p.add_argument(
        "--include-empty",
        action="store_true",
        help="Allow rows with an empty/missing reasoning trace into the sample.",
    )
    p.add_argument(
        "--decisive-only",
        action="store_true",
        help="Only sample pairs whose verdict is not 'Same'.",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Also write a single combined PDF with all sampled pairs.",
    )
    p.add_argument(
        "--max-image-px",
        type=int,
        default=1100,
        help="Downscale images so the longest side is at most this many px. Default: 1100.",
    )
    p.add_argument(
        "--wrap",
        type=int,
        default=100,
        help="Character width to wrap text at. Default: 100.",
    )
    p.add_argument(
        "--lines-per-page",
        type=int,
        default=58,
        help="Reasoning/text lines per page. Default: 58.",
    )
    return p.parse_args()


def _resolve_prompt_file(arg: Path | None, parquet: Path) -> Path | None:
    """Pick the prompt YAML: explicit arg, else keyword-match the parquet name."""
    if arg is not None:
        if str(arg).lower() == "none":
            return None
        if not arg.exists():
            raise SystemExit(f"--prompt-file not found: {arg}")
        return arg
    stem = parquet.stem.lower()
    for keyword, prompt_stem in PROMPT_BY_KEYWORD.items():
        if keyword in stem:
            candidate = PROMPT_DIR / f"{prompt_stem}.yaml"
            if candidate.exists():
                return candidate
    print(f"[WARN] could not auto-resolve a prompt config from '{parquet.stem}'; "
          "no prompt page will be rendered (pass --prompt-file to set one).")
    return None


def _load_prompt(path: Path) -> tuple[str, str]:
    """Return (system, user_template) verbatim from a prompt YAML."""
    cfg = yaml.safe_load(path.read_text()) or {}
    return str(cfg.get("system", "")).rstrip(), str(cfg.get("user_template", "")).rstrip()


def _parse_verdict(raw: str) -> tuple[str | None, str | None]:
    """Best-effort pull (answer_label, explanation) from a model_response JSON blob."""
    if not raw or not str(raw).strip():
        return None, None
    text = str(raw)
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if isinstance(obj, dict):
        label = obj.get("answer")
        expl = obj.get("explanation") or obj.get("reasoning") or obj.get("rationale")
        return (str(label) if label is not None else None,
                str(expl) if expl is not None else None)
    return None, None


def _load_image(path: str | None, max_px: int) -> Image.Image | None:
    if not path:
        return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    long_side = max(img.size)
    if long_side > max_px:
        scale = max_px / long_side
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    return img


def _interpret(label: str | None) -> str:
    """Human-readable gloss of the ordinal label (A relative to B)."""
    gloss = {
        "MuchLess": "strongly prefer Image B",
        "Less": "prefer Image B",
        "Same": "no preference (about equal)",
        "More": "prefer Image A",
        "MuchMore": "strongly prefer Image A",
    }
    return gloss.get(str(label), "—")


def _text_block_pages(pdf: PdfPages, header: str, body: str, wrap: int,
                      lines_per_page: int) -> None:
    """Render an arbitrary text body across as many monospace pages as needed."""
    raw_lines: list[str] = []
    for para in str(body).splitlines():
        if not para.strip():
            raw_lines.append("")
            continue
        raw_lines.extend(textwrap.wrap(para, width=wrap) or [""])
    if not raw_lines:
        raw_lines = ["(empty)"]

    total_pages = (len(raw_lines) + lines_per_page - 1) // lines_per_page
    for pi in range(total_pages):
        chunk = raw_lines[pi * lines_per_page:(pi + 1) * lines_per_page]
        suffix = f"  (p.{pi + 1}/{total_pages})" if total_pages > 1 else ""
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.06, 0.975, f"{header}{suffix}", fontsize=9, fontweight="bold", va="top")
        fig.text(0.06, 0.945, "\n".join(chunk), fontsize=7.5, family="monospace",
                 va="top", ha="left", linespacing=1.25)
        pdf.savefig(fig)
        plt.close(fig)


def _prompt_page(pdf: PdfPages, title: str, system: str, user_template: str,
                 wrap: int, lines_per_page: int) -> None:
    body = (
        "================ SYSTEM ================\n"
        f"{system}\n\n"
        "================ USER TEMPLATE ================\n"
        f"{user_template}\n"
    )
    _text_block_pages(pdf, f"{title}  —  exact prompt", body, wrap, lines_per_page)


def _cover_page(pdf: PdfPages, row: pd.Series, title: str, max_px: int) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)

    left = _load_image(row.get("presented_left_path"), max_px)
    right = _load_image(row.get("presented_right_path"), max_px)

    ax_l = fig.add_axes([0.04, 0.50, 0.44, 0.40])
    ax_r = fig.add_axes([0.52, 0.50, 0.44, 0.40])
    for ax, img, cap, sid in (
        (ax_l, left, "Image A  (presented left)", row.get("sample_id_a")),
        (ax_r, right, "Image B  (presented right)", row.get("sample_id_b")),
    ):
        ax.axis("off")
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "image unavailable", ha="center", va="center",
                    fontsize=10, color="crimson")
        ax.set_title(f"{cap}\n{sid}", fontsize=9)

    label = row.get("presented_label")
    vlabel, expl = _parse_verdict(row.get("model_response"))
    verdict = vlabel or label

    lines = [
        f"pair_id:          {row.get('pair_id')}",
        f"sample A vs B:    {row.get('sample_id_a')}   vs   {row.get('sample_id_b')}",
        f"presented_order:  {row.get('presented_order')}    is_swapped: {row.get('is_swapped')}",
        f"presented_label:  {label}    relative_label: {row.get('relative_label')}",
        "",
        f"VERDICT:  {verdict}   ->   {_interpret(verdict)}",
    ]
    if expl:
        lines += ["", "Model explanation:", textwrap.fill(expl, width=104)]

    fig.text(0.04, 0.46, "\n".join(lines), fontsize=8.5, family="monospace",
             va="top", ha="left", wrap=True)
    fig.text(0.04, 0.015,
             "Exact prompt is on the prompt page (top of this file / report). "
             "Reasoning trace follows on the next page(s).",
             fontsize=7, style="italic", color="dimgray", va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def _render_pair(pdf: PdfPages, row: pd.Series, title: str,
                 args: argparse.Namespace) -> None:
    _cover_page(pdf, row, title, args.max_image_px)
    _text_block_pages(pdf, f"{row.get('pair_id')}  —  reasoning trace",
                      row.get("model_reasoning") or "", args.wrap, args.lines_per_page)


def main() -> None:
    args = _parse_args()
    if not args.output_parquet.exists():
        raise SystemExit(f"Parquet not found: {args.output_parquet}")

    df = pd.read_parquet(args.output_parquet)
    if "model_reasoning" not in df.columns:
        raise SystemExit(f"{args.output_parquet} has no 'model_reasoning' column.")

    title = args.title or args.output_parquet.stem

    prompt_file = _resolve_prompt_file(args.prompt_file, args.output_parquet)
    system = user_template = ""
    if prompt_file is not None:
        system, user_template = _load_prompt(prompt_file)
        print(f"Embedding exact prompt from {prompt_file}")

    def _prompt(pdf: PdfPages) -> None:
        if prompt_file is not None:
            _prompt_page(pdf, title, system, user_template, args.wrap, args.lines_per_page)

    eligible = df
    if not args.include_empty:
        reasoning = df["model_reasoning"].fillna("").astype(str).str.strip()
        eligible = eligible[reasoning.str.len() > 0]
    if args.decisive_only and "presented_label" in eligible.columns:
        eligible = eligible[eligible["presented_label"].astype(str) != "Same"]
    if eligible.empty:
        raise SystemExit("No eligible rows after filtering "
                         "(try --include-empty / drop --decisive-only).")

    k = min(args.n, len(eligible))
    if k < args.n:
        print(f"[WARN] only {k} eligible rows (requested {args.n}).")
    sample = eligible.sample(n=k, random_state=args.seed).reset_index(drop=True)

    out_dir = args.out or args.output_parquet.with_suffix("").with_name(
        f"{args.output_parquet.stem}.reasoning_pdfs"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in sample.iterrows():
        pid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(row.get("pair_id") or f"row{i}"))
        pdf_path = out_dir / f"{i:02d}_{pid}.pdf"
        with PdfPages(pdf_path) as pdf:
            _prompt(pdf)              # exact prompt once, at the top of the file
            _render_pair(pdf, row, title, args)
        print(f"  wrote {pdf_path}")

    if args.combined:
        combined = out_dir / f"_combined_{k}.pdf"
        with PdfPages(combined) as pdf:
            _prompt(pdf)              # exact prompt once, at the very top of the report
            for _, row in sample.iterrows():
                _render_pair(pdf, row, title, args)
        print(f"  wrote {combined}  (combined, {k} pairs)")

    print(f"Done: {k} pair PDF(s) -> {out_dir}")


if __name__ == "__main__":
    main()
