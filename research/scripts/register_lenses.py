#!/usr/bin/env python
"""The lens registry: 1 record for each fitted Jacobian lens on disk.

Why
---
A lens file is a measurement instrument. Two files with the same name give
different readouts, because the corpus that fitted them is different. Until
now the only record of a corpus was a directory name and a wiki page. This
script builds the record from the files themselves.

For each file it records:

| Field | Source |
|---|---|
| sha256, size, mtime | the file on disk |
| n_prompts, d_model, source_layers | the checkpoint keys |
| save dtype | the dtype of 1 stored Jacobian |
| meta | the `meta` key, if the file carries 1 |
| provenance | a `provenance_shard*.json` beside the file |
| annotation | this script, for the files that predate `meta` |

An old file has no `meta` key. The `annotation` table below holds what we
know about those files from the wiki and from the fit logs. A new fit writes
the same facts into `meta`, so the annotation table does not need to grow.

The script reads the checkpoint header only. It maps the file, so it does not
copy the Jacobians into memory. It needs no GPU.

Run
---
    .venv-3.12/bin/python research/scripts/register_lenses.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path("/share/ju/matt/mllmsci")
JLENS = REPO / "outputs/_monocle/jlens"
OUT_DEFAULT = JLENS / "LENS_REGISTRY.json"
DOC_DEFAULT = REPO / "research/docs/lens-registry.md"
CHUNK = 8 << 20

#: The command that builds this registry again.
REGEN = ".venv-3.12/bin/python research/scripts/register_lenses.py"

#: What we know about the files that predate the `meta` key. The key is the
#: path under the repository root.
ANNOTATIONS: dict[str, dict] = {
    "outputs/_monocle/jlens/gemma4_12b_lens.pt": {
        "corpus": "128 WikiText prompts",
        "estimand": "gemma-4 maps residual content to logits, broad text",
        "target_kind": "span",
        "prompt_kind": "text only, no image",
        "dim_batch": 16,
        "fitted": "July 2026",
        "note": "layers 6..46, stride 6, plus 46",
    },
    # Step 1 of research/docs/lens-refusal-probe-plan.md (2026-09-06).
    "outputs/_monocle/jlens/varA/half0/gemma4_12b_lens.pt": {
        "corpus": "WikiText prompts 0-127 of corpus_wikitext_1k.json",
        "estimand": "the wikitext map again; half 0 of the variance floor",
        "fitted": "2026-09-06",
    },
    "outputs/_monocle/jlens/varA/half1/gemma4_12b_lens.pt": {
        "corpus": "WikiText prompts 128-255 of corpus_wikitext_1k.json",
        "estimand": "the wikitext map again; half 1 of the variance floor",
        "fitted": "2026-09-06",
    },
    "outputs/_monocle/jlens/wikitext1k/gemma4_12b_lens.pt": {
        "corpus": "1,000 WikiText prompts, the reference scale",
        "estimand": "the wikitext map at the paper's scale",
        "fitted": "2026-09-06",
    },
    "outputs/_monocle/jlens/mm_textafter/gemma4_12b_lens.pt": {
        "corpus": "128 cyclomedia images, the 7 single-image case prompts in rotation",
        "estimand": "the text positions after the image map to logits; the question read",
        "fitted": "2026-09-06",
    },
    "outputs/_monocle/jlens/mm_textafter/half_even/gemma4_12b_lens.pt": {
        "corpus": "the even shards of mm_textafter, 64 images",
        "estimand": "the question read again; half 0 of its variance floor",
        "fitted": "2026-09-06",
    },
    "outputs/_monocle/jlens/mm_textafter/half_odd/gemma4_12b_lens.pt": {
        "corpus": "the odd shards of mm_textafter, 64 images",
        "estimand": "the question read again; half 1 of its variance floor",
        "fitted": "2026-09-06",
    },
    "outputs/_monocle/jlens/urban/gemma4_12b_lens.pt": {
        "corpus": "corpus_urban_1k.json",
        "estimand": "the same map, under urban text",
        "target_kind": "span",
        "prompt_kind": "text only, no image",
        "dim_batch": None,
        "fitted": "2026-07-23",
        "note": "the topic control for the WikiText lens",
    },
    "outputs/_monocle/jlens/mm/gemma4_12b_lens.pt": {
        "corpus": "128 cyclomedia images, describe contexts",
        "estimand": "patch content in a street image surfaces as these words",
        "target_kind": "span",
        "prompt_kind": "describe",
        "dim_batch": 8,
        "fitted": "2026-07-23",
        "note": "the lens that judgment_trace reads as `mm`",
    },
    "outputs/_monocle/jlens/mm_all47/gemma4_12b_lens.pt": {
        "corpus": "128 cyclomedia images",
        "estimand": "the same map, at every layer",
        "target_kind": "span",
        "prompt_kind": "describe",
        "dim_batch": 4,
        "fitted": "merged 2026-08-26",
        "note": "47 layers, 16 shards",
    },
    "outputs/_monocle/jlens/ladder/R0/gemma4_12b_lens.pt": {
        "corpus": "128 cyclomedia images, describe contexts",
        "estimand": "the ladder baseline; it copies the `mm` lens",
        "target_kind": "span",
        "prompt_kind": "describe",
        "dim_batch": 8,
        "fitted": "2026-08-24",
        "note": "see research/docs/lens-corpus-faithfulness.md",
    },
    "outputs/_monocle/jlens/ladder/R1/gemma4_12b_lens.pt": {
        "corpus": "128 cyclomedia images, describe contexts",
        "estimand": "the target span axis alone",
        "target_kind": "last",
        "prompt_kind": "describe",
        "dim_batch": 8,
        "fitted": "2026-08-24",
        "note": "see research/docs/lens-corpus-faithfulness.md",
    },
    "outputs/_monocle/jlens/ladder/R2/gemma4_12b_lens.pt": {
        "corpus": "128 cyclomedia images, rating prompt, case subway_safety",
        "estimand": "the prompt text axis; the answer-token rung",
        "target_kind": "last",
        "prompt_kind": "rating",
        "dim_batch": 4,
        "fitted": "merged 2026-08-26",
        "note": "see research/docs/lens-corpus-faithfulness.md",
    },
    "outputs/_monocle/jlens/ladder/R3/gemma4_12b_lens.pt": {
        "corpus": "images from the analysis pool, held out, case subway_safety",
        "estimand": "the image content axis; the faithful lens",
        "target_kind": "last",
        "prompt_kind": "rating",
        "dim_batch": 4,
        "fitted": "not fitted",
        "note": "see research/docs/lens-corpus-faithfulness.md",
    },
    "outputs/_monocle/jlens/lrp/gemma4_12b_lens.pt": {
        "corpus": "128 WikiText prompts",
        "estimand": "the WikiText lens under the LRP rules",
        "target_kind": "span",
        "prompt_kind": "text only, no image",
        "dim_batch": None,
        "fitted": "2026-08-23",
        "note": "rules = lrp; see scale.json beside the file",
    },
    "outputs/_monocle/jlens/mm/lrp/gemma4_12b_lens.pt": {
        "corpus": "128 cyclomedia images, describe contexts",
        "estimand": "the `mm` lens under the LRP rules",
        "target_kind": "span",
        "prompt_kind": "describe",
        "dim_batch": None,
        "fitted": "2026-08-23",
        "note": "rules = lrp; see scale.json beside the file",
    },
}

#: The short name of each merged lens, and its file.
MERGED: list[tuple[str, Path]] = [
    ("wikitext", JLENS / "gemma4_12b_lens.pt"),
    ("urban", JLENS / "urban/gemma4_12b_lens.pt"),
    ("mm", JLENS / "mm/gemma4_12b_lens.pt"),
    ("mm_all47", JLENS / "mm_all47/gemma4_12b_lens.pt"),
    ("ladder_R0", JLENS / "ladder/R0/gemma4_12b_lens.pt"),
    ("ladder_R1", JLENS / "ladder/R1/gemma4_12b_lens.pt"),
    ("ladder_R2", JLENS / "ladder/R2/gemma4_12b_lens.pt"),
    ("ladder_R3", JLENS / "ladder/R3/gemma4_12b_lens.pt"),
    ("lrp_wikitext", JLENS / "lrp/gemma4_12b_lens.pt"),
    ("lrp_mm", JLENS / "mm/lrp/gemma4_12b_lens.pt"),
    # Step 1 of research/docs/lens-refusal-probe-plan.md (2026-09-06).
    ("varA_half0", JLENS / "varA/half0/gemma4_12b_lens.pt"),
    ("varA_half1", JLENS / "varA/half1/gemma4_12b_lens.pt"),
    ("wikitext1k", JLENS / "wikitext1k/gemma4_12b_lens.pt"),
    ("mm_textafter", JLENS / "mm_textafter/gemma4_12b_lens.pt"),
    ("mm_textafter_even", JLENS / "mm_textafter/half_even/gemma4_12b_lens.pt"),
    ("mm_textafter_odd", JLENS / "mm_textafter/half_odd/gemma4_12b_lens.pt"),
]

#: The ladder rungs whose shard parts also go into the registry.
LADDER_RUNGS = ["R0", "R1", "R2", "R3"]


def log(m: str) -> None:
    print(f"[registry] {m}", flush=True)


def sha256_of(path: Path) -> str:
    """Stream the file. A large lens must not go into memory twice."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def header_of(path: Path) -> dict:
    """The checkpoint keys, with no copy of the Jacobians."""
    out: dict = {"readable": True, "error": None}
    try:
        try:
            ck = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True)
        except (RuntimeError, TypeError, ValueError):
            ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - the registry must not stop
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"}

    out["keys"] = sorted(str(k) for k in ck)
    out["is_lens"] = "J" in ck
    out["n_prompts"] = int(ck["n_prompts"]) if "n_prompts" in ck else None
    out["d_model"] = int(ck["d_model"]) if "d_model" in ck else None

    layers = ck.get("source_layers")
    if layers is None and "J" in ck:
        layers = sorted(ck["J"])
    if layers is not None:
        layers = [int(v) for v in layers]
        out["source_layers"] = layers
        out["n_layers"] = len(layers)
        out["layer_min"] = min(layers) if layers else None
        out["layer_max"] = max(layers) if layers else None
    else:
        out["source_layers"] = None
        out["n_layers"] = None
        out["layer_min"] = None
        out["layer_max"] = None

    dtype = None
    if "J" in ck and ck["J"]:
        first = sorted(ck["J"])[0]
        dtype = str(ck["J"][first].dtype).replace("torch.", "")
    out["tensor_dtype"] = dtype
    out["save_dtype"] = (
        str(ck["save_dtype"]).replace("torch.", "")
        if "save_dtype" in ck else None)
    meta = ck.get("meta")
    out["meta"] = meta if isinstance(meta, dict) and meta else None
    out["has_meta"] = bool(out["meta"])
    del ck
    return out


def side_files(path: Path) -> dict:
    """The JSON that a fitter or a merge left beside the lens."""
    out: dict = {}
    merge = path.with_suffix(".merge.json")
    if merge.exists():
        try:
            out["merge"] = json.loads(merge.read_text())
        except json.JSONDecodeError:
            out["merge"] = None
    prov = sorted(path.parent.glob("provenance_shard*.json"))
    if prov:
        try:
            out["provenance"] = json.loads(prov[0].read_text())
            out["provenance_files"] = len(prov)
        except json.JSONDecodeError:
            out["provenance"] = None
    scale = path.parent / "scale.json"
    if scale.exists():
        try:
            out["scale"] = json.loads(scale.read_text())
        except json.JSONDecodeError:
            out["scale"] = None
    return out


def record(name: str, path: Path, kind: str) -> dict:
    rel = str(path.relative_to(REPO))
    stat = path.stat()
    log(f"{name}: hash {stat.st_size / 1e6:.0f} MB")
    digest = sha256_of(path)
    head = header_of(path)
    rec = {
        "name": name,
        "kind": kind,
        "path": rel,
        "sha256": digest,
        "sha256_12": digest[:12],
        "size_bytes": int(stat.st_size),
        "size_mb": round(stat.st_size / 1e6, 1),
        "mtime": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
        "annotation": ANNOTATIONS.get(rel),
    }
    rec.update(head)
    rec.update(side_files(path))
    return rec


def collect() -> tuple[list[dict], list[dict]]:
    """The records, and the named lenses that are not on disk."""
    rows: list[dict] = []
    missing: list[dict] = []
    for name, path in MERGED:
        if path.exists():
            rows.append(record(name, path, "merged"))
        else:
            rel = str(path.relative_to(REPO))
            log(f"{name}: {rel} is not on disk; skip")
            missing.append({
                "name": name, "path": rel,
                "annotation": ANNOTATIONS.get(rel)})
    for rung in LADDER_RUNGS:
        rung_dir = JLENS / "ladder" / rung
        for shard in sorted(rung_dir.glob("shard*.pt")):
            rows.append(record(f"ladder_{rung}_{shard.stem}", shard, "shard"))
    return (rows, missing)


def layers_text(rec: dict) -> str:
    if not rec.get("n_layers"):
        return "unknown"
    return (f"{rec['n_layers']} ({rec['layer_min']}..{rec['layer_max']})")


def corpus_text(rec: dict) -> str:
    meta = rec.get("meta") or {}
    if meta.get("corpus"):
        return str(meta["corpus"])
    ann = rec.get("annotation") or {}
    if ann.get("corpus"):
        return str(ann["corpus"])
    prov = rec.get("provenance") or {}
    if prov.get("rung"):
        return (f"rung {prov['rung']}, case {prov.get('case', '?')}, "
                f"target {prov.get('target_kind', '?')}, "
                f"prompt {prov.get('prompt_kind', '?')}")
    return "unknown"


def estimand_text(rec: dict) -> str:
    meta = rec.get("meta") or {}
    if meta.get("estimand"):
        return str(meta["estimand"])
    ann = rec.get("annotation") or {}
    return str(ann.get("estimand", "unknown"))


def write_doc(
    rows: list[dict], missing: list[dict], doc: Path, out: Path,
) -> None:
    merged = [r for r in rows if r["kind"] == "merged"]
    shards = [r for r in rows if r["kind"] == "shard"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    no_meta = [r for r in merged if not r.get("has_meta")]

    lines: list[str] = []
    lines.append("# Lens registry")
    lines.append("")
    lines.append(f"Generated: {now}. Do not edit this page by hand.")
    lines.append("")
    lines.append("```bash")
    lines.append(REGEN)
    lines.append("```")
    lines.append("")
    lines.append("## What this page is")
    lines.append("")
    lines.append(
        "A lens file is an instrument. Its readout depends on the corpus "
        "that fitted it. This page names each lens on disk, and gives the "
        "hash that identifies it.")
    lines.append("")
    lines.append(
        f"The machine-readable copy is `{out.relative_to(REPO)}`. It carries "
        "the full hash, the layer list, and the `meta` dict.")
    lines.append("")
    lines.append("## The lenses")
    lines.append("")
    lines.append(
        "| Name | Path | sha256 | Prompts or images | Layers | Dtype | "
        "Corpus | Estimand |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in merged:
        dtype = r.get("tensor_dtype") or "unknown"
        n_p = r.get("n_prompts")
        lines.append(
            f"| `{r['name']}` | `{r['path']}` | `{r['sha256_12']}` | "
            f"{n_p if n_p is not None else 'unknown'} | {layers_text(r)} | "
            f"{dtype} | {corpus_text(r)} | {estimand_text(r)} |")
    lines.append("")
    if missing:
        lines.append("## A named lens that is not on disk")
        lines.append("")
        lines.append(
            "The wiki names these lenses, but no file holds them. Do not "
            "cite a result from one of them.")
        lines.append("")
        lines.append("| Name | Path | Estimand |")
        lines.append("|---|---|---|")
        for r in missing:
            ann = r.get("annotation") or {}
            lines.append(
                f"| `{r['name']}` | `{r['path']}` | "
                f"{ann.get('estimand', 'unknown')} |")
        lines.append("")
    lines.append("## The ladder shard parts")
    lines.append("")
    lines.append(
        "A merge reads these. Do not read them as a lens. They are here so "
        "that a merge is reproducible.")
    lines.append("")
    lines.append("| Name | sha256 | Prompts | Layers | Dtype | Rung |")
    lines.append("|---|---|---|---|---|---|")
    for r in shards:
        prov = r.get("provenance") or {}
        lines.append(
            f"| `{r['name']}` | `{r['sha256_12']}` | "
            f"{r.get('n_prompts', 'unknown')} | {layers_text(r)} | "
            f"{r.get('tensor_dtype') or 'unknown'} | "
            f"{prov.get('rung', 'unknown')} |")
    lines.append("")
    lines.append("## What we do not know")
    lines.append("")
    lines.append(
        "A file that a fit wrote before review item C7 has no `meta` key. "
        "For such a file the table above shows an annotation that this "
        "script holds, not a fact that the file carries. These items stay "
        "unknown:")
    lines.append("")
    lines.append("| Item | Why it matters |")
    lines.append("|---|---|")
    lines.append(
        "| The corpus, in the file | A second copy of the file cannot be "
        "traced back to its fit. |")
    lines.append(
        "| `dim_batch` | The batch changes the numerics. Two fits at "
        "different batches are not the same lens. |")
    lines.append(
        "| The fit jobs | We cannot name the SLURM jobs, the code commit, or "
        "the model revision. |")
    lines.append("")
    if no_meta:
        lines.append("Files with no `meta` key:")
        lines.append("")
        for r in no_meta:
            lines.append(f"- `{r['path']}`")
        lines.append("")
    lines.append(
        "A new fit writes the corpus, the estimator, `dim_batch`, and the "
        "date into the file's `meta` dict (review item C7). Read `meta` "
        "first. Use the annotation only when `meta` is absent.")
    lines.append("")
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(lines))


def print_table(rows: list[dict]) -> None:
    log("")
    log(f"{'name':>20} {'sha256':>14} {'prompts':>8} {'layers':>10} "
        f"{'dtype':>9} {'MB':>7} {'meta':>5}")
    for r in rows:
        log(f"{r['name']:>20} {r['sha256_12']:>14} "
            f"{str(r.get('n_prompts')):>8} {layers_text(r):>10} "
            f"{str(r.get('tensor_dtype')):>9} {r['size_mb']:>7.0f} "
            f"{'yes' if r.get('has_meta') else 'no':>5}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--doc", type=Path, default=DOC_DEFAULT)
    args = ap.parse_args()

    rows, missing = collect()
    print_table(rows)

    payload = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": str(REPO),
        "regenerate": REGEN,
        "n_files": len(rows),
        "lenses": rows,
        "not_on_disk": missing,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    write_doc(rows, missing, args.doc, args.out)
    log("")
    log(f"wrote {args.out}")
    log(f"wrote {args.doc}")


if __name__ == "__main__":
    main()
