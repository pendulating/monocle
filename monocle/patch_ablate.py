"""Ablate a patch at the INPUT, with the model's own marker for `no patch`.

Why this exists
---------------
`patch_map_causal` measures a patch by a DONOR SWAP at layer 20: it replaces
the patch's residual with the same cell of another photograph. That reference
has 2 problems, and both showed up in the numbers.

1. The value is relative to the donors, not to the patch being absent. The map
   answers "what if this cell were THAT cell", so a reader who takes it as
   "how dangerous is this cell" reads something we never measured.
2. Layer 20 is not local. Gemma-4 gives the image block a BIDIRECTIONAL mask,
   so from block 0 every patch can read every other patch. After 20 blocks the
   vector at 1 cell carries a share of the whole scene. That is why 1 swap can
   move `c` by 3.5 logits, and why 256 of them sum to 10x `c`.

The input does not have either problem.

The 2 stages of the image path
------------------------------
| Stage | Attention | Result |
|---|---|---|
| pixels to vectors | none at all | a soft token reads its own 48 px and its (x, y) |
| decoder blocks 0-47 | yes, bidirectional over the image | the patches mix |

Gemma-4 is encoder-free: the vision config holds no layers and no heads, only
`LayerNorm -> Linear -> LayerNorm -> + posemb -> RMSNorm -> Linear`. Thus a
patch is still only itself at the input, and nowhere after it.

The model's own placeholder
---------------------------
`image_position_ids == (-1, -1)` marks a slot that holds no patch. The
processor pads every image this way, so our 256-patch photograph already
carries 24 such slots. `get_image_features` strips them before it scatters the
features into the text sequence:

    padding_mask = (image_position_ids == -1).all(dim=-1)
    vision_outputs = vision_outputs[~padding_mask]

So a patch goes away in 2 edits that must agree:

1. mark its slot `(-1, -1)`, which removes 1 feature;
2. remove its `<|image|>` token, which removes 1 slot in the text sequence.

The model then reads a picture of 255 patches. The position embedding is
factorized per patch, so the survivors keep their true coordinates: the model
sees a hole at (3, 7), not a shifted grid.

What NOT to use
---------------
| Candidate | Why not |
|---|---|
| `<pad>`, `<unk>`, `<unusedN>` | no training example puts a text token between image patches |
| a zero or mean vector at L20 | there is no token at L20, and Gemma carries massive activations there |
| a grey square | `patch_ln1` is a LayerNorm, so EVERY flat colour gives the same soft token. It is still content, not absence |

Sign
----
`delta = c_ablated - c0`. The map is `-delta`, the CONTRIBUTION, which is the
same convention `causal_maps` draws. A patch that holds up the positive label
makes `c` fall when it goes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import patch_attribution as pa  # noqa: E402
from monocle import patch_causal as pc  # noqa: E402
from monocle import patch_map_causal as pmc  # noqa: E402
from monocle import safety_workspace as sw  # noqa: E402
from monocle import single as sg  # noqa: E402

#: Tensors with 1 entry for each token of the text sequence.
SEQ_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids",
            "token_type_ids", "position_ids")
#: Tensors with 1 entry for each soft-token slot of the image.
SLOT_KEYS = ("pixel_values", "image_position_ids")
#: The `none` arm must reproduce `c` to this much. Above it, the rebuild is
#: wrong, and no map from it is readable. This is a MACHINERY test.
NULL_TOL = 1e-3
#: An image whose whole-image ablation moves `c` less than this cannot carry a
#: parts-against-whole ratio, because the denominator is noise. This is a
#: property of the IMAGE, not a fault, so it flags 1 image and never stops a
#: battery. Some photographs really do leave the judgment where the prompt
#: put it.
WORTH_FLOOR = 2.0
#: But if most images fail that test, the ablation itself is dead. A battery
#: needs this share of its images above the floor.
WORTH_SHARE = 0.5


def log(m: str) -> None:
    print(f"[ablate] {m}", flush=True)


# ---------------------------------------------------------------------------
# Building the ablated inputs
# ---------------------------------------------------------------------------
def split_keys(inputs: dict) -> tuple[list[str], list[str], list[str]]:
    """(sequence tensors, slot tensors, everything else).

    An unknown tensor raises. A new key in the processor output would
    otherwise pass through unedited and break the alignment in silence.
    """
    seq, slot, other = [], [], []
    for k, v in inputs.items():
        if not torch.is_tensor(v):
            other.append(k)
        elif k in SEQ_KEYS:
            seq.append(k)
        elif k in SLOT_KEYS:
            slot.append(k)
        else:
            raise ValueError(
                f"unknown tensor {k!r} in the model inputs. Add it to "
                "SEQ_KEYS or SLOT_KEYS, or the ablation will not edit it.")
    if "input_ids" not in seq or "image_position_ids" not in slot:
        raise ValueError(
            "the inputs need input_ids and image_position_ids; "
            f"got {sorted(inputs)}")
    return seq, slot, other


def ablate_row(inputs: dict, slots: list[int], patches) -> dict:
    """1 row with `patches` removed. `inputs` must hold exactly 1 row.

    `slots[i]` is the sequence position of the `<|image|>` token of patch `i`.
    """
    if int(inputs["input_ids"].shape[0]) != 1:
        raise ValueError(
            f"ablate_row needs 1 row, got {inputs['input_ids'].shape[0]}")
    drop = sorted({int(p) for p in patches})
    bad = [p for p in drop if p < 0 or p >= len(slots)]
    if bad:
        raise ValueError(f"patch {bad} is outside 0..{len(slots) - 1}")

    seq_keys, slot_keys, other = split_keys(inputs)
    length = int(inputs["input_ids"].shape[1])
    keep = torch.ones(length, dtype=torch.bool)
    for p in drop:
        keep[slots[p]] = False

    out: dict[str, Any] = {k: inputs[k] for k in other}
    for k in seq_keys:
        out[k] = inputs[k][:, keep.to(inputs[k].device)]
    for k in slot_keys:
        out[k] = inputs[k].clone()
    if drop:
        idx = torch.tensor(drop, device=out["image_position_ids"].device)
        out["image_position_ids"][0, idx] = -1
    return out


def substitute_row(
    inputs: dict, donor_pixels: torch.Tensor, patches,
) -> dict:
    """1 row where `patches` hold the DONOR's pixels. Nothing is removed.

    An ablation takes a square away and leaves a hole. This puts other content
    in the square instead. The 2 answer different questions:

    | Edit | Asks |
    |---|---|
    | ablate | what does the judgment lose when this square goes |
    | substitute | what does the judgment lose when this square is other content |

    A map whose ranking works under BOTH is a map of content. A map that
    works only under the ablation ranks holes.

    The edit is 1 line, and no token count changes, because the slot stays a
    slot. The vision path holds no attention, so a substituted row reaches the
    decoder as the donor's own patch, with the TARGET's coordinates.
    """
    drop = sorted({int(p) for p in patches})
    n_slots = int(inputs["pixel_values"].shape[1])
    bad = [p for p in drop if p < 0 or p >= n_slots]
    if bad:
        raise ValueError(f"patch {bad} is outside 0..{n_slots - 1}")
    if donor_pixels.shape[-1] != inputs["pixel_values"].shape[-1]:
        raise ValueError(
            f"the donor rows are {donor_pixels.shape[-1]} wide and the target "
            f"rows are {inputs['pixel_values'].shape[-1]} wide")
    out = {k: (v.clone() if torch.is_tensor(v) else v)
           for k, v in inputs.items()}
    if drop:
        idx = torch.tensor(drop, device=out["pixel_values"].device)
        out["pixel_values"][0, idx, :] = donor_pixels[idx, :].to(
            out["pixel_values"].dtype).to(out["pixel_values"].device)
    return out


def substitution_batch(
    inputs: dict, donor_pixels: torch.Tensor, groups: list,
    image_token_id: int,
) -> dict:
    """1 row for each group. Unlike an ablation, sizes may differ freely."""
    if not groups:
        raise ValueError("no groups to substitute")
    rows = [substitute_row(inputs, donor_pixels, g) for g in groups]
    for r in rows:
        check_alignment(r, image_token_id)
    seq_keys, slot_keys, other = split_keys(inputs)
    out: dict[str, Any] = {k: inputs[k] for k in other}
    for k in seq_keys + slot_keys:
        out[k] = torch.cat([r[k] for r in rows], dim=0)
    return out


def stack_rows(rows: list) -> dict:
    """Stack single-row inputs built from DIFFERENT photographs.

    `ablation_batch` stacks several edits of ONE photograph. This stacks one
    edit of each of several photographs, which is what lets a generation run
    over many images at once.

    Every row must already have the same sequence length and the same slot
    count. Padding is refused rather than added: a pad inside a bidirectional
    image block is the very thing this module exists to avoid. Rows that keep
    the same NUMBER of cells always agree in length, so the caller groups by
    that count.
    """
    if not rows:
        raise ValueError("no rows to stack")
    lengths = {int(r["input_ids"].shape[1]) for r in rows}
    if len(lengths) != 1:
        raise ValueError(
            f"rows are {sorted(lengths)} tokens long. Stacking would need a "
            "pad, and a pad inside the image block is never safe. Group the "
            "rows by how many cells they keep.")
    slots = {int((r["image_position_ids"] != -1).all(-1).sum()) for r in rows}
    if len(slots) != 1:
        raise ValueError(f"rows hold {sorted(slots)} unpadded patches")
    if any(int(r["input_ids"].shape[0]) != 1 for r in rows):
        raise ValueError("stack_rows takes single-row inputs")

    seq_keys, slot_keys, other = split_keys(rows[0])
    out = {k: rows[0][k] for k in other}
    for k in seq_keys + slot_keys:
        out[k] = torch.cat([r[k] for r in rows], dim=0)
    return out


def check_alignment(row: dict, image_token_id: int) -> int:
    """The feature count must equal the slot count. Returns the count."""
    n_tokens = int((row["input_ids"] == image_token_id).sum())
    n_valid = int((row["image_position_ids"] != -1).all(-1).sum())
    if n_tokens != n_valid:
        raise ValueError(
            f"{n_tokens} image tokens against {n_valid} unpadded patches. "
            "The 2 edits of an ablation must always agree.")
    return n_tokens


def ablation_batch(
    inputs: dict, slots: list[int], groups: list, image_token_id: int,
) -> dict:
    """1 row for each group. Every group must drop the same patch count.

    Rows of different lengths would need padding, and a pad inside the image
    block would be the very thing this module avoids. So the caller batches
    groups of one size.
    """
    if not groups:
        raise ValueError("no groups to ablate")
    sizes = {len({int(p) for p in g}) for g in groups}
    if len(sizes) != 1:
        raise ValueError(
            f"a batch needs 1 group size, got {sorted(sizes)}. Batch the "
            "groups by size.")
    rows = [ablate_row(inputs, slots, g) for g in groups]
    for r in rows:
        check_alignment(r, image_token_id)

    seq_keys, slot_keys, other = split_keys(inputs)
    out: dict[str, Any] = {k: inputs[k] for k in other}
    for k in seq_keys + slot_keys:
        out[k] = torch.cat([r[k] for r in rows], dim=0)
    return out


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------
def contrast(
    model: Any, lens_model: Any, inputs: dict, pos_id: int, neg_id: int,
) -> np.ndarray:
    """`c` for every row, in float32."""
    with torch.no_grad():
        h = pc.final_residual(model, lens_model, inputs, graph=False)
    return pc.fp32_contrast(lens_model, h, pos_id, neg_id).float().cpu().numpy()


def ablation_map(
    model: Any, lens_model: Any, inputs: dict, slots: list[int],
    c0: float, pos_id: int, neg_id: int, *,
    batch: int = 8, group: int = 1, image_token_id: int,
) -> np.ndarray:
    """`c` change when each patch, or each `group` x `group` block, goes away.

    Returns [n_patches] when `group` is 1, and [n_patches / group**2] when it
    is more, in row-major order.

    **Every chunk is the same size, and the zero is measured at that size.**
    Measured on gemma-4-12b, the same input with NO intervention gives a `c`
    that moves by up to 0.67 logits when the batch size changes, while 16
    identical rows INSIDE one batch agree to the last bit. So the last chunk
    of a 64-cell map, being short, would carry a different offset from the
    rest, and the map would hold a step at the chunk boundary.

    A no-op row cannot simply ride along: an ablation SHORTENS the sequence,
    so a no-op and an ablated row cannot sit in one batch. Instead every chunk
    is padded to exactly `batch` rows, and the zero is measured once in a
    batch of `batch` identical no-op rows.
    """
    groups = (pmc.patch_groups(len(slots), group) if group > 1
              else [[i] for i in range(len(slots))])
    zero = float(contrast(
        model, lens_model,
        ablation_batch(inputs, slots, [[]] * batch, image_token_id),
        pos_id, neg_id)[0])
    deltas = np.zeros(len(groups), dtype=np.float64)
    for start in range(0, len(groups), batch):
        chunk = list(groups[start:start + batch])
        n = len(chunk)
        while len(chunk) < batch:          # pad, so every batch is one size
            chunk.append(chunk[-1])
        rows = ablation_batch(inputs, slots, chunk, image_token_id)
        cs = contrast(model, lens_model, rows, pos_id, neg_id)
        deltas[start:start + n] = cs[:n] - zero
    return deltas


def substitution_map(
    model: Any, lens_model: Any, inputs: dict, slots: list[int],
    donor_pixels: torch.Tensor, c0: float, pos_id: int, neg_id: int, *,
    batch: int = 8, group: int = 1, image_token_id: int,
) -> np.ndarray:
    """`c` change when each cell holds the DONOR's pixels instead of its own.

    The twin of `ablation_map`. That one asks what the judgment loses when a
    square goes; this one asks what it loses when the square is other content.
    A cell where the 2 agree is a cell whose CONTENT carries the judgment.
    """
    groups = (pmc.patch_groups(len(slots), group) if group > 1
              else [[i] for i in range(len(slots))])
    # Same batch-size discipline as `ablation_map`. A substitution does not
    # change the sequence length, so a no-op row COULD ride along, but the
    # zero is measured the same way here so the 2 maps stay comparable.
    zero = float(contrast(
        model, lens_model,
        substitution_batch(inputs, donor_pixels, [[]] * batch,
                           image_token_id),
        pos_id, neg_id)[0])
    deltas = np.zeros(len(groups), dtype=np.float64)
    for start in range(0, len(groups), batch):
        chunk = list(groups[start:start + batch])
        n = len(chunk)
        while len(chunk) < batch:
            chunk.append(chunk[-1])
        rows = substitution_batch(inputs, donor_pixels, chunk, image_token_id)
        cs = contrast(model, lens_model, rows, pos_id, neg_id)
        deltas[start:start + n] = cs[:n] - zero
    return deltas


def controls(
    model: Any, lens_model: Any, inputs: dict, slots: list[int],
    c0: float, pos_id: int, neg_id: int, *, image_token_id: int,
    batch: int = 1,
) -> dict:
    """3 arms. 2 of them gate a map, and the third records the batch offset.

    | Arm | Batch | Meaning | Pass |
    |---|---|---|---|
    | `none` | 1 | the rebuild changes nothing | `c` comes back |
    | `zero` | `batch` | the map's own reference | recorded, not gated |
    | `all` | `batch` | the picture goes away | `c` moves a lot |

    The `none` arm is not a no-op: it runs the same rebuild, so it catches a
    wrong index, a lost tensor, or a bad concatenation. It stays at batch 1
    because it is compared against `c0`, which the caller reads at batch 1.

    **The whole must share the map's zero.** `ablation_map` measures every
    cell against a no-op row in a batch of `batch`. In bf16 the same row gives
    a `c` that moves by up to 0.67 logits when the batch size changes, so a
    whole measured at batch 1 sits on a different reference from the parts.
    The sum of 256 cells then carries 256 copies of that offset, and
    `parts_over_whole` compares 2 quantities that were never on one scale.
    Thus `image_worth` is `zero - c_all`, both at `batch`.

    `batch_offset` reports the size of the effect for the record.
    """
    none_row = ablation_batch(inputs, slots, [[]], image_token_id)
    c_none = float(contrast(model, lens_model, none_row, pos_id, neg_id)[0])
    zero_rows = ablation_batch(inputs, slots, [[]] * batch, image_token_id)
    zero = float(contrast(model, lens_model, zero_rows, pos_id, neg_id)[0])
    whole = [list(range(len(slots)))] * batch
    all_rows = ablation_batch(inputs, slots, whole, image_token_id)
    c_all = float(contrast(model, lens_model, all_rows, pos_id, neg_id)[0])
    return {
        "c0": c0, "c_none": c_none, "c_all": c_all, "zero": zero,
        "batch": batch, "batch_offset": zero - c_none,
        "null_error": abs(c_none - c0), "image_worth": zero - c_all,
        "null_ok": abs(c_none - c0) <= NULL_TOL,
        "worth_ok": abs(zero - c_all) >= WORTH_FLOOR,
    }


def gate_rebuild(ctl: dict) -> None:
    """Stop when the REBUILD is wrong. This is never a property of an image."""
    if not ctl["null_ok"]:
        raise SystemExit(
            f"the null arm moved `c` by {ctl['null_error']:.4f}, over "
            f"{NULL_TOL}. The rebuild changes the prompt, so no map from it "
            "is readable.")


def gate_battery(worths: list[float]) -> None:
    """Stop when MOST images are worth nothing. 1 weak image is data.

    A single photograph that the model barely reads is a finding, and the
    battery keeps it with `worth_ok` false. Every photograph reading near 0
    means the ablation does not reach the model.
    """
    if not worths:
        raise SystemExit("no image was measured")
    ok = sum(1 for w in worths if abs(w) >= WORTH_FLOOR)
    share = ok / len(worths)
    if share < WORTH_SHARE:
        raise SystemExit(
            f"only {ok} of {len(worths)} images move `c` by {WORTH_FLOOR} "
            "when the whole picture goes. The ablation does not reach the "
            "model, or the prompt does not read the picture.")


def slots_of(inputs: dict, image_token_id: int) -> list[int]:
    """The sequence position of each patch, in patch order."""
    return [int(x) for x in
            sw.image_token_blocks(inputs, image_token_id, expect=1)[0]]


# ---------------------------------------------------------------------------
# CLI: prove the path on 1 image before a battery runs
# ---------------------------------------------------------------------------
def smoke(case: str, model_dir: str, device: str, group: int) -> int:
    from PIL import Image

    from monocle import attn_route as ar
    from monocle import jlens_read

    proc, model, tmpl = ar.load_eager(model_dir, device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    pos_id, neg_id = sg.contrast_ids(proc.tokenizer, case)
    image_token_id = int(model.config.image_token_id)

    row = sg.load_single_rows(1, case)[0]
    img = Image.open(row["path"]).convert("RGB")
    inputs = sg.build_single_inputs(proc, tmpl, img, case, device=device)
    slots = slots_of(inputs, image_token_id)
    log(f"{row['image_id']}: {len(slots)} patches, "
        f"{int(inputs['input_ids'].shape[1])} tokens")

    c0 = float(contrast(model, lens_model, inputs, pos_id, neg_id)[0])
    ctl = controls(model, lens_model, inputs, slots, c0, pos_id, neg_id,
                   image_token_id=image_token_id)
    log(f"controls: c0={ctl['c0']:+.4f} none={ctl['c_none']:+.4f} "
        f"(error {ctl['null_error']:.5f}) all={ctl['c_all']:+.4f} "
        f"(the image is worth {ctl['image_worth']:+.3f})")
    gate_rebuild(ctl)
    if not ctl["worth_ok"]:
        log(f"NOTE: this image is worth under {WORTH_FLOOR}, so a "
            "parts-against-whole ratio from it means little")

    m = ablation_map(model, lens_model, inputs, slots, c0, pos_id, neg_id,
                     batch=8, group=group, image_token_id=image_token_id)
    contrib = -m
    log(f"map: {contrib.size} cells, max |contribution| "
        f"{np.abs(contrib).max():.3f}, sum {contrib.sum():+.3f}, "
        f"the whole image is worth {ctl['image_worth']:+.3f}")
    log(f"sum of the parts / the whole = "
        f"{contrib.sum() / ctl['image_worth']:+.3f}")
    print(json.dumps({k: v for k, v in ctl.items()}, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Input-level patch ablation.")
    ap.add_argument("--case", default="subway_safety",
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return smoke(args.case, args.model_dir, args.device, args.group)


if __name__ == "__main__":
    raise SystemExit(main())
