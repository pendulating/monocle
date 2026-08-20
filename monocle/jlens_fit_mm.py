"""Fit a MULTIMODAL Jacobian lens (sub/jacobian-lens) for gemma-4-12B.

Tier-2 companion to monocle.jlens_fit. Where the text-only fitter backprops
through the bare language decoder on WikiText prompts, this variant fits J_l
over MULTIMODAL prompts (cyclomedia street image + a short neutral caption
context), with:

  * SOURCE positions restricted to the image-patch token positions, and
  * TARGET (cotangent-injection) positions from the first patch onward.

That makes J_l literally the patch -> verbalization transport, fitted
on-distribution:

    lens_l(h) = unembed(J_l @ h),  J_l = E[dh_target/dh_l  |  patch positions]

The estimator mechanics are identical to jlens.fitting.jacobian_for_prompt
(one forward replicated dim_batch times, retained graph, ceil(d_model/dim_batch)
backward passes, one-hot cotangents), but the source-average and cotangent
masks are DECOUPLED so the source can be the patch block while the target is
the whole downstream span.

Two load-bearing invariants (wiki concept-jlens-monocle "Gotchas"):
  (a) block outputs come from jlens's ActivationRecorder hooks on
      lens_model.layers, never from HF output_hidden_states.
  (b) from_hf is called with force_bos=False: gemma-4 ships add_bos_token=False
      because the chat template emits <bos> itself; the default True would
      double-BOS every subsequent multimodal build_inputs on the SHARED
      processor tokenizer.

Usage (klara, .venv-nightly + LD_PRELOAD -- see monocle/monocle.sub):
    sbatch monocle/monocle.sub monocle.jlens_fit_mm --smoke
    sbatch monocle/monocle.sub monocle.jlens_fit_mm --shard 0/4
    ... (shards 1/4..3/4) then merge with the text fitter's --merge:
    python -m monocle.jlens_fit --merge \\
        outputs/_monocle/jlens/mm/shard*.pt \\
        --out-dir outputs/_monocle/jlens/mm
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import MODEL_DIR_DEFAULT  # noqa: E402

DEFAULT_IMAGES = str(REPO / "outputs/_monocle/jlens/mm/fit_images.json")
DEFAULT_OUT_DIR = str(REPO / "outputs/_monocle/jlens/mm")
# Sources must all be < target (default final layer, 47). start_graph_at=min
# bounds the retained autograd graph, so nothing below layer 6 is retained.
DEFAULT_SOURCE_LAYERS = [6, 12, 18, 24, 30, 36, 42, 46]

#: Deterministic neutral describe-contexts. The context for image i (by GLOBAL
#: index, before sharding) is DESCRIBE_CONTEXTS[i % 5], so shard slicing never
#: skews the context mix. These come AFTER the patches in the sequence, so they
#: cannot leak into the patch residuals under causal attention -- they only
#: shape the downstream (target) span the cotangent is injected over.
DESCRIBE_CONTEXTS = [
    "Describe this scene.",
    "What do you see in this image?",
    "Give a detailed description of this photograph.",
    "Describe everything visible here.",
    "What is shown in this picture?",
]


def log(msg: str) -> None:
    print(f"[jlens-fit-mm] {msg}", flush=True)


def _load_activation_recorder():
    """Import ActivationRecorder, tolerating both venvs.

    jlens is pip-installed in .venv-nightly but NOT in the plain .venv the CPU
    test-suite runs in. Try the installed import first; on failure put the
    vendored copy on sys.path and retry. This keeps jacobian_for_inputs
    importable and CPU-testable in either environment.
    """
    try:
        from jlens.hooks import ActivationRecorder
    except ModuleNotFoundError:
        sys.path.insert(0, str(REPO / "sub/jacobian-lens"))
        from jlens.hooks import ActivationRecorder
    return ActivationRecorder


def _load_jacobian_lens():
    """Import JacobianLens with the same two-venv fallback as above."""
    try:
        from jlens import JacobianLens
    except ModuleNotFoundError:
        sys.path.insert(0, str(REPO / "sub/jacobian-lens"))
        from jlens import JacobianLens
    return JacobianLens


# ---------------------------------------------------------------------------
# A) Pure core: dependency-injectable, CPU-testable
# ---------------------------------------------------------------------------
def _is_replicable(value: object) -> bool:
    """Whether a value should be replicated along the batch axis.

    Only floating and integer tensors are replicated; bool tensors and
    non-tensor values (e.g. a ``use_cache=False`` flag) pass through unchanged.
    """
    if not torch.is_tensor(value):
        return False
    return value.is_floating_point() or (
        not value.is_floating_point() and value.dtype != torch.bool
    )


def _replicate_inputs(
    inputs: dict, dim_batch: int, *, use_repeat: bool
) -> dict:
    """Replicate every floating/int tensor in ``inputs`` along dim 0 to
    ``dim_batch``.

    Prefers ``expand`` (a zero-copy view); ``use_repeat=True`` materialises a
    contiguous copy instead, for the fallback when a model forward rejects the
    non-contiguous expanded tensors. Non-tensor / bool values pass through.
    """
    out: dict = {}
    for key, value in inputs.items():
        if _is_replicable(value):
            trailing = value.dim() - 1
            if use_repeat:
                out[key] = value.repeat(dim_batch, *([1] * trailing))
            else:
                out[key] = value.expand(dim_batch, *([-1] * trailing))
        else:
            out[key] = value
    return out


def _resolve_layers(
    source_layers: Sequence[int], target_layer: int | None, n_layers: int
) -> tuple[list[int], int]:
    """Resolve/validate source + target layer indices (negatives count from
    the end); enforce sources all < target and in range."""
    target = n_layers - 1 if target_layer is None else target_layer
    if target < 0:
        target += n_layers
    if not 0 <= target < n_layers:
        raise ValueError(
            f"target_layer={target_layer} out of range for {n_layers} layers")
    sources = sorted({l + n_layers if l < 0 else l for l in source_layers})
    if not sources:
        raise ValueError("source_layers is empty")
    if sources[0] < 0 or sources[-1] >= n_layers:
        raise ValueError(
            f"source_layers {sorted(source_layers)} out of range for "
            f"{n_layers} layers")
    if sources[-1] >= target:
        raise ValueError(
            f"source_layers must all be < target_layer={target}; "
            f"got max={sources[-1]}")
    return sources, target


def jacobian_for_inputs(
    model,
    lens_model,
    inputs: dict,
    source_positions: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    source_layers: Sequence[int],
    target_layer: int | None = None,
    dim_batch: int = 8,
) -> dict[int, torch.Tensor]:
    """Per-layer Jacobian estimator ``J_l`` for one multimodal input.

    Identical mechanics to :func:`jlens.fitting.jacobian_for_prompt` -- one
    forward on the inputs replicated ``dim_batch`` times, a retained graph, and
    ``ceil(d_model / dim_batch)`` backward passes each computing ``dim_batch``
    rows of ``J_l`` via one-hot cotangents -- but the source-average positions
    and the cotangent-injection (target) positions are supplied separately so
    the source can be the patch block while the target is the downstream span.

    Args:
        model: Any callable; ``model(**batched_inputs)`` runs a forward. Its
            return value is ignored -- outputs are read only via the recorder
            hooks on ``lens_model.layers``.
        lens_model: Duck-typed; needs ``.layers`` (indexable nn.Modules),
            ``.n_layers``, ``.d_model``.
        inputs: Dict of tensors with batch dim 1 (plus optional non-tensor
            passthroughs). Every floating/int tensor is replicated to
            ``dim_batch`` along dim 0.
        source_positions: 1-D LongTensor of sequence positions to average the
            Jacobian over (the patch block).
        target_mask: 1-D BoolTensor ``[seq_len]`` marking cotangent-injection
            positions.
        source_layers: Layer indices ``l`` to compute ``J_l`` at.
        target_layer: Layer to differentiate. Defaults to the final layer
            (``n_layers - 1``); negative indices count from the end.
        dim_batch: Output dimensions computed per backward pass (the input is
            replicated this many times).

    Returns:
        ``{layer: [d_model, d_model]}`` fp32 CPU tensors.
    """
    ActivationRecorder = _load_activation_recorder()

    n_layers, d_model = lens_model.n_layers, lens_model.d_model
    source_layers, target_layer = _resolve_layers(
        source_layers, target_layer, n_layers)

    source_positions = torch.as_tensor(source_positions).long().flatten()
    if source_positions.numel() == 0:
        raise ValueError("source_positions is empty")
    target_mask = torch.as_tensor(target_mask).bool().flatten()
    if int(target_mask.sum()) == 0:
        raise ValueError("target_mask has no True positions")

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            lens_model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        # One forward on the inputs replicated dim_batch times. Prefer the
        # zero-copy expand; fall back to a contiguous repeat if the model
        # rejects the non-contiguous view. The retained graph is reused for
        # every backward pass below.
        try:
            batched = _replicate_inputs(inputs, dim_batch, use_repeat=False)
            model(**batched)
            target_activation = recorder.activations[target_layer]
        except torch.cuda.OutOfMemoryError:
            # OOM is a RuntimeError subclass, but a contiguous repeat uses MORE
            # memory -- never retry, let smoke calibration see it.
            raise
        except (RuntimeError, ValueError):
            recorder.activations.clear()
            batched = _replicate_inputs(inputs, dim_batch, use_repeat=True)
            model(**batched)
            target_activation = recorder.activations[target_layer]

        source_activations = [
            recorder.activations[layer] for layer in source_layers
        ]

        target_positions = target_mask.nonzero(as_tuple=True)[0].to(
            target_activation.device)
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n = min(dim_batch, d_model - dim_start)
            # One-hot cotangent at output dim (dim_start + b) for batch element
            # b, set at every target position. Yields rows dim_start..+n of J_l.
            cotangent.zero_()
            cotangent[
                batch_indices[:n, None],
                target_positions[None, :],
                dim_start + batch_indices[:n, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                positions = source_positions.to(grad.device, non_blocking=True)
                rows = grad[:n, positions, :].float().mean(dim=1)
                jacobians[layer][dim_start : dim_start + n, :] = rows.cpu()
            del grads

    return jacobians


# ---------------------------------------------------------------------------
# B) Gemma-specific wrappers
# ---------------------------------------------------------------------------
def build_mm_inputs(
    proc,
    tmpl: str,
    image_path: str,
    context: str,
    *,
    image_token_id: int,
    device: str = "cuda:0",
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Build multimodal chat inputs for one image + describe-context.

    Uses the production prompt path (extract.build_inputs, no system message).
    SOURCE positions are the image-patch token positions; the TARGET mask
    covers every position from the first patch onward, excluding the final
    sequence position (mirroring the reference fitter's next-token exclusion --
    the last position has no next-token target).

    Returns ``(inputs, source_positions, target_mask)``.
    """
    from monocle import extract

    image = extract.open_image(image_path)
    inputs = extract.build_inputs(
        proc, tmpl, image, user_text=context, device=device)
    source_positions = extract.image_token_positions(inputs, image_token_id)

    seq_len = int(inputs["input_ids"].shape[1])
    first_patch = int(source_positions[0])
    target_mask = torch.arange(seq_len) >= first_patch
    target_mask[seq_len - 1] = False  # exclude the final position (no target)

    # Match the verified readout path: no KV cache under the grad forward.
    inputs = dict(inputs)
    inputs["use_cache"] = False
    return inputs, source_positions, target_mask


def load_lens_model(model_dir: str):
    """Load gemma-4 via the verified monocle recipe and wrap for jlens.

    force_bos=False is LOAD-BEARING (wiki gotcha (b)): the chat template emits
    its own <bos>, and the default True would double-BOS the shared processor
    tokenizer on every subsequent multimodal build_inputs.

    Returns ``(proc, model, tmpl, lens_model, image_token_id)``.
    """
    try:
        import jlens
    except ModuleNotFoundError:
        sys.path.insert(0, str(REPO / "sub/jacobian-lens"))
        import jlens

    from monocle import extract

    proc, model, tmpl = extract.load_model(model_dir)
    lens_model = jlens.from_hf(model, proc.tokenizer, force_bos=False)
    image_token_id = int(model.config.image_token_id)
    return proc, model, tmpl, lens_model, image_token_id


# ---------------------------------------------------------------------------
# C) CLI
# ---------------------------------------------------------------------------
def parse_shard(spec: str) -> tuple[int, int]:
    i, n = spec.split("/")
    i, n = int(i), int(n)
    if not 0 <= i < n:
        raise ValueError(f"bad shard spec {spec!r}")
    return i, n


def load_images(path: str) -> list[dict]:
    entries = json.load(open(path))
    if not isinstance(entries, list):
        raise ValueError(f"{path} is not a JSON list")
    return entries


def cmd_smoke(args: argparse.Namespace) -> int:
    """Calibrate dim_batch: time one image at each setting, report peak mem."""
    entries = load_images(args.images)
    if not entries:
        log("no images to smoke on")
        return 1
    entry = entries[0]

    proc, model, tmpl, lens_model, image_token_id = load_lens_model(
        args.model_dir)
    inputs, source_positions, target_mask = build_mm_inputs(
        proc, tmpl, entry["path"], DESCRIBE_CONTEXTS[0],
        image_token_id=image_token_id)
    seq_len = int(inputs["input_ids"].shape[1])
    log(f"smoke image={entry.get('recording_id', entry['path'])} "
        f"seq_len={seq_len} n_source={int(source_positions.numel())} "
        f"n_target={int(target_mask.sum())}")

    for dim_batch in (4, 8, 12, 16):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            jacobian_for_inputs(
                model, lens_model, inputs, source_positions, target_mask,
                source_layers=DEFAULT_SOURCE_LAYERS, dim_batch=dim_batch)
        except torch.cuda.OutOfMemoryError:
            log(f"dim_batch={dim_batch}: OOM")
            torch.cuda.empty_cache()
            continue
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        log(f"dim_batch={dim_batch}: {dt:.0f}s/image, peak {peak:.1f} GiB, "
            f"est. {dt * 32 / 3600:.1f} h per 32-image shard")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    import os

    JacobianLens = _load_jacobian_lens()
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stdout)

    entries = load_images(args.images)[: args.n_images]
    shard_i, shard_n = parse_shard(args.shard)
    # Context rotates by GLOBAL index (before sharding); shard slicing keeps
    # each item's global index so the context mix is unskewed.
    indexed = list(enumerate(entries))[shard_i::shard_n]
    log(f"shard {shard_i}/{shard_n}: {len(indexed)} of {len(entries)} images, "
        f"dim_batch={args.dim_batch}, sources={DEFAULT_SOURCE_LAYERS}")

    proc, model, tmpl, lens_model, image_token_id = load_lens_model(
        args.model_dir)
    d_model = lens_model.d_model
    source_layers, target_layer = _resolve_layers(
        DEFAULT_SOURCE_LAYERS, None, lens_model.n_layers)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"fit_ckpt_shard{shard_i}of{shard_n}.pt"

    # Running state: sum of per-image Jacobians, success count, resume index.
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        for key, expected in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("mode", "multimodal"),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint {ckpt} was fitted with {key}={state[key]!r}, "
                    f"not {expected!r}; delete it to refit")
        jacobian_sum = state["jacobian_sum"]
        n_done = state["n_done"]
        next_idx = state["next_idx"]
        log(f"  resuming: {next_idx}/{len(indexed)} images processed")
    else:
        jacobian_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32)
            for layer in source_layers
        }
        n_done = 0
        next_idx = 0

    def write_checkpoint() -> None:
        tmp = f"{ckpt}.tmp.{os.getpid()}"
        torch.save(
            {
                "jacobian_sum": jacobian_sum,
                "n_done": n_done,
                "next_idx": next_idx,
                "source_layers": source_layers,
                "target_layer": target_layer,
                "mode": "multimodal",
            },
            tmp,
        )
        os.replace(tmp, ckpt)

    n_skipped = 0
    for local_idx, (global_idx, entry) in enumerate(indexed):
        if local_idx < next_idx:
            continue
        context = DESCRIBE_CONTEXTS[global_idx % len(DESCRIBE_CONTEXTS)]
        image_id = entry.get("recording_id") or Path(entry["path"]).name

        try:
            inputs, source_positions, target_mask = build_mm_inputs(
                proc, tmpl, entry["path"], context,
                image_token_id=image_token_id)
        except Exception as exc:  # noqa: BLE001 -- unreadable image / build fail
            log(f"  skipping image {local_idx} ({image_id}): {exc}")
            n_skipped += 1
            next_idx = local_idx + 1
            continue

        seq_len = int(inputs["input_ids"].shape[1])
        n_source = int(source_positions.numel())
        n_target = int(target_mask.sum())
        t0 = time.time()
        per_image_J = jacobian_for_inputs(
            model, lens_model, inputs, source_positions, target_mask,
            source_layers=source_layers, target_layer=target_layer,
            dim_batch=args.dim_batch)

        # Convergence diagnostic: relative shift of the running mean, max over
        # source layers (falls ~1/n once settled).
        if n_done > 0:
            mean_rel_change = max(
                (
                    (per_image_J[l] - jacobian_sum[l] / n_done).norm()
                    / ((n_done + 1) * (jacobian_sum[l] / n_done).norm())
                ).item()
                for l in source_layers
            )
        else:
            mean_rel_change = float("nan")

        for layer in source_layers:
            jacobian_sum[layer] += per_image_J[layer]
        n_done += 1
        next_idx = local_idx + 1

        log(f"  image {local_idx + 1}/{len(indexed)} ({image_id})  "
            f"seq_len={seq_len} n_source={n_source} n_target={n_target}  "
            f"{time.time() - t0:.0f}s  max_d_mean={mean_rel_change:.2e}")

        if next_idx % 2 == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no images were processed successfully")
    jacobian_mean = {l: jacobian_sum[l] / n_done for l in source_layers}
    lens = JacobianLens(
        jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)
    out = out_dir / f"shard{shard_i}of{shard_n}.pt"
    lens.save(str(out))
    log(f"saved {out} ({lens!r}); {n_done} fitted, {n_skipped} skipped")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Calibrate dim_batch on one image; no fitting.")
    ap.add_argument("--shard", default="0/1", help="i/n image-slice shard.")
    ap.add_argument("--n-images", type=int, default=128,
                    help="Images from the list to fit over (before sharding).")
    ap.add_argument("--dim-batch", type=int, default=8)
    ap.add_argument("--images", default=DEFAULT_IMAGES,
                    help="JSON list of {path, recording_id, dataset, face}.")
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        return cmd_smoke(args)
    return cmd_fit(args)


if __name__ == "__main__":
    sys.exit(main())
