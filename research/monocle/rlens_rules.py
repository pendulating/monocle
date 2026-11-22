"""LRP rules for the lens backward pass — the R-lens.

What this changes, and what it does not
---------------------------------------
The J-lens transports a residual with an averaged gradient,
`J_l = E[dh_target/dh_l]`. This module does not touch that fit. It changes
only what flows BACKWARD through gemma-4 while the fit runs, so a raw
gradient gives way to a Layer-wise Relevance Propagation coefficient.

The reason is in `research/docs/r-lens-proposal.md`. Our J-lens is
unreadable below L30, and the image stops acting on the answer above L30, so
the readable window and the causal window never meet.

The 4 rules
-----------
| Rule | Site | What the backward does |
|---|---|---|
| norm | RMSNorm | the denominator becomes a constant |
| identity | GELU | the slope becomes y/x, not dy/dx |
| half | the GeGLU product | each branch gets half, with no double count |
| softmax | attention | relevance flows through V alone |

Every forward value stays bit-identical, and each rule proves it:

  * norm — `detach` never changes a value.
  * identity — the value comes from `y.detach()`, and the term that carries
    the new slope is exactly 0.
  * half — `0.5 * (a + a)` is exact in IEEE arithmetic.
  * softmax — `detach` never changes a value.

Thus `check()` can demand `max|logits_lrp - logits_plain| == 0`, and a
failure is a real fault and not a rounding error.

Why the softmax rule needs no half rule
---------------------------------------
AttnLRP splits relevance at both matmuls of an attention head. Once the
softmax is a constant, the query-key matmul carries no relevance at all, and
the remaining matmul has 1 differentiable input. A half rule there is a
factor of 0.5 on every head, which a lens readout does not see. So the
constant-softmax rule alone is the whole attention rule.

Warning: the softmax rule needs EAGER attention. SDPA computes the softmax
inside a fused kernel that never calls `torch.nn.functional.softmax`, so the
patch would do nothing and say nothing. `lrp_rules` refuses a model that is
not eager.

Usage:
    from research.monocle import rlens_rules as rr
    with rr.lrp_rules(model):
        ...                       # the fit, unchanged

    sbatch research/monocle/monocle.sub monocle.rlens_rules --check
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from torch import nn

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Below this the ratio y/x is not a usable slope, so the rule falls back to
#: the slope of the activation at 0.
RATIO_FLOOR = 1e-6

#: The step that measures the slope of an activation at 0.
SLOPE_PROBE = 1e-3

#: Class-name suffixes that name the 2 sites a rule patches.
NORM_SUFFIX = "RMSNorm"
MLP_SUFFIX = "MLP"

#: These compute the softmax inside a fused kernel, so the patch cannot see
#: it. A model that names one of them must be reloaded eager.
FUSED_ATTENTION = frozenset(
    {"sdpa", "flash_attention_2", "flash_attention_3", "flex_attention"})


def log(m: str) -> None:
    print(f"[rlens] {m}", flush=True)


# ---------------------------------------------------------------------------
# The rules, as pure tensor functions
# ---------------------------------------------------------------------------
def straight_through(
    x: torch.Tensor, y: torch.Tensor, slope: torch.Tensor,
) -> torch.Tensor:
    """`y` in the forward, `slope * dx` in the backward.

    `x - x.detach()` is exactly 0, so the value is `y` to the last bit. The
    same term carries the whole gradient, because it is the only part of the
    expression that autograd can see.
    """
    return y.detach() + (x - x.detach()) * slope


def identity_slope(
    x: torch.Tensor, y: torch.Tensor, zero: float = 1.0,
    floor: float = RATIO_FLOOR,
) -> torch.Tensor:
    """The LRP identity-rule slope y/x, with a guard where x is near 0.

    The guard matters because y/x is 0/0 at the origin. The limit there is
    the slope of the activation at 0, which `slope_at_zero` measures once.
    """
    big = x.abs() > floor
    denom = torch.where(big, x, torch.ones_like(x))
    return torch.where(
        big, y / denom, torch.full_like(x, zero)).detach()


def half_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a * b`, with half of the relevance to each branch.

    A plain product sends `b` to `a` and `a` to `b`, which counts the same
    relevance twice. The half rule sends half to each.
    """
    return 0.5 * (a * b.detach() + a.detach() * b)


def slope_at_zero(act_fn: Any, probe: float = SLOPE_PROBE) -> float:
    """A central difference across 0. GELU gives 0.5, and ReLU gives 0.5."""
    x = torch.tensor([probe, -probe], dtype=torch.float32)
    y = act_fn(x).double()
    return float((y[0] - y[1]) / (2.0 * probe))


def lrp_norm(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    """`Gemma4UnifiedRMSNorm._norm` with a constant denominator.

    The body repeats the original line for line, and adds `detach`. The
    installer checks that against the real class, so a change to gemma-4 in a
    later transformers release fails loudly.
    """
    mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
    return hidden_states * torch.pow(mean_squared, -0.5).detach()


def make_mlp_forward(*, identity: bool, half: bool, zero: float):
    """A GeGLU forward that carries the identity rule and the half rule."""

    def forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        acted = self.act_fn(gate)
        if identity:
            acted = straight_through(
                gate, acted, identity_slope(gate, acted, zero))
        up = self.up_proj(x)
        product = half_product(acted, up) if half else acted * up
        return self.down_proj(product)

    return forward


def make_softmax(original, report: Optional[dict] = None):
    """A softmax whose output is a constant to the backward pass.

    It counts its own calls. A config check can miss a fused kernel, but a
    count of 0 after a forward pass cannot: it says the rule never ran.
    """

    def softmax(*args, **kwargs):
        if report is not None:
            report["softmax_calls"] += 1
        return original(*args, **kwargs).detach()

    return softmax


# ---------------------------------------------------------------------------
# Where the rules go
# ---------------------------------------------------------------------------
def classes_with_suffix(model: nn.Module, suffix: str) -> list[type]:
    """Every distinct class in the model whose name ends with `suffix`."""
    seen: dict[type, None] = {}
    for module in model.modules():
        kind = type(module)
        if kind.__name__.endswith(suffix):
            seen.setdefault(kind, None)
    return list(seen)


def one_instance(model: nn.Module, kind: type) -> nn.Module:
    """The first module of that exact class. Used for the install check."""
    for module in model.modules():
        if type(module) is kind:
            return module
    raise LookupError(f"no instance of {kind.__name__} in the model")


def attention_kinds(config: Any) -> set[str]:
    """Every attention implementation the config names, at any depth."""
    found: set[str] = set()
    value = getattr(config, "_attn_implementation", None)
    if isinstance(value, str):
        found.add(value)
    elif isinstance(value, dict):
        found.update(str(v) for v in value.values())
    for name in ("text_config", "vision_config", "audio_config"):
        sub = getattr(config, name, None)
        if sub is not None and sub is not config:
            found |= attention_kinds(sub)
    return found


def _probe(
    width: int, n: int = 4, like: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """A small input for an install check. `like` fixes device and dtype."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(1, n, width, generator=generator, dtype=torch.float32)
    if like is None:
        return x
    return x.to(device=like.device, dtype=like.dtype)


def _check_exact(name: str, plain: torch.Tensor, ruled: torch.Tensor) -> None:
    gap = float((plain.float() - ruled.float()).abs().max())
    if gap != 0.0:
        raise RuntimeError(
            f"the {name} rule changed the forward value by {gap:.3e}; the "
            f"rule must be exact, so this is a fault and not noise")


# ---------------------------------------------------------------------------
# The context manager
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def lrp_rules(
    model: nn.Module, *, norm: bool = True, identity: bool = True,
    half: bool = True, softmax: bool = True, verify: bool = True,
) -> Iterator[dict]:
    """Install the LRP rules on `model`, and take them off again after.

    The rules live on the CLASSES, not on the instances, so every block in
    the model gets them at once. Each flag turns 1 rule off, which is what an
    ablation of the rule set needs.

    Warning: this patches classes that other models in the same process
    share. Do not hold 2 models of the same family and rule only 1 of them.
    """
    report: dict = {"norm": 0, "mlp": 0, "softmax": False,
                    "softmax_calls": 0, "zero_slope": []}
    undo: list = []

    if softmax:
        fused = attention_kinds(model.config) & FUSED_ATTENTION
        if fused:
            raise RuntimeError(
                f"the softmax rule needs eager attention, but the config "
                f"names {sorted(fused)}; load with attn_route.load_eager")

    try:
        if norm:
            for kind in classes_with_suffix(model, NORM_SUFFIX):
                original = kind._norm
                if verify:
                    # `_norm` reads only `self.eps`, so a CPU tensor tests it
                    # whatever the module holds. `v_norm` has no parameter.
                    site = one_instance(model, kind)
                    x = _probe(8)
                    _check_exact("norm", original(site, x), lrp_norm(site, x))
                kind._norm = lrp_norm
                undo.append((kind, "_norm", original))
                report["norm"] += 1

        if identity or half:
            for kind in classes_with_suffix(model, MLP_SUFFIX):
                site = one_instance(model, kind)
                if not all(hasattr(site, a) for a in
                           ("gate_proj", "up_proj", "down_proj", "act_fn")):
                    continue
                zero = slope_at_zero(site.act_fn)
                ruled = make_mlp_forward(
                    identity=identity, half=half, zero=zero)
                original = kind.forward
                if verify:
                    x = _probe(int(site.gate_proj.in_features),
                               like=site.gate_proj.weight)
                    with torch.no_grad():
                        _check_exact("mlp", original(site, x), ruled(site, x))
                kind.forward = ruled
                undo.append((kind, "forward", original))
                report["mlp"] += 1
                report["zero_slope"].append(round(zero, 6))

        if softmax:
            original = nn.functional.softmax
            nn.functional.softmax = make_softmax(original, report)
            undo.append((nn.functional, "softmax", original))
            report["softmax"] = True

        yield report
    finally:
        for target, name, value in reversed(undo):
            setattr(target, name, value)


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------
def match_scale(
    relevance: dict, jacobian: dict,
) -> tuple[dict, dict]:
    """Give each R_l the Frobenius norm of the J_l at the same layer.

    The rules shrink relevance at every block. Measured at L20 on 1 image,
    the local R vector is 0.004 times the length of the J vector.

    A top-k readout cannot see a scale, but `scoring.score_patches` takes a
    softmax over the lens logits, and a softmax can see one. A 250-fold
    shrink flattens that softmax, so the quadrant test would report no signal
    where the ranking is in fact perfect.

    One scalar for each layer removes the problem, and no ranking moves.
    Returns the rescaled matrices and the scalar that each layer took.
    """
    out, scales = {}, {}
    for layer, r in relevance.items():
        if layer not in jacobian:
            raise KeyError(f"layer {layer} is not in the J lens")
        r_norm = float(r.float().norm())
        if r_norm == 0.0:
            raise ValueError(f"R_{layer} is all zeros; the fit failed")
        scale = float(jacobian[layer].float().norm()) / r_norm
        out[layer] = r.float() * scale
        scales[layer] = scale
    return out, scales


# ---------------------------------------------------------------------------
# The stage-1 gate: exact forward, different backward
# ---------------------------------------------------------------------------
def check(case: str, layer: int, model_dir: str, device: str) -> int:
    """Prove the 2 halves of the contract on the real model.

    1. The rules do not move a single logit.
    2. The rules DO move the gradient, or they did not install.
    """
    from PIL import Image

    from research.monocle import attn_route as ar
    from research.monocle import jlens_read
    from research.monocle import patch_attribution as pa
    from research.monocle import single as sg

    proc, model, tmpl = ar.load_eager(model_dir, device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    pos_id, neg_id = sg.contrast_ids(proc.tokenizer, case)
    image_token_id = int(model.config.image_token_id)

    row = next(iter(sg.load_single_rows(1, case, seed=20260821)))
    image = Image.open(row["path"]).convert("RGB")
    inputs = sg.build_single_inputs(proc, tmpl, image, case, device=device)
    answer_pos = int(inputs["input_ids"].shape[1]) - 1
    log(f"case={case} image={row['image_id']} layer={layer} "
        f"seq_len={answer_pos + 1}")

    with torch.no_grad():
        plain_logits = model(**inputs, use_cache=False).logits.float()
    with lrp_rules(model) as report:
        log(f"installed: {report}")
        with torch.no_grad():
            ruled_logits = model(**inputs, use_cache=False).logits.float()
        calls = int(report["softmax_calls"])
    gap = float((plain_logits - ruled_logits).abs().max())
    log(f"forward: max |dlogit| = {gap:.3e}  "
        f"{'PASS' if gap == 0.0 else 'FAIL'}")
    log(f"softmax rule ran {calls} times  "
        f"{'PASS' if calls > 0 else 'FAIL (a fused kernel hides it)'}")

    grads_j, acts, c_plain = pa.attribute_pair(
        model, lens_model, inputs, [layer], pos_id, neg_id, answer_pos)
    with lrp_rules(model):
        grads_r, _, c_ruled = pa.attribute_pair(
            model, lens_model, inputs, [layer], pos_id, neg_id, answer_pos)
    log(f"contrast: c_plain={c_plain:+.4f}  c_ruled={c_ruled:+.4f}")

    slots = (inputs["input_ids"][0] == image_token_id).nonzero().flatten()
    a = grads_j[layer][0, slots, :].float()
    b = grads_r[layer][0, slots, :].float()
    cos = float(torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0))
    scale = float(b.norm() / a.norm().clamp_min(1e-30))
    per_patch = torch.nn.functional.cosine_similarity(a, b, dim=1)
    log(f"backward at L{layer}, {slots.numel()} patches: "
        f"cosine={cos:+.4f}  |R|/|J|={scale:.3f}")
    log(f"  per-patch cosine: median={per_patch.median():+.4f}  "
        f"min={per_patch.min():+.4f}  max={per_patch.max():+.4f}")
    moved = cos < 0.999
    log(f"backward: {'PASS (the rules changed it)' if moved else 'FAIL'}")

    ok = gap == 0.0 and moved and calls > 0
    log(f"stage-1 gate: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[list] = None) -> int:
    import argparse

    from research.monocle import patch_attribution as pa
    from research.monocle import single as sg
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="run the stage-1 gate on the real model")
    ap.add_argument("--case", default="subway_safety",
                    choices=list(sg.CASE_RATING))
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    if not a.check:
        ap.error("nothing to do; pass --check")
    return check(a.case, a.layer, a.model_dir, a.device)


if __name__ == "__main__":
    raise SystemExit(main())
