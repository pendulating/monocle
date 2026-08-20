"""Canonical-run prompt path for the pairwise Jacobian-lens study.

The lens must read the SAME forward pass the CVPR battery ran. This module
builds that forward pass. It does not copy the production prompt code — it
calls it. Every string that reaches the model comes from
``dagspaces.urbanpairvqa.stages.pairwise_vqa`` and
``dagspaces.common.vllm_inference``, so the prompt cannot drift away from the
battery.

Why this module exists
----------------------
``monocle/safety_workspace.py`` was written against the pre-consolidation
subway prompt (the 2026-06-29 run). The 2026-08-11 consolidation changed three
things that each break the lens read:

| Mismatch | Old monocle build | Canonical battery |
|---|---|---|
| Image layout | ``images_then_text`` | ``interleaved_labels`` |
| System turn | a persona string | ``system: null`` — no system turn |
| Abstention | 5 labels, no NotSure | NotSure in the enum + a guidance line |

The layout mismatch is the worst of the three. The registry gate demands
``interleaved_labels`` for gemma-4-12b because that architecture does not bind
image B without the text anchor. A lens read under ``images_then_text``
measures a different forward pass, so every per-patch map over image B would
be wrong.

Source of truth
---------------
The prompt config comes from the canonical run's OWN ``.hydra/config.yaml``,
reached through the registry symlink — not from the YAML that sits in
``conf/prompt/`` today. A prompt file can change after a run; the run's
recorded config cannot.

Scope
-----
gemma-4-12b only. Monocle binds ``Gemma4UnifiedForConditionalGeneration`` and
all three fitted lenses are gemma-4-12b (48 layers, d_model 3840). A qwen3.5-9b
lens study needs its own lens fit first.

Warning: prefer the ``proxy`` kind. The ``trace`` runs set
``enable_thinking=True`` and sample at temperature 1.0, so the answer position
sits after a long sampled reasoning block. The residual there then depends on
sampled text, not on the images alone, which confounds a per-patch attribution.
The ``proxy`` runs are greedy and emit the label immediately.

Every import that needs a GPU stays inside a function, so the CPU test suite
(``tests/test_canonical_prompt.py``) can import this module.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: The registry directory. See vlm-narratives-docs/canonical-run-registry.md.
REGISTRY = REPO / "notebooks/cvpr/canonical_data"

#: The only model this study can lens. See the Scope note above.
MODEL = "gemma-4-12b"

#: The seven cases of the consolidated battery.
CASES = (
    "subway_safety",
    "libraries",
    "schools",
    "road_quality",
    "parks_plazas",
    "restaurants",
    "street_photography",
)

KINDS = ("proxy", "trace")

#: The teacher-forced JSON prefix, and the DEFAULT. The last sequence position
#: then literally emits the label token, which is the position the lens reads.
#: This matches how the model writes the object: under greedy decoding
#: gemma-4-12b emits `{\n  "answer": "More"\n}`, with a newline and a 2-space
#: indent. Taken from the `answer` column of the registered runs.
FORCE_PREFIX = '{\n  "answer": "'

#: The compact prefix the 2026-07-24 rung-B run used. Kept only to reproduce
#: that run.
#:
#: Warning: this form is OFF-DISTRIBUTION and it fails silently, per case.
#: Measured on 150 pairs per case, mean p(prod label) at L47 under the
#: production prompt (jobs 199648 / 199649):
#:
#: | Case | compact | natural |
#: |---|---|---|
#: | subway_safety | 0.778 | 0.774 |
#: | road_quality | 0.001 | 0.769 |
#: | restaurants | 0.001 | 0.771 |
#:
#: subway_safety — the only case rung B ran — is unaffected, so that result
#: stands. Any other case reads as a flat zero, with nothing to distinguish it
#: from "the judgment never enters the channel". Never use this for new work.
FORCE_PREFIX_COMPACT = '{"answer": "'

#: The five ordinal labels, in scale order. NotSure is deliberately absent —
#: it is an abstention, not a point on the scale. See ``label_classes``.
ORDINAL = ("MuchLess", "Less", "Same", "More", "MuchMore")

NOT_SURE = "NotSure"


def log(m: str) -> None:
    print(f"[canonical] {m}", flush=True)


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------
def registry_dir(case: str, kind: str = "proxy", model: str = MODEL) -> Path:
    """Path to one registered run's link directory.

    Raises ValueError on an unknown case or kind, and FileNotFoundError when
    the run is not registered.
    """
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; expected one of {list(CASES)}")
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {list(KINDS)}")
    if model != MODEL:
        raise ValueError(
            f"model {model!r} is not lensable — monocle binds Gemma4Unified and "
            f"the fitted lenses are {MODEL}. Fit a lens for {model!r} first.")
    d = REGISTRY / kind / f"{case}__{model}"
    if not d.is_dir():
        raise FileNotFoundError(
            f"{d} is not registered; run "
            f"`python scripts/register_canonical_runs.py show`")
    return d


def hydra_config(case: str, kind: str = "proxy") -> DictConfig:
    """The full resolved config the registered run used.

    Read through the registry symlink, so a provenance table names the
    registry. Raises FileNotFoundError when the run directory holds no
    ``.hydra/config.yaml``.
    """
    p = registry_dir(case, kind) / "stage" / ".hydra" / "config.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"no recorded hydra config at {p}")
    cfg = OmegaConf.load(p)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"{p} did not parse to a DictConfig")
    return cfg


def prompt_cfg(case: str, kind: str = "proxy") -> DictConfig:
    """A minimal cfg that carries the run's recorded prompt block.

    The production helpers all read ``cfg.prompt.*``, so this is the smallest
    object that drives them. ``model`` comes along because the chat render
    needs ``model.chat_template_kwargs`` (the trace runs set
    ``enable_thinking``).
    """
    full = hydra_config(case, kind)
    if "prompt" not in full:
        raise KeyError(f"{case}/{kind}: recorded config has no `prompt` block")
    node = OmegaConf.create({
        "prompt": full.prompt,
        "model": full.get("model", {}),
    })
    layout = str(node.prompt.get("image_layout", "") or "")
    if layout != "interleaved_labels":
        log(f"WARNING {case}/{kind}: recorded image_layout={layout!r}, not "
            f"interleaved_labels — gemma-4-12b does not bind image B without "
            f"the text anchor")
    return node


def load_results(
    case: str,
    kind: str = "proxy",
    n: Optional[int] = None,
    seed: int = 777,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Rows of one registered run, optionally subsampled.

    Reads ``results.parquet`` through the registry symlink. A subsample is a
    seeded ``DataFrame.sample`` over the whole run, so it does not inherit the
    pair-generation order. ``repeat_idx == 0`` is NOT filtered here — a caller
    that wants one presentation per canonical pair must say so.
    """
    p = registry_dir(case, kind) / "results.parquet"
    df = pd.read_parquet(p, columns=columns)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed)
    return df.reset_index(drop=True)


def load_pairs(case: str, kind: str = "proxy") -> pd.DataFrame:
    """The pair manifest of one registered run."""
    return pd.read_parquet(registry_dir(case, kind) / "pairs.parquet")


# ---------------------------------------------------------------------------
# Prompt construction — production functions only
# ---------------------------------------------------------------------------
_PAIRWISE_VQA_PATH = REPO / "dagspaces/urbanpairvqa/stages/pairwise_vqa.py"
_pairwise_vqa_module: Any = None


def _load_pairwise_vqa() -> Any:
    """The production pairwise stage module, loaded by file path.

    ``dagspaces.urbanpairvqa.stages.__init__`` imports ``trace_extract``,
    which needs langextract. That package is absent from the lens venvs, and
    the lens does not use it. A direct load by path skips the package
    ``__init__`` and keeps this module CPU-safe.

    ``pairwise_vqa`` itself imports only pandas and omegaconf at module level.
    """
    global _pairwise_vqa_module
    if _pairwise_vqa_module is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "monocle._prod_pairwise_vqa", _PAIRWISE_VQA_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {_PAIRWISE_VQA_PATH}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _pairwise_vqa_module = mod
    return _pairwise_vqa_module


def _production():
    """The production prompt helpers. Imported here, never copied.

    ``vllm_inference`` imports only pandas and omegaconf at module level, so
    this stays CPU-safe.
    """
    from dagspaces.common.vllm_inference import _to_image_type_blocks
    return _load_pairwise_vqa(), _to_image_type_blocks


def system_text(cfg: DictConfig) -> Optional[str]:
    """The system-turn text, or None when the run sends no system turn.

    Delegates to the production resolver, which normalizes an empty string to
    None. ``system: ""`` renders a vestigial empty turn and is never what the
    battery wants.
    """
    pairwise_vqa, _ = _production()
    return pairwise_vqa._resolve_system_prompt(cfg)


def user_text(pair_id: str, cfg: DictConfig) -> str:
    """The rendered user text for one pair.

    Delegates to the production renderer, so the abstention guidance, the
    ``Pair ID:`` line, and the interpretation line all appear exactly as they
    did in the run.
    """
    pairwise_vqa, _ = _production()
    return pairwise_vqa._render_pair_prompt({"pair_id": pair_id}, cfg)


def build_messages(
    row: dict, cfg: DictConfig, *, for_hf: bool = True,
) -> list[dict]:
    """Chat messages for one pair, exactly as the run built them.

    Runs the production preprocess callback and keeps only ``messages``. The
    callback also builds sampling params and the guided-decoding schema; the
    lens does not use those.

    ``row`` needs ``pair_id``, ``presented_left_path``, and
    ``presented_right_path``. The ``prompt`` field is rendered here when it is
    absent, the same way the stage renders it before preprocess runs.

    With ``for_hf`` the image blocks are rewritten to ``{"type": "image"}`` —
    the rewrite the gemma4_unified path applies before it renders the chat
    template. The HF processor wants the same block type.
    """
    pairwise_vqa, to_image_blocks = _production()
    row = dict(row)
    if "pair_id" not in row:
        raise KeyError("row has no pair_id")
    row.setdefault("prompt", user_text(str(row["pair_id"]), cfg))
    preprocess = pairwise_vqa._make_pairwise_preprocess(cfg)
    messages = preprocess(row)["messages"]
    return to_image_blocks(messages) if for_hf else messages


def chat_template_kwargs(cfg: DictConfig) -> dict:
    """``model.chat_template_kwargs`` of the run, as a plain dict.

    The proxy runs record none. The trace runs record
    ``{"enable_thinking": True}``, which adds a reasoning channel before the
    answer — see the warning in the module docstring.
    """
    node = cfg.get("model", None)
    if node is None:
        return {}
    raw = node.get("chat_template_kwargs", None)
    if not raw:
        return {}
    return dict(OmegaConf.to_container(raw, resolve=True)
                if OmegaConf.is_config(raw) else raw)


def render_chat_text(
    tokenizer: Any,
    tmpl: str,
    messages: list[dict],
    cfg: Optional[DictConfig] = None,
    *,
    force_prefix: str = FORCE_PREFIX,
) -> str:
    """Chat text for the lens forward pass.

    Mirrors the gemma4_unified branch of ``run_vllm_inference``: the recorded
    ``model.chat_template_kwargs`` pass through, the gemma-4 template renders
    the ``<|image|>`` placeholders, and a generation prompt is added. The
    forced JSON prefix is then appended so the last position emits the label.

    Pass ``force_prefix=""`` to get the text the run itself sent.
    """
    kwargs = chat_template_kwargs(cfg) if cfg is not None else {}
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template=tmpl, **kwargs)
    return text + force_prefix


def build_pair_inputs(
    proc: Any,
    tmpl: str,
    images: list[Any],
    row: dict,
    cfg: DictConfig,
    *,
    force_prefix: str = FORCE_PREFIX,
    device: str = "cuda:0",
) -> dict:
    """Model inputs for one pair, on the canonical prompt path.

    ``images`` must be [image A, image B] in PRESENTED order — that is
    ``presented_left_path`` then ``presented_right_path``, which is not
    ``image_path_a``/``image_path_b`` when ``is_swapped`` is true.
    """
    if len(images) != 2:
        raise ValueError(f"expected 2 images, got {len(images)}")
    messages = build_messages(row, cfg)
    text = render_chat_text(
        proc.tokenizer, tmpl, messages, cfg, force_prefix=force_prefix)
    return proc(text=[text], images=[images], return_tensors="pt").to(device)


# ---------------------------------------------------------------------------
# Contrast conditions
# ---------------------------------------------------------------------------
#: The neutral question. It asks for the same 5-point comparison but never
#: names an attribute, a unit, or a city. Everything else — the label list, the
#: interpretation line, the abstention guidance, the Pair ID line, the image
#: layout, the absent system turn — matches the production condition, so the
#: only variable is whether the prompt asks for the judgment.
NEUTRAL_TEMPLATE = """Image A is the first image and Image B is the second image.
Compare the two images.

Return exactly one label from:
- MuchLess
- Less
- Same
- More
- MuchMore

Interpret this as "Image A is <label> than Image B". Use "Same" when they look
about the same.
"""

#: The axis question, in the same shape. ``{axis}`` takes the recovered phrase.
AXIS_TEMPLATE = """Image A is the first image and Image B is the second image.
Which image looks more {axis}?

Return exactly one label from:
- MuchLess
- Less
- Same
- More
- MuchMore

Interpret this as "Image A looks <label> {axis} than Image B". Use "Same" when
they look about the same.
"""

#: GEPA-recovered axis phrases, by case. A case that is absent has no axis
#: condition — the axis arm is then skipped rather than guessed.
RECOVERED_AXIS = {
    "subway_safety": "brightly lit",
}


def build_conditions(
    case: str, kind: str = "proxy", conds: Optional[list[str]] = None,
) -> dict[str, DictConfig]:
    """{condition: cfg} for the prompt arms of the lens study.

    Every arm is the canonical config with ONLY ``prompt.user_template``
    changed. The system turn stays absent, the layout stays
    ``interleaved_labels``, and the abstention guidance still appends. Thus a
    prod-vs-neutral difference measures the question, and nothing else.

    Warning: the pre-2026-08-11 build did not do this. Its neutral arm carried
    "You are a helpful assistant." and its axis arm carried a long judge
    persona, while the prod arm carried the old subway persona. That contrast
    varied the persona and the question together.

    ``conds`` defaults to ``["prod", "neutral"]`` plus ``"axis"`` when the case
    has a recovered phrase.
    """
    base = prompt_cfg(case, kind)
    axis = RECOVERED_AXIS.get(case)
    if conds is None:
        conds = ["prod", "neutral"] + (["axis"] if axis else [])

    out: dict[str, DictConfig] = {}
    for cond in conds:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        if cond == "prod":
            pass
        elif cond == "neutral":
            cfg.prompt.user_template = NEUTRAL_TEMPLATE
        elif cond == "axis":
            if not axis:
                raise ValueError(
                    f"case {case!r} has no recovered axis phrase; add one to "
                    f"RECOVERED_AXIS or drop the axis condition")
            cfg.prompt.user_template = AXIS_TEMPLATE.format(axis=axis)
        else:
            raise ValueError(f"unknown condition {cond!r}")
        out[cond] = cfg
    return out


# ---------------------------------------------------------------------------
# Label first-token machinery, with abstention
# ---------------------------------------------------------------------------
def not_sure_enabled(cfg: DictConfig) -> bool:
    """Whether the run put the abstention label in the answer enum."""
    pairwise_vqa, _ = _production()
    return pairwise_vqa._not_sure_enabled(cfg)


def not_sure_label(cfg: DictConfig) -> str:
    pairwise_vqa, _ = _production()
    return pairwise_vqa._not_sure_label(cfg)


def label_classes(
    tokenizer: Any, cfg: DictConfig,
) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    """First-token ids for the run's answer labels, after the Much* collapse.

    Returns ``(label_first, class_ids, collapse)``:

    - ``label_first`` — {label: first token id} over every label the run's
      enum allowed, which includes the abstention label when the run enabled
      it.
    - ``class_ids`` — {class: id} over the DISTINGUISHABLE first-token
      classes. MuchLess and MuchMore share the token "Much", so they collapse
      to one class. A single first token cannot separate them.
    - ``collapse`` — {label: class}, for scoring a canonical label against a
      first-token read.

    Raises RuntimeError when two labels that the collapse does not merge still
    share a first token — that would silently fold two classes into one.
    """
    labels = list(ORDINAL)
    if not_sure_enabled(cfg):
        labels.append(not_sure_label(cfg))

    label_first = {
        lbl: int(tokenizer.encode(lbl, add_special_tokens=False)[0])
        for lbl in labels
    }
    collapse = {lbl: ("Much*" if lbl in ("MuchLess", "MuchMore") else lbl)
                for lbl in labels}

    class_ids: dict[str, int] = {}
    for lbl in labels:
        class_ids.setdefault(collapse[lbl], label_first[lbl])

    for cls, tid in class_ids.items():
        twins = [c for c, t in class_ids.items() if t == tid and c != cls]
        if twins:
            raise RuntimeError(
                f"first-token collision across classes {[cls, *twins]} "
                f"(token id {tid}) — the collapse does not cover it")
    return label_first, class_ids, collapse


def presented_images(row: dict) -> tuple[str, str]:
    """(left path, right path) in PRESENTED order.

    The battery swaps presentation order for half the rows. A lens read that
    used ``image_path_a``/``image_path_b`` would put image B on the left for
    those rows and every per-patch map would be mirrored.
    """
    left = str(row.get("presented_left_path", "") or "").strip()
    right = str(row.get("presented_right_path", "") or "").strip()
    if not left or not right:
        raise KeyError(
            "row lacks presented_left_path / presented_right_path — a lens "
            "read must use presented order, not image_path_a/b")
    return left, right
