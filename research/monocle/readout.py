"""J-lens readouts of an (images, prompt) row: segments, position selectors, reads.

A read at layer l and position p is ``unembed(lens.transport(h_l[p], l))``, where
``h_l`` is the output of block l (``jlens.hooks.ActivationRecorder``). This is
the read of the F10 scripts (``research/scripts/f10/hop_lean.py``) and of
``jlens_read.py``. The dagspace ``research.dagspaces.lensread`` runs it over a
table; see ``research/docs/lensread.md``.

The module has 3 parts:

1. Pure functions over a token sequence: ``segments`` and ``select_positions``
   (the position selectors), and the delta scores of ``top_delta``.
2. ``Reader``: one row in, the top-k words, the answer and (optional) the store
   arrays out. It calls 3 functions (tokenize, forward, read), thus a test can
   give fakes.
3. ``load_reader``: the real functions for a HF model and a J++ lens.

The segments of a sequence:

| Segment | Positions |
|---------|-----------|
| ``prefix`` | before the first image token |
| ``image`` | the image tokens (all blocks) |
| ``text`` | after the first image token, not image and not gen |
| ``gen`` | the generation prompt at the end (and ``force_prefix``) |
| ``all`` | every position |

With the layout ``text_first``, the prompt words are in ``prefix``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

__version__ = "1.0.0"

# The yes forms come first, then the no forms, in the same order. Qwen3.6-27B writes the
# forms with a leading space, thus the no-space forms alone give a false margin.
ANSWERS = ("Yes", "yes", " Yes", " yes", "No", "no", " No", " no")

#: The word list of the F10 stores: a space and 3 or more lower-case letters.
WORD_RE = re.compile(r" [a-z]{3,}")

#: The number of positions in 1 lens pass (as ``hop_all.BATCH``).
BATCH = 256

SEGMENTS = ("prefix", "image", "text", "gen", "all")
KINDS = ("last", "tail", "head", "match", "index", "segment", "top_delta")


# ---------------------------------------------------------------------------
# The helpers of the F10 scripts (hop_all.py imports them from here)
# ---------------------------------------------------------------------------
def pooled_margin(logits: torch.Tensor) -> torch.Tensor:
    """Give log p(any yes form) - log p(any no form). The first half of the last axis holds the yes forms."""
    half = logits.shape[-1] // 2
    return torch.logsumexp(logits[..., :half], -1) - torch.logsumexp(logits[..., half:], -1)


def parts(seq: list[int], image: int) -> tuple[list[int], list[int], list[int]]:
    """Give the indices of the prefix, the patches and the tail. The patches must be 1 block."""
    hits = [i for i, t in enumerate(seq) if t == image]
    if not hits or hits != list(range(hits[0], hits[-1] + 1)):
        raise RuntimeError("the patch tokens are not 1 block")
    return list(range(hits[0])), hits, list(range(hits[-1] + 1, len(seq)))


# ---------------------------------------------------------------------------
# Segments and selectors
# ---------------------------------------------------------------------------
def segments(seq: Sequence[int], image_token: Optional[int], n_gen: int) -> Dict[str, List[int]]:
    """Split ``seq`` into the segments of the module docstring.

    ``n_gen`` is the number of generation-prompt tokens at the end.
    """
    n = len(seq)
    if not 0 <= n_gen <= n:
        raise ValueError(f"n_gen={n_gen} is not in [0, {n}]")
    gen = list(range(n - n_gen, n))
    image = [i for i, t in enumerate(seq) if image_token is not None and t == image_token]
    first = image[0] if image else n - n_gen
    image_set, gen_set = set(image), set(gen)
    prefix = list(range(min(first, n - n_gen)))
    text = [i for i in range(first, n) if i not in image_set and i not in gen_set]
    return {"prefix": prefix, "image": image, "text": text, "gen": gen, "all": list(range(n))}


def parse_selectors(specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Check the selector specs and give them defaults. Each spec gets a ``label``."""
    out = []
    for raw in specs:
        s = dict(raw)
        kind = s.get("kind")
        if kind not in KINDS:
            raise ValueError(f"unknown selector kind {kind!r}; use one of {KINDS}")
        s.setdefault("optional", False)
        if kind in ("tail", "head"):
            s.setdefault("n", 1)
            s.setdefault("segment", "all" if kind == "tail" else "text")
            if int(s["n"]) < 1:
                raise ValueError(f"{kind}: n must be 1 or more")
        if kind == "match":
            if not s.get("text"):
                raise ValueError("match: give text")
            s.setdefault("which", "last")
            s.setdefault("occurrence", "all")
            if s["which"] not in ("last", "all"):
                raise ValueError("match: which is last or all")
            if s["occurrence"] not in ("first", "last", "all"):
                raise ValueError("match: occurrence is first, last or all")
        if kind == "index":
            if not s.get("at"):
                raise ValueError("index: give at")
            s["at"] = [int(i) for i in s["at"]]
        if kind == "segment":
            s.setdefault("name", "image")
        if kind == "top_delta":
            s.setdefault("k", 5)
            s.setdefault("metric", "read_js")
            s.setdefault("segment", "all")
            s.setdefault("layers", None)
            if s["metric"] not in ("read_js", "residual"):
                raise ValueError("top_delta: metric is read_js or residual")
        for key in ("segment", "name"):
            if key in s and s[key] not in SEGMENTS:
                raise ValueError(f"{kind}: unknown segment {s[key]!r}; use one of {SEGMENTS}")
        s.setdefault("label", _default_label(s))
        out.append(s)
    labels = [s["label"] for s in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"the selector labels are not unique: {labels}")
    return out


def _default_label(s: Mapping[str, Any]) -> str:
    kind = s["kind"]
    if kind in ("tail", "head"):
        return f"{kind}{s['n']}_{s['segment']}"
    if kind == "match":
        return f"match:{s['text']}"
    if kind == "index":
        return "index:" + ",".join(map(str, s["at"]))
    if kind == "segment":
        return f"segment:{s['name']}"
    if kind == "top_delta":
        return f"top_delta{s['k']}_{s['metric']}"
    return kind


def find_subsequence(seq: Sequence[int], sub: Sequence[int]) -> List[int]:
    """Give the start of each occurrence of ``sub`` in ``seq``."""
    m = len(sub)
    if m == 0:
        return []
    return [i for i in range(len(seq) - m + 1) if list(seq[i:i + m]) == list(sub)]


def select_positions(
    specs: Sequence[Mapping[str, Any]],
    seq: Sequence[int],
    segs: Mapping[str, List[int]],
    encode: Callable[[str], List[int]],
) -> Dict[int, List[str]]:
    """Run the selectors except ``top_delta``. Give {position: [labels]}.

    A selector that selects nothing raises, unless it has ``optional: true``.
    """
    n = len(seq)
    chosen: Dict[int, List[str]] = {}
    for s in specs:
        kind = s["kind"]
        if kind == "top_delta":
            continue
        if kind == "last":
            hits = [n - 1]
        elif kind == "tail":
            hits = segs[s["segment"]][-int(s["n"]):]
        elif kind == "head":
            hits = segs[s["segment"]][:int(s["n"])]
        elif kind == "match":
            ids = encode(str(s["text"]))
            starts = find_subsequence(seq, ids)
            if s["occurrence"] == "first":
                starts = starts[:1]
            elif s["occurrence"] == "last":
                starts = starts[-1:]
            hits = []
            for st in starts:
                hits += [st + len(ids) - 1] if s["which"] == "last" else list(range(st, st + len(ids)))
        elif kind == "index":
            hits = []
            for i in s["at"]:
                j = i + n if i < 0 else i
                if not 0 <= j < n:
                    raise ValueError(f"{s['label']}: index {i} is outside a sequence of {n} tokens")
                hits.append(j)
        elif kind == "segment":
            hits = list(segs[s["name"]])
        else:  # pragma: no cover - parse_selectors checks the kinds
            raise ValueError(kind)
        if not hits and not s["optional"]:
            raise ValueError(f"the selector {s['label']} selects no position")
        for p in hits:
            chosen.setdefault(int(p), [])
            if s["label"] not in chosen[int(p)]:
                chosen[int(p)].append(s["label"])
    return chosen


def residual_delta(h: torch.Tensor) -> torch.Tensor:
    """The relative L2 change of the residual between consecutive read layers.

    ``h`` is [positions, layers, d]. Gives [positions, layers - 1].
    """
    h = h.float()
    step = (h[:, 1:] - h[:, :-1]).norm(dim=-1)
    return step / h[:, :-1].norm(dim=-1).clamp_min(1e-12)


def js_divergence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The Jensen-Shannon divergence (nats) of the softmax of 2 logit tensors, over the last axis."""
    la, lb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
    lm = torch.logsumexp(torch.stack([la, lb]), 0) - math.log(2.0)
    kl_a = (la.exp() * (la - lm)).sum(-1)
    kl_b = (lb.exp() * (lb - lm)).sum(-1)
    return 0.5 * (kl_a + kl_b)


def top_k_positions(scores: Mapping[int, float], k: int) -> List[int]:
    """The k positions with the largest scores (ties: the earlier position)."""
    order = sorted(scores, key=lambda p: (-scores[p], p))
    return order[:k]


# ---------------------------------------------------------------------------
# The reader of 1 row
# ---------------------------------------------------------------------------
@dataclass
class Tokenized:
    """The model inputs of a row and their token ids."""

    inputs: Any
    seq: List[int]
    n_gen: int


@dataclass
class RowReadout:
    """The output of 1 row."""

    results: List[Dict[str, Any]]
    answer: Dict[str, Any]
    index: List[Dict[str, Any]] = field(default_factory=list)
    reads: Optional[np.ndarray] = None      # [P, L, V] float16 over the word list
    residuals: Optional[np.ndarray] = None  # [P, L, d] float16


class Reader:
    """Read the J-lens at the selected positions of each row.

    The 3 functions:

    - ``tokenize(images, prompt) -> Tokenized``
    - ``forward(inputs) -> ({layer: [seq, d]}, final_logits [vocab])``
    - ``read(h [n, d], layer) -> logits [n, vocab]`` (the lens read)
    """

    def __init__(
        self,
        *,
        tokenize: Callable[[List[Any], str], Tokenized],
        forward: Callable[[Any], Tuple[Dict[int, torch.Tensor], torch.Tensor]],
        read: Callable[[torch.Tensor, int], torch.Tensor],
        decode: Callable[[List[int]], str],
        encode: Callable[[str], List[int]],
        layers: Sequence[int],
        image_token: Optional[int],
        selectors: Sequence[Mapping[str, Any]],
        top_k: int = 20,
        keep_ids: Optional[Sequence[int]] = None,
        answer_forms: Optional[Sequence[str]] = ANSWERS,
        store: bool = False,
    ) -> None:
        self.tokenize, self.forward, self.read = tokenize, forward, read
        self.decode, self.encode = decode, encode
        self.layers = [int(l) for l in layers]
        if not self.layers:
            raise ValueError("give 1 or more layers")
        self.image_token = image_token
        self.selectors = parse_selectors(selectors)
        if not self.selectors:
            raise ValueError("give 1 or more position selectors")
        self.top_k = int(top_k)
        self.keep = None if keep_ids is None else torch.as_tensor(list(keep_ids), dtype=torch.long)
        self.forms = list(answer_forms) if answer_forms else []
        self.form_ids = [self.encode(f)[0] for f in self.forms]
        self.store = bool(store)

    # -- the lens reads -----------------------------------------------------
    def _reads(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """The lens logits [n, vocab] of h [n, d], in passes of BATCH positions."""
        out = []
        for i in range(0, h.shape[0], BATCH):
            x = self.read(h[i:i + BATCH], layer).float()
            if not torch.isfinite(x).all():
                raise RuntimeError(f"a lens read at layer {layer} is not finite")
            out.append(x)
        return torch.cat(out)

    def _delta_scores(self, spec: Mapping[str, Any], acts: Dict[int, torch.Tensor],
                      candidates: List[int]) -> Dict[int, float]:
        """The score of each candidate: the largest change over the consecutive layer pairs."""
        layers = self.layers
        if spec.get("layers"):
            lo, hi = int(spec["layers"][0]), int(spec["layers"][1])
            layers = [l for l in layers if lo <= l <= hi]
        if len(layers) < 2:
            raise ValueError(f"{spec['label']}: the layer range holds fewer than 2 read layers")
        scores: List[float] = []
        # Passes of BATCH candidates: a full-vocabulary read of every patch at
        # every layer does not fit in memory at one time.
        for i in range(0, len(candidates), BATCH):
            idx = torch.as_tensor(candidates[i:i + BATCH], dtype=torch.long)
            if spec["metric"] == "residual":
                h = torch.stack([acts[l][idx.to(acts[l].device)].float().cpu() for l in layers], dim=1)
                score = residual_delta(h).max(dim=1).values
            else:
                score, prev = None, None
                for l in layers:
                    cur = self._reads(acts[l][idx.to(acts[l].device)], l)
                    if prev is not None:
                        js = js_divergence(prev, cur)
                        score = js if score is None else torch.maximum(score, js)
                    prev = cur
                assert score is not None
            scores += score.float().cpu().tolist()
        return dict(zip(candidates, scores))

    # -- one row -------------------------------------------------------------
    def read_row(self, sample_id: Any, images: List[Any], prompt: str) -> RowReadout:
        tk = self.tokenize(images, prompt)
        seq = tk.seq
        segs = segments(seq, self.image_token, tk.n_gen)
        chosen = select_positions(self.selectors, seq, segs, self.encode)
        acts, final_logits = self.forward(tk.inputs)
        deltas: Dict[int, float] = {}
        for s in self.selectors:
            if s["kind"] != "top_delta":
                continue
            cand = segs[s["segment"]]
            if not cand:
                if s["optional"]:
                    continue
                raise ValueError(f"the selector {s['label']} has no candidate position")
            scores = self._delta_scores(s, acts, cand)
            for p in top_k_positions(scores, int(s["k"])):
                chosen.setdefault(p, [])
                chosen[p].append(s["label"])
                deltas[p] = max(deltas.get(p, -math.inf), scores[p])
        positions = sorted(chosen)
        seg_of = {}
        for name in ("prefix", "image", "text", "gen"):
            for p in segs[name]:
                seg_of[p] = name
        pos_t = torch.as_tensor(positions, dtype=torch.long)

        results: List[Dict[str, Any]] = []
        reads_store, res_store = [], []
        for l in self.layers:
            h = acts[l][pos_t.to(acts[l].device)]
            logits = self._reads(h, l)
            logprob = torch.log_softmax(logits, -1)
            sub_logits = logits if self.keep is None else logits[:, self.keep.to(logits.device)]
            sub_lp = logprob if self.keep is None else logprob[:, self.keep.to(logprob.device)]
            k = min(self.top_k, sub_logits.shape[-1])
            top = sub_logits.topk(k, dim=-1)
            for row, p in enumerate(positions):
                for rank in range(k):
                    j = int(top.indices[row, rank])
                    tid = j if self.keep is None else int(self.keep[j])
                    results.append({
                        "sample_id": sample_id, "position": p, "segment": seg_of.get(p, ""),
                        "token_at": self.decode([seq[p]]), "selectors": ",".join(chosen[p]),
                        "delta": deltas.get(p, float("nan")), "layer": l, "rank": rank,
                        "token": self.decode([tid]), "token_id": tid,
                        "logit": float(top.values[row, rank]), "logprob": float(sub_lp[row, j]),
                    })
            if self.store:
                reads_store.append(sub_logits.half().cpu().numpy())
                res_store.append(h.float().half().cpu().numpy())

        lp = torch.log_softmax(final_logits.float(), -1)
        answer: Dict[str, Any] = {
            "sample_id": sample_id, "top": self.decode([int(final_logits.argmax())]),
            "tokens": len(seq), "n_patches": len(segs["image"]), "positions": len(positions),
        }
        if self.forms:
            ids = torch.as_tensor(self.form_ids, dtype=torch.long, device=lp.device)
            vals = lp[ids]
            answer["yes_no"] = float(lp[self.encode("Yes")[0]] - lp[self.encode("No")[0]])
            answer["margin"] = float(pooled_margin(vals))
            for f, v in zip(self.forms, vals.tolist()):
                answer[f"lp_{f}"] = float(v)

        out = RowReadout(results=results, answer=answer)
        if self.store:
            out.reads = np.stack(reads_store, axis=1)      # [P, L, V]
            out.residuals = np.stack(res_store, axis=1)    # [P, L, d]
            out.index = [{"sample_id": sample_id, "position": p, "segment": seg_of.get(p, ""),
                          "selectors": ",".join(chosen[p]), "delta": deltas.get(p, float("nan"))}
                         for p in positions]
        return out


# ---------------------------------------------------------------------------
# The real model and lens
# ---------------------------------------------------------------------------
def word_ids(tokenizer: Any, vocab_size: int) -> List[int]:
    """The ids of the F10 word list: a space and 3 or more lower-case letters."""
    n = min(vocab_size, len(tokenizer))
    return [i for i in range(n) if WORD_RE.fullmatch(tokenizer.decode([i]))]


def render(tokenizer: Any, prompt: str, n_images: int, layout: str, generation: bool,
           force_prefix: str = "") -> str:
    """The chat text of 1 user turn with ``n_images`` images and the prompt."""
    images = [{"type": "image"} for _ in range(n_images)]
    text = [{"type": "text", "text": prompt}]
    if layout == "image_first":
        content = images + text
    elif layout == "text_first":
        content = text + images
    else:
        raise ValueError(f"unknown layout {layout!r}; use image_first or text_first")
    msgs = [{"role": "user", "content": content}]
    out = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=generation,
                                        enable_thinking=False)
    return out + (force_prefix if generation else "")


def load_reader(model_cfg: Mapping[str, Any], readout_cfg: Mapping[str, Any]) -> Tuple[Reader, Dict[str, Any]]:
    """Load the HF model, the processor and the J++ lens. Give the reader and its manifest."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from jlens.hooks import ActivationRecorder

    from dagspaces.common.model_registry import resolve_model_source
    from research.monocle import jlens_read

    model_dir = resolve_model_source(str(model_cfg["model_dir"]), stage_name="lensread")
    dtype = getattr(torch, str(model_cfg.get("dtype", "float32")))
    device_map = model_cfg.get("device_map") or "cuda:0"
    proc = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=dtype, device_map=device_map, attn_implementation="sdpa").eval()
    tok = proc.tokenizer
    lens_model = jlens_read.wrap_for_unembed(model, tok)
    lens = jlens_read.load_lens(str(model_cfg["lens_path"]))
    n_layers = int(model_cfg["n_layers"])
    layers = list(readout_cfg.get("layers") or range(0, n_layers, int(readout_cfg.get("layer_step", 4))))
    bad = [l for l in layers if l not in lens.source_layers]
    if bad:
        raise ValueError(f"the lens has no map for layers {bad}; it has {sorted(lens.source_layers)}")
    first_device = next(model.parameters()).device
    head_device = model.get_output_embeddings().weight.device
    layout = str(readout_cfg.get("layout", "image_first"))
    force_prefix = str(readout_cfg.get("force_prefix") or "")

    def encode(text: str) -> List[int]:
        return tok.encode(text, add_special_tokens=False)

    def tokenize(images: List[Any], prompt: str) -> Tokenized:
        full = render(tok, prompt, len(images), layout, True, force_prefix)
        bare = render(tok, prompt, len(images), layout, False)
        if not full.startswith(bare):
            raise RuntimeError("the generation prompt is not a suffix of the chat text")
        n_gen = len(encode(full)) - len(encode(bare))
        inputs = proc(text=[full], images=[images], return_tensors="pt").to(first_device)
        return Tokenized(inputs=dict(inputs), seq=inputs["input_ids"][0].tolist(), n_gen=n_gen)

    def forward(inputs: Any) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        with ActivationRecorder(lens_model.layers, at=layers) as rec, torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0, -1].float()
            acts = {l: rec.activations[l][0] for l in layers}
        return acts, logits

    def read(h: torch.Tensor, layer: int) -> torch.Tensor:
        with torch.inference_mode():
            return lens_model.unembed(lens.transport(h.to(head_device), layer)).float()

    vocab_size = int(model.get_output_embeddings().weight.shape[0])
    vocab_mode = str(readout_cfg.get("vocab", "words"))
    if vocab_mode == "words":
        keep = word_ids(tok, vocab_size)
    elif vocab_mode == "all":
        keep = None
    else:
        raise ValueError(f"unknown vocab {vocab_mode!r}; use words or all")
    image_token = getattr(model.config, "image_token_id", None)
    reader = Reader(
        tokenize=tokenize, forward=forward, read=read,
        decode=lambda ids: tok.decode(ids), encode=encode,
        layers=layers, image_token=None if image_token is None else int(image_token),
        selectors=list(readout_cfg.get("positions") or [{"kind": "last"}]),
        top_k=int(readout_cfg.get("top_k", 20)), keep_ids=keep,
        answer_forms=list(readout_cfg.get("answer_forms") or []) or None,
        store=bool(readout_cfg.get("store", False)),
    )
    manifest = {
        "version": __version__, "model": str(model_cfg.get("key", "")), "model_dir": model_dir,
        "lens": str(model_cfg["lens_path"]), "dtype": str(dtype).replace("torch.", ""),
        "device_map": str(device_map), "layers": layers, "layout": layout,
        "force_prefix": force_prefix, "vocab": vocab_mode,
        "vocab_size": vocab_size if keep is None else len(keep),
        "selectors": reader.selectors, "top_k": reader.top_k, "answer_forms": reader.forms,
        "d": int(getattr(model.config.get_text_config(), "hidden_size", 0)),
    }
    return reader, {"manifest": manifest, "vocab": keep, "tokenizer": tok}
