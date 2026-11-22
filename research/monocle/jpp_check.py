"""The positive control of a released J++ lens, on text with a known middle step.

The J++ README gives this test: at a mid layer, the lens reads the hidden step
of "the organ that pumps blood" ("heart") before the model writes a number. If
the lens passes here, a failure on image patches comes from the patches, not
from the adapter (`jlens_read.load_lens`). The script prints the top tokens at
7 depths, and the rank of the wanted tokens at each depth.

Use (a GPU job):
    python -m research.monocle.jpp_check --lens outputs/_monocle/jlens/jpp/gemma-4-31b/lens.pt \
        --model-dir /share/ju/matt/zoo/models/Gemma-4-31B-it
"""
from __future__ import annotations

import argparse

import torch

from research.monocle import jlens_read, patch_word_gif

PROMPTS = (
    ("Fact: In humans, the organ that pumps blood through the body has this many chambers: ", ("heart", " heart")),
    ("Fact: The number of legs on the animal that spins webs is", ("spider", " spider", " spiders")),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    proc, model, _ = patch_word_gif.load_model(a.model_dir, a.device)
    tok = proc.tokenizer
    lens_model = jlens_read.wrap_for_unembed(model, tok)
    lens = jlens_read.load_lens(a.lens)
    print(lens, {k: lens.meta.get(k) for k in ("hf_model_name", "lrp_mode", "num_prompts_trained_on")}, flush=True)
    n = lens_model.n_layers
    layers = sorted({min(round(n * f), lens.source_layers[-1]) for f in (1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8)})
    from jlens.hooks import ActivationRecorder

    for text, wanted in PROMPTS:
        ids = tok(tok.bos_token + text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        with ActivationRecorder(lens_model.layers, at=layers) as rec, torch.inference_mode():
            out = model(**ids, use_cache=False)
            acts = {i: rec.activations[i].detach() for i in rec.activations}
        print(f"\n{text!r}\nmodel next: {[tok.decode([i]) for i in out.logits[0, -1].topk(8).indices.tolist()]}")
        want = [i for w in wanted for i in tok(w, add_special_tokens=False).input_ids[:1]]
        for layer in layers:
            with torch.inference_mode():
                lg = lens_model.unembed(lens.transport(acts[layer][0, -1:, :].float(), layer)).float()[0]
            order = lg.argsort(descending=True)
            rank = min(int((order == i).nonzero()[0, 0]) for i in want)
            print(f"L{layer:<3} rank {rank:<7} {[tok.decode([i]) for i in order[:10].tolist()]}", flush=True)


if __name__ == "__main__":
    main()
