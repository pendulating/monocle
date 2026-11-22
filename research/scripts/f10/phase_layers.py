"""F10.20: give the layers of each phase of the judgment, for the prompt figure.

Each rule uses 1 fixed token and a fixed threshold. A band is the first unbroken run of layers that pass.

| Phase | Token | Rule | Source |
|-------|-------|------|--------|
| image | mean patch | R2 of the margin is at least 0.9 of its peak; the band runs to the peak. The value at L0 is in the table | `t2_patch_depth.csv` |
| sense | ` cool` | Axis 1 holds at least 0.40 of the NTA variance, and the score gives abs(r) of at least 0.95 with it | `t2_axis_by_layer.csv` |
| standard | `?` | Gain over the mean patch is at least half of its peak; the band ends at the verdict onset | `t2_gain_by_layer.csv` |
| verdict | last token | r of the read margin with the answer margin is at least 0.95, to the last saved layer | `t2_verdict_by_layer.csv` |

Version 1.1.0 reads the per-layer files of F10.26 (`research/figures/f10/prompts/`, prompt `place cool`),
which hold all 3 models. The phase "scene" is now "image". When no layer reaches the verdict
threshold, the band is the peak layer alone, and `reached` is false (Qwen3.6-27B, r .92 at L60).
`n_layers` gives the depth of the model, for the layer scale of the figure.

Use:
    python research/scripts/f10/phase_layers.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "research/figures/f10"
MODELS = ("gemma-4-31b", "qwen3.5-9b", "qwen3.6-27b")
N_LAYERS = {"gemma-4-31b": 60, "qwen3.5-9b": 32, "qwen3.6-27b": 64}
PROMPTS = FIG / "prompts"
CELL = "adjective == 'cool' and noun == 'place'"
__version__ = "1.1.0"


def first_run(layers: list[int], ok: list[bool]) -> list[int]:
    """Give the first unbroken run of layers that pass."""
    run: list[int] = []
    for layer, passed in zip(layers, ok):
        if passed:
            run.append(layer)
        elif run:
            break
    return run


def bands(model: str) -> list[dict]:
    n = N_LAYERS[model]
    patch = pd.read_csv(PROMPTS / "t2_patch_depth.csv").query(CELL).query("model == @model").sort_values("layer")
    peak = patch.loc[patch["r2"].idxmax()]
    start = patch[patch["r2"] >= 0.9 * peak["r2"]]["layer"].min()
    rows = [{"model": model, "n_layers": n, "phase": "image", "first": 0, "onset": int(start), "last": int(peak["layer"]), "value_l0": patch["r2"].iloc[0], "peak": peak["r2"], "reached": True}]

    sense = pd.read_csv(PROMPTS / "t2_axis_by_layer.csv").query(CELL).query("model == @model and position == 'adjective'").sort_values("layer")
    run = first_run(sense["layer"].tolist(), ((sense["share1"] >= 0.40) & (sense["r_axis1_coollist"].abs() >= 0.95)).tolist())
    rows.append({"model": model, "n_layers": n, "phase": "sense", "first": run[0], "onset": run[0], "last": run[-1], "peak": sense["share1"].max(), "reached": True})

    last = pd.read_csv(PROMPTS / "t2_verdict_by_layer.csv").query(CELL).query("model == @model").sort_values("layer")
    verdict = first_run(last["layer"].tolist(), (last["r_last_token"] >= 0.95).tolist())
    reached = bool(verdict)
    if not reached:
        verdict = [int(last.loc[last["r_last_token"].idxmax(), "layer"])]

    gain = pd.read_csv(PROMPTS / "t2_gain_by_layer.csv").query(CELL).query("model == @model and position == 'qm'").sort_values("layer")
    onset = int(gain[gain["gain"] >= 0.5 * gain["gain"].max()]["layer"].min())
    rows.append({"model": model, "n_layers": n, "phase": "standard", "first": onset, "onset": onset, "last": verdict[0], "peak": gain["gain"].max(), "reached": True})
    rows.append({"model": model, "n_layers": n, "phase": "verdict", "first": verdict[0], "onset": verdict[0], "last": verdict[-1], "peak": last["r_last_token"].max(), "reached": reached})
    return rows


def main() -> None:
    out = pd.DataFrame([r for m in MODELS for r in bands(m)])
    out.to_csv(FIG / "mechanism/phase_layers.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
