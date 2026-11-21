import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Positional bias and test-retest reliability

    2 camera-ready tables, plus a third that folds them into 1. Each reads the
    run parquets of the canonical registry, because both measures need the label
    of each comparison. Neither touches W&B or a case export.

    ## What the battery does about the side

    The sampler swaps the presentation order on every second row
    (`counterbalance_mode: balanced`), thus 50% of all rows show the canonical
    B image first. The stage undoes the swap when it writes `relative_label`.
    The raw bias cancels in the aggregate, and the de-swapped mean is the
    number that says whether it did.

    **Warning: the repeats do not remove the bias.** `repeat_fraction: 0.10`
    draws 100,000 more observations of pairs the battery already holds. A
    repeat gets its own swap state by parity, thus about half come in the same
    order and half in the opposite order. The repeats measure reliability, and
    the 50% swap removes the bias.

    ## The 2 measures

    | Column | Meaning | No effect |
    |--------|---------|-----------|
    | first | Share of decided judgments that favour the first image | 0.50 |
    | $\bar{s}$ | Mean score on the $-2$ to $+2$ scale, after the de-swap | 0 |
    | same | Sign agreement of a repeat that kept the order | 1 |
    | swap | Sign agreement of a repeat that swapped the sides | 1 |

    A decided judgment is any label other than Same or NotSure. A sign
    agreement keeps the repeats where both judgments are decided and counts
    the share that name the same winner. 0.5 is chance there, and a value
    below it means the side beats the content.

    ## Warning: the 2 halves of a repeat answer 2 questions

    | Half | Facility case | Image case ($\dagger$) |
    |------|---------------|------------------------|
    | same | fresh photographs, same side | same photographs: decoding stability |
    | swap | fresh photographs, other side | same photographs: order sensitivity |

    A facility case draws fresh photographs of both units on each observation,
    thus its repeat mixes image variation with model noise. Road quality and
    street photography repeat the identical 2 images. The temperature is 0.0,
    thus a disagreement on the same 2 images in the same order comes from the
    batch composition in vLLM, not from sampling.

    ## Warning: 2 floors run before the table prints

    | Floor | Constant | What it removes |
    |-------|----------|-----------------|
    | Decided judgments | `RL.MIN_DECIDED` (10,000) | A bias cell where the model abstained on nearly every row |
    | Decided repeats | `RL.MIN_BOTH` (1,000) | A reliability cell with too few repeats to read |

    Schools with qwen3.5-9b abstains on 99.7% of the rows, thus both floors
    dash it.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # The shared modules sit in the parent, `notebooks/cvpr/`, because every
    # folder here uses them. This notebook lives one level down.
    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import marimo as mo

    import _canonical as C
    import _reliability as RL

    # The gate. Stop when the registry no longer matches the disk, because
    # then the numbers describe runs that moved or changed.
    C.verify_or_raise()

    NOTEBOOK_VERSION = RL.VERSION
    OUT_DIR = _here / "outputs"
    return C, NOTEBOOK_VERSION, OUT_DIR, RL, mo


@app.cell(hide_code=True)
def _(C, mo):
    arm_pick = mo.ui.radio(options=C.arms(), value="canonical", label="Prompt arm")
    use_cache = mo.ui.checkbox(
        value=True, label="Read the cached measures (uncheck to read every parquet again)"
    )
    full_width = mo.ui.checkbox(
        value=False, label="Full width (table*, spans the 2 columns)"
    )
    mo.md(
        f"""
    ## 1. Controls

    {arm_pick}
    {use_cache}
    {full_width}

    **Canonical is the default for a reason.** An ablation arm holds a
    different number of rows, and its caption states that number. Read an arm
    to compare, never to replace a paper number.

    The cache is 1 JSON file for each run, named by the run hash under
    `master/outputs/reliability_cache/`. A new run never reads a stale file.
    The first read of 14 runs takes about 3 minutes.
    """
    )
    return arm_pick, full_width, use_cache


@app.cell
def _(RL, arm_pick, use_cache):
    bias, retest = RL.build(arm=arm_pick.value, use_cache=bool(use_cache.value))
    return bias, retest


@app.cell(hide_code=True)
def _(RL, bias, mo, retest):
    if bias.empty:
        _msg = ("**Nothing to show.** The registry holds no proxy run of this arm. "
                "Run `scripts/register_canonical_runs.py register` first.")
    else:
        _printed = bias[bias["n_decided"] >= RL.MIN_DECIDED]
        _first = _printed["first_wins"]
        _rel = _printed["relative_mean"].abs()
        _opp = retest[(retest["subset"] == "opposite_order")
                      & (retest["n_decided"] >= RL.MIN_BOTH)]["sign_agree"]
        _msg = f"""
    ## 2. The headline

    **{len(bias)}** cells, **{bias['n_rows'].iloc[0]:,}** rows each,
    **{bias['swapped_share'].mean():.0%}** swapped.

    | Quantity | Value |
    |----------|-------|
    | Cells that favour the first image (first $> 0.5$) | {int((_first > 0.5).sum())} of {len(_printed)} |
    | Raw first-image share, range | {_first.min():.2f} to {_first.max():.2f} |
    | De-swapped mean, largest absolute value | {_rel.max():.3f} |
    | Swapped-repeat sign agreement, range | {_opp.min():.2f} to {_opp.max():.2f} |
    | Swapped-repeat cells at or below chance (0.5) | {int((_opp <= 0.5).sum())} of {len(_opp)} |

    The first row says the bias is real and the third row says the swap
    removed it. The last 2 rows say what the swap cannot fix: at the level of
    1 pair, the side often beats the content. The proxy correlations survive
    because each unit or polygon rests on hundreds to thousands of
    comparisons, and the positional pull cancels across them.
    """
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Positional bias, by cell

    `decided_share` is the share of rows that are neither Same nor NotSure.
    `first_wins` reads only those rows. `presented_mean` is the raw score
    (positive favours the first image), and `relative_mean` is the same score
    after the de-swap, with its 95% half-width in `relative_hw`.
    """)
    return


@app.cell
def _(RL, bias):
    bias_table = RL.bias_wide(bias)
    bias_table
    return (bias_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Test-retest sign agreement, by cell

    The sign agreement over the decided repeats, split by whether the repeat
    kept the order or swapped the sides. The long frame below the table keeps
    the 6-label exact agreement, Cohen $\kappa$, the Spearman correlation of
    the scores, and the abstention agreement, for each of the 3 subsets.
    """)
    return


@app.cell
def _(RL, retest):
    retest_table = RL.retest_wide(retest)
    retest_table
    return (retest_table,)


@app.cell
def _(retest):
    retest
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The LaTeX

    3 floats. The first 2 hold 5 columns each and sit in 1 column of the CVPR
    template at `\small`. The third folds both into 7 columns and scales the
    box to the column width with `\resizebox`. Take the pair, or take the one,
    never all 3.

    Every count in a caption comes from the data, thus an arm with a different
    row count writes its own number.
    """)
    return


@app.cell
def _(RL, bias, full_width, mo, retest):
    bias_latex = RL.bias_to_latex(bias, retest, full_width=bool(full_width.value))
    mo.md(f"```latex\n{bias_latex}\n```")
    return (bias_latex,)


@app.cell
def _(RL, bias, full_width, mo, retest):
    retest_latex = RL.retest_to_latex(retest, bias, full_width=bool(full_width.value))
    mo.md(f"```latex\n{retest_latex}\n```")
    return (retest_latex,)


@app.cell
def _(RL, bias, full_width, mo, retest):
    combined_latex = RL.combined_to_latex(bias, retest, full_width=bool(full_width.value))
    mo.md(f"```latex\n{combined_latex}\n```")
    return (combined_latex,)


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the table files")
    mo.md(
        f"""
    ## 6. Export

    {save}

    The same files come from a command, without marimo:

    ```bash
    .venv-vllm025cu129/bin/python notebooks/cvpr/_reliability.py
    ```
    """
    )
    return (save,)


@app.cell
def _(
    NOTEBOOK_VERSION,
    OUT_DIR,
    arm_pick,
    bias,
    bias_latex,
    combined_latex,
    mo,
    retest,
    retest_latex,
    save,
):
    if save.value:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        _suffix = "" if arm_pick.value == "canonical" else f"_{arm_pick.value}"
        _files = {
            f"positional_bias{_suffix}_v{NOTEBOOK_VERSION}.tex": bias_latex,
            f"test_retest{_suffix}_v{NOTEBOOK_VERSION}.tex": retest_latex,
            f"order_effects{_suffix}_v{NOTEBOOK_VERSION}.tex": combined_latex,
        }
        for _name, _text in _files.items():
            (OUT_DIR / _name).write_text(_text + "\n")
        bias.to_csv(OUT_DIR / f"positional_bias{_suffix}_v{NOTEBOOK_VERSION}.csv",
                    index=False)
        retest.to_csv(OUT_DIR / f"test_retest{_suffix}_v{NOTEBOOK_VERSION}.csv",
                      index=False)
        _out = mo.md(
            "Wrote " + ", ".join(f"`{n}`" for n in _files)
            + f", and the 2 CSV files, to `{OUT_DIR}`."
        )
    else:
        _out = mo.md("Press the button to write the files.")
    _out
    return


if __name__ == "__main__":
    app.run()
