"""FacDB facility-type explorer.

Marimo notebook for picking the next FacDB sub-dataset to materialize.

What it does:
    1. Loads the frozen 4-level FacDB categorization hierarchy
       (`facdomain` -> `facgroup` -> `facsubgrp` -> `factype`).
    2. Pulls the full FacDB once via the same Socrata fetcher the
       curation CLI uses, caches it locally, and shows live row counts at
       every level.
    3. Shows which sub-datasets are already materialized under
       `curation/facdb_*` so we don't repeat work.
    4. Interactive drill-down: pick a level, get filtered options, see
       row counts, and copy a ready-to-run `facdb-facilities` CLI line.

Run with:
    marimo edit notebooks/machine-beholder/explore_facdb_facility_types.py
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro():
    import marimo as mo

    mo.md(
        """
        # FacDB facility-type explorer

        The NYC DCP **Facilities Database** (Socrata `ji82-xba5`) carries
        ~34.7k POIs organized in a 4-level hierarchy
        (`facdomain` -> `facgroup` -> `facsubgrp` -> `factype`).

        Use this notebook to pick the next FacDB sub-dataset to
        materialize via:

        ```bash
        python -m dagspaces.common.curation facdb-facilities \\
            --out curation/facdb_<slug> \\
            --facgroup "<GROUP>"  # or --facdomain / --facsubgrp / --factype
        ```

        The hierarchy is loaded from the frozen
        `dagspaces/common/curation/facdb/categorization.json`; row
        counts come from a one-time live FacDB pull, cached locally.
        """
    )
    return (mo,)


@app.cell
def _imports():
    import json
    from pathlib import Path

    import polars as pl

    REPO_ROOT = Path("/share/pierson/matt/mllmsci")
    CACHE_DIR = REPO_ROOT / "notebooks" / "machine-beholder" / ".cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FACDB_CACHE = CACHE_DIR / "facdb_full.parquet"
    CURATION_DIR = REPO_ROOT / "curation"
    return CURATION_DIR, FACDB_CACHE, json, pl


@app.cell
def _load_hierarchy():
    """Frozen 4-level hierarchy from `categorization.json`."""
    from dagspaces.common.curation.facdb import load_categorization

    cat = load_categorization()
    hierarchy = cat["hierarchy"]
    domains = cat["domains"]
    groups = cat["groups"]
    subgroups = cat["subgroups"]
    types = cat["types"]
    return cat, domains, groups, hierarchy, subgroups, types


@app.cell
def _hierarchy_summary(cat, domains, groups, mo, subgroups, types):
    mo.md(f"""
    **Frozen dictionary `{cat['version']}`**

    | Level | Distinct values |
    |---|---:|
    | `facdomain`  | {len(domains)} |
    | `facgroup`   | {len(groups)} |
    | `facsubgrp`  | {len(subgroups)} |
    | `factype`    | {len(types)} |
    """)
    return


@app.cell
def _fetch_full_facdb(FACDB_CACHE, pl):
    """Pull the full FacDB once and cache as parquet (~34.7k rows).

    Re-uses `dagspaces.common.curation.facdb.fetch_facdb` so we get
    the exact same column subset and pagination handling the curation
    CLI uses. Subsequent runs hit the parquet on disk.
    """
    from dagspaces.common.curation.facdb.fetch import fetch_facdb

    if FACDB_CACHE.exists():
        full_df = pl.read_parquet(FACDB_CACHE)
        cache_msg = (
            f"Loaded **{full_df.height:,}** cached FacDB rows from "
            f"`{FACDB_CACHE}`. Delete the file to force a refresh."
        )
    else:
        result = fetch_facdb(cache_path=str(FACDB_CACHE))
        full_df = result.df
        cache_msg = (
            f"Fetched **{full_df.height:,}** rows in {result.elapsed_s:.1f}s "
            f"({result.pages} page(s)). Cached to `{FACDB_CACHE.name}`."
        )

    # Normalize hierarchy columns to upper-case so they line up with the
    # frozen dictionary (the curation CLI does the same in `normalize.py`).
    for col in ("facdomain", "facgroup", "facsubgrp", "factype"):
        if col in full_df.columns:
            full_df = full_df.with_columns(pl.col(col).str.to_uppercase())
    return cache_msg, full_df


@app.cell
def _show_cache_msg(cache_msg, mo):
    mo.md(cache_msg)
    return


@app.cell
def _level_counts(full_df, pl):
    """Per-level row counts for the live FacDB pull."""

    def _summary(col: str) -> pl.DataFrame:
        return (
            full_df.group_by(col)
            .agg(pl.len().alias("rows"))
            .sort("rows", descending=True)
            .rename({col: "value"})
        )

    domain_counts = _summary("facdomain")
    group_counts = _summary("facgroup")
    subgroup_counts = _summary("facsubgrp")
    type_counts = _summary("factype")
    return domain_counts, group_counts, subgroup_counts, type_counts


@app.cell
def _domain_header(mo):
    mo.md("""
    ### Row counts by `facdomain`
    """)
    return


@app.cell
def _domain_table(domain_counts, mo):
    mo.ui.table(domain_counts.to_pandas(), page_size=10, selection=None)
    return


@app.cell
def _group_header(mo):
    mo.md("""
    ### Row counts by `facgroup`
    """)
    return


@app.cell
def _group_table(group_counts, mo):
    mo.ui.table(group_counts.to_pandas(), page_size=15, selection=None)
    return


@app.cell
def _subgroup_header(mo):
    mo.md("""
    ### Row counts by `facsubgrp` (top 30)
    """)
    return


@app.cell
def _subgroup_table(mo, subgroup_counts):
    mo.ui.table(subgroup_counts.head(30).to_pandas(), page_size=15, selection=None)
    return


@app.cell
def _type_header(mo):
    mo.md("""
    ### Row counts by `factype` (top 50)
    """)
    return


@app.cell
def _type_table(mo, type_counts):
    mo.ui.table(type_counts.head(50).to_pandas(), page_size=20, selection=None)
    return


@app.cell
def _already_materialized(CURATION_DIR, json, pl):
    """Scan `curation/facdb_*` to see what's already been built."""
    materialized: list[dict] = []
    for _sub in sorted(CURATION_DIR.glob("facdb_*")):
        _manifest = _sub / "manifest.json"
        if not _manifest.is_file():
            materialized.append({"dir": _sub.name, "status": "MISSING_MANIFEST"})
            continue
        _m = json.loads(_manifest.read_text())
        materialized.append(
            {
                "dir": _sub.name,
                "status": _m.get("status", "?"),
                "filters": json.dumps(_m.get("filters", {}), separators=(",", ":")),
                "raw_rows": _m.get("raw_rows"),
                "publishable_rows": _m.get("publishable_rows"),
                "polygon_match_pct": round(
                    _m.get("polygon_match_rate_overall_pct") or 0.0, 1
                ),
                "coverage_km2": round(_m.get("coverage_area_km2") or 0.0, 3),
                "built_at": (_m.get("built_at") or "")[:10],
            }
        )
    materialized_df = pl.DataFrame(materialized) if materialized else pl.DataFrame()
    return materialized, materialized_df


@app.cell
def _materialized_header(materialized: list[dict], mo):
    mo.md(f"""
    ### Already materialized — `{len(materialized)}` sub-dataset(s) "
        "under `curation/facdb_*`
    """)
    return


@app.cell
def _materialized_table(materialized_df, mo):
    if materialized_df.is_empty():
        _out = mo.md("_None yet — `curation/facdb_*` is empty._")
    else:
        _out = mo.ui.table(materialized_df.to_pandas(), page_size=20, selection=None)
    _out
    return


@app.cell(hide_code=True)
def _drilldown_intro(mo):
    mo.md("""
    ---

    ## Interactive drill-down

    Pick a level, choose a value (or "(any)" to skip), and the next
    dropdown filters down to the children of your selection. Once
    you've narrowed in, the **CLI command** cell below shows the
    exact `facdb-facilities` invocation to materialize that filter.
    """)
    return


@app.cell
def _domain_picker(domains, mo):
    domain_picker = mo.ui.dropdown(
        options=["(any)"] + sorted(domains),
        value="(any)",
        label="`facdomain`",
    )
    domain_picker
    return (domain_picker,)


@app.cell(hide_code=True)
def _group_picker(domain_picker, groups, hierarchy, mo):
    if domain_picker.value == "(any)":
        group_options = sorted(groups)
    else:
        group_options = sorted(hierarchy[domain_picker.value].keys())
    group_picker = mo.ui.dropdown(
        options=["(any)"] + group_options,
        value="(any)",
        label=f"`facgroup` ({len(group_options)} option(s))",
    )
    group_picker
    return (group_picker,)


@app.cell(hide_code=True)
def _subgroup_picker(domain_picker, group_picker, hierarchy, mo, subgroups):
    if group_picker.value != "(any)" and domain_picker.value != "(any)":
        subgroup_options = sorted(
            hierarchy[domain_picker.value][group_picker.value].keys()
        )
    elif group_picker.value != "(any)":
        subgroup_options = sorted(
            {
                sg
                for dom in hierarchy
                for grp, sgs in hierarchy[dom].items()
                if grp == group_picker.value
                for sg in sgs
            }
        )
    elif domain_picker.value != "(any)":
        subgroup_options = sorted(
            {
                sg
                for grp in hierarchy[domain_picker.value]
                for sg in hierarchy[domain_picker.value][grp]
            }
        )
    else:
        subgroup_options = sorted(subgroups)
    subgroup_picker = mo.ui.dropdown(
        options=["(any)"] + subgroup_options,
        value="(any)",
        label=f"`facsubgrp` ({len(subgroup_options)} option(s))",
    )
    subgroup_picker
    return (subgroup_picker,)


@app.cell(hide_code=True)
def _type_picker(
    domain_picker,
    group_picker,
    hierarchy,
    mo,
    subgroup_picker,
    types,
):
    if (
        domain_picker.value != "(any)"
        and group_picker.value != "(any)"
        and subgroup_picker.value != "(any)"
    ):
        type_options = sorted(
            hierarchy[domain_picker.value][group_picker.value][subgroup_picker.value]
        )
    elif subgroup_picker.value != "(any)":
        type_options = sorted(
            {
                t
                for dom in hierarchy
                for grp in hierarchy[dom]
                for sg, ts in hierarchy[dom][grp].items()
                if sg == subgroup_picker.value
                for t in ts
            }
        )
    elif group_picker.value != "(any)":
        type_options = sorted(
            {
                t
                for dom in hierarchy
                for grp, sgs in hierarchy[dom].items()
                if grp == group_picker.value
                for sg, ts in sgs.items()
                for t in ts
            }
        )
    elif domain_picker.value != "(any)":
        type_options = sorted(
            {
                t
                for grp, sgs in hierarchy[domain_picker.value].items()
                for sg, ts in sgs.items()
                for t in ts
            }
        )
    else:
        type_options = sorted(types)
    type_picker = mo.ui.dropdown(
        options=["(any)"] + type_options,
        value="(any)",
        label=f"`factype` ({len(type_options)} option(s))",
    )
    type_picker
    return (type_picker,)


@app.cell(hide_code=True)
def _apply_selection(
    domain_picker,
    full_df,
    group_picker,
    pl,
    subgroup_picker,
    type_picker,
):
    """Apply the picker filters to the live FacDB."""
    filt = full_df
    chosen: dict[str, str] = {}
    if domain_picker.value != "(any)":
        chosen["facdomain"] = domain_picker.value
        filt = filt.filter(pl.col("facdomain") == domain_picker.value)
    if group_picker.value != "(any)":
        chosen["facgroup"] = group_picker.value
        filt = filt.filter(pl.col("facgroup") == group_picker.value)
    if subgroup_picker.value != "(any)":
        chosen["facsubgrp"] = subgroup_picker.value
        filt = filt.filter(pl.col("facsubgrp") == subgroup_picker.value)
    if type_picker.value != "(any)":
        chosen["factype"] = type_picker.value
        filt = filt.filter(pl.col("factype") == type_picker.value)
    return chosen, filt


@app.cell
def _selection_summary(chosen: dict[str, str], filt, full_df, mo):
    n = filt.height
    mo.md(
        f"""
        ### Selection

        - Filter: **{chosen if chosen else '(none — full FacDB)'}**
        - Live row count: **{n:,}** of {full_df.height:,}
        """
    )
    return (n,)


@app.cell
def _selection_boro_breakdown(filt, mo, n, pl):
    if n:
        by_boro = (
            filt.group_by("boro")
            .agg(pl.len().alias("rows"))
            .sort("rows", descending=True)
        )
        _out = mo.vstack(
            [
                mo.md("**Borough breakdown**"),
                mo.ui.table(by_boro.to_pandas(), page_size=10, selection=None),
            ]
        )
    else:
        _out = mo.md("")
    _out
    return


@app.cell(hide_code=True)
def _selection_factype_breakdown(filt, mo, n, pl):
    """If we haven't drilled all the way to factype, show the type mix."""
    if n and "factype" in filt.columns:
        type_mix = (
            filt.group_by("factype")
            .agg(pl.len().alias("rows"))
            .sort("rows", descending=True)
        )
        if type_mix.height > 1:
            _out = mo.vstack(
                [
                    mo.md(
                        f"**`factype` mix within selection ({type_mix.height} types)**"
                    ),
                    mo.ui.table(type_mix.to_pandas(), page_size=15, selection=None),
                ]
            )
        else:
            _out = mo.md("")
    else:
        _out = mo.md("")
    _out
    return


@app.cell(hide_code=True)
def _cli_command(chosen: dict[str, str], mo, n):
    """Render a copy-pasteable `facdb-facilities` invocation."""

    def _slug(d: dict[str, str]) -> str:
        if not d:
            return "facdb_full"
        for level in ("factype", "facsubgrp", "facgroup", "facdomain"):
            if level in d:
                v = d[level].lower()
                v = "".join(c if c.isalnum() else "_" for c in v).strip("_")
                v = "_".join(filter(None, v.split("_")))
                return f"facdb_{v}"
        return "facdb_custom"

    def _flag(level: str) -> str:
        return {
            "facdomain": "--facdomain",
            "facgroup": "--facgroup",
            "facsubgrp": "--facsubgrp",
            "factype": "--factype",
        }[level]

    if not chosen:
        _out = mo.md(
            "_Pick at least one filter above to generate a CLI command "
            "(running with no filters pulls the full ~34.7k FacDB)._"
        )
    else:
        slug = _slug(chosen)
        flags = " \\\n    ".join(f'{_flag(k)} "{v}"' for k, v in chosen.items())
        cmd = (
            "python -m dagspaces.common.curation facdb-facilities \\\n"
            f"    --out curation/{slug} \\\n"
            f"    {flags}"
        )
        cyclo_target = slug.removeprefix("facdb_")
        _out = mo.md(
            f"""
    ### Suggested CLI

    Materializes **{n:,}** raw FacDB rows into `curation/{slug}/`.

    ```bash
    {cmd}
    ```

    After it finishes, materialize Cyclomedia coverage with:

    ```bash
    sbatch --export=ALL,OUTPUT_FILENAME=cyclomedia_near_{cyclo_target}.parquet \\
    scripts/materialize_scaffolding_cyclomedia.sub \\
    curation/{slug}
    ```
    """
        )
    _out
    return


@app.cell(hide_code=True)
def _candidates_intro(mo):
    mo.md("""
    ---

    ## Materialization candidates

    Heuristic shortlist for the next sub-dataset: levels with a
    "Goldilocks" row count (large enough to matter, small enough to
    materialize and inspect quickly), excluding anything we've
    already done.
    """)
    return


@app.cell
def _already_filter_values(json, materialized: list[dict]):
    """Flatten the filter values already used in past materializations."""
    already: list[str] = []
    for _m in materialized:
        _f_str = _m.get("filters")
        if not isinstance(_f_str, str):
            continue
        try:
            _f = json.loads(_f_str)
        except Exception:
            _f = {}
        for _vals in _f.values():
            if isinstance(_vals, list):
                already.extend(_vals)
    return (already,)


@app.cell
def _candidate_groups(already: list[str], group_counts, mo, pl):
    candidates_grp = (
        group_counts.with_columns(
            pl.col("value").is_in(already).alias("already_done")
        )
        .filter(~pl.col("already_done"))
        .filter(pl.col("rows").is_between(50, 5_000))
        .sort("rows")
    )
    _out = mo.vstack(
        [
            mo.md(
                f"**{candidates_grp.height} `facgroup` candidate(s)** "
                "(50–5,000 rows, not already materialized)"
            ),
            mo.ui.table(
                candidates_grp.drop("already_done").to_pandas(),
                page_size=25,
                selection=None,
            ),
        ]
    )
    _out
    return


@app.cell
def _candidate_subgroups(already: list[str], mo, pl, subgroup_counts):
    candidates_sg = (
        subgroup_counts.with_columns(
            pl.col("value").is_in(already).alias("already_done")
        )
        .filter(~pl.col("already_done"))
        .filter(pl.col("rows").is_between(50, 2_500))
        .sort("rows")
    )
    _out = mo.vstack(
        [
            mo.md(
                f"**{candidates_sg.height} `facsubgrp` candidate(s)** "
                "(50–2,500 rows, not already materialized)"
            ),
            mo.ui.table(
                candidates_sg.drop("already_done").to_pandas(),
                page_size=25,
                selection=None,
            ),
        ]
    )
    _out
    return


@app.cell(hide_code=True)
def _footer(mo):
    mo.md("""
    ---

    ### References

    - Wiki: [[facdb-curation]] — full pipeline / validation / output schema
    - Code: `dagspaces/common/curation/facdb/` — fetch, normalize, validate
    - Frozen dictionary: `dagspaces/common/curation/facdb/categorization.json`
      (baked from `curation/facilities_data_dictionary.xlsx`, 25v2)
    - Downstream consumer: `scripts/materialize_scaffolding_cyclomedia.sub`
      + per-unit facing filter (see [[concept-facing-filter]])
    """)
    return


if __name__ == "__main__":
    app.run()
