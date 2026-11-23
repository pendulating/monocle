import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # F4 — the Instagram captions of 6 models, by NTA

    The caption prompt shows each model 1 image and asks: "Look and write an
    Instagram caption for what's in front of me." It runs on 2 image sets,
    the selector below picks 1: `caption` (restaurant storefronts) and
    `street_caption` (street views, D133). The selector shows a case only
    when its caption analysis exists. A model
    can write more than 1 caption and some hashtags in 1 answer. DeepSeek-V4-Flash
    extracts the captions (x2 extraction); a script extracts the hashtags.
    The images of both sets come from samples stratified by NTA, thus each
    caption has the NTA of its image.

    The caption analysis by NTA (`research/scripts/f4/caption_nta_analysis.py`, D101
    to D108) gives the entropy of the caption words and of the hashtags for
    each (model, NTA). This notebook asks 2 questions:

    1. Does the entropy go up or down with the demographics of the NTA? (section 1)
    2. Which hashtags, caption words and captions do the models write for the
       NTAs at 1 end of a demographic feature? (section 2)

    **The entropy measure.** The notebook uses the rarefied entropy (`_rare`).
    For each model, the script draws the same number `k` of tokens from each
    NTA, 200 times, and takes the mean entropy in bits. A plug-in entropy goes
    up with the number of tokens, and the NTAs hold different numbers of
    images, thus the NTA size would drive a plug-in value. `k` is the token
    count of the smallest NTA (3 images or more); section 0 gives it.

    **Warnings:**

    - The entropy of 1 model in 1 NTA describes that model, not the NTA. Compare
      the sign of a correlation across models; do not compare entropy values
      across models (each model has its own `k`).
    - The NTA is a large unit. A feature of the NTA is not a feature of the
      place in the image.
    - The captions often name the restaurant or the neighborhood. Thus a word or
      a hashtag of a group of NTAs can come from the names of its restaurants and
      places, not from a choice of the model.
    - Compare the 2 cases through the sign of a correlation only. The images,
      the NTAs and the `k` of the 2 cases differ.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # `_f4cap` sits beside this notebook and puts `research/notebooks/cvpr/` on the path.
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    import marimo as mo

    from research.notebooks.facct import _f4 as F
    from research.notebooks.facct import _f4cap as C
    from research.notebooks.cvpr import _style as style
    from research.notebooks.cvpr import _traces as T

    style.apply_house_style()

    NOTEBOOK_VERSION = "1.10.0"
    return C, F, NOTEBOOK_VERSION, Path, T, mo


@app.cell(hide_code=True)
def _(C, mo):
    _cases = C.available_cases()
    case_pick = mo.ui.dropdown(options=_cases, value=_cases[0], label="Case")
    case_pick
    return (case_pick,)


@app.cell(hide_code=True)
def _(C, F, case_pick, mo):
    case = case_pick.value
    measures, acs = C.load_measures(case), C.load_acs()
    captions, hashtags = C.load_captions(case), C.load_hashtags(case)
    _k = C.manifest(case)["rarefy_k"]
    if F.DATA == "pooled":
        _name = "the pool of 2 batches (D132): the canonical batch and batch 2 together"
    elif not F.DATA:
        _name = "the canonical batch only, for history (`F4_DATA=canonical`)"
    else:
        _name = f"`F4_DATA={F.DATA}`"
    _ntas = C.load_ntas(case).groupby("nta").size()
    _rows = "\n".join(
        f"| {g.replace('_', ' ')} | {captions.doc_id[captions.model == g].nunique() if g != C.ALL else captions.doc_id.nunique():,} "
        f"| {(captions.model == g).sum() if g != C.ALL else len(captions):,} "
        f"| {(hashtags.model == g).sum() if g != C.ALL else len(hashtags):,} "
        f"| {_k[g]['word_k']} | {_k[g]['tag_k']} |"
        for g in C.GROUPS
    )
    _small = [g for g in C.GROUPS if _k[g]["tag_k"] < 10]
    _warn = (
        f"\n\n**Warning:** the hashtag `k` of {', '.join(_small)} is under 10. A rarefied "
        "entropy of so few hashtags is weak; read its correlations with care."
        if _small else ""
    )
    mo.md(
        f"## 0. The data\n\nThis notebook reads {_name}, at "
        f"`{C.nta_dir(case).relative_to(F.REPO_ROOT)}` (`_f4cap` {C.__version__}). "
        f"{len(_ntas)} NTAs hold {int(_ntas.sum()):,} images ({C.CASES[case][1]}s; median "
        f"{_ntas.median():g} images, range {_ntas.min()} to {_ntas.max()}). "
        f"{len(acs)} NTAs have residents in the ACS table.\n\n"
        "| Model | Answers with a caption | Captions | Hashtags (1 for each answer) | `k` words | `k` hashtags |\n"
        "|-------|------------------------|----------|------------------------------|-----------|--------------|\n"
        + _rows + _warn
    )
    return acs, captions, case, hashtags, measures


@app.cell(hide_code=True)
def _(F, mo):
    min_pick = mo.ui.number(start=3, stop=90, step=1, value=F.nta_min_images(),
                            label="Images in 1 NTA, minimum")
    sig_pick = mo.ui.checkbox(value=False, label="Only the significant tests")
    mo.md(f"""
    ## 1. The entropy and the NTA features

    1 test for each (model, measure, feature): the Spearman correlation over
    the NTAs of the rarefied entropy with the ACS feature, as test 2 of the
    traces notebook (section 4.10). `all models` is the 6 models together,
    with its own `k`. The NTAs that enter have residents and the minimum
    number of images below (the default is the NTA minimum of
    `_f4`). `p_bonf` is `p` times the number of tests of the case
    (Bonferroni). In the
    figure, a bold value with a star is significant (`p_bonf` < 0.05).

    {mo.hstack([min_pick, sig_pick], justify="start", gap=2)}
    """)
    return min_pick, sig_pick


@app.cell
def _(C, acs, measures, min_pick):
    entropy_cor = C.entropy_correlations(measures, acs, int(min_pick.value))
    return (entropy_cor,)


@app.cell(hide_code=True)
def _(C, case, entropy_cor, mo, sig_pick):
    _n = entropy_cor.ntas.max()
    _sig = int((entropy_cor.sign != 0).sum())
    _fig = C.plot_correlations(entropy_cor, title=f"{case}: rarefied entropy against the NTA features ({_n} NTAs)")
    _t = entropy_cor[entropy_cor.sign != 0] if sig_pick.value else entropy_cor
    mo.vstack([
        mo.md(f"{_sig} of {len(entropy_cor)} tests are significant, over {_n} NTAs."),
        _fig,
        _t,
    ])
    return


@app.cell(hide_code=True)
def _(C, F, mo):
    sc_model = mo.ui.dropdown(options=list(C.GROUPS), value=C.ALL, label="Model")
    sc_measure = mo.ui.dropdown(options={v: k for k, v in C.MEASURES.items()}, value="hashtags", label="Measure")
    sc_feature = mo.ui.dropdown(options={C.FEATURE_LABELS[f]: f for f in F.NTA_FEATURES},
                                value=C.FEATURE_LABELS["pct_asian_nh"], label="Feature")
    mo.md(f"""
    ### 1.1 1 test as a scatter

    1 dot for each NTA.

    {mo.hstack([sc_model, sc_measure, sc_feature], justify="start", gap=2)}
    """)
    return sc_feature, sc_measure, sc_model


@app.cell
def _(C, acs, measures, min_pick, sc_feature, sc_measure, sc_model):
    C.plot_entropy_scatter(measures, acs, sc_model.value, sc_measure.value, sc_feature.value,
                           int(min_pick.value))
    return


@app.cell(hide_code=True)
def _(C, F, mo):
    g_feature = mo.ui.dropdown(options={C.FEATURE_LABELS[f]: f for f in F.NTA_FEATURES},
                               value=C.FEATURE_LABELS["median_household_income"], label="Feature")
    g_side = mo.ui.radio(options=["bottom", "top"], value="bottom", label="End", inline=True)
    g_pct = mo.ui.slider(1, 50, value=5, step=1, label="Percent of the NTAs", show_value=True)
    g_model = mo.ui.dropdown(options=list(C.GROUPS), value=C.ALL, label="Model")
    g_kind = mo.ui.radio(options=list(C.KINDS), value="hashtags", label="Item", inline=True)
    g_sort = mo.ui.radio(options={"most frequent in the group": "share",
                                  "most over-used against the other NTAs (z)": "z"},
                         value="most frequent in the group", label="Order")
    g_n = mo.ui.slider(10, 100, value=30, step=5, label="Rows", show_value=True)
    mo.md(f"""
    ## 2. The captions of the NTAs at 1 end of a feature

    The group is the NTAs at the bottom or the top of 1 feature: the given
    percent of the NTAs of section 1 (each NTA counts 1, not its residents).
    The rest is the other NTAs of section 1. The unit is 1 caption answer: an
    item counts 1 time for each answer that holds it.

    | Column | Meaning |
    |--------|---------|
    | `group_answers`, `rest_answers` | the answers that hold the item |
    | `group_pct`, `rest_pct` | the same, in percent of all the answers of the selected model(s) in those NTAs |
    | `z` | the log-odds z of the group against the rest (informative Dirichlet prior, Monroe, Colaresi and Quinn 2008); positive: the group uses the item more; empty under 5 answers in total |

    "Most frequent" shows what dominates the group, and much of it is common
    in all NTAs: compare `group_pct` with `rest_pct`. "Most over-used" shows
    what sets the group apart. Caption words drop the English stopwords.

    {mo.hstack([g_feature, g_side, g_pct], justify="start", gap=2)}
    {mo.hstack([g_model, g_kind, g_n], justify="start", gap=2)}
    {g_sort}
    """)
    return g_feature, g_kind, g_model, g_n, g_pct, g_side, g_sort


@app.cell
def _(C, T, acs, captions, case, g_feature, g_kind, g_model, g_n, g_pct, g_side, g_sort, hashtags,
      measures, min_pick):
    _feats = C.eligible_ntas(measures, acs, int(min_pick.value))
    group, cut = C.nta_group(_feats, g_feature.value, g_side.value, g_pct.value)
    rest = _feats.index.difference(group.index)
    items = C.group_items(g_kind.value, captions, hashtags, group.index, rest, model=g_model.value,
                          sort=g_sort.value, n=int(g_n.value), stopwords=T.default_stopwords(), case=case)
    return cut, group, items, rest


@app.cell(hide_code=True)
def _(C, cut, g_feature, g_kind, g_side, group, items, mo, rest):
    _f = g_feature.value
    _op = "at or under" if g_side.value == "bottom" else "at or over"
    _ntas = group[["NTAName", "BoroName", _f, "restaurants"]].rename(
        columns={_f: C.FEATURE_LABELS[_f], "restaurants": "images"})
    mo.vstack([
        mo.md(
            f"**The group:** {len(group)} NTAs with {C.FEATURE_LABELS[_f]} {_op} {cut:,.1f}; "
            f"the rest: {len(rest)} NTAs. Answers of the selection: "
            f"{items.attrs.get('group_answers', 0):,} in the group, "
            f"{items.attrs.get('rest_answers', 0):,} in the rest."
        ),
        _ntas,
        mo.md(f"**The {g_kind.value} of the group:**"),
        items,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    s_n = mo.ui.slider(5, 100, value=20, step=5, label="Captions", show_value=True)
    s_seed = mo.ui.number(start=0, stop=10_000, step=1, value=0, label="Seed")
    mo.md(f"""
    ### 2.1 Captions of the group

    A random sample of the captions of the group, for the model of section 2.
    Change the seed for another sample.

    {mo.hstack([s_n, s_seed], justify="start", gap=2)}
    """)
    return s_n, s_seed


@app.cell
def _(C, captions, g_model, group, s_n, s_seed):
    C.caption_sample(captions, group.index, model=g_model.value, n=int(s_n.value), seed=int(s_seed.value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 The hashtags of each NTA, on 1 map

    Street captions, the 6 models together. For each NTA, up to 10 hashtags
    with the highest log-odds z against all the other NTAs (Monroe et al. 2008,
    informative prior). The count is images: a tag counts 1 time for each
    image, if 1 or more models write it. Thus 1 image that gets the same tag
    from all 6 models counts 1, not 6. A tag must be on 5 images in the city
    and 3 images in the NTA. The word size shows the z inside the NTA. The number of tags
    follows the NTA area: a tag that does not fit at 1 pt drops out. The words
    are vector text, so zoom in on the PDF to read the small NTAs. Hatched NTAs
    have no tag that passes the filter.

    The fill is the probability that a caption of the NTA (a caption span, not
    a hashtag) names its place: the NTA of the image, or its borough
    (`_f4cap.flag_place_names`). A caption that names an other place counts as
    no. The ramp is 1 hue, with equal steps in OKLab (`_f4cap.place_cmap`).
    All the NTA maps use 1 top for the ramp. The table below gives the rate of each NTA.
    """)
    return


@app.cell
def _(C):
    # The fill of the maps: the probability that a caption (not a hashtag) names its NTA or borough.
    place_flags = C.flag_place_names(C.load_captions("street_caption"))
    place = C.place_rate(place_flags)
    place_cd = C.place_rate(C.district_hashtags(place_flags))
    place_vmax = C.place_vmax(place)
    place.join(C.load_ntas("street_caption").drop_duplicates("nta").set_index("nta")[["NTAName", "BoroName"]]) \
        .sort_values("p", ascending=False)
    return place, place_cd, place_flags, place_vmax


@app.cell
def _(C, place, place_vmax):
    tag_top = C.nta_top_tags(C.load_hashtags("street_caption"), n=10, min_in_nta=3)
    tag_fig = C.plot_tag_map(tag_top, fill=place["p"], vmax=place_vmax)
    tag_fig
    return tag_fig, tag_top


@app.cell
def _(C, place, tag_fig, tag_top):
    tag_caption = C.tag_map_caption(tag_top, tag_fig, rate=place)
    tag_top.sort_values(["nta", "rank"])
    return (tag_caption,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 The same hashtags on a cartogram

    Each NTA with tags gets the same area, because each NTA holds 50 or 51
    street views. Thus Manhattan grows and Staten Island shrinks, as on a
    transit map. An NTA with no tags gets 0.6 of that area. The warp is
    Dougenik, Chrisman and Niemeyer (1985), over simplified shores.
    """)
    return


@app.cell
def _(C, place, place_vmax, tag_top):
    import geopandas as _gpd

    _ntas = _gpd.read_file(C.NTA_SHAPES)
    _warped, _err = C.nta_cartogram(_ntas["NTA2020"].isin(tag_top["nta"]))
    carto_fig = C.plot_tag_map(tag_top, ntas=_warped, fill=place["p"], vmax=place_vmax)
    carto_caption = C.tag_map_caption(tag_top, carto_fig, cartogram=True, rate=place)
    carto_fig
    return carto_caption, carto_fig


@app.cell(hide_code=True)
def _(mo):
    cd_go = mo.ui.run_button(label="Fit the district map")
    mo.md(f"""
    ### 2.4 The hashtags of each community district, for print

    The NTA maps need a zoom: on a 7 in page, an NTA is about 0.3 in across.
    This map uses the 59 community districts (CDTA) with street views, 5
    hashtags each, ranked against the other districts, and every district
    shows all 5. The count is images, as in section 2.2, and a tag must be on
    5 images in the district. Thus a tag that 1 or 2 images get (for example
    the name of 1 shop) does not show. The cartogram starts from an area in proportion to the NTAs
    with street views of each district; the districts with none (JFK, Jamaica
    Bay, Fresh Kills) keep their shape and do not push on their neighbours.
    A district whose tags do not fit at 4 pt then gets more area, round by
    round (`_f4cap.fit_district_map`); a district that is still short gets a
    lower floor (3.5 pt, then 3 pt). The caption names these districts.

    {cd_go}
    """)
    return (cd_go,)


@app.cell
def _(C, cd_go, mo, place_cd, style):
    cd_top = C.nta_top_tags(C.district_hashtags(C.load_hashtags("street_caption")), n=5, min_in_nta=5)
    cd_fig = cd_caption = None
    # Headless: `python f4_captions.py --fit` presses the 2 fit buttons.
    if cd_go.value or "fit" in mo.cli_args():
        with mo.status.spinner("Fitting the district map (about 13 minutes)"):
            cd_fig = C.fit_district_map(cd_top, border=style.PAL["warmgrey"], fill=place_cd["p"],
                                        vmax=C.place_vmax(place_cd))
        cd_caption = C.tag_map_caption(cd_top, cd_fig, cartogram=True, unit="district", rate=place_cd)
    cd_fig if cd_fig is not None else mo.md("_Press the button to fit the map. The export then writes it._")
    return cd_caption, cd_fig, cd_top


@app.cell(hide_code=True)
def _(mo):
    cd4_go = mo.ui.run_button(label="Fit the 4 model maps")
    mo.md(f"""
    ### 2.5 The hashtags of each community district, for each model

    1 full page, 4 district maps: 1 for each thinking model
    (`_f4.THINKING_MODELS`), with the hashtags of that model only. The 2
    Llama models are not on the page: llama-3.2-90b tags only 4,948 of the
    10,000 images. The rule is the rule of section 2.4 (images, 5 in the
    district), applied to each model alone, with 3 tags for each district.
    Each panel is 3.4 in wide, thus 5 tags do not fit: with 5 tags, 232 to
    249 of 295 fit, nearly all at the 2 pt floor. The type starts at 3 pt,
    then a district that is still short gets a lower floor (2.6, 2.3, 2 pt).
    The 4 fits run in 4 processes at the same time
    (`_f4cap.fit_model_district_maps`), in about 40 minutes.

    {cd4_go}
    """)
    return (cd4_go,)


@app.cell
def _(C, cd4_go, mo, place_flags, style):
    cd4_fits = cd4_fig = cd4_caption = None
    if cd4_go.value or "fit" in mo.cli_args():
        with mo.status.spinner("Fitting the 4 model maps (about 40 minutes)"):
            cd4_fits = C.fit_model_district_maps(C.load_hashtags("street_caption"), n=3, min_images=5,
                                                 width_in=3.425, min_pt=3.0, floors=(2.6, 2.3, 2.0),
                                                 max_pts=(8, 7, 6, 5, 4.5, 4, 3.5, 3), rounds=20, boost=1.35,
                                                 border=style.PAL["warmgrey"])
        _cd = C.district_hashtags(place_flags)
        _fill = {m: C.place_rate(_cd, model=m)["p"] for m in cd4_fits}
        cd4_fig = C.plot_model_district_maps(cd4_fits, border=style.PAL["warmgrey"], fill=_fill)
        cd4_caption = (C.model_maps_caption(cd4_fits, cd4_fig)
                       + C.place_caption(C.place_rate(_cd[_cd["model"].isin(list(cd4_fits))]), "district")
                       + " Each panel fills its districts from the captions of its model only, on 1 ramp.")
    cd4_fig if cd4_fig is not None else mo.md("_Press the button to fit the maps. The export then writes them._")
    return cd4_caption, cd4_fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.6 The city in the captions

    All the street views are in NYC. The first map puts the words of section
    2.2 on an other fill (`_f4cap.redraw_tag_map`, no new fit). The second map
    shows the other cities of each NTA in place of the hashtags
    (`_f4cap.other_city_top`): the word size is the number of images.

    1. The probability that a caption of the NTA names NYC (`_f4cap.CITY_NAMES`:
       "nyc", "new york", "big apple" and others). Borough and NTA names do not
       count here. The unit is the caption, as in section 2.2.
    2. The probability that an image of the NTA gets 1 or more captions that
       give an other city (`_f4cap.OTHER_CITIES`: Philadelphia, Chicago,
       Washington DC, Toronto and others). The unit is the image, because these
       captions are rare. A name before a street or shop word, a hedge
       ("nyc or philly"), a comparison and a template blank do not count
       (`_f4cap.flag_city_names`).

    **Warning:** the second map is sparse. Most NTAs have 0 images of 50, and
    the top NTA has 4. Read the table below the map before you rank NTAs.
    llama-3.2-11b writes most of these captions.
    """)
    return


@app.cell
def _(C, place_flags):
    city_flags = C.flag_city_names(place_flags)
    city = C.flag_rate(city_flags, "names_city")
    other = C.flag_rate(city_flags, "names_other", unit="image")
    return city, city_flags, other


@app.cell
def _(C, city, city_flags, tag_fig, tag_top):
    city_fig = C.redraw_tag_map(tag_fig, tag_top, fill=city["p"], vmax=C.place_vmax(city),
                                fill_label="P(caption names NYC)")
    city_caption = C.tag_map_caption(tag_top, tag_fig) + C.city_caption(city, city_flags)
    city_fig
    return city_caption, city_fig


@app.cell
def _(C, city_flags, other, style):
    # The words are the other cities of each NTA, not the hashtags: this map has its own fit.
    other_top = C.other_city_top(city_flags)
    other_fig = C.plot_tag_map(other_top, prefix="", ceiling=C.count_ceiling(other_top),
                               fill=other["p"], vmax=C.place_vmax(other),
                               fill_label="P(image gets a caption that names an other city)",
                               fill_color=style.PAL["coral"])
    other_caption = C.other_city_map_caption(other_top, other_fig, other, city_flags)
    other_fig
    return other_caption, other_fig


@app.cell
def _(C, city_flags, other):
    # The NTAs with an other city, and the cities, from the most to the least.
    _cities = C.other_cities(city_flags).groupby("nta")[["other", "captions"]].apply(
        lambda d: ", ".join(f"{o} ({c})" for o, c in zip(d["other"], d["captions"])))
    _names = C.load_ntas("street_caption").drop_duplicates("nta").set_index("nta")[["NTAName", "BoroName"]]
    other[other["hits"] > 0].join(_names).join(_cities.rename("cities")).sort_values("p", ascending=False)
    return


@app.cell
def _(city_flags):
    # The captions that give an other city, by model: the share of the captions of each model.
    city_flags.groupby("model")[["names_city", "names_other"]].mean().mul(100).round(3) \
        .rename(columns=lambda c: f"% {c}")
    return


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the files")
    mo.md(f"""
    ## 3. Export

    Writes the table and the figure of section 1 for the selected case to
    `research/notebooks/facct/figures/<root>/`: `f4_<case>_entropy_nta_features.csv`
    and `.png`. Also writes the hashtag maps of sections 2.2 and 2.3:
    `f4_street_caption_nta_hashtags` and `f4_street_caption_nta_hashtags_cartogram`
    (`.pdf`, `.png` and `.json`), and the district map of section 2.4:
    `f4_street_caption_cdta_hashtags_cartogram`, and the 4 model maps of
    section 2.5: `f4_street_caption_cdta_hashtags_by_model` (each only after
    its button). Also the 2 city maps of section 2.6:
    `f4_street_caption_nta_city` and `f4_street_caption_nta_other_city`.

    {save}
    """)
    return (save,)


@app.cell
def _(C, F, NOTEBOOK_VERSION, Path, carto_caption, carto_fig, case, cd4_caption, cd4_fig, cd_caption, cd_fig,
      city_caption, city_fig, entropy_cor, mo, other_caption, other_fig, save, tag_caption, tag_fig):
    # Headless: `python f4_captions.py --write` presses this button.
    if save.value or "write" in mo.cli_args():
        _dir = Path(__file__).resolve().parent / "figures" / F.FIGURES
        _dir.mkdir(parents=True, exist_ok=True)
        entropy_cor.to_csv(_dir / f"f4_{case}_entropy_nta_features.csv", index=False)
        C.plot_correlations(entropy_cor).savefig(_dir / f"f4_{case}_entropy_nta_features.png", dpi=200,
                                                  bbox_inches="tight")
        C.save_figure(tag_fig, "f4_street_caption_nta_hashtags", "The hashtags of each NTA", tag_caption,
                      ["f4", "street_caption", "hashtags", "nta", "map"])
        C.save_figure(carto_fig, "f4_street_caption_nta_hashtags_cartogram", "The hashtags of each NTA, on a cartogram",
                      carto_caption, ["f4", "street_caption", "hashtags", "nta", "map", "cartogram"])
        C.save_figure(city_fig, "f4_street_caption_nta_city", "The captions that name NYC, by NTA", city_caption,
                      ["f4", "street_caption", "hashtags", "nta", "map", "city"])
        C.save_figure(other_fig, "f4_street_caption_nta_other_city", "The captions that name an other city, by NTA",
                      other_caption, ["f4", "street_caption", "nta", "map", "city", "misidentification"])
        _n = 14
        if cd_fig is not None:
            C.save_figure(cd_fig, "f4_street_caption_cdta_hashtags_cartogram",
                          "The hashtags of each community district", cd_caption,
                          ["f4", "street_caption", "hashtags", "community district", "map", "cartogram"])
            _n += 3
        if cd4_fig is not None:
            C.save_figure(cd4_fig, "f4_street_caption_cdta_hashtags_by_model",
                          "The hashtags of each community district, for each model", cd4_caption,
                          ["f4", "street_caption", "hashtags", "community district", "map", "cartogram",
                           "small multiples", "model"])
            _n += 3
        _msg = mo.md(f"Wrote {_n} files to `research/notebooks/facct/figures/{F.FIGURES}/`. Notebook "
                     f"{NOTEBOOK_VERSION}, `_f4cap` {C.__version__}.")
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
