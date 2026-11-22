"""Descriptive statistics of the 2025_1 Cyclomedia corpus.

This notebook is load-bearing for the CVPR paper. It writes the corpus
paragraph: how many recordings we hold, where they are, how far apart they
are, how many cube faces we read, and how many images qualify for each of the
7 canonical prompts.

The corpus is the 5 `*_2025_1k` datasets of the v1 catalog — 1 pull for each
borough, recorded in 2025. The catalog under `/share/ju/cyclomedia/catalog/v1`
is the only source of a recording count here. The curation manifests under
`curation/` are the only source of a case count.

Run it:
    .venv-mllmsci-vllm025cu129/bin/marimo edit \
        research/notebooks/cvpr/master/corpus_descriptives.py

Export it (headless):
    .venv-mllmsci-vllm025cu129/bin/marimo export html \
        research/notebooks/cvpr/master/corpus_descriptives.py \
        -o research/notebooks/cvpr/master/outputs/corpus_descriptives.html

Warning: use `.venv-mllmsci-vllm025cu129`. It has marimo, polars, pyproj and
scipy.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    # The shared modules sit in the parent, `research/notebooks/cvpr/`, because every
    # folder here uses them. This notebook lives one level down.
    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    from matplotlib.transforms import Affine2D
    from PIL import Image
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    import _canonical as C
    import _style as style

    style.apply_house_style()

    REPO_ROOT = _here.parents[3]
    OUT_DIR = _here / "outputs"
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR = _here / "figures"
    FIG_DIR.mkdir(exist_ok=True)
    return (
        Affine2D,
        C,
        FIG_DIR,
        Image,
        OUT_DIR,
        REPO_ROOT,
        Transformer,
        cKDTree,
        mo,
        np,
        pd,
        pl,
        plt,
    )


@app.cell
def _():
    # Raise NOTEBOOK_VERSION when you change how a number is computed.
    NOTEBOOK_VERSION = "1.0.0"

    CATALOG_ROOT = "/share/ju/cyclomedia/catalog/v1"

    # The corpus. 1 pull for each borough, recorded in 2025. `2025_1` is the
    # pull tag; `1k` is the render size of a cube face.
    DATASETS = [
        "bronx_2025_1k",
        "brooklyn_2025_1k",
        "manhattan_2025_1k",
        "queens_2025_1k",
        "si_2025_1k",
    ]

    # A recording holds 6 cube faces. The pipelines read the 4 street-level
    # faces and drop the sky face (U) and the road-surface face (D).
    FACES_ALL = ["F", "B", "L", "R", "U", "D"]
    FACES_USED = ["F", "B", "L", "R"]

    # NAD83 / New York Long Island, US survey feet. 1 ft = 0.3048 m exactly.
    PROJ_CRS = "EPSG:2263"
    FT_TO_M = 0.3048

    # 2 recordings belong to the same drive when they share a `group` and the
    # time between them is 5 s or less. The vehicle records about every 5 m,
    # thus a longer gap means a new street or a stop.
    DRIVE_GAP_MAX_S = 5.0

    # A `group` is a directory of the pull, not a vehicle. 2 vehicles that
    # record at the same time can land in 1 group, and then the recording that
    # follows in time sits kilometres away. A step of more than 50 m in 5 s
    # asks for 36 km/h between 2 shots that stand 5 m apart, thus it is a
    # step between 2 vehicles and not a step along 1 drive. The cut drops
    # about 0.3% of the steps.
    DRIVE_STEP_MAX_M = 50.0

    BOROUGH_LABEL = {
        "bronx": "Bronx",
        "brooklyn": "Brooklyn",
        "manhattan": "Manhattan",
        "queens": "Queens",
        "staten_island": "Staten Island",
    }

    # --- the cubemap figure -------------------------------------------------
    # The seed of the draw. Change it to get another recording; keep it to get
    # the same recording again.
    CUBE_SEED = 777
    # The borough that the draw reads. Set it to None to draw from the whole
    # corpus.
    CUBE_BOROUGH = "manhattan"
    # Set a `recording_id` here to pin 1 recording and to ignore the draw.
    CUBE_RECORDING = None
    # The width of 1 CVPR column, in inches, and the height of the figure.
    CUBE_FIG_SIZE = (3.25, 1.55)
    return (
        BOROUGH_LABEL,
        CATALOG_ROOT,
        CUBE_BOROUGH,
        CUBE_FIG_SIZE,
        CUBE_RECORDING,
        CUBE_SEED,
        DATASETS,
        DRIVE_GAP_MAX_S,
        DRIVE_STEP_MAX_M,
        FACES_ALL,
        FACES_USED,
        FT_TO_M,
        NOTEBOOK_VERSION,
        PROJ_CRS,
    )


@app.cell(hide_code=True)
def _(NOTEBOOK_VERSION, mo):
    mo.md(
        f"""
    # The 2025_1 Cyclomedia corpus — descriptive statistics

    **Notebook version {NOTEBOOK_VERSION}**

    The corpus is the 5 `*_2025_1k` datasets of the v1 Cyclomedia catalog: 1
    pull for each borough of New York City, recorded in 2025. Every number
    below comes from that catalog or from a curation manifest. This notebook
    reads no run result and no W&B run.

    | Term | Meaning |
    |------|---------|
    | Recording | 1 panorama position on the street, with 1 `recording_id` |
    | Face | 1 cube face of a recording. 6 exist: F, B, L, R, U, D |
    | Street-level face | F, B, L, R. The pipelines read these 4 and drop U and D |
    | Qualifying image | 1 face that passes the filter of a case |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 1. The recordings""")
    return


@app.cell
def _(CATALOG_ROOT, DATASETS, pl):
    # 1 row for each (dataset, recording_id). The catalog holds 1 row for each
    # (recording_id, face), thus `unique` collapses the 6 faces.
    catalog = pl.scan_parquet(
        f"{CATALOG_ROOT}/by_dataset/**/*.parquet", hive_partitioning=True
    ).filter(pl.col("dataset").is_in(DATASETS))

    recordings = (
        catalog.select(
            "dataset",
            "borough",
            "recording_id",
            "group",
            "latitude",
            "longitude",
            "recordedAt",
        )
        .unique(subset=["dataset", "recording_id"])
        .collect()
    )
    return catalog, recordings


@app.cell
def _(BOROUGH_LABEL, pl, recordings):
    boroughs = (
        recordings.group_by("borough")
        .agg(
            pl.len().alias("recordings"),
            pl.col("dataset").first().alias("dataset"),
            pl.col("recordedAt").min().alias("first_recorded"),
            pl.col("recordedAt").max().alias("last_recorded"),
        )
        .with_columns(
            pl.col("borough").replace_strict(BOROUGH_LABEL).alias("Borough"),
            (pl.col("recordings") / pl.col("recordings").sum() * 100).alias(
                "share_pct"
            ),
        )
        .sort("recordings", descending=True)
        .select(
            "Borough",
            "dataset",
            "recordings",
            "share_pct",
            "first_recorded",
            "last_recorded",
        )
    )
    n_recordings = int(recordings.height)
    return boroughs, n_recordings


@app.cell(hide_code=True)
def _(boroughs, mo, n_recordings, recordings):
    mo.md(
        f"""
    The corpus holds **{n_recordings:,} recordings**. The camera recorded them
    between **{recordings["recordedAt"].min():%Y-%m-%d}** and
    **{recordings["recordedAt"].max():%Y-%m-%d}**.

    {mo.ui.table(boroughs.to_pandas(), selection=None)}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 2. The distance between recordings

    2 measurements answer this question, and they answer it differently.

    | Measurement | What it counts |
    |-------------|----------------|
    | Along-drive step | The distance to the next recording of the same drive. This is the design spacing of the camera. |
    | Nearest neighbour | The distance to the closest other recording anywhere in the borough. A second pass on the same street lowers this number. |

    A drive is a run of recordings that share a `group` and that follow each
    other within 5 s.
    """
    )
    return


@app.cell
def _(FT_TO_M, PROJ_CRS, Transformer, pl, recordings):
    # Project to the state plane once. A distance in degrees is not a distance.
    _tf = Transformer.from_crs("EPSG:4326", PROJ_CRS, always_xy=True)
    _x, _y = _tf.transform(
        recordings["longitude"].to_numpy(), recordings["latitude"].to_numpy()
    )
    xy = recordings.with_columns(
        pl.Series("x_m", _x * FT_TO_M),
        pl.Series("y_m", _y * FT_TO_M),
    )
    return (xy,)


@app.cell
def _(DRIVE_GAP_MAX_S, DRIVE_STEP_MAX_M, pl, xy):
    # The along-drive step. Sort inside the drive by time, then take the
    # distance to the previous recording. Read the 2 cuts in the constants
    # cell: the gap cut removes a new street, and the length cut removes a
    # step between 2 vehicles of the same group.
    _candidates = (
        xy.sort(["dataset", "group", "recordedAt"])
        .with_columns(
            (pl.col("x_m") - pl.col("x_m").shift(1))
            .over(["dataset", "group"])
            .alias("dx"),
            (pl.col("y_m") - pl.col("y_m").shift(1))
            .over(["dataset", "group"])
            .alias("dy"),
            (pl.col("recordedAt") - pl.col("recordedAt").shift(1))
            .over(["dataset", "group"])
            .dt.total_milliseconds()
            .alias("gap_ms"),
        )
        .drop_nulls("dx")
        .filter(pl.col("gap_ms") <= DRIVE_GAP_MAX_S * 1000)
        .with_columns(
            ((pl.col("dx") ** 2 + pl.col("dy") ** 2).sqrt()).alias("step_m")
        )
        .select("borough", "step_m")
    )
    steps = _candidates.filter(pl.col("step_m") <= DRIVE_STEP_MAX_M)
    steps_dropped = _candidates.height - steps.height
    return steps, steps_dropped


@app.cell
def _(BOROUGH_LABEL, cKDTree, np, pl, xy):
    # The nearest-neighbour distance. 1 tree for each borough, then 1 tree over
    # the whole corpus. `k=2` because the first neighbour of a point is itself.
    def _nn_metres(frame):
        pts = np.column_stack(
            [frame["x_m"].to_numpy(), frame["y_m"].to_numpy()]
        )
        dist, _ = cKDTree(pts).query(pts, k=2, workers=-1)
        return dist[:, 1]

    _rows = []
    for _boro in sorted(xy["borough"].unique().to_list()):
        _sub = xy.filter(pl.col("borough") == _boro)
        _rows.append(
            pl.DataFrame(
                {"borough": [_boro] * _sub.height, "nn_m": _nn_metres(_sub)}
            )
        )
    nn = pl.concat(_rows)
    nn_corpus = _nn_metres(xy)
    _ = BOROUGH_LABEL
    return nn, nn_corpus


@app.cell
def _(BOROUGH_LABEL, nn, nn_corpus, np, pl, steps):
    def _summary(frame, col):
        return frame.group_by("borough").agg(
            pl.len().alias(f"{col}_n"),
            pl.col(col).mean().alias(f"{col}_mean_m"),
            pl.col(col).median().alias(f"{col}_median_m"),
            pl.col(col).quantile(0.05).alias(f"{col}_p05_m"),
            pl.col(col).quantile(0.95).alias(f"{col}_p95_m"),
        )

    spacing = (
        _summary(steps, "step_m")
        .join(_summary(nn, "nn_m"), on="borough")
        .with_columns(pl.col("borough").replace_strict(BOROUGH_LABEL).alias("Borough"))
        .cast({"step_m_n": pl.Int64, "nn_m_n": pl.Int64})
        .sort("Borough")
    )

    _corpus_row = pl.DataFrame(
        {
            "borough": ["all"],
            "step_m_n": [steps.height],
            "step_m_mean_m": [steps["step_m"].mean()],
            "step_m_median_m": [steps["step_m"].median()],
            "step_m_p05_m": [steps["step_m"].quantile(0.05)],
            "step_m_p95_m": [steps["step_m"].quantile(0.95)],
            "nn_m_n": [len(nn_corpus)],
            "nn_m_mean_m": [float(np.mean(nn_corpus))],
            "nn_m_median_m": [float(np.median(nn_corpus))],
            "nn_m_p05_m": [float(np.percentile(nn_corpus, 5))],
            "nn_m_p95_m": [float(np.percentile(nn_corpus, 95))],
            "Borough": ["Whole corpus"],
        }
    )
    spacing_table = pl.concat([spacing, _corpus_row.select(spacing.columns)]).select(
        "Borough",
        pl.col("step_m_n").alias("drive_steps"),
        pl.col("step_m_mean_m").alias("step_mean_m"),
        pl.col("step_m_median_m").alias("step_median_m"),
        pl.col("step_m_p05_m").alias("step_p05_m"),
        pl.col("step_m_p95_m").alias("step_p95_m"),
        pl.col("nn_m_mean_m").alias("nn_mean_m"),
        pl.col("nn_m_median_m").alias("nn_median_m"),
        pl.col("nn_m_p05_m").alias("nn_p05_m"),
        pl.col("nn_m_p95_m").alias("nn_p95_m"),
    )
    return (spacing_table,)


@app.cell(hide_code=True)
def _(mo, spacing_table, steps_dropped):
    _all = spacing_table.filter(spacing_table["Borough"] == "Whole corpus").row(
        0, named=True
    )
    _kept = int(_all["drive_steps"])
    mo.md(
        f"""
    Over the whole corpus the mean along-drive step is
    **{_all["step_mean_m"]:.2f} m** and the median is
    **{_all["step_median_m"]:.2f} m**. The mean nearest-neighbour distance is
    **{_all["nn_mean_m"]:.2f} m** and the median is
    **{_all["nn_median_m"]:.2f} m**.

    The step columns count {_kept:,} steps. The 50 m cut dropped
    {steps_dropped:,} more steps ({steps_dropped / (_kept + steps_dropped) * 100:.2f}%),
    which are steps between 2 vehicles of 1 group.

    {mo.ui.table(spacing_table.to_pandas().round(3), selection=None)}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 3. The image faces""")
    return


@app.cell
def _(BOROUGH_LABEL, FACES_ALL, FACES_USED, catalog, pl):
    _faces = (
        catalog.group_by(["borough", "face"]).agg(pl.len().alias("n")).collect()
    )
    faces_by_borough = (
        _faces.pivot(values="n", index="borough", on="face")
        .with_columns(
            pl.sum_horizontal(FACES_USED).alias("street_level_faces"),
            pl.sum_horizontal(FACES_ALL).alias("all_faces"),
            pl.col("borough").replace_strict(BOROUGH_LABEL).alias("Borough"),
        )
        .sort("street_level_faces", descending=True)
        .select(["Borough"] + FACES_ALL + ["street_level_faces", "all_faces"])
    )
    n_faces_used = int(faces_by_borough["street_level_faces"].sum())
    n_faces_all = int(faces_by_borough["all_faces"].sum())
    return faces_by_borough, n_faces_all, n_faces_used


@app.cell(hide_code=True)
def _(faces_by_borough, mo, n_faces_all, n_faces_used):
    mo.md(
        f"""
    The corpus holds **{n_faces_all:,} cube faces** on disk. The pipelines read
    the 4 street-level faces, thus we use **{n_faces_used:,} image faces**. The
    sky face (U) and the road-surface face (D) stay unread.

    {mo.ui.table(faces_by_borough.to_pandas(), selection=None)}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 4. The cubemap of 1 recording

    A recording is 1 cyclorama. The camera writes the panorama onto the 6
    faces of a cube, and the pipelines read 4 of them. The figure draws 1
    recording that the seed selects. It draws the same cube 2 times, from
    above and from below, thus every face is visible.

    The faces are north-referenced: F points north, R east, B south, L west.
    The camera heading does not rotate them.
    """
    )
    return


@app.cell
def _(CUBE_BOROUGH, CUBE_RECORDING, CUBE_SEED, catalog, pl, recordings):
    # Draw 1 recording. A candidate qualifies when it holds all 6 faces, thus
    # the draw takes 64 candidates and keeps the first one that qualifies.
    #
    # Warning: sort before the draw. `unique` does not hold a row order, thus
    # the same seed picks another recording on the next run and the figure of
    # the paper changes under you.
    _pool = (
        recordings
        if CUBE_BOROUGH is None
        else recordings.filter(pl.col("borough") == CUBE_BOROUGH)
    )
    _candidates = (
        [CUBE_RECORDING]
        if CUBE_RECORDING
        else _pool.sort(["dataset", "recording_id"])
        .sample(n=64, seed=CUBE_SEED)["recording_id"]
        .to_list()
    )
    _faces = (
        catalog.filter(pl.col("recording_id").is_in(_candidates))
        .select("recording_id", "face", "image_path")
        .collect()
    )
    _full = (
        _faces.group_by("recording_id")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") == 6)["recording_id"]
        .to_list()
    )
    cube_id = next(r for r in _candidates if r in _full)
    cube_paths = dict(
        _faces.filter(pl.col("recording_id") == cube_id)
        .select("face", "image_path")
        .iter_rows()
    )
    cube_meta = recordings.filter(pl.col("recording_id") == cube_id).row(
        0, named=True
    )
    return cube_id, cube_meta, cube_paths


@app.cell
def _(Affine2D, Image, np, plt):
    # The world frame: x east, y north, z up. Each face carries its outward
    # normal, the direction of the image right edge, and the direction of the
    # image top edge. The 4 street-level faces are north-referenced.
    FACE_FRAME = {
        "F": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
        "R": ((1, 0, 0), (0, -1, 0), (0, 0, 1)),
        "B": ((0, -1, 0), (-1, 0, 0), (0, 0, 1)),
        "L": ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "U": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        "D": ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
    }

    def _camera(az_deg, el_deg):
        """Return the view axis, the screen right axis and the screen up axis."""
        az, el = np.radians(az_deg), np.radians(el_deg)
        view = np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        right = np.cross([0, 0, 1.0], view)
        right /= np.linalg.norm(right)
        return view, right, np.cross(view, right)

    def draw_cube(ax, paths, az, el, half=0.5, label_size=6.0):
        """Draw the cube of 1 recording on `ax`, seen from `az` and `el`.

        The projection is axonometric, thus a square face maps to a
        parallelogram and 1 affine transform carries the whole face image.
        Only the faces that point at the camera are drawn. The function
        returns the 2D corners, which `set_common_scale` needs.
        """
        view, right, up = _camera(az, el)
        corners = []
        for name, (n, rt, tp) in FACE_FRAME.items():
            n, rt, tp = (np.array(a, float) for a in (n, rt, tp))
            if n @ view <= 0:
                continue
            centre = n * half
            to2d = lambda p: np.array([p @ right, p @ up])
            bl = to2d(centre - rt * half - tp * half)
            br = to2d(centre + rt * half - tp * half)
            tl = to2d(centre - rt * half + tp * half)
            tr = to2d(centre + rt * half + tp * half)
            affine = Affine2D(
                matrix=np.array(
                    [
                        [br[0] - bl[0], tl[0] - bl[0], bl[0]],
                        [br[1] - bl[1], tl[1] - bl[1], bl[1]],
                        [0, 0, 1.0],
                    ]
                )
            )
            ax.imshow(
                np.asarray(Image.open(paths[name])),
                extent=(0, 1, 0, 1),
                origin="upper",
                transform=affine + ax.transData,
                interpolation="antialiased",
                zorder=1,
            )
            ax.add_patch(
                plt.Polygon(
                    np.array([bl, br, tr, tl]),
                    closed=True,
                    fill=False,
                    ec="#1a1a1a",
                    lw=0.6,
                    zorder=2,
                )
            )
            ax.text(
                *to2d(centre),
                name,
                ha="center",
                va="center",
                zorder=3,
                fontsize=label_size,
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.14",
                    fc="#1a1a1a",
                    ec="none",
                    alpha=0.72,
                ),
            )
            corners += [bl, br, tr, tl]
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.grid(False)
        return np.array(corners)

    def set_common_scale(axes, corner_sets, pad=0.04):
        """Give every cube the same scale on the page.

        Each axes fits its own cube by default, thus 2 cubes of 1 recording
        get 2 different sizes and the reader sees 2 objects. This function
        applies 1 half-extent to all of them, around the centre of each cube.
        """
        reach = max(
            max(np.ptp(c[:, 0]), np.ptp(c[:, 1])) for c in corner_sets
        ) / 2 + pad
        for ax, corners in zip(axes, corner_sets):
            cx = (corners[:, 0].max() + corners[:, 0].min()) / 2
            cy = (corners[:, 1].max() + corners[:, 1].min()) / 2
            ax.set_xlim(cx - reach, cx + reach)
            ax.set_ylim(cy - reach, cy + reach)

    return draw_cube, set_common_scale


@app.cell
def _(
    CUBE_FIG_SIZE,
    FIG_DIR,
    cube_id,
    cube_paths,
    draw_cube,
    plt,
    set_common_scale,
):
    # 2 views of 1 cube. The first looks down from the north-east and shows
    # F, R and U. The second looks up from the south-west and shows B, L and D.
    cube_fig, _axes = plt.subplots(1, 2, figsize=CUBE_FIG_SIZE)
    _corners = [
        draw_cube(_axes[0], cube_paths, az=35.0, el=22.0),
        draw_cube(_axes[1], cube_paths, az=215.0, el=-22.0),
    ]
    set_common_scale(_axes, _corners)
    cube_fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02)

    cube_png = FIG_DIR / "corpus_cubemap.png"
    cube_pdf = FIG_DIR / "corpus_cubemap.pdf"
    cube_fig.savefig(cube_png, bbox_inches="tight", pad_inches=0.01, dpi=600)
    cube_fig.savefig(cube_pdf, bbox_inches="tight", pad_inches=0.01)
    _ = cube_id
    return cube_fig, cube_pdf, cube_png


@app.cell(hide_code=True)
def _(cube_fig, cube_id, cube_meta, cube_pdf, cube_png, mo):
    mo.md(
        f"""
    Recording **{cube_id}**, dataset `{cube_meta["dataset"]}`, at
    {cube_meta["latitude"]:.5f}, {cube_meta["longitude"]:.5f}, recorded
    {cube_meta["recordedAt"]:%Y-%m-%d %H:%M}.

    Wrote `{cube_png.name}` and `{cube_pdf.name}` to `master/figures/`. The
    figure is {cube_fig.get_size_inches()[0]:.2f} in wide, which is 1 CVPR
    column.
    """
    )
    return


@app.cell(hide_code=True)
def _(cube_fig):
    cube_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 5. The qualifying images for each canonical prompt

    A case selects its images in 1 of 2 ways.

    | Mode | How it selects | Cases |
    |------|----------------|-------|
    | `unit` | Keep a face that looks at 1 unit of a city dataset. The unit is a school, a library, a park, a restaurant or a subway entrance. | 5 |
    | `image` | Keep a face of a citywide stratified sample. No unit exists. | 2 |

    The `unit` filter runs in 2 steps. Step 1 keeps every street-level face
    inside the coverage buffer of a unit. Step 2 is the facing filter: it keeps
    a face only when the ray of that face reaches the unit. A qualifying image
    is a face that passes both steps.
    """
    )
    return


@app.cell
def _(REPO_ROOT):
    # case -> where its images come from. The `unit` cases each hold a curation
    # directory with a materialize manifest; the 2 `image` cases share the
    # citywide stratified manifest.
    CITYWIDE = "data/cyclomedia/cyclomedia_all_2025_citywide_500k.parquet"
    CASE_SPEC = {
        "schools": dict(
            mode="unit",
            source="FacDB K-12 schools",
            curation="curation/facdb_schools_k_12",
            images="curation/facdb_schools_k_12/cyclomedia_near_schools_facing.parquet",
        ),
        "libraries": dict(
            mode="unit",
            source="FacDB libraries",
            curation="curation/facdb_libraries",
            images="curation/facdb_libraries/cyclomedia_near_libraries_facing.parquet",
        ),
        "parks_plazas": dict(
            mode="unit",
            source="FacDB parks and plazas",
            curation="curation/facdb_parks_plazas",
            images="curation/facdb_parks_plazas/cyclomedia_near_parks_facing.parquet",
        ),
        "restaurants": dict(
            mode="unit",
            source="DOHMH inspected restaurants",
            curation="curation/dohmh_restaurants_inspected_all",
            images="curation/dohmh_restaurants_inspected_all/cyclomedia_near_restaurants_facing.parquet",
        ),
        "subway_safety": dict(
            mode="unit",
            source="MTA subway entrances",
            curation="curation/subway_entrances_all",
            images="curation/subway_entrances_all/cyclomedia_near_subway_facing.parquet",
        ),
        "road_quality": dict(
            mode="image",
            source="citywide stratified sample",
            curation=None,
            images=CITYWIDE,
        ),
        "street_photography": dict(
            mode="image",
            source="citywide stratified sample",
            curation=None,
            images=CITYWIDE,
        ),
    }
    CASE_SPEC = {
        k: {**v, "images": str(REPO_ROOT / v["images"]),
            "curation": str(REPO_ROOT / v["curation"]) if v["curation"] else None}
        for k, v in CASE_SPEC.items()
    }
    return (CASE_SPEC,)


@app.cell
def _(C, CASE_SPEC, pl):
    import json

    def _question(case):
        runs = C.runs(kind="proxy", case=case)
        return runs[0].question if runs else ""

    _rows = []
    for _case, _spec in CASE_SPEC.items():
        _lf = pl.scan_parquet(_spec["images"])
        _cols = _lf.collect_schema().names()
        _aggs = [
            pl.len().alias("qualifying_images"),
            pl.col("recording_id").n_unique().alias("recordings"),
        ]
        if "unit_uid" in _cols:
            _aggs.append(pl.col("unit_uid").n_unique().alias("units_with_images"))
        _stat = _lf.select(_aggs).collect().to_dicts()[0]

        _in_buffer = None
        _units_total = None
        if _spec["curation"]:
            _man = json.loads(
                open(f"{_spec['curation']}/cyclomedia_materialize_manifest.json").read()
            )
            _in_buffer = _man["rows"]
            _units_total = _man["units_count"]

        _rows.append(
            {
                "case": _case,
                "question": _question(_case),
                "mode": _spec["mode"],
                "unit_source": _spec["source"],
                "units_published": _units_total,
                "units_with_images": _stat.get("units_with_images"),
                "images_in_buffer": _in_buffer,
                "qualifying_images": _stat["qualifying_images"],
                "recordings": _stat["recordings"],
            }
        )
    cases = pl.DataFrame(_rows, strict=False).with_columns(
        (pl.col("qualifying_images") / pl.col("images_in_buffer") * 100).alias(
            "facing_keep_pct"
        )
    )
    return (cases,)


@app.cell(hide_code=True)
def _(cases, mo, pl):
    _unit = cases.filter(pl.col("mode") == "unit")
    mo.md(
        f"""
    The 5 `unit` cases hold **{int(_unit["qualifying_images"].sum()):,}
    qualifying images** over **{int(_unit["units_with_images"].sum()):,}
    units**. The 2 `image` cases share 1 citywide manifest of
    **{int(cases.filter(pl.col("mode") == "image")["qualifying_images"][0]):,}**
    faces.

    {mo.ui.table(cases.to_pandas(), selection=None)}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 6. The exports""")
    return


@app.cell
def _(
    NOTEBOOK_VERSION,
    OUT_DIR,
    boroughs,
    cases,
    faces_by_borough,
    mo,
    n_faces_all,
    n_faces_used,
    n_recordings,
    pl,
    recordings,
    spacing_table,
):
    _stem = f"corpus_descriptives_v{NOTEBOOK_VERSION}"
    _all = spacing_table.filter(spacing_table["Borough"] == "Whole corpus").row(
        0, named=True
    )
    headline = pl.DataFrame(
        {
            "statistic": [
                "recordings",
                "image_faces_on_disk",
                "image_faces_used",
                "along_drive_step_mean_m",
                "along_drive_step_median_m",
                "nearest_neighbour_mean_m",
                "nearest_neighbour_median_m",
                "first_recorded",
                "last_recorded",
            ],
            "value": [
                f"{n_recordings}",
                f"{n_faces_all}",
                f"{n_faces_used}",
                f"{_all['step_mean_m']:.3f}",
                f"{_all['step_median_m']:.3f}",
                f"{_all['nn_mean_m']:.3f}",
                f"{_all['nn_median_m']:.3f}",
                f"{recordings['recordedAt'].min():%Y-%m-%d}",
                f"{recordings['recordedAt'].max():%Y-%m-%d}",
            ],
        }
    )

    headline.write_csv(OUT_DIR / f"{_stem}_headline.csv")
    boroughs.write_csv(OUT_DIR / f"{_stem}_boroughs.csv")
    spacing_table.write_csv(OUT_DIR / f"{_stem}_spacing.csv")
    faces_by_borough.write_csv(OUT_DIR / f"{_stem}_faces.csv")
    cases.write_csv(OUT_DIR / f"{_stem}_cases.csv")

    mo.md(
        f"""
    Wrote 5 files to `master/outputs/`, with the stem `{_stem}`:
    `_headline.csv`, `_boroughs.csv`, `_spacing.csv`, `_faces.csv`,
    `_cases.csv`.

    {mo.ui.table(headline.to_pandas(), selection=None)}
    """
    )
    return


if __name__ == "__main__":
    app.run()
