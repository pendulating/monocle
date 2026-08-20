---
title: Cyclomedia Depth Maps
category: concept
created: 2026-07-12
updated: 2026-07-29
tags:
  - cyclomedia
  - depth
  - geometry
  - calibration
sources: []
---

# Cyclomedia Depth Maps

Every Cyclomedia recording ships a depth render per cube face at `depthmaps_faces/{F,R,B,L,U,D}.png` — 1024×1024 RGB PNGs, produced by the *same* StreetSmart `PanoramaRendering` call as the RGB faces but with `depthMap=true`. They therefore share the cube geometry in [[guide-cyclomedia-browser]] **exactly, pixel for pixel**: depth and RGB line up with no resampling.

Coverage is **100%** across all 5.24M recordings in the catalog (`depthmap_present` in [[cyclomedia-catalog]]).

## Encoding

Reverse-engineered — **Cyclomedia publishes no spec** for this (the developer portal documents the render API but not the depth format).

```python
code = R * 256 + G      # 16-bit, big-endian across R and G
                        # B channel is unused (identically 0)
code == 0               # NO RETURN — sky, or beyond the depth model
```

Implemented in `dagspaces/common/cyclomedia/depth.py` (`decode_depth`, `NO_RETURN`, `to_metres`).

**Evidence:**
- The blue channel is identically zero on every face.
- The `U` (up) face is **98.7% zeros** — i.e. sky. `D` (down) is **100% valid** — i.e. ground. Exactly what a no-return sentinel predicts.
- Binning the `D` face's code against the ray geometry of a flat ground plane at the known camera height (`groundLevelOffset` ≈ 2.23 m) gives a clean **monotonic** curve, R² = 0.998.
- The stitched depth panorama is visually coherent with the RGB panorama: sky reads as no-return, the road is a smooth gradient, facades are coherent surfaces, avenues recede correctly.

`code` **increases with distance**. That much is solid.

## Metric calibration — RESOLVED 2026-07-29

```python
range_m = (code - 16384) / 250.0      # 4 mm per code step, 0 … 196.6 m
```

`range_m` is the **Euclidean distance from the camera centre along that pixel's ray**, not the perpendicular z-distance to the face plane. (A flat ground plane produces a *varying* code across the `D` face — 16930 at nadir rising to ~17370 at the corners — exactly as range does; z-depth would be constant.)

Exposed as `to_metres()` in `dagspaces/common/cyclomedia/depth.py`. The independent implementation and the calibration scripts live in the `cyclomedia` repo: `pull/depth.py`, `calibration/`, `docs/depth_maps.md`.

### What broke the deadlock

The problem was never the fitting, it was the **anchor**. The `D` face spans only ~2.2–3.9 m, so nothing measured there can separate linear from log from inverse. The fix was to stop using the ground plane entirely and use **known camera baselines** instead.

**1. Scale, from collinear camera pairs.** If camera B sits on the ray camera A shoots along, both rays terminate on the *same* surface point, so `range_A − range_B` must equal the catalogued `|AB|`. This needs no feature matching, and because it differences two codes it is completely blind to the zero-point. Over 893 inlier pairs 4–25 m apart:

> **scale = 249.86 ± 0.17 codes/m**, and it does **not drift with distance** — 249.78 / 249.91 / 249.86 / 249.73 / 250.01 across bins from ~19 m to ~52 m.

That flatness is the result that kills the log and inverse-depth hypotheses. It also lands within 0.06% of a round 250 codes/m, i.e. a 4 mm quantum.

**2. Zero-point, from cross-projection consistency.** Unproject a pixel from camera A, reproject the 3D point into a neighbouring camera B, and require B's *own* depth map to agree at that pixel. Non-collinear rays make the triangle closure sensitive to the zero-point (and nearly flat in scale — the two tests are complementary). Consensus over 30 pairs peaks at **16388**, with 2¹⁴ = 16384 statistically tied.

**3. Facade planarity, as an independent check.** A wrong zero-point bows a flat wall. RMS deviation from the best-fit plane for one facade spanning 18–40 m:

| zero-point | 15800 | 16200 | 16300 | **16384** | 16450 | 16600 | 17000 |
|---|---|---|---|---|---|---|---|
| RMS | 15.6 cm | 6.4 cm | 4.8 cm | **4.0 cm** | 4.2 cm | 6.3 cm | 14.8 cm |

At the calibrated constants a facade comes out planar to ~4 cm RMS and vertical to 2.5°, and the road comes out horizontal to ~1°.

Residual uncertainty: scale to 0.07%, zero-point to about ±8 codes (±3 cm of range bias). Both are well inside the error of the depth model itself, which smooths surfaces and clips near ~120 m even though the encoding reaches 196.6 m.

### ⚠️ Two traps that produce confident wrong numbers

**Do not anchor on `groundLevelOffset`.** This is the trap the earlier analysis fell into. The catalog lists mount heights clustering at 2.2259 m with outlier fleets at 2.3156 and 2.9856 m — but the depth render **ignores them**. The `D`-face nadir code is ~16930 for *every* fleet, and fitting the real observed road 6–25 m out recovers 2.15–2.28 m for all of them, including the 2.99 m fleet. The rendered camera sits at a fixed nominal ~2.18 m above the road. Anchoring the scale there gives 245.76 codes/m, which is **24 σ** from the baseline measurement.

**It is not IEEE float16.** Reading `(R<<8)|G` as a big-endian half float is seductive because the `D` face then lands at 3.06–3.88 m, close to a plausible mount height. It is wrong: observed codes run past 45000, where the float16 sign bit is set, so ~2% of pixels would decode negative or NaN while remaining spatially continuous. It also fails the ground-plane slope test, implying `d = 1.98 + 1.08·r` — a "plane" that does not pass through the camera.

## Working with depth

- **Stitch codes, then colorize.** Colorizing per-face first gives each face its own stretch and makes faces mutually incomparable. `cube_to_equirect` is dtype-agnostic precisely so a `(H,W) uint16` code panorama can be built and colorized once. Compute `vmin`/`vmax` across all six faces together.
- **Never interpolate.** Nearest-neighbour only. Bilinear sampling across a depth discontinuity invents surfaces, and blends the `0` sentinel into real distances.
- **Never treat `0` as "very close."** It is the no-return sentinel and sits at the *bottom* of the numeric range while meaning "infinitely far / unknown." Mask with `code > 0`.
- **`U` and `D` are degenerate by construction.** `U` is nearly all sky (tiny valid fraction); `D` is fully valid but spans a very narrow range (it only ever sees ground a few metres away). Neither is representative — don't compute global statistics over all six faces without accounting for this.
- **`to_metres()` returns NaN for no-return**, not 0 and not a negative number. Decoding the sentinel affinely would give −65.5 m and silently poison any mean. Use `np.nanmedian` / `np.isfinite`.
- **Screen for corrupt faces.** A clean depth face has `B` identically zero. Roughly 4% of recordings have at least one face carrying a rectangular corrupted block with `B > 0`; those pixels decode to garbage. Separately, ~920–970 recordings are missing at least one depth face outright.

## See also

- [[guide-cyclomedia-browser]] — the viewer, and the cube geometry these share
- [[cyclomedia-catalog]] — `depthmap_present`, `groundLevelOffset`, `height`
- [[troubleshooting]] — the analysis traps above, in short form
