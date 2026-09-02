"""Build the run manifest and the shard manifests for one place.

The annotator reads a shard manifest per driver process. This script
turns a full manifest into that input:

1. It keeps the rows for one place.
2. It probes each video for the duration and the height.
3. It gives a proxy path to each video taller than the target height.
   A 4K source costs 24 MB per decoded frame, which the queue cannot
   hold. The model sees about 1 MP per frame, so a proxy changes no
   annotation.
4. It splits the rows into shards of near equal total duration, so the
   drivers finish at about the same time.

The proxy files themselves come from the sbatch script, which runs
ffmpeg over the `src_path` and `video_path` pairs this script writes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

FFPROBE_DEFAULT = "/share/pierson/matt/zoo/ffmpeg-libs/n7.1/bin/ffprobe"
CHUNK_CORPUS = "/share/ju/robot_norms/embeddings/chunk_corpus.parquet"


def probe_env(ffprobe: str) -> dict[str, str]:
    """Add the FFmpeg library folder to the child library search path.

    The zoo build of ffprobe does not find its own libraries without it.
    """
    env = os.environ.copy()
    lib = Path(ffprobe).resolve().parent.parent / "lib"
    if lib.is_dir():
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib}:{existing}" if existing else str(lib)
    return env


def probe(path: str, ffprobe: str) -> tuple[float, int]:
    """Return the duration in seconds and the frame height of a video."""
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=height:format=duration",
        "-of", "json", path,
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True, env=probe_env(ffprobe)
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0.0, 0
    try:
        data = json.loads(out)
        height = int(data["streams"][0]["height"])
        duration = float(data["format"]["duration"])
    except (KeyError, IndexError, ValueError, TypeError):
        return 0.0, 0
    return duration, height


def shard_by_duration(df: pd.DataFrame, n_shards: int) -> list[pd.DataFrame]:
    """Split rows into shards of near equal total duration."""
    order = df.sort_values("duration_s", ascending=False)
    loads = [0.0] * n_shards
    picks: list[list[int]] = [[] for _ in range(n_shards)]
    for pos, row in enumerate(order.to_dict("records")):
        target = loads.index(min(loads))
        picks[target].append(pos)
        loads[target] += float(row["duration_s"])
    return [order.iloc[idx].reset_index(drop=True) for idx in picks]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Full manifest parquet.")
    parser.add_argument("--place", required=True, help="Place slug to keep.")
    parser.add_argument("--out-dir", required=True, help="Folder for the outputs.")
    parser.add_argument("--shards", type=int, default=4, help="Number of shards.")
    parser.add_argument("--max-height", type=int, default=720,
                        help="Taller videos get a proxy path.")
    parser.add_argument("--ffprobe", default=FFPROBE_DEFAULT)
    parser.add_argument("--corpus", default=CHUNK_CORPUS,
                        help="Utterance corpus to filter for the speech channel.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proxies = out_dir / "proxies"

    df = pd.read_parquet(args.manifest)
    df = df[df["place"] == args.place].reset_index(drop=True)
    if df.empty:
        print(f"no rows for place {args.place}", file=sys.stderr)
        return 1
    print(f"{len(df)} video(s) at {args.place}")

    rows = []
    for i, row in enumerate(df.to_dict("records"), start=1):
        duration, height = probe(str(row["src_path" if "src_path" in row else "video_path"]),
                                 args.ffprobe)
        src = str(row["video_path"])
        row["src_path"] = src
        row["duration_s"] = duration
        row["height"] = height
        # A tall source needs a proxy. The path may not exist yet.
        if height > args.max_height:
            row["video_path"] = str(proxies / f"{row['sample_id']}.mp4")
        else:
            row["video_path"] = src
        rows.append(row)
        if i % 20 == 0:
            print(f"  probed {i}/{len(df)}")

    run = pd.DataFrame(rows)

    # A file which does not probe holds no video stream. Some folders keep
    # a truncated file of a few hundred bytes. The annotator cannot read
    # one, so it goes to a separate list instead of into a shard.
    bad = run[run["duration_s"] <= 0]
    if not bad.empty:
        skipped = out_dir / "unreadable.parquet"
        bad.to_parquet(skipped, index=False)
        print(f"warning: {len(bad)} video(s) hold no video stream; see {skipped}")
        for name in bad["sample_id"]:
            print(f"    skipped {name}")
        run = run[run["duration_s"] > 0].reset_index(drop=True)

    run.to_parquet(out_dir / "run_manifest.parquet", index=False)
    print(f"wrote {out_dir / 'run_manifest.parquet'} ({len(run)} rows)")
    print(f"proxies needed: {(run['video_path'] != run['src_path']).sum()}")
    print(f"total footage: {run['duration_s'].sum() / 3600:.1f} h")

    for i, shard in enumerate(shard_by_duration(run, args.shards)):
        path = out_dir / f"shard_{i}.parquet"
        shard.to_parquet(path, index=False)
        print(f"  shard_{i}: {len(shard)} rows, {shard['duration_s'].sum() / 3600:.1f} h")

    # The speech channel reads transcripts, not video. It keeps the rows of
    # every deployment at this place, also the ones whose video we drop.
    corpus_path = Path(args.corpus)
    if corpus_path.is_file():
        corpus = pd.read_parquet(corpus_path)
        subset = corpus[corpus["place"] == args.place].reset_index(drop=True)
        out = out_dir / "corpus.parquet"
        subset.to_parquet(out, index=False)
        print(f"wrote {out} ({len(subset)} utterances)")
    else:
        print(f"warning: no corpus at {corpus_path}; the speech channel needs one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
