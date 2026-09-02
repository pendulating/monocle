"""Command line interface for the TrashBot annotator.

Subcommands:
  manifest     Build a video manifest parquet from one or more data roots.
  run          Run the C++ binary over manifest rows.
  consolidate  Merge JSONL outputs into one long-format parquet.
  speech       Classify transcript utterances (to the robot, about it).
  report       Distance report over the proxemic events.
  embed-frames Cut event keyframes for the repeat-interaction embeddings.
  repeats      Score event pairs by embedding cosine similarity.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .consolidate import consolidate_dir
from .manifest import write_manifest
from .proxemics import write_report
from .runner import DEFAULT_MODEL, run_manifest


def cmd_manifest(args: argparse.Namespace) -> int:
    """Handle the manifest subcommand."""
    df = write_manifest(args.data_root, args.out)
    print(f"Manifest rows: {len(df)}")
    print(f"Wrote {args.out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Handle the run subcommand."""
    manifest = pd.read_parquet(args.manifest)
    results = run_manifest(
        manifest,
        binary=args.binary,
        server_url=args.server_url,
        model=args.model,
        out_dir=args.out_dir,
        fps=args.fps,
        batch=args.batch,
        limit=args.limit,
        overlays_dir=args.overlays_dir,
        max_windows=args.max_windows,
        skip_existing=args.skip_existing,
    )
    failed = [r for r in results if not r.ok]
    for result in failed:
        print(f"FAILED {result.sample_id}: {result.stderr_tail}")
    print(f"Videos processed: {len(results)}; failures: {len(failed)}")
    print(f"JSONL outputs in {args.out_dir}")
    return 1 if failed else 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    """Handle the consolidate subcommand."""
    manifest = pd.read_parquet(args.manifest) if args.manifest else None
    df, warnings = consolidate_dir(args.out_dir, args.out, manifest=manifest)
    for name in sorted(warnings):
        print(f"WARNING {name}: {warnings[name]}")
    print(f"Consolidated rows: {len(df)}")
    print(f"Wrote {args.out}")
    return 0


def cmd_speech(args: argparse.Namespace) -> int:
    """Handle the speech subcommand."""
    from .speech import classify_corpus, consolidate_speech, load_utterance_corpus

    corpus = load_utterance_corpus(args.corpus)
    print(f"Utterances in corpus: {len(corpus)}")
    warnings = classify_corpus(
        corpus,
        server_url=args.server_url,
        out_jsonl=args.jsonl,
        model=args.model,
        prompt_template=args.prompt_template,
        limit=args.limit,
        resume=not args.no_resume,
    )
    for name in sorted(warnings):
        print(f"{name}: {warnings[name]}")
    df = consolidate_speech(args.jsonl, args.out)
    to_robot = int((df["address"] == "to_robot").sum())
    eavesdrop = int(df["eavesdrop"].sum())
    print(f"Speech rows: {len(df)}; to_robot: {to_robot}; eavesdrop: {eavesdrop}")
    print(f"Wrote {args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Handle the report subcommand."""
    report, text = write_report(args.annotations, args.out, group_by=args.by)
    print(text)
    print(f"Report rows: {len(report)}")
    print(f"Wrote {args.out}")
    return 0


def cmd_embed_frames(args: argparse.Namespace) -> int:
    """Handle the embed-frames subcommand."""
    from .repeats import extract_event_frames

    annotations = pd.read_parquet(args.annotations)
    manifest = extract_event_frames(annotations, args.out_dir, limit=args.limit)
    print(f"Keyframes written: {len(manifest)}")
    print(f"Wrote {args.out_dir}/frames_manifest.parquet")
    print(
        "Next: sbatch --export=ALL,FRAMES_MANIFEST=<path>,OUT=<path> "
        "scripts/embed_events.sub"
    )
    return 0


def cmd_repeats(args: argparse.Namespace) -> int:
    """Handle the repeats subcommand."""
    from .repeats import cluster_repeats, find_repeats

    embeddings = pd.read_parquet(args.embeddings)
    frames = pd.read_parquet(args.frames_manifest)
    pairs = find_repeats(
        embeddings,
        frames,
        min_sim=args.min_sim,
        cross_sample_only=not args.within_sample,
    )
    clusters = cluster_repeats(pairs)
    pairs.to_parquet(args.out, index=False)
    clusters_out = str(args.out).replace(".parquet", "_clusters.parquet")
    clusters.to_parquet(clusters_out, index=False)
    cross_day = int((pairs["same_place"] & ~pairs["same_date"]).sum())
    print(f"Pairs over threshold {args.min_sim}: {len(pairs)}")
    print(f"Cross-day pairs at one place: {cross_day}")
    print(f"Clusters: {len(clusters)}")
    print(f"Wrote {args.out} and {clusters_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="trashbot-annotate",
        description="TrashBot deployment footage annotator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("manifest", help="Build a video manifest parquet.")
    p_manifest.add_argument(
        "--data-root",
        required=True,
        action="append",
        help="Root folder of video datasets. Repeat the flag for more roots.",
    )
    p_manifest.add_argument("--out", required=True, help="Output parquet path.")
    p_manifest.set_defaults(func=cmd_manifest)

    p_run = sub.add_parser("run", help="Run the C++ annotator over manifest rows.")
    p_run.add_argument("--manifest", required=True, help="Manifest parquet path.")
    p_run.add_argument("--binary", default="build/trashbot_pipe", help="C++ binary path.")
    p_run.add_argument("--server-url", required=True, help="vLLM base URL, e.g. http://h:8000/v1.")
    p_run.add_argument("--model", default=DEFAULT_MODEL, help="Served model name.")
    p_run.add_argument("--out-dir", required=True, help="Folder for per-video JSONL files.")
    p_run.add_argument("--limit", type=int, default=None, help="Stop after N videos.")
    p_run.add_argument("--fps", type=float, default=None, help="Window sampling fps override.")
    p_run.add_argument("--batch", type=int, default=None, help="Frames per window override.")
    p_run.add_argument("--overlays-dir", default=None, help="Folder for annotated overlay mp4 files.")
    p_run.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Stop each video after N windows. Use for smoke runs.",
    )
    p_run.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows whose JSONL output exists. Use to resume a batch.",
    )
    p_run.set_defaults(func=cmd_run)

    p_cons = sub.add_parser("consolidate", help="Merge JSONL outputs into one parquet.")
    p_cons.add_argument("--out-dir", required=True, help="Folder with per-video JSONL files.")
    p_cons.add_argument("--out", required=True, help="Output parquet path.")
    p_cons.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest parquet for dataset, place, and capture type joins.",
    )
    p_cons.set_defaults(func=cmd_consolidate)

    p_speech = sub.add_parser(
        "speech", help="Classify transcript utterances: to the robot, about it."
    )
    p_speech.add_argument(
        "--corpus",
        default="/share/ju/robot_norms/embeddings/chunk_corpus.parquet",
        help="Utterance corpus parquet (chunk or caption corpus).",
    )
    p_speech.add_argument("--server-url", required=True, help="vLLM base URL.")
    p_speech.add_argument("--model", default=DEFAULT_MODEL, help="Served model name.")
    p_speech.add_argument("--jsonl", required=True, help="Streaming JSONL output path.")
    p_speech.add_argument("--out", required=True, help="Output parquet path.")
    p_speech.add_argument("--prompt-template", default=None, help="Speech prompt override.")
    p_speech.add_argument("--limit", type=int, default=None, help="Stop after N calls.")
    p_speech.add_argument(
        "--no-resume",
        action="store_true",
        help="Classify every utterance again, even ones in the JSONL.",
    )
    p_speech.set_defaults(func=cmd_speech)

    p_report = sub.add_parser("report", help="Distance report over proxemic events.")
    p_report.add_argument("--annotations", required=True, help="Consolidated parquet path.")
    p_report.add_argument("--out", required=True, help="Report parquet path.")
    p_report.add_argument(
        "--by",
        action="append",
        default=None,
        help="Extra grouping column (place, dataset, control_mode). Repeatable.",
    )
    p_report.set_defaults(func=cmd_report)

    p_frames = sub.add_parser(
        "embed-frames", help="Cut event keyframes for the repeat pipeline."
    )
    p_frames.add_argument("--annotations", required=True, help="Consolidated parquet path.")
    p_frames.add_argument("--out-dir", required=True, help="Folder for frames and manifest.")
    p_frames.add_argument("--limit", type=int, default=None, help="Stop after N windows.")
    p_frames.set_defaults(func=cmd_embed_frames)

    p_repeats = sub.add_parser(
        "repeats", help="Score event pairs by embedding cosine similarity."
    )
    p_repeats.add_argument("--embeddings", required=True, help="Embeddings parquet path.")
    p_repeats.add_argument(
        "--frames-manifest", required=True, help="Frames manifest parquet path."
    )
    p_repeats.add_argument("--out", required=True, help="Pairs parquet path.")
    p_repeats.add_argument(
        "--min-sim", type=float, default=0.60, help="Cosine similarity threshold."
    )
    p_repeats.add_argument(
        "--within-sample",
        action="store_true",
        help="Also score pairs inside one clip.",
    )
    p_repeats.set_defaults(func=cmd_repeats)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the trashbot-annotate console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    return int(func(args))


if __name__ == "__main__":
    sys.exit(main())
