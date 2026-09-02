"""Distance reports over the consolidated proxemic events.

The report answers the deployment questions with distance
distributions, one per proxemic event type:

* trash_deposit — how close the robot and the person are at a deposit.
* trash_toss — from what distance people throw trash at the robot.
* gaze_at_robot — at what distance people first look at the robot.
* talk_to_robot — at what distance people talk to the robot.
* annoyance — at what distance people show irritation ("too close").
* avoidance — at what distance people step away.

The distributions describe the observed footage. They are not causal
thresholds. Treat the metre values as model estimates with noise; the
zone counts are the robust view.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .taxonomy import PROXEMIC_EVENTS, ZONE_KEYS

# Report questions in display order: (event key, question text).
REPORT_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("trash_deposit", "How close are people at a trash deposit?"),
    ("trash_toss", "From what distance do people toss trash at TrashBot?"),
    ("gaze_at_robot", "At what distance do people first look at the robot?"),
    ("talk_to_robot", "At what distance do people talk to the robot?"),
    ("annoyance", "At what distance do people show annoyance (too close)?"),
    ("avoidance", "At what distance do people step away from the robot?"),
)

# Column order of the report parquet.
REPORT_COLUMNS: tuple[str, ...] = (
    "group",
    "event_key",
    "event",
    "n_events",
    "n_with_distance",
    "distance_min",
    "distance_q25",
    "distance_median",
    "distance_q75",
    "distance_max",
    *(f"zone_{zone}" for zone in ZONE_KEYS),
)

_LABEL_TO_KEY: dict[str, str] = {label: key for key, label in PROXEMIC_EVENTS.items()}


def event_rows(annotations: pd.DataFrame) -> pd.DataFrame:
    """Return the proxemic event rows with a normalized event_key column."""
    df = annotations[annotations["event"].notna()].copy()
    df["event_key"] = df["event"].map(_LABEL_TO_KEY)
    return df[df["event_key"].notna()]


def _summarize_group(df: pd.DataFrame, group: str) -> list[dict[str, object]]:
    """Summarize one group of event rows into report rows."""
    rows: list[dict[str, object]] = []
    for key, _question in REPORT_QUESTIONS:
        sub = df[df["event_key"] == key]
        distances = pd.to_numeric(sub["distance_m"], errors="coerce").dropna()
        row: dict[str, object] = {
            "group": group,
            "event_key": key,
            "event": PROXEMIC_EVENTS[key],
            "n_events": int(len(sub)),
            "n_with_distance": int(len(distances)),
            "distance_min": float(distances.min()) if len(distances) else None,
            "distance_q25": float(distances.quantile(0.25)) if len(distances) else None,
            "distance_median": float(distances.median()) if len(distances) else None,
            "distance_q75": float(distances.quantile(0.75)) if len(distances) else None,
            "distance_max": float(distances.max()) if len(distances) else None,
        }
        zone_counts = sub["zone"].value_counts()
        for zone in ZONE_KEYS:
            row[f"zone_{zone}"] = int(zone_counts.get(zone, 0))
        rows.append(row)
    return rows


def build_report(
    annotations: pd.DataFrame, group_by: list[str] | None = None
) -> pd.DataFrame:
    """Build the distance report table.

    The "(all)" group always appears. Each value of each group_by
    column adds one more group, named "column=value".
    """
    events = event_rows(annotations)
    rows = _summarize_group(events, "(all)")
    for column in group_by or []:
        if column not in events.columns:
            continue
        for value, sub in events.groupby(column, dropna=True):
            rows.extend(_summarize_group(sub, f"{column}={value}"))
    return pd.DataFrame(rows, columns=pd.Index(REPORT_COLUMNS))


def format_report(report: pd.DataFrame) -> str:
    """Format the report as readable text, one block per group."""
    lines: list[str] = []
    for group, sub in report.groupby("group", sort=False):
        lines.append(f"== {group} ==")
        for row in sub.to_dict("records"):
            question = dict(REPORT_QUESTIONS)[str(row["event_key"])]
            lines.append(f"  {question}")
            n = row["n_events"]
            n_d = row["n_with_distance"]
            if not n:
                lines.append("    No events found.")
                continue
            zones = ", ".join(
                f"{zone}={row[f'zone_{zone}']}"
                for zone in ZONE_KEYS
                if row[f"zone_{zone}"]
            )
            lines.append(f"    events={n} with_distance={n_d} zones: {zones or '-'}")
            if n_d:
                lines.append(
                    "    metres: "
                    f"min={row['distance_min']:.1f} "
                    f"q25={row['distance_q25']:.1f} "
                    f"median={row['distance_median']:.1f} "
                    f"q75={row['distance_q75']:.1f} "
                    f"max={row['distance_max']:.1f}"
                )
        lines.append("")
    return "\n".join(lines)


def write_report(
    annotations_path: str | Path,
    out_path: str | Path,
    group_by: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Read the annotations, build the report, and write the parquet.

    Returns the report frame and its text form.
    """
    annotations = pd.read_parquet(annotations_path)
    report = build_report(annotations, group_by=group_by)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(out, index=False)
    return report, format_report(report)
