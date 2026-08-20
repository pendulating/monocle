#!/usr/bin/env python3
"""Measure the GPU supply of a SLURM partition.

The tool answers 4 questions:

  1. What GPUs does the partition hold?              `inventory`
  2. How many are free, and how often?               `history`
  3. How long does a job wait? How often does it     `waits`
     lose the node to a preemption?
  4. All of the above, as a markdown page.           `report`

`history` and `waits` read the SLURM accounting database with `sacct`, which
keeps about 90 days on this cluster. No poll loop is necessary: the tool
replays the start time and the end time of each job and rebuilds the occupancy
of the past. `snapshot` writes a row of the live state to a CSV, for a cron
that wants a record which does not depend on that replay.

A node of the `gpu` partition also belongs to the partition of its owner. Thus
the tool counts EVERY job on the node, not only the jobs of the partition you
name. The competition for a GPU comes mostly from the owner.

Usage:
  python3 scripts/gpu_census.py inventory
  python3 scripts/gpu_census.py inventory --constraint 'a6000|6000ada|a40|a100|l40s'
  python3 scripts/gpu_census.py history --days 14
  python3 scripts/gpu_census.py history --days 14 --constraint 'a6000|6000ada' --by-hour
  python3 scripts/gpu_census.py waits --days 14
  python3 scripts/gpu_census.py report --days 30 --out vlm-narratives-docs/gpu-partition-census.md
  python3 scripts/gpu_census.py snapshot --csv multirun/gpu_snapshots.csv

Every subcommand takes `--json` and writes the numbers instead of the table.

See `vlm-narratives-docs/gpu-partition-census.md`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# The GPU models of this cluster.
#
# SLURM gives the model as a GRES type name. It does not give the memory, and
# the memory is what decides if a model fits. `short` is the feature tag that
# a `--constraint` uses.
# ---------------------------------------------------------------------------
GPU_INFO = {
    # gres type name                      short        GB   architecture
    "nvidia_b200":                        ("b200",     180, "blackwell"),
    "nvidia_h200_nvl":                    ("h200",     141, "hopper"),
    "nvidia_h100_nvl":                    ("h100",      94, "hopper"),
    "nvidia_rtx_pro_6000_blackwell_server_edition":
                                          ("blackwell", 96, "blackwell"),
    "nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition":
                                          ("6000maxq",  96, "blackwell"),
    "nvidia_a100-sxm4-80gb":              ("a100",      80, "ampere"),
    "nvidia_a100_80gb_pcie":              ("a100",      80, "ampere"),
    "nvidia_a100-pcie-40gb":              ("a100",      40, "ampere"),
    "nvidia_l40s":                        ("l40s",      48, "ada"),
    "nvidia_rtx_6000_ada_generation":     ("6000ada",   48, "ada"),
    "nvidia_rtx_a6000":                   ("a6000",     48, "ampere"),
    "nvidia_a40":                         ("a40",       48, "ampere"),
    "tesla_v100-sxm3-32gb-h":             ("v100",      32, "volta"),
    "tesla_v100s-pcie-32gb":              ("v100",      32, "volta"),
    "nvidia_rtx_a5500":                   ("a5500",     24, "ampere"),
    "nvidia_rtx_a5000":                   ("a5000",     24, "ampere"),
    "nvidia_geforce_rtx_3090":            ("3090",      24, "ampere"),
    "nvidia_titan_rtx":                   ("titanrtx",  24, "turing"),
    "quadro_rtx_6000":                    ("quadro6000", 24, "turing"),
    "nvidia_l4":                          ("l4",        24, "ada"),
    "tesla_t4":                           ("t4",        16, "turing"),
    "nvidia_titan_xp":                    ("titanxp",   12, "pascal"),
    "nvidia_titan_x_pascal":              ("titanxp",   12, "pascal"),
    "nvidia_titan_x":                     ("titanxp",   12, "pascal"),
    "nvidia_geforce_gtx_titan_x":         ("titanx",    12, "maxwell"),
    "nvidia_geforce_rtx_2080_ti":         ("2080ti",    11, "turing"),
    "nvidia_geforce_gtx_1080_ti":         ("1080ti",    11, "pascal"),
}

UNKNOWN = ("?", 0, "?")


def gpu_meta(gres_type: str):
    return GPU_INFO.get(gres_type, UNKNOWN)


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------
def sh(args) -> str:
    """Run a command and give its stdout. An empty string means it failed."""
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[census] {args[0]} failed: {exc}", file=sys.stderr)
        return ""
    if p.returncode != 0 and not p.stdout:
        print(f"[census] {args[0]} failed: {p.stderr.strip()[:400]}", file=sys.stderr)
        return ""
    return p.stdout


_HOST_CACHE: dict[str, list[str]] = {}


def expand_hostlist(spec: str) -> list[str]:
    """Give the node names of a SLURM host list such as `nikola-compute-[01-03]`."""
    spec = (spec or "").strip()
    if not spec or spec in ("None", "None assigned"):
        return []
    if "[" not in spec:
        return [h for h in spec.split(",") if h]
    hit = _HOST_CACHE.get(spec)
    if hit is None:
        hit = [h for h in sh(["scontrol", "show", "hostnames", spec]).split() if h]
        _HOST_CACHE[spec] = hit
    return hit


# ---------------------------------------------------------------------------
# The live inventory
# ---------------------------------------------------------------------------
_GRES_RE = re.compile(r"(?:^|,)gpu:(?:([A-Za-z0-9_.\-]+):)?(\d+)")


def parse_gres(field: str) -> dict[str, int]:
    """Give {gres type: count} of the GPUs in a `sinfo %G` field.

    A `shard:` entry is not a GPU that a job can hold alone, thus we drop it.
    """
    out: dict[str, int] = {}
    for typ, count in _GRES_RE.findall(field or ""):
        out[typ or "untyped"] = out.get(typ or "untyped", 0) + int(count)
    return out


class Node:
    __slots__ = ("name", "gres", "features", "cpus", "mem_mb", "state")

    def __init__(self, name, gres, features, cpus, mem_mb, state):
        self.name = name
        self.gres = gres
        self.features = features
        self.cpus = cpus
        self.mem_mb = mem_mb
        self.state = state

    @property
    def gpus(self) -> int:
        return sum(self.gres.values())

    @property
    def main_type(self) -> str:
        """The GPU model of the node. A mixed node gives its largest group."""
        if not self.gres:
            return ""
        return max(self.gres.items(), key=lambda kv: kv[1])[0]


def read_nodes(partition: str) -> dict[str, Node]:
    """Give the GPU nodes of a partition, keyed by node name."""
    raw = sh(["sinfo", "-h", "-p", partition, "-N", "-o", "%N|%G|%f|%c|%m|%t"])
    nodes: dict[str, Node] = {}
    for line in raw.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        name, gres, feats, cpus, mem, state = (p.strip() for p in parts[:6])
        gres_map = parse_gres(gres)
        if not gres_map:
            continue  # a CPU node holds no GPU
        nodes[name] = Node(
            name=name,
            gres=gres_map,
            features={f for f in feats.split(",") if f},
            cpus=int(cpus) if cpus.isdigit() else 0,
            mem_mb=int(mem) if mem.isdigit() else 0,
            state=state,
        )
    return nodes


def constraint_ok(features: set[str], expr: str) -> bool:
    """Test a node against a SLURM feature expression.

    `|` is OR and `&` or `,` is AND, as `sbatch --constraint` reads them. A
    count such as `[a6000*2]` is not supported, and the tool tells you so.
    """
    if not expr:
        return True
    for or_term in expr.split("|"):
        want = [t.strip() for t in re.split(r"[&,]", or_term) if t.strip()]
        if want and all(t in features for t in want):
            return True
    return False


def select_nodes(nodes: dict[str, Node], constraint: str,
                 min_vram: int = 0) -> dict[str, Node]:
    """Keep the nodes that match the constraint AND hold a large enough GPU."""
    out = {}
    for n, nd in nodes.items():
        if constraint and not constraint_ok(nd.features, constraint):
            continue
        if min_vram and not any(gpu_meta(t)[1] >= min_vram for t in nd.gres):
            continue
        out[n] = nd
    return out


def suggest_constraint(nodes: dict[str, Node], min_vram: int) -> str:
    """Give the SLURM feature expression that reaches these nodes.

    A hand-written list of models goes stale: the cluster adds a card and the
    list does not know it. This builds the list from the memory of each model.
    """
    tags: set[str] = set()
    for nd in nodes.values():
        for typ in nd.gres:
            short = gpu_meta(typ)[1] >= min_vram and gpu_meta(typ)[0]
            if short and short in nd.features:
                tags.add(short)
    return "|".join(sorted(tags))


def live_alloc() -> dict[str, dict[str, int]]:
    """Give the GPUs that each node holds now, as {node: {gres type: count}}.

    `scontrol show node` gives AllocTRES, which names the model.
    """
    raw = sh(["scontrol", "show", "node", "--oneliner"])
    out: dict[str, dict[str, int]] = {}
    for line in raw.splitlines():
        m = re.search(r"NodeName=(\S+)", line)
        if not m:
            continue
        node = m.group(1)
        m2 = re.search(r"AllocTRES=(\S*)", line)
        held: dict[str, int] = {}
        if m2:
            for typ, count in re.findall(r"gres/gpu:([^=,]+)=(\d+)", m2.group(1)):
                held[typ] = held.get(typ, 0) + int(count)
        out[node] = held
    return out


# ---------------------------------------------------------------------------
# The accounting history
# ---------------------------------------------------------------------------
SACCT_FIELDS = "JobID,User,Partition,NodeList,AllocTRES,ReqTRES,Submit,Start,End,State,Elapsed"


class Job:
    __slots__ = ("jobid", "user", "partition", "nodelist", "gpus", "req_gpus",
                 "submit", "start", "end", "state")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def _ts(text: str):
    text = (text or "").strip()
    if not text or text in ("Unknown", "None", "N/A"):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def _tres_gpus(field: str) -> dict[str, int]:
    """Give {gres type: count} of the GPUs of a TRES string."""
    out: dict[str, int] = {}
    for typ, count in re.findall(r"gres/gpu:([^=,]+)=(\d+)", field or ""):
        out[typ] = out.get(typ, 0) + int(count)
    if out:
        return out
    m = re.search(r"(?:^|,)gres/gpu=(\d+)", field or "")
    if m and int(m.group(1)) > 0:
        return {"untyped": int(m.group(1))}
    return {}


def read_jobs(days: float, nodes: dict[str, Node], end: datetime | None = None) -> list[Job]:
    """Read every job that touched these nodes in the window."""
    nodelist = ",".join(sorted(nodes))
    args = ["sacct", "-a", "-X", "-P", "-n", "-N", nodelist,
            "-S", f"now-{_hours(days)}hours", "-o", SACCT_FIELDS]
    if end is not None:
        args += ["-E", end.strftime("%Y-%m-%dT%H:%M:%S")]
    raw = sh(args)
    jobs: list[Job] = []
    for line in raw.splitlines():
        f = line.split("|")
        if len(f) < 11:
            continue
        jobs.append(Job(
            jobid=f[0], user=f[1], partition=f[2], nodelist=f[3],
            gpus=_tres_gpus(f[4]), req_gpus=_tres_gpus(f[5]),
            submit=_ts(f[6]), start=_ts(f[7]), end=_ts(f[8]),
            state=f[9].split()[0] if f[9] else "",
        ))
    return jobs


def _hours(days: float) -> int:
    return max(1, int(round(days * 24)))


# ---------------------------------------------------------------------------
# The occupancy replay
# ---------------------------------------------------------------------------
def occupancy(jobs, nodes, t0: datetime, t1: datetime, step_s: int = 300):
    """Rebuild how many GPUs of each model were busy over the window.

    The result is a grid of samples, 1 for each `step_s` seconds. A job that
    spans more than 1 node gives its GPUs to the nodes in equal parts, which is
    an approximation for a multi-node job.
    """
    events: dict[str, list[tuple[float, float]]] = defaultdict(list)
    e0, e1 = t0.timestamp(), t1.timestamp()
    dropped = 0
    for j in jobs:
        if not j.gpus or j.start is None:
            continue
        hosts = expand_hostlist(j.nodelist)
        if not hosts:
            continue
        hits = [h for h in hosts if h in nodes]
        if not hits:
            continue
        share = len(hits) / len(hosts)
        s = max(e0, j.start.timestamp())
        e = min(e1, (j.end or t1).timestamp())
        if e <= s:
            continue
        for typ, n in j.gpus.items():
            if typ == "untyped":
                kinds = {nodes[h].main_type for h in hits}
                if len(kinds) != 1:
                    dropped += 1
                    continue
                typ = kinds.pop()
            events[typ].append((s, n * share))
            events[typ].append((e, -n * share))

    # Sample at the MIDDLE of each period, not at its edge. A job starts and
    # ends on a whole minute, thus a sample on the edge hits the event itself
    # and counts a job that ends exactly there as already gone.
    n_samples = max(1, int((e1 - e0) // step_s))
    grid = [e0 + (k + 0.5) * step_s for k in range(n_samples)]
    series: dict[str, list[float]] = {}
    for typ, evs in events.items():
        evs.sort()
        times = [t for t, _ in evs]
        cum, acc = [], 0.0
        for _, d in evs:
            acc += d
            cum.append(acc)
        out = []
        for t in grid:
            i = bisect_right(times, t) - 1
            out.append(cum[i] if i >= 0 else 0.0)
        series[typ] = out
    return grid, series, dropped


def capacity_by_type(nodes: dict[str, Node]) -> dict[str, int]:
    cap: dict[str, int] = {}
    for nd in nodes.values():
        for typ, n in nd.gres.items():
            cap[typ] = cap.get(typ, 0) + n
    return cap


def free_series(alloc: list[float], cap: int) -> list[float]:
    return [max(0.0, cap - a) for a in alloc]


def pct_at_least(free: list[float], k: int) -> float:
    if not free:
        return 0.0
    return 100.0 * sum(1 for v in free if v >= k) / len(free)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def table(rows: list[list[str]], head: list[str]) -> str:
    """Render a markdown table that is also readable in a terminal."""
    cols = len(head)
    width = [len(h) for h in head]
    for r in rows:
        for i in range(cols):
            width[i] = max(width[i], len(str(r[i])))
    def line(cells):
        return "| " + " | ".join(str(c).ljust(width[i]) for i, c in enumerate(cells)) + " |"
    out = [line(head), "|" + "|".join("-" * (w + 2) for w in width) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Subcommand: inventory
# ---------------------------------------------------------------------------
def cmd_inventory(a) -> None:
    nodes = read_nodes(a.partition)
    sel = select_nodes(nodes, a.constraint, a.min_vram)
    if not sel:
        print(f"no GPU node of partition `{a.partition}` matches {a.constraint!r}")
        return
    held = live_alloc()

    per_type: dict[str, dict] = defaultdict(lambda: {"nodes": 0, "gpus": 0, "alloc": 0})
    for nd in sel.values():
        for typ, n in nd.gres.items():
            rec = per_type[typ]
            rec["nodes"] += 1
            rec["gpus"] += n
            rec["alloc"] += min(n, held.get(nd.name, {}).get(typ, 0))

    rows = []
    for typ, rec in sorted(per_type.items(), key=lambda kv: -kv[1]["gpus"]):
        short, gb, arch = gpu_meta(typ)
        free = rec["gpus"] - rec["alloc"]
        rows.append([short, f"{gb} GB", arch, rec["nodes"], rec["gpus"],
                     rec["alloc"], free,
                     f"{100.0 * rec['alloc'] / rec['gpus']:.0f}%", typ])
    total_g = sum(r["gpus"] for r in per_type.values())
    total_a = sum(r["alloc"] for r in per_type.values())

    if a.json:
        print(json.dumps({"partition": a.partition, "constraint": a.constraint,
                          "nodes": len(sel), "gpus": total_g, "alloc": total_a,
                          "by_type": {t: dict(v) for t, v in per_type.items()}}, indent=2))
        return

    print(f"# GPU inventory — partition `{a.partition}`"
          + (f", constraint `{a.constraint}`" if a.constraint else "")
          + (f", {a.min_vram} GB or more" if a.min_vram else ""))
    if a.min_vram:
        print(f"\nEquivalent SLURM constraint: `{suggest_constraint(sel, a.min_vram)}`")
    print(f"\n{len(sel)} nodes, {total_g} GPUs, {total_a} busy now, {total_g - total_a} free\n")
    print(table(rows, ["model", "memory", "arch", "nodes", "GPUs",
                       "busy", "free", "used", "gres type"]))

    if a.by_node:
        print("\n## By node\n")
        nrows = []
        for name, nd in sorted(sel.items()):
            busy = sum(held.get(name, {}).values())
            nrows.append([name, gpu_meta(nd.main_type)[0], nd.gpus, busy,
                          nd.gpus - busy, nd.cpus, f"{nd.mem_mb // 1024} GB", nd.state])
        print(table(nrows, ["node", "model", "GPUs", "busy", "free",
                            "CPUs", "RAM", "state"]))


# ---------------------------------------------------------------------------
# Subcommand: history
# ---------------------------------------------------------------------------
def collect_history(a):
    nodes = read_nodes(a.partition)
    sel = select_nodes(nodes, a.constraint, a.min_vram)
    if not sel:
        raise SystemExit(f"no GPU node of `{a.partition}` matches {a.constraint!r}")
    t1 = datetime.now()
    t0 = t1 - timedelta(hours=_hours(a.days))
    jobs = read_jobs(a.days, sel)
    grid, series, dropped = occupancy(jobs, sel, t0, t1, a.step)
    cap = capacity_by_type(sel)
    return sel, jobs, grid, series, dropped, cap, t0, t1


def history_stats(series, cap, grid, thresholds):
    """Give the per-model statistics and the statistics of the whole pool."""
    per_type = {}
    for typ, total in sorted(cap.items(), key=lambda kv: -kv[1]):
        alloc = series.get(typ, [0.0] * len(grid))
        free = free_series(alloc, total)
        per_type[typ] = {
            "capacity": total,
            "mean_busy": statistics.fmean(alloc) if alloc else 0.0,
            "mean_free": statistics.fmean(free) if free else 0.0,
            "p10_free": quantile(free, 0.10),
            "p50_free": quantile(free, 0.50),
            "p90_free": quantile(free, 0.90),
            "pct_free_ge": {k: pct_at_least(free, k) for k in thresholds},
            "_free": free,
        }
    pool_cap = sum(cap.values())
    pool_alloc = [0.0] * len(grid)
    for typ in cap:
        s = series.get(typ)
        if not s:
            continue
        for i, v in enumerate(s):
            pool_alloc[i] += v
    pool_free = free_series(pool_alloc, pool_cap)
    pool = {
        "capacity": pool_cap,
        "mean_busy": statistics.fmean(pool_alloc) if pool_alloc else 0.0,
        "mean_free": statistics.fmean(pool_free) if pool_free else 0.0,
        "p10_free": quantile(pool_free, 0.10),
        "p50_free": quantile(pool_free, 0.50),
        "p90_free": quantile(pool_free, 0.90),
        "pct_free_ge": {k: pct_at_least(pool_free, k) for k in thresholds},
        "_free": pool_free,
    }
    return per_type, pool


def cmd_history(a) -> None:
    sel, jobs, grid, series, dropped, cap, t0, t1 = collect_history(a)
    thresholds = a.thresholds
    per_type, pool = history_stats(series, cap, grid, thresholds)

    if a.json:
        out = {"partition": a.partition, "constraint": a.constraint,
               "days": a.days, "step_s": a.step, "samples": len(grid),
               "pool": {k: v for k, v in pool.items() if not k.startswith("_")},
               "by_type": {gpu_meta(t)[0] + "/" + t:
                           {k: v for k, v in s.items() if not k.startswith("_")}
                           for t, s in per_type.items()}}
        print(json.dumps(out, indent=2))
        return

    print(f"# GPU availability — partition `{a.partition}`"
          + (f", constraint `{a.constraint}`" if a.constraint else ""))
    print(f"\n{t0:%Y-%m-%d %H:%M} to {t1:%Y-%m-%d %H:%M} "
          f"({a.days:g} days, {len(grid)} samples, 1 for each {a.step}s)")
    print(f"{len(sel)} nodes, {pool['capacity']} GPUs, {len(jobs)} jobs replayed"
          + (f", {dropped} GPU claims of unknown model dropped" if dropped else ""))
    print(f"\nMean occupancy {100.0 * pool['mean_busy'] / max(1, pool['capacity']):.1f}%. "
          f"A free GPU is one that no job holds; a job of yours still has to "
          f"outrank the queue to reach it.\n")

    head = ["model", "GPUs", "mean free", "p10", "median", "p90"] + \
           [f"≥{k} free" for k in thresholds]
    rows = []
    for typ, s in per_type.items():
        rows.append([gpu_meta(typ)[0], s["capacity"], f"{s['mean_free']:.1f}",
                     f"{s['p10_free']:.0f}", f"{s['p50_free']:.0f}",
                     f"{s['p90_free']:.0f}"]
                    + [f"{s['pct_free_ge'][k]:.0f}%" for k in thresholds])
    rows.append(["POOL", pool["capacity"], f"{pool['mean_free']:.1f}",
                 f"{pool['p10_free']:.0f}", f"{pool['p50_free']:.0f}",
                 f"{pool['p90_free']:.0f}"]
                + [f"{pool['pct_free_ge'][k]:.0f}%" for k in thresholds])
    print(table(rows, head))
    print("\n`≥N free` is the part of the window when N or more GPUs held no job.")

    if a.by_hour:
        print("\n## Free GPUs of the pool, by hour of the day (local time)\n")
        by_h: dict[int, list[float]] = defaultdict(list)
        for t, v in zip(grid, pool["_free"]):
            by_h[datetime.fromtimestamp(t).hour].append(v)
        hrows = []
        for h in range(24):
            vals = by_h.get(h, [])
            if not vals:
                continue
            mean = statistics.fmean(vals)
            bar = "#" * int(round(40.0 * mean / max(1.0, pool["capacity"])))
            hrows.append([f"{h:02d}:00", f"{mean:.1f}",
                          f"{quantile(vals, 0.10):.0f}",
                          f"{quantile(vals, 0.90):.0f}", bar])
        print(table(hrows, ["hour", "mean free", "p10", "p90", ""]))

        print("\n## Free GPUs of the pool, by day of the week\n")
        by_d: dict[int, list[float]] = defaultdict(list)
        for t, v in zip(grid, pool["_free"]):
            by_d[datetime.fromtimestamp(t).weekday()].append(v)
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        drows = []
        for d in range(7):
            vals = by_d.get(d, [])
            if not vals:
                continue
            drows.append([names[d], f"{statistics.fmean(vals):.1f}",
                          f"{quantile(vals, 0.10):.0f}", f"{quantile(vals, 0.90):.0f}"])
        print(table(drows, ["day", "mean free", "p10", "p90"]))


# ---------------------------------------------------------------------------
# Subcommand: waits
# ---------------------------------------------------------------------------
def wait_and_preempt(jobs, nodes, partition, user=None):
    """Give the queue delay and the preemption rate of the GPU jobs."""
    waits: list[tuple[int, float]] = []          # (GPUs asked, seconds waited)
    runtimes: list[float] = []
    states = Counter()
    preempt_nodes = Counter()
    preempt_life: list[float] = []
    for j in jobs:
        if not j.gpus:
            continue
        if j.partition != partition:
            continue
        if user and j.user != user:
            continue
        n = sum(j.gpus.values())
        states[j.state] += 1
        if j.start and j.submit:
            waits.append((n, max(0.0, (j.start - j.submit).total_seconds())))
        if j.start and j.end:
            life = (j.end - j.start).total_seconds()
            runtimes.append(life)
            if j.state == "PREEMPTED":
                preempt_life.append(life)
                for h in expand_hostlist(j.nodelist):
                    if h in nodes:
                        preempt_nodes[h] += 1
    return waits, runtimes, states, preempt_nodes, preempt_life


def cmd_waits(a) -> None:
    nodes = read_nodes(a.partition)
    sel = select_nodes(nodes, a.constraint, a.min_vram)
    jobs = read_jobs(a.days, sel)
    waits, runtimes, states, pnodes, plife = wait_and_preempt(
        jobs, sel, a.partition, a.user)

    total = sum(states.values())
    npre = states.get("PREEMPTED", 0)
    secs = [w for _, w in waits]

    if a.json:
        print(json.dumps({
            "partition": a.partition, "days": a.days, "user": a.user,
            "jobs": total, "preempted": npre,
            "preempt_rate": (npre / total if total else 0.0),
            "wait_s": {"median": quantile(secs, 0.5), "p90": quantile(secs, 0.9),
                       "mean": statistics.fmean(secs) if secs else 0.0},
            "states": dict(states)}, indent=2))
        return

    print(f"# Queue delay and preemption — partition `{a.partition}`"
          + (f", user `{a.user}`" if a.user else ""))
    print(f"\n{total} GPU jobs in the last {a.days:g} days"
          + (f", constraint `{a.constraint}`" if a.constraint else "") + "\n")
    if not total:
        print("No job matches. A job that never started has no delay to report.")
        return

    print("## Delay from submit to start\n")
    print(table([[f"{quantile(secs, 0.5) / 60:.1f} min",
                  f"{quantile(secs, 0.9) / 60:.1f} min",
                  f"{(statistics.fmean(secs) if secs else 0) / 60:.1f} min",
                  f"{100.0 * sum(1 for s in secs if s < 300) / max(1, len(secs)):.0f}%",
                  len(secs)]],
                ["median", "p90", "mean", "under 5 min", "jobs"]))
    print("\nWarning: a job that still waits has no start time, thus it is absent. "
          "The true delay is longer than this table shows.")

    buckets = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 999)]
    brows = []
    for lo, hi in buckets:
        vals = [w for n, w in waits if lo <= n <= hi]
        if not vals:
            continue
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi == 999 else f"{lo}-{hi}")
        brows.append([label, len(vals), f"{quantile(vals, 0.5) / 60:.1f} min",
                      f"{quantile(vals, 0.9) / 60:.1f} min"])
    if brows:
        print("\n### By the number of GPUs the job asks for\n")
        print(table(brows, ["GPUs", "jobs", "median wait", "p90 wait"]))

    print("\n## How the jobs ended\n")
    srows = [[s, c, f"{100.0 * c / total:.1f}%"]
             for s, c in states.most_common(10)]
    print(table(srows, ["state", "jobs", "share"]))

    if npre:
        print(f"\n**Preemption rate {100.0 * npre / total:.1f}%.** "
              f"A preempted job ran {hms(quantile(plife, 0.5))} at the median "
              f"and {hms(quantile(plife, 0.1))} at the 10th percentile "
              f"before it lost the node.")
        print("\n### The nodes that took the job back most often\n")
        prows = [[n, c, gpu_meta(sel[n].main_type)[0] if n in sel else "?"]
                 for n, c in pnodes.most_common(12)]
        print(table(prows, ["node", "preemptions", "model"]))
    if runtimes:
        print(f"\nMedian job life on this partition: {hms(quantile(runtimes, 0.5))}.")


# ---------------------------------------------------------------------------
# Subcommand: snapshot
# ---------------------------------------------------------------------------
def cmd_snapshot(a) -> None:
    """Append 1 row for each GPU model to a CSV. Made for a cron."""
    nodes = read_nodes(a.partition)
    sel = select_nodes(nodes, a.constraint, a.min_vram)
    held = live_alloc()
    now = datetime.now().replace(microsecond=0).isoformat()

    per_type: dict[str, dict] = defaultdict(lambda: {"nodes": 0, "gpus": 0,
                                                     "alloc": 0, "idle_nodes": 0})
    for nd in sel.values():
        busy_here = held.get(nd.name, {})
        for typ, n in nd.gres.items():
            rec = per_type[typ]
            rec["nodes"] += 1
            rec["gpus"] += n
            rec["alloc"] += min(n, busy_here.get(typ, 0))
        if not sum(busy_here.values()):
            per_type[nd.main_type]["idle_nodes"] += 1

    new = not os.path.exists(a.csv)
    os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
    with open(a.csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "partition", "constraint", "model",
                        "gres_type", "nodes", "gpus", "busy", "free", "idle_nodes"])
        for typ, rec in sorted(per_type.items()):
            w.writerow([now, a.partition, a.constraint, gpu_meta(typ)[0], typ,
                        rec["nodes"], rec["gpus"], rec["alloc"],
                        rec["gpus"] - rec["alloc"], rec["idle_nodes"]])
    if not a.quiet:
        print(f"[census] appended {len(per_type)} rows to {a.csv}")


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------
def partition_meta(partition: str) -> dict:
    raw = sh(["scontrol", "show", "partition", partition])
    out = {}
    for key in ("PriorityTier", "PreemptMode", "MaxTime", "DefaultTime",
                "TotalNodes", "State"):
        m = re.search(rf"{key}=(\S+)", raw)
        if m:
            out[key] = m.group(1)
    return out


def neighbour_partitions(nodes: dict[str, Node], skip: str) -> list[tuple[str, int, int]]:
    """Give the partitions that share these nodes, with their priority tier.

    A partition of a higher tier takes the node back from a lower one.
    """
    raw = sh(["scontrol", "show", "partition", "--oneliner"])
    out = []
    for line in raw.splitlines():
        m = re.search(r"PartitionName=(\S+)", line)
        t = re.search(r"PriorityTier=(\d+)", line)
        n = re.search(r"\sNodes=(\S+)", line)
        if not (m and t and n):
            continue
        name = m.group(1)
        if name == skip or name.endswith("-interactive"):
            continue
        shared = sum(1 for h in expand_hostlist(n.group(1)) if h in nodes)
        if shared:
            out.append((name, int(t.group(1)), shared))
    return sorted(out, key=lambda r: (-r[1], -r[2]))


def cmd_report(a) -> None:
    sel, jobs, grid, series, dropped, cap, t0, t1 = collect_history(a)
    per_type, pool = history_stats(series, cap, grid, a.thresholds)
    meta = partition_meta(a.partition)
    held = live_alloc()

    lines: list[str] = []
    W = lines.append
    W(f"# GPU census — partition `{a.partition}`")
    W("")
    W(f"Written by `scripts/gpu_census.py report` on {t1:%Y-%m-%d %H:%M}. "
      f"The window is {a.days:g} days, from {t0:%Y-%m-%d} to {t1:%Y-%m-%d}.")
    if a.constraint:
        W("")
        W(f"Constraint: `{a.constraint}`.")
    W("")
    W("## What the partition is")
    W("")
    W(table([[k, v] for k, v in meta.items()], ["field", "value"]))
    W("")
    W("A partition of a HIGHER priority tier takes a node back from a lower "
      "one. These partitions share the nodes of this one. `scontrol` shows "
      "only the partitions that you may use, thus the owner partition that "
      "preempts you is often absent from this table:")
    W("")
    nb = neighbour_partitions(sel, a.partition)
    W(table([[n, t, c, "takes the node from you" if t > int(meta.get("PriorityTier", 0))
              else "you take the node from it"] for n, t, c in nb[:15]],
            ["partition", "tier", "shared nodes", "effect"]))
    W("")

    W("## What GPUs it holds")
    W("")
    inv: dict[str, dict] = defaultdict(lambda: {"nodes": 0, "gpus": 0, "alloc": 0})
    for nd in sel.values():
        for typ, n in nd.gres.items():
            rec = inv[typ]
            rec["nodes"] += 1
            rec["gpus"] += n
            rec["alloc"] += min(n, held.get(nd.name, {}).get(typ, 0))
    rows = []
    for typ, rec in sorted(inv.items(), key=lambda kv: -kv[1]["gpus"]):
        short, gb, arch = gpu_meta(typ)
        rows.append([short, f"{gb} GB", arch, rec["nodes"], rec["gpus"],
                     rec["gpus"] - rec["alloc"], typ])
    W(table(rows, ["model", "memory", "arch", "nodes", "GPUs", "free now", "gres type"]))
    W("")
    W(f"Total: {len(sel)} nodes and {pool['capacity']} GPUs.")
    W("")

    W("## How often a GPU is free")
    W("")
    W(f"The tool replayed {len(jobs)} jobs and sampled the result every "
      f"{a.step} seconds. It counts EVERY job on the node, thus the owner of "
      f"the node is in these numbers too.")
    W("")
    head = ["model", "GPUs", "mean free", "p10", "median", "p90"] + \
           [f"≥{k} free" for k in a.thresholds]
    rows = []
    for typ, s in per_type.items():
        rows.append([gpu_meta(typ)[0], s["capacity"], f"{s['mean_free']:.1f}",
                     f"{s['p10_free']:.0f}", f"{s['p50_free']:.0f}",
                     f"{s['p90_free']:.0f}"]
                    + [f"{s['pct_free_ge'][k]:.0f}%" for k in a.thresholds])
    rows.append(["POOL", pool["capacity"], f"{pool['mean_free']:.1f}",
                 f"{pool['p10_free']:.0f}", f"{pool['p50_free']:.0f}",
                 f"{pool['p90_free']:.0f}"]
                + [f"{pool['pct_free_ge'][k]:.0f}%" for k in a.thresholds])
    W(table(rows, head))
    W("")
    W(f"Mean occupancy of the pool: "
      f"{100.0 * pool['mean_busy'] / max(1, pool['capacity']):.1f}%.")
    W("")

    W("### By hour of the day (local time)")
    W("")
    by_h: dict[int, list[float]] = defaultdict(list)
    for t, v in zip(grid, pool["_free"]):
        by_h[datetime.fromtimestamp(t).hour].append(v)
    hrows = []
    for h in range(24):
        vals = by_h.get(h, [])
        if vals:
            mean = statistics.fmean(vals)
            hrows.append([f"{h:02d}:00", f"{mean:.1f}",
                          f"{quantile(vals, 0.10):.0f}", f"{quantile(vals, 0.90):.0f}",
                          "#" * int(round(40.0 * mean / max(1.0, pool["capacity"])))])
    W(table(hrows, ["hour", "mean free", "p10", "p90", ""]))
    W("")

    W("## What a job of yours can expect")
    W("")
    waits, runtimes, states, pnodes, plife = wait_and_preempt(jobs, sel, a.partition)
    total = sum(states.values())
    if total:
        secs = [w for _, w in waits]
        npre = states.get("PREEMPTED", 0)
        W(table([[total, f"{quantile(secs, 0.5) / 60:.1f} min",
                  f"{quantile(secs, 0.9) / 60:.1f} min",
                  f"{100.0 * npre / total:.1f}%",
                  hms(quantile(plife, 0.5)) if plife else "n/a",
                  hms(quantile(runtimes, 0.5)) if runtimes else "n/a"]],
                ["GPU jobs", "median wait", "p90 wait", "preempted",
                 "median life before preemption", "median job life"]))
        W("")
        if pnodes:
            W("The nodes that took a job back most often:")
            W("")
            W(table([[n, c, gpu_meta(sel[n].main_type)[0]]
                     for n, c in pnodes.most_common(10)],
                    ["node", "preemptions", "model"]))
            W("")
    else:
        W("No job of this partition ran on these nodes in the window.")
        W("")

    W("## How to read this")
    W("")
    W("- A free GPU holds no job. Your job must still outrank the queue.")
    W("- A constraint selects a NODE, not a GPU. A node that matches can hold "
      "a second model, and an untyped `--gres=gpu:1` can land on it. The "
      "table above shows every model of every node that matched.")
    W("- A job that spans more than 1 node gives its GPUs to the nodes in "
      "equal parts. This is an approximation.")
    W("- The capacity is the capacity of today. A node that the cluster added "
      "last week counts over the whole window.")
    W("- A job that still waits has no start time, thus the delay table "
      "reports less than the truth.")
    W("- Build this page again with:")
    W("")
    W("```bash")
    cons = f" \\\n      --constraint '{a.constraint}'" if a.constraint else ""
    W(f"python3 scripts/gpu_census.py report --days {a.days:g}{cons} \\\n"
      f"      --out {a.out}")
    W("```")
    W("")

    text = "\n".join(lines)
    if a.out == "-":
        print(text)
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(text)
    print(f"[census] wrote {a.out} ({len(lines)} lines)")


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure the GPU supply of a SLURM partition.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    def common(sp):
        sp.add_argument("--partition", default="gpu", help="default: gpu")
        sp.add_argument("--constraint", default="",
                        help="SLURM feature expression, such as 'a6000|6000ada'")
        sp.add_argument("--min-vram", type=int, default=0, metavar="GB",
                        help="keep only nodes with a GPU of this memory or more")
        sp.add_argument("--json", action="store_true", help="write JSON")
        return sp

    sub = p.add_subparsers(dest="cmd", required=True)

    s = common(sub.add_parser("inventory", help="what GPUs the partition holds now"))
    s.add_argument("--by-node", action="store_true", help="add a row for each node")
    s.set_defaults(func=cmd_inventory)

    s = common(sub.add_parser("history", help="how often a GPU is free"))
    s.add_argument("--days", type=float, default=14.0)
    s.add_argument("--step", type=int, default=300, help="sample period in seconds")
    s.add_argument("--by-hour", action="store_true", help="add the daily profile")
    s.add_argument("--thresholds", type=int, nargs="+", default=[1, 8, 32, 64],
                   help="report the part of the window with N or more free")
    s.set_defaults(func=cmd_history)

    s = common(sub.add_parser("waits", help="queue delay and preemption rate"))
    s.add_argument("--days", type=float, default=14.0)
    s.add_argument("--user", default="", help="only the jobs of this user")
    s.set_defaults(func=cmd_waits)

    s = common(sub.add_parser("report", help="write the whole census as markdown"))
    s.add_argument("--days", type=float, default=30.0)
    s.add_argument("--step", type=int, default=300)
    s.add_argument("--thresholds", type=int, nargs="+", default=[1, 8, 32, 64])
    s.add_argument("--out", default="vlm-narratives-docs/gpu-partition-census.md",
                   help="`-` writes to the terminal")
    s.set_defaults(func=cmd_report)

    s = common(sub.add_parser("snapshot", help="append the live state to a CSV"))
    s.add_argument("--csv", default="multirun/gpu_snapshots.csv")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_snapshot)

    a = p.parse_args()
    if getattr(a, "constraint", "") and "[" in a.constraint:
        raise SystemExit("a constraint with a count, such as [a6000*2], is not supported")
    a.func(a)


if __name__ == "__main__":
    main()
