"""Status of the version 2 extraction arms: shards done, running, failed, and the rerun command.

Use:
    python research/scripts/check_trace_arms.py

A FAILED shard is rerun with `submit_trace_shards --arm <arm> --only case:shard --base-dir <base>`.
See `vlm-narratives-docs/langextract-trace-extraction.md`, "Submit without a monitor".
"""
import glob, json, os, subprocess, sys
from collections import Counter, defaultdict

ROOT = "/share/ju/matt/mllmsci"
ARMS = {
    "CUESV2GQ27": "gemma_by_qwen27_v2", "CUESV2QG31": "qwen_by_gemma31_v2",
    "CUESV2Q": "qwen_by_gemma_v2", "CUESV2QQ": "qwen_self_v2",
    "CUESV2": "gemma_by_qwen_v2", "CUESV2GG": "gemma_self_v2",
    "CUESV2GG31": "gemma_by_gemma31_v2", "CUESV2QQ27": "qwen_by_qwen27_v2",
    "CUESV3GQ27": "gemma_by_qwen27_v3", "CUESV3QG31": "qwen_by_gemma31_v3",
    "CUESV3GG31": "gemma_by_gemma31_v3", "CUESV3QQ27": "qwen_by_qwen27_v3",
}

def sacct_states(jobids):
    if not jobids:
        return {}
    out = subprocess.run(["sacct", "-j", ",".join(jobids), "-X", "-n", "-P", "-o", "JobID,State,Elapsed,NodeList"],
                         capture_output=True, text=True).stdout
    states = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            states[parts[0]] = (parts[1].split()[0], parts[2] if len(parts) > 2 else "", parts[3] if len(parts) > 3 else "")
    return states

for exp, arm in ARMS.items():
    base = os.path.join(ROOT, "multirun", f"2026-09-20_{exp}")
    sub = os.path.join(base, "submitted_jobs.txt")
    if not os.path.exists(sub):
        print(f"== {exp} ({arm}): not submitted"); continue
    rows = [l.rstrip("\n").split("\t") for l in open(sub) if l.strip()]
    # the LAST submission of each (case, shard) is the live one
    live = {}
    for jid, case, shard in rows:
        live[(case, int(shard))] = jid
    states = sacct_states(sorted(set(live.values())))
    done = Counter(); total = Counter(); chunks = Counter(); st = Counter(); timeouts = []
    for (case, shard), jid in live.items():
        total[case] += 1
        d = os.path.join(base, "runs", f"{case}__{shard:04d}", "outputs", "extractions")
        if glob.glob(os.path.join(d, "*_s*.parquet")):
            done[case] += 1
        chunks[case] += len(glob.glob(os.path.join(d, "chunks", "*.parquet")))
        s = states.get(jid, ("UNKNOWN", "", ""))[0]
        st[s] += 1
        if s in ("TIMEOUT", "FAILED", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"):
            timeouts.append((case, shard, jid, s))
    print(f"== {exp} ({arm}): shards {sum(total.values())} done {sum(done.values())} chunks {sum(chunks.values())} states {dict(st)}")
    for case in sorted(total):
        print(f"   {case:<20} done {done[case]}/{total[case]} chunks {chunks[case]}")
    if timeouts:
        print(f"   NEEDS RERUN ({len(timeouts)}): " + ", ".join(f"{c}#{s}:{jid}:{state}" for c, s, jid, state in timeouts[:8]))
        print(f"   -> python -m dagspaces.pairwisevqa.submit_trace_shards --arm {arm} --only-missing --base-dir {base} ...")
