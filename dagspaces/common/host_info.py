"""Host resource detection for CPU, GPU, and memory.

This module detects host resources without torch imports. GPU queries use
nvidia-smi only, so a pre-fork parent process never initialises CUDA.

Detection sources:
- CUDA_VISIBLE_DEVICES and SLURM environment variables
- nvidia-smi inventory query
- cgroup limits, SLURM memory variables, and system memory

The wandb_logger module previously held these functions. This module is now
the single source of truth for them.
"""

import os
import subprocess
from functools import lru_cache
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# GPU / compute detection (nvidia-smi only — never imports torch)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _query_nvidia_smi_inventory() -> List[Dict[str, Any]]:
    """Query GPU inventory via nvidia-smi without touching torch.cuda.

    Best practice for pre-fork multiprocessing is to avoid initialising CUDA
    in the parent process.  Scheduler env vars and nvidia-smi provide the
    same inventory data without poisoning a later fork-based worker launch.

    Tries two nvidia-smi field sets: one with compute_cap (not available on
    all driver versions) and one without, falling back gracefully.
    """
    field_sets = [
        ["index", "uuid", "name", "memory.total", "compute_cap"],
        ["index", "uuid", "name", "memory.total"],
    ]
    try:
        proc = None
        fields: List[str] = []
        for candidate_fields in field_sets:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={','.join(candidate_fields)}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0:
                fields = candidate_fields
                break
        if proc is None or proc.returncode != 0:
            return []

        gpus: List[Dict[str, Any]] = []
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(fields):
                continue
            gpu_info: Dict[str, Any] = {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
            }
            try:
                gpu_info["total_memory_gb"] = round(float(parts[3]) / 1024.0, 2)
            except Exception:
                pass
            if len(parts) > 4 and parts[4]:
                gpu_info["compute_capability"] = parts[4]
                try:
                    major_s, minor_s = parts[4].split(".", 1)
                    gpu_info["major"] = int(major_s)
                    gpu_info["minor"] = int(minor_s)
                except Exception:
                    pass
            gpus.append(gpu_info)
        return gpus
    except Exception:
        return []


def _collect_gpu_details() -> List[Dict[str, Any]]:
    """Return a copy of the cached GPU inventory."""
    return _query_nvidia_smi_inventory().copy()


def _detect_num_gpus() -> int:
    """Detect the number of GPUs allocated to this job.

    Priority order:
    1. CUDA_VISIBLE_DEVICES environment variable (set by launcher)
    2. SLURM_GPUS_PER_NODE or SLURM_GPUS_ON_NODE
    3. nvidia-smi inventory (does not initialize CUDA in-process)
    4. Return 0 if no GPUs detected
    """
    # Priority 1: CUDA_VISIBLE_DEVICES (most reliable for actual allocation)
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            # Parse comma-separated GPU indices (e.g., "0,1,2,3" -> 4 GPUs)
            gpu_indices = [x.strip() for x in cuda_visible.split(",") if x.strip()]
            if gpu_indices:
                return len(gpu_indices)
    except Exception:
        pass

    # Priority 2: SLURM environment variables
    try:
        slurm_gpus = os.environ.get("SLURM_GPUS_PER_NODE") or os.environ.get(
            "SLURM_GPUS_ON_NODE"
        )
        if slurm_gpus:
            # Can be a number like "4" or format like "gpu:4"
            try:
                if ":" in slurm_gpus:
                    return int(slurm_gpus.split(":")[-1])
                return int(slurm_gpus)
            except Exception:
                pass
    except Exception:
        pass

    # Priority 3: GPU inventory from nvidia-smi
    gpu_details = _collect_gpu_details()
    if gpu_details:
        return len(gpu_details)

    return 0


def _detect_gpu_type() -> Optional[str]:
    """Detect the GPU type/model name via nvidia-smi (no torch import).

    Returns a normalised GPU type string (e.g. 'NVIDIA RTX A6000').
    """
    gpu_details = _collect_gpu_details()
    if gpu_details:
        gpu_name = gpu_details[0].get("name")
        if gpu_name:
            return str(gpu_name)
    return None


# ---------------------------------------------------------------------------
# CPU / memory detection
# ---------------------------------------------------------------------------

def _parse_cpus_on_node(val: str) -> int:
    """Parse SLURM_CPUS_ON_NODE format (e.g., '32', '16(x2)', '2,2')."""
    if not isinstance(val, str):
        return -1
    try:
        v = val.strip()
        if "(x" in v and v.endswith(")"):
            import re as _re

            m = _re.match(r"^(\d+)\(x(\d+)\)$", v)
            if m:
                a = int(m.group(1))
                b = int(m.group(2))
                return max(1, a * b)
        if "," in v:
            parts = [p for p in v.split(",") if p.strip()]
            acc = 0
            for p in parts:
                try:
                    acc += int(p)
                except Exception:
                    return -1
            return max(1, acc)
        return max(1, int(v))
    except Exception:
        return -1


def _detect_num_cpus() -> Optional[int]:
    """Detect the number of CPUs allocated to this job.

    Priority order:
    1. SLURM_CPUS_PER_TASK or SLURM_CPUS_ON_NODE (job allocation)
    2. os.cpu_count() (system total)
    """
    try:
        cpt = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpt is not None and str(cpt).strip() != "":
            return int(cpt)

        con = os.environ.get("SLURM_CPUS_ON_NODE")
        if con is not None and str(con).strip() != "":
            cpus = _parse_cpus_on_node(con)
            if cpus > 0:
                return cpus
    except Exception:
        pass

    try:
        return os.cpu_count()
    except Exception:
        pass

    return None


def _read_int_file(path: str) -> int:
    """Read an integer from a file, handling 'max' keyword."""
    try:
        with open(path, "r") as f:
            s = f.read().strip()
        if s.lower() == "max":
            return -1
        return int(s)
    except Exception:
        return -1


def _detect_cgroup_mem_limit_bytes() -> int:
    """Return cgroup memory limit in bytes when available; otherwise -1.

    Supports cgroup v2 (memory.max) and v1 (memory.limit_in_bytes).
    """
    v2 = "/sys/fs/cgroup/memory.max"
    lim = _read_int_file(v2)
    if lim > 0:
        try:
            if lim > (1 << 56):
                return -1
        except Exception:
            pass
        return lim

    v1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    lim = _read_int_file(v1)
    if lim > 0:
        try:
            if lim > (1 << 56):
                return -1
        except Exception:
            pass
        return lim

    return -1


def _detect_slurm_job_mem_bytes() -> int:
    """Infer SLURM job memory allocation in bytes from env vars.

    Prefers SLURM_MEM_PER_NODE; otherwise uses SLURM_MEM_PER_CPU *
    SLURM_CPUS_ON_NODE.  Values are MB according to SLURM docs.
    """
    try:
        mem_per_node_mb = os.environ.get("SLURM_MEM_PER_NODE")
        if mem_per_node_mb:
            mb = int(mem_per_node_mb)
            if mb > 0:
                return mb * 1024 * 1024
    except Exception:
        pass

    try:
        mem_per_cpu_mb = os.environ.get("SLURM_MEM_PER_CPU")
        cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE")
        if mem_per_cpu_mb and cpus_on_node:
            mb = int(mem_per_cpu_mb)
            cpus = _parse_cpus_on_node(cpus_on_node)
            if mb > 0 and cpus > 0:
                return mb * cpus * 1024 * 1024
    except Exception:
        pass

    return -1


def _detect_memory_gb() -> Optional[float]:
    """Detect available memory in GB.

    Priority order:
    1. cgroup memory limit (container/SLURM cgroup)
    2. SLURM env-based memory inference
    3. System total memory (psutil/os.sysconf)
    """
    cg = _detect_cgroup_mem_limit_bytes()
    if cg > 0:
        return cg / (1024 ** 3)

    sj = _detect_slurm_job_mem_bytes()
    if sj > 0:
        return sj / (1024 ** 3)

    try:
        import psutil  # type: ignore

        tot = int(getattr(psutil.virtual_memory(), "total", 0))
        if tot > 0:
            return tot / (1024 ** 3)
    except Exception:
        pass

    try:
        tot = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        if tot > 0:
            return tot / (1024 ** 3)
    except Exception:
        pass

    return None
