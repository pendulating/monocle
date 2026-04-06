"""Runtime patches for Python's multiprocessing resource tracker.

This module adds two layers of hardening:

1. Spawn the resource-tracker helper process with a custom entrypoint that
   tolerates double-unregister attempts (which normally surface as KeyError
   tracebacks) and swallows benign cleanup errors such as removing already
   deleted POSIX shared memory segments.
2. Wrap the cleanup callables registered with multiprocessing so that missing
   resources (FileNotFoundError, ProcessLookupError, etc.) never escalate to the
   user.

The patches are idempotent and safe to call from every process (driver and Ray
workers).  They only modify private multiprocessing internals, so the logic is
guarded behind explicit opt-in in :func:`apply_patch`.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import warnings
from functools import wraps
from typing import Callable

_PATCH_APPLIED = False
_PATCH_MODULE = "dagspaces.common.resource_tracker_patch"


def _wrap_cleanup_funcs(rt_module) -> None:
    """Wrap cleanup functions so missing resources don't raise."""

    for rtype, func in list(rt_module._CLEANUP_FUNCS.items()):  # noqa: SLF001
        if getattr(func, "__mllmsci_wrapped__", False):
            continue

        @wraps(func)
        def _wrapped(name: str, _orig: Callable[[str], None] = func) -> None:
            with contextlib.suppress(FileNotFoundError, ProcessLookupError, OSError):
                _orig(name)

        _wrapped.__mllmsci_wrapped__ = True  # type: ignore[attr-defined]
        rt_module._CLEANUP_FUNCS[rtype] = _wrapped  # noqa: SLF001


def _patched_main(fd: int) -> None:
    """Replica of multiprocessing.resource_tracker.main with safer handling."""

    import multiprocessing.resource_tracker as rt

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if rt._HAVE_SIGMASK:  # noqa: SLF001
        signal.pthread_sigmask(signal.SIG_UNBLOCK, rt._IGNORED_SIGNALS)  # noqa: SLF001

    for stream in (sys.stdin, sys.stdout):
        with contextlib.suppress(Exception):
            stream.close()

    cache = {rtype: set() for rtype in rt._CLEANUP_FUNCS.keys()}  # noqa: SLF001
    try:
        with open(fd, "rb") as pipe:
            for raw_line in pipe:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    cmd, name, rtype = line.decode("ascii").split(":")
                    cleanup_func = rt._CLEANUP_FUNCS.get(rtype)  # noqa: SLF001
                    if cleanup_func is None:
                        raise ValueError(
                            f"Cannot register {name!r} for automatic cleanup: "
                            f"unknown resource type {rtype!r}"
                        )

                    if cmd == "REGISTER":
                        cache.setdefault(rtype, set()).add(name)
                    elif cmd == "UNREGISTER":
                        cache.setdefault(rtype, set()).discard(name)
                    elif cmd == "PROBE":
                        continue
                    else:
                        raise RuntimeError(f"unrecognized command {cmd!r}")
                except Exception:
                    exc_type, exc, tb = sys.exc_info()
                    if exc_type is KeyError:
                        continue
                    with contextlib.suppress(Exception):
                        sys.excepthook(exc_type, exc, tb)
    finally:
        for rtype, entries in cache.items():
            if entries:
                with contextlib.suppress(Exception):
                    warnings.warn(
                        "resource_tracker: There appear to be %d leaked %s objects "
                        "to clean up at shutdown"
                        % (len(entries), rtype)
                    )
            cleanup_func = rt._CLEANUP_FUNCS.get(rtype)  # noqa: SLF001
            if cleanup_func is None:
                continue
            for name in entries:
                try:
                    cleanup_func(name)
                except Exception as err:
                    with contextlib.suppress(Exception):
                        warnings.warn(f"resource_tracker: {name!r}: {err}")


def run_patched_resource_tracker(fd: int) -> None:
    """Entry point executed inside the helper process."""
    import multiprocessing.resource_tracker as rt

    _wrap_cleanup_funcs(rt)
    _patched_main(fd)


def apply_patch() -> None:
    """Install resource-tracker patches (idempotent)."""
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    import multiprocessing.resource_tracker as rt

    _wrap_cleanup_funcs(rt)

    original_ensure_running = rt.ResourceTracker.ensure_running

    def ensure_running_with_patch(self):  # type: ignore[override]
        with self._lock:
            if self._lock._recursion_count() > 1:
                return self._reentrant_call_error()
            if self._fd is not None:
                if self._check_alive():
                    return
                os.close(self._fd)
                try:
                    if self._pid is not None:
                        os.waitpid(self._pid, 0)
                except ChildProcessError:
                    pass
                self._fd = None
                self._pid = None
                warnings.warn(
                    "resource_tracker: process died unexpectedly, relaunching.  "
                    "Some resources might leak."
                )

            fds_to_pass = []
            with contextlib.suppress(Exception):
                fds_to_pass.append(sys.stderr.fileno())

            cmd = (
                f"from {_PATCH_MODULE} import run_patched_resource_tracker as _rp;"
                " _rp(%d)"
            )
            r, w = os.pipe()
            try:
                fds_to_pass.append(r)
                exe = rt.spawn.get_executable()  # noqa: SLF001
                args = [exe] + rt.util._args_from_interpreter_flags()  # noqa: SLF001
                args += ["-c", cmd % r]
                try:
                    if rt._HAVE_SIGMASK:  # noqa: SLF001
                        signal.pthread_sigmask(
                            signal.SIG_BLOCK, rt._IGNORED_SIGNALS  # noqa: SLF001
                        )
                    pid = rt.util.spawnv_passfds(exe, args, fds_to_pass)  # noqa: SLF001
                finally:
                    if rt._HAVE_SIGMASK:  # noqa: SLF001
                        signal.pthread_sigmask(
                            signal.SIG_UNBLOCK, rt._IGNORED_SIGNALS  # noqa: SLF001
                        )
            except Exception:
                os.close(w)
                raise
            else:
                self._fd = w
                self._pid = pid
            finally:
                os.close(r)

    rt.ResourceTracker.ensure_running = ensure_running_with_patch
    rt._resource_tracker.ensure_running = ensure_running_with_patch.__get__(  # noqa: SLF001
        rt._resource_tracker, rt.ResourceTracker
    )
    rt.ensure_running = rt._resource_tracker.ensure_running  # noqa: SLF001

    _PATCH_APPLIED = True


__all__ = ["apply_patch", "run_patched_resource_tracker"]
