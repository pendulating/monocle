"""Job-only fixes for DeepSeek-V4 on vLLM 0.25.0+cu129 on the B300 (sm_103).

`scripts/f4/extract.sbatch` puts this folder on PYTHONPATH, thus every Python
process of the job (the driver, the vLLM engine core, the worker) runs this
file at start. Each fix has its own switch. Nothing in the venv changes.

Fix 1, F4X_PRELOAD_CUDART=1: bind FlashInfer to the real CUDA runtime.
    `flashinfer.comm.cuda_ipc` runs `CudaRTLibrary()` at import. It takes the
    FIRST line of /proc/self/maps that holds "libcudart". The DeepSeek-V4
    kernels load TileLang, and TileLang maps its own `libcudart_stub.so` at a
    lower address than the real runtime. vLLM imports `flashinfer.comm` late,
    in `kernel_warmup`, thus FlashInfer then binds the stub, and the stub has
    no `cudaDeviceReset` (smoke job 92495). `import torch` maps the real
    `libcudart.so.12`, and the import of `flashinfer.comm.cuda_ipc` right after
    it binds that file and caches the module. Nothing here starts CUDA.

Fix 2, F4X_TOPK_FALLBACK=1: decode top-k through `persistent_topk`.
    The sparse attention indexer uses `torch.ops._C.cooperative_topk` on a
    decode step of 32 rows or fewer. On the B300 that launch fails with "no
    kernel image is available for execution on the device" (smoke job 92682):
    the cu129 wheel holds cooperative_topk only as sm_100a code, and sm_100a
    code does not run on sm_103. `persistent_topk` has plain sm_100 code, and
    plain sm_100 code runs on sm_103. The fix rewrites the gate
    `use_cooperative_topk` to False when the module loads, thus the indexer
    takes its own `persistent_topk` branch. The fix asserts that the gate
    appears exactly 1 time; else it changes nothing and prints why.
"""
import importlib.abc
import importlib.machinery
import os
import sys

_TOPK_MODULE = "vllm.model_executor.layers.sparse_attn_indexer"
_TOPK_EDITS = (
    ("        use_cooperative_topk = (\n", "        use_cooperative_topk = False and (\n"),
)


def _say(msg):
    print(f"[f4x-hook] pid={os.getpid()} {msg}", flush=True)


class _TopkLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        src = self.get_data(self.path).decode("utf-8")
        for old, new in _TOPK_EDITS:
            if src.count(old) != 1:
                _say(f"top-k fallback NOT applied: {old.strip()!r} occurs {src.count(old)} times")
                return super().get_code(fullname)
        for old, new in _TOPK_EDITS:
            src = src.replace(old, new)
        if os.environ.get("F4X_HOOK_VERBOSE") == "1":
            _say("top-k fallback applied: decode uses persistent_topk")
        return compile(src, self.path, "exec", dont_inherit=True)


class _TopkFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TOPK_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            return spec
        spec.loader = _TopkLoader(fullname, spec.origin)
        return spec


if os.environ.get("F4X_TOPK_FALLBACK") == "1":
    sys.meta_path.insert(0, _TopkFinder())

if os.environ.get("F4X_PRELOAD_CUDART") == "1":
    try:
        import torch  # noqa: F401  (maps the real libcudart)
        import flashinfer.comm.cuda_ipc as _ipc

        if os.environ.get("F4X_HOOK_VERBOSE") == "1":
            _lib = getattr(_ipc.cudart, "lib", None)
            _say(f"flashinfer cudart = {getattr(_lib, '_name', _lib)}")
    except Exception as exc:  # never stop the interpreter
        _say(f"cudart preload skipped: {type(exc).__name__}: {exc}")
