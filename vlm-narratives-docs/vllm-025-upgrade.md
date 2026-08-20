# The vLLM 0.25 upgrade

Date: 2026-08-12.

`.venv-vllm025cu129` is now the default environment for all launchers. It
replaces a two-environment split that was difficult to keep correct.

| Environment | vLLM | torch | Status |
|-------------|------|-------|--------|
| `.venv-vllm025cu129` | 0.25.0+cu129 | 2.11.0+cu129 | **default** |
| `.venv-nightly` | 0.23.1 | 2.11.0+cu129 | kept; was the gemma-4-12b path |
| `.venv-3.12` | 0.19.0 | 2.10.0+cu128 | kept; was the qwen3.5 path |
| `.venv` | — | — | empty since 2026-08-11; do not use |

Both older environments stay on disk. To go back for one run:

```bash
export MLLMSCI_VENV_ACTIVATE=/share/pierson/matt/mllmsci/.venv-3.12/bin/activate
```

## Build the environment

```bash
bash scripts/build_venv_vllm025.sh
```

The version pins for the CUDA stack are in `requirements-vllm025cu129.txt`.
The other packages come from `pyproject.toml`. The script installs torch
first, from the cu129 index, so that no later step can pull the default-index
cu128 build in as a dependency.

`flash-attn` has no published wheel for torch 2.11+cu129, and a source build
costs about an hour. The script copies the build that UAIR already made,
because both environments use CPython 3.12 and the same torch ABI.

**Warning:** after you change this environment, sync the `/scratch` mirrors
again. See [scratch-mirrors.md](scratch-mirrors.md).

## API changes in the pipeline code

We tested the vLLM import surface against 0.25 before the upgrade. Two names
disappeared.

| Name | Status in 0.25 | What we did |
|------|----------------|-------------|
| `vllm.sampling_params.GuidedDecodingParams` | removed | Use `StructuredOutputsParams` |
| `vllm.config.EngineArgs` | removed | The `vllm.engine.arg_utils` fallback covers it |

`_build_sampling_params` in `dagspaces/common/vllm_inference.py` already tried
`StructuredOutputsParams` first, so it needed no change.

`serialize_arrow_unfriendly_in_row` in `dagspaces/common/stage_utils.py` did
need a correction. It imported `GuidedDecodingParams` and `SamplingParams` in
one statement. When only the first name is absent, the whole statement fails
and the function loses the `SamplingParams` branch too. The function now
imports each type on its own.

**Note:** this was not only a vLLM 0.25 problem. vLLM 0.19 in `.venv-3.12` also
lacks `GuidedDecodingParams`, so the function never serialized a
`SamplingParams` object on the old default environment either.

## FFmpeg: a new runtime dependency

vLLM 0.25 imports `vllm.multimodal.video`, which imports torchcodec, which
opens `libtorchcodec_coreN.so` with dlopen. That library needs the FFmpeg
`libav*.so.N`. vLLM 0.19 never loaded it, so this dependency is new.

Without a fix, `import vllm` stops with `Could not load libtorchcodec`. The
node has no FFmpeg 4 to 7, and the `libavfilter` of the system FFmpeg 8 needs
`GLIBCXX_3.4.32`, which the anaconda-base libstdc++ does not give.

The fix is a self-contained FFmpeg 7.1 LGPL shared build outside the venv, at
`/share/pierson/matt/zoo/ffmpeg-libs/n7.1/lib`. Its `libav*.so` have no
libstdc++ dependency, so the GLIBCXX wall is gone.

**Warning:** `ld.so` reads `LD_LIBRARY_PATH` one time, when the process starts.
An in-process load of `server.env` thus cannot help dlopen.
`scripts/activate_stage_venv.sh` does the prepend in the SLURM shell, before
python starts. Each launcher sources that script, **including the monitor**,
which sets `MLLMSCI_SKIP_SCRATCH_VENV=1` to keep the FFmpeg path but skip the
venv swap.

If you start a driver by hand, prepend the directory yourself:

```bash
export LD_LIBRARY_PATH=/share/pierson/matt/zoo/ffmpeg-libs/n7.1/lib:$LD_LIBRARY_PATH
```

## Engine kwargs

We tested each of the 26 `engine_kwargs` keys in the configs against the 0.25
`EngineArgs` signature. Only `guided_decoding_backend` is not accepted.
`filter_vllm_engine_kwargs` removes it and prints a line, so the effect is
harmless. `structured_outputs_config` is the replacement, and the configs
already use it.
