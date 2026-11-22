# Site configs

This folder holds the cluster settings of this site (the Cornell Tech
`pierson`, `ju` and `klara` partitions). The framework does not need it.

| Folder | Content |
|--------|---------|
| `conf/common/hydra/launcher/` | The launchers of this cluster |
| `conf/ocr/hydra/launcher/` | The OCR launcher with 4 GPUs (tensor parallel 2) |

The search-path plugin (`hydra_plugins/mllmsci_searchpath/`) adds `site/conf/`.
A launcher here takes the place of the generic framework launcher with the same
name (`slurm_gpu_2x` and so on). Thus jobs on this cluster keep the settings of
the cluster: the `sbatch` shim, `source ~/.bashrc`, the `LD_PRELOAD` of the
system libstdc++, `TORCH_CUDA_ARCH_LIST=8.6` and the partition names.

These launchers read `SLURM_PARTITION` (default `pierson`). The generic
launchers read `IMAGE_FORWARD_SLURM_PARTITION`. See `server.env.example`.

**Another cluster:** delete this folder, or write your own launchers here. The
generic launchers in `dagspaces/common/conf/hydra/launcher/` work with
`server.env` only.
