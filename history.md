# History

## 2026-07-07

- Fixed GPU dashboard forecast failures on abbreviated Slurm array task lists like `0,19,38-57,...%20` by running Slurm state fetches with `SLURM_BITSTR_LEN=0`, so `ArrayTaskId` values are expanded before parsing.

## 2026-06-29

- Pulled latest `SLURM_CLI` from `origin/main` through commit `68203d6`, adding the `gpu query` JSON surface. Fixed `gpu query plan` and `gpu query forecast` so fragmented aggregate GPU availability no longer reports immediate start for single-node multi-GPU requests when `max_colocated_available` is below the request size.

## 2026-06-01

- Made colocated GPU availability CPU/memory-aware so raw free GPUs stranded on full nodes no longer show as schedulable. Added Cardinal-style CPU-full node regressions.
- Fixed GPU availability forecasting so down or non-responsive Slurm nodes are excluded from total and colocated availability. Added regression tests for `DOWN+NOT_RESPONDING` quad nodes.
