# History

## 2026-06-01

- Made colocated GPU availability CPU/memory-aware so raw free GPUs stranded on full nodes no longer show as schedulable. Added Cardinal-style CPU-full node regressions.
- Fixed GPU availability forecasting so down or non-responsive Slurm nodes are excluded from total and colocated availability. Added regression tests for `DOWN+NOT_RESPONDING` quad nodes.
