# Architecture

## High-Level Modules

- `cli.py`: `gpu` command router (`launch/search/dash/remote`).
- `interactive_slurm.py`: launch/search/dash resource resolution and execution flows.
- `launch_flow.py`: shared launch workflow used by both `gpu launch` and dashboard-triggered launches.
- `dash_ui.py`: curses dashboard rendering and key handling.
- `dashboard_forecast.py`: forecast panel rendering helpers and chart annotations for the dashboard.
- `dash_logic.py`: Slurm job fetch/cancel/join action helpers.
- `remote_access.py`: shared direct remote-open API with editor CLI selection.
- `forecast_core.py`: Slurm parsing and forecast time-series computation.
- `forecast_cli.py`: terminal forecast rendering (used by dash forecast panel).
- `slurm_backend.py`: `srun`/`sbatch`/host helpers.
- `search_logic.py` / `search_ui.py`: two-phase probe generation and UI.

## Command Flow

### `gpu launch`

1. parse args
2. resolve account/resources/timeout/UI mode
3. run terminal `srun` or background VS Code flow

### `gpu search`

1. resolve account/resources/bounds/notify email
2. build two-phase probes
3. submit probes sequentially via `sbatch`

### `gpu dash`

1. poll jobs via `squeue`
2. render rows + forecast panel
3. actions:
   - cancel via `scancel`
   - join via `remote_access.open_remote_target`
   - relocate to another OSC login cluster via `remote_access.open_remote_target`
   - relaunch a dashboard-specific batch-submit workflow via the shared launch-flow module

### `gpu remote`

1. normalize host aliases
2. resolve editor command (`--editor` -> env var -> auto detect)
3. run direct remote URI open command
4. return structured result text for CLI/dash display
