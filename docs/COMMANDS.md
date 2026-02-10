# Commands

## `gpu` / `gpu launch`

Runs the interactive allocation launcher.

Examples:

```bash
gpu
gpu launch --time 01:00:00 --gpus 1 --cpus 8 --mem 50G --account PXXXX
gpu launch --ui vscode --timeout-mode notify --notify-email you@osu.edu
```

## `gpu search`

Submits two-phase halving probes via `sbatch`.

Examples:

```bash
gpu search
gpu search --max-time 04:00:00 --min-time 00:30:00 --max-gpus 4 --min-gpus 1
gpu search --dry-run --yes
```

## `gpu dash`

Live dashboard for pending/running jobs.

Keyboard actions in curses mode:

- `j`/`k` or arrow keys: move focus
- `space`: select focused job
- `a`: toggle all
- `c`: cancel selected/focused jobs
- `v`: join focused running job via remote open
- `r`: refresh
- `q`: quit

Examples:

```bash
gpu dash
gpu dash --user your_user
gpu dash --editor cursor
```

## `gpu remote`

Open direct remote editor target.

Examples:

```bash
gpu remote pitzer
gpu remote c0318 --path ./project
gpu remote cardinal --editor antigravity
gpu remote ascend --dry-run
```
