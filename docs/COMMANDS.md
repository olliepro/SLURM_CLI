# Commands

## `gpu` / `gpu launch`

Runs the interactive allocation launcher.

The resource picker includes a `Partition` row with `Auto` plus the current
cluster's visible partitions.

Examples:

```bash
gpu
gpu launch --time 01:00:00 --gpus 1 --cpus 8 --mem 50G --account PXXXX
gpu launch --time 00:30:00 --gpus 1 --partition quad --account PXXXX
gpu launch --ui vscode --timeout-mode notify --notify-email you@osu.edu
```

## `gpu search`

Submits two-phase halving probes via `sbatch`.

Examples:

```bash
gpu search
gpu search --max-time 04:00:00 --min-time 00:30:00 --max-gpus 4 --min-gpus 1
gpu search --max-time 02:00:00 --max-gpus 1 --partition debug-nextgen
gpu search --dry-run --yes
```

## `gpu dash`

Live dashboard for pending/running jobs.

Keyboard actions in curses mode:

- `j`/`k` or arrow keys: move focus
- `space`: select focused job
- `a`: toggle all
- `c`: cancel selected/focused jobs
- `v`: join focused running job via remote open without leaving the dashboard
- `n`: choose account/resources, submit a held batch allocation, and return to the dashboard
- `r`: relocate remote editor to another OSC login cluster; Escape returns to the dashboard from this picker
- `q` or Escape: quit

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

## `gpu query`

Non-interactive, read-only JSON surface for agents/scripts. No curses, no
prompts, no side effects; every subcommand prints one JSON object to stdout.
Output carries `schema_version` for forward compatibility.

Subcommands:

- `plan`: answers "where should I schedule this and when can it start?" —
  returns the recommended partition (debug-aware), whether the request
  `qualifies_for_debug`, `advice` on how to reach the high-priority debug queue,
  and one `options` entry per candidate partition with a capacity-forecast
  `start_estimate`.
- `recommend`: partition routing + debug eligibility only (no Slurm fetch when
  `--cluster` is given).
- `forecast`: availability time-series and `earliest_free` for one partition.
- `avail`: current free GPUs (and `max_colocated_available`) per GPU partition.
- `jobs`: your pending/running jobs with start ETAs.

Start estimates are **capacity-based, not priority-aware** — see the `caveats`
field in the output. They reflect when GPUs free up, not Slurm's
priority/backfill ordering.

Examples:

```bash
gpu query plan --gpus 1 --time 00:30:00 --cpus 4 --mem 32G
gpu query recommend --gpus 4 --time 02:00:00 --cluster cardinal
gpu query forecast --partition gpu --gpus 2 --horizon-hours 8
gpu query avail
gpu query jobs --user your_user
```
