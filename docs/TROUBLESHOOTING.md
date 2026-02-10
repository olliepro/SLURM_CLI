# Troubleshooting

## `ModuleNotFoundError: No module named 'slurm_cli'`

Install the package first:

```bash
python -m pip install -e .
```

## No supported remote editor command found

`gpu remote` checks this order: `--editor`, then `GPU_REMOTE_EDITOR`/`SLURM_CLI_EDITOR`, then auto-detect from `code`, `cursor`, `windsurf`, `codium`, `antigravity`.

Set one explicitly:

```bash
gpu remote pitzer --editor cursor
```

Or set it in your shell profile:

```bash
export GPU_REMOTE_EDITOR=cursor
```

## `gpu` runs git push instead of this CLI

Oh My Zsh `git` plugin can define `gpu='git push upstream'`, which shadows the installed `gpu` command.

Check what `gpu` resolves to:

```bash
type -a gpu
```

If you see an alias first, add this immediately after `source "$ZSH/oh-my-zsh.sh"` in `~/.zshrc`:

```zsh
unalias gpu 2>/dev/null || true
```

Then reload your shell:

```bash
source ~/.zshrc
```

## `squeue` / `scontrol` / `srun` / `sbatch` not found

You are likely outside a Slurm-enabled environment, or Slurm tools are not on PATH.

## `gpu remote` fails with DBus / remote-peer errors

`gpu remote` is direct-only and requires a working editor CLI session on the machine where you run it.

- Verify your editor command resolves correctly:

```bash
type -a "${GPU_REMOTE_EDITOR:-code}"
```

- Confirm it can open any folder from this shell:

```bash
"${GPU_REMOTE_EDITOR:-code}" --new-window .
```

- On headless nodes, run `gpu remote` from a login/desktop session where remote editor launching is supported.

## Dash join fails to connect to compute node host

Your SSH config may be missing compute-node wildcard rules and `ProxyJump` entries.

- Confirm `~/.ssh/config` has `Host c????`, `Host p????`, and `Host a????` sections with the right `ProxyJump` values.
- Confirm OSC login aliases exist: `osc-pitzer-login`, `osc-cardinal-login`, `osc-ascend-login`.
- Keep `User` and `IdentityFile` set to your own values for all related host entries.

## No TTY / curses issues

When no interactive terminal is available, commands fall back to non-curses behavior where supported. For scripted flows, prefer explicit flags and `--yes` where available.

## Search emails and notify behavior

`gpu search` requires a notify email for begin notifications. You can pass it explicitly:

```bash
gpu search --notify-email you@osu.edu
```
