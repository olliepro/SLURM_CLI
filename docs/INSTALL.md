# Install

## Requirements

- Python 3.9+
- Slurm client commands available on PATH (`squeue`, `scontrol`, `srun`, `sbatch`, `scancel`)
- Optional for remote editor open: at least one editor CLI command on PATH (`code`, `cursor`, `windsurf`, `codium`, `antigravity`)

## Install

```bash
python -m pip install .
```

Optional (if your local pip supports editable installs):

```bash
python -m pip install -e .
```

## Verify

```bash
gpu --help
gpu launch --help
gpu search --help
gpu dash --help
gpu remote --help
```

## Required SSH Config for Remote Open

`gpu remote` and dash join (`v` in `gpu dash`) require SSH host aliases that match OSC login hosts and compute node patterns.

Add entries like this in `~/.ssh/config`:

```sshconfig
Host osc-pitzer-login
    HostName pitzer.osc.edu
    User <your_username>
    IdentityFile <path_to_ssh_key>

Host osc-ascend-login
    HostName ascend.osc.edu
    User <your_username>
    IdentityFile <path_to_ssh_key>

Host osc-cardinal-login
    HostName cardinal.osc.edu
    User <your_username>
    IdentityFile <path_to_ssh_key>

Host c????
    ProxyJump osc-cardinal-login
    HostName %h.ten.osc.edu
    User <your_username>
    IdentityFile <path_to_ssh_key>

Host p????
    ProxyJump osc-pitzer-login
    HostName %h.ten.osc.edu
    User <your_username>
    IdentityFile <path_to_ssh_key>

Host a????
    ProxyJump osc-ascend-login
    HostName %h.ten.osc.edu
    User <your_username>
    IdentityFile <path_to_ssh_key>
```

Use your own username and key path values.

## Remote Open Behavior

`gpu remote` always uses the direct URI launcher:

```bash
<editor> --new-window --folder-uri vscode-remote://ssh-remote+<host>/<path>
```

Editor command resolution order:

1. `--editor ...`
2. `GPU_REMOTE_EDITOR` or `SLURM_CLI_EDITOR`
3. first available from: `code`, `cursor`, `windsurf`, `codium`, `antigravity`
