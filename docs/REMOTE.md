# Remote Behavior

`gpu remote` and dash join actions use shared logic in `src/slurm_cli/remote_access.py`.

## Alias Expansion

These aliases are supported:

- `pitzer` -> `osc-pitzer-login`
- `cardinal` -> `osc-cardinal-login`
- `ascend` -> `osc-ascend-login`

Any other host string is used as-is.

## Required SSH Config

Configure `~/.ssh/config` with login aliases and compute-node proxy rules so editor remote URIs can resolve correctly:

**Set `User` and `IdentityFile` to your own values.**

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

## Launch Path

Remote open is direct-only and always executes:

```bash
<editor> --new-window --folder-uri vscode-remote://ssh-remote+<host>/<path>
```

Editor command resolution order:

1. explicit `--editor`
2. `GPU_REMOTE_EDITOR` or `SLURM_CLI_EDITOR`
3. first available command on PATH: `code`, `cursor`, `windsurf`, `codium`, `antigravity`

## Dash Integration

Inside `gpu dash`, pressing `v` on a running job calls the same remote API.
