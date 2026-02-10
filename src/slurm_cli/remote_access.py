from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote

EDITOR_ENV_VARS = ("GPU_REMOTE_EDITOR", "SLURM_CLI_EDITOR")
EDITOR_CANDIDATES = ("code", "cursor", "windsurf", "codium", "antigravity")
EDITOR_ALIASES = {
    "vscode": "code",
    "code": "code",
    "cursor": "cursor",
    "windsurf": "windsurf",
    "codium": "codium",
    "vscodium": "codium",
    "antigravity": "antigravity",
    "gravity": "antigravity",
}


@dataclass(frozen=True)
class EditorCommand:
    """Resolved editor command used for direct remote URI opens.

    Args:
        command: Executable token passed as argv[0] to `subprocess.run`.
        source: Resolution source (`flag`, `env`, `auto`).

    Returns:
        Immutable editor command descriptor for diagnostics.

    Example:
        >>> resolve_editor_command(preferred="cursor")
        EditorCommand(command='cursor', source='flag')
    """

    command: str
    source: str


@dataclass(frozen=True)
class RemoteOpenRequest:
    """Input request for opening a direct remote editor target.

    Args:
        host: Hostname or OSC alias (`pitzer`, `cardinal`, `ascend`).
        work_dir: Folder path to open on the remote host.
        editor: Preferred editor command or alias (`code`, `cursor`, `antigravity`).
        dry_run: When true, return command details without executing.

    Returns:
        Immutable request object with normalized helpers.

    Example:
        >>> request = RemoteOpenRequest(host="pitzer", work_dir=Path("."))
        >>> request.resolved_host()
        'osc-pitzer-login'
    """

    host: str
    work_dir: Optional[Path] = None
    editor: Optional[str] = None
    dry_run: bool = False

    def resolved_host(self) -> str:
        """Return canonical hostname after OSC alias expansion."""

        return normalize_osc_host(host=self.host)

    def resolved_work_dir(self) -> Path:
        """Return existing working directory, falling back to current directory."""

        if self.work_dir is None:
            return Path.cwd()
        return self.work_dir if self.work_dir.exists() else Path.cwd()

    def direct_command(self, editor_command: str) -> list[str]:
        """Return direct remote URI command arguments for one editor CLI."""

        uri = f"vscode-remote://ssh-remote+{self.resolved_host()}{quote(str(self.resolved_work_dir()))}"
        return [editor_command, "--new-window", "--folder-uri", uri]


@dataclass(frozen=True)
class RemoteOpenResult:
    """Outcome of a remote open attempt.

    Args:
        ok: Whether launch succeeded.
        message: Human-readable status message.
        command: Command argv associated with the outcome.

    Returns:
        Immutable result object for CLI/UI status rendering.
    """

    ok: bool
    message: str
    command: list[str]

    def command_text(self) -> str:
        """Return canonical shell-quoted command text for logs/UI."""

        if not self.command:
            return ""
        return " ".join(shlex.quote(part) for part in self.command)


def normalize_osc_host(host: str) -> str:
    """Expand OSC shorthand host aliases into login hostnames.

    Args:
        host: Raw host token provided by a caller.

    Returns:
        Canonical hostname with OSC aliases expanded.

    Example:
        >>> normalize_osc_host(host="cardinal")
        'osc-cardinal-login'
    """

    normalized = host.strip()
    assert normalized, "host is required"
    alias_map = {
        "pitzer": "osc-pitzer-login",
        "cardinal": "osc-cardinal-login",
        "ascend": "osc-ascend-login",
    }
    return alias_map.get(normalized, normalized)


def normalize_editor_token(editor: Optional[str]) -> Optional[str]:
    """Normalize editor aliases into concrete CLI commands.

    Args:
        editor: Raw editor token from CLI or environment.

    Returns:
        Normalized command token, or `None` when unset/blank.
    """

    if editor is None:
        return None
    token = editor.strip()
    if not token:
        return None
    return EDITOR_ALIASES.get(token.lower(), token)


def resolve_editor_command(preferred: Optional[str]) -> Optional[EditorCommand]:
    """Resolve the editor CLI command used for remote opens.

    Args:
        preferred: Optional explicit editor command or alias from caller.

    Returns:
        Resolved command descriptor, or `None` when no supported editor exists.

    Example:
        >>> resolve_editor_command(preferred='vscode')
        EditorCommand(command='code', source='flag')
    """

    from_flag = normalize_editor_token(editor=preferred)
    if from_flag is not None:
        return EditorCommand(command=from_flag, source="flag")
    from_env = _editor_from_env()
    if from_env is not None:
        return EditorCommand(command=from_env, source="env")
    from_auto = _editor_from_candidates()
    if from_auto is not None:
        return EditorCommand(command=from_auto, source="auto")
    return None


def _editor_from_env() -> Optional[str]:
    """Resolve editor command from configured environment variables.

    Args:
        None.

    Returns:
        First normalized editor token found in configured env vars, else `None`.
    """

    for env_name in EDITOR_ENV_VARS:
        from_env = normalize_editor_token(editor=os.environ.get(env_name))
        if from_env is not None:
            return from_env
    return None


def _editor_from_candidates() -> Optional[str]:
    """Resolve editor command from installed PATH candidates.

    Args:
        None.

    Returns:
        First editor command whose executable exists on PATH, else `None`.
    """

    for candidate in EDITOR_CANDIDATES:
        if shutil.which(candidate):
            return candidate
    return None


def open_remote_target(request: RemoteOpenRequest) -> RemoteOpenResult:
    """Open a remote editor target using direct URI launch only.

    Args:
        request: Remote open request including host and optional path.

    Returns:
        Structured result describing success/failure and attempted command.

    Example:
        >>> req = RemoteOpenRequest(host="c0318", dry_run=True)
        >>> open_remote_target(request=req).ok
        True
    """

    editor_command = resolve_editor_command(preferred=request.editor)
    if editor_command is None:
        return RemoteOpenResult(
            ok=False,
            message=_missing_editor_message(),
            command=[],
        )
    command = request.direct_command(editor_command=editor_command.command)
    if request.dry_run:
        return RemoteOpenResult(
            ok=True,
            message=f"Dry run: direct remote command generated ({editor_command.source})",
            command=command,
        )
    return _execute(command=command, cwd=request.resolved_work_dir())


def _execute(command: list[str], cwd: Path) -> RemoteOpenResult:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, cwd=str(cwd))
    except FileNotFoundError as exc:
        return RemoteOpenResult(
            ok=False,
            message=f"{command[0]} not found on PATH ({exc})",
            command=command,
        )
    except OSError as exc:
        return RemoteOpenResult(
            ok=False,
            message=f"remote command failed: {exc}",
            command=command,
        )
    if proc.returncode == 0:
        return RemoteOpenResult(
            ok=True,
            message="Opened remote editor",
            command=command,
        )
    return RemoteOpenResult(
        ok=False,
        message=_result_text(proc=proc),
        command=command,
    )


def _result_text(proc: subprocess.CompletedProcess[str]) -> str:
    text = (proc.stderr or proc.stdout or "").strip()
    if text:
        return text.splitlines()[-1]
    return f"command failed with exit code {proc.returncode}"


def _missing_editor_message() -> str:
    """Return standardized guidance when no editor command can be resolved."""

    supported = ", ".join(EDITOR_CANDIDATES)
    env_vars = ", ".join(EDITOR_ENV_VARS)
    return (
        "No supported editor CLI found on PATH. "
        f"Tried: {supported}. Set one with --editor or env var ({env_vars})."
    )
