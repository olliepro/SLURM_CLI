from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

from slurm_cli.config_store import Config, find_account_entry, record_account_use
from slurm_cli.constants import (
    DEFAULT_CPUS,
    DEFAULT_MEM_GB,
    DEFAULT_SHELL,
    DEFAULT_TIME_MINUTES,
    DEFAULT_TIMEOUT_LIMIT_SECONDS,
    DEFAULT_GPUS,
    MAX_CPUS,
    TIMEOUT_IMPATIENT,
    TIMEOUT_NOTIFY,
    UI_TERMINAL,
    UI_VSCODE,
)
from slurm_cli.format_utils import (
    mem_to_gb,
    minutes_to_slurm_time,
    parse_mem,
    parse_time_string,
    sanitize_text,
)
from slurm_cli.partition_policy import list_partition_names, validate_partition_name
from slurm_cli.pickers import (
    AccountPicker,
    ResourcePicker,
    TimeoutSettingsPicker,
    UIModePicker,
)
from slurm_cli.slurm_backend import (
    build_sbatch,
    build_srun,
    get_node_for_job,
    open_vscode_on_host,
    print_release_instructions,
    start_allocation_background,
    submit_batch_job,
)


@dataclass(frozen=True)
class AccountSelection:
    """Typed account selection persisted in the config store.

    Inputs:
    - `account`: Slurm account identifier.
    - `label`: human-readable description.
    - `last_used`: unix timestamp for recency sorting.

    Outputs:
    - Immutable account selection with config serialization helper.
    """

    account: str
    label: str
    last_used: float

    def to_record(self) -> dict[str, Any]:
        """Return canonical config-store record representation."""

        return {
            "account": self.account,
            "label": self.label,
            "last_used": self.last_used,
        }


@dataclass(frozen=True)
class ResourceSelection:
    """Resolved allocation resources for one launch request.

    Inputs:
    - Requested walltime, GPU/CPU counts, memory, and optional partition.

    Outputs:
    - Immutable launch resource bundle shared by CLI and dashboard flows.
    """

    time_str: str
    time_minutes: int
    gpus: int
    cpus: int
    mem_str: str
    partition: Optional[str]


@dataclass(frozen=True)
class TimeoutSelection:
    """Resolved timeout behavior for VS Code allocation waits.

    Inputs:
    - `mode`: timeout policy (`impatient` or `notify`).
    - `limit_seconds`: wait limit before fallback/exit.
    - `email`: notification address for notify mode.

    Outputs:
    - Immutable timeout configuration.
    """

    mode: str
    limit_seconds: int
    email: str


@dataclass(frozen=True)
class LaunchFlowResult:
    """Structured outcome for launch workflows.

    Inputs:
    - `ok`: whether the workflow completed successfully.
    - `message`: status text suitable for dashboard reuse.
    - `exit_code`: process exit code used by CLI wrappers on failure.

    Outputs:
    - Immutable launch result shared across CLI and embedded callers.
    """

    ok: bool
    message: str
    exit_code: int = 0


class LaunchFlowAbort(RuntimeError):
    """Internal control-flow exception carrying CLI-style exit metadata."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


def build_default_launch_namespace() -> argparse.Namespace:
    """Return the default `gpu launch` argument namespace.

    Inputs:
    - None.

    Outputs:
    - Namespace matching `gpu launch` defaults for embedded dashboard launches.
    """

    return argparse.Namespace(
        account=None,
        cpus=None,
        dry_run=False,
        gpus=None,
        mem=None,
        notify_email=None,
        partition=None,
        shell=DEFAULT_SHELL,
        time=None,
        timeout_limit=None,
        timeout_mode=None,
        ui=None,
    )


def run_launch_command(args: argparse.Namespace) -> None:
    """Run the CLI launch flow and preserve CLI exit semantics.

    Inputs:
    - `args`: parsed `gpu launch` arguments.

    Outputs:
    - Returns on success or raises `SystemExit` with the preserved exit code.
    """

    result = run_launch_flow(args=args, embedded=False)
    if not result.ok:
        print(result.message)
        raise SystemExit(result.exit_code)


def run_launch_flow(
    args: argparse.Namespace,
    embedded: bool,
) -> LaunchFlowResult:
    """Resolve and execute the shared launch workflow.

    Inputs:
    - `args`: parsed or synthesized launch arguments.
    - `embedded`: whether the workflow should return control to its caller.

    Outputs:
    - Structured result for CLI wrappers or dashboard relaunch handling.

    Example:
        >>> result = run_launch_flow(  # doctest: +SKIP
        ...     args=build_default_launch_namespace(),
        ...     embedded=True,
        ... )
    """

    cfg = Config.load()
    try:
        account_entry = resolve_account(
            args=args,
            cfg=cfg,
            persist_selection=True,
        )
        resources = resolve_resources(args=args, cfg=cfg)
        ui_mode = resolve_ui_mode(args=args, cfg=cfg)
        timeout = resolve_timeout(args=args, cfg=cfg, ui_mode=ui_mode)
        _save_launch_defaults(
            cfg=cfg,
            resources=resources,
            ui_mode=ui_mode,
            timeout=timeout,
        )
        if ui_mode == UI_TERMINAL:
            return run_terminal_mode(
                resources=resources,
                account=account_entry.account,
                shell=args.shell,
                timeout=timeout,
                dry_run=args.dry_run,
                embedded=embedded,
            )
        return run_vscode_mode(
            resources=resources,
            account=account_entry.account,
            timeout=timeout,
            dry_run=args.dry_run,
            embedded=embedded,
        )
    except LaunchFlowAbort as exc:
        return LaunchFlowResult(
            ok=False,
            message=exc.message,
            exit_code=exc.exit_code,
        )


def run_dashboard_launch_flow(args: argparse.Namespace) -> LaunchFlowResult:
    """Resolve and submit a dashboard allocation as a background batch job.

    Inputs:
    - `args`: parsed or synthesized launch arguments.

    Outputs:
    - Structured result describing the submission outcome for dashboard relaunch.

    Example:
        >>> result = run_dashboard_launch_flow(  # doctest: +SKIP
        ...     args=build_default_launch_namespace(),
        ... )
    """

    cfg = Config.load()
    try:
        account_entry = resolve_account(
            args=args,
            cfg=cfg,
            persist_selection=not args.dry_run,
        )
        resources = resolve_resources(args=args, cfg=cfg)
        _save_dashboard_launch_defaults(cfg=cfg, resources=resources)
        return submit_dashboard_batch(
            resources=resources,
            account=account_entry.account,
            dry_run=args.dry_run,
        )
    except LaunchFlowAbort as exc:
        return LaunchFlowResult(
            ok=False,
            message=exc.message,
            exit_code=exc.exit_code,
        )


def safe_cli_text(raw: Optional[str]) -> Optional[str]:
    """Return a sanitized printable CLI string, or `None` when blank."""

    if raw is None:
        return None
    value = sanitize_text(raw.strip())
    return value if value else None


def resolve_account(
    args: argparse.Namespace,
    cfg: Config,
    persist_selection: bool,
) -> AccountSelection:
    """Resolve the account used by the launch workflow.

    Inputs:
    - `args`: launch arguments with optional explicit account override.
    - `cfg`: persisted CLI config.
    - `persist_selection`: whether to store the chosen account in config.

    Outputs:
    - Typed `AccountSelection`.
    """

    account_arg = safe_cli_text(args.account)
    if args.account is not None and account_arg is None and args.account.strip():
        _fail("--account must contain printable characters.")
    if account_arg:
        entry = find_account_entry(cfg, account_arg)
        label = entry.get("label", "") if entry else ""
        if not label:
            label = _prompt_account_description(account_id=account_arg)
        selection = AccountSelection(
            account=account_arg,
            label=label,
            last_used=time.time(),
        )
    else:
        picker = AccountPicker(cfg.recent_accounts, cfg.last_account)
        raw_selection = picker.run()
        if raw_selection is None:
            _cancel()
        assert (
            raw_selection is not None
        ), "AccountPicker returned None after cancel path"
        selection = AccountSelection(
            account=raw_selection["account"],
            label=raw_selection["label"],
            last_used=float(raw_selection["last_used"]),
        )
    if persist_selection:
        record_account_use(cfg, selection.to_record())
    return selection


def resolve_resources(args: argparse.Namespace, cfg: Config) -> ResourceSelection:
    """Resolve the resource bundle for one launch.

    Inputs:
    - `args`: launch arguments.
    - `cfg`: persisted CLI config.

    Outputs:
    - Canonical `ResourceSelection`.
    """

    partition_name, available_partitions = resolve_partition_selection(
        partition_arg=args.partition,
        cached_partition=cfg.last_partition,
    )
    time_cli = safe_cli_text(args.time)
    if args.time is not None and time_cli is None and args.time.strip():
        _fail("--time must contain printable characters.")
    time_minutes_cli = parse_time_string(time_cli) if time_cli else None
    if time_cli and time_minutes_cli is None:
        _fail("--time must be HH:MM:SS or DD-HH:MM:SS")
    time_initial = (
        time_minutes_cli
        or parse_time_string(cfg.last_time or "")
        or DEFAULT_TIME_MINUTES
    )
    gpus_cli = args.gpus if args.gpus is not None else None
    if gpus_cli is not None:
        gpus_initial = gpus_cli
    elif cfg.last_gpus is not None:
        gpus_initial = cfg.last_gpus
    else:
        gpus_initial = DEFAULT_GPUS
    cpus_cli = args.cpus if args.cpus is not None else None
    if cpus_cli is not None and not (1 <= cpus_cli <= MAX_CPUS):
        _fail(f"--cpus must be between 1 and {MAX_CPUS}")
    if cpus_cli is not None:
        cpus_initial = cpus_cli
    elif isinstance(cfg.last_cpus, int) and cfg.last_cpus > 0:
        cpus_initial = max(1, min(MAX_CPUS, cfg.last_cpus))
    elif isinstance(cfg.last_cpus, float) and cfg.last_cpus > 0:
        cpus_initial = max(1, min(MAX_CPUS, int(cfg.last_cpus)))
    else:
        cpus_initial = DEFAULT_CPUS
    mem_cli_norm = parse_mem(args.mem) if args.mem else None
    if args.mem and not mem_cli_norm:
        _fail("--mem must be like 50G or 50000M")
    mem_initial_gb = mem_to_gb(mem_cli_norm) if mem_cli_norm else None
    if mem_initial_gb is None:
        mem_initial_gb = mem_to_gb(cfg.last_mem) or DEFAULT_MEM_GB
    cpus_value = cpus_cli if cpus_cli is not None else cpus_initial
    if time_cli and gpus_cli is not None and mem_cli_norm:
        minutes_value = time_minutes_cli or time_initial
        return ResourceSelection(
            time_str=minutes_to_slurm_time(minutes_value),
            time_minutes=minutes_value,
            gpus=gpus_cli,
            cpus=cpus_value,
            mem_str=mem_cli_norm,
            partition=partition_name,
        )
    picker = ResourcePicker(
        time_minutes=time_initial,
        gpus=gpus_initial,
        cpus=cpus_initial,
        mem_gb=mem_initial_gb,
        initial_partition=partition_name,
        available_partitions=available_partitions,
    )
    result = picker.run()
    if result is None:
        _cancel()
    assert result is not None, "ResourcePicker returned None after cancel path"
    time_str, gpus, cpus, mem_str, selected_partition = result
    time_minutes = parse_time_string(time_str) or time_initial
    mem_norm = parse_mem(mem_str) or f"{mem_initial_gb}G"
    return ResourceSelection(
        time_str=time_str,
        time_minutes=time_minutes,
        gpus=gpus,
        cpus=cpus,
        mem_str=mem_norm,
        partition=selected_partition,
    )


def resolve_ui_mode(args: argparse.Namespace, cfg: Config) -> str:
    """Resolve the chosen launch attach mode."""

    if args.ui:
        return args.ui
    initial = cfg.last_ui if cfg.last_ui in (UI_TERMINAL, UI_VSCODE) else UI_TERMINAL
    picker = UIModePicker(initial)
    ui_mode = picker.run()
    if ui_mode is None:
        _cancel()
    assert ui_mode is not None, "UIModePicker returned None after cancel path"
    return ui_mode


def resolve_timeout(
    args: argparse.Namespace,
    cfg: Config,
    ui_mode: str,
) -> TimeoutSelection:
    """Resolve timeout settings for VS Code launches."""

    mode_cli = args.timeout_mode
    limit_cli = (
        args.timeout_limit if args.timeout_limit and args.timeout_limit >= 15 else None
    )
    email_cli = safe_cli_text(args.notify_email)
    if (
        args.notify_email is not None
        and email_cli is None
        and args.notify_email.strip()
    ):
        _fail("--notify-email must contain printable characters.")
    limit_initial = cfg.last_timeout_limit_seconds or DEFAULT_TIMEOUT_LIMIT_SECONDS
    limit_initial = limit_cli or limit_initial

    def finalize(mode: str, seconds: int, email: str) -> TimeoutSelection:
        normalized_seconds = max(15, seconds)
        if mode == TIMEOUT_NOTIFY:
            if not email:
                email = _prompt_email()
        else:
            email = ""
        return TimeoutSelection(
            mode=mode,
            limit_seconds=normalized_seconds,
            email=email,
        )

    overrides_supplied = any(
        [mode_cli is not None, limit_cli is not None, args.notify_email is not None]
    )
    if overrides_supplied:
        mode = mode_cli or cfg.last_timeout_mode or TIMEOUT_IMPATIENT
        email = email_cli or cfg.last_notify_email or ""
        return finalize(mode=mode, seconds=limit_initial, email=email)
    if ui_mode == UI_TERMINAL:
        return finalize(mode=TIMEOUT_IMPATIENT, seconds=limit_initial, email="")
    mode_initial = cfg.last_timeout_mode or TIMEOUT_IMPATIENT
    email_initial = cfg.last_notify_email or ""
    picker = TimeoutSettingsPicker(mode_initial, limit_initial, email_initial)
    result = picker.run()
    if result is None:
        _cancel()
    assert result is not None, "TimeoutSettingsPicker returned None after cancel path"
    mode, limit_seconds, email = result
    return finalize(mode=mode, seconds=limit_seconds, email=email)


def resolve_partition_selection(
    partition_arg: Optional[str],
    cached_partition: Optional[str] = None,
) -> tuple[Optional[str], tuple[str, ...]]:
    """Resolve the requested partition override and visible choices."""

    available_partitions = list_partition_names()
    partition_cli = safe_cli_text(partition_arg)
    if partition_arg is not None and partition_cli is None and partition_arg.strip():
        _fail("--partition must contain printable characters.")
    if partition_cli is None:
        cached_name = _valid_cached_partition(
            cached_partition=cached_partition,
            available_partitions=available_partitions,
        )
        return cached_name, available_partitions
    try:
        partition_name = validate_partition_name(
            partition_name=partition_cli,
            available_partitions=available_partitions,
        )
    except ValueError:
        available_text = ", ".join(available_partitions) if available_partitions else ""
        suffix = f" Known partitions: {available_text}" if available_text else ""
        _fail(f"--partition must match a partition on this cluster.{suffix}")
    return partition_name, available_partitions


def _valid_cached_partition(
    cached_partition: Optional[str],
    available_partitions: tuple[str, ...],
) -> Optional[str]:
    """Return cached partition only when it is valid on this cluster."""

    cached_cli = safe_cli_text(cached_partition)
    if cached_cli is None:
        return None
    try:
        return validate_partition_name(
            partition_name=cached_cli,
            available_partitions=available_partitions,
        )
    except ValueError:
        return None


def run_terminal_mode(
    resources: ResourceSelection,
    account: str,
    shell: str,
    timeout: TimeoutSelection,
    dry_run: bool,
    embedded: bool,
) -> LaunchFlowResult:
    """Run the terminal allocation flow.

    Inputs:
    - `resources`: resolved launch resources.
    - `account`: selected Slurm account.
    - `shell`: program executed under `srun --pty`.
    - `timeout`: retained for interface parity with VS Code mode.
    - `dry_run`: whether to print the command and exit.
    - `embedded`: whether control must return to the caller afterward.

    Outputs:
    - Structured launch result. In CLI mode, success usually does not return.
    """

    _ = timeout
    cmd = build_srun(
        gpus=resources.gpus,
        cpus=resources.cpus,
        time_str=resources.time_str,
        account=account,
        shell=shell,
        mem=resources.mem_str,
        partition=resources.partition,
    )
    if dry_run:
        _print_command(command=cmd)
        return LaunchFlowResult(
            ok=True, message="Dry run: printed terminal allocation command."
        )
    if sys.stdin.isatty() and not _confirm(command=cmd):
        _cancel()
    if not embedded:
        if not sys.stdin.isatty():
            os.execvp(cmd[0], cmd)
        try:
            os.execvp(cmd[0], cmd)
        except Exception:
            subprocess.run(args=cmd, check=False)
        return LaunchFlowResult(ok=True, message="")
    completed = subprocess.run(args=cmd, check=False)
    if completed.returncode != 0:
        return LaunchFlowResult(
            ok=False,
            message=f"Terminal allocation exited with code {completed.returncode}.",
            exit_code=completed.returncode,
        )
    return LaunchFlowResult(
        ok=True,
        message="Terminal allocation ended; returned to dashboard.",
    )


def run_vscode_mode(
    resources: ResourceSelection,
    account: str,
    timeout: TimeoutSelection,
    dry_run: bool,
    embedded: bool,
) -> LaunchFlowResult:
    """Run the VS Code allocation flow with optional dashboard return."""

    if dry_run:
        _print_command(
            command=_background_srun_command(
                resources=resources,
                account=account,
                job_name="slurmcli-vscode",
            )
        )
        return LaunchFlowResult(
            ok=True, message="Dry run: printed VS Code allocation command."
        )
    proc, job_id = start_allocation_background(
        gpus=resources.gpus,
        cpus=resources.cpus,
        time_str=resources.time_str,
        account=account,
        mem=resources.mem_str,
        job_name="slurmcli-vscode",
        partition=resources.partition,
    )
    print("Starting allocation in the background for VS Code…")
    if job_id is None:
        return LaunchFlowResult(
            ok=False,
            message="Unable to determine Slurm job id for VS Code allocation.",
            exit_code=3,
        )
    node = wait_for_node(job_id=job_id, timeout_seconds=timeout.limit_seconds)
    if not node:
        try:
            proc.terminate()
        except Exception:
            pass
        if timeout.mode == TIMEOUT_NOTIFY:
            return notify_batch_fallback(
                resources=resources,
                account=account,
                email=timeout.email,
                job_name="slurmcli-vscode",
                dry_run=dry_run,
                message=(
                    "Allocation did not start within the timeout limit. "
                    "Submitting a batch job to notify you when it begins..."
                ),
            )
        if embedded:
            return LaunchFlowResult(
                ok=False,
                message=f"Timed out waiting for compute node for job {job_id}.",
                exit_code=4,
            )
        raise LaunchFlowAbort(
            message=f"Timed out waiting for a compute node for job {job_id}.",
            exit_code=4,
        )
    print(f"Allocated node: {node}\nJob id: {job_id}")
    open_result = open_vscode_on_host(hostname=node)
    print_release_instructions(job_id=job_id)
    if open_result != 0:
        return LaunchFlowResult(
            ok=False,
            message=f"Remote editor command exited with code {open_result}.",
            exit_code=open_result,
        )
    if embedded:
        return LaunchFlowResult(
            ok=True,
            message="Opened remote editor; returned to dashboard.",
        )
    return LaunchFlowResult(ok=True, message="")


def submit_dashboard_batch(
    resources: ResourceSelection,
    account: str,
    dry_run: bool,
) -> LaunchFlowResult:
    """Submit one dashboard-requested allocation as a held batch job.

    Inputs:
    - `resources`: resolved allocation resources.
    - `account`: selected Slurm account.
    - `dry_run`: whether to print the command instead of submitting it.

    Outputs:
    - Structured submission result for dashboard relaunch handling.
    """

    sbatch_cmd = build_sbatch(
        gpus=resources.gpus,
        cpus=resources.cpus,
        time_str=resources.time_str,
        account=account,
        mem=resources.mem_str,
        email=None,
        job_name="slurmcli-dashboard",
        partition=resources.partition,
    )
    if dry_run:
        _print_command(command=sbatch_cmd)
        return LaunchFlowResult(
            ok=True,
            message="Dry run: printed dashboard allocation command.",
        )
    job_id = submit_batch_job(
        gpus=resources.gpus,
        cpus=resources.cpus,
        time_str=resources.time_str,
        account=account,
        mem=resources.mem_str,
        email=None,
        job_name="slurmcli-dashboard",
        partition=resources.partition,
    )
    if not job_id:
        return LaunchFlowResult(
            ok=False,
            message="ERROR: sbatch submission failed.",
            exit_code=3,
        )
    return LaunchFlowResult(
        ok=True,
        message=f"Submitted background allocation job {job_id}.",
    )


def wait_for_node(job_id: str, timeout_seconds: int) -> Optional[str]:
    """Wait for a Slurm allocation to resolve to a node name.

    Inputs:
    - `job_id`: Slurm job id for the waiting allocation.
    - `timeout_seconds`: maximum wait time.

    Outputs:
    - Allocated node name when available, else `None`.
    """

    assert timeout_seconds > 0, "timeout_seconds must be positive"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        node = get_node_for_job(job_id)
        if node:
            return node
        time.sleep(0.5)
    return None


def notify_batch_fallback(
    resources: ResourceSelection,
    account: str,
    email: str,
    job_name: str,
    dry_run: bool,
    message: str,
) -> LaunchFlowResult:
    """Submit a notify-on-begin batch fallback for VS Code mode."""

    sbatch_cmd = build_sbatch(
        gpus=resources.gpus,
        cpus=resources.cpus,
        time_str=resources.time_str,
        account=account,
        mem=resources.mem_str,
        email=email,
        job_name=job_name,
        partition=resources.partition,
    )
    if dry_run:
        print(message)
        _print_command(command=sbatch_cmd)
        return LaunchFlowResult(
            ok=True, message="Dry run: printed notify fallback command."
        )
    print(message)
    print("Submitting batch job that will notify you when the allocation begins…")
    job_id = submit_batch_job(
        gpus=resources.gpus,
        cpus=resources.cpus,
        time_str=resources.time_str,
        account=account,
        mem=resources.mem_str,
        email=email,
        job_name=job_name,
        partition=resources.partition,
    )
    if not job_id:
        return LaunchFlowResult(
            ok=False,
            message="ERROR: sbatch submission failed.",
            exit_code=3,
        )
    print(f"Submitted batch job: {job_id}")
    print_release_instructions(job_id=job_id, batch=True)
    return LaunchFlowResult(
        ok=True,
        message=f"Submitted notify fallback job {job_id}.",
    )


def _save_launch_defaults(
    cfg: Config,
    resources: ResourceSelection,
    ui_mode: str,
    timeout: TimeoutSelection,
) -> None:
    """Persist the latest successful launch defaults."""

    _save_resource_defaults(cfg=cfg, resources=resources)
    cfg.last_ui = ui_mode
    cfg.last_timeout_mode = timeout.mode
    cfg.last_notify_email = timeout.email or cfg.last_notify_email
    cfg.last_timeout_limit_seconds = timeout.limit_seconds
    cfg.save()


def _save_dashboard_launch_defaults(cfg: Config, resources: ResourceSelection) -> None:
    """Persist dashboard allocation defaults without overwriting attach settings."""

    _save_resource_defaults(cfg=cfg, resources=resources)
    cfg.save()


def _save_resource_defaults(cfg: Config, resources: ResourceSelection) -> None:
    """Persist the shared resource defaults for future allocation requests."""

    cfg.last_time = resources.time_str
    cfg.last_mem = resources.mem_str
    cfg.last_gpus = resources.gpus
    cfg.last_cpus = resources.cpus
    cfg.last_partition = resources.partition


def _background_srun_command(
    resources: ResourceSelection,
    account: str,
    job_name: str,
) -> list[str]:
    """Return the background `srun` command used by VS Code mode."""

    cmd = ["srun"]
    if resources.gpus > 0:
        cmd.append(f"--gres=gpu:{resources.gpus}")
    cmd.extend(
        [
            f"--cpus-per-task={resources.cpus}",
            f"--time={resources.time_str}",
            f"--account={account}",
            f"--mem={resources.mem_str}",
            f"--job-name={job_name}",
        ]
    )
    if resources.partition:
        cmd.append(f"--partition={resources.partition}")
    cmd.extend(["--pty", "sleep", "infinity"])
    return cmd


def _confirm(command: list[str]) -> bool:
    """Prompt the user before launching an interactive terminal allocation."""

    pretty = " ".join(shlex.quote(part) for part in command)
    print("\nAbout to run:\n  " + pretty)
    answer = safe_cli_text(input("Proceed? [Y/n]: "))
    normalized = "" if answer is None else answer.lower()
    return normalized in ("", "y", "yes")


def _prompt_account_description(account_id: str) -> str:
    """Prompt for a label when an account is not yet known locally."""

    while True:
        value = safe_cli_text(input(f"Description for account {account_id}: "))
        if value:
            return value
        print("Description is required.")


def _prompt_email() -> str:
    """Prompt for a notify email address."""

    while True:
        value = safe_cli_text(input("Notify email: "))
        if value:
            return value
        print("Email is required for notify mode.")


def _print_command(command: list[str]) -> None:
    """Print a shell-quoted command line for dry runs."""

    print(" ".join(shlex.quote(part) for part in command))


def _cancel() -> None:
    raise LaunchFlowAbort(message="Canceled.", exit_code=1)


def _fail(message: str) -> None:
    raise LaunchFlowAbort(message=message, exit_code=2)
