#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NoReturn, Optional, Sequence
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from slurm_cli.config_store import Config, find_account_entry, record_account_use
from slurm_cli.constants import (
    DEFAULT_MEM_GB,
    DEFAULT_SHELL,
    DEFAULT_TIMEOUT_LIMIT_SECONDS,
    DEFAULT_TIME_MINUTES,
    DEFAULT_GPUS,
    DEFAULT_CPUS,
    MAX_CPUS,
    SEARCH_DEFAULT_MIN_GPUS,
    SEARCH_DEFAULT_MIN_TIME_MINUTES,
    SEARCH_JOB_PREFIX,
    SEARCH_SUBMIT_GAP_SECONDS,
    SEARCH_SWITCH_MINUTES,
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
from slurm_cli.dash_ui import run_dash_dashboard
from slurm_cli.pickers import (
    AccountPicker,
    ResourcePicker,
    TimeoutSettingsPicker,
    UIModePicker,
)
from slurm_cli.search_ui import (
    SearchBoundsPicker,
    SearchEmailPicker,
    SearchSubmissionDashboard,
)
from slurm_cli.search_logic import (
    SEARCH_STATUS_FAILED,
    SearchBounds,
    SearchProbe,
    SearchSubmissionResult,
    build_probe_command,
    build_search_probes,
    submit_search_probes,
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


@dataclass
class ResourceSelection:
    time_str: str
    time_minutes: int
    gpus: int
    cpus: int
    mem_str: str


@dataclass
class TimeoutSelection:
    mode: str
    limit_seconds: int
    email: str


@dataclass
class SearchSelection:
    account: str
    resources: ResourceSelection
    bounds: SearchBounds
    notify_email: str


def build_launch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple Slurm interactive job launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shell", default=DEFAULT_SHELL, help="Shell/program to run (terminal mode)"
    )
    parser.add_argument("--time", help="Skip UI and provide time as HH:MM:SS")
    parser.add_argument("--account", help="Account to charge (overrides saved default)")
    parser.add_argument(
        "--gpus", type=int, choices=range(0, 5), help="Number of GPUs 0-4 (skip UI)"
    )
    parser.add_argument(
        "--cpus",
        type=int,
        help="Number of CPUs per task (skip UI)",
    )
    parser.add_argument("--mem", help="Memory request (e.g., 50G, 50000M)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the command and exit"
    )
    parser.add_argument("--ui", choices=[UI_TERMINAL, UI_VSCODE], help="Attach mode")
    parser.add_argument(
        "--timeout-mode",
        choices=[TIMEOUT_IMPATIENT, TIMEOUT_NOTIFY],
        help="Timeout behavior override",
    )
    parser.add_argument(
        "--timeout-limit",
        type=int,
        help="Timeout limit in seconds for impatient mode (>=15).",
    )
    parser.add_argument("--notify-email", help="Email for notify mode override")
    return parser


def parse_launch_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_launch_parser()
    return parser.parse_args(args=list(argv) if argv is not None else None)


def _parse_time_minutes_arg(raw: str) -> int:
    value = safe_cli_text(raw)
    if value is None:
        raise argparse.ArgumentTypeError("time must contain printable characters")
    minutes = parse_time_string(value)
    if minutes is None:
        raise argparse.ArgumentTypeError("time must be HH:MM:SS or DD-HH:MM:SS")
    return minutes


def build_search_parser() -> argparse.ArgumentParser:
    """Create the parser for the `search` subcommand flags."""

    parser = argparse.ArgumentParser(
        prog="interactive_slurm.py search",
        description="Submit parallel two-phase halving search probes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--account", help="Account to charge (overrides saved default)")
    parser.add_argument(
        "--cpus",
        type=int,
        help="CPUs per task for all search probes",
    )
    parser.add_argument("--mem", help="Memory request (e.g., 50G, 50000M)")
    parser.add_argument("--notify-email", help="Email for BEGIN notifications")
    parser.add_argument(
        "--max-time",
        dest="max_time_minutes",
        type=_parse_time_minutes_arg,
        help="Maximum time in HH:MM:SS or DD-HH:MM:SS",
    )
    parser.add_argument(
        "--min-time",
        dest="min_time_minutes",
        type=_parse_time_minutes_arg,
        help="Minimum time in HH:MM:SS or DD-HH:MM:SS",
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        choices=range(1, 5),
        help="Maximum GPU count (1-4)",
    )
    parser.add_argument(
        "--min-gpus",
        type=int,
        choices=range(1, 5),
        help="Minimum GPU count (1-4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned commands and exit"
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    return parser


def parse_search_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse validated search args from `argv`.

    Example:
        >>> parse_search_args(['--max-time', '01:00:00', '--max-gpus', '2']).max_gpus
        2
    """

    parser = build_search_parser()
    args = parser.parse_args(args=list(argv) if argv is not None else None)
    if (
        args.max_time_minutes is not None
        and args.min_time_minutes is not None
        and args.min_time_minutes > args.max_time_minutes
    ):
        parser.error("--min-time must be <= --max-time.")
    if (
        args.max_gpus is not None
        and args.min_gpus is not None
        and args.min_gpus > args.max_gpus
    ):
        parser.error("--min-gpus must be <= --max-gpus.")
    return args


def build_dash_parser() -> argparse.ArgumentParser:
    """Create parser for the `dash` subcommand."""

    return argparse.ArgumentParser(
        prog="interactive_slurm.py dash",
        description="Interactive dashboard for pending/running jobs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def parse_dash_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse arguments for the `dash` subcommand."""

    parser = build_dash_parser()
    return parser.parse_args(args=list(argv) if argv is not None else None)


def parse_args() -> argparse.Namespace:
    return parse_launch_args(argv=None)


def safe_cli_text(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = sanitize_text(raw.strip())
    return value if value else None


def resolve_account(
    args: argparse.Namespace,
    cfg: Config,
    persist_selection: bool = True,
) -> Dict[str, Any]:
    account_arg = safe_cli_text(args.account)
    if args.account is not None and account_arg is None:
        fail("--account must contain printable characters.")
    if account_arg:
        entry = find_account_entry(cfg, account_arg)
        label = entry.get("label", "") if entry else ""
        if not label:
            label = prompt_account_description(account_arg)
        selected = {"account": account_arg, "label": label, "last_used": time.time()}
        if persist_selection:
            record_account_use(cfg, selected)
        return selected
    picker = AccountPicker(cfg.recent_accounts, cfg.last_account)
    selection = picker.run()
    if selection is None:
        cancel()
    if persist_selection:
        record_account_use(cfg, selection)
    return selection


def resolve_resources(args: argparse.Namespace, cfg: Config) -> ResourceSelection:
    time_cli = safe_cli_text(args.time)
    if args.time is not None and time_cli is None and args.time.strip():
        fail("--time must contain printable characters.")
    time_minutes_cli = parse_time_string(time_cli) if time_cli else None
    if time_cli and time_minutes_cli is None:
        fail("--time must be HH:MM:SS or DD-HH:MM:SS")
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
        fail(f"--cpus must be between 1 and {MAX_CPUS}")
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
        fail("--mem must be like 50G or 50000M")
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
        )
    picker = ResourcePicker(time_initial, gpus_initial, cpus_initial, mem_initial_gb)
    result = picker.run()
    if result is None:
        cancel()
    time_str, gpus, cpus, mem_str = result
    time_minutes = parse_time_string(time_str) or time_initial
    mem_norm = parse_mem(mem_str) or f"{mem_initial_gb}G"
    return ResourceSelection(
        time_str=time_str,
        time_minutes=time_minutes,
        gpus=gpus,
        cpus=cpus,
        mem_str=mem_norm,
    )


def _resolve_cpus(cpus_cli: Optional[int], cfg: Config) -> int:
    if cpus_cli is not None and not (1 <= cpus_cli <= MAX_CPUS):
        fail(f"--cpus must be between 1 and {MAX_CPUS}")
    if cpus_cli is not None:
        return cpus_cli
    if isinstance(cfg.last_cpus, (int, float)) and cfg.last_cpus > 0:
        return max(1, min(MAX_CPUS, int(cfg.last_cpus)))
    return DEFAULT_CPUS


def _resolve_mem(mem_arg: Optional[str], cfg: Config) -> tuple[Optional[str], int]:
    mem_cli = parse_mem(mem_arg) if mem_arg else None
    if mem_arg and mem_cli is None:
        fail("--mem must be like 50G or 50000M")
    mem_initial = mem_to_gb(mem_cli) if mem_cli else None
    if mem_initial is None:
        mem_initial = mem_to_gb(cfg.last_mem) or DEFAULT_MEM_GB
    return mem_cli, mem_initial


def _initial_search_max_time(args: argparse.Namespace, cfg: Config) -> int:
    if args.max_time_minutes is not None:
        return args.max_time_minutes
    if cfg.last_search_max_time_minutes:
        return cfg.last_search_max_time_minutes
    return parse_time_string(cfg.last_time or "") or DEFAULT_TIME_MINUTES


def _initial_search_max_gpus(args: argparse.Namespace, cfg: Config) -> int:
    if args.max_gpus is not None:
        return args.max_gpus
    if cfg.last_search_max_gpus:
        return max(1, min(4, cfg.last_search_max_gpus))
    if cfg.last_gpus is not None:
        return max(1, min(4, cfg.last_gpus))
    return max(1, min(4, DEFAULT_GPUS))


def resolve_search_resources(args: argparse.Namespace, cfg: Config) -> ResourceSelection:
    cpus_value = _resolve_cpus(cpus_cli=args.cpus, cfg=cfg)
    mem_cli, mem_initial = _resolve_mem(mem_arg=args.mem, cfg=cfg)
    time_initial = _initial_search_max_time(args=args, cfg=cfg)
    gpus_initial = _initial_search_max_gpus(args=args, cfg=cfg)
    if args.max_time_minutes is not None and args.max_gpus is not None and mem_cli:
        return ResourceSelection(
            time_str=minutes_to_slurm_time(args.max_time_minutes),
            time_minutes=args.max_time_minutes,
            gpus=args.max_gpus,
            cpus=cpus_value,
            mem_str=mem_cli,
        )
    picker = ResourcePicker(time_initial, gpus_initial, cpus_value, mem_initial)
    result = picker.run()
    if result is None:
        cancel()
    time_str, gpus, cpus, mem_str = result
    if gpus < 1:
        fail("search requires at least 1 GPU.")
    time_minutes = parse_time_string(time_str) or time_initial
    mem_norm = parse_mem(mem_str) or f"{mem_initial}G"
    return ResourceSelection(
        time_str=time_str,
        time_minutes=time_minutes,
        gpus=gpus,
        cpus=cpus,
        mem_str=mem_norm,
    )


def _bounded_min_time(value: int, max_time: int) -> int:
    return max(SEARCH_DEFAULT_MIN_TIME_MINUTES, min(max_time, value))


def _bounded_min_gpus(value: int, max_gpus: int) -> int:
    return max(SEARCH_DEFAULT_MIN_GPUS, min(max_gpus, value))


def resolve_search_bounds(
    args: argparse.Namespace,
    cfg: Config,
    resources: ResourceSelection,
) -> SearchBounds:
    min_time_seed = (
        args.min_time_minutes
        or cfg.last_search_min_time_minutes
        or SEARCH_DEFAULT_MIN_TIME_MINUTES
    )
    min_gpu_seed = args.min_gpus or cfg.last_search_min_gpus or SEARCH_DEFAULT_MIN_GPUS
    min_time_initial = _bounded_min_time(min_time_seed, resources.time_minutes)
    min_gpu_initial = _bounded_min_gpus(min_gpu_seed, resources.gpus)
    if args.min_time_minutes is not None and args.min_gpus is not None:
        min_time = args.min_time_minutes
        min_gpus = args.min_gpus
    else:
        picker = SearchBoundsPicker(
            max_time_minutes=resources.time_minutes,
            max_gpus=resources.gpus,
            min_time_minutes=min_time_initial,
            min_gpus=min_gpu_initial,
        )
        result = picker.run()
        if result is None:
            cancel()
        min_time, min_gpus = result
    if min_time > resources.time_minutes:
        fail("--min-time must be <= max search time.")
    if min_gpus > resources.gpus:
        fail("--min-gpus must be <= max search GPUs.")
    return SearchBounds(
        max_time_minutes=resources.time_minutes,
        min_time_minutes=min_time,
        max_gpus=resources.gpus,
        min_gpus=min_gpus,
        switch_minutes=SEARCH_SWITCH_MINUTES,
    )


def resolve_search_email(args: argparse.Namespace, cfg: Config) -> str:
    email_cli = safe_cli_text(args.notify_email)
    if (
        args.notify_email is not None
        and email_cli is None
        and args.notify_email.strip()
    ):
        fail("--notify-email must contain printable characters.")
    if email_cli:
        return email_cli
    picker = SearchEmailPicker(initial_email=cfg.last_notify_email)
    selection = picker.run()
    if not selection:
        cancel()
    return selection


def resolve_search_selection(
    args: argparse.Namespace,
    cfg: Config,
    persist_selection: bool,
) -> SearchSelection:
    """Resolve account, resource maxima, bounds, and notify email for search."""

    account_entry = resolve_account(
        args=args,
        cfg=cfg,
        persist_selection=persist_selection,
    )
    resources = resolve_search_resources(args=args, cfg=cfg)
    bounds = resolve_search_bounds(args=args, cfg=cfg, resources=resources)
    notify_email = resolve_search_email(args=args, cfg=cfg)
    return SearchSelection(
        account=account_entry["account"],
        resources=resources,
        bounds=bounds,
        notify_email=notify_email,
    )


def resolve_ui_mode(args: argparse.Namespace, cfg: Config) -> str:
    if args.ui:
        return args.ui
    initial = cfg.last_ui if cfg.last_ui in (UI_TERMINAL, UI_VSCODE) else UI_TERMINAL
    picker = UIModePicker(initial)
    ui_mode = picker.run()
    if ui_mode is None:
        cancel()
    return ui_mode


def resolve_timeout(
    args: argparse.Namespace, cfg: Config, ui_mode: str
) -> TimeoutSelection:
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
        fail("--notify-email must contain printable characters.")
    limit_initial = cfg.last_timeout_limit_seconds or DEFAULT_TIMEOUT_LIMIT_SECONDS
    limit_initial = limit_cli or limit_initial

    def finalize(mode: str, seconds: int, email: str) -> TimeoutSelection:
        seconds = max(15, seconds)
        if mode == TIMEOUT_NOTIFY:
            if not email:
                email = prompt_email()
        else:
            email = ""
        return TimeoutSelection(mode, seconds, email)

    overrides_supplied = any(
        [
            mode_cli is not None,
            limit_cli is not None,
            args.notify_email is not None,
        ]
    )
    if overrides_supplied:
        mode = mode_cli or cfg.last_timeout_mode or TIMEOUT_IMPATIENT
        email = email_cli or cfg.last_notify_email or ""
        return finalize(mode, limit_initial, email)

    if ui_mode == UI_TERMINAL:
        return finalize(TIMEOUT_IMPATIENT, limit_initial, "")

    mode_initial = cfg.last_timeout_mode or TIMEOUT_IMPATIENT
    email_initial = cfg.last_notify_email or ""
    picker = TimeoutSettingsPicker(mode_initial, limit_initial, email_initial)
    result = picker.run()
    if result is None:
        cancel()
    mode, limit_seconds, email = result
    return finalize(mode, limit_seconds, email)


def run_terminal_mode(
    resources: ResourceSelection,
    account: str,
    shell: str,
    timeout: TimeoutSelection,
    dry_run: bool,
) -> None:
    cmd = build_srun(
        gpus=resources.gpus,
        cpus=resources.cpus,
        time_str=resources.time_str,
        account=account,
        shell=shell,
        mem=resources.mem_str,
    )

    print_cmd(cmd, dry_run)
    if dry_run:
        return
    if not sys.stdin.isatty():
        os.execvp(cmd[0], cmd)
    if confirm(cmd):
        try:
            os.execvp(cmd[0], cmd)
        except Exception:
            os.system(" ".join(shlex.quote(part) for part in cmd))


def run_vscode_mode(
    resources: ResourceSelection, account: str, timeout: TimeoutSelection, dry_run: bool
) -> None:
    proc, job_id = start_allocation_background(
        gpus=resources.gpus,
        cpus=resources.cpus,
        time_str=resources.time_str,
        account=account,
        mem=resources.mem_str,
        job_name="slurmcli-vscode",
    )
    print("Starting allocation in the background for VS Code…")
    assert job_id is not None, "srun command failed"

    node = wait_for_node(job_id, timeout.limit_seconds)

    if not node:
        try:
            proc.terminate()
        except Exception:
            pass
        if timeout.mode == TIMEOUT_NOTIFY:
            notify_batch_fallback(
                resources,
                account,
                timeout.email,
                "slurmcli-vscode",
                dry_run,
                "Allocation did not start within the timeout limit. Submitting a batch job to notify you when it begins...",
            )
            return
        print(f"Timed out waiting for a compute node for job {job_id}.")
        sys.exit(4)
    print(f"Allocated node: {node}\nJob id: {job_id}")
    open_vscode_on_host(node)
    print_release_instructions(job_id)


def _skip_search_confirmation(assume_yes: bool) -> bool:
    return assume_yes or not sys.stdin.isatty()


def _has_failed_submissions(results: Sequence[SearchSubmissionResult]) -> bool:
    return any(result.status == SEARCH_STATUS_FAILED for result in results)


def _search_submitter(
    selection: SearchSelection,
    probes: Sequence[SearchProbe],
    callback: Optional[Callable[[SearchSubmissionResult], None]] = None,
    dry_run: bool = False,
) -> List[SearchSubmissionResult]:
    return submit_search_probes(
        probes=probes,
        account=selection.account,
        email=selection.notify_email,
        gap_seconds=SEARCH_SUBMIT_GAP_SECONDS,
        job_prefix=SEARCH_JOB_PREFIX,
        dry_run=dry_run,
        status_callback=callback,
    )


def _print_search_plan(selection: SearchSelection, probes: Sequence[SearchProbe]) -> None:
    print("\n=== Search plan ===")
    print(f"Account: {selection.account}")
    print(f"Notify email: {selection.notify_email}")
    print(
        f"Fixed resources: cpus={selection.resources.cpus} mem={selection.resources.mem_str}"
    )
    print(
        "Bounds: "
        f"max={minutes_to_slurm_time(selection.bounds.max_time_minutes)}"
        f"/g{selection.bounds.max_gpus} "
        f"min={minutes_to_slurm_time(selection.bounds.min_time_minutes)}"
        f"/g{selection.bounds.min_gpus}"
    )
    for probe in probes:
        print(f"  {probe.summary_line(prefix=SEARCH_JOB_PREFIX)}")


def _print_search_dry_run(
    selection: SearchSelection,
    probes: Sequence[SearchProbe],
) -> None:
    print("\nDry-run sbatch commands:")
    for probe in probes:
        cmd = build_probe_command(
            probe=probe,
            account=selection.account,
            email=selection.notify_email,
            job_prefix=SEARCH_JOB_PREFIX,
        )
        print(" ".join(shlex.quote(part) for part in cmd))


def _confirm_search_submission() -> bool:
    ans = safe_cli_text(input("Submit these search jobs? [Y/n]: "))
    normalized = "" if ans is None else ans.lower()
    return normalized in ("", "y", "yes")


def _should_use_search_dashboard() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _print_submission_progress(result: SearchSubmissionResult) -> None:
    print(result.summary_line())


def _run_search_submission(
    selection: SearchSelection,
    probes: Sequence[SearchProbe],
    require_confirmation: bool,
) -> tuple[List[SearchSubmissionResult], bool]:
    if _should_use_search_dashboard():
        dashboard = SearchSubmissionDashboard(probes=probes, job_prefix=SEARCH_JOB_PREFIX)
        results = dashboard.run(
            submitter=lambda callback: _search_submitter(
                selection=selection,
                probes=probes,
                callback=callback,
                dry_run=False,
            ),
            require_confirmation=require_confirmation,
        )
        if dashboard.was_canceled:
            return [], True
        if results is not None:
            return results, False
    if require_confirmation and not _confirm_search_submission():
        return [], True
    return (
        _search_submitter(
            selection=selection,
            probes=probes,
            callback=_print_submission_progress,
            dry_run=False,
        ),
        False,
    )


def _request_text_confirmation(assume_yes: bool) -> bool:
    if _skip_search_confirmation(assume_yes=assume_yes):
        return True
    return _confirm_search_submission()


def _should_confirm_in_dashboard(assume_yes: bool) -> bool:
    return not _skip_search_confirmation(assume_yes=assume_yes)


def _submit_and_report(
    selection: SearchSelection,
    probes: Sequence[SearchProbe],
    assume_yes: bool,
) -> None:
    results, canceled_by_user = _run_search_submission(
        selection=selection,
        probes=probes,
        require_confirmation=_should_confirm_in_dashboard(assume_yes=assume_yes),
    )
    if canceled_by_user:
        cancel()
    _print_search_summary(results=results)
    if _has_failed_submissions(results):
        sys.exit(3)


def _print_search_summary(results: Sequence[SearchSubmissionResult]) -> None:
    print("\n=== Search submission report ===")
    for result in results:
        print(f"  {result.summary_line()}")
    submitted = [result.job_id for result in results if result.job_id]
    if submitted:
        print("\nSubmitted job ids:")
        for job_id in submitted:
            print(f"  {job_id}")
        print("Release any probe manually with: scancel <job_id>")
    if _has_failed_submissions(results):
        print("One or more submissions failed.")


def run_search_mode(
    selection: SearchSelection,
    dry_run: bool,
    assume_yes: bool,
) -> None:
    """Submit a two-phase search probe sequence and report outcomes."""

    probes = build_search_probes(
        bounds=selection.bounds,
        cpus=selection.resources.cpus,
        mem_str=selection.resources.mem_str,
    )
    _print_search_plan(selection=selection, probes=probes)
    if dry_run:
        if not _request_text_confirmation(assume_yes=assume_yes):
            cancel()
        _print_search_dry_run(selection=selection, probes=probes)
        return
    _submit_and_report(selection=selection, probes=probes, assume_yes=assume_yes)


def wait_for_node(job_id: str, timeout_seconds: int) -> Optional[str]:
    assert timeout_seconds > 0, "timeout_seconds must be positive"
    total_wait = timeout_seconds
    deadline = time.time() + total_wait
    while time.time() < deadline:
        node = get_node_for_job(job_id)
        if node:
            return node
        time.sleep(0.5)
    return None


def print_cmd(cmd: list, dry_run: bool) -> None:
    if dry_run:
        pretty = " ".join(shlex.quote(part) for part in cmd)
        print(pretty)


def notify_batch_fallback(
    resources: ResourceSelection,
    account: str,
    email: str,
    job_name: str,
    dry_run: bool,
    message: str,
) -> None:
    sbatch_cmd = build_sbatch(
        gpus=resources.gpus,
        cpus=resources.cpus,
        time_str=resources.time_str,
        account=account,
        mem=resources.mem_str,
        email=email,
        job_name=job_name,
    )
    if dry_run:
        print(message)
        print(" ".join(shlex.quote(part) for part in sbatch_cmd))
        return
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
    )
    if not job_id:
        print("ERROR: sbatch submission failed.")
        sys.exit(3)
    print(f"Submitted batch job: {job_id}")
    print_release_instructions(job_id, batch=True)


def confirm(cmd: list) -> bool:
    pretty = " ".join(shlex.quote(part) for part in cmd)
    print("\nAbout to run:\n  " + pretty)
    ans = safe_cli_text(input("Proceed? [Y/n]: "))
    ans = "" if ans is None else ans.lower()
    return ans in ("", "y", "yes")


def prompt_account_description(account_id: str) -> str:
    while True:
        value = safe_cli_text(input(f"Description for account {account_id}: "))
        if value:
            return value
        print("Description is required.")


def prompt_email() -> str:
    while True:
        value = safe_cli_text(input("Notify email: "))
        if value:
            return value
        print("Email is required for notify mode.")


def cancel() -> NoReturn:
    print("Canceled.")
    sys.exit(1)


def fail(message: str) -> NoReturn:
    print(message)
    sys.exit(2)


def _save_launch_defaults(
    cfg: Config,
    resources: ResourceSelection,
    ui_mode: str,
    timeout: TimeoutSelection,
) -> None:
    cfg.last_time = resources.time_str
    cfg.last_mem = resources.mem_str
    cfg.last_gpus = resources.gpus
    cfg.last_cpus = resources.cpus
    cfg.last_ui = ui_mode
    cfg.last_timeout_mode = timeout.mode
    cfg.last_notify_email = timeout.email or cfg.last_notify_email
    cfg.last_timeout_limit_seconds = timeout.limit_seconds
    cfg.save()


def run_launch_command(args: argparse.Namespace) -> None:
    cfg = Config.load()
    account_entry = resolve_account(args=args, cfg=cfg, persist_selection=True)
    resources = resolve_resources(args, cfg)
    ui_mode = resolve_ui_mode(args, cfg)
    timeout = resolve_timeout(args, cfg, ui_mode)
    _save_launch_defaults(
        cfg=cfg,
        resources=resources,
        ui_mode=ui_mode,
        timeout=timeout,
    )
    if ui_mode == UI_TERMINAL:
        run_terminal_mode(
            resources, account_entry["account"], args.shell, timeout, args.dry_run
        )
    else:
        run_vscode_mode(resources, account_entry["account"], timeout, args.dry_run)


def _save_search_defaults(cfg: Config, selection: SearchSelection) -> None:
    cfg.last_time = selection.resources.time_str
    cfg.last_mem = selection.resources.mem_str
    cfg.last_gpus = selection.resources.gpus
    cfg.last_cpus = selection.resources.cpus
    cfg.last_notify_email = selection.notify_email
    cfg.last_search_max_time_minutes = selection.resources.time_minutes
    cfg.last_search_min_time_minutes = selection.bounds.min_time_minutes
    cfg.last_search_max_gpus = selection.resources.gpus
    cfg.last_search_min_gpus = selection.bounds.min_gpus
    cfg.save()


def run_search_command(args: argparse.Namespace) -> None:
    cfg = Config.load()
    selection = resolve_search_selection(
        args=args,
        cfg=cfg,
        persist_selection=not args.dry_run,
    )
    run_search_mode(
        selection=selection,
        dry_run=args.dry_run,
        assume_yes=args.yes,
    )
    if not args.dry_run:
        _save_search_defaults(cfg=cfg, selection=selection)


def run_dash_command(args: argparse.Namespace) -> None:
    _ = args
    user_name = safe_cli_text(os.environ.get("USER"))
    if not user_name:
        print("ERROR: USER is not set.")
        sys.exit(2)
    exit_code = run_dash_dashboard(user_name=user_name)
    if exit_code != 0:
        sys.exit(exit_code)


def _is_dash_invocation(argv: Sequence[str]) -> bool:
    return bool(argv) and argv[0] == "dash"


def _is_search_invocation(argv: Sequence[str]) -> bool:
    return bool(argv) and argv[0] == "search"


def main() -> None:
    argv = sys.argv[1:]
    if _is_dash_invocation(argv=argv):
        dash_args = parse_dash_args(argv=argv[1:])
        run_dash_command(args=dash_args)
        return
    if _is_search_invocation(argv=argv):
        search_args = parse_search_args(argv=argv[1:])
        run_search_command(args=search_args)
        return
    launch_args = parse_launch_args(argv=argv)
    run_launch_command(args=launch_args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cancel()
