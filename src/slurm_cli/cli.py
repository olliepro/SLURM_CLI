from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from slurm_cli.interactive_slurm import (
    parse_dash_args,
    parse_launch_args,
    parse_search_args,
    run_dash_command,
    run_launch_command,
    run_search_command,
)
from slurm_cli.remote_access import (
    RemoteOpenRequest,
    open_remote_target,
)


def build_gpu_parser() -> argparse.ArgumentParser:
    """Build root parser for `gpu` command with discoverable subcommands.

    Returns:
        Parser describing available command groups.

    Example:
        >>> build_gpu_parser().prog
        'gpu'
    """

    parser = argparse.ArgumentParser(
        prog="gpu",
        description="OSC-first Slurm workflow CLI for launch/search/dash/remote.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("launch", "search", "dash", "remote"),
        help="Subcommand to run. Defaults to launch when omitted.",
    )
    return parser


def build_remote_parser() -> argparse.ArgumentParser:
    """Build parser for the `gpu remote` subcommand.

    Returns:
        Parser with target host and remote-launch controls.

    Example:
        >>> build_remote_parser().prog
        'gpu remote'
    """

    parser = argparse.ArgumentParser(
        prog="gpu remote",
        description="Open a remote editor session to a host.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("host", help="Target hostname or OSC alias")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Remote folder path (defaults to current working directory)",
    )
    parser.add_argument(
        "--editor",
        default=None,
        help="Editor CLI command or alias (e.g. code, cursor, codium, antigravity)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command that would run without executing it",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the `gpu` command router and return a process exit code.

    Args:
        argv: Optional argv list without program name.

    Returns:
        POSIX-style process exit code.

    Example:
        >>> main(argv=["--help"])  # doctest: +SKIP
        0
    """

    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if not raw_args:
        return _run_launch(argv=[])
    if raw_args[0] in ("-h", "--help"):
        build_gpu_parser().print_help()
        return 0
    command = raw_args[0]
    if command == "launch":
        return _run_launch(argv=raw_args[1:])
    if command == "search":
        return _run_search(argv=raw_args[1:])
    if command == "dash":
        return _run_dash(argv=raw_args[1:])
    if command == "remote":
        return _run_remote(argv=raw_args[1:])
    if command.startswith("-"):
        return _run_launch(argv=raw_args)
    build_gpu_parser().error(f"unknown command: {command}")
    return 2


def _run_launch(argv: Sequence[str]) -> int:
    args = parse_launch_args(argv=argv)
    run_launch_command(args=args)
    return 0


def _run_search(argv: Sequence[str]) -> int:
    args = parse_search_args(argv=argv)
    run_search_command(args=args)
    return 0


def _run_dash(argv: Sequence[str]) -> int:
    args = parse_dash_args(argv=argv)
    run_dash_command(args=args)
    return 0


def _run_remote(argv: Sequence[str]) -> int:
    args = build_remote_parser().parse_args(args=list(argv))
    result = open_remote_target(
        request=RemoteOpenRequest(
            host=args.host,
            work_dir=args.path,
            editor=args.editor,
            dry_run=args.dry_run,
        )
    )
    if result.command:
        print(result.command_text())
    print(result.message)
    return 0 if result.ok else 2
