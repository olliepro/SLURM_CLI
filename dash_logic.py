from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


DASH_RUNNING = "R"
DASH_PENDING = "PD"
DASH_SQUEUE_FORMAT = "%i\t%t\t%j\t%M\t%L\t%R\t%N\t%Z"


@dataclass(frozen=True)
class DashJob:
    """One dashboard row derived from `squeue` output.

    Args:
        job_id: Slurm job identifier.
        state_compact: Compact state token (`PD` or `R`).
        name: Slurm job name.
        time_used: Elapsed runtime string.
        time_left: Remaining time string.
        reason: Pending/running reason field.
        node_list: NodeList string from Slurm.
        work_dir: Slurm working directory.

    Returns:
        Immutable typed job record with canonical display helpers.

    Example:
        >>> DashJob('1', 'R', 'demo', '0:01', '0:59', 'None', 'c001', '/tmp').is_running()
        True
    """

    job_id: str
    state_compact: str
    name: str
    time_used: str
    time_left: str
    reason: str
    node_list: str
    work_dir: str

    def is_running(self) -> bool:
        """Return true if this job is currently running."""

        return self.state_compact == DASH_RUNNING

    def is_pending(self) -> bool:
        """Return true if this job is currently pending."""

        return self.state_compact == DASH_PENDING

    def display_row(self) -> str:
        """Render a compact row string for terminal dashboard views."""

        return (
            f"{self.job_id:>8} {self.state_compact:<2} {self.name[:22]:<22} "
            f"{self.time_used:>9} {self.time_left:>10} "
            f"{self.reason[:18]:<18} {self.node_list[:16]:<16}"
        )


@dataclass(frozen=True)
class DashActionResult:
    """Result for dashboard actions such as cancel or join.

    Args:
        ok: Whether the action succeeded.
        message: Human-readable status message.
        affected_job_ids: Job ids targeted by the action.

    Returns:
        Immutable result object with display helper for status lines.
    """

    ok: bool
    message: str
    affected_job_ids: List[str]

    def summary_line(self) -> str:
        """Return a concise action outcome string for UI status bars."""

        state = "OK" if self.ok else "ERROR"
        targets = ",".join(self.affected_job_ids) or "-"
        return f"{state}: {self.message} ({targets})"


def fetch_dash_jobs(user_name: str) -> List[DashJob]:
    """Fetch pending/running jobs for one user from Slurm.

    Args:
        user_name: Unix username used with `squeue -u`.

    Returns:
        Sorted dashboard jobs: running first, then pending, then descending id.

    Example:
        >>> fetch_dash_jobs(user_name='demo')  # doctest: +SKIP
        [DashJob(...)]
    """

    assert user_name, "user_name is required"
    output = _squeue_output(user_name=user_name)
    jobs = _parse_jobs(output=output)
    return sorted(jobs, key=_dash_sort_key)


def cancel_dash_jobs(job_ids: List[str]) -> DashActionResult:
    """Cancel one or more jobs with `scancel`.

    Args:
        job_ids: Job ids selected in dashboard order.

    Returns:
        Structured cancel result with command outcome details.
    """

    unique_ids = _dedupe_job_ids(job_ids=job_ids)
    if not unique_ids:
        return DashActionResult(False, "No jobs selected", [])
    try:
        proc = subprocess.run(
            ["scancel", *unique_ids],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return DashActionResult(False, "scancel not found on PATH", unique_ids)
    message = _result_message(proc=proc)
    return DashActionResult(proc.returncode == 0, message, unique_ids)


def resolve_primary_host(node_list: str) -> Optional[str]:
    """Resolve the first hostname from a Slurm NodeList expression.

    Args:
        node_list: Raw NodeList string from `squeue`.

    Returns:
        First concrete hostname, or `None` when unavailable.
    """

    normalized = node_list.strip()
    if not normalized:
        return None
    if "[" not in normalized and "," not in normalized:
        return normalized
    return _expand_first_host(node_list=normalized)


def join_job_via_remote(job: DashJob) -> DashActionResult:
    """Open VS Code remote through zsh `remote()` for a running job.

    Args:
        job: Running dashboard job to join.

    Returns:
        Success/failure result describing the join attempt.

    Example:
        >>> job = DashJob('1', 'R', 'demo', '0:01', '0:59', '', 'c001', '/tmp')
        >>> join_job_via_remote(job=job)  # doctest: +SKIP
        DashActionResult(ok=True, ...)
    """

    if not job.is_running():
        return DashActionResult(False, "Only running jobs can be joined", [job.job_id])
    host = resolve_primary_host(node_list=job.node_list)
    if not host:
        return DashActionResult(False, "Unable to resolve job host", [job.job_id])
    cwd = _join_cwd(work_dir=job.work_dir)
    cmd = ["zsh", "-ic", f"remote {shlex.quote(host)}"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
    except FileNotFoundError:
        return DashActionResult(False, "zsh not found on PATH", [job.job_id])
    except OSError as exc:
        return DashActionResult(False, f"join failed: {exc}", [job.job_id])
    message = "Opened VS Code remote" if proc.returncode == 0 else _result_message(proc=proc)
    return DashActionResult(proc.returncode == 0, message, [job.job_id])


def _squeue_output(user_name: str) -> str:
    cmd = [
        "squeue",
        "-h",
        "-u",
        user_name,
        "-t",
        "PD,R",
        "-o",
        DASH_SQUEUE_FORMAT,
    ]
    return subprocess.check_output(cmd, text=True)


def _parse_jobs(output: str) -> List[DashJob]:
    jobs: List[DashJob] = []
    for raw_line in output.splitlines():
        parsed = _parse_dash_line(raw_line=raw_line)
        if parsed is not None:
            jobs.append(parsed)
    return jobs


def _parse_dash_line(raw_line: str) -> Optional[DashJob]:
    pieces = raw_line.split("\t", maxsplit=7)
    if len(pieces) != 8:
        return None
    state = pieces[1].strip()
    if state not in (DASH_PENDING, DASH_RUNNING):
        return None
    return DashJob(
        job_id=pieces[0].strip(),
        state_compact=state,
        name=pieces[2].strip(),
        time_used=pieces[3].strip(),
        time_left=pieces[4].strip(),
        reason=pieces[5].strip(),
        node_list=pieces[6].strip(),
        work_dir=pieces[7].strip(),
    )


def _dash_sort_key(job: DashJob) -> tuple[int, int]:
    state_rank = 0 if job.is_running() else 1
    job_id_number = int(job.job_id) if job.job_id.isdigit() else 0
    return state_rank, -job_id_number


def _dedupe_job_ids(job_ids: List[str]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for job_id in job_ids:
        if job_id in seen:
            continue
        seen.add(job_id)
        unique.append(job_id)
    return unique


def _expand_first_host(node_list: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["scontrol", "show", "hostnames", node_list],
            text=True,
        )
    except Exception:
        return None
    hosts = [line.strip() for line in out.splitlines() if line.strip()]
    return hosts[0] if hosts else None


def _join_cwd(work_dir: str) -> Path:
    if not work_dir:
        return Path.cwd()
    candidate = Path(work_dir)
    return candidate if candidate.exists() else Path.cwd()


def _result_message(proc: subprocess.CompletedProcess[str]) -> str:
    text = (proc.stderr or proc.stdout or "").strip()
    if text:
        return text.splitlines()[-1]
    if proc.returncode == 0:
        return "Command succeeded"
    return f"Command failed with code {proc.returncode}"
