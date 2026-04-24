from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from slurm_cli.format_utils import format_hours_minutes_compact
from slurm_cli.remote_access import (
    RemoteOpenRequest,
    open_remote_target,
)

DASH_RUNNING = "R"
DASH_PENDING = "PD"
DASH_SQUEUE_FORMAT = "%i\t%t\t%j\t%M\t%L\t%S\t%R\t%N\t%Z"
BLAME_SQUEUE_FORMAT = "%u|%a|%b|%D|%l|%t"
UNKNOWN_START_VALUES = {"", "N/A", "Unknown", "None", "(null)"}


@dataclass(frozen=True)
class DashTableLayout:
    """Column layout used to render dashboard job tables.

    Inputs:
    - Total available row width after selection/focus prefixes.

    Outputs:
    - Immutable column widths plus aligned header/render helpers.
    """

    name_width: int
    reason_width: int
    node_width: int
    job_id_width: int = 8
    state_width: int = 2
    time_used_width: int = 8
    time_left_width: int = 8
    eta_width: int = 6

    @classmethod
    def from_width(cls, total_width: int) -> "DashTableLayout":
        """Build a best-effort layout for one available row width."""

        fixed_width = 39
        flexible_extra = max(0, total_width - fixed_width - 20)
        name_width = 8 + (flexible_extra // 2)
        reason_width = 6 + (flexible_extra // 4)
        node_width = 6 + flexible_extra - (flexible_extra // 2) - (flexible_extra // 4)
        return cls(
            name_width=name_width,
            reason_width=reason_width,
            node_width=node_width,
        )

    def header_row(self) -> str:
        """Return the aligned dashboard header row."""

        return " ".join(
            [
                f"{'JOBID':>{self.job_id_width}}",
                f"{'ST':<{self.state_width}}",
                _clip_text(text="NAME", width=self.name_width, align="left"),
                f"{'USED':>{self.time_used_width}}",
                f"{'LEFT':>{self.time_left_width}}",
                f"{'ETA':>{self.eta_width}}",
                _clip_text(text="REASON", width=self.reason_width, align="left"),
                _clip_text(text="NODE", width=self.node_width, align="left"),
            ]
        )

    def render_job(self, job: "DashJob", as_of: datetime | None) -> str:
        """Return one aligned job row for the dashboard table."""

        return " ".join(
            [
                f"{job.job_id:>{self.job_id_width}}",
                f"{job.state_compact:<{self.state_width}}",
                _clip_text(text=job.name, width=self.name_width, align="left"),
                _clip_text(
                    text=job.time_used, width=self.time_used_width, align="right"
                ),
                _clip_text(
                    text=job.time_left, width=self.time_left_width, align="right"
                ),
                _clip_text(
                    text=job.eta_text(as_of=as_of), width=self.eta_width, align="right"
                ),
                _clip_text(text=job.reason, width=self.reason_width, align="left"),
                _clip_text(text=job.node_list, width=self.node_width, align="left"),
            ]
        )


@dataclass(frozen=True)
class DashJob:
    """One dashboard row derived from `squeue` output.

    Args:
        job_id: Slurm job identifier.
        state_compact: Compact state token (`PD` or `R`).
        name: Slurm job name.
        time_used: Elapsed runtime string.
        time_left: Remaining time string.
        start_time: Slurm actual/expected start time when available.
        reason: Pending/running reason field.
        node_list: NodeList string from Slurm.
        work_dir: Slurm working directory.

    Returns:
        Immutable typed job record with canonical display helpers.

    Example:
        >>> DashJob('1', 'R', 'demo', '0:01', '0:59', 'None', 'c001', 'workdir').is_running()
        True
    """

    job_id: str
    state_compact: str
    name: str
    time_used: str
    time_left: str
    start_time: datetime | None
    reason: str
    node_list: str
    work_dir: str

    def is_running(self) -> bool:
        """Return true if this job is currently running."""

        return self.state_compact == DASH_RUNNING

    def is_pending(self) -> bool:
        """Return true if this job is currently pending."""

        return self.state_compact == DASH_PENDING

    def eta_text(self, as_of: datetime | None) -> str:
        """Return the canonical start ETA text from one refresh timestamp.

        Inputs:
        - `as_of`: refresh timestamp used as the ETA reference point.

        Outputs:
        - `0h00m` for running jobs, `--` when start time is unknown, else a
          compact hours/minutes offset clamped at zero.
        """

        if self.is_running():
            return format_hours_minutes_compact(total_minutes=0)
        if as_of is None or self.start_time is None:
            return "--"
        delta_minutes = int(max(0.0, (self.start_time - as_of).total_seconds()) // 60)
        return format_hours_minutes_compact(total_minutes=delta_minutes)


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


@dataclass(frozen=True)
class BlameRecord:
    """Aggregated GPU usage stats for one user.

    Args:
        username: Unix username.
        account: Slurm account/project.
        running_gpus: Total GPUs currently running.
        pending_gpus: Total GPUs waiting in queue.
        avg_request_minutes: Average time limit of running jobs.
    """

    username: str
    account: str
    running_gpus: int
    pending_gpus: int
    avg_request_minutes: float
    full_name: str
    coordinator_name: str


def fetch_blame_records() -> List[BlameRecord]:
    """Fetch usage stats for all users with active GPU jobs.

    Returns:
        List of records sorted by gpu_count descending.
    """

    cmd = [
        "squeue",
        "-h",
        "-t",
        "R,PD",
        "-o",
        BLAME_SQUEUE_FORMAT,
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError:
        return []

    return _parse_blame_output(output)


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


def join_job_via_remote(
    job: DashJob,
    editor: Optional[str] = None,
) -> DashActionResult:
    """Open remote editor for a running dashboard job.

    Args:
        job: Running dashboard job to join.
        editor: Optional editor command or alias override.

    Returns:
        Success/failure result describing the join attempt.

    Example:
        >>> job = DashJob('1', 'R', 'demo', '0:01', '0:59', '', 'c001', 'workdir')
        >>> join_job_via_remote(job=job, editor='cursor')  # doctest: +SKIP
        DashActionResult(ok=True, ...)
    """

    if not job.is_running():
        return DashActionResult(False, "Only running jobs can be joined", [job.job_id])
    host = resolve_primary_host(node_list=job.node_list)
    if not host:
        return DashActionResult(False, "Unable to resolve job host", [job.job_id])
    result = open_remote_target(
        request=RemoteOpenRequest(
            host=host,
            work_dir=_join_cwd(work_dir=job.work_dir),
            editor=editor,
        )
    )
    return DashActionResult(
        ok=result.ok, message=result.message, affected_job_ids=[job.job_id]
    )


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
    pieces = raw_line.split("\t", maxsplit=8)
    if len(pieces) != 9:
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
        start_time=_parse_dash_start_time(value=pieces[5].strip()),
        reason=pieces[6].strip(),
        node_list=pieces[7].strip(),
        work_dir=pieces[8].strip(),
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


def _parse_dash_start_time(value: str) -> datetime | None:
    """Parse one `squeue %S` timestamp into a local datetime."""

    if value in UNKNOWN_START_VALUES:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _clip_text(text: str, width: int, align: str) -> str:
    """Return one field clipped and padded to a target width."""

    if width <= 0:
        return ""
    trimmed = text[:width]
    return f"{trimmed:<{width}}" if align == "left" else f"{trimmed:>{width}}"


def _parse_blame_output(output: str) -> List[BlameRecord]:
    # store: username -> dict of accumulated stats
    stats: dict[str, dict[str, Any]] = {}

    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        user, account, gres, nodes_str, time_str, state = parts
        user = user.strip()
        state = state.strip()
        if not user:
            continue

        # Parse GPUs
        gpus_per_node = _parse_gres_gpu_count(gres)
        if gpus_per_node == 0:
            continue

        try:
            nodes = int(nodes_str)
        except ValueError:
            nodes = 1

        total_job_gpus = gpus_per_node * nodes

        # Parse Time Limit
        minutes = _parse_slurm_duration(time_str)

        entry = stats.setdefault(
            user,
            {
                "account": account.strip(),
                "running_gpus": 0,
                "pending_gpus": 0,
                "total_req_minutes_run": 0.0,
                "run_count": 0,
            },
        )

        if state == DASH_RUNNING:
            entry["running_gpus"] += total_job_gpus
            entry["total_req_minutes_run"] += minutes
            entry["run_count"] += 1
        elif state == DASH_PENDING:
            entry["pending_gpus"] += total_job_gpus
            # Include pending jobs in the requested average
            entry["total_req_minutes_run"] += minutes
            entry["run_count"] += 1

    # Collect unique accounts
    # Collect unique accounts
    unique_accounts = {data["account"] for data in stats.values()}
    coordinators = _resolve_account_coordinators(unique_accounts)

    records = []
    for user, data in stats.items():
        count = data["run_count"]
        avg = data["total_req_minutes_run"] / count if count > 0 else 0.0

        acc = data["account"]
        coord_name = coordinators.get(acc, "")

        records.append(
            BlameRecord(
                username=user,
                account=acc,
                running_gpus=data["running_gpus"],
                pending_gpus=data["pending_gpus"],
                avg_request_minutes=avg,
                full_name=_resolve_full_name(user),
                coordinator_name=coord_name,
            )
        )

    # Sort by Running GPUs descending
    return sorted(
        records,
        key=lambda r: (r.running_gpus, r.pending_gpus, r.avg_request_minutes),
        reverse=True,
    )


def _resolve_full_name(username: str) -> str:
    try:
        cmd = ["getent", "passwd", username]
        out = subprocess.check_output(cmd, text=True).strip()
        parts = out.split(":")
        if len(parts) >= 5:
            # GECOS field: Full Name,Room Number,Work Phone,Home Phone
            return parts[4].split(",")[0]
    except (subprocess.CalledProcessError, IndexError, FileNotFoundError):
        pass
    return ""


def _resolve_account_coordinators(accounts: set[str]) -> dict[str, str]:
    """Return map of account -> coordinator full name."""
    if not accounts:
        return {}

    # We can batch query if supported, but let's do a single sacctmgr call with commas
    acct_list = ",".join(sorted(accounts))
    # format=Account,Coordinators returns Account|Coordinators|
    cmd = [
        "sacctmgr",
        "show",
        "account",
        acct_list,
        "withcoordinator",
        "format=Account,Coordinators",
        "-n",
        "-p",
    ]

    mapping = {}

    try:
        out = subprocess.check_output(cmd, text=True)
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) >= 2:
                acc_name = parts[0]
                coords = parts[1]  # comma separated usernames
                if coords:
                    # Pick first coordinator
                    first_coord = coords.split(",")[0].strip()
                    if first_coord:
                        mapping[acc_name] = _resolve_full_name(first_coord)
    except Exception:
        pass

    return mapping


def _parse_gres_gpu_count(gres: str) -> int:

    # Handle "gpu:2", "gpu:a100:4", "gpu:2(S_static)"
    # We want the number after the last colon that is digital
    # but gres format is type:name:count or type:count
    # examples: gpu:2, gpu:v100:1.
    if "gpu" not in gres:
        return 0

    # Extract parts specifically for gpu entry if comma separated
    gpu_part = ""
    for part in gres.split(","):
        if "gpu" in part:
            gpu_part = part
            break

    if not gpu_part:
        return 0

    # gpu:types:count or gpu:count
    # split by colon
    subparts = gpu_part.split(":")
    # look for the first part that is an integer? No, the count is usually the last number.
    # But wait, "gpu:2" -> ["gpu", "2"]. "gpu:a100:2" -> ["gpu", "a100", "2"].
    # "gpu" -> ["gpu"] (implies 1? No usually 0/error).

    # Let's try to match the last integer segment
    import re

    match = re.search(r"gpu:(?:[^:]+:)?(\d+)", gpu_part)
    if match:
        return int(match.group(1))
    return 0


def _parse_slurm_duration(time_str: str) -> float:
    # 1-02:03:04, 02:03:04, 20:00
    try:
        days = 0
        if "-" in time_str:
            d_part, time_str = time_str.split("-", 1)
            days = int(d_part)

        parts = time_str.split(":")
        seconds = 0
        if len(parts) == 3:  # HH:MM:SS
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:  # MM:SS
            seconds = int(parts[0]) * 60 + int(parts[1])

        return (days * 24 * 60) + (seconds / 60.0)
    except (ValueError, IndexError):
        return 0.0
